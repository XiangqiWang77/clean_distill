#!/usr/bin/env bash
#SBATCH --job-name=go20-train2
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@1800
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%j-train2.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%j-train2.err

set -Eeuo pipefail
: "${CSD_GO_CONFIG:?export CSD_GO_CONFIG}"
: "${CSD_GO_TRAIN_TAG:?export CSD_GO_TRAIN_TAG}"
# shellcheck disable=SC1090
source "$CSD_GO_CONFIG"

case "$CSD_GO_TRAIN_TAG" in
  privileged_16) BRANCH=privileged; EPISODES=16; PREPARED="$CSD_GO_PREPARED_16_ROOT"; DISTILL_SHA="$CSD_GO_DISTILL_16_SHA256" ;;
  privileged_64) BRANCH=privileged; EPISODES=64; PREPARED="$CSD_GO_PREPARED_64_ROOT"; DISTILL_SHA="$CSD_GO_DISTILL_64_SHA256" ;;
  trsd_16)       BRANCH=clean;      EPISODES=16; PREPARED="$CSD_GO_PREPARED_16_ROOT"; DISTILL_SHA="$CSD_GO_DISTILL_16_SHA256" ;;
  trsd_64)       BRANCH=clean;      EPISODES=64; PREPARED="$CSD_GO_PREPARED_64_ROOT"; DISTILL_SHA="$CSD_GO_DISTILL_64_SHA256" ;;
  *) echo "Unknown CSD_GO_TRAIN_TAG=$CSD_GO_TRAIN_TAG" >&2; exit 2 ;;
esac

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_GO_CONDA_ENV"
cd "$CSD_GO_CODE_ROOT"
export PYTHONPATH="$CSD_GO_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_GO_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_GO_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/go20_train2_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_GO_RUN_ROOT/logs"

[[ -s "$CSD_GO_RUN_ROOT/TWO_GPU_SMOKE.json" ]]
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 2 ]]
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
DISTILL_QUERIES="$PREPARED/distill_queries.jsonl"
[[ "$(wc -l < "$DISTILL_QUERIES")" == "$EPISODES" ]]
[[ "$(sha256sum "$DISTILL_QUERIES" | awk '{print $1}')" == "$DISTILL_SHA" ]]

TRAIN_DIR="$CSD_GO_RUN_ROOT/train/$CSD_GO_TRAIN_TAG"
FINAL_CHECKPOINT="$TRAIN_DIR/checkpoints/episode_$(printf '%04d' "$EPISODES")"
if [[ "$EPISODES" == 16 ]]; then
  CHECKPOINTS=0,16
  ROLLING_INTERVAL=4
else
  CHECKPOINTS=0,16,32,48,64
  ROLLING_INTERVAL=8
fi

# shellcheck disable=SC1091
source "$CSD_GO_CODE_ROOT/scripts/clean_self_distill/slurm/persistent_wait.sh"
resume=()
[[ ! -s "$TRAIN_DIR/run_manifest.json" ]] || resume=(--resume)
trust_args=()
if [[ "$BRANCH" == clean ]]; then
  trust_args=(
    --trust-region-kl-budget "$CSD_GO_TRUST_REGION_KL_BUDGET"
    --trust-region-binary-search-steps "$CSD_GO_TRUST_REGION_BINARY_SEARCH_STEPS"
  )
fi

CSD_CHILD_PID=
trap csd_forward_usr1 USR1
if [[ ! -s "$FINAL_CHECKPOINT/checkpoint_manifest.json" ]]; then
  "$CSD_GO_PYTHON" scripts/clean_self_distill/04_persistent_train.py \
    --branch "$BRANCH" \
    --queries "$DISTILL_QUERIES" \
    --model "$CSD_GO_MODEL_LOCAL_DIR" \
    --model-id "$CSD_GO_MODEL_ID" \
    --revision "$CSD_GO_MODEL_REVISION" \
    --output-dir "$TRAIN_DIR" \
    --episodes "$EPISODES" \
    --scientific-checkpoints "$CHECKPOINTS" \
    --rolling-checkpoint-interval "$ROLLING_INTERVAL" \
    --max-sequence-tokens "$CSD_GO_TRAIN_MAX_SEQUENCE_TOKENS" \
    --max-rollout-tokens "$CSD_GO_TRAIN_MAX_ROLLOUT_TOKENS" \
    --learning-rate "$CSD_GO_LEARNING_RATE" \
    --lora-rank "$CSD_GO_LORA_RANK" \
    --lora-alpha "$CSD_GO_LORA_ALPHA" \
    --seed "$CSD_GO_TRAIN_SEED" \
    --distill-token-chunk-size "$CSD_GO_DISTILL_TOKEN_CHUNK_SIZE" \
    --device-map balanced \
    "${trust_args[@]}" \
    "${resume[@]}" &
  CSD_CHILD_PID=$!
  rc=0
  csd_wait_for_child "$CSD_CHILD_PID" || rc=$?
  CSD_CHILD_PID=
  if (( rc == 75 )); then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  (( rc == 0 ))
fi
[[ -s "$FINAL_CHECKPOINT/checkpoint_manifest.json" ]]
printf 'complete\njob_id=%s\nmethod=%s\nepisodes=%s\ngpus=2\n' \
  "$SLURM_JOB_ID" "$CSD_GO_TRAIN_TAG" "$EPISODES" \
  > "$CSD_GO_RUN_ROOT/$CSD_GO_TRAIN_TAG.TRAIN_COMPLETE"
