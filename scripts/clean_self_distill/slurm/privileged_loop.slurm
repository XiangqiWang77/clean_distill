#!/usr/bin/env bash
#SBATCH --job-name=privileged-sd-main
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@3600

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"

cd "$CSD_ONLINE_CODE_ROOT"
export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$CSD_RUN_ROOT/tmp/privileged_${SLURM_JOB_ID}"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

# shellcheck disable=SC1091
source "$CSD_ONLINE_CODE_ROOT/scripts/clean_self_distill/slurm/persistent_wait.sh"
CSD_OUTPUT="${CSD_TIMEBOX_OUTPUT:-$CSD_RUN_ROOT/core_privileged_loop}"
mkdir -p "$CSD_OUTPUT"
CSD_RESUME=()
[[ ! -s "$CSD_OUTPUT/run_manifest.json" ]] || CSD_RESUME+=(--resume)

CSD_CHILD_PID=
trap csd_forward_usr1 USR1
"$CSD_TTT_PYTHON" scripts/clean_self_distill/04_persistent_train.py \
  --branch privileged \
  --variant correct_wrong_signed \
  --queries "$CSD_PREPARED_ROOT/distill_queries.jsonl" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --output-dir "$CSD_OUTPUT" \
  --episodes "${CSD_TIMEBOX_EPISODES:-$CSD_DISTILL_EPISODES}" \
  --scientific-checkpoints "${CSD_TIMEBOX_CHECKPOINTS:-$CSD_SCIENTIFIC_CHECKPOINTS}" \
  --rolling-checkpoint-interval "${CSD_TIMEBOX_ROLLING_INTERVAL:-$CSD_ROLLING_CHECKPOINT_INTERVAL}" \
  --max-sequence-tokens "$CSD_TRAIN_MAX_SEQUENCE_TOKENS" \
  --max-rollout-tokens "${CSD_TIMEBOX_ROLLOUT_TOKENS:-$CSD_TRAIN_MAX_SEQUENCE_TOKENS}" \
  --learning-rate "$CSD_LEARNING_RATE" \
  --lora-rank "$CSD_LORA_RANK" \
  --lora-alpha "$CSD_LORA_ALPHA" \
  --seed "$CSD_RUN_SEED" \
  --distill-token-chunk-size "$CSD_DISTILL_TOKEN_CHUNK_SIZE" \
  "${CSD_RESUME[@]}" &
CSD_CHILD_PID=$!
CSD_RC=0
csd_wait_for_child "$CSD_CHILD_PID" || CSD_RC=$?
CSD_CHILD_PID=
if (( CSD_RC == 75 )); then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$CSD_RC"
