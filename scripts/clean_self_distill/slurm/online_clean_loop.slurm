#!/usr/bin/env bash
#SBATCH --job-name=three-points-clean
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
export HF_HOME="$CSD_RUN_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$CSD_RUN_ROOT/tmp/online_${SLURM_JOB_ID}"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

# shellcheck disable=SC1091
source "$CSD_ONLINE_CODE_ROOT/scripts/clean_self_distill/slurm/persistent_wait.sh"
CSD_OUTPUT="$CSD_RUN_ROOT/core_clean_loop"
mkdir -p "$CSD_OUTPUT"
CSD_RESUME=()
[[ ! -s "$CSD_OUTPUT/run_manifest.json" ]] || CSD_RESUME+=(--resume)

CSD_CHILD_PID=
trap csd_forward_usr1 USR1
"$CSD_TTT_PYTHON" scripts/clean_self_distill/04_online_clean_loop.py \
  --queries "$CSD_PREPARED_ROOT/distill_queries.jsonl" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --output-dir "$CSD_OUTPUT" \
  --episodes "$CSD_DISTILL_EPISODES" \
  --scientific-checkpoints "$CSD_SCIENTIFIC_CHECKPOINTS" \
  --rolling-checkpoint-interval "$CSD_ROLLING_CHECKPOINT_INTERVAL" \
  --seed "$CSD_RUN_SEED" \
  --num-candidates "$CSD_NUM_CANDIDATES" \
  --proposal-oversample "$CSD_PROPOSAL_OVERSAMPLE" \
  --proposal-max-rounds "$CSD_PROPOSAL_MAX_ROUNDS" \
  --min-accepted-candidates "$CSD_MIN_ACCEPTED_CANDIDATES" \
  --max-sequence-tokens "$CSD_TRAIN_MAX_SEQUENCE_TOKENS" \
  --max-rollout-tokens "$CSD_TRAIN_MAX_SEQUENCE_TOKENS" \
  --learning-rate "$CSD_LEARNING_RATE" \
  --lora-rank "$CSD_LORA_RANK" \
  --lora-alpha "$CSD_LORA_ALPHA" \
  --distill-token-chunk-size "$CSD_DISTILL_TOKEN_CHUNK_SIZE" \
  --ridge-lambda "$CSD_RIDGE_LAMBDA" \
  --residual-step-size "$CSD_RESIDUAL_STEP_SIZE" \
  --max-support-tokens "$CSD_MAX_SUPPORT_TOKENS" \
  --max-tokens-per-candidate "$CSD_MAX_TOKENS_PER_CANDIDATE" \
  --hard-negatives "$CSD_HARD_NEGATIVES" \
  --frontier-positive-weight "$CSD_FRONTIER_POSITIVE_WEIGHT" \
  --frontier-negative-weight "$CSD_FRONTIER_NEGATIVE_WEIGHT" \
  --frontier-target-margin "$CSD_FRONTIER_TARGET_MARGIN" \
  --max-update-norm "$CSD_MAX_UPDATE_NORM" \
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
