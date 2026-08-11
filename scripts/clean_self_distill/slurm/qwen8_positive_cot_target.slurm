#!/usr/bin/env bash
#SBATCH --job-name=q8-pcot-target
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -Eeuo pipefail
: "${CSD_PCOT_CONFIG:?submitter must export CSD_PCOT_CONFIG}"
source "$CSD_PCOT_CONFIG"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_PCOT_CONDA_ENV"
cd "$CSD_PCOT_CODE_ROOT"

export PYTHONPATH="$CSD_PCOT_TORCH_OVERLAY:$CSD_PCOT_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_PCOT_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_PCOT_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/pcot_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_PCOT_RUN_ROOT/logs" "$CSD_PCOT_RUN_ROOT/status"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader \
  | grep -Eqi "${CSD_PCOT_GPU_NAME_REGEX:-H100|H200|RTX.*6000}"
[[ "$(sha256sum "$CSD_PCOT_QUERIES" | awk '{print $1}')" == "$CSD_PCOT_QUERIES_SHA256" ]]
[[ "$(sha256sum "$CSD_PCOT_LABELS" | awk '{print $1}')" == "$CSD_PCOT_LABELS_SHA256" ]]
[[ "$(sha256sum "$CSD_PCOT_HISTORICAL_RAW_LOG" | awk '{print $1}')" == "$CSD_PCOT_HISTORICAL_RAW_SHA256" ]]
[[ "$(sha256sum "$CSD_PCOT_HISTORICAL_PROJECTED_LOG" | awk '{print $1}')" == "$CSD_PCOT_HISTORICAL_PROJECTED_SHA256" ]]
[[ "$(sha256sum "$CSD_PCOT_LONG_HORIZON_NLL" | awk '{print $1}')" == "$CSD_PCOT_LONG_HORIZON_NLL_SHA256" ]]
[[ -s "$CSD_PCOT_MODEL_LOCAL_DIR/config.json" ]]

date '+started=%Y-%m-%dT%H:%M:%S%z'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$CSD_PCOT_PYTHON" scripts/clean_self_distill/39_positive_cot_target_study.py \
  --queries "$CSD_PCOT_QUERIES" \
  --labels "$CSD_PCOT_LABELS" \
  --model "$CSD_PCOT_MODEL_LOCAL_DIR" \
  --model-id "$CSD_PCOT_MODEL_ID" \
  --revision "$CSD_PCOT_MODEL_REVISION" \
  --run-root "$CSD_PCOT_RUN_ROOT" \
  --num-queries "$CSD_PCOT_NUM_QUERIES" \
  --reference-token-cap "$CSD_PCOT_REFERENCE_TOKEN_CAP" \
  --answer-token-cap "$CSD_PCOT_ANSWER_TOKEN_CAP" \
  --max-prompt-tokens "$CSD_PCOT_MAX_PROMPT_TOKENS" \
  --context-window "$CSD_PCOT_CONTEXT_WINDOW" \
  --loops "$CSD_PCOT_LOOPS" \
  --loop-rate "$CSD_PCOT_LOOP_RATE" \
  --epsilon "$CSD_PCOT_EPSILON" \
  --binary-search-steps "$CSD_PCOT_BINARY_SEARCH_STEPS" \
  --full-vocab-chunk-size "$CSD_PCOT_FULL_VOCAB_CHUNK" \
  --bootstrap-resamples "$CSD_PCOT_BOOTSTRAP_RESAMPLES" \
  --bootstrap-seed "$CSD_PCOT_BOOTSTRAP_SEED" \
  --dtype "$CSD_PCOT_DTYPE" \
  --attn-implementation "$CSD_PCOT_ATTN_IMPLEMENTATION" \
  --historical-raw-log "$CSD_PCOT_HISTORICAL_RAW_LOG" \
  --historical-projected-log "$CSD_PCOT_HISTORICAL_PROJECTED_LOG" \
  --long-horizon-nll "$CSD_PCOT_LONG_HORIZON_NLL"

[[ -s "$CSD_PCOT_RUN_ROOT/COMPLETE.json" ]]
jq -e ".queries == $CSD_PCOT_NUM_QUERIES" "$CSD_PCOT_RUN_ROOT/COMPLETE.json" >/dev/null
printf 'complete\njob_id=%s\nrun_root=%s\n' "$SLURM_JOB_ID" "$CSD_PCOT_RUN_ROOT" \
  > "$CSD_PCOT_RUN_ROOT/status/job_${SLURM_JOB_ID}.complete"
date '+finished=%Y-%m-%dT%H:%M:%S%z'
