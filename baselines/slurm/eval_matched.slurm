#!/usr/bin/env bash
#SBATCH --job-name=baseline-eval
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%4

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_PREPARED_ROOT:?export CSD_PREPARED_ROOT}"
: "${BASELINE_METHOD:?export BASELINE_METHOD}"
: "${BASELINE_EVAL_ROOT:?export BASELINE_EVAL_ROOT}"
: "${BASELINE_EPISODE:=16}"

case "$BASELINE_METHOD" in base|privileged_sd|trsd|demopsd|grpo) ;; *) exit 2 ;; esac
if [[ "$BASELINE_METHOD" == base ]]; then
  [[ "$BASELINE_EPISODE" == 0 ]]
  [[ -z "${BASELINE_CHECKPOINT:-}" ]]
else
  : "${BASELINE_CHECKPOINT:?export BASELINE_CHECKPOINT for adapter methods}"
fi
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"

export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_SCRATCH_ROOT/hf" HF_HUB_CACHE="$CSD_SCRATCH_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/tmp/csdbe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$TMPDIR" "$BASELINE_EVAL_ROOT/predictions" "$BASELINE_EVAL_ROOT/status"

SHARD=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
OUTPUT="$BASELINE_EVAL_ROOT/predictions/shard_${SHARD}.jsonl"
ADAPTER_ARGS=()
[[ -z "${BASELINE_CHECKPOINT:-}" ]] || ADAPTER_ARGS=(--adapter "$BASELINE_CHECKPOINT")
"$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
  --queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
  --output "$OUTPUT" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --method "$BASELINE_METHOD" \
  --checkpoint-episode "$BASELINE_EPISODE" \
  --engine vllm \
  --batch-size 64 \
  --gpu-memory-utilization 0.88 \
  --max-lora-rank 8 \
  --sample-count 1 \
  --max-new-tokens 10240 \
  --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS" \
  --context-window "$CSD_CONTEXT_WINDOW_TOKENS" \
  --attn-implementation sdpa \
  --num-shards 4 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --seed "$CSD_RUN_SEED" \
  "${ADAPTER_ARGS[@]}"
printf 'completed_at=%s\njob_id=%s\nshard=%s\n' \
  "$(date -Is)" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" \
  > "$BASELINE_EVAL_ROOT/status/shard_${SHARD}.done"
