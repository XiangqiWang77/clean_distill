#!/usr/bin/env bash
#SBATCH --job-name=matched-eval
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%3

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_PREPARED_ROOT:?export CSD_PREPARED_ROOT}"
: "${CSD_EVAL_ROOT:?export CSD_EVAL_ROOT}"
: "${CSD_EVAL_METHOD:?export CSD_EVAL_METHOD}"
: "${CSD_CHECKPOINT_DIR:?export CSD_CHECKPOINT_DIR}"
: "${CSD_CHECKPOINT_EPISODE:?export CSD_CHECKPOINT_EPISODE}"
CSD_EXPECTED_GPU_REGEX=${CSD_EXPECTED_GPU_REGEX:-H100}

case "$CSD_EVAL_METHOD" in
  base|trsd|privileged_sd) ;;
  *) printf 'unsupported method: %s\n' "$CSD_EVAL_METHOD" >&2; exit 2 ;;
esac

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
export TMPDIR="$CSD_EVAL_ROOT/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$CSD_EVAL_ROOT/predictions" "$CSD_EVAL_ROOT/status"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i "$CSD_EXPECTED_GPU_REGEX" >/dev/null
CSD_ADAPTER_ARGS=()
if [[ "$CSD_EVAL_METHOD" == base ]]; then
  [[ "$CSD_CHECKPOINT_EPISODE" == 0 ]]
else
  [[ -s "$CSD_CHECKPOINT_DIR/checkpoint_manifest.json" ]]
  CSD_ADAPTER_ARGS=(--adapter "$CSD_CHECKPOINT_DIR")
fi

CSD_SHARD=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
CSD_OUTPUT="$CSD_EVAL_ROOT/predictions/shard_${CSD_SHARD}.jsonl"
CSD_DONE="$CSD_EVAL_ROOT/status/shard_${CSD_SHARD}.done"
if [[ ! -s "$CSD_DONE" ]]; then
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
    --output "$CSD_OUTPUT" \
    --model "$CSD_MODEL_LOCAL_DIR" \
    --model-id "$CSD_MODEL_ID" \
    --revision "$CSD_MODEL_REVISION" \
    --method "$CSD_EVAL_METHOD" \
    --checkpoint-episode "$CSD_CHECKPOINT_EPISODE" \
    "${CSD_ADAPTER_ARGS[@]}" \
    --sample-count 1 \
    --max-new-tokens 10240 \
    --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS" \
    --context-window "$CSD_CONTEXT_WINDOW_TOKENS" \
    --attn-implementation sdpa \
    --num-shards 4 \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --seed "$CSD_RUN_SEED"
  printf 'completed_at=%s\njob_id=%s\nshard=%s\n' \
    "$(date -Is)" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" > "$CSD_DONE"
fi
