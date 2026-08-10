#!/usr/bin/env bash
#SBATCH --job-name=q8-int-eval
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --time=11:50:00
#SBATCH --array=0-3%3

set -Eeuo pipefail
: "${CSD_I8_CONFIG:?export CSD_I8_CONFIG}"
: "${CSD_I8_METHOD:?export CSD_I8_METHOD}"
: "${CSD_I8_EPISODE:?export CSD_I8_EPISODE}"
: "${CSD_I8_CHECKPOINT:?export CSD_I8_CHECKPOINT}"
: "${CSD_I8_EVAL_ROOT:?export CSD_I8_EVAL_ROOT}"
# shellcheck disable=SC1090
source "$CSD_I8_CONFIG"

case "$CSD_I8_METHOD" in
  privileged_sd)
    [[ "$CSD_I8_EPISODE" == 16 || "$CSD_I8_EPISODE" == 32 || "$CSD_I8_EPISODE" == 48 ]]
    ;;
  trsd)
    [[ "$CSD_I8_EPISODE" == 32 || "$CSD_I8_EPISODE" == 48 ]]
    ;;
  *) printf 'unsupported method: %s\n' "$CSD_I8_METHOD" >&2; exit 2 ;;
esac
[[ -s "$CSD_I8_CHECKPOINT/checkpoint_manifest.json" ]]
[[ -d "$CSD_I8_SNAPSHOT_ROOT/src" ]]
[[ -f "$CSD_I8_SNAPSHOT_ROOT/scripts/clean_self_distill/05_heldout_eval.py" ]]

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_I8_CONDA_ENV"
cd "$CSD_I8_SNAPSHOT_ROOT"

# Deliberately use the conda cu126 stack used by the retained frozen-protocol
# Qwen3-8B 0/16/64 evaluations.  The cu128 overlay is only for the archived
# Privilege-SD training replay.
export PYTHONPATH="$CSD_I8_SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_I8_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_I8_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/tmp/q8_i8_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_I8_EVAL_ROOT/predictions" "$CSD_I8_EVAL_ROOT/status"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100
"$CSD_I8_PYTHON" - <<'PY'
import torch
assert str(torch.__version__).endswith("+cu126"), torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
print(f"frozen_eval_torch={torch.__version__} cuda={torch.version.cuda}")
PY

shard=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
output="$CSD_I8_EVAL_ROOT/predictions/shard_${shard}.jsonl"
done_file="$CSD_I8_EVAL_ROOT/status/shard_${shard}.done"
if [[ ! -s "$done_file" ]]; then
  "$CSD_I8_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$CSD_I8_PREPARED_ROOT/heldout_queries.jsonl" \
    --output "$output" \
    --model "$CSD_I8_MODEL_LOCAL_DIR" \
    --model-id "$CSD_I8_MODEL_ID" \
    --revision "$CSD_I8_MODEL_REVISION" \
    --method "$CSD_I8_METHOD" \
    --checkpoint-episode "$CSD_I8_EPISODE" \
    --adapter "$CSD_I8_CHECKPOINT" \
    --sample-count 1 \
    --max-new-tokens "$CSD_I8_EVAL_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$CSD_I8_MAX_PROMPT_TOKENS" \
    --context-window "$CSD_I8_CONTEXT_WINDOW_TOKENS" \
    --attn-implementation sdpa \
    --num-shards 4 \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --seed "$CSD_I8_EVAL_SEED"
  printf 'completed_at=%s\njob_id=%s\nmethod=%s\nepisode=%s\nshard=%s\n' \
    "$(date -Is)" "$SLURM_JOB_ID" "$CSD_I8_METHOD" "$CSD_I8_EPISODE" \
    "$SLURM_ARRAY_TASK_ID" > "$done_file"
fi
