#!/usr/bin/env bash
#SBATCH --job-name=trsd-final143
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_LONG_ROOT:?export CSD_LONG_ROOT}"
: "${CSD_PREPARED_ROOT:?export CSD_PREPARED_ROOT}"
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
export TMPDIR="$CSD_LONG_ROOT/tmp_final143_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$CSD_LONG_ROOT/final143/predictions" "$CSD_LONG_ROOT/final143/status"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

CSD_SHARD=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
CSD_OUTPUT="$CSD_LONG_ROOT/final143/predictions/shard_${CSD_SHARD}.jsonl"
CSD_DONE="$CSD_LONG_ROOT/final143/status/shard_${CSD_SHARD}.done"
CSD_CHECKPOINT="$CSD_LONG_ROOT/train/checkpoints/episode_0016"
[[ -s "$CSD_CHECKPOINT/checkpoint_manifest.json" ]]

if [[ ! -s "$CSD_DONE" ]]; then
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
    --output "$CSD_OUTPUT" \
    --model "$CSD_MODEL_LOCAL_DIR" \
    --model-id "$CSD_MODEL_ID" \
    --revision "$CSD_MODEL_REVISION" \
    --method trsd \
    --checkpoint-episode 16 \
    --adapter "$CSD_CHECKPOINT" \
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
