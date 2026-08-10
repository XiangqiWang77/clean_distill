#!/usr/bin/env bash
#SBATCH --job-name=score-base-p64
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_PREPARED_ROOT:?export CSD_PREPARED_ROOT}"
: "${CSD_BUDGET_EVAL_ROOT:?export CSD_BUDGET_EVAL_ROOT}"
: "${CSD_PAIR_SCORE_ROOT:?export CSD_PAIR_SCORE_ROOT}"

# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"
export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$CSD_PAIR_SCORE_ROOT"

score() {
  local root=$1
  local output=$2
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    --predictions "$root/predictions/shard_00.jsonl" \
    --predictions "$root/predictions/shard_01.jsonl" \
    --predictions "$root/predictions/shard_02.jsonl" \
    --predictions "$root/predictions/shard_03.jsonl" \
    --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
    --sample-count 1 \
    --output "$output"
}

score "$CSD_BUDGET_EVAL_ROOT/base" "$CSD_PAIR_SCORE_ROOT/base.jsonl"
score "$CSD_BUDGET_EVAL_ROOT/privileged_ep64" "$CSD_PAIR_SCORE_ROOT/privileged_ep64.jsonl"
printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" > "$CSD_PAIR_SCORE_ROOT/COMPLETE"
