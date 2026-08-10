#!/usr/bin/env bash
#SBATCH --job-name=proj64-score
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%j-score.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%j-score.err

set -Eeuo pipefail
: "${CSD_PT_CONFIG:?export CSD_PT_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_PT_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_PT_CONDA_ENV"
cd "$CSD_PT_CODE_ROOT"
export PYTHONPATH="$CSD_PT_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

VARIANTS=(opsd_original opsd_projected demopsd_original demopsd_projected)
for variant in "${VARIANTS[@]}"; do
  root="$CSD_PT_RUN_ROOT/table14/$variant/math"
  mkdir -p "$root/scored"
  for shard in 0 1 2 3; do
    shard_name=$(printf '%02d' "$shard")
    "$CSD_PT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
      --predictions "$root/predictions/shard_${shard_name}.jsonl" \
      --labels "$CSD_PT_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
      --sample-count 1 \
      --output "$root/scored/shard_${shard_name}.jsonl"
  done
done

