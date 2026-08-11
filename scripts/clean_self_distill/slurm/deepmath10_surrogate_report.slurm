#!/usr/bin/env bash
#SBATCH --job-name=d10-figures
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -Eeuo pipefail
: "${CSD_D10_CONFIG:?export CSD_D10_CONFIG}"
: "${CSD_D10_EXEC_ROOT:?export CSD_D10_EXEC_ROOT}"
source "$CSD_D10_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_D10_CONDA_ENV"
cd "$CSD_D10_EXEC_ROOT"
export PYTHONPATH="$CSD_D10_EXEC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

for shard in 00 01; do
  [[ -s "$CSD_D10_RUN_ROOT/status/shard_${shard}.complete" ]]
done
"$CSD_D10_PYTHON" scripts/clean_self_distill/34_plot_deepmath10_surrogate_study.py \
  --manifest "$CSD_D10_MANIFEST" \
  --shard "$CSD_D10_RUN_ROOT/evidence/shard_00.jsonl" \
  --shard "$CSD_D10_RUN_ROOT/evidence/shard_01.jsonl" \
  --output-dir "$CSD_D10_RUN_ROOT/figures" \
  --bootstrap-resamples "$CSD_D10_BOOTSTRAP_RESAMPLES" \
  --bootstrap-seed "$CSD_D10_BOOTSTRAP_SEED"
printf 'complete\njob_id=%s\n' "$SLURM_JOB_ID" > "$CSD_D10_RUN_ROOT/REPORT_COMPLETE"
