#!/usr/bin/env bash
#SBATCH --job-name=trsd64-interim
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_PREPARED_ROOT:?export CSD_PREPARED_ROOT}"
: "${CSD_BUDGET_EVAL_ROOT:?export CSD_BUDGET_EVAL_ROOT}"
: "${CSD_INTERIM_ROOT:?export CSD_INTERIM_ROOT}"

# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"

export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$CSD_INTERIM_ROOT/tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$CSD_INTERIM_ROOT"

"$CSD_TTT_PYTHON" scripts/clean_self_distill/11_interim_trsd64_snapshot.py \
  --trsd-prediction "$CSD_BUDGET_EVAL_ROOT/trsd_ep64/predictions/shard_00.jsonl" \
  --trsd-prediction "$CSD_BUDGET_EVAL_ROOT/trsd_ep64/predictions/shard_01.jsonl" \
  --trsd-prediction "$CSD_BUDGET_EVAL_ROOT/trsd_ep64/predictions/shard_02.jsonl" \
  --trsd-prediction "$CSD_BUDGET_EVAL_ROOT/trsd_ep64/predictions/shard_03.jsonl" \
  --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
  --reference "base=$CSD_BUDGET_EVAL_ROOT/partial_scored/base.jsonl" \
  --reference "privileged_ep16=$CSD_BUDGET_EVAL_ROOT/partial_scored/privileged_ep16.jsonl" \
  --reference "privileged_ep64=$CSD_BUDGET_EVAL_ROOT/partial_scored/privileged_ep64.jsonl" \
  --output-dir "$CSD_INTERIM_ROOT"
