#!/usr/bin/env bash
#SBATCH --job-name=q8-a2-table
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-component-ablations64-20260809/logs/%j-report.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-component-ablations64-20260809/logs/%j-report.err

set -Eeuo pipefail
: "${CSD_A2_CONFIG:?export CSD_A2_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_A2_CONFIG"
CSD_A2_CODE_ROOT=${CSD_A2_CODE_ROOT_OVERRIDE:-$CSD_A2_CODE_ROOT}
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_A2_CONDA_ENV"
cd "$CSD_A2_CODE_ROOT"
export PYTHONPATH="$CSD_A2_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$CSD_A2_PYTHON" scripts/clean_self_distill/24_qwen8_component_ablation_table.py \
  --config "$CSD_A2_CONFIG"
