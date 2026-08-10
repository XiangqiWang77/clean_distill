#!/usr/bin/env bash
#SBATCH --job-name=q8-core-a2
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-component-ablations64-20260809/logs/%j-core-report.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-component-ablations64-20260809/logs/%j-core-report.err

set -Eeuo pipefail
: "${CSD_A2_CONFIG:?export CSD_A2_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_A2_CONFIG"

for variant in independent_token_budgets fixed_global_alpha; do
  scored="$CSD_A2_RUN_ROOT/eval/$variant/scored.jsonl"
  until [[ -s "$scored" ]] && [[ "$(wc -l < "$scored")" == 143 ]]; do
    sleep 60
  done
done

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_A2_CONDA_ENV"
cd "$CSD_A2_CODE_ROOT"
export PYTHONPATH="$CSD_A2_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$CSD_A2_PYTHON" scripts/clean_self_distill/24_qwen8_component_ablation_table.py \
  --config "$CSD_A2_CONFIG" \
  --include independent_token_budgets \
  --include fixed_global_alpha
