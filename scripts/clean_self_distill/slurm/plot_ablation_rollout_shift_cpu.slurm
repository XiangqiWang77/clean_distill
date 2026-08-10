#!/usr/bin/env bash
#SBATCH --job-name=trsd-fig-e
#SBATCH --partition=scavenge
#SBATCH --account=pi_mg269
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/da839/clean_distill
export PATH=/home/da839/anaconda3/bin:/usr/bin:/bin
export PYTHONPATH=/home/da839/clean_distill${PYTHONPATH:+:${PYTHONPATH}}
export MPLCONFIGDIR=/tmp/trsd-mpl-${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"

/home/da839/anaconda3/bin/python \
  scripts/clean_self_distill/29_plot_ablation_rollout_shift.py

for output in \
  docs/figures/qwen3_8b_ablation_rollout_shift/rollout_target_shift_distribution.png \
  docs/figures/qwen3_8b_ablation_rollout_shift/rollout_target_shift_distribution.pdf \
  docs/figures/qwen3_8b_ablation_rollout_shift/rollout_target_shift_distribution.svg \
  docs/figures/qwen3_8b_ablation_rollout_shift/summary.json; do
  test -s "${output}"
done

echo "Panel (e) rendering complete."
