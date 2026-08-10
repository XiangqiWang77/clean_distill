#!/usr/bin/env bash
#SBATCH --job-name=prompt-var-plot
#SBATCH --account=pi_mg269
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -Eeuo pipefail

CSD_REPO=/home/da839/clean_distill
CSD_MECHANISM=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-short-empirical-20260807/mechanism
CSD_OUTPUT="$CSD_REPO/docs/figures/prompt_sensitivity_multiquery/prompt_sensitivity_multiquery"

cd "$CSD_REPO"
export PATH=/home/da839/anaconda3/bin:/usr/bin:/bin
export PYTHONPATH="$CSD_REPO${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/prompt-sensitivity-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

/home/da839/anaconda3/bin/python \
  scripts/clean_self_distill/30_plot_prompt_sensitivity.py \
  --posthoc \
  "$CSD_MECHANISM/query_00.jsonl" \
  "$CSD_MECHANISM/query_01.jsonl" \
  "$CSD_MECHANISM/query_02.jsonl" \
  --output-prefix "$CSD_OUTPUT" \
  --example-token Therefore

for CSD_SUFFIX in png pdf svg csv; do
  test -s "${CSD_OUTPUT}.${CSD_SUFFIX}"
done
test -s "${CSD_OUTPUT}_summary.csv"
test -s "${CSD_OUTPUT}_example.csv"
