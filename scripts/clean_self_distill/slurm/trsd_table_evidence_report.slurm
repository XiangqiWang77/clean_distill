#!/usr/bin/env bash
#SBATCH --job-name=trsd-tables
#SBATCH --partition=devel
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-table-report-20260808/logs/report-%j.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-table-report-20260808/logs/report-%j.err

set -euo pipefail

REPO=/home/da839/clean_distill
SCRATCH=/home/da839/scratch_pi_mg269/da839/clean_distill/runs
EVAL_ROOT="$SCRATCH/budget-prompt-eval-20260807"
OUT="$SCRATCH/trsd-table-report-20260808"
STAGE="$REPO/docs/experiments/trsd_table_report_20260808"
T16_EVAL_ROOT="$EVAL_ROOT/trsd_ep16_rkl_current"
T16_SCORED="$EVAL_ROOT/partial_scored/trsd_ep16_rkl_current.jsonl"
EPISODE_STYLE="$SCRATCH/trsd-style64-final-20260808-v2/matched64_episode_style_metrics.csv"

mkdir -p "$OUT/logs" "$STAGE"

# Fail before invoking the builder unless the current explicit-budget,
# reverse-KL T16 evaluation is fully generated and scored.  The builder repeats
# the 143-row, source-count, query-match, seed, and protocol checks.
for SHARD in 00 01 02 03; do
  [[ -s "$T16_EVAL_ROOT/predictions/shard_${SHARD}.jsonl" ]]
  [[ -s "$T16_EVAL_ROOT/status/shard_${SHARD}.done" ]]
done
[[ -s "$T16_SCORED" ]]

python3 "$REPO/scripts/clean_self_distill/12_build_table_evidence_bundle.py" \
  --scored-root "$EVAL_ROOT/partial_scored" \
  --prediction-root "$EVAL_ROOT" \
  --style-root "$SCRATCH/trsd-style64-final-20260808-v2" \
  --epsilon-csv "$REPO/docs/results/trsd_epsilon_one_episode.csv" \
  --p16-journal "$SCRATCH/csd-h100-clean-sd-20260806-130355/timebox12h/privileged/episodes.jsonl" \
  --p16-manifest "$SCRATCH/csd-h100-clean-sd-20260806-130355/timebox12h/privileged/run_manifest.json" \
  --p64-journal "$SCRATCH/csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/episodes.jsonl" \
  --p64-manifest "$SCRATCH/csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/run_manifest.json" \
  --t64-journal "$SCRATCH/reverse-kl-matched64-20260807/trsd/train/episodes.jsonl" \
  --t64-manifest "$SCRATCH/reverse-kl-matched64-20260807/trsd/train/run_manifest.json" \
  --out "$OUT" \
  --stage "$STAGE"

python3 "$REPO/scripts/clean_self_distill/13_plot_trsd_three_claim_story.py" \
  --bundle "$OUT" \
  --episode-csv "$EPISODE_STYLE" \
  --stage "$STAGE"
