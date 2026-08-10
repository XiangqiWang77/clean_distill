#!/usr/bin/env bash
#SBATCH --job-name=srpo-table
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%j.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%j.err

set -Eeuo pipefail
: "${CSD_SRPO_CONFIG:?export CSD_SRPO_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_SRPO_CONFIG"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_SRPO_CONDA_ENV"
cd "$CSD_SRPO_SNAPSHOT_ROOT"
export PYTHONPATH="$CSD_SRPO_SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

Q8_INTERMEDIATE=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-intermediate-checkpoints-20260809
OUTPUT="$CSD_SRPO_RUN_ROOT/results/complete_math_table"
"$CSD_SRPO_PYTHON" scripts/clean_self_distill/27_cross_model_math_table.py \
  --qwen8-existing-eval-root /home/da839/scratch_pi_mg269/da839/clean_distill/runs/budget-prompt-eval-20260807/partial_scored \
  --qwen8-new-eval-root "$Q8_INTERMEDIATE/eval" \
  --qwen8-privileged-journal /home/da839/scratch_pi_mg269/da839/clean_distill/runs/csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/episodes.jsonl \
  --qwen8-trsd-journal /home/da839/scratch_pi_mg269/da839/clean_distill/runs/reverse-kl-matched64-20260807/trsd/train/episodes.jsonl \
  --qwen17-root /home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-1.7b-fourway-20260808 \
  --gptoss20-root /home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809 \
  --srpo-root "$CSD_SRPO_RUN_ROOT" \
  --output-dir "$OUTPUT"

[[ -s "$OUTPUT/README.md" ]]
[[ -s "$OUTPUT/complete_math_table.csv" ]]
[[ -s "$OUTPUT/summary.json" ]]
printf 'complete\njob_id=%s\noutput=%s\n' "$SLURM_JOB_ID" "$OUTPUT" \
  > "$CSD_SRPO_RUN_ROOT/FINAL_REPORT_COMPLETE"
