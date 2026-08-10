#!/usr/bin/env bash
#SBATCH --job-name=trsd-score-report
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_PREPARED_ROOT:?export CSD_PREPARED_ROOT}"
: "${CSD_BUDGET_EVAL_ROOT:?export CSD_BUDGET_EVAL_ROOT}"
: "${CSD_TRSD64_JOURNAL:?export CSD_TRSD64_JOURNAL}"
: "${CSD_FINAL_REPORT_ROOT:?export CSD_FINAL_REPORT_ROOT}"

# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"

export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$CSD_FINAL_REPORT_ROOT/tmp_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$CSD_FINAL_REPORT_ROOT/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CSD_FINAL_REPORT_ROOT/scored"

CSD_LABELS="$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl"
CSD_PRIV64_JOURNAL=${CSD_PRIV64_JOURNAL:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/episodes.jsonl}
CSD_MECHANISM_CSV=${CSD_MECHANISM_CSV:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-short-empirical-20260807/report/mechanism_query_wrapper.csv}
CSD_EPSILON_SENSITIVITY_CSV=${CSD_EPSILON_SENSITIVITY_CSV:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-short-empirical-20260807/report/epsilon_sensitivity.csv}
CSD_HISTORICAL_TRSD16_SCORED=${CSD_HISTORICAL_TRSD16_SCORED:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-short-empirical-20260807/report/scored/trsd_episode_0016_10240.jsonl}
CSD_HISTORICAL_BASE_SCORED=${CSD_HISTORICAL_BASE_SCORED:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-short-empirical-20260807/report/scored/base_10240.jsonl}

[[ -s "$CSD_HISTORICAL_TRSD16_SCORED" ]] || { echo "missing historical TRSD-16 scored file: $CSD_HISTORICAL_TRSD16_SCORED" >&2; exit 2; }
[[ -s "$CSD_HISTORICAL_BASE_SCORED" ]] || { echo "missing historical Base scored file: $CSD_HISTORICAL_BASE_SCORED" >&2; exit 2; }

score_method() {
  local input_root=$1
  local output=$2
  local prediction_args=()
  local shard
  for shard in 00 01 02 03; do
    prediction_args+=(--predictions "$input_root/predictions/shard_${shard}.jsonl")
  done
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    "${prediction_args[@]}" \
    --labels "$CSD_LABELS" \
    --sample-count 1 \
    --output "$output"
}

CSD_BASE_SCORED="$CSD_FINAL_REPORT_ROOT/scored/base.jsonl"
CSD_PRIV16_SCORED="$CSD_FINAL_REPORT_ROOT/scored/privileged_ep16.jsonl"
CSD_TRSD16_SCORED="$CSD_FINAL_REPORT_ROOT/scored/trsd_ep16_rkl_current.jsonl"
CSD_PRIV64_SCORED="$CSD_FINAL_REPORT_ROOT/scored/privileged_ep64.jsonl"
CSD_TRSD64_SCORED="$CSD_FINAL_REPORT_ROOT/scored/trsd_ep64.jsonl"

score_method "$CSD_BUDGET_EVAL_ROOT/base" "$CSD_BASE_SCORED"
score_method "$CSD_BUDGET_EVAL_ROOT/privileged_ep16" "$CSD_PRIV16_SCORED"
score_method "$CSD_BUDGET_EVAL_ROOT/trsd_ep16_rkl_current" "$CSD_TRSD16_SCORED"
score_method "$CSD_BUDGET_EVAL_ROOT/privileged_ep64" "$CSD_PRIV64_SCORED"
score_method "$CSD_BUDGET_EVAL_ROOT/trsd_ep64" "$CSD_TRSD64_SCORED"

CSD_REPORT_ARGS=(
  --base-scored "$CSD_BASE_SCORED"
  --privileged16-scored "$CSD_PRIV16_SCORED"
  --trsd16-scored "$CSD_TRSD16_SCORED"
  --historical-trsd16-scored "$CSD_HISTORICAL_TRSD16_SCORED"
  --historical-base-scored "$CSD_HISTORICAL_BASE_SCORED"
  --privileged64-scored "$CSD_PRIV64_SCORED"
  --trsd64-scored "$CSD_TRSD64_SCORED"
  --privileged64-journal "$CSD_PRIV64_JOURNAL"
  --trsd64-journal "$CSD_TRSD64_JOURNAL"
  --output-dir "$CSD_FINAL_REPORT_ROOT"
)
if [[ -s "$CSD_MECHANISM_CSV" ]]; then
  CSD_REPORT_ARGS+=(--mechanism-csv "$CSD_MECHANISM_CSV")
fi
if [[ -s "$CSD_EPSILON_SENSITIVITY_CSV" ]]; then
  CSD_REPORT_ARGS+=(--epsilon-sensitivity-csv "$CSD_EPSILON_SENSITIVITY_CSV")
fi

"$CSD_TTT_PYTHON" scripts/clean_self_distill/10_final_trsd_report.py \
  "${CSD_REPORT_ARGS[@]}"

printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" \
  > "$CSD_FINAL_REPORT_ROOT/REPORT_COMPLETE"
