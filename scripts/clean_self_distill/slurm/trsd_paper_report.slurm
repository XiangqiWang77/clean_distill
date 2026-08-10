#!/usr/bin/env bash
#SBATCH --job-name=trsd-paper-report
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_SHORT_ROOT:?export CSD_SHORT_ROOT}"
: "${CSD_LONG_ROOT:?export CSD_LONG_ROOT}"
: "${CSD_PREPARED_ROOT:?export CSD_PREPARED_ROOT}"
: "${CSD_FULL_EVAL_ROOT:?export CSD_FULL_EVAL_ROOT}"

# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"

export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$CSD_SHORT_ROOT/tmp_report_${SLURM_JOB_ID}/matplotlib"
export TMPDIR="$CSD_SHORT_ROOT/tmp_report_${SLURM_JOB_ID}/tmp"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR" "$CSD_SHORT_ROOT/report/scored"

CSD_REPORT="$CSD_SHORT_ROOT/report"
CSD_BASE_SCORED="$CSD_REPORT/scored/base_10240.jsonl"
CSD_PRIVILEGED_SCORED="$CSD_REPORT/scored/privileged_sd_10240.jsonl"
CSD_TRSD_SCORED="$CSD_REPORT/scored/trsd_episode_0016_10240.jsonl"

CSD_BASE_ARGS=()
CSD_PRIVILEGED_ARGS=()
CSD_TRSD_ARGS=()
for CSD_SHARD in 00 01 02 03; do
  CSD_BASE_ARGS+=(--predictions "$CSD_FULL_EVAL_ROOT/predictions/episode_0000/shard_${CSD_SHARD}.jsonl")
  CSD_PRIVILEGED_ARGS+=(--predictions "$CSD_FULL_EVAL_ROOT/predictions/episode_0064/shard_${CSD_SHARD}.jsonl")
  CSD_TRSD_ARGS+=(--predictions "$CSD_LONG_ROOT/final143/predictions/shard_${CSD_SHARD}.jsonl")
done

"$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
  "${CSD_BASE_ARGS[@]}" \
  --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
  --sample-count 1 \
  --output "$CSD_BASE_SCORED"

"$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
  "${CSD_PRIVILEGED_ARGS[@]}" \
  --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
  --sample-count 1 \
  --output "$CSD_PRIVILEGED_SCORED"

"$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
  "${CSD_TRSD_ARGS[@]}" \
  --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
  --sample-count 1 \
  --output "$CSD_TRSD_SCORED"

"$CSD_TTT_PYTHON" scripts/clean_self_distill/08_short_empirical_report.py report \
  --episodes "$CSD_SHORT_ROOT/train/episodes.jsonl" \
  --scored 0:"$CSD_SHORT_ROOT/eval/episode_0000.scored.jsonl" \
  --scored 1:"$CSD_SHORT_ROOT/eval/episode_0001.scored.jsonl" \
  --scored 2:"$CSD_SHORT_ROOT/eval/episode_0002.scored.jsonl" \
  --mechanism "$CSD_SHORT_ROOT/mechanism/query_00.jsonl" \
  --mechanism "$CSD_SHORT_ROOT/mechanism/query_01.jsonl" \
  --mechanism "$CSD_SHORT_ROOT/mechanism/query_02.jsonl" \
  --epsilon 0.004 \
  --out-dir "$CSD_REPORT"

"$CSD_TTT_PYTHON" scripts/clean_self_distill/09_paper_result_bundle.py \
  --base "$CSD_BASE_SCORED" \
  --privileged "$CSD_PRIVILEGED_SCORED" \
  --trsd "$CSD_TRSD_SCORED" \
  --pilot-report "$CSD_REPORT/summary.json" \
  --output-dir "$CSD_REPORT"

if [[ -s docs/results/trsd_epsilon_one_episode.csv ]]; then
  cp docs/results/trsd_epsilon_one_episode.csv "$CSD_REPORT/epsilon_sensitivity.csv"
fi
printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" > "$CSD_REPORT/REPORT_COMPLETE"
