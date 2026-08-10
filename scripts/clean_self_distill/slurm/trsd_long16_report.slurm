#!/usr/bin/env bash
#SBATCH --job-name=trsd-long16-report
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
: "${CSD_LONG_ROOT:?export CSD_LONG_ROOT}"
: "${CSD_SHORT_ROOT:?export CSD_SHORT_ROOT}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"
export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$CSD_LONG_ROOT/tmp_report_${SLURM_JOB_ID}/matplotlib"
export TMPDIR="$CSD_LONG_ROOT/tmp_report_${SLURM_JOB_ID}/tmp"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR" "$CSD_LONG_ROOT/report"

"$CSD_TTT_PYTHON" scripts/clean_self_distill/08_short_empirical_report.py report \
  --episodes "$CSD_LONG_ROOT/train/episodes.jsonl" \
  --scored 0:"$CSD_LONG_ROOT/eval/episode_0000.scored.jsonl" \
  --scored 4:"$CSD_LONG_ROOT/eval/episode_0004.scored.jsonl" \
  --scored 8:"$CSD_LONG_ROOT/eval/episode_0008.scored.jsonl" \
  --scored 12:"$CSD_LONG_ROOT/eval/episode_0012.scored.jsonl" \
  --scored 16:"$CSD_LONG_ROOT/eval/episode_0016.scored.jsonl" \
  --mechanism "$CSD_SHORT_ROOT/mechanism/query_00.jsonl" \
  --mechanism "$CSD_SHORT_ROOT/mechanism/query_01.jsonl" \
  --mechanism "$CSD_SHORT_ROOT/mechanism/query_02.jsonl" \
  --epsilon 0.004 \
  --out-dir "$CSD_LONG_ROOT/report"

printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" > "$CSD_LONG_ROOT/report/REPORT_COMPLETE"
