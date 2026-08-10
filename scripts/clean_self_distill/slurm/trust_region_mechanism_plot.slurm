#!/usr/bin/env bash
#SBATCH --job-name=trsd-mech-plot
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# CPU-only rendering stage. Submit with an afterok dependency on the H100
# collector; this job never polls, loads a model, or generates a trajectory.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG before sbatch}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT before sbatch}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"

export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$CSD_RUN_ROOT/tmp/trsd_mechanism_plot_${SLURM_JOB_ID}/matplotlib"
export TMPDIR="$CSD_RUN_ROOT/tmp/trsd_mechanism_plot_${SLURM_JOB_ID}/tmp"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR" "$CSD_RUN_ROOT/status"

CSD_MECHANISM_ROOT=${CSD_MECHANISM_ROOT:-$CSD_RUN_ROOT/trust_region_mechanism}
CSD_MECHANISM_INPUT="$CSD_MECHANISM_ROOT/raw/query_0022.jsonl"
CSD_COLLECT_MARKER="$CSD_MECHANISM_ROOT/status/collect_query_0022.done"
CSD_PLOT_MARKER="$CSD_MECHANISM_ROOT/status/plot_query_0022.done"
CSD_OUTPUT_PREFIX="$CSD_MECHANISM_ROOT/figures/trust_region_mechanism_query_0022"
CSD_LABELS="$CSD_PREPARED_ROOT/distill_labels.sealed.jsonl"

[[ -s "$CSD_COLLECT_MARKER" ]]
[[ -s "$CSD_MECHANISM_INPUT" ]]
[[ -s "$CSD_LABELS" ]]
mkdir -p "$(dirname "$CSD_OUTPUT_PREFIX")"

CSD_EXPECTED_OUTPUTS=(
  "${CSD_OUTPUT_PREFIX}.png"
  "${CSD_OUTPUT_PREFIX}.pdf"
  "${CSD_OUTPUT_PREFIX}.csv"
  "${CSD_OUTPUT_PREFIX}_epsilon_sweep.csv"
  "${CSD_OUTPUT_PREFIX}_summary.csv"
)

if [[ -f "$CSD_PLOT_MARKER" ]]; then
  for CSD_OUTPUT in "${CSD_EXPECTED_OUTPUTS[@]}"; do
    [[ -s "$CSD_OUTPUT" ]]
  done
  exit 0
fi

"$CSD_TTT_PYTHON" scripts/clean_self_distill/07_plot_trust_region_mechanism.py \
  --posthoc "$CSD_MECHANISM_INPUT" \
  --labels "$CSD_LABELS" \
  --output-prefix "$CSD_OUTPUT_PREFIX" \
  --bootstrap-replicates 10000 \
  --seed 20260807

for CSD_OUTPUT in "${CSD_EXPECTED_OUTPUTS[@]}"; do
  [[ -s "$CSD_OUTPUT" ]]
done
CSD_MARKER_TMP="${CSD_PLOT_MARKER}.tmp.$$"
printf 'completed_at=%s\njob_id=%s\ninput=%s\noutput_prefix=%s\n' \
  "$(date -Is)" "$SLURM_JOB_ID" "$CSD_MECHANISM_INPUT" \
  "$CSD_OUTPUT_PREFIX" > "$CSD_MARKER_TMP"
mv "$CSD_MARKER_TMP" "$CSD_PLOT_MARKER"
