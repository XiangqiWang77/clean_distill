#!/usr/bin/env bash
# Submit the one-query H100 collector and its afterok CPU renderer.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_TRSD_FINAL_CHECKPOINT:?export the adopted rolling-36 or completed final checkpoint}"

CSD_MECHANISM_ROOT=${CSD_MECHANISM_ROOT:-}
CSD_EXPORTS="ALL,CSD_RUN_CONFIG=$CSD_RUN_CONFIG,CSD_ONLINE_CODE_ROOT=$CSD_ONLINE_CODE_ROOT,CSD_TRSD_FINAL_CHECKPOINT=$CSD_TRSD_FINAL_CHECKPOINT"
if [[ -n "$CSD_MECHANISM_ROOT" ]]; then
  CSD_EXPORTS="$CSD_EXPORTS,CSD_MECHANISM_ROOT=$CSD_MECHANISM_ROOT"
fi

CSD_COLLECT_JOB_REF=$(sbatch --parsable --export="$CSD_EXPORTS" \
  scripts/clean_self_distill/slurm/trust_region_mechanism_collect.slurm)
CSD_COLLECT_JOB_ID=${CSD_COLLECT_JOB_REF%%;*}
CSD_PLOT_JOB_REF=$(sbatch --parsable --dependency="afterok:$CSD_COLLECT_JOB_ID" \
  --export="$CSD_EXPORTS" \
  scripts/clean_self_distill/slurm/trust_region_mechanism_plot.slurm)
CSD_PLOT_JOB_ID=${CSD_PLOT_JOB_REF%%;*}

printf 'collector_job_id=%s\nplot_job_id=%s\n' \
  "$CSD_COLLECT_JOB_ID" "$CSD_PLOT_JOB_ID"
