#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT=/home/da839/clean_distill
CONFIG=${CSD_A2_CONFIG:-$CODE_ROOT/configs/clean_self_distill/qwen3_8b_component_ablations64.env}
# shellcheck disable=SC1090
source "$CONFIG"
mkdir -p "$CSD_A2_RUN_ROOT/logs"
export CSD_A2_CONFIG="$CONFIG"
dependency=${CSD_A2_DEPENDENCY:-afterok:21781795:21783044}

array_job=$(sbatch --parsable --dependency="$dependency" \
  --export=ALL,CSD_A2_CONFIG="$CONFIG" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/qwen8_component_ablation64_b200.slurm")
report_job=$(sbatch --parsable --dependency="afterok:$array_job" \
  --export=ALL,CSD_A2_CONFIG="$CONFIG" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/qwen8_component_ablation64_report.slurm")

printf '{"array_job":"%s","report_job":"%s","episodes":64,"max_h100":4,"dependency":"%s","conda_env":"TTT"}\n' \
  "$array_job" "$report_job" "$dependency" > "$CSD_A2_RUN_ROOT/SUBMISSION.json"
printf 'array=%s report=%s\n' "$array_job" "$report_job"
squeue -r -j "$array_job,$report_job" -o '%.18i %.12P %.24j %.2t %.10M %.10l %R'
