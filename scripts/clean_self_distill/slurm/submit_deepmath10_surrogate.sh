#!/usr/bin/env bash
# Submit the highest-priority two-H100 Qwen3-8B DeepMath-10 mechanism study.
set -Eeuo pipefail

CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CONFIG=${1:-$CODE_ROOT/configs/clean_self_distill/deepmath10_surrogate.env}
source "$CONFIG"

[[ "$(sha256sum "$CSD_D10_DEEPMATH" | awk '{print $1}')" == "$CSD_D10_DEEPMATH_SHA256" ]]
[[ "$(sha256sum "$CSD_D10_MANIFEST" | awk '{print $1}')" == "$CSD_D10_MANIFEST_SHA256" ]]
[[ "$(jq -er '.schema_version' "$CSD_D10_MANIFEST")" == trsd-deepmath10-surrogate-study-v1 ]]
[[ "$(jq -er '.total' "$CSD_D10_MANIFEST")" == "$CSD_D10_EXPECTED_ROWS" ]]
[[ "$CSD_D10_NUM_SHARDS" == 2 ]]
[[ ! -e "$CSD_D10_RUN_ROOT/SUBMISSION.json" ]]

snapshot="$CSD_D10_RUN_ROOT/code_snapshot"
[[ ! -e "$snapshot" ]]
mkdir -p "$snapshot/scripts/clean_self_distill" "$CSD_D10_RUN_ROOT/logs"
cp -a "$CODE_ROOT/src" "$snapshot/src"
cp "$CODE_ROOT/scripts/clean_self_distill/33_deepmath10_surrogate_study.py" \
  "$CODE_ROOT/scripts/clean_self_distill/34_plot_deepmath10_surrogate_study.py" \
  "$snapshot/scripts/clean_self_distill/"

common_export="ALL,CSD_D10_CONFIG=$CONFIG,CSD_D10_EXEC_ROOT=$snapshot"
study_job=$(sbatch --parsable --nice=0 \
  --output="$CSD_D10_RUN_ROOT/logs/%A_%a-study.out" \
  --error="$CSD_D10_RUN_ROOT/logs/%A_%a-study.err" \
  --export="$common_export" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/deepmath10_surrogate_eval.slurm")
report_job=$(sbatch --parsable --dependency="afterok:$study_job" \
  --output="$CSD_D10_RUN_ROOT/logs/%j-report.out" \
  --error="$CSD_D10_RUN_ROOT/logs/%j-report.err" \
  --export="$common_export" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/deepmath10_surrogate_report.slurm")

STUDY_JOB="$study_job" REPORT_JOB="$report_job" CSD_D10_RUN_ROOT="$CSD_D10_RUN_ROOT" "$CSD_D10_PYTHON" -c \
  'import json,os,pathlib; p=pathlib.Path(os.environ["CSD_D10_RUN_ROOT"])/"SUBMISSION.json"; p.write_text(json.dumps({"schema_version":"trsd-deepmath10-submission-v1","study_array":os.environ["STUDY_JOB"],"report":os.environ["REPORT_JOB"],"peak_h100":2,"scheduler_nice":0,"queries":3116,"figures":["figure1_nuisance_vs_useful_signal","figure2_local_surrogate_reliability"]},indent=2)+"\n")'

printf 'study=%s report=%s peak_h100=2 nice=0 queries=3116 figures=2\n' "$study_job" "$report_job"
squeue -r -j "$study_job,$report_job" -o '%.18i %.12P %.30j %.2t %.10M %.10l %.6Q %R'
