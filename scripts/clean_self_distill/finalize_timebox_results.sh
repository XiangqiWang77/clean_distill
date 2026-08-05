#!/usr/bin/env bash
# Score completed predictions and build the 12-hour Clean-SD main table.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
CSD_ONLINE_CODE_ROOT=${CSD_ONLINE_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export CSD_RUN_CONFIG CSD_ONLINE_CODE_ROOT
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
cd "$CSD_ONLINE_CODE_ROOT"
export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

bash scripts/clean_self_distill/score_timebox_main.sh
bash scripts/clean_self_distill/score_timebox_horizon.sh

CSD_TIMEBOX="$CSD_RUN_ROOT/timebox12h"
CSD_RESULTS="$CSD_TIMEBOX/results"
mkdir -p "$CSD_RESULTS"
CSD_RESOURCE_ARGS=()
if [[ -n "${CSD_SLURM_ACCOUNTING:-}" ]]; then
  [[ -s "$CSD_SLURM_ACCOUNTING" ]]
  CSD_RESOURCE_ARGS+=(--slurm-accounting "$CSD_SLURM_ACCOUNTING")
fi

"$CSD_TTT_PYTHON" scripts/clean_self_distill/report_timebox_efficiency.py \
  --timebox-dir "$CSD_TIMEBOX" \
  "${CSD_RESOURCE_ARGS[@]}" \
  --json-output "$CSD_RESULTS/efficiency.json" \
  --markdown-output "$CSD_RESULTS/efficiency.md"

"$CSD_TTT_PYTHON" scripts/clean_self_distill/report_timebox_main_table.py \
  --base-scored "$CSD_TIMEBOX/main_eval/scored/base.jsonl" \
  --csd-t-scored "$CSD_TIMEBOX/main_eval/scored/csd_t.jsonl" \
  --clean64-scored "$CSD_TIMEBOX/main_eval/scored/clean_sd.jsonl" \
  --privileged64-scored "$CSD_TIMEBOX/main_eval/scored/privileged_sd.jsonl" \
  --clean16-scored "$CSD_TIMEBOX/horizon_eval/scored/clean_sd/episode_0016.jsonl" \
  --clean32-scored "$CSD_TIMEBOX/horizon_eval/scored/clean_sd/episode_0032.jsonl" \
  --clean48-scored "$CSD_TIMEBOX/horizon_eval/scored/clean_sd/episode_0048.jsonl" \
  --privileged16-scored "$CSD_TIMEBOX/horizon_eval/scored/privileged_sd/episode_0016.jsonl" \
  --privileged32-scored "$CSD_TIMEBOX/horizon_eval/scored/privileged_sd/episode_0032.jsonl" \
  --privileged48-scored "$CSD_TIMEBOX/horizon_eval/scored/privileged_sd/episode_0048.jsonl" \
  --clean-journal "$CSD_TIMEBOX/clean/episodes.jsonl" \
  --privileged-journal "$CSD_TIMEBOX/privileged/episodes.jsonl" \
  --clean-proposals "$CSD_TIMEBOX/clean/online_proposals.jsonl" \
  --resource-report "$CSD_RESULTS/efficiency.json" \
  --json-output "$CSD_RESULTS/main_table.json" \
  --markdown-output "$CSD_RESULTS/main_table.md"

printf 'main_table=%s\n' "$CSD_RESULTS/main_table.md"
