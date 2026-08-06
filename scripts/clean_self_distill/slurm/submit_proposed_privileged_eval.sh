#!/usr/bin/env bash
# Submit independent Self-Proposed Privileged-SD held-out evaluations.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
[[ $# -eq 1 ]]
CSD_PHASE=$1
CSD_ONLINE_CODE_ROOT=${CSD_ONLINE_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export CSD_RUN_CONFIG CSD_ONLINE_CODE_ROOT
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
cd "$CSD_ONLINE_CODE_ROOT"
mkdir -p "$CSD_RUN_ROOT/logs"

case "$CSD_PHASE" in
  main)
    [[ -s "$CSD_RUN_ROOT/timebox12h/proposed_privileged/checkpoints/episode_0064/checkpoint_manifest.json" ]]
    sbatch --parsable --export=ALL \
      --output="$CSD_RUN_ROOT/logs/proposed-privileged-main-%A_%a.out" \
      --error="$CSD_RUN_ROOT/logs/proposed-privileged-main-%A_%a.err" \
      scripts/clean_self_distill/slurm/proposed_privileged_main_eval.slurm
    ;;
  horizon)
    for CSD_EPISODE in 0016 0032 0048; do
      [[ -s "$CSD_RUN_ROOT/timebox12h/proposed_privileged/checkpoints/episode_${CSD_EPISODE}/checkpoint_manifest.json" ]]
    done
    sbatch --parsable --export=ALL \
      --output="$CSD_RUN_ROOT/logs/proposed-privileged-horizon-%A_%a.out" \
      --error="$CSD_RUN_ROOT/logs/proposed-privileged-horizon-%A_%a.err" \
      scripts/clean_self_distill/slurm/proposed_privileged_horizon_eval.slurm
    ;;
  *)
    echo "Usage: $0 main|horizon" >&2
    exit 2
    ;;
esac
