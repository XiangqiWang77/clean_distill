#!/usr/bin/env bash
# Submit one standalone phase of the three-method Acc@1 evaluation.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
[[ $# -le 1 ]]
CSD_PHASE=${1:-all}
CSD_ONLINE_CODE_ROOT=${CSD_ONLINE_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export CSD_RUN_CONFIG CSD_ONLINE_CODE_ROOT
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
cd "$CSD_ONLINE_CODE_ROOT"

csd_require_clean_checkpoint() {
  [[ -s "$CSD_RUN_ROOT/timebox12h/clean/checkpoints/episode_0064/checkpoint_manifest.json" ]]
}

csd_require_privileged_checkpoint() {
  [[ -s "$CSD_RUN_ROOT/timebox12h/privileged/checkpoints/episode_0064/checkpoint_manifest.json" ]]
}

case "$CSD_PHASE" in
  base)
    CSD_ARRAY=0-3%4
    ;;
  clean)
    csd_require_clean_checkpoint
    CSD_ARRAY=4-7%4
    ;;
  privileged)
    csd_require_privileged_checkpoint
    CSD_ARRAY=8-11%4
    ;;
  final)
    csd_require_clean_checkpoint
    csd_require_privileged_checkpoint
    CSD_ARRAY=4-11%4
    ;;
  all)
    csd_require_clean_checkpoint
    csd_require_privileged_checkpoint
    CSD_ARRAY=0-11%4
    ;;
  *)
    echo "Usage: $0 [base|clean|privileged|final|all]" >&2
    exit 2
    ;;
esac

mkdir -p "$CSD_RUN_ROOT/logs"
sbatch --parsable --export=ALL --array="$CSD_ARRAY" \
  --output="$CSD_RUN_ROOT/logs/timebox-main-%A_%a.out" \
  --error="$CSD_RUN_ROOT/logs/timebox-main-%A_%a.err" \
  scripts/clean_self_distill/slurm/timebox_main_eval.slurm
