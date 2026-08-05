#!/usr/bin/env bash
# Submit the independent episode-16/32/48 held-out accuracy array.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
CSD_ONLINE_CODE_ROOT=${CSD_ONLINE_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export CSD_RUN_CONFIG CSD_ONLINE_CODE_ROOT
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"

for CSD_BRANCH in clean privileged; do
  for CSD_EPISODE in 0016 0032 0048; do
    [[ -s "$CSD_RUN_ROOT/timebox12h/$CSD_BRANCH/checkpoints/episode_${CSD_EPISODE}/checkpoint_manifest.json" ]]
  done
done

cd "$CSD_ONLINE_CODE_ROOT"
sbatch --parsable --export=ALL \
  scripts/clean_self_distill/slurm/timebox_horizon_eval.slurm

