#!/usr/bin/env bash
# Submit Base/TRSD-16/TRSD-32/TRSD-36/Privileged-64 at a 10,240-token cap.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
CSD_ONLINE_CODE_ROOT=${CSD_ONLINE_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export CSD_RUN_CONFIG CSD_ONLINE_CODE_ROOT
export CSD_TRSD_EVAL_MAX_NEW_TOKENS=${CSD_TRSD_EVAL_MAX_NEW_TOKENS:-10240}
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"

for CSD_CHECKPOINT in \
  "$CSD_TIMEBOX_OUTPUT/checkpoints/episode_0016" \
  "$CSD_TIMEBOX_OUTPUT/checkpoints/episode_0032" \
  "$CSD_TIMEBOX_OUTPUT/checkpoints/rolling_episode_0036" \
  "/home/da839/scratch_pi_mg269/da839/clean_distill/runs/csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/checkpoints/episode_0064"; do
  [[ -s "$CSD_CHECKPOINT/checkpoint_manifest.json" ]]
done

cd "$CSD_ONLINE_CODE_ROOT"
mkdir -p "$CSD_RUN_ROOT/logs"
sbatch --parsable \
  --export="ALL,CSD_RUN_CONFIG=$CSD_RUN_CONFIG,CSD_ONLINE_CODE_ROOT=$CSD_ONLINE_CODE_ROOT,CSD_TRSD_EVAL_MAX_NEW_TOKENS=$CSD_TRSD_EVAL_MAX_NEW_TOKENS" \
  --output="$CSD_RUN_ROOT/logs/trsd-main-table-%A_%a.out" \
  --error="$CSD_RUN_ROOT/logs/trsd-main-table-%A_%a.err" \
  scripts/clean_self_distill/slurm/trust_region_checkpoint_eval.slurm
