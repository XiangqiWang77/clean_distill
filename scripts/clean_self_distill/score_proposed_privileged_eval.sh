#!/usr/bin/env bash
# Offline label join for isolated Self-Proposed Privileged-SD predictions.
set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
[[ $# -eq 1 ]]
CSD_PHASE=$1
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
cd "$CSD_ONLINE_CODE_ROOT"
export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

csd_score_shards() {
  local CSD_SOURCE_DIR=$1
  local CSD_OUTPUT=$2
  local CSD_STATUS_PREFIX=$3
  local CSD_PREDICTION_ARGS=()
  local CSD_SHARD_INDEX CSD_SHARD_LABEL CSD_PREDICTION CSD_DONE
  for CSD_SHARD_INDEX in 0 1 2 3; do
    CSD_SHARD_LABEL=$(printf '%02d' "$CSD_SHARD_INDEX")
    CSD_PREDICTION="$CSD_SOURCE_DIR/shard_${CSD_SHARD_LABEL}.jsonl"
    CSD_DONE="${CSD_STATUS_PREFIX}${CSD_SHARD_LABEL}.done"
    [[ -s "$CSD_PREDICTION" && -s "$CSD_DONE" ]]
    CSD_PREDICTION_ARGS+=(--predictions "$CSD_PREDICTION")
  done
  mkdir -p "$(dirname "$CSD_OUTPUT")"
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    "${CSD_PREDICTION_ARGS[@]}" \
    --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
    --sample-count 1 \
    --output "$CSD_OUTPUT"
}

case "$CSD_PHASE" in
  main)
    CSD_DEST="$CSD_RUN_ROOT/timebox12h/proposed_privileged_main_eval"
    csd_score_shards \
      "$CSD_DEST/predictions/proposed_privileged_sd" \
      "$CSD_DEST/scored/proposed_privileged_sd.jsonl" \
      "$CSD_DEST/status/proposed_privileged_sd_shard_"
    ;;
  horizon)
    CSD_DEST="$CSD_RUN_ROOT/timebox12h/proposed_privileged_horizon_eval"
    for CSD_EPISODE in 0016 0032 0048; do
      csd_score_shards \
        "$CSD_DEST/predictions/proposed_privileged_sd/episode_${CSD_EPISODE}" \
        "$CSD_DEST/scored/proposed_privileged_sd/episode_${CSD_EPISODE}.jsonl" \
        "$CSD_DEST/status/proposed_privileged_sd_episode_${CSD_EPISODE}_shard_"
    done
    ;;
  *)
    echo "Usage: $0 main|horizon" >&2
    exit 2
    ;;
esac
