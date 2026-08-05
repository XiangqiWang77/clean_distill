#!/usr/bin/env bash
# Offline label join for the three reported Acc@1 prediction sets.  The
# specialized teacher is temporary and is measured inside Clean-SD episodes.
set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
cd "$CSD_ONLINE_CODE_ROOT"
export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CSD_DEST="$CSD_RUN_ROOT/timebox12h/main_eval"
for CSD_METHOD in base clean_sd privileged_sd; do
  CSD_PREDICTION_ARGS=()
  for CSD_SHARD_INDEX in 0 1 2 3; do
    CSD_SHARD_LABEL=$(printf '%02d' "$CSD_SHARD_INDEX")
    CSD_PREDICTION="$CSD_DEST/predictions/$CSD_METHOD/shard_${CSD_SHARD_LABEL}.jsonl"
    CSD_DONE="$CSD_DEST/status/${CSD_METHOD}_shard_${CSD_SHARD_LABEL}.done"
    [[ -s "$CSD_PREDICTION" && -s "$CSD_DONE" ]]
    CSD_PREDICTION_ARGS+=(--predictions "$CSD_PREDICTION")
  done
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    "${CSD_PREDICTION_ARGS[@]}" \
    --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
    --sample-count 1 \
    --output "$CSD_DEST/scored/${CSD_METHOD}.jsonl"
done
