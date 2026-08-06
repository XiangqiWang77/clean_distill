#!/usr/bin/env bash
# Offline label join for intermediate Clean-SD checkpoints.  Self-Proposed
# Privileged-SD has a separate scorer; legacy privileged_sd scores are untouched.
set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
cd "$CSD_ONLINE_CODE_ROOT"
export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CSD_DEST="$CSD_RUN_ROOT/timebox12h/horizon_eval"
for CSD_METHOD in clean_sd; do
  for CSD_EPISODE in 0016 0032 0048; do
    CSD_PREDICTION_ARGS=()
    for CSD_SHARD_INDEX in 0 1 2 3; do
      CSD_SHARD_LABEL=$(printf '%02d' "$CSD_SHARD_INDEX")
      CSD_PREDICTION="$CSD_DEST/predictions/$CSD_METHOD/episode_${CSD_EPISODE}/shard_${CSD_SHARD_LABEL}.jsonl"
      CSD_DONE="$CSD_DEST/status/${CSD_METHOD}_episode_${CSD_EPISODE}_shard_${CSD_SHARD_LABEL}.done"
      [[ -s "$CSD_PREDICTION" && -s "$CSD_DONE" ]]
      CSD_PREDICTION_ARGS+=(--predictions "$CSD_PREDICTION")
    done
    "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
      "${CSD_PREDICTION_ARGS[@]}" \
      --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
      --sample-count 1 \
      --output "$CSD_DEST/scored/$CSD_METHOD/episode_${CSD_EPISODE}.jsonl"
  done
done
