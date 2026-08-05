#!/usr/bin/env bash
# Submit one standalone phase of the four-method Acc@1 evaluation.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
[[ $# -le 1 ]]
CSD_PHASE=${1:-all}
CSD_ONLINE_CODE_ROOT=${CSD_ONLINE_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export CSD_RUN_CONFIG CSD_ONLINE_CODE_ROOT
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
cd "$CSD_ONLINE_CODE_ROOT"
export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CSD_PROBE_ROOT="$CSD_RUN_ROOT/timebox12h/heldout_probes"
csd_merge_probes() {
  [[ -s "$CSD_PROBE_ROOT/shard_00.done" && -s "$CSD_PROBE_ROOT/shard_01.done" ]]
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/merge_empirical_proposals.py \
    --queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
    --shard "$CSD_PROBE_ROOT/shard_00.jsonl" \
    --shard "$CSD_PROBE_ROOT/shard_01.jsonl" \
    --bind-by-problem-sha256 \
    --output "$CSD_PROBE_ROOT/merged.jsonl" \
    --manifest "$CSD_PROBE_ROOT/merged.manifest.json"
}

csd_require_final_checkpoints() {
  [[ -s "$CSD_RUN_ROOT/timebox12h/clean/checkpoints/episode_0064/checkpoint_manifest.json" ]]
  [[ -s "$CSD_RUN_ROOT/timebox12h/privileged/checkpoints/episode_0064/checkpoint_manifest.json" ]]
}

case "$CSD_PHASE" in
  early)
    csd_merge_probes
    CSD_ARRAY=0-7%2
    ;;
  final)
    csd_require_final_checkpoints
    CSD_ARRAY=8-15%4
    ;;
  all)
    csd_merge_probes
    csd_require_final_checkpoints
    CSD_ARRAY=0-15%4
    ;;
  *)
    echo "Usage: $0 [early|final|all]" >&2
    exit 2
    ;;
esac

sbatch --parsable --export=ALL --array="$CSD_ARRAY" \
  scripts/clean_self_distill/slurm/timebox_main_eval.slurm
