#!/usr/bin/env bash
# Merge completed held-out probes and submit the standalone four-method Acc@1 array.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
CSD_ONLINE_CODE_ROOT=${CSD_ONLINE_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export CSD_RUN_CONFIG CSD_ONLINE_CODE_ROOT
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
cd "$CSD_ONLINE_CODE_ROOT"
export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CSD_PROBE_ROOT="$CSD_RUN_ROOT/timebox12h/heldout_probes"
[[ -s "$CSD_PROBE_ROOT/shard_00.done" && -s "$CSD_PROBE_ROOT/shard_01.done" ]]
[[ -s "$CSD_RUN_ROOT/timebox12h/clean/checkpoints/episode_0064/checkpoint_manifest.json" ]]
[[ -s "$CSD_RUN_ROOT/timebox12h/privileged/checkpoints/episode_0064/checkpoint_manifest.json" ]]

"$CSD_TTT_PYTHON" scripts/clean_self_distill/merge_empirical_proposals.py \
  --queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
  --shard "$CSD_PROBE_ROOT/shard_00.jsonl" \
  --shard "$CSD_PROBE_ROOT/shard_01.jsonl" \
  --bind-by-problem-sha256 \
  --output "$CSD_PROBE_ROOT/merged.jsonl" \
  --manifest "$CSD_PROBE_ROOT/merged.manifest.json"

sbatch --parsable --export=ALL \
  scripts/clean_self_distill/slurm/timebox_main_eval.slurm

