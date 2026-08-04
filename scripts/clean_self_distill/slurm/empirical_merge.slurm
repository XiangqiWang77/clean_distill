#!/usr/bin/env bash
#SBATCH --job-name=csd8b-merge
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?The empirical submitter must export CSD_RUN_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
# shellcheck disable=SC1091
source "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/empirical_common.sh"
csd_assert_gpu 'RTX.*6000|PRO.*6000'

CSD_OUTPUT="$CSD_RUN_ROOT/proposals/proposals.jsonl"
CSD_MANIFEST="$CSD_RUN_ROOT/proposals/manifest.json"
CSD_MARKER="$CSD_RUN_ROOT/status/proposals_merged.done"
if [[ -f "$CSD_MARKER" ]]; then
  [[ -s "$CSD_OUTPUT" && -s "$CSD_MANIFEST" ]]
  exit 0
fi

CSD_SHARD_ARGS=()
for ((index=0; index<CSD_PROPOSAL_NUM_SHARDS; index++)); do
  label=$(printf '%03d' "$index")
  shard="$CSD_RUN_ROOT/proposals/shards/proposals_${label}.jsonl"
  [[ -s "$shard" ]]
  CSD_SHARD_ARGS+=(--shard "$shard")
done
"$CSD_TTT_PYTHON" scripts/clean_self_distill/merge_empirical_proposals.py \
  --queries "$CSD_PREPARED_ROOT/proposal_queries.jsonl" \
  "${CSD_SHARD_ARGS[@]}" \
  --output "$CSD_OUTPUT" \
  --manifest "$CSD_MANIFEST"

"$CSD_TTT_PYTHON" scripts/clean_self_distill/split_empirical_proposals.py \
  --proposals "$CSD_OUTPUT" \
  --distill-queries "$CSD_PREPARED_ROOT/distill_queries.jsonl" \
  --dev-queries "$CSD_PREPARED_ROOT/dev_queries.jsonl" \
  --heldout-queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
  --output-dir "$CSD_RUN_ROOT/proposals/splits" \
  --manifest "$CSD_RUN_ROOT/proposals/splits/manifest.json"

csd_atomic_marker "$CSD_MARKER"
