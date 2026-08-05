#!/usr/bin/env bash
#SBATCH --job-name=csd8b-propose
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --requeue

set -Eeuo pipefail

: "${SLURM_ARRAY_TASK_ID:?Proposal generation must be an array job}"
: "${CSD_RUN_CONFIG:?The empirical submitter must export CSD_RUN_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
# shellcheck disable=SC1091
source "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/empirical_common.sh"
csd_assert_gpu "$CSD_GPU_NAME_REGEX"
trap csd_stop_requeue_watchdog EXIT
csd_start_requeue_watchdog "$CSD_PROPOSAL_WALLTIME"

CSD_SHARD_INDEX=$SLURM_ARRAY_TASK_ID
(( CSD_SHARD_INDEX >= 0 && CSD_SHARD_INDEX < CSD_PROPOSAL_NUM_SHARDS ))
CSD_LABEL=$(printf '%03d' "$CSD_SHARD_INDEX")
CSD_OUTPUT="$CSD_RUN_ROOT/proposals/shards/proposals_${CSD_LABEL}.jsonl"
CSD_MARKER="$CSD_RUN_ROOT/status/proposals_${CSD_LABEL}.done"
mkdir -p "$(dirname "$CSD_OUTPUT")"
if [[ -f "$CSD_MARKER" ]]; then
  [[ -s "$CSD_OUTPUT" ]]
  exit 0
fi

# Each accepted row is appended atomically enough for the proposer's strict
# ordered-prefix resume check.  Requeued allocations never repeat completed
# queries and never widen the information boundary beyond query-only JSONL.
"$CSD_TTT_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
  repair-proposals --path "$CSD_OUTPUT"
"$CSD_TTT_PYTHON" scripts/clean_self_distill/01_propose.py \
  --input "$CSD_PREPARED_ROOT/proposal_queries.jsonl" \
  --output "$CSD_OUTPUT" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --revision "$CSD_MODEL_REVISION" \
  --num-candidates "$CSD_NUM_CANDIDATES" \
  --proposal-oversample "$CSD_PROPOSAL_OVERSAMPLE" \
  --max-rounds "$CSD_PROPOSAL_MAX_ROUNDS" \
  --min-accepted-candidates "$CSD_MIN_ACCEPTED_CANDIDATES" \
  --max-new-tokens 1536 \
  --temperature 0.8 \
  --solver-temperature 0.3 \
  --verifier-temperature 0 \
  --stage-max-attempts 2 \
  --top-p 0.95 \
  --num-shards "$CSD_PROPOSAL_NUM_SHARDS" \
  --shard-index "$CSD_SHARD_INDEX" \
  --resume \
  --seed "$CSD_RUN_SEED"

csd_atomic_marker "$CSD_MARKER"
