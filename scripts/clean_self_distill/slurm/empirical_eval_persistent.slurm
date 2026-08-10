#!/usr/bin/env bash
#SBATCH --job-name=csd8b-long-eval
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --requeue

set -Eeuo pipefail

: "${SLURM_ARRAY_TASK_ID:?Persistent checkpoint evaluation must be an array job}"
: "${CSD_RUN_CONFIG:?The empirical submitter must export CSD_RUN_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
# shellcheck disable=SC1091
source "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/empirical_common.sh"
csd_assert_model_gpu
trap csd_stop_requeue_watchdog EXIT
csd_start_requeue_watchdog "$CSD_EVAL_WALLTIME"

CSD_CHECKPOINTS=(250 500 750 1000)
CSD_TASKS_PER_BRANCH=$((${#CSD_CHECKPOINTS[@]} * CSD_EVAL_NUM_SHARDS))
CSD_TOTAL_TASKS=$((2 * CSD_TASKS_PER_BRANCH))
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < CSD_TOTAL_TASKS ))
CSD_BRANCH_INDEX=$((SLURM_ARRAY_TASK_ID / CSD_TASKS_PER_BRANCH))
CSD_BRANCH_OFFSET=$((SLURM_ARRAY_TASK_ID % CSD_TASKS_PER_BRANCH))
CSD_CHECKPOINT_INDEX=$((CSD_BRANCH_OFFSET / CSD_EVAL_NUM_SHARDS))
CSD_SHARD_INDEX=$((CSD_BRANCH_OFFSET % CSD_EVAL_NUM_SHARDS))
CSD_EPISODE=${CSD_CHECKPOINTS[$CSD_CHECKPOINT_INDEX]}
CSD_EPISODE_LABEL=$(printf '%04d' "$CSD_EPISODE")
CSD_SHARD_LABEL=$(printf '%03d' "$CSD_SHARD_INDEX")

case "$CSD_BRANCH_INDEX" in
  0) CSD_METHOD=trsd ;;
  1) CSD_METHOD=privileged_sd ;;
  *) echo "Unexpected persistent branch index $CSD_BRANCH_INDEX" >&2; exit 2 ;;
esac

CSD_CHECKPOINT="$CSD_RUN_ROOT/persistent/$CSD_METHOD/checkpoints/episode_${CSD_EPISODE_LABEL}"
CSD_CHECKPOINT_MANIFEST="$CSD_CHECKPOINT/checkpoint_manifest.json"
CSD_OUTPUT="$CSD_RUN_ROOT/persistent_eval/predictions/$CSD_METHOD/episode_${CSD_EPISODE_LABEL}/shard_${CSD_SHARD_LABEL}.jsonl"
CSD_MARKER="$CSD_RUN_ROOT/status/long_${CSD_METHOD}_${CSD_EPISODE_LABEL}_${CSD_SHARD_LABEL}.done"
[[ -s "$CSD_CHECKPOINT_MANIFEST" ]]
mkdir -p "$(dirname "$CSD_OUTPUT")"
if [[ -f "$CSD_MARKER" ]]; then
  [[ -s "$CSD_OUTPUT" ]]
  exit 0
fi

"$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
  --queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
  --output "$CSD_OUTPUT" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --adapter "$CSD_CHECKPOINT" \
  --method "$CSD_METHOD" \
  --checkpoint-episode "$CSD_EPISODE" \
  --max-new-tokens "$CSD_EVAL_MAX_NEW_TOKENS" \
  --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS" \
  --context-window "$CSD_CONTEXT_WINDOW_TOKENS" \
  --num-shards "$CSD_EVAL_NUM_SHARDS" \
  --shard-index "$CSD_SHARD_INDEX" \
  --seed "$CSD_RUN_SEED"

[[ -s "$CSD_OUTPUT" ]]
csd_atomic_marker "$CSD_MARKER"
