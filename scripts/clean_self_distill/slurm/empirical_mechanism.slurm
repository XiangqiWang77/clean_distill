#!/usr/bin/env bash
#SBATCH --job-name=csd8b-mechanism
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

: "${SLURM_ARRAY_TASK_ID:?Mechanism evaluation must be an array job}"
: "${CSD_RUN_CONFIG:?The empirical submitter must export CSD_RUN_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
# shellcheck disable=SC1091
source "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/empirical_common.sh"
csd_assert_gpu "$CSD_GPU_NAME_REGEX"
trap csd_stop_requeue_watchdog EXIT
csd_start_requeue_watchdog "$CSD_EVAL_WALLTIME"

CSD_TOTAL_TASKS=$((2 * CSD_EVAL_NUM_SHARDS))
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < CSD_TOTAL_TASKS ))
CSD_TEACHER_INDEX=$((SLURM_ARRAY_TASK_ID / CSD_EVAL_NUM_SHARDS))
CSD_SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % CSD_EVAL_NUM_SHARDS))
CSD_SHARD_LABEL=$(printf '%03d' "$CSD_SHARD_INDEX")
case "$CSD_TEACHER_INDEX" in
  0) CSD_TEACHER_TYPE=pre_decision_privilege ;;
  1) CSD_TEACHER_TYPE=post_outcome_privilege ;;
  *) echo "Unexpected mechanism teacher index $CSD_TEACHER_INDEX" >&2; exit 2 ;;
esac

CSD_OUTPUT="$CSD_RUN_ROOT/mechanism/raw/$CSD_TEACHER_TYPE/shard_${CSD_SHARD_LABEL}.jsonl"
CSD_MARKER="$CSD_RUN_ROOT/status/mechanism_${CSD_TEACHER_TYPE}_${CSD_SHARD_LABEL}.done"
mkdir -p "$(dirname "$CSD_OUTPUT")"
if [[ -f "$CSD_MARKER" ]]; then
  [[ -s "$CSD_OUTPUT" ]]
  exit 0
fi

CSD_BASE_ARGS=()
for ((index=0; index<CSD_EVAL_NUM_SHARDS; index++)); do
  label=$(printf '%03d' "$index")
  base_path="$CSD_RUN_ROOT/short/predictions/base/shard_${label}.jsonl"
  [[ -s "$base_path" ]]
  CSD_BASE_ARGS+=(--base-predictions "$base_path")
done
CSD_LABEL_ARGS=()
if [[ "$CSD_TEACHER_TYPE" == post_outcome_privilege ]]; then
  CSD_LABEL_ARGS+=(--labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl")
fi

"$CSD_TTT_PYTHON" scripts/clean_self_distill/07_mechanism_eval.py \
  --teacher-type "$CSD_TEACHER_TYPE" \
  --queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
  "${CSD_BASE_ARGS[@]}" \
  "${CSD_LABEL_ARGS[@]}" \
  --output "$CSD_OUTPUT" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --max-new-tokens "$CSD_EVAL_MAX_NEW_TOKENS" \
  --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS" \
  --context-window "$CSD_CONTEXT_WINDOW_TOKENS" \
  --num-shards "$CSD_EVAL_NUM_SHARDS" \
  --shard-index "$CSD_SHARD_INDEX" \
  --seed "$CSD_RUN_SEED"

[[ -s "$CSD_OUTPUT" ]]
csd_atomic_marker "$CSD_MARKER"
