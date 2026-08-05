#!/usr/bin/env bash
#SBATCH --job-name=csd8b-short
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

: "${SLURM_ARRAY_TASK_ID:?Short-term evaluation must be an array job}"
: "${CSD_RUN_CONFIG:?The empirical submitter must export CSD_RUN_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
# shellcheck disable=SC1091
source "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/empirical_common.sh"
csd_assert_model_gpu
trap csd_stop_requeue_watchdog EXIT
csd_start_requeue_watchdog "$CSD_EVAL_WALLTIME"

CSD_METHOD_COUNT=5
CSD_TOTAL_TASKS=$((CSD_METHOD_COUNT * CSD_EVAL_NUM_SHARDS))
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < CSD_TOTAL_TASKS ))
CSD_METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / CSD_EVAL_NUM_SHARDS))
CSD_SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % CSD_EVAL_NUM_SHARDS))
CSD_SHARD_LABEL=$(printf '%03d' "$CSD_SHARD_INDEX")

case "$CSD_METHOD_INDEX" in
  0) CSD_METHOD=base; CSD_PROGRAM=heldout; CSD_SUPPORT_VARIANT=none ;;
  1) CSD_METHOD=csd_t; CSD_PROGRAM=heldout; CSD_SUPPORT_VARIANT=correct_wrong_signed ;;
  2) CSD_METHOD=csd_t_correct_only; CSD_PROGRAM=heldout; CSD_SUPPORT_VARIANT=correct_only ;;
  3) CSD_METHOD=csd_sd; CSD_PROGRAM=short_distill; CSD_SUPPORT_VARIANT=correct_wrong_signed ;;
  4) CSD_METHOD=privileged_sd; CSD_PROGRAM=short_distill; CSD_SUPPORT_VARIANT=correct_wrong_signed ;;
  *) echo "Unexpected short-term method index $CSD_METHOD_INDEX" >&2; exit 2 ;;
esac

CSD_OUTPUT="$CSD_RUN_ROOT/short/predictions/$CSD_METHOD/shard_${CSD_SHARD_LABEL}.jsonl"
CSD_MARKER="$CSD_RUN_ROOT/status/short_${CSD_METHOD}_${CSD_SHARD_LABEL}.done"
mkdir -p "$(dirname "$CSD_OUTPUT")"
if [[ -f "$CSD_MARKER" ]]; then
  [[ -s "$CSD_OUTPUT" ]]
  exit 0
fi

CSD_COMMON_EVAL_ARGS=(
  --queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl"
  --output "$CSD_OUTPUT"
  --model "$CSD_MODEL_LOCAL_DIR"
  --model-id "$CSD_MODEL_ID"
  --revision "$CSD_MODEL_REVISION"
  --max-new-tokens "$CSD_EVAL_MAX_NEW_TOKENS"
  --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS"
  --context-window "$CSD_CONTEXT_WINDOW_TOKENS"
  --num-shards "$CSD_EVAL_NUM_SHARDS"
  --shard-index "$CSD_SHARD_INDEX"
  --seed "$CSD_RUN_SEED"
)
CSD_RIDGE_ARGS=(
  --ridge-lambda "$CSD_RIDGE_LAMBDA"
  --residual-step-size "$CSD_RESIDUAL_STEP_SIZE"
  --max-tokens-per-candidate "$CSD_MAX_TOKENS_PER_CANDIDATE"
  --max-support-tokens "$CSD_MAX_SUPPORT_TOKENS"
  --hard-negatives "$CSD_HARD_NEGATIVES"
  --max-sequence-tokens "$CSD_TRAIN_MAX_SEQUENCE_TOKENS"
  --reasoning-token-weight "$CSD_REASONING_TOKEN_WEIGHT"
  --answer-token-weight "$CSD_ANSWER_TOKEN_WEIGHT"
  --frontier-positive-weight "$CSD_FRONTIER_POSITIVE_WEIGHT"
  --frontier-negative-weight "$CSD_FRONTIER_NEGATIVE_WEIGHT"
  --frontier-target-margin "$CSD_FRONTIER_TARGET_MARGIN"
  --frontier-max-tokens "$CSD_FRONTIER_MAX_TOKENS"
  --frontier-negative-probability-floor "$CSD_FRONTIER_NEGATIVE_PROBABILITY_FLOOR"
  --max-update-norm "$CSD_MAX_UPDATE_NORM"
)

if [[ "$CSD_PROGRAM" == heldout ]]; then
  CSD_METHOD_ARGS=()
  if [[ "$CSD_METHOD" == csd_t* ]]; then
    CSD_METHOD_ARGS+=(
      --proposals "$CSD_RUN_ROOT/proposals/splits/heldout_proposals.jsonl"
      --support-variant "$CSD_SUPPORT_VARIANT"
      "${CSD_RIDGE_ARGS[@]}"
    )
  fi
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    "${CSD_COMMON_EVAL_ARGS[@]}" \
    --method "$CSD_METHOD" \
    --checkpoint-episode 0 \
    "${CSD_METHOD_ARGS[@]}"
else
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/06_short_term_eval.py \
    "${CSD_COMMON_EVAL_ARGS[@]}" \
    --method "$CSD_METHOD" \
    --proposals "$CSD_RUN_ROOT/proposals/splits/heldout_proposals.jsonl" \
    --support-variant "$CSD_SUPPORT_VARIANT" \
    --learning-rate "$CSD_LEARNING_RATE" \
    --weight-decay "$CSD_WEIGHT_DECAY" \
    --lora-rank "$CSD_LORA_RANK" \
    --lora-alpha "$CSD_LORA_ALPHA" \
    --distill-top-k "$CSD_DISTILL_TOP_K" \
    --distill-temperature "$CSD_DISTILL_TEMPERATURE" \
    "${CSD_RIDGE_ARGS[@]}"
fi

[[ -s "$CSD_OUTPUT" ]]
csd_atomic_marker "$CSD_MARKER"
