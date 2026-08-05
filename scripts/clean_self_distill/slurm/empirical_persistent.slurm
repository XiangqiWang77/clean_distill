#!/usr/bin/env bash
#SBATCH --job-name=csd8b-persist
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@3600

set -Eeuo pipefail

: "${SLURM_ARRAY_TASK_ID:?Persistent training must be an array job}"
: "${CSD_RUN_CONFIG:?The empirical submitter must export CSD_RUN_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
# shellcheck disable=SC1091
source "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/empirical_common.sh"
csd_assert_model_gpu

case "$SLURM_ARRAY_TASK_ID" in
  0)
    CSD_BRANCH=clean
    CSD_VARIANT=correct_wrong_signed
    CSD_METHOD=clean_sd
    ;;
  1)
    CSD_BRANCH=privileged
    CSD_VARIANT=correct_wrong_signed
    CSD_METHOD=privileged_sd
    ;;
  *) echo "Unexpected persistent array index $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

CSD_OUTPUT="$CSD_RUN_ROOT/persistent/$CSD_METHOD"
CSD_COMPLETE="$CSD_OUTPUT/COMPLETE.json"
if [[ -s "$CSD_COMPLETE" ]]; then
  exit 0
fi
mkdir -p "$CSD_OUTPUT"
CSD_RESUME_ARGS=()
if [[ -s "$CSD_OUTPUT/run_manifest.json" ]]; then
  CSD_RESUME_ARGS+=(--resume)
fi

CSD_CHILD_PID=
csd_forward_usr1() {
  if [[ -n "$CSD_CHILD_PID" ]]; then
    kill -USR1 "$CSD_CHILD_PID" 2>/dev/null || true
  fi
}
trap csd_forward_usr1 USR1

"$CSD_TTT_PYTHON" scripts/clean_self_distill/04_persistent_train.py \
  --branch "$CSD_BRANCH" \
  --variant "$CSD_VARIANT" \
  --queries "$CSD_PREPARED_ROOT/distill_queries.jsonl" \
  --proposals "$CSD_RUN_ROOT/proposals/splits/distill_proposals.jsonl" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --output-dir "$CSD_OUTPUT" \
  --episodes "$CSD_DISTILL_EPISODES" \
  --scientific-checkpoints "$CSD_SCIENTIFIC_CHECKPOINTS" \
  --rolling-checkpoint-interval "$CSD_ROLLING_CHECKPOINT_INTERVAL" \
  --max-sequence-tokens "$CSD_TRAIN_MAX_SEQUENCE_TOKENS" \
  --max-rollout-tokens "$CSD_TRAIN_MAX_SEQUENCE_TOKENS" \
  --learning-rate "$CSD_LEARNING_RATE" \
  --weight-decay "$CSD_WEIGHT_DECAY" \
  --lora-rank "$CSD_LORA_RANK" \
  --lora-alpha "$CSD_LORA_ALPHA" \
  --seed "$CSD_RUN_SEED" \
  --train-temperature "$CSD_EVAL_TEMPERATURE" \
  --top-p "$CSD_EVAL_TOP_P" \
  --top-k "$CSD_EVAL_TOP_K" \
  --distill-top-k "$CSD_DISTILL_TOP_K" \
  --distill-temperature "$CSD_DISTILL_TEMPERATURE" \
  --ridge-lambda "$CSD_RIDGE_LAMBDA" \
  --residual-step-size "$CSD_RESIDUAL_STEP_SIZE" \
  --max-tokens-per-candidate "$CSD_MAX_TOKENS_PER_CANDIDATE" \
  --max-support-tokens "$CSD_MAX_SUPPORT_TOKENS" \
  --hard-negatives "$CSD_HARD_NEGATIVES" \
  --ridge-max-length "$CSD_TRAIN_MAX_SEQUENCE_TOKENS" \
  --reasoning-token-weight "$CSD_REASONING_TOKEN_WEIGHT" \
  --answer-token-weight "$CSD_ANSWER_TOKEN_WEIGHT" \
  --frontier-positive-weight "$CSD_FRONTIER_POSITIVE_WEIGHT" \
  --frontier-negative-weight "$CSD_FRONTIER_NEGATIVE_WEIGHT" \
  --frontier-target-margin "$CSD_FRONTIER_TARGET_MARGIN" \
  --frontier-max-tokens "$CSD_FRONTIER_MAX_TOKENS" \
  --frontier-negative-probability-floor "$CSD_FRONTIER_NEGATIVE_PROBABILITY_FLOOR" \
  --max-update-norm "$CSD_MAX_UPDATE_NORM" \
  "${CSD_RESUME_ARGS[@]}" &
CSD_CHILD_PID=$!

set +e
wait "$CSD_CHILD_PID"
CSD_RC=$?
set -e
CSD_CHILD_PID=

if (( CSD_RC == 75 )); then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
(( CSD_RC == 0 )) || exit "$CSD_RC"
[[ -s "$CSD_COMPLETE" ]]
