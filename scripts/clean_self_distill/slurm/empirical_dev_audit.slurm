#!/usr/bin/env bash
#SBATCH --job-name=csd8b-dev-audit
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

CSD_OUTPUT="$CSD_RUN_ROOT/dev/dev_audit.json"
CSD_MARKER="$CSD_RUN_ROOT/status/dev_audit.done"
if [[ -f "$CSD_MARKER" ]]; then
  [[ -s "$CSD_OUTPUT" ]]
  exit 0
fi

"$CSD_TTT_PYTHON" scripts/clean_self_distill/09_build_dev_audit.py \
  --queries "$CSD_PREPARED_ROOT/dev_queries.jsonl" \
  --proposals "$CSD_RUN_ROOT/proposals/splits/dev_proposals.jsonl" \
  --output "$CSD_OUTPUT" \
  --ridge-lambda "$CSD_RIDGE_LAMBDA" \
  --residual-step-size "$CSD_RESIDUAL_STEP_SIZE" \
  --reasoning-token-weight "$CSD_REASONING_TOKEN_WEIGHT" \
  --answer-token-weight "$CSD_ANSWER_TOKEN_WEIGHT" \
  --frontier-positive-weight "$CSD_FRONTIER_POSITIVE_WEIGHT" \
  --frontier-negative-weight "$CSD_FRONTIER_NEGATIVE_WEIGHT" \
  --frontier-target-margin "$CSD_FRONTIER_TARGET_MARGIN" \
  --max-support-tokens "$CSD_MAX_SUPPORT_TOKENS" \
  --max-tokens-per-candidate "$CSD_MAX_TOKENS_PER_CANDIDATE" \
  --num-candidates "$CSD_NUM_CANDIDATES" \
  --minimum-accepted-candidates "$CSD_MIN_ACCEPTED_CANDIDATES" \
  --learning-rate "$CSD_LEARNING_RATE" \
  --training-max-sequence-tokens "$CSD_TRAIN_MAX_SEQUENCE_TOKENS" \
  --distill-token-chunk-size "$CSD_DISTILL_TOKEN_CHUNK_SIZE" \
  --evaluation-max-new-tokens "$CSD_EVAL_MAX_NEW_TOKENS" \
  --evaluation-temperature "$CSD_EVAL_TEMPERATURE" \
  --evaluation-top-p "$CSD_EVAL_TOP_P" \
  --evaluation-top-k "$CSD_EVAL_TOP_K"

[[ -s "$CSD_OUTPUT" ]]
csd_atomic_marker "$CSD_MARKER"
