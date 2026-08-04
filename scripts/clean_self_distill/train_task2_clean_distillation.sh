#!/usr/bin/env bash
set -euo pipefail

# Paper Task 2 / CSD-SD:
#   construct the same query-local ridge teacher, distill it into a rank-8
#   student LoRA for 3 same-prefix steps, destroy teacher state, evaluate the
#   student, then reset exactly to the base checkpoint before the next query.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
MODEL_REVISION=${MODEL_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}
EVAL_DATA=${EVAL_DATA:-data/verl/amc23+aime24+aime25/val.parquet}
RUN_SEED=${RUN_SEED:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/clean_self_distill/task2_clean_distillation/seed_${RUN_SEED}}
PROPOSALS=${PROPOSALS:-${OUTPUT_ROOT}/eval_proposals.jsonl}
NUM_CANDIDATES=${NUM_CANDIDATES:-10}
MIN_ACCEPTED_CANDIDATES=${MIN_ACCEPTED_CANDIDATES:-4}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-8192}
FORCE_PROPOSE=${FORCE_PROPOSE:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-}

if [[ ! -f "$EVAL_DATA" ]]; then
  echo "Missing EVAL_DATA: $EVAL_DATA" >&2
  echo "Download it first with: python scripts/download_data.py --dataset amc23+aime24+aime25 --split val" >&2
  exit 1
fi

if [[ ! -f "$PROPOSALS" || "$FORCE_PROPOSE" == "1" ]]; then
  "$PYTHON_BIN" scripts/clean_self_distill/01_propose.py \
    --input "$EVAL_DATA" \
    --output "$PROPOSALS" \
    --model "$MODEL_PATH" \
    --revision "$MODEL_REVISION" \
    --num-candidates "$NUM_CANDIDATES" \
    --min-accepted-candidates "$MIN_ACCEPTED_CANDIDATES" \
    --proposal-oversample 2 \
    --max-rounds 4 \
    --temperature 0.8 \
    --solver-temperature 0.3 \
    --verifier-temperature 0 \
    --stage-max-attempts 2 \
    --top-p 0.95 \
    --seed "$RUN_SEED"
fi

EVAL_ARGS=()
if [[ -n "$MAX_EVAL_SAMPLES" ]]; then
  EVAL_ARGS+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
fi

"$PYTHON_BIN" scripts/clean_self_distill/03_train_eval.py \
  --mode task2 \
  --eval-data "$EVAL_DATA" \
  --proposals "$PROPOSALS" \
  --model "$MODEL_PATH" \
  --revision "$MODEL_REVISION" \
  --output-dir "$OUTPUT_ROOT/evaluation" \
  --ridge-lambda 0.1 \
  --residual-step-size 0.8 \
  --max-support-tokens 768 \
  --max-tokens-per-candidate 96 \
  --hard-negatives 8 \
  --reasoning-token-weight 0.25 \
  --answer-token-weight 1.0 \
  --frontier-positive-weight 8.0 \
  --frontier-negative-weight 8.0 \
  --frontier-max-tokens 24 \
  --frontier-negative-probability-floor 0.25 \
  --max-update-norm 2.0 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --distillation-steps 3 \
  --learning-rate 2e-5 \
  --weight-decay 0 \
  --distill-top-k 64 \
  --distill-temperature 1.0 \
  --distill-token-clip 0 \
  --train-temperature 0.8 \
  --train-max-new-tokens 512 \
  --eval-samples 1 \
  --eval-max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
  --eval-temperature 0 \
  --top-p 0.95 \
  --top-k 20 \
  --seed "$RUN_SEED" \
  "${EVAL_ARGS[@]}"

echo "Task 2 complete: $OUTPUT_ROOT/evaluation/summary.json"
