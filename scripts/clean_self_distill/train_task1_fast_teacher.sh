#!/usr/bin/env bash
set -euo pipefail

# Paper Task 1 / CSD-T:
#   target -> skill card -> 10 verified candidates -> ridge teacher -> direct decode
# No Fit/Check split and no mock-exam gate: all accepted candidates specialize.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
MODEL_REVISION=${MODEL_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}
EVAL_DATA=${EVAL_DATA:-data/verl/amc23+aime24+aime25/val.parquet}
RUN_SEED=${RUN_SEED:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/clean_self_distill/task1_fast_teacher/seed_${RUN_SEED}}
PROPOSALS=${PROPOSALS:-${OUTPUT_ROOT}/eval_proposals.jsonl}
ADAPTER_DIR=${ADAPTER_DIR:-${OUTPUT_ROOT}/ridge_adapters}
NUM_CANDIDATES=${NUM_CANDIDATES:-10}
MIN_ACCEPTED_CANDIDATES=${MIN_ACCEPTED_CANDIDATES:-4}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-8192}
FORCE_PROPOSE=${FORCE_PROPOSE:-0}
FORCE_SPECIALIZE=${FORCE_SPECIALIZE:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-}
PRIVILEGED_CONTROL=${PRIVILEGED_CONTROL:-1}

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

if [[ ! -f "$ADAPTER_DIR/manifest.jsonl" || "$FORCE_SPECIALIZE" == "1" ]]; then
  "$PYTHON_BIN" scripts/clean_self_distill/02_specialize.py \
    --proposals "$PROPOSALS" \
    --output-dir "$ADAPTER_DIR" \
    --model "$MODEL_PATH" \
    --revision "$MODEL_REVISION" \
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
    --max-update-norm 2.0
fi

EVAL_ARGS=()
if [[ -n "$MAX_EVAL_SAMPLES" ]]; then
  EVAL_ARGS+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
fi
if [[ "$PRIVILEGED_CONTROL" == "1" ]]; then
  EVAL_ARGS+=(--privileged-control)
fi

"$PYTHON_BIN" scripts/clean_self_distill/03_train_eval.py \
  --mode task1 \
  --eval-data "$EVAL_DATA" \
  --proposals "$PROPOSALS" \
  --adapter-dir "$ADAPTER_DIR" \
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
  --eval-samples 1 \
  --eval-max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
  --eval-temperature 0 \
  --top-p 1.0 \
  --top-k 0 \
  --seed "$RUN_SEED" \
  "${EVAL_ARGS[@]}"

echo "Task 1 complete: $OUTPUT_ROOT/evaluation/summary.json"
