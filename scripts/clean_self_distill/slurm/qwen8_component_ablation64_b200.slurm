#!/usr/bin/env bash
#SBATCH --job-name=q8-a2
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=160G
#SBATCH --time=11:50:00
#SBATCH --array=0-5%4
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-component-ablations64-20260809/logs/%A_%a.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-component-ablations64-20260809/logs/%A_%a.err

set -Eeuo pipefail
: "${CSD_A2_CONFIG:?export CSD_A2_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_A2_CONFIG"
CSD_A2_CODE_ROOT=${CSD_A2_CODE_ROOT_OVERRIDE:-$CSD_A2_CODE_ROOT}

case "$SLURM_ARRAY_TASK_ID" in
  0) TAG=independent_token_budgets;     EXTRA=(--projection-scope token) ;;
  1) TAG=fixed_global_alpha;            EXTRA=(--projection-scope fixed --fixed-projection-alpha "$CSD_A2_FIXED_ALPHA") ;;
  2) TAG=arithmetic_probability_path;   EXTRA=(--projection-path arithmetic) ;;
  3) TAG=forward_kl_student_loss;       EXTRA=(--student-kl-direction forward) ;;
  4) TAG=without_same_prefix_scoring;   EXTRA=(--disable-same-prefix-scoring) ;;
  5) TAG=realized_update_guard;         EXTRA=(--update-guard) ;;
  *) exit 2 ;;
esac

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_A2_CONDA_ENV"
cd "$CSD_A2_CODE_ROOT"
export PYTHONPATH="$CSD_A2_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_A2_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_A2_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/q8_a2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_A2_RUN_ROOT/logs"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100

TRAIN_DIR="$CSD_A2_RUN_ROOT/train/$TAG"
FINAL_CHECKPOINT="$TRAIN_DIR/checkpoints/episode_0064"
mkdir -p "$TRAIN_DIR"
resume=()
[[ ! -s "$TRAIN_DIR/run_manifest.json" ]] || resume=(--resume)
if [[ ! -s "$FINAL_CHECKPOINT/checkpoint_manifest.json" ]]; then
  set +e
  "$CSD_A2_PYTHON" scripts/clean_self_distill/04_persistent_train.py \
    --branch clean \
    --queries "$CSD_A2_TRAIN_QUERIES" \
    --model "$CSD_A2_MODEL_LOCAL_DIR" \
    --model-id "$CSD_A2_MODEL_ID" \
    --revision "$CSD_A2_MODEL_REVISION" \
    --output-dir "$TRAIN_DIR" \
    --episodes "$CSD_A2_EPISODES" \
    --scientific-checkpoints 0,16,32,48,64 \
    --rolling-checkpoint-interval 8 \
    --max-sequence-tokens "$CSD_A2_MAX_SEQUENCE_TOKENS" \
    --max-rollout-tokens "$CSD_A2_MAX_ROLLOUT_TOKENS" \
    --learning-rate "$CSD_A2_LEARNING_RATE" \
    --lora-rank "$CSD_A2_LORA_RANK" \
    --lora-alpha "$CSD_A2_LORA_ALPHA" \
    --seed "$CSD_A2_SEED" \
    --distill-token-chunk-size "$CSD_A2_TOKEN_CHUNK_SIZE" \
    --trust-region-kl-budget "$CSD_A2_KL_BUDGET" \
    --trust-region-binary-search-steps "$CSD_A2_BINARY_SEARCH_STEPS" \
    --attn-implementation sdpa \
    "${EXTRA[@]}" \
    "${resume[@]}"
  rc=$?
  set -e
  if (( rc == 75 )); then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  (( rc == 0 ))
fi
[[ -s "$FINAL_CHECKPOINT/checkpoint_manifest.json" ]]

EVAL_DIR="$CSD_A2_RUN_ROOT/eval/$TAG"
PREDICTIONS="$EVAL_DIR/predictions.jsonl"
SCORED="$EVAL_DIR/scored.jsonl"
mkdir -p "$EVAL_DIR"
if [[ ! -s "$SCORED" ]] || [[ "$(wc -l < "$SCORED")" != 143 ]]; then
  "$CSD_A2_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$CSD_A2_HELDOUT_QUERIES" \
    --output "$PREDICTIONS" \
    --model "$CSD_A2_MODEL_LOCAL_DIR" \
    --model-id "$CSD_A2_MODEL_ID" \
    --revision "$CSD_A2_MODEL_REVISION" \
    --method trsd \
    --checkpoint-episode 64 \
    --adapter "$FINAL_CHECKPOINT" \
    --engine vllm \
    --batch-size "$CSD_A2_EVAL_BATCH_SIZE" \
    --gpu-memory-utilization 0.88 \
    --max-lora-rank "$CSD_A2_LORA_RANK" \
    --max-new-tokens "$CSD_A2_EVAL_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$CSD_A2_MAX_PROMPT_TOKENS" \
    --context-window "$CSD_A2_CONTEXT_WINDOW_TOKENS" \
    --num-shards 1 --shard-index 0 --sample-count 1 \
    --seed "$CSD_A2_EVAL_SEED" \
    --attn-implementation sdpa
  "$CSD_A2_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    --predictions "$PREDICTIONS" \
    --labels "$CSD_A2_HELDOUT_LABELS" \
    --output "$SCORED" \
    --sample-count 1
fi
[[ "$(wc -l < "$SCORED")" == 143 ]]
printf 'complete\nmethod=%s\nepisodes=64\n' "$TAG" > "$CSD_A2_RUN_ROOT/$TAG.TASK_COMPLETE"
