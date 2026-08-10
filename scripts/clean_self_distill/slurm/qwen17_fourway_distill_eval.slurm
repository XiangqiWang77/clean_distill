#!/usr/bin/env bash
#SBATCH --job-name=q17-fourway
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=11:50:00
#SBATCH --array=0-3%4
#SBATCH --requeue
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-1.7b-fourway-20260808/logs/%A_%a.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-1.7b-fourway-20260808/logs/%A_%a.err

set -Eeuo pipefail
: "${CSD_Q17_CONFIG:?export CSD_Q17_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_Q17_CONFIG"

case "$SLURM_ARRAY_TASK_ID" in
  0) TAG=privileged_16; BRANCH=privileged; METHOD=privileged_sd; EPISODES=16 ;;
  1) TAG=trsd_16;       BRANCH=clean;      METHOD=trsd;         EPISODES=16 ;;
  2) TAG=privileged_64; BRANCH=privileged; METHOD=privileged_sd; EPISODES=64 ;;
  3) TAG=trsd_64;       BRANCH=clean;      METHOD=trsd;         EPISODES=64 ;;
  *) echo "Unsupported array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_Q17_CONDA_ENV"
cd "$CSD_Q17_CODE_ROOT"

export PYTHONPATH="$CSD_Q17_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_Q17_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_Q17_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/q17_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_Q17_RUN_ROOT/logs"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100

DISTILL_QUERIES="$CSD_Q17_PREPARED_ROOT/distill_queries.jsonl"
HELDOUT_QUERIES="$CSD_Q17_PREPARED_ROOT/heldout_queries.jsonl"
HELDOUT_LABELS="$CSD_Q17_PREPARED_ROOT/heldout_labels.sealed.jsonl"
[[ "$(sha256sum "$DISTILL_QUERIES" | awk '{print $1}')" == "$CSD_Q17_DISTILL_QUERIES_SHA256" ]]
[[ "$(sha256sum "$HELDOUT_QUERIES" | awk '{print $1}')" == "$CSD_Q17_HELDOUT_QUERIES_SHA256" ]]
[[ "$(sha256sum "$HELDOUT_LABELS" | awk '{print $1}')" == "$CSD_Q17_HELDOUT_LABELS_SHA256" ]]
[[ -s "$CSD_Q17_MODEL_LOCAL_DIR/model.safetensors.index.json" ]]

run_eval() {
  local eval_tag=$1
  local eval_method=$2
  local checkpoint_episode=$3
  local adapter=${4:-}
  local eval_dir="$CSD_Q17_RUN_ROOT/eval/$eval_tag"
  local predictions="$eval_dir/predictions.jsonl"
  local scored="$eval_dir/scored.jsonl"
  mkdir -p "$eval_dir"
  if [[ -s "$scored" ]] && [[ "$(wc -l < "$scored")" == 143 ]]; then
    echo "$eval_tag evaluation already complete"
    return
  fi
  local adapter_args=()
  [[ -z "$adapter" ]] || adapter_args=(--adapter "$adapter")
  "$CSD_Q17_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$HELDOUT_QUERIES" \
    --output "$predictions" \
    --model "$CSD_Q17_MODEL_LOCAL_DIR" \
    --model-id "$CSD_Q17_MODEL_ID" \
    --revision "$CSD_Q17_MODEL_REVISION" \
    --method "$eval_method" \
    --checkpoint-episode "$checkpoint_episode" \
    --engine vllm \
    --batch-size 64 \
    --gpu-memory-utilization 0.88 \
    --max-lora-rank 8 \
    --max-new-tokens "$CSD_Q17_EVAL_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$CSD_Q17_MAX_PROMPT_TOKENS" \
    --context-window "$CSD_Q17_CONTEXT_WINDOW_TOKENS" \
    --num-shards 1 \
    --shard-index 0 \
    --sample-count 1 \
    --seed "$CSD_Q17_EVAL_SEED" \
    --attn-implementation sdpa \
    "${adapter_args[@]}"
  [[ "$(wc -l < "$predictions")" == 143 ]]
  "$CSD_Q17_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    --predictions "$predictions" \
    --labels "$HELDOUT_LABELS" \
    --output "$scored" \
    --sample-count 1
  [[ "$(wc -l < "$scored")" == 143 ]]
}

echo "task=$SLURM_ARRAY_TASK_ID tag=$TAG branch=$BRANCH episodes=$EPISODES host=$(hostname)"

# Base shares task 0's H100, keeping the experiment at exactly four GPU jobs.
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  run_eval base base 0
  "$CSD_Q17_PYTHON" scripts/clean_self_distill/14_qwen17_fourway_report.py \
    --run-root "$CSD_Q17_RUN_ROOT"
fi

TRAIN_DIR="$CSD_Q17_RUN_ROOT/train/$TAG"
mkdir -p "$TRAIN_DIR"
if [[ "$EPISODES" == 16 ]]; then
  CHECKPOINTS=0,16
  ROLLING_INTERVAL=4
else
  CHECKPOINTS=0,16,32,48,64
  ROLLING_INTERVAL=8
fi
FINAL_CHECKPOINT="$TRAIN_DIR/checkpoints/episode_$(printf '%04d' "$EPISODES")"

if [[ ! -s "$FINAL_CHECKPOINT/checkpoint_manifest.json" ]]; then
  RESUME_ARGS=()
  [[ ! -s "$TRAIN_DIR/run_manifest.json" ]] || RESUME_ARGS=(--resume)
  "$CSD_Q17_PYTHON" scripts/clean_self_distill/04_persistent_train.py \
    --branch "$BRANCH" \
    --queries "$DISTILL_QUERIES" \
    --model "$CSD_Q17_MODEL_LOCAL_DIR" \
    --model-id "$CSD_Q17_MODEL_ID" \
    --revision "$CSD_Q17_MODEL_REVISION" \
    --output-dir "$TRAIN_DIR" \
    --episodes "$EPISODES" \
    --scientific-checkpoints "$CHECKPOINTS" \
    --rolling-checkpoint-interval "$ROLLING_INTERVAL" \
    --max-sequence-tokens "$CSD_Q17_TRAIN_MAX_SEQUENCE_TOKENS" \
    --max-rollout-tokens "$CSD_Q17_TRAIN_MAX_ROLLOUT_TOKENS" \
    --learning-rate "$CSD_Q17_LEARNING_RATE" \
    --lora-rank "$CSD_Q17_LORA_RANK" \
    --lora-alpha "$CSD_Q17_LORA_ALPHA" \
    --seed "$CSD_Q17_TRAIN_SEED" \
    --distill-token-chunk-size "$CSD_Q17_DISTILL_TOKEN_CHUNK_SIZE" \
    --trust-region-kl-budget "$CSD_Q17_TRUST_REGION_KL_BUDGET" \
    --trust-region-binary-search-steps "$CSD_Q17_TRUST_REGION_BINARY_SEARCH_STEPS" \
    --attn-implementation sdpa \
    "${RESUME_ARGS[@]}"
fi
[[ -s "$FINAL_CHECKPOINT/checkpoint_manifest.json" ]]

run_eval "$TAG" "$METHOD" "$EPISODES" "$FINAL_CHECKPOINT"

printf 'complete\njob_id=%s\narray_task_id=%s\nmethod=%s\nepisodes=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$METHOD" "$EPISODES" \
  > "$CSD_Q17_RUN_ROOT/$TAG.TASK_COMPLETE"

"$CSD_Q17_PYTHON" scripts/clean_self_distill/14_qwen17_fourway_report.py \
  --run-root "$CSD_Q17_RUN_ROOT"
