#!/usr/bin/env bash
#SBATCH --job-name=q17b64
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=160G
#SBATCH --time=11:50:00
#SBATCH --array=0-1%2
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-1.7b-baselines64-20260809/logs/%A_%a.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-1.7b-baselines64-20260809/logs/%A_%a.err

set -Eeuo pipefail
: "${CSD_Q17B64_CONFIG:?export CSD_Q17B64_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_Q17B64_CONFIG"

case "$SLURM_ARRAY_TASK_ID" in
  0) METHOD=demopsd; DISPLAY=DemoPSD ;;
  1) METHOD=grpo;    DISPLAY=GRPO ;;
  *) exit 2 ;;
esac

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_Q17B64_CONDA_ENV"
cd "$CSD_Q17B64_CODE_ROOT"

export PYTHONPATH="$CSD_Q17B64_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_Q17B64_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_Q17B64_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/q17b64_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export VLLM_CACHE_ROOT="$TMPDIR/vllm" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_Q17B64_RUN_ROOT/logs"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100
[[ -s "$CSD_Q17B64_MODEL_LOCAL_DIR/model.safetensors.index.json" ]]
[[ "$(wc -l < "$CSD_Q17B64_PREPARED_ROOT/distill_queries.jsonl")" == "$CSD_Q17B64_EPISODES" ]]
[[ "$(sha256sum "$CSD_Q17B64_PREPARED_ROOT/distill_queries.jsonl" | awk '{print $1}')" == "$CSD_Q17B64_DISTILL_QUERIES_SHA256" ]]
[[ "$(sha256sum "$CSD_Q17B64_TRAIN_LABELS" | awk '{print $1}')" == "$CSD_Q17B64_TRAIN_LABELS_SHA256" ]]
[[ "$(sha256sum "$CSD_Q17B64_PREPARED_ROOT/heldout_queries.jsonl" | awk '{print $1}')" == "$CSD_Q17B64_HELDOUT_QUERIES_SHA256" ]]
[[ "$(sha256sum "$CSD_Q17B64_PREPARED_ROOT/heldout_labels.sealed.jsonl" | awk '{print $1}')" == "$CSD_Q17B64_HELDOUT_LABELS_SHA256" ]]
[[ "$(wc -l < "$CSD_Q17B64_BASE_SCORED")" == 143 ]]

requeue_requested=0
child_pid=
handle_usr1() {
  requeue_requested=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
}
trap handle_usr1 USR1

run_child() {
  requeue_requested=0
  "$@" &
  child_pid=$!
  local rc=0
  wait "$child_pid" || rc=$?
  child_pid=
  if (( requeue_requested )); then
    echo "checkpoint-safe requeue requested for job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  (( rc == 0 ))
}

TRAIN_DIR="$CSD_Q17B64_RUN_ROOT/train/$METHOD"
FINAL_CHECKPOINT="$TRAIN_DIR/checkpoints/episode_0064"
mkdir -p "$TRAIN_DIR"
if [[ ! -s "$FINAL_CHECKPOINT/checkpoint_manifest.json" ]]; then
  resume=()
  [[ ! -s "$TRAIN_DIR/run_manifest.json" ]] || resume=(--resume)
  run_child "$CSD_Q17B64_PYTHON" baselines/train.py \
    --method "$METHOD" \
    --queries "$CSD_Q17B64_PREPARED_ROOT/distill_queries.jsonl" \
    --labels "$CSD_Q17B64_TRAIN_LABELS" \
    --model "$CSD_Q17B64_MODEL_LOCAL_DIR" \
    --model-id "$CSD_Q17B64_MODEL_ID" \
    --revision "$CSD_Q17B64_MODEL_REVISION" \
    --output-dir "$TRAIN_DIR" \
    --episodes "$CSD_Q17B64_EPISODES" \
    --group-size "$CSD_Q17B64_GROUP_SIZE" \
    --max-sequence-tokens "$CSD_Q17B64_MAX_SEQUENCE_TOKENS" \
    --teacher-max-sequence-tokens "$CSD_Q17B64_TEACHER_MAX_SEQUENCE_TOKENS" \
    --max-rollout-tokens "$CSD_Q17B64_MAX_ROLLOUT_TOKENS" \
    --privileged-max-tokens "$CSD_Q17B64_PRIVILEGED_MAX_TOKENS" \
    --learning-rate "$CSD_Q17B64_LEARNING_RATE" \
    --lora-rank "$CSD_Q17B64_LORA_RANK" \
    --lora-alpha "$CSD_Q17B64_LORA_ALPHA" \
    --token-chunk-size "$CSD_Q17B64_TOKEN_CHUNK_SIZE" \
    --checkpoint-interval "$CSD_Q17B64_CHECKPOINT_INTERVAL" \
    --seed "$CSD_Q17B64_TRAIN_SEED" \
    --train-temperature "$CSD_Q17B64_TRAIN_TEMPERATURE" \
    --top-p "$CSD_Q17B64_TOP_P" \
    --top-k "$CSD_Q17B64_TOP_K" \
    --demopsd-beta "$CSD_Q17B64_DEMOPSD_BETA" \
    --demopsd-alpha-max "$CSD_Q17B64_DEMOPSD_ALPHA_MAX" \
    --ema-rate "$CSD_Q17B64_EMA_RATE" \
    --grpo-clip-epsilon "$CSD_Q17B64_GRPO_CLIP_EPSILON" \
    --grpo-kl-coefficient "$CSD_Q17B64_GRPO_KL_COEFFICIENT" \
    --attn-implementation sdpa \
    "${resume[@]}"
fi
[[ -s "$FINAL_CHECKPOINT/checkpoint_manifest.json" ]]

EVAL_DIR="$CSD_Q17B64_RUN_ROOT/eval/$METHOD"
PREDICTIONS="$EVAL_DIR/predictions.jsonl"
SCORED="$EVAL_DIR/scored.jsonl"
mkdir -p "$EVAL_DIR"
if [[ ! -s "$SCORED" ]] || [[ "$(wc -l < "$SCORED")" != 143 ]]; then
  run_child "$CSD_Q17B64_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$CSD_Q17B64_PREPARED_ROOT/heldout_queries.jsonl" \
    --output "$PREDICTIONS" \
    --model "$CSD_Q17B64_MODEL_LOCAL_DIR" \
    --model-id "$CSD_Q17B64_MODEL_ID" \
    --revision "$CSD_Q17B64_MODEL_REVISION" \
    --method "$METHOD" \
    --checkpoint-episode "$CSD_Q17B64_EPISODES" \
    --adapter "$FINAL_CHECKPOINT" \
    --engine vllm \
    --batch-size "$CSD_Q17B64_EVAL_BATCH_SIZE" \
    --gpu-memory-utilization 0.88 \
    --max-lora-rank "$CSD_Q17B64_LORA_RANK" \
    --max-new-tokens "$CSD_Q17B64_EVAL_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$CSD_Q17B64_MAX_PROMPT_TOKENS" \
    --context-window "$CSD_Q17B64_CONTEXT_WINDOW_TOKENS" \
    --num-shards 1 --shard-index 0 --sample-count 1 \
    --seed "$CSD_Q17B64_EVAL_SEED" \
    --attn-implementation sdpa
  [[ "$(wc -l < "$PREDICTIONS")" == 143 ]]
  "$CSD_Q17B64_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    --predictions "$PREDICTIONS" \
    --labels "$CSD_Q17B64_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
    --output "$SCORED" \
    --sample-count 1
fi
[[ "$(wc -l < "$SCORED")" == 143 ]]

printf 'complete\njob_id=%s\narray_task_id=%s\nmethod=%s\nepisodes=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$METHOD" "$CSD_Q17B64_EPISODES" \
  > "$CSD_Q17B64_RUN_ROOT/$METHOD.TASK_COMPLETE"

(
  flock -x 9
  if [[ -s "$CSD_Q17B64_RUN_ROOT/eval/demopsd/scored.jsonl" ]] \
    && [[ -s "$CSD_Q17B64_RUN_ROOT/eval/grpo/scored.jsonl" ]]; then
    "$CSD_Q17B64_PYTHON" baselines/build_comparison_table.py \
      --entry "Base|0|$CSD_Q17B64_BASE_SCORED" \
      --entry "DemoPSD 64|64|$CSD_Q17B64_RUN_ROOT/eval/demopsd/scored.jsonl" \
      --entry "GRPO 64|64|$CSD_Q17B64_RUN_ROOT/eval/grpo/scored.jsonl" \
      --model-label "$CSD_Q17B64_MODEL_LABEL" \
      --output-dir "$CSD_Q17B64_RUN_ROOT/results"
  fi
) 9> "$CSD_Q17B64_RUN_ROOT/.report.lock"

echo "$DISPLAY 64 train+eval complete"
