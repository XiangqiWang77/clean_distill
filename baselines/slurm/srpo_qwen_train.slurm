#!/usr/bin/env bash
#SBATCH --job-name=srpo-qwen
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=160G
#SBATCH --time=11:50:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%j.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%j.err

set -Eeuo pipefail
: "${CSD_SRPO_CONFIG:?export CSD_SRPO_CONFIG}"
: "${CSD_SRPO_MODEL_KEY:?export CSD_SRPO_MODEL_KEY}"
CSD_SRPO_EXPECTED_GPUS=${CSD_SRPO_EXPECTED_GPUS:-1}
# shellcheck disable=SC1090
source "$CSD_SRPO_CONFIG"

case "$CSD_SRPO_MODEL_KEY" in
  q17)
    MODEL_ID=$CSD_SRPO_Q17_MODEL_ID
    MODEL_LABEL=$CSD_SRPO_Q17_MODEL_LABEL
    MODEL=$CSD_SRPO_Q17_MODEL
    REVISION=$CSD_SRPO_Q17_REVISION
    GENERATION_BATCH_SIZE=4
    ;;
  q8)
    MODEL_ID=$CSD_SRPO_Q8_MODEL_ID
    MODEL_LABEL=$CSD_SRPO_Q8_MODEL_LABEL
    MODEL=$CSD_SRPO_Q8_MODEL
    REVISION=$CSD_SRPO_Q8_REVISION
    GENERATION_BATCH_SIZE=2
    ;;
  *) printf 'unsupported model key: %s\n' "$CSD_SRPO_MODEL_KEY" >&2; exit 2 ;;
esac

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_SRPO_CONDA_ENV"
cd "$CSD_SRPO_SNAPSHOT_ROOT"
export PYTHONPATH="$CSD_SRPO_SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_SRPO_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_SRPO_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/srpo_${CSD_SRPO_MODEL_KEY}_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_SRPO_RUN_ROOT/logs"

[[ "$CSD_SRPO_EXPECTED_GPUS" =~ ^[1-9][0-9]*$ ]]
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == "$CSD_SRPO_EXPECTED_GPUS" ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100
[[ -s "$MODEL/config.json" ]]
[[ "$(wc -l < "$CSD_SRPO_TRAIN_QUERIES")" == "$CSD_SRPO_EPISODES" ]]
[[ -s "$CSD_SRPO_TRAIN_LABELS" ]]

requeue_requested=0
child_pid=
handle_usr1() {
  requeue_requested=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
}
trap handle_usr1 USR1

TRAIN_DIR="$CSD_SRPO_RUN_ROOT/train/$CSD_SRPO_MODEL_KEY"
FINAL_CHECKPOINT="$TRAIN_DIR/checkpoints/episode_0064"
mkdir -p "$TRAIN_DIR"
if [[ ! -s "$FINAL_CHECKPOINT/checkpoint_manifest.json" ]]; then
  resume=()
  device_map=()
  [[ ! -s "$TRAIN_DIR/run_manifest.json" ]] || resume=(--resume)
  (( CSD_SRPO_EXPECTED_GPUS == 1 )) || device_map=(--device-map balanced)
  "$CSD_SRPO_PYTHON" baselines/train.py \
    --method srpo \
    --queries "$CSD_SRPO_TRAIN_QUERIES" \
    --labels "$CSD_SRPO_TRAIN_LABELS" \
    --model "$MODEL" \
    --model-id "$MODEL_ID" \
    --revision "$REVISION" \
    --output-dir "$TRAIN_DIR" \
    --episodes "$CSD_SRPO_EPISODES" \
    --group-size "$CSD_SRPO_GROUP_SIZE" \
    --generation-batch-size "$GENERATION_BATCH_SIZE" \
    --max-sequence-tokens "$CSD_SRPO_MAX_SEQUENCE_TOKENS" \
    --teacher-max-sequence-tokens "$CSD_SRPO_TEACHER_MAX_SEQUENCE_TOKENS" \
    --max-rollout-tokens "$CSD_SRPO_MAX_ROLLOUT_TOKENS" \
    --privileged-max-tokens "$CSD_SRPO_PRIVILEGED_MAX_TOKENS" \
    --learning-rate "$CSD_SRPO_LEARNING_RATE" \
    --weight-decay "$CSD_SRPO_WEIGHT_DECAY" \
    --lora-rank "$CSD_SRPO_LORA_RANK" \
    --lora-alpha "$CSD_SRPO_LORA_ALPHA" \
    --token-chunk-size "$CSD_SRPO_TOKEN_CHUNK_SIZE" \
    --checkpoint-interval "$CSD_SRPO_CHECKPOINT_INTERVAL" \
    --seed "$CSD_SRPO_TRAIN_SEED" \
    --train-temperature "$CSD_SRPO_TRAIN_TEMPERATURE" \
    --top-p "$CSD_SRPO_TRAIN_TOP_P" \
    --top-k "$CSD_SRPO_TRAIN_TOP_K" \
    --ema-rate "$CSD_SRPO_EMA_RATE" \
    --srpo-top-k "$CSD_SRPO_TOP_K" \
    --srpo-entropy-beta "$CSD_SRPO_ENTROPY_BETA" \
    --srpo-jsd-alpha "$CSD_SRPO_JSD_ALPHA" \
    --srpo-grpo-clip-epsilon-high "$CSD_SRPO_GRPO_CLIP_HIGH" \
    --srpo-warmup-steps "$CSD_SRPO_WARMUP_STEPS" \
    --srpo-rollout-importance-clip "$CSD_SRPO_ROLLOUT_IS_CLIP" \
    --grpo-clip-epsilon "$CSD_SRPO_GRPO_CLIP_LOW" \
    --grpo-kl-coefficient "$CSD_SRPO_GRPO_KL" \
    --attn-implementation sdpa \
    "${device_map[@]}" \
    "${resume[@]}" &
  child_pid=$!
  rc=0
  wait "$child_pid" || rc=$?
  child_pid=
  if (( requeue_requested )); then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  (( rc == 0 ))
fi
[[ -s "$FINAL_CHECKPOINT/checkpoint_manifest.json" ]]
printf 'complete\njob_id=%s\nmodel=%s\nepisodes=%s\ngpus=%s\n' \
  "$SLURM_JOB_ID" "$MODEL_LABEL" "$CSD_SRPO_EPISODES" "$CSD_SRPO_EXPECTED_GPUS" \
  > "$CSD_SRPO_RUN_ROOT/${CSD_SRPO_MODEL_KEY}.TRAIN_COMPLETE"
