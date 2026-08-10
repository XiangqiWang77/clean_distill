#!/usr/bin/env bash
#SBATCH --job-name=srpo-qeval
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --time=11:50:00
#SBATCH --array=0-7%4
#SBATCH --requeue
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%A_%a.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%A_%a.err

set -Eeuo pipefail
: "${CSD_SRPO_CONFIG:?export CSD_SRPO_CONFIG}"
: "${CSD_SRPO_MODEL_KEY:?export CSD_SRPO_MODEL_KEY}"
# shellcheck disable=SC1090
source "$CSD_SRPO_CONFIG"

case "$CSD_SRPO_MODEL_KEY" in
  q17)
    MODEL_ID=$CSD_SRPO_Q17_MODEL_ID
    MODEL=$CSD_SRPO_Q17_MODEL
    REVISION=$CSD_SRPO_Q17_REVISION
    ;;
  q8)
    MODEL_ID=$CSD_SRPO_Q8_MODEL_ID
    MODEL=$CSD_SRPO_Q8_MODEL
    REVISION=$CSD_SRPO_Q8_REVISION
    ;;
  *) printf 'unsupported model key: %s\n' "$CSD_SRPO_MODEL_KEY" >&2; exit 2 ;;
esac

if (( SLURM_ARRAY_TASK_ID < 4 )); then EPISODE=16; else EPISODE=64; fi
SHARD=$((SLURM_ARRAY_TASK_ID % 4))
CHECKPOINT="$CSD_SRPO_RUN_ROOT/train/$CSD_SRPO_MODEL_KEY/checkpoints/episode_$(printf '%04d' "$EPISODE")"
EVAL_ROOT="$CSD_SRPO_RUN_ROOT/eval/$CSD_SRPO_MODEL_KEY/episode_$(printf '%04d' "$EPISODE")"
OUTPUT="$EVAL_ROOT/predictions/shard_$(printf '%02d' "$SHARD").jsonl"
DONE="$EVAL_ROOT/status/shard_$(printf '%02d' "$SHARD").done"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_SRPO_CONDA_ENV"
cd "$CSD_SRPO_SNAPSHOT_ROOT"
export PYTHONPATH="$CSD_SRPO_SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_SRPO_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_SRPO_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/srpo_qeval_${CSD_SRPO_MODEL_KEY}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$(dirname "$OUTPUT")" "$(dirname "$DONE")"

[[ -s "$CHECKPOINT/checkpoint_manifest.json" ]]
[[ "$(wc -l < "$CSD_SRPO_HELDOUT_QUERIES")" == 143 ]]
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100

if [[ ! -s "$DONE" ]]; then
  "$CSD_SRPO_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$CSD_SRPO_HELDOUT_QUERIES" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --model-id "$MODEL_ID" \
    --revision "$REVISION" \
    --method srpo \
    --checkpoint-episode "$EPISODE" \
    --adapter "$CHECKPOINT" \
    --engine vllm \
    --batch-size 64 \
    --gpu-memory-utilization 0.88 \
    --max-lora-rank "$CSD_SRPO_LORA_RANK" \
    --max-new-tokens "$CSD_SRPO_EVAL_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$CSD_SRPO_EVAL_MAX_PROMPT_TOKENS" \
    --context-window "$CSD_SRPO_QWEN_CONTEXT_WINDOW" \
    --num-shards "$CSD_SRPO_EVAL_SHARDS" \
    --shard-index "$SHARD" \
    --sample-count 1 \
    --seed "$CSD_SRPO_EVAL_SEED" \
    --attn-implementation sdpa
  [[ -s "$OUTPUT" ]]
  printf 'complete\njob_id=%s\nmodel=%s\nepisode=%s\nshard=%s\n' \
    "$SLURM_JOB_ID" "$CSD_SRPO_MODEL_KEY" "$EPISODE" "$SHARD" > "$DONE"
fi
