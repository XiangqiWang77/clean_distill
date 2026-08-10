#!/usr/bin/env bash
#SBATCH --job-name=srpo-goeval
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=240G
#SBATCH --time=11:50:00
#SBATCH --array=0-7%2
#SBATCH --requeue
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%A_%a.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%A_%a.err

set -Eeuo pipefail
: "${CSD_SRPO_CONFIG:?export CSD_SRPO_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_SRPO_CONFIG"

if (( SLURM_ARRAY_TASK_ID < 4 )); then EPISODE=16; else EPISODE=64; fi
SHARD=$((SLURM_ARRAY_TASK_ID % 4))
CHECKPOINT="$CSD_SRPO_RUN_ROOT/train/go20/checkpoints/episode_$(printf '%04d' "$EPISODE")"
EVAL_ROOT="$CSD_SRPO_RUN_ROOT/eval/go20/episode_$(printf '%04d' "$EPISODE")"
OUTPUT="$EVAL_ROOT/predictions/shard_$(printf '%02d' "$SHARD").jsonl"
DONE="$EVAL_ROOT/status/shard_$(printf '%02d' "$SHARD").done"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_SRPO_CONDA_ENV"
cd "$CSD_SRPO_SNAPSHOT_ROOT"
export PYTHONPATH="$CSD_SRPO_SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_SRPO_GO_HF_HOME" HF_HUB_CACHE="$CSD_SRPO_GO_HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/srpo_goeval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$(dirname "$OUTPUT")" "$(dirname "$DONE")"

[[ -s "$CHECKPOINT/checkpoint_manifest.json" ]]
[[ "$(wc -l < "$CSD_SRPO_HELDOUT_QUERIES")" == 143 ]]
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 2 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100

if [[ ! -s "$DONE" ]]; then
  "$CSD_SRPO_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$CSD_SRPO_HELDOUT_QUERIES" \
    --output "$OUTPUT" \
    --model "$CSD_SRPO_GO_MODEL" \
    --model-id "$CSD_SRPO_GO_MODEL_ID" \
    --revision "$CSD_SRPO_GO_REVISION" \
    --method srpo \
    --checkpoint-episode "$EPISODE" \
    --adapter "$CHECKPOINT" \
    --engine vllm \
    --batch-size 64 \
    --gpu-memory-utilization 0.88 \
    --tensor-parallel-size 2 \
    --max-lora-rank "$CSD_SRPO_LORA_RANK" \
    --max-new-tokens "$CSD_SRPO_EVAL_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$CSD_SRPO_EVAL_MAX_PROMPT_TOKENS" \
    --context-window "$CSD_SRPO_GO_CONTEXT_WINDOW" \
    --num-shards "$CSD_SRPO_EVAL_SHARDS" \
    --shard-index "$SHARD" \
    --sample-count 1 \
    --seed "$CSD_SRPO_EVAL_SEED"
  [[ -s "$OUTPUT" ]]
  printf 'complete\njob_id=%s\nmodel=go20\nepisode=%s\nshard=%s\n' \
    "$SLURM_JOB_ID" "$EPISODE" "$SHARD" > "$DONE"
fi
