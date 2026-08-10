#!/usr/bin/env bash
#SBATCH --job-name=go20-math
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=240G
#SBATCH --time=05:50:00
#SBATCH --array=0-3%3
#SBATCH --requeue
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%A_%a-math.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%A_%a-math.err

set -Eeuo pipefail
: "${CSD_GO_CONFIG:?export CSD_GO_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_GO_CONFIG"

case "$SLURM_ARRAY_TASK_ID" in
  0) TAG=privileged_16; METHOD=privileged_sd; EPISODES=16 ;;
  1) TAG=trsd_16;       METHOD=trsd;         EPISODES=16 ;;
  2) TAG=privileged_64; METHOD=privileged_sd; EPISODES=64 ;;
  3) TAG=trsd_64;       METHOD=trsd;         EPISODES=64 ;;
  *) exit 2 ;;
esac

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_GO_CONDA_ENV"
cd "$CSD_GO_CODE_ROOT"
export PYTHONPATH="$CSD_GO_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_GO_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_GO_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/go20_math_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_GO_RUN_ROOT/logs"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 2 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100

PREPARED="$CSD_GO_PREPARED_16_ROOT"
CHECKPOINT="$CSD_GO_RUN_ROOT/train/$TAG/checkpoints/episode_$(printf '%04d' "$EPISODES")"
[[ -s "$CHECKPOINT/checkpoint_manifest.json" ]]
EVAL_DIR="$CSD_GO_RUN_ROOT/eval/$TAG"
PREDICTIONS="$EVAL_DIR/predictions.jsonl"
SCORED="$EVAL_DIR/scored.jsonl"
mkdir -p "$EVAL_DIR"
if [[ ! -s "$SCORED" ]] || [[ "$(wc -l < "$SCORED")" != 143 ]]; then
  "$CSD_GO_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$PREPARED/heldout_queries.jsonl" \
    --output "$PREDICTIONS" \
    --model "$CSD_GO_MODEL_LOCAL_DIR" \
    --model-id "$CSD_GO_MODEL_ID" \
    --revision "$CSD_GO_MODEL_REVISION" \
    --method "$METHOD" \
    --checkpoint-episode "$EPISODES" \
    --adapter "$CHECKPOINT" \
    --engine vllm \
    --batch-size "$CSD_GO_MATH_BATCH_SIZE" \
    --gpu-memory-utilization 0.88 \
    --tensor-parallel-size 2 \
    --max-lora-rank "$CSD_GO_LORA_RANK" \
    --max-new-tokens "$CSD_GO_EVAL_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$CSD_GO_MAX_PROMPT_TOKENS" \
    --context-window "$CSD_GO_CONTEXT_WINDOW_TOKENS" \
    --num-shards 1 --shard-index 0 --sample-count 1 \
    --seed "$CSD_GO_EVAL_SEED"
  [[ "$(wc -l < "$PREDICTIONS")" == 143 ]]
  "$CSD_GO_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    --predictions "$PREDICTIONS" \
    --labels "$PREPARED/heldout_labels.sealed.jsonl" \
    --output "$SCORED" \
    --sample-count 1
fi
[[ "$(wc -l < "$SCORED")" == 143 ]]
"$CSD_GO_PYTHON" scripts/clean_self_distill/14_qwen17_fourway_report.py \
  --run-root "$CSD_GO_RUN_ROOT" \
  --model-id "$CSD_GO_MODEL_ID" \
  --model-label "$CSD_GO_MODEL_LABEL"
