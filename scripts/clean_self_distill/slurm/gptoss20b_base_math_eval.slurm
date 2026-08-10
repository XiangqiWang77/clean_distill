#!/usr/bin/env bash
#SBATCH --job-name=go20-base
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=240G
#SBATCH --time=05:50:00
#SBATCH --requeue
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%j-base-math.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%j-base-math.err

set -Eeuo pipefail
: "${CSD_GO_CONFIG:?export CSD_GO_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_GO_CONFIG"
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
export TMPDIR="/tmp/go20_base_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_GO_RUN_ROOT/logs" "$CSD_GO_RUN_ROOT/eval/base"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 2 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100

PREDICTIONS="$CSD_GO_RUN_ROOT/eval/base/predictions.jsonl"
SCORED="$CSD_GO_RUN_ROOT/eval/base/scored.jsonl"
if [[ ! -s "$SCORED" ]] || [[ "$(wc -l < "$SCORED")" != 143 ]]; then
  "$CSD_GO_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$CSD_GO_PREPARED_16_ROOT/heldout_queries.jsonl" \
    --output "$PREDICTIONS" \
    --model "$CSD_GO_MODEL_LOCAL_DIR" \
    --model-id "$CSD_GO_MODEL_ID" \
    --revision "$CSD_GO_MODEL_REVISION" \
    --method base --checkpoint-episode 0 \
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
    --labels "$CSD_GO_PREPARED_16_ROOT/heldout_labels.sealed.jsonl" \
    --output "$SCORED" \
    --sample-count 1
fi
[[ "$(wc -l < "$SCORED")" == 143 ]]
"$CSD_GO_PYTHON" scripts/clean_self_distill/14_qwen17_fourway_report.py \
  --run-root "$CSD_GO_RUN_ROOT" \
  --model-id "$CSD_GO_MODEL_ID" \
  --model-label "$CSD_GO_MODEL_LABEL"
