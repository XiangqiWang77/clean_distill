#!/usr/bin/env bash
#SBATCH --job-name=cost64-train
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=11:50:00
#SBATCH --array=0-2%2
#SBATCH --requeue
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%A_%a-cost.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%A_%a-cost.err

set -Eeuo pipefail
: "${CSD_PT_CONFIG:?export CSD_PT_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_PT_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_PT_CONDA_ENV"
cd "$CSD_PT_CODE_ROOT"

export PYTHONPATH="$CSD_PT_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_PT_SCRATCH_ROOT/hf" HF_HUB_CACHE="$CSD_PT_SCRATCH_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$CSD_PT_RUN_ROOT/tmp_cost_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  OUTPUT="$CSD_PT_RUN_ROOT/table16/grpo/train"
  mkdir -p "$OUTPUT"
  RESUME_ARGS=()
  [[ ! -s "$OUTPUT/checkpoints/LATEST.json" ]] || RESUME_ARGS=(--resume)
  "$CSD_PT_PYTHON" baselines/train.py \
    --method grpo \
    --queries "$CSD_PT_PREPARED_ROOT/distill_queries.jsonl" \
    --labels "$CSD_PT_TRAIN_LABELS" \
    --model "$CSD_PT_MODEL_LOCAL_DIR" \
    --model-id "$CSD_PT_MODEL_ID" \
    --revision "$CSD_PT_MODEL_REVISION" \
    --output-dir "$OUTPUT" \
    --episodes 64 \
    --group-size 8 \
    --generation-batch-size 8 \
    --max-sequence-tokens 10240 \
    --teacher-max-sequence-tokens 40960 \
    --max-rollout-tokens 10240 \
    --learning-rate 1e-6 \
    --token-chunk-size 64 \
    --seed 0 \
    --checkpoint-interval 1 \
    --attn-implementation sdpa \
    "${RESUME_ARGS[@]}"
else
  GUARD_ARGS=()
  VARIANT=trsd
  if [[ "$SLURM_ARRAY_TASK_ID" == 2 ]]; then
    VARIANT=trsd_update_guard
    GUARD_ARGS=(--update-guard)
  fi
  OUTPUT="$CSD_PT_RUN_ROOT/table16/$VARIANT/train"
  mkdir -p "$OUTPUT"
  RESUME_ARGS=()
  [[ ! -s "$OUTPUT/checkpoints/LATEST.json" ]] || RESUME_ARGS=(--resume)
  "$CSD_PT_PYTHON" scripts/clean_self_distill/04_persistent_train.py \
    --branch clean \
    --queries "$CSD_PT_PREPARED_ROOT/distill_queries.jsonl" \
    --model "$CSD_PT_MODEL_LOCAL_DIR" \
    --model-id "$CSD_PT_MODEL_ID" \
    --revision "$CSD_PT_MODEL_REVISION" \
    --output-dir "$OUTPUT" \
    --episodes 64 \
    --scientific-checkpoints 0,16,32,48,64 \
    --rolling-checkpoint-interval 1 \
    --max-sequence-tokens 10240 \
    --max-rollout-tokens 10240 \
    --learning-rate 2e-5 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --seed 0 \
    --train-temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --distill-token-chunk-size 128 \
    --trust-region-kl-budget 0.004 \
    --trust-region-binary-search-steps 6 \
    --attn-implementation sdpa \
    "${GUARD_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
fi
