#!/usr/bin/env bash
#SBATCH --job-name=proj64-train
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=11:50:00
#SBATCH --array=0-3%2
#SBATCH --requeue
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%A_%a-train.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%A_%a-train.err

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
export TMPDIR="$CSD_PT_RUN_ROOT/tmp_train_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

METHODS=(opsd opsd demopsd demopsd)
PROJECTIONS=(0 1 0 1)
VARIANTS=(opsd_original opsd_projected demopsd_original demopsd_projected)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
PROJECTED="${PROJECTIONS[$SLURM_ARRAY_TASK_ID]}"
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
PROJECTION_ARGS=()
[[ "$PROJECTED" == 0 ]] || PROJECTION_ARGS=(--trajectory-projection)
OUTPUT="$CSD_PT_RUN_ROOT/table14/$VARIANT/train"
mkdir -p "$TMPDIR" "$OUTPUT"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null
[[ "$(sha256sum "$CSD_PT_PREPARED_ROOT/distill_queries.jsonl" | awk '{print $1}')" == "$CSD_PT_DISTILL_QUERIES_SHA256" ]]
[[ "$(sha256sum "$CSD_PT_TRAIN_LABELS" | awk '{print $1}')" == "$CSD_PT_TRAIN_LABELS_SHA256" ]]

RESUME_ARGS=()
[[ ! -s "$OUTPUT/checkpoints/LATEST.json" ]] || RESUME_ARGS=(--resume)
"$CSD_PT_PYTHON" baselines/train.py \
  --method "$METHOD" \
  --queries "$CSD_PT_PREPARED_ROOT/distill_queries.jsonl" \
  --labels "$CSD_PT_TRAIN_LABELS" \
  --model "$CSD_PT_MODEL_LOCAL_DIR" \
  --model-id "$CSD_PT_MODEL_ID" \
  --revision "$CSD_PT_MODEL_REVISION" \
  --output-dir "$OUTPUT" \
  --episodes "$CSD_PT_EPISODES" \
  --group-size "$CSD_PT_GROUP_SIZE" \
  --generation-batch-size "$CSD_PT_GROUP_SIZE" \
  --max-sequence-tokens "$CSD_PT_MAX_SEQUENCE_TOKENS" \
  --teacher-max-sequence-tokens "$CSD_PT_TEACHER_MAX_SEQUENCE_TOKENS" \
  --max-rollout-tokens "$CSD_PT_MAX_ROLLOUT_TOKENS" \
  --privileged-max-tokens "$CSD_PT_PRIVILEGED_MAX_TOKENS" \
  --learning-rate "$CSD_PT_LEARNING_RATE" \
  --token-chunk-size "$CSD_PT_TOKEN_CHUNK_SIZE" \
  --projection-kl-budget "$CSD_PT_PROJECTION_KL_BUDGET" \
  --projection-binary-search-steps "$CSD_PT_PROJECTION_SEARCH_STEPS" \
  --seed "$CSD_PT_SEED" \
  --checkpoint-interval 1 \
  --attn-implementation sdpa \
  "${PROJECTION_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
