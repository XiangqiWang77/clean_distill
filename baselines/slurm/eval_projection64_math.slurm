#!/usr/bin/env bash
#SBATCH --job-name=proj64-math
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --time=11:50:00
#SBATCH --array=0-15%2
#SBATCH --requeue
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%A_%a-math.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%A_%a-math.err

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
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="$CSD_PT_RUN_ROOT/tmp_math_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

VARIANTS=(opsd_original opsd_projected demopsd_original demopsd_projected)
BRANCHES=(opsd opsd demopsd demopsd)
METHOD_IDS=(
  baseline:opsd:official_generalized_jsd_fixed_teacher_v1
  baseline:opsd:official_generalized_jsd_fixed_teacher_v1:trajectory_projection_v1
  baseline:demopsd:exact_full_vocab_v1
  baseline:demopsd:exact_full_vocab_v1:trajectory_projection_v1
)
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
VARIANT="${VARIANTS[$METHOD_INDEX]}"
BRANCH="${BRANCHES[$METHOD_INDEX]}"
METHOD_ID="${METHOD_IDS[$METHOD_INDEX]}"
CHECKPOINT="$CSD_PT_RUN_ROOT/table14/$VARIANT/train/checkpoints/episode_0064"
OUTPUT_ROOT="$CSD_PT_RUN_ROOT/table14/$VARIANT/math"
OUTPUT="$OUTPUT_ROOT/predictions/shard_$(printf '%02d' "$SHARD_INDEX").jsonl"
mkdir -p "$TMPDIR" "$OUTPUT_ROOT/predictions" "$OUTPUT_ROOT/status"

"$CSD_PT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
  --queries "$CSD_PT_PREPARED_ROOT/heldout_queries.jsonl" \
  --output "$OUTPUT" \
  --model "$CSD_PT_MODEL_LOCAL_DIR" \
  --model-id "$CSD_PT_MODEL_ID" \
  --revision "$CSD_PT_MODEL_REVISION" \
  --adapter "$CHECKPOINT" \
  --method "$VARIANT" \
  --expected-branch "$BRANCH" \
  --expected-method-id "$METHOD_ID" \
  --checkpoint-episode 64 \
  --engine vllm \
  --batch-size 64 \
  --gpu-memory-utilization 0.88 \
  --max-lora-rank 8 \
  --sample-count 1 \
  --max-new-tokens 10240 \
  --max-prompt-tokens 8192 \
  --context-window 40960 \
  --attn-implementation sdpa \
  --num-shards 4 \
  --shard-index "$SHARD_INDEX" \
  --seed 20260808

printf 'completed_at=%s\njob_id=%s\nvariant=%s\nshard=%s\n' \
  "$(date -Is)" "$SLURM_JOB_ID" "$VARIANT" "$SHARD_INDEX" \
  > "$OUTPUT_ROOT/status/shard_$(printf '%02d' "$SHARD_INDEX").done"

