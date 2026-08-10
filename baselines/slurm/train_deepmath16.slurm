#!/usr/bin/env bash
#SBATCH --job-name=baseline-dm16
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_PREPARED_ROOT:?export CSD_PREPARED_ROOT}"
: "${BASELINE_RUN_ROOT:?export BASELINE_RUN_ROOT}"

# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"

export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_SCRATCH_ROOT/hf" HF_HUB_CACHE="$CSD_SCRATCH_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$BASELINE_RUN_ROOT/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

METHODS=(demopsd grpo)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="$BASELINE_RUN_ROOT/$METHOD/train"
mkdir -p "$TMPDIR" "$OUTPUT"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

RESUME_ARGS=()
if [[ -s "$OUTPUT/checkpoints/LATEST.json" ]]; then
  RESUME_ARGS=(--resume)
fi
"$CSD_TTT_PYTHON" baselines/train.py \
  --method "$METHOD" \
  --queries "$CSD_PREPARED_ROOT/distill_queries.jsonl" \
  --labels "$CSD_PREPARED_ROOT/distill_labels.sealed.jsonl" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --output-dir "$OUTPUT" \
  --episodes 16 \
  --group-size 8 \
  --max-sequence-tokens 10240 \
  --teacher-max-sequence-tokens 40960 \
  --max-rollout-tokens 10240 \
  --privileged-max-tokens 2048 \
  --learning-rate 1e-6 \
  --token-chunk-size 64 \
  --checkpoint-interval 1 \
  --attn-implementation sdpa \
  "${RESUME_ARGS[@]}"
