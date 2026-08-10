#!/usr/bin/env bash
#SBATCH --job-name=trsd-style-array
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%3

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_SHORT_ROOT:?export CSD_SHORT_ROOT}"
: "${CSD_PREPARED_ROOT:?export CSD_PREPARED_ROOT}"

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
export TMPDIR="$CSD_SHORT_ROOT/tmp_mechanism_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$CSD_SHORT_ROOT/mechanism"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

CSD_QUERY_LABEL=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
CSD_OUTPUT="$CSD_SHORT_ROOT/mechanism/query_${CSD_QUERY_LABEL}.jsonl"
if [[ -s "$CSD_OUTPUT" ]]; then
  exit 0
fi

"$CSD_TTT_PYTHON" scripts/clean_self_distill/06_trust_region_mechanism.py \
  --queries "$CSD_PREPARED_ROOT/dev_queries.jsonl" \
  --adapter "$CSD_SHORT_ROOT/train/checkpoints/episode_0002" \
  --output "$CSD_OUTPUT" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --query-index "$SLURM_ARRAY_TASK_ID" \
  --max-new-tokens 10240 \
  --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS" \
  --context-window "$CSD_CONTEXT_WINDOW_TOKENS" \
  --alpha-grid 0,0.25,0.5,0.75,1 \
  --epsilon-grid 0.001,0.002,0.004,0.008,0.016 \
  --selected-epsilon 0.004 \
  --training-epsilon 0.004 \
  --binary-search-steps "$CSD_TRUST_REGION_BINARY_SEARCH_STEPS" \
  --full-vocab-chunk-size "$CSD_DISTILL_TOKEN_CHUNK_SIZE" \
  --attn-implementation sdpa \
  --seed "$CSD_RUN_SEED"

[[ -s "$CSD_OUTPUT" ]]
