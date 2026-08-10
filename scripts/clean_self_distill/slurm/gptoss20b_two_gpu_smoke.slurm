#!/usr/bin/env bash
#SBATCH --job-name=go20-2g-smoke
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%j-2gpu-smoke.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%j-2gpu-smoke.err

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
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/go20_2gpu_smoke_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_GO_RUN_ROOT/logs"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 2 ]]
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$CSD_GO_PYTHON" scripts/clean_self_distill/21_gptoss_two_gpu_smoke.py \
  --model "$CSD_GO_MODEL_LOCAL_DIR" \
  --revision "$CSD_GO_MODEL_REVISION" \
  --sequence-tokens "$CSD_GO_TRAIN_MAX_SEQUENCE_TOKENS" \
  --output "$CSD_GO_RUN_ROOT/TWO_GPU_SMOKE.json"
[[ -s "$CSD_GO_RUN_ROOT/TWO_GPU_SMOKE.json" ]]
