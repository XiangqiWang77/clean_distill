#!/usr/bin/env bash
#SBATCH --job-name=q8-p-replay
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00

set -Eeuo pipefail
: "${CSD_I8_CONFIG:?export CSD_I8_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_I8_CONFIG"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_I8_CONDA_ENV"
cd "$CSD_I8_ARCHIVED_CODE_ROOT"

export PYTHONPATH="$CSD_I8_TORCH_OVERLAY:$CSD_I8_ARCHIVED_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_I8_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_I8_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/q8_priv_replay_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_I8_PRIVILEGED_REPLAY_ROOT"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100
"$CSD_I8_PYTHON" - <<'PY'
import torch
assert str(torch.__version__).endswith("+cu128"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
print(f"replay_torch={torch.__version__} cuda={torch.version.cuda}")
PY

resume=()
[[ ! -s "$CSD_I8_PRIVILEGED_REPLAY_ROOT/run_manifest.json" ]] || resume=(--resume)
"$CSD_I8_PYTHON" \
  "$CSD_I8_SNAPSHOT_ROOT/scripts/clean_self_distill/25_replay_privileged_intermediate.py" \
  --queries "$CSD_I8_PREPARED_ROOT/distill_queries.jsonl" \
  --original-journal "$CSD_I8_PRIVILEGED_JOURNAL" \
  --original-checkpoint "$CSD_I8_PRIVILEGED64_CHECKPOINT" \
  --expected-adapter-sha256 "$CSD_I8_PRIVILEGED64_ADAPTER_SHA256" \
  --model "$CSD_I8_MODEL_LOCAL_DIR" \
  --model-id "$CSD_I8_MODEL_ID" \
  --revision "$CSD_I8_MODEL_REVISION" \
  --output-dir "$CSD_I8_PRIVILEGED_REPLAY_ROOT" \
  "${resume[@]}"

[[ -s "$CSD_I8_PRIVILEGED_REPLAY_ROOT/REPLAY_VERIFIED.json" ]]
