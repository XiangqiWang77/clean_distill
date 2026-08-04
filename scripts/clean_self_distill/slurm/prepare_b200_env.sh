#!/usr/bin/env bash
# Build the small Python dependency layer used with YCRC's shared sm_100 torch.
set -Eeuo pipefail
umask 027

CSD_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
# shellcheck disable=SC1091
source "$CSD_REPO_ROOT/configs/clean_self_distill/b200_poc.env"

[[ "$CSD_B200_ENV_ROOT" == "$CSD_SCRATCH_ROOT"/* ]]
[[ "$CSD_B200_PYTHON" == "$CSD_B200_ENV_ROOT/bin/python" ]]
mkdir -p "$CSD_SCRATCH_ROOT/envs" "$CSD_SCRATCH_ROOT/tmp"

# Sourcing Lmod can return a non-zero status even though it defines `module`.
# Keep that site quirk outside the script's error trap.
source /etc/profile.d/modules.sh 2>/dev/null || true
module --force purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
[[ "${CONDA_DEFAULT_ENV:-}" == "$CSD_CONDA_ENV" ]]
module load "$CSD_PYTORCH_MODULE"

python -m venv --system-site-packages "$CSD_B200_ENV_ROOT"
TMPDIR="$CSD_SCRATCH_ROOT/tmp" \
PIP_CACHE_DIR="$CSD_SCRATCH_ROOT/pip-cache" \
PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$CSD_B200_PYTHON" -m pip install --no-cache-dir --only-binary=:all: \
    'transformers==4.57.6' \
    'peft==0.19.1' \
    'accelerate==1.14.0' \
    'huggingface-hub==0.36.2' \
    'safetensors==0.8.0' \
    'tokenizers==0.22.2' \
    'pyarrow==23.0.1' \
    'tqdm==4.67.3'

"$CSD_B200_PYTHON" - <<'PY'
import os
import torch
import transformers
import peft
import pyarrow

assert os.environ.get("CONDA_DEFAULT_ENV") == "TTT"
assert torch.__version__ == "2.9.1"
assert torch.version.cuda == "12.8"
assert "sm_100" in torch._C._cuda_getArchFlags().split()
print(
    "b200_python_ready=1",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"transformers={transformers.__version__}",
    f"peft={peft.__version__}",
    f"pyarrow={pyarrow.__version__}",
)
PY

"$CSD_B200_PYTHON" "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/launcher_support.py" \
  check-budget \
  --path "$CSD_SCRATCH_ROOT/hf" \
  --path "$CSD_SCRATCH_ROOT/data/verl" \
  --path "$CSD_B200_ENV_ROOT" \
  --max-bytes "$CSD_MAX_DOWNLOAD_BYTES"
