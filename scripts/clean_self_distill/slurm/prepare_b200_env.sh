#!/usr/bin/env bash
# Validate the existing CUDA 12.8 overlay with the real TTT interpreter.
set -Eeuo pipefail
umask 027

CSD_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
# shellcheck disable=SC1091
source "$CSD_REPO_ROOT/configs/clean_self_distill/b200_poc.env"

[[ "$CSD_TTT_PYTHON" == /home/da839/.conda/envs/TTT/bin/python ]]
[[ -x "$CSD_TTT_PYTHON" ]]
[[ "$CSD_TORCH_OVERLAY" == /home/da839/scratch_pi_mg269/da839/mfspd/pydeps-cu128 ]]
[[ -d "$CSD_TORCH_OVERLAY/torch" ]]

# Sourcing Lmod can return a non-zero status even though it defines `module`.
# Keep that site quirk outside the script's error trap.
source /etc/profile.d/modules.sh 2>/dev/null || true
module --force purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
[[ "${CONDA_DEFAULT_ENV:-}" == "$CSD_CONDA_ENV" ]]
[[ "$CONDA_PREFIX" == "${CSD_TTT_PYTHON%/bin/python}" ]]
export CSD_TTT_PYTHON CSD_TORCH_OVERLAY
export PYTHONPATH="$CSD_TORCH_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$CSD_TTT_PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

import math_verify
import peft
import pyarrow
import torch
import transformers

assert os.environ.get("CONDA_DEFAULT_ENV") == "TTT"
expected_python = Path(os.environ["CSD_TTT_PYTHON"]).resolve()
overlay = Path(os.environ["CSD_TORCH_OVERLAY"]).resolve()
torch_path = Path(torch.__file__).resolve()
arch_flags = torch._C._cuda_getArchFlags().split()
assert Path(sys.executable).resolve() == expected_python, sys.executable
assert Path(os.environ["CONDA_PREFIX"]).resolve() == expected_python.parent.parent
assert torch_path.is_relative_to(overlay), (torch_path, overlay)
assert str(torch.__version__).endswith("+cu128"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert "sm_100" in arch_flags, arch_flags
print(
    "b200_overlay_validated=1",
    f"python={sys.executable}",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"torch_module={torch_path}",
    f"overlay={overlay}",
    f"arch_flags={' '.join(arch_flags)}",
    f"transformers={transformers.__version__}",
    f"peft={peft.__version__}",
    f"pyarrow={pyarrow.__version__}",
)
PY

# Validate only the model/data roots newly downloaded for this task. The
# pre-existing overlay is reused read-only and intentionally excluded.
"$CSD_TTT_PYTHON" "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/launcher_support.py" \
  check-budget \
  --path "$CSD_SCRATCH_ROOT/hf" \
  --path "$CSD_SCRATCH_ROOT/data/verl" \
  --max-bytes "$CSD_MAX_DOWNLOAD_BYTES"
