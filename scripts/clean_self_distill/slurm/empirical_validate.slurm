#!/usr/bin/env bash
#SBATCH --job-name=csd8b-validate
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --mem=256G
#SBATCH --time=01:00:00

# This is a high-memory code/protocol validation job, not a model smoke test.
# It intentionally does not source empirical_common.sh: validation occurs
# before the formal commit and is allowed to inspect a dirty candidate tree.
set -Eeuo pipefail
umask 027

CSD_REPO_ROOT=${CSD_REPO_ROOT:-/home/da839/clean_distill}
CSD_TTT_PYTHON=${CSD_TTT_PYTHON:-/home/da839/.conda/envs/TTT/bin/python}
CSD_TORCH_OVERLAY=${CSD_TORCH_OVERLAY:-/home/da839/scratch_pi_mg269/da839/mfspd/pydeps-cu128}
CSD_VALIDATION_ROOT=${CSD_VALIDATION_ROOT:-/home/da839/scratch_pi_mg269/da839/clean_distill/validation}

cd "$CSD_REPO_ROOT"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate TTT
[[ "$(readlink -f "$CONDA_PREFIX/bin/python")" == "$(readlink -f "$CSD_TTT_PYTHON")" ]]
[[ -d "$CSD_TORCH_OVERLAY/torch" ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i 'RTX.*6000|PRO.*6000' >/dev/null

export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$CSD_VALIDATION_ROOT/tmp/job_${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
mkdir -p "$TMPDIR" "$CSD_VALIDATION_ROOT/logs" "$CSD_VALIDATION_ROOT/status"

# Environment provenance check only: no model weights or inference are loaded.
"$CSD_TTT_PYTHON" - <<'PY'
from pathlib import Path
import torch

overlay = Path("/home/da839/scratch_pi_mg269/da839/mfspd/pydeps-cu128").resolve()
module = Path(torch.__file__).resolve()
assert module.is_relative_to(overlay), (module, overlay)
assert str(torch.__version__).endswith("+cu128"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert "sm_90" in torch._C._cuda_getArchFlags().split()
PY

git diff --check
mapfile -t CSD_PYTHON_FILES < <(
  { git ls-files '*.py'; git ls-files --others --exclude-standard '*.py'; } | sort -u
)
(( ${#CSD_PYTHON_FILES[@]} > 0 ))
"$CSD_TTT_PYTHON" -m py_compile "${CSD_PYTHON_FILES[@]}"

mapfile -t CSD_SHELL_FILES < <(
  { git ls-files '*.sh' '*.slurm'; git ls-files --others --exclude-standard '*.sh' '*.slurm'; } | sort -u
)
(( ${#CSD_SHELL_FILES[@]} > 0 ))
for path in "${CSD_SHELL_FILES[@]}"; do
  bash -n "$path"
done

"$CSD_TTT_PYTHON" -m pytest -q -p no:cacheprovider

CSD_MARKER="$CSD_VALIDATION_ROOT/status/validation_${SLURM_JOB_ID}.done"
CSD_TEMPORARY="${CSD_MARKER}.tmp.$$"
printf 'completed_at=%s\njob_id=%s\nhead=%s\n' \
  "$(date -Is)" "$SLURM_JOB_ID" "$(git rev-parse HEAD)" > "$CSD_TEMPORARY"
mv "$CSD_TEMPORARY" "$CSD_MARKER"
