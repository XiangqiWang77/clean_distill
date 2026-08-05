#!/usr/bin/env bash
# Shared, fail-closed environment bootstrap for the empirical PoC jobs.

set -Eeuo pipefail
umask 027

: "${CSD_RUN_CONFIG:?The empirical submitter must export CSD_RUN_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
export CSD_RUN_CONFIG CSD_RUN_ROOT CSD_REPO_ROOT CSD_CODE_ROOT
export CSD_GIT_COMMIT CSD_CODE_TREE_SHA256 CSD_TORCH_OVERLAY
export CSD_MODEL_ID CSD_MODEL_REVISION
export CSD_GPU_PARTITION CSD_GPU_GRES CSD_GPU_NAME_REGEX
export CSD_GPU_CAPABILITY_MAJOR CSD_GPU_CAPABILITY_MINOR CSD_GPU_ARCH_FLAG

cd "$CSD_REPO_ROOT"
[[ "$(readlink -f "$CSD_REPO_ROOT")" == "$(readlink -f "$CSD_CODE_ROOT")" ]]
[[ "$(cat .csd-commit)" == "$CSD_GIT_COMMIT" ]]
[[ "$(cat .csd-tree-sha256)" == "$CSD_CODE_TREE_SHA256" ]]
[[ "$CSD_CODE_TREE_SHA256" =~ ^[0-9a-f]{64}$ ]]

export TMPDIR="$CSD_RUN_ROOT/tmp/job_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-single}"
mkdir -p "$TMPDIR" "$CSD_RUN_ROOT/status"
CSD_ACTUAL_CODE_HASH=$(
  tar --format=gnu --sort=name --mtime='UTC 1970-01-01' \
    --owner=0 --group=0 --numeric-owner \
    --exclude='./.csd-commit' --exclude='./.csd-tree-sha256' \
    -cf - . | sha256sum | awk '{print $1}'
)
[[ "$CSD_ACTUAL_CODE_HASH" == "$CSD_CODE_TREE_SHA256" ]]
unset CSD_ACTUAL_CODE_HASH

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
[[ "${CONDA_DEFAULT_ENV:-}" == "$CSD_CONDA_ENV" ]]
[[ "$(readlink -f "$CONDA_PREFIX/bin/python")" == "$(readlink -f "$CSD_TTT_PYTHON")" ]]
[[ -x "$CSD_TTT_PYTHON" ]]
[[ -d "$CSD_TORCH_OVERLAY/torch" ]]

export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_RUN_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HF_HUB_CACHE"

case "$(readlink -f "$CSD_MODEL_LOCAL_DIR")" in
  "$(readlink -f "$CSD_SCRATCH_ROOT")"/*) ;;
  *) echo "Model is outside the task scratch root" >&2; exit 2 ;;
esac
case "$(readlink -f "$CSD_RUN_ROOT")" in
  "$(readlink -f "$CSD_SCRATCH_ROOT")"/*) ;;
  *) echo "Run root is outside the task scratch root" >&2; exit 2 ;;
esac

csd_atomic_marker() {
  local target=$1
  local temporary="${target}.tmp.$$"
  mkdir -p "$(dirname "$target")"
  printf 'completed_at=%s\njob_id=%s\ngit_commit=%s\n' \
    "$(date -Is)" "${SLURM_JOB_ID:-none}" "$CSD_GIT_COMMIT" > "$temporary"
  mv "$temporary" "$target"
}

csd_assert_gpu() {
  local required=$1
  local count
  count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  (( count == 1 ))
  nvidia-smi --query-gpu=name --format=csv,noheader | rg -i "$required" >/dev/null
}

csd_assert_model_gpu() {
  csd_assert_gpu "$CSD_GPU_NAME_REGEX"
  "$CSD_TTT_PYTHON" - <<'PY'
import os
import re
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
name = torch.cuda.get_device_name(0)
assert re.search(os.environ["CSD_GPU_NAME_REGEX"], name, re.IGNORECASE), name
expected = (
    int(os.environ["CSD_GPU_CAPABILITY_MAJOR"]),
    int(os.environ["CSD_GPU_CAPABILITY_MINOR"]),
)
assert torch.cuda.get_device_capability(0) == expected, (
    torch.cuda.get_device_capability(0),
    expected,
)
assert os.environ["CSD_GPU_ARCH_FLAG"] in torch._C._cuda_getArchFlags().split()
PY
}

csd_start_requeue_watchdog() {
  local walltime=$1
  local hours minutes seconds total delay
  IFS=: read -r hours minutes seconds <<< "$walltime"
  total=$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
  delay=$((total - 600))
  (( delay > 0 ))
  (
    sleep "$delay"
    printf 'checkpoint-safe requeue requested at %s for job %s\n' \
      "$(date -Is)" "$SLURM_JOB_ID" >&2
    scontrol requeue "$SLURM_JOB_ID"
  ) &
  CSD_REQUEUE_WATCHDOG_PID=$!
}

csd_stop_requeue_watchdog() {
  if [[ -n "${CSD_REQUEUE_WATCHDOG_PID:-}" ]]; then
    kill "$CSD_REQUEUE_WATCHDOG_PID" 2>/dev/null || true
    wait "$CSD_REQUEUE_WATCHDOG_PID" 2>/dev/null || true
    CSD_REQUEUE_WATCHDOG_PID=
  fi
}
