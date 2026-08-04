#!/usr/bin/env bash
# Submit prefetch -> B200 array -> afterok report.  Set DRY_RUN=1 to print only.
set -Eeuo pipefail
umask 027

CSD_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
# shellcheck disable=SC1091
source "$CSD_REPO_ROOT/configs/clean_self_distill/b200_poc.env"

CSD_GIT_COMMIT=$(git -C "$CSD_REPO_ROOT" rev-parse HEAD)
CSD_GIT_STATUS=$(git -C "$CSD_REPO_ROOT" status --porcelain=v1 --untracked-files=all)
[[ "$CSD_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]]
if [[ -n "$CSD_GIT_STATUS" ]]; then
  echo "Formal CSD submission requires a clean committed worktree" >&2
  exit 2
fi

# The requested experiment is the full 143-query scaled run.  A smoke profile
# remains available only when a caller opts into it explicitly.
RUN_PROFILE=${RUN_PROFILE:-full}
case "$RUN_PROFILE" in
  smoke)
    CSD_NUM_SHARDS=${NUM_SHARDS:-$CSD_SMOKE_NUM_SHARDS}
    CSD_MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-$CSD_SMOKE_MAX_EVAL_SAMPLES}
    CSD_NUM_CANDIDATES=${NUM_CANDIDATES:-$CSD_SMOKE_NUM_CANDIDATES}
    CSD_GPU_WALLTIME=${GPU_WALLTIME:-$CSD_SMOKE_GPU_WALLTIME}
    ;;
  full)
    CSD_NUM_SHARDS=${NUM_SHARDS:-$CSD_FULL_NUM_SHARDS}
    CSD_MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES-$CSD_FULL_MAX_EVAL_SAMPLES}
    CSD_NUM_CANDIDATES=${NUM_CANDIDATES:-$CSD_FULL_NUM_CANDIDATES}
    CSD_GPU_WALLTIME=${GPU_WALLTIME:-$CSD_FULL_GPU_WALLTIME}
    ;;
  *) echo "RUN_PROFILE must be smoke or full" >&2; exit 2 ;;
esac

CSD_SCRATCH_ROOT=${SCRATCH_ROOT:-$CSD_SCRATCH_ROOT}
CSD_CPU_PARTITION=${CPU_PARTITION:-$CSD_CPU_PARTITION}
CSD_RUN_SEED=${RUN_SEED:-0}
CSD_RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-${RUN_PROFILE}}
CSD_DRY_RUN=${DRY_RUN:-0}
CSD_RESUBMIT=${RESUBMIT:-0}
CSD_PREFETCH_ONLY=${PREFETCH_ONLY:-0}
CSD_ASSETS_READY=${ASSETS_READY:-0}
CSD_UPSTREAM_AFTEROK_JOB_ID=${UPSTREAM_AFTEROK_JOB_ID:-}

[[ "$CSD_RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "$CSD_NUM_SHARDS" =~ ^[0-9]+$ ]]
if [[ "$RUN_PROFILE" == full ]]; then
  (( CSD_NUM_SHARDS == 16 ))
else
  (( CSD_NUM_SHARDS == 2 ))
fi
[[ "$CSD_MAX_CONCURRENT_SHARDS" =~ ^[0-9]+$ ]]
(( CSD_MAX_CONCURRENT_SHARDS == 2 ))
[[ "$CSD_NUM_CANDIDATES" =~ ^[0-9]+$ ]] && (( CSD_NUM_CANDIDATES >= 1 ))
[[ "$CSD_PROPOSAL_OVERSAMPLE" =~ ^[0-9]+$ ]]
[[ "$CSD_PROPOSAL_MAX_ROUNDS" =~ ^[0-9]+$ ]] && (( CSD_PROPOSAL_MAX_ROUNDS >= 1 ))
[[ "$CSD_MIN_ACCEPTED_CANDIDATES" =~ ^[0-9]+$ ]]
(( CSD_MIN_ACCEPTED_CANDIDATES >= 1 && CSD_MIN_ACCEPTED_CANDIDATES <= CSD_NUM_CANDIDATES ))
[[ "$CSD_GPU_WALLTIME" =~ ^[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$ ]]
if [[ -n "$CSD_MAX_EVAL_SAMPLES" ]]; then
  [[ "$CSD_MAX_EVAL_SAMPLES" =~ ^[0-9]+$ ]]
  (( CSD_MAX_EVAL_SAMPLES >= CSD_NUM_SHARDS ))
fi
[[ "$CSD_MODEL_ID" == Qwen/Qwen3-4B ]]
[[ "$CSD_MODEL_REVISION" == 1cfa9a7208912126459214e8b04321603b3df60c ]]
[[ "$CSD_ACCOUNT" == pi_mg269 ]]
[[ "$CSD_GPU_PARTITION" == scavenge_gpu ]]
[[ "$CSD_CONDA_ENV" == TTT ]]
[[ "$CSD_TTT_PYTHON" == /home/da839/.conda/envs/TTT/bin/python ]]
[[ -x "$CSD_TTT_PYTHON" ]]
[[ "$CSD_TORCH_OVERLAY" == /home/da839/scratch_pi_mg269/da839/mfspd/pydeps-cu128 ]]
[[ -d "$CSD_TORCH_OVERLAY/torch" ]]
(( CSD_EVAL_MAX_NEW_TOKENS == 8192 ))
(( CSD_MAX_DOWNLOAD_BYTES < 10000000000 ))
[[ "$CSD_ASSETS_READY" == 0 || "$CSD_ASSETS_READY" == 1 ]]
if [[ -n "$CSD_UPSTREAM_AFTEROK_JOB_ID" ]]; then
  [[ "$CSD_UPSTREAM_AFTEROK_JOB_ID" =~ ^[0-9]+$ ]]
  [[ "$CSD_PREFETCH_ONLY" != 1 ]]
fi
if [[ "$CSD_ASSETS_READY" == 1 ]]; then
  [[ -z "$CSD_UPSTREAM_AFTEROK_JOB_ID" ]]
  [[ "$CSD_PREFETCH_ONLY" != 1 ]]
fi

CSD_RUN_ROOT="$CSD_SCRATCH_ROOT/runs/$CSD_RUN_ID"
CSD_DATA_ROOT="$CSD_SCRATCH_ROOT/data/verl"
CSD_EVAL_DATA="$CSD_DATA_ROOT/$CSD_DATASET_NAME/$CSD_DATASET_SPLIT.parquet"
CSD_HF_HOME="$CSD_SCRATCH_ROOT/hf"
CSD_HF_HUB_CACHE="$CSD_HF_HOME/hub"
CSD_ASSET_ROOT="$CSD_SCRATCH_ROOT/assets"
CSD_MODEL_MANIFEST="$CSD_ASSET_ROOT/qwen3-4b-${CSD_MODEL_REVISION}.json"
CSD_RUN_CONFIG="$CSD_RUN_ROOT/config/run.env"
CSD_JOBS_FILE="$CSD_RUN_ROOT/config/jobs.env"
mkdir -p "$CSD_RUN_ROOT"/{config,logs,status,shards,tmp,report_inputs} \
  "$CSD_DATA_ROOT" "$CSD_HF_HUB_CACHE" "$CSD_ASSET_ROOT"

CSD_CONFIG_TMP="${CSD_RUN_CONFIG}.tmp.$$"
{
  for CSD_NAME in \
    CSD_REPO_ROOT CSD_RUN_ROOT CSD_GIT_COMMIT CSD_ACCOUNT CSD_GPU_PARTITION CSD_CPU_PARTITION \
    CSD_CONDA_ENV CSD_MAX_CONCURRENT_SHARDS CSD_SCRATCH_ROOT CSD_TTT_PYTHON CSD_TORCH_OVERLAY \
    CSD_DATA_ROOT CSD_EVAL_DATA CSD_HF_HOME \
    CSD_HF_HUB_CACHE CSD_ASSET_ROOT CSD_MODEL_MANIFEST CSD_MODEL_ID \
    CSD_MODEL_REVISION CSD_DATASET_NAME CSD_DATASET_SPLIT CSD_MAX_DOWNLOAD_BYTES \
    CSD_EVAL_MAX_NEW_TOKENS CSD_PROPOSAL_OVERSAMPLE CSD_PROPOSAL_MAX_ROUNDS \
    CSD_MIN_ACCEPTED_CANDIDATES CSD_NUM_SHARDS CSD_MAX_EVAL_SAMPLES \
    CSD_NUM_CANDIDATES CSD_GPU_WALLTIME CSD_PREFETCH_WALLTIME \
    CSD_REPORT_WALLTIME CSD_RUN_SEED CSD_RUN_ID RUN_PROFILE; do
    printf '%s=%q\n' "$CSD_NAME" "${!CSD_NAME}"
  done
} > "$CSD_CONFIG_TMP"
if [[ -f "$CSD_RUN_CONFIG" ]]; then
  if ! cmp -s "$CSD_CONFIG_TMP" "$CSD_RUN_CONFIG"; then
    rm -f "$CSD_CONFIG_TMP"
    echo "RUN_ID=$CSD_RUN_ID already has a different immutable configuration" >&2
    exit 2
  fi
  rm -f "$CSD_CONFIG_TMP"
else
  mv "$CSD_CONFIG_TMP" "$CSD_RUN_CONFIG"
fi

if [[ -f "$CSD_JOBS_FILE" && "$CSD_RESUBMIT" != 1 ]]; then
  echo "RUN_ID=$CSD_RUN_ID was already submitted; use RESUBMIT=1 to resume incomplete stages" >&2
  exit 2
fi

csd_submit() {
  local dry_id=$1
  shift
  if [[ "$CSD_DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    printf '%s\n' "$dry_id"
    return
  fi
  local result
  result=$("$@")
  result=${result%%;*}
  [[ "$result" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch output: $result" >&2; return 1; }
  printf '%s\n' "$result"
}

csd_validate_ready_assets() {
  (
    set -Eeuo pipefail
    module purge
    module load miniconda
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CSD_CONDA_ENV"
    [[ "${CONDA_DEFAULT_ENV:-}" == "$CSD_CONDA_ENV" ]]
    [[ "$CONDA_PREFIX" == "${CSD_TTT_PYTHON%/bin/python}" ]]
    [[ -x "$CSD_TTT_PYTHON" ]]
    [[ -d "$CSD_TORCH_OVERLAY/torch" ]]
    export CSD_TTT_PYTHON CSD_TORCH_OVERLAY
    export PYTHONPATH="$CSD_TORCH_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
    local csd_python="$CSD_TTT_PYTHON"
    export HF_HOME="$CSD_HF_HOME"
    export HF_HUB_CACHE="$CSD_HF_HUB_CACHE"
    export HF_HUB_DISABLE_XET=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    "$csd_python" - <<'PY'
import os
import sys
from pathlib import Path

import math_verify
import peft
import pyarrow
import torch
import transformers

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
    "ready_assets_environment_gate=passed",
    f"python={sys.executable}",
    f"torch={torch.__version__}",
    f"torch_module={torch_path}",
    f"overlay={overlay}",
    f"arch_flags={' '.join(arch_flags)}",
)
PY
    "$csd_python" "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/launcher_support.py" \
      validate-dataset --dataset "$CSD_EVAL_DATA"
    "$csd_python" "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/launcher_support.py" \
      verify-model --model "$CSD_MODEL_ID" --revision "$CSD_MODEL_REVISION" \
      --cache-dir "$CSD_HF_HUB_CACHE" --manifest "$CSD_MODEL_MANIFEST"
    # Only model/data downloaded for this task count toward the 9.9 GB cap;
    # the existing CUDA 12.8 overlay is reused and deliberately excluded.
    "$csd_python" "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/launcher_support.py" \
      check-budget --path "$CSD_HF_HOME" --path "$CSD_DATA_ROOT" \
      --max-bytes "$CSD_MAX_DOWNLOAD_BYTES"
  )
}

CSD_EXPORT="ALL,CSD_RUN_CONFIG=$CSD_RUN_CONFIG"
if [[ "$CSD_ASSETS_READY" == 1 ]]; then
  if [[ "$CSD_DRY_RUN" != 1 ]]; then
    csd_validate_ready_assets
  fi
  CSD_PREFETCH_JOB_ID=validated_local
elif [[ -n "$CSD_UPSTREAM_AFTEROK_JOB_ID" ]]; then
  # Reuse the shared, pinned scratch assets only after a prior validated chain
  # has succeeded. This avoids concurrent writers in the dedicated HF cache.
  CSD_PREFETCH_JOB_ID=$CSD_UPSTREAM_AFTEROK_JOB_ID
else
  CSD_PREFETCH_JOB_ID=$(csd_submit DRY_PREFETCH \
    sbatch --parsable --account "$CSD_ACCOUNT" --partition "$CSD_CPU_PARTITION" \
    --time "$CSD_PREFETCH_WALLTIME" --export "$CSD_EXPORT" \
    --output "$CSD_RUN_ROOT/logs/prefetch-%j.out" \
    --error "$CSD_RUN_ROOT/logs/prefetch-%j.err" \
    "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/prefetch_assets.slurm")
fi
if [[ "$CSD_PREFETCH_ONLY" == 1 ]]; then
  if [[ "$CSD_DRY_RUN" != 1 ]]; then
    {
      printf 'CSD_PREFETCH_JOB_ID=%q\n' "$CSD_PREFETCH_JOB_ID"
      printf 'submitted_at=%q\n' "$(date -Is)"
    } > "${CSD_JOBS_FILE}.tmp.$$"
    mv "${CSD_JOBS_FILE}.tmp.$$" "$CSD_JOBS_FILE"
  fi
  printf 'run_root=%s\nprefetch_job=%s\n' "$CSD_RUN_ROOT" "$CSD_PREFETCH_JOB_ID"
  exit 0
fi
CSD_ARRAY_DEPENDENCY=()
if [[ "$CSD_PREFETCH_JOB_ID" =~ ^[0-9]+$ ]]; then
  CSD_ARRAY_DEPENDENCY=(--dependency "afterok:$CSD_PREFETCH_JOB_ID")
fi
CSD_ARRAY_JOB_ID=$(csd_submit DRY_ARRAY \
  sbatch --parsable --account "$CSD_ACCOUNT" --partition "$CSD_GPU_PARTITION" \
  --time "$CSD_GPU_WALLTIME" \
  --array "0-$((CSD_NUM_SHARDS - 1))%$CSD_MAX_CONCURRENT_SHARDS" \
  --gres gpu:b200:1 --requeue "${CSD_ARRAY_DEPENDENCY[@]}" --export "$CSD_EXPORT" \
  --output "$CSD_RUN_ROOT/logs/gpu-%A_%a.out" \
  --error "$CSD_RUN_ROOT/logs/gpu-%A_%a.err" \
  "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/run_shard.slurm")
CSD_REPORT_JOB_ID=$(csd_submit DRY_REPORT \
  sbatch --parsable --account "$CSD_ACCOUNT" --partition "$CSD_CPU_PARTITION" \
  --time "$CSD_REPORT_WALLTIME" --dependency "afterok:$CSD_ARRAY_JOB_ID" \
  --export "$CSD_EXPORT" \
  --output "$CSD_RUN_ROOT/logs/report-%j.out" \
  --error "$CSD_RUN_ROOT/logs/report-%j.err" \
  "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/merge_report.slurm")

if [[ "$CSD_DRY_RUN" != 1 ]]; then
  {
    printf 'CSD_PREFETCH_JOB_ID=%q\n' "$CSD_PREFETCH_JOB_ID"
    printf 'CSD_UPSTREAM_AFTEROK_JOB_ID=%q\n' "$CSD_UPSTREAM_AFTEROK_JOB_ID"
    printf 'CSD_ARRAY_JOB_ID=%q\n' "$CSD_ARRAY_JOB_ID"
    printf 'CSD_REPORT_JOB_ID=%q\n' "$CSD_REPORT_JOB_ID"
    printf 'submitted_at=%q\n' "$(date -Is)"
  } > "${CSD_JOBS_FILE}.tmp.$$"
  mv "${CSD_JOBS_FILE}.tmp.$$" "$CSD_JOBS_FILE"
fi
printf 'run_root=%s\nasset_or_upstream_job=%s\narray_job=%s\nreport_job=%s\n' \
  "$CSD_RUN_ROOT" "$CSD_PREFETCH_JOB_ID" "$CSD_ARRAY_JOB_ID" "$CSD_REPORT_JOB_ID"
