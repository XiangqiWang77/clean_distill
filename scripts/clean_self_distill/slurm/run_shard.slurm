#!/usr/bin/env bash
#SBATCH --job-name=csd-b200
#SBATCH --account=pi_mg269
#SBATCH --partition=scavenge_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --requeue

set -Eeuo pipefail
umask 027

: "${CSD_RUN_CONFIG:?submit_b200_poc.sh must export CSD_RUN_CONFIG}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"

CSD_SHARD_INDEX=$SLURM_ARRAY_TASK_ID
(( CSD_SHARD_INDEX >= 0 && CSD_SHARD_INDEX < CSD_NUM_SHARDS ))
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  (( SLURM_ARRAY_TASK_COUNT == CSD_NUM_SHARDS ))
fi
CSD_SHARD_LABEL=$(printf '%03d' "$CSD_SHARD_INDEX")
CSD_SHARD_ROOT="$CSD_RUN_ROOT/shards/shard_${CSD_SHARD_LABEL}"
CSD_STATUS_FILE="$CSD_SHARD_ROOT/status.env"
CSD_STAGE=bootstrap
CSD_FAILED_COMMAND=
CSD_WATCHDOG_PID=
mkdir -p "$CSD_SHARD_ROOT" "$CSD_SHARD_ROOT/markers" "$CSD_RUN_ROOT/tmp"

csd_write_status() {
  local state=$1 rc=${2:-0}
  local temporary="${CSD_STATUS_FILE}.tmp.$$"
  {
    printf 'state=%q\n' "$state"
    printf 'stage=%q\n' "$CSD_STAGE"
    printf 'exit_code=%q\n' "$rc"
    printf 'job_id=%q\n' "${SLURM_JOB_ID:-none}"
    printf 'array_job_id=%q\n' "${SLURM_ARRAY_JOB_ID:-none}"
    printf 'shard_index=%q\n' "$CSD_SHARD_INDEX"
    printf 'failed_command=%q\n' "$CSD_FAILED_COMMAND"
    printf 'updated_at=%q\n' "$(date -Is)"
  } > "$temporary"
  mv "$temporary" "$CSD_STATUS_FILE"
}
csd_mark_done() {
  local marker=$1 temporary="${marker}.tmp.$$"
  printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "${SLURM_JOB_ID:-none}" > "$temporary"
  mv "$temporary" "$marker"
}
csd_stop_watchdog() {
  if [[ -n "$CSD_WATCHDOG_PID" ]]; then
    kill "$CSD_WATCHDOG_PID" 2>/dev/null || true
    wait "$CSD_WATCHDOG_PID" 2>/dev/null || true
    CSD_WATCHDOG_PID=
  fi
}
csd_on_exit() {
  local rc=$?
  csd_stop_watchdog
  if (( rc == 0 )); then
    csd_write_status complete 0
  else
    csd_write_status failed "$rc"
  fi
}
csd_start_requeue_watchdog() {
  local hours minutes seconds walltime_seconds watchdog_delay
  IFS=: read -r hours minutes seconds <<< "$CSD_GPU_WALLTIME"
  walltime_seconds=$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
  watchdog_delay=$((walltime_seconds - 600))
  (( watchdog_delay > 0 ))
  (
    watchdog_sleep_pid=
    csd_watchdog_cleanup() {
      if [[ -n "$watchdog_sleep_pid" ]]; then
        kill "$watchdog_sleep_pid" 2>/dev/null || true
        wait "$watchdog_sleep_pid" 2>/dev/null || true
      fi
      exit 0
    }
    trap csd_watchdog_cleanup TERM INT HUP
    sleep "$watchdog_delay" &
    watchdog_sleep_pid=$!
    wait "$watchdog_sleep_pid"
    watchdog_sleep_pid=
    printf 'automatic_requeue_requested=%s job=%s shard=%s\n' \
      "$(date -Is)" "$SLURM_JOB_ID" "$CSD_SHARD_INDEX" >&2
    scontrol requeue Incomplete "$SLURM_JOB_ID"
  ) &
  CSD_WATCHDOG_PID=$!
}
trap 'CSD_FAILED_COMMAND=$BASH_COMMAND' ERR
trap csd_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
csd_write_status running 0
csd_start_requeue_watchdog

cd "$CSD_REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$CSD_GIT_COMMIT" ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]]
CSD_STAGE=environment
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
[[ "${CONDA_DEFAULT_ENV:-}" == "$CSD_CONDA_ENV" ]]
[[ "$CSD_TTT_PYTHON" == /home/da839/.conda/envs/TTT/bin/python ]]
[[ "$CONDA_PREFIX" == "${CSD_TTT_PYTHON%/bin/python}" ]]
[[ "$CSD_TORCH_OVERLAY" == /home/da839/scratch_pi_mg269/da839/mfspd/pydeps-cu128 ]]
[[ -d "$CSD_TORCH_OVERLAY/torch" ]]
export CSD_TTT_PYTHON CSD_TORCH_OVERLAY
export PYTHONPATH="$CSD_TORCH_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
CSD_PYTHON="$CSD_TTT_PYTHON"
[[ -x "$CSD_PYTHON" ]]
"$CSD_PYTHON" - <<'PY'
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
assert Path(sys.executable).resolve() == expected_python, sys.executable
assert Path(os.environ["CONDA_PREFIX"]).resolve() == expected_python.parent.parent
assert torch_path.is_relative_to(overlay), (torch_path, overlay)
assert str(torch.__version__).endswith("+cu128"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.device_count() == 1
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
arch_flags = torch._C._cuda_getArchFlags().split()
assert "B200" in name.upper(), name
assert capability == (10, 0), capability
assert "sm_100" in arch_flags, arch_flags
probe = torch.arange(8, device="cuda", dtype=torch.float32)
probe = (probe + 1).square().sum()
torch.cuda.synchronize()
assert probe.item() == 204.0, probe.item()
print(
    "b200_environment_gate=passed",
    f"python={sys.executable}",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"torch_module={torch_path}",
    f"overlay={overlay}",
    f"arch_flags={' '.join(arch_flags)}",
    f"gpu={name}",
    f"capability={capability}",
)
PY
mapfile -t CSD_GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#CSD_GPU_NAMES[@]} == 1 ))
[[ "${CSD_GPU_NAMES[0]}" == *B200* ]]

[[ -s "$CSD_EVAL_DATA" ]]
export HF_HOME="$CSD_HF_HOME"
export HF_HUB_CACHE="$CSD_HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TMPDIR="$CSD_RUN_ROOT/tmp/job_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${CSD_SHARD_LABEL}"
mkdir -p "$TMPDIR"
"$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
  verify-model --model "$CSD_MODEL_ID" --revision "$CSD_MODEL_REVISION" \
  --cache-dir "$CSD_HF_HUB_CACHE" --manifest "$CSD_MODEL_MANIFEST"

CSD_LIMIT_ARGS=()
if [[ -n "$CSD_MAX_EVAL_SAMPLES" ]]; then
  CSD_LIMIT_ARGS+=(--max-samples "$CSD_MAX_EVAL_SAMPLES")
fi
CSD_EVAL_LIMIT_ARGS=()
if [[ -n "$CSD_MAX_EVAL_SAMPLES" ]]; then
  CSD_EVAL_LIMIT_ARGS+=(--max-eval-samples "$CSD_MAX_EVAL_SAMPLES")
fi
CSD_VALIDATE_COMMON=(
  --dataset "$CSD_EVAL_DATA"
  --num-shards "$CSD_NUM_SHARDS"
  --shard-index "$CSD_SHARD_INDEX"
  --model "$CSD_MODEL_ID"
  --revision "$CSD_MODEL_REVISION"
)
if [[ -n "$CSD_MAX_EVAL_SAMPLES" ]]; then
  CSD_VALIDATE_COMMON+=(--max-samples "$CSD_MAX_EVAL_SAMPLES")
fi

CSD_PROPOSALS="$CSD_SHARD_ROOT/proposals/eval_proposals.jsonl"
CSD_PROPOSAL_MARKER="$CSD_SHARD_ROOT/markers/proposals.done"
mkdir -p "$(dirname "$CSD_PROPOSALS")"
CSD_STAGE=proposals
if [[ ! -f "$CSD_PROPOSAL_MARKER" ]]; then
  "$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
    repair-proposals --path "$CSD_PROPOSALS"
  "$CSD_PYTHON" scripts/clean_self_distill/01_propose.py \
    --input "$CSD_EVAL_DATA" \
    --output "$CSD_PROPOSALS" \
    --model "$CSD_MODEL_ID" \
    --revision "$CSD_MODEL_REVISION" \
    --num-candidates "$CSD_NUM_CANDIDATES" \
    --proposal-oversample "$CSD_PROPOSAL_OVERSAMPLE" \
    --max-rounds "$CSD_PROPOSAL_MAX_ROUNDS" \
    --min-accepted-candidates "$CSD_MIN_ACCEPTED_CANDIDATES" \
    --max-new-tokens 1536 \
    --temperature 0.8 \
    --solver-temperature 0.3 \
    --verifier-temperature 0 \
    --top-p 0.95 \
    --num-shards "$CSD_NUM_SHARDS" \
    --shard-index "$CSD_SHARD_INDEX" \
    --resume \
    --seed "$CSD_RUN_SEED" \
    "${CSD_LIMIT_ARGS[@]}"
  "$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
    validate-shard "${CSD_VALIDATE_COMMON[@]}" \
    --kind proposal --artifact "$CSD_PROPOSALS"
  csd_mark_done "$CSD_PROPOSAL_MARKER"
fi

CSD_ADAPTER_DIR="$CSD_SHARD_ROOT/task1/adapters"
CSD_ADAPTER_MARKER="$CSD_SHARD_ROOT/markers/adapters.done"
CSD_STAGE=task1_adapters
if [[ ! -f "$CSD_ADAPTER_MARKER" ]]; then
  mkdir -p "$CSD_ADAPTER_DIR"
  "$CSD_PYTHON" scripts/clean_self_distill/02_specialize.py \
    --proposals "$CSD_PROPOSALS" \
    --output-dir "$CSD_ADAPTER_DIR" \
    --model "$CSD_MODEL_ID" \
    --revision "$CSD_MODEL_REVISION" \
    --ridge-lambda 0.1 \
    --residual-step-size 0.8 \
    --max-support-tokens 256 \
    --max-tokens-per-candidate 64 \
    --hard-negatives 8 \
    --max-length 8192
  "$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
    validate-shard "${CSD_VALIDATE_COMMON[@]}" \
    --kind adapter --artifact "$CSD_ADAPTER_DIR/manifest.jsonl"
  csd_mark_done "$CSD_ADAPTER_MARKER"
fi

CSD_TASK1_DIR="$CSD_SHARD_ROOT/task1/evaluation"
CSD_TASK1_JSONL="$CSD_TASK1_DIR/eval_task1_fast_teacher.jsonl"
CSD_TASK1_MARKER="$CSD_SHARD_ROOT/markers/task1.done"
CSD_STAGE=task1_base_privileged_csdt
if [[ ! -f "$CSD_TASK1_MARKER" ]]; then
  mkdir -p "$CSD_TASK1_DIR"
  "$CSD_PYTHON" scripts/clean_self_distill/03_train_eval.py \
    --mode task1 \
    --eval-data "$CSD_EVAL_DATA" \
    --proposals "$CSD_PROPOSALS" \
    --adapter-dir "$CSD_ADAPTER_DIR" \
    --model "$CSD_MODEL_ID" \
    --revision "$CSD_MODEL_REVISION" \
    --output-dir "$CSD_TASK1_DIR" \
    --privileged-control \
    --ridge-lambda 0.1 \
    --residual-step-size 0.8 \
    --max-support-tokens 256 \
    --max-length 8192 \
    --eval-max-new-tokens "$CSD_EVAL_MAX_NEW_TOKENS" \
    --eval-samples 1 \
    --eval-temperature 0 \
    --top-p 1.0 \
    --top-k 0 \
    --num-shards "$CSD_NUM_SHARDS" \
    --shard-index "$CSD_SHARD_INDEX" \
    --seed "$CSD_RUN_SEED" \
    "${CSD_EVAL_LIMIT_ARGS[@]}"
  "$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
    validate-shard "${CSD_VALIDATE_COMMON[@]}" \
    --kind task1 --artifact "$CSD_TASK1_JSONL"
  csd_mark_done "$CSD_TASK1_MARKER"
fi

CSD_TASK2_DIR="$CSD_SHARD_ROOT/task2/evaluation"
CSD_TASK2_JSONL="$CSD_TASK2_DIR/eval_task2_clean_distillation.jsonl"
CSD_TASK2_MARKER="$CSD_SHARD_ROOT/markers/task2.done"
CSD_STAGE=task2_csd_sd
if [[ ! -f "$CSD_TASK2_MARKER" ]]; then
  mkdir -p "$CSD_TASK2_DIR"
  "$CSD_PYTHON" scripts/clean_self_distill/03_train_eval.py \
    --mode task2 \
    --eval-data "$CSD_EVAL_DATA" \
    --proposals "$CSD_PROPOSALS" \
    --model "$CSD_MODEL_ID" \
    --revision "$CSD_MODEL_REVISION" \
    --output-dir "$CSD_TASK2_DIR" \
    --ridge-lambda 0.1 \
    --residual-step-size 0.8 \
    --max-support-tokens 256 \
    --hard-negatives 8 \
    --max-length 8192 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --distillation-steps 3 \
    --learning-rate 2e-5 \
    --distill-top-k 64 \
    --distill-temperature 1.0 \
    --distill-token-clip 0 \
    --train-max-new-tokens 512 \
    --eval-max-new-tokens "$CSD_EVAL_MAX_NEW_TOKENS" \
    --eval-samples 1 \
    --eval-temperature 0 \
    --top-p 1.0 \
    --top-k 0 \
    --num-shards "$CSD_NUM_SHARDS" \
    --shard-index "$CSD_SHARD_INDEX" \
    --seed "$CSD_RUN_SEED" \
    "${CSD_EVAL_LIMIT_ARGS[@]}"
  "$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
    validate-shard "${CSD_VALIDATE_COMMON[@]}" \
    --kind task2 --artifact "$CSD_TASK2_JSONL"
  csd_mark_done "$CSD_TASK2_MARKER"
fi

CSD_STAGE=done
printf 'shard_complete=%s shard=%s/%s\n' "$(date -Is)" "$CSD_SHARD_INDEX" "$CSD_NUM_SHARDS"
