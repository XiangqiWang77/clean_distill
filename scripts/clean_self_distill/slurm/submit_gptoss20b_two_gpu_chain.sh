#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT=/home/da839/clean_distill
CONFIG=${CSD_GO_CONFIG:-$CODE_ROOT/configs/clean_self_distill/gptoss20b_fiveway.env}
# shellcheck disable=SC1090
source "$CONFIG"
mkdir -p "$CSD_GO_RUN_ROOT/logs"

SUBMISSION=${CSD_GO_2GPU_SUBMISSION_PATH:-$CSD_GO_RUN_ROOT/SUBMISSION_2GPU.json}
if [[ -s "$SUBMISSION" ]] && [[ "${CSD_GO_ALLOW_2GPU_RESUBMIT:-0}" != 1 ]]; then
  echo "Two-GPU submission already exists: $SUBMISSION" >&2
  exit 2
fi

running_h100=$(squeue -h -u "$USER" -p gpu_h100 -t R -o '%i' | wc -l)
if (( running_h100 + 2 > 8 )); then
  echo "Refusing submission: $running_h100 H100 jobs already run; next job needs two GPUs." >&2
  exit 2
fi

export CSD_GO_CONFIG="$CONFIG"
smoke_job=${CSD_GO_TWO_GPU_SMOKE_JOB_ID:-}
if [[ -n "$smoke_job" ]]; then
  [[ -s "$CSD_GO_RUN_ROOT/TWO_GPU_SMOKE.json" ]]
  [[ "$(sacct -j "$smoke_job" -X -n -o State | xargs)" == COMPLETED ]]
else
  smoke_job=$(sbatch --parsable --export=ALL,CSD_GO_CONFIG="$CONFIG" \
    "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_two_gpu_smoke.slurm")
fi
base_job=$(sbatch --parsable --dependency="afterok:$smoke_job" --export=ALL,CSD_GO_CONFIG="$CONFIG" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_base_math_eval.slurm")
p16_job=$(sbatch --parsable --dependency="afterok:$base_job" --export=ALL,CSD_GO_CONFIG="$CONFIG",CSD_GO_TRAIN_TAG=privileged_16 \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_train_single_2gpu.slurm")
p64_job=$(sbatch --parsable --dependency="afterok:$p16_job" --export=ALL,CSD_GO_CONFIG="$CONFIG",CSD_GO_TRAIN_TAG=privileged_64 \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_train_single_2gpu.slurm")
t16_job=$(sbatch --parsable --dependency="afterok:$p64_job" --export=ALL,CSD_GO_CONFIG="$CONFIG",CSD_GO_TRAIN_TAG=trsd_16 \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_train_single_2gpu.slurm")
t64_job=$(sbatch --parsable --dependency="afterok:$t16_job" --export=ALL,CSD_GO_CONFIG="$CONFIG",CSD_GO_TRAIN_TAG=trsd_64 \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_train_single_2gpu.slurm")
math_job=$(sbatch --parsable --dependency="afterok:$t64_job" --export=ALL,CSD_GO_CONFIG="$CONFIG" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_math_eval.slurm")
logic_job=$(sbatch --parsable --dependency="afterok:$math_job" --export=ALL,CSD_GO_CONFIG="$CONFIG" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_logic_eval.slurm")

env CSD_GO_RUN_ROOT="$CSD_GO_RUN_ROOT" SUBMISSION="$SUBMISSION" \
  SMOKE_JOB="$smoke_job" BASE_JOB="$base_job" P16_JOB="$p16_job" P64_JOB="$p64_job" \
  T16_JOB="$t16_job" T64_JOB="$t64_job" MATH_JOB="$math_job" LOGIC_JOB="$logic_job" \
  "$CSD_GO_PYTHON" -c 'import json,os,pathlib; pathlib.Path(os.environ["SUBMISSION"]).write_text(json.dumps({"schema_version":"gptoss20b-two-gpu-chain-v1","smoke":os.environ["SMOKE_JOB"],"base_math":os.environ["BASE_JOB"],"privileged_16":os.environ["P16_JOB"],"privileged_64":os.environ["P64_JOB"],"trsd_16":os.environ["T16_JOB"],"trsd_64":os.environ["T64_JOB"],"math_eval_array":os.environ["MATH_JOB"],"logic_array":os.environ["LOGIC_JOB"],"gpus_per_task":2,"peak_h100":8},indent=2)+"\n")'

printf 'smoke=%s base=%s p16=%s p64=%s t16=%s t64=%s math=%s logic=%s\n' \
  "$smoke_job" "$base_job" "$p16_job" "$p64_job" "$t16_job" "$t64_job" "$math_job" "$logic_job"
squeue -r -j "$smoke_job,$base_job,$p16_job,$p64_job,$t16_job,$t64_job,$math_job,$logic_job" \
  -o '%.18i %.12P %.28j %.2t %.10M %.10l %.4D %R'
