#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT=/home/da839/clean_distill
CONFIG="$CODE_ROOT/configs/clean_self_distill/qwen3_8b_logic_eval.env"
RUN_ROOT=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-logic-threeway-10k-20260808

source "$CONFIG"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmp"

# Six H100s must already account for the four Qwen3-1.7B math jobs and the
# two Qwen3-8B DemoPSD/GRPO math jobs. This array adds exactly the final two.
running_h100=$(squeue -h -u "$USER" -p gpu_h100 -t R -o '%i' | wc -l)
if (( running_h100 > 6 )); then
  echo "Refusing submission: $running_h100 H100 jobs are already running; at most 6 are allowed before this 2-GPU array." >&2
  exit 2
fi

export CSD_LOGIC_CONFIG="$CONFIG"
job_id=$(sbatch --parsable --export=ALL,CSD_LOGIC_CONFIG="$CONFIG" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/qwen8_logic_threeway_eval.slurm")
printf '%s\n' "$job_id" | tee "$RUN_ROOT/SLURM_ARRAY_JOB_ID"
