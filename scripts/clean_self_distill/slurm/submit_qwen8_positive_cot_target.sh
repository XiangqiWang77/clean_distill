#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
config="${1:-$repo_root/configs/clean_self_distill/qwen3_8b_positive_cot_target.env}"
config="$(realpath "$config")"
source "$config"

mkdir -p "$CSD_PCOT_RUN_ROOT/logs" "$CSD_PCOT_RUN_ROOT/status"
job_id="$(
  sbatch --parsable \
    --export="ALL,CSD_PCOT_CONFIG=$config" \
    --output="$CSD_PCOT_RUN_ROOT/logs/%x-%j.out" \
    --error="$CSD_PCOT_RUN_ROOT/logs/%x-%j.err" \
    "$repo_root/scripts/clean_self_distill/slurm/qwen8_positive_cot_target.slurm"
)"
printf 'job_id=%s\nrun_root=%s\nconfig=%s\n' "$job_id" "$CSD_PCOT_RUN_ROOT" "$config"
