#!/usr/bin/env bash
#SBATCH --job-name=q8-int-wait
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-intermediate-checkpoints-20260809/logs/%j-wait-report.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-intermediate-checkpoints-20260809/logs/%j-wait-report.err

set -Eeuo pipefail
: "${CSD_I8_CONFIG:?export CSD_I8_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_I8_CONFIG"

for tag in privileged_32 privileged_48 trsd_32 trsd_48; do
  for shard in 00 01 02 03; do
    done_file="$CSD_I8_RUN_ROOT/eval/$tag/status/shard_${shard}.done"
    until [[ -s "$done_file" ]]; do sleep 60; done
  done
done
until [[ -s "$CSD_I8_DYNAMICS_ROOT/DYNAMICS_COMPLETE" ]]; do sleep 60; done

export CSD_I8_CONFIG
bash "$CSD_I8_CODE_ROOT/scripts/clean_self_distill/slurm/qwen8_intermediate_report.slurm"
