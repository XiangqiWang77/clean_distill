#!/usr/bin/env bash
#SBATCH --job-name=restore-throttle
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00

set -Eeuo pipefail
: "${CSD_TARGET_ARRAY_JOB:?export CSD_TARGET_ARRAY_JOB}"
: "${CSD_TARGET_ARRAY_THROTTLE:?export CSD_TARGET_ARRAY_THROTTLE}"

if squeue -h -j "$CSD_TARGET_ARRAY_JOB" | grep -q .; then
  scontrol update \
    JobId="$CSD_TARGET_ARRAY_JOB" \
    ArrayTaskThrottle="$CSD_TARGET_ARRAY_THROTTLE"
  echo "restored job $CSD_TARGET_ARRAY_JOB throttle to $CSD_TARGET_ARRAY_THROTTLE"
else
  echo "job $CSD_TARGET_ARRAY_JOB already left the queue; nothing to restore"
fi
