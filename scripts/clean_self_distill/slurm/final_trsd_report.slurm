#!/usr/bin/env bash
#SBATCH --job-name=trsd-final-report
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

: "${CSD_BASE_SCORED:?set CSD_BASE_SCORED to the scored Base JSONL}"
: "${CSD_PRIV16_SCORED:?set CSD_PRIV16_SCORED to the scored Privilege-SD-16 JSONL}"
: "${CSD_TRSD16_SCORED:?set CSD_TRSD16_SCORED to the current matched reverse-KL TRSD-16 JSONL}"
: "${CSD_PRIV64_SCORED:?set CSD_PRIV64_SCORED to the scored Privilege-SD-64 JSONL}"
: "${CSD_TRSD64_SCORED:?set CSD_TRSD64_SCORED to the scored TRSD-64 JSONL}"

CSD_PRIV64_JOURNAL=${CSD_PRIV64_JOURNAL:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/episodes.jsonl}
CSD_TRSD64_JOURNAL=${CSD_TRSD64_JOURNAL:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/reverse-kl-matched64-20260807/trsd/train/episodes.jsonl}
CSD_HISTORICAL_TRSD16_SCORED=${CSD_HISTORICAL_TRSD16_SCORED:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-short-empirical-20260807/report/scored/trsd_episode_0016_10240.jsonl}
CSD_HISTORICAL_BASE_SCORED=${CSD_HISTORICAL_BASE_SCORED:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-short-empirical-20260807/report/scored/base_10240.jsonl}
CSD_MECHANISM_CSV=${CSD_MECHANISM_CSV:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-short-empirical-20260807/report/mechanism_query_wrapper.csv}
CSD_FINAL_REPORT_ROOT=${CSD_FINAL_REPORT_ROOT:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-final-matched-20260807}

mkdir -p "$CSD_FINAL_REPORT_ROOT" "$CSD_FINAL_REPORT_ROOT/matplotlib"
export MPLCONFIGDIR="$CSD_FINAL_REPORT_ROOT/matplotlib"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

cd /home/da839/clean_distill
/home/da839/.conda/envs/TTT/bin/python \
  scripts/clean_self_distill/10_final_trsd_report.py \
  --base-scored "$CSD_BASE_SCORED" \
  --privileged16-scored "$CSD_PRIV16_SCORED" \
  --trsd16-scored "$CSD_TRSD16_SCORED" \
  --historical-trsd16-scored "$CSD_HISTORICAL_TRSD16_SCORED" \
  --historical-base-scored "$CSD_HISTORICAL_BASE_SCORED" \
  --privileged64-scored "$CSD_PRIV64_SCORED" \
  --trsd64-scored "$CSD_TRSD64_SCORED" \
  --privileged64-journal "$CSD_PRIV64_JOURNAL" \
  --trsd64-journal "$CSD_TRSD64_JOURNAL" \
  --mechanism-csv "$CSD_MECHANISM_CSV" \
  --output-dir "$CSD_FINAL_REPORT_ROOT"
