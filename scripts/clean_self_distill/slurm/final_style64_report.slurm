#!/usr/bin/env bash
#SBATCH --job-name=trsd-style64
#SBATCH --account=pi_mg269
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -Eeuo pipefail

CSD_STYLE_REPORT_ROOT=${CSD_STYLE_REPORT_ROOT:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-style64-final-20260808}
CSD_PRIV64_JOURNAL=${CSD_PRIV64_JOURNAL:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/episodes.jsonl}
CSD_TRSD64_JOURNAL=${CSD_TRSD64_JOURNAL:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/reverse-kl-matched64-20260807/trsd/train/episodes.jsonl}
CSD_MECHANISM_CSV=${CSD_MECHANISM_CSV:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/trsd-short-empirical-20260807/report/mechanism_query_wrapper.csv}

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate TTT
cd /home/da839/clean_distill

mkdir -p "$CSD_STYLE_REPORT_ROOT" "$CSD_STYLE_REPORT_ROOT/matplotlib" "$CSD_STYLE_REPORT_ROOT/tmp_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$CSD_STYLE_REPORT_ROOT/matplotlib"
export TMPDIR="$CSD_STYLE_REPORT_ROOT/tmp_${SLURM_JOB_ID}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

/home/da839/.conda/envs/TTT/bin/python \
  scripts/clean_self_distill/11_final_style64_summary.py \
  --privileged64-journal "$CSD_PRIV64_JOURNAL" \
  --trsd64-journal "$CSD_TRSD64_JOURNAL" \
  --mechanism-csv "$CSD_MECHANISM_CSV" \
  --output-dir "$CSD_STYLE_REPORT_ROOT"
