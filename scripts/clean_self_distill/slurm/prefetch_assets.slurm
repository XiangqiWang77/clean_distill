#!/usr/bin/env bash
#SBATCH --job-name=csd-prefetch
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -Eeuo pipefail
umask 027

: "${CSD_RUN_CONFIG:?submit_b200_poc.sh must export CSD_RUN_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"

CSD_STATUS_FILE="$CSD_RUN_ROOT/status/prefetch.status"
CSD_STAGE=bootstrap
CSD_FAILED_COMMAND=
mkdir -p "$CSD_RUN_ROOT/status" "$CSD_DATA_ROOT" "$CSD_HF_HUB_CACHE"

csd_write_status() {
  local state=$1
  local rc=${2:-0}
  local temporary="${CSD_STATUS_FILE}.tmp.$$"
  {
    printf 'state=%q\n' "$state"
    printf 'stage=%q\n' "$CSD_STAGE"
    printf 'exit_code=%q\n' "$rc"
    printf 'job_id=%q\n' "${SLURM_JOB_ID:-none}"
    printf 'failed_command=%q\n' "$CSD_FAILED_COMMAND"
    printf 'updated_at=%q\n' "$(date -Is)"
  } > "$temporary"
  mv "$temporary" "$CSD_STATUS_FILE"
}
trap 'CSD_FAILED_COMMAND=$BASH_COMMAND' ERR
trap 'rc=$?; if (( rc == 0 )); then csd_write_status complete 0; else csd_write_status failed "$rc"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
csd_write_status running 0

cd "$CSD_REPO_ROOT"
CSD_STAGE=environment
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
[[ "${CONDA_DEFAULT_ENV:-}" == "$CSD_CONDA_ENV" ]]
# YCRC's module-provided conda currently updates CONDA_PREFIX while leaving
# the base interpreter first on PATH. Select the activated environment
# explicitly so every node runs the requested TTT packages.
CSD_PYTHON="$CONDA_PREFIX/bin/python"
[[ -x "$CSD_PYTHON" ]]
"$CSD_PYTHON" -c 'import huggingface_hub, pyarrow, torch, transformers'

export HF_HOME="$CSD_HF_HOME"
export HF_HUB_CACHE="$CSD_HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

CSD_STAGE=dataset
if [[ ! -s "$CSD_EVAL_DATA" ]]; then
  "$CSD_PYTHON" scripts/download_data.py \
    --dataset "$CSD_DATASET_NAME" \
    --split "$CSD_DATASET_SPLIT" \
    --dst "$CSD_DATA_ROOT"
fi
"$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
  validate-dataset --dataset "$CSD_EVAL_DATA"

CSD_STAGE=model
"$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
  prefetch-model \
  --model "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --cache-dir "$CSD_HF_HUB_CACHE" \
  --manifest "$CSD_MODEL_MANIFEST" \
  --other-path "$CSD_DATA_ROOT" \
  --max-bytes "$CSD_MAX_DOWNLOAD_BYTES"

CSD_STAGE=budget
[[ -x "$CSD_B200_PYTHON" ]]
"$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
  check-budget \
  --path "$CSD_HF_HOME" \
  --path "$CSD_DATA_ROOT" \
  --path "$CSD_B200_ENV_ROOT" \
  --max-bytes "$CSD_MAX_DOWNLOAD_BYTES"

CSD_STAGE=done
printf 'prefetch_complete=%s\n' "$(date -Is)"
