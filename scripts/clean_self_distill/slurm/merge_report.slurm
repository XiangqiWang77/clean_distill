#!/usr/bin/env bash
#SBATCH --job-name=csd-report
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -Eeuo pipefail
umask 027
: "${CSD_RUN_CONFIG:?submit_b200_poc.sh must export CSD_RUN_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"

CSD_STATUS_FILE="$CSD_RUN_ROOT/status/report.status"
CSD_STAGE=bootstrap
CSD_FAILED_COMMAND=
mkdir -p "$CSD_RUN_ROOT/status" "$CSD_RUN_ROOT/report_inputs" "$CSD_RUN_ROOT/restarts"
csd_write_status() {
  local state=$1 rc=${2:-0} temporary="${CSD_STATUS_FILE}.tmp.$$"
  {
    printf 'state=%q\nstage=%q\nexit_code=%q\njob_id=%q\nfailed_command=%q\nupdated_at=%q\n' \
      "$state" "$CSD_STAGE" "$rc" "${SLURM_JOB_ID:-none}" "$CSD_FAILED_COMMAND" "$(date -Is)"
  } > "$temporary"
  mv "$temporary" "$CSD_STATUS_FILE"
}
trap 'CSD_FAILED_COMMAND=$BASH_COMMAND' ERR
trap 'rc=$?; if (( rc == 0 )); then csd_write_status complete 0; else csd_write_status failed "$rc"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
csd_write_status running 0

CSD_REPORT_DONE="$CSD_RUN_ROOT/status/report.done"
if [[ -f "$CSD_REPORT_DONE" ]]; then
  CSD_STAGE=already_complete
  exit 0
fi

cd "$CSD_REPO_ROOT"
CSD_STAGE=environment
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
[[ "${CONDA_DEFAULT_ENV:-}" == "$CSD_CONDA_ENV" ]]
CSD_PYTHON=$(command -v python)
"$CSD_PYTHON" -c 'import pyarrow'

CSD_STAGE=validate_shards
CSD_TASK_ARGS=()
for (( CSD_SHARD_INDEX=0; CSD_SHARD_INDEX<CSD_NUM_SHARDS; CSD_SHARD_INDEX++ )); do
  CSD_SHARD_LABEL=$(printf '%03d' "$CSD_SHARD_INDEX")
  CSD_SHARD_ROOT="$CSD_RUN_ROOT/shards/shard_${CSD_SHARD_LABEL}"
  CSD_PROPOSAL="$CSD_SHARD_ROOT/proposals/eval_proposals.jsonl"
  CSD_TASK1="$CSD_SHARD_ROOT/task1/evaluation/eval_task1_fast_teacher.jsonl"
  CSD_TASK2="$CSD_SHARD_ROOT/task2/evaluation/eval_task2_clean_distillation.jsonl"
  CSD_VALIDATE=(
    --dataset "$CSD_EVAL_DATA" --num-shards "$CSD_NUM_SHARDS"
    --shard-index "$CSD_SHARD_INDEX" --model "$CSD_MODEL_ID"
    --revision "$CSD_MODEL_REVISION"
  )
  if [[ -n "$CSD_MAX_EVAL_SAMPLES" ]]; then
    CSD_VALIDATE+=(--max-samples "$CSD_MAX_EVAL_SAMPLES")
  fi
  "$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
    validate-shard "${CSD_VALIDATE[@]}" --kind proposal --artifact "$CSD_PROPOSAL"
  "$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
    validate-shard "${CSD_VALIDATE[@]}" --kind task1 --artifact "$CSD_TASK1"
  "$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
    validate-shard "${CSD_VALIDATE[@]}" --kind task2 --artifact "$CSD_TASK2"
  CSD_TASK_ARGS+=(
    --proposal "$CSD_PROPOSAL" --task1 "$CSD_TASK1" --task2 "$CSD_TASK2"
  )
done

CSD_STAGE=prepare_report_dataset
CSD_REPORT_DATASET="$CSD_RUN_ROOT/report_inputs/eval_subset.parquet"
CSD_REPORT_METADATA="$CSD_RUN_ROOT/report_inputs/metadata.json"
CSD_PREPARE_ARGS=(
  --dataset "$CSD_EVAL_DATA" --output "$CSD_REPORT_DATASET"
  --metadata "$CSD_REPORT_METADATA"
)
if [[ -n "$CSD_MAX_EVAL_SAMPLES" ]]; then
  CSD_PREPARE_ARGS+=(--max-samples "$CSD_MAX_EVAL_SAMPLES")
fi
"$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
  prepare-report-dataset "${CSD_PREPARE_ARGS[@]}"
mapfile -t CSD_REPORT_DATA_ARGS < <(
  "$CSD_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
    report-args --metadata "$CSD_REPORT_METADATA"
)

CSD_STAGE=report
CSD_REPORT_ATTEMPT="$CSD_RUN_ROOT/report.attempt.${SLURM_JOB_ID:-$$}"
"$CSD_PYTHON" scripts/clean_self_distill/report_poc.py \
  "${CSD_REPORT_DATA_ARGS[@]}" \
  "${CSD_TASK_ARGS[@]}" \
  --output-dir "$CSD_REPORT_ATTEMPT" \
  --model "$CSD_MODEL_ID" \
  --max-tokens "$CSD_EVAL_MAX_NEW_TOKENS"

if [[ -e "$CSD_RUN_ROOT/report" ]]; then
  mv "$CSD_RUN_ROOT/report" "$CSD_RUN_ROOT/restarts/report.$(date +%s)"
fi
mv "$CSD_REPORT_ATTEMPT" "$CSD_RUN_ROOT/report"
printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "${SLURM_JOB_ID:-none}" > "${CSD_REPORT_DONE}.tmp.$$"
mv "${CSD_REPORT_DONE}.tmp.$$" "$CSD_REPORT_DONE"
CSD_STAGE=done
