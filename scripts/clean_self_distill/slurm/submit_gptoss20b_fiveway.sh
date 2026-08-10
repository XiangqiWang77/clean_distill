#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT=/home/da839/clean_distill
CONFIG=${CSD_GO_CONFIG:-$CODE_ROOT/configs/clean_self_distill/gptoss20b_fiveway.env}
# shellcheck disable=SC1090
source "$CONFIG"
mkdir -p "$CSD_GO_RUN_ROOT/logs"

if [[ -s "$CSD_GO_RUN_ROOT/SUBMISSION.json" ]] && [[ "${CSD_GO_ALLOW_RESUBMIT:-0}" != 1 ]]; then
  echo "Submission already exists: $CSD_GO_RUN_ROOT/SUBMISSION.json" >&2
  exit 2
fi

running_h100=$(squeue -h -u "$USER" -p gpu_h100 -t R -o '%i' | wc -l)
if (( running_h100 > 0 )); then
  echo "Refusing submission: $running_h100 H100 jobs are already running; the final stage needs all eight slots." >&2
  exit 2
fi

[[ -s "$CSD_GO_MODEL_LOCAL_DIR/model.safetensors.index.json" ]]
[[ "$(sha256sum "$CSD_GO_MODEL_LOCAL_DIR/config.json" | awk '{print $1}')" == "$CSD_GO_MODEL_CONFIG_SHA256" ]]
[[ "$(sha256sum "$CSD_GO_MODEL_LOCAL_DIR/model.safetensors.index.json" | awk '{print $1}')" == "$CSD_GO_MODEL_INDEX_SHA256" ]]
[[ "$(wc -l < "$CSD_GO_PREPARED_16_ROOT/distill_queries.jsonl")" == 16 ]]
[[ "$(wc -l < "$CSD_GO_PREPARED_64_ROOT/distill_queries.jsonl")" == 64 ]]
[[ "$(sha256sum "$CSD_GO_PREPARED_16_ROOT/distill_queries.jsonl" | awk '{print $1}')" == "$CSD_GO_DISTILL_16_SHA256" ]]
[[ "$(sha256sum "$CSD_GO_PREPARED_64_ROOT/distill_queries.jsonl" | awk '{print $1}')" == "$CSD_GO_DISTILL_64_SHA256" ]]
[[ "$(sha256sum "$CSD_GO_PREPARED_16_ROOT/heldout_queries.jsonl" | awk '{print $1}')" == "$CSD_GO_HELDOUT_QUERIES_SHA256" ]]
[[ "$(sha256sum "$CSD_GO_PREPARED_16_ROOT/heldout_labels.sealed.jsonl" | awk '{print $1}')" == "$CSD_GO_HELDOUT_LABELS_SHA256" ]]

export CSD_GO_CONFIG="$CONFIG"
smoke_job=$(sbatch --parsable --export=ALL,CSD_GO_CONFIG="$CONFIG" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_smoke.slurm")
base_math_job=$(sbatch --parsable --dependency="afterok:$smoke_job" --export=ALL,CSD_GO_CONFIG="$CONFIG" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_base_math_eval.slurm")
train_job=$(sbatch --parsable --dependency="afterok:$smoke_job" --export=ALL,CSD_GO_CONFIG="$CONFIG" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_train.slurm")
math_job=$(sbatch --parsable --dependency="afterok:$train_job" --export=ALL,CSD_GO_CONFIG="$CONFIG" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_math_eval.slurm")
logic_job=$(sbatch --parsable --dependency="afterok:$base_math_job:$math_job" --export=ALL,CSD_GO_CONFIG="$CONFIG" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/gptoss20b_logic_eval.slurm")

env SMOKE_JOB="$smoke_job" BASE_MATH_JOB="$base_math_job" TRAIN_JOB="$train_job" \
  MATH_JOB="$math_job" LOGIC_JOB="$logic_job" CSD_GO_RUN_ROOT="$CSD_GO_RUN_ROOT" \
  "$CSD_GO_PYTHON" -c 'import json,os,pathlib; pathlib.Path(os.environ["CSD_GO_RUN_ROOT"],"SUBMISSION.json").write_text(json.dumps({"smoke":os.environ["SMOKE_JOB"],"base_math":os.environ["BASE_MATH_JOB"],"train_array":os.environ["TRAIN_JOB"],"math_eval_array":os.environ["MATH_JOB"],"logic_array":os.environ["LOGIC_JOB"],"peak_h100":8},indent=2)+"\n")'

printf 'smoke=%s base_math=%s train=%s math_eval=%s logic=%s peak_h100=8\n' \
  "$smoke_job" "$base_math_job" "$train_job" "$math_job" "$logic_job"
squeue -r -j "$smoke_job,$base_math_job,$train_job,$math_job,$logic_job" \
  -o '%.18i %.12P %.28j %.2t %.10M %.10l %R'
