#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG=${CSD_Q17B64_CONFIG:-/home/da839/clean_distill/configs/clean_self_distill/qwen3_1p7b_baselines64.env}
TRAIN_SCRIPT=/home/da839/clean_distill/baselines/slurm/qwen17_deepmath64_train_eval.slurm
RESTORE_SCRIPT=/home/da839/clean_distill/baselines/slurm/restore_array_throttle.slurm
[[ -s "$CONFIG" && -s "$TRAIN_SCRIPT" && -s "$RESTORE_SCRIPT" ]]
# shellcheck disable=SC1090
source "$CONFIG"

mkdir -p "$CSD_Q17B64_RUN_ROOT/logs"
if [[ -s "$CSD_Q17B64_RUN_ROOT/SUBMISSION.json" ]] && [[ "${CSD_Q17B64_ALLOW_RESUBMIT:-0}" != 1 ]]; then
  echo "submission already recorded at $CSD_Q17B64_RUN_ROOT/SUBMISSION.json" >&2
  exit 2
fi

[[ -s "$CSD_Q17B64_MODEL_LOCAL_DIR/model.safetensors.index.json" ]]
[[ "$(wc -l < "$CSD_Q17B64_PREPARED_ROOT/distill_queries.jsonl")" == 64 ]]
[[ "$(sha256sum "$CSD_Q17B64_PREPARED_ROOT/distill_queries.jsonl" | awk '{print $1}')" == "$CSD_Q17B64_DISTILL_QUERIES_SHA256" ]]
[[ "$(sha256sum "$CSD_Q17B64_TRAIN_LABELS" | awk '{print $1}')" == "$CSD_Q17B64_TRAIN_LABELS_SHA256" ]]
[[ "$(wc -l < "$CSD_Q17B64_BASE_SCORED")" == 143 ]]

# Six GPT-OSS logic shards plus these two Qwen jobs preserve the global cap of
# eight H100s.  The dependent CPU job restores all eight logic shards as soon
# as both Qwen array tasks have released their GPUs.
throttle_changed=0
job_id=
restore_armed=0
rollback_submission() {
  if (( ! restore_armed )); then
    [[ -z "$job_id" ]] || scancel "$job_id" || true
    if (( throttle_changed )); then
      scontrol update \
        JobId="$CSD_Q17B64_GPTOSS_LOGIC_JOB_ID" \
        ArrayTaskThrottle=8 || true
    fi
  fi
}
trap rollback_submission EXIT

if squeue -h -j "$CSD_Q17B64_GPTOSS_LOGIC_JOB_ID" | grep -q .; then
  scontrol update \
    JobId="$CSD_Q17B64_GPTOSS_LOGIC_JOB_ID" \
    ArrayTaskThrottle=6
  throttle_changed=1
fi

job_id=$(sbatch --parsable \
  --export=ALL,CSD_Q17B64_CONFIG="$CONFIG" \
  "$TRAIN_SCRIPT")
RESTORE_JOB_ID=$(sbatch --parsable \
  --dependency="afterany:$job_id" \
  --export=ALL,CSD_TARGET_ARRAY_JOB="$CSD_Q17B64_GPTOSS_LOGIC_JOB_ID",CSD_TARGET_ARRAY_THROTTLE=8 \
  "$RESTORE_SCRIPT")
restore_armed=1
trap - EXIT

printf '{\n  "schema_version": "qwen17-baselines64-submission-v1",\n  "submitted_at": "%s",\n  "train_eval_array_job_id": "%s",\n  "restore_throttle_job_id": "%s",\n  "methods": ["demopsd", "grpo"],\n  "episodes": 64,\n  "group_size": 8,\n  "peak_added_h100": 2,\n  "global_h100_cap": 8,\n  "gptoss_logic_job_id": "%s",\n  "gptoss_logic_temporary_throttle": 6\n}\n' \
  "$(date -Is)" "$job_id" "$RESTORE_JOB_ID" "$CSD_Q17B64_GPTOSS_LOGIC_JOB_ID" \
  > "$CSD_Q17B64_RUN_ROOT/SUBMISSION.json"

echo "submitted Qwen3-1.7B DemoPSD/GRPO array $job_id (two H100s)"
echo "submitted throttle restore job $RESTORE_JOB_ID"
squeue -r -j "$job_id","$RESTORE_JOB_ID" \
  -o '%.18i %.12P %.28j %.2t %.10M %.10l %R'
