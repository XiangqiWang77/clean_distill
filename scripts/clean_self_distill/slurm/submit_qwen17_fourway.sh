#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG=${CSD_Q17_CONFIG:-/home/da839/clean_distill/configs/clean_self_distill/qwen3_1p7b_fourway.env}
SCRIPT=/home/da839/clean_distill/scripts/clean_self_distill/slurm/qwen17_fourway_distill_eval.slurm
[[ -s "$CONFIG" ]]
[[ -s "$SCRIPT" ]]
# shellcheck disable=SC1090
source "$CONFIG"

mkdir -p "$CSD_Q17_RUN_ROOT/logs"
if [[ -s "$CSD_Q17_RUN_ROOT/SUBMITTED_JOB_ID" ]] && [[ "${CSD_Q17_ALLOW_RESUBMIT:-0}" != 1 ]]; then
  echo "Submission already recorded: $(<"$CSD_Q17_RUN_ROOT/SUBMITTED_JOB_ID")"
  exit 2
fi

[[ -s "$CSD_Q17_MODEL_LOCAL_DIR/model.safetensors.index.json" ]]
[[ "$(wc -l < "$CSD_Q17_SOURCE_PREPARED_ROOT/distill_queries.jsonl")" == 1000 ]]
[[ "$(sha256sum "$CSD_Q17_SOURCE_PREPARED_ROOT/distill_queries.jsonl" | awk '{print $1}')" == "$CSD_Q17_DISTILL_SOURCE_SHA256" ]]
[[ "$(wc -l < "$CSD_Q17_PREPARED_16_ROOT/distill_queries.jsonl")" == 16 ]]
[[ "$(wc -l < "$CSD_Q17_PREPARED_64_ROOT/distill_queries.jsonl")" == 64 ]]
[[ "$(sha256sum "$CSD_Q17_PREPARED_16_ROOT/distill_queries.jsonl" | awk '{print $1}')" == "$CSD_Q17_DISTILL_16_SHA256" ]]
[[ "$(sha256sum "$CSD_Q17_PREPARED_64_ROOT/distill_queries.jsonl" | awk '{print $1}')" == "$CSD_Q17_DISTILL_64_SHA256" ]]
for prepared in "$CSD_Q17_PREPARED_16_ROOT" "$CSD_Q17_PREPARED_64_ROOT"; do
  [[ "$(wc -l < "$prepared/heldout_queries.jsonl")" == 143 ]]
  [[ "$(wc -l < "$prepared/heldout_labels.sealed.jsonl")" == 143 ]]
  [[ "$(sha256sum "$prepared/heldout_queries.jsonl" | awk '{print $1}')" == "$CSD_Q17_HELDOUT_QUERIES_SHA256" ]]
  [[ "$(sha256sum "$prepared/heldout_labels.sealed.jsonl" | awk '{print $1}')" == "$CSD_Q17_HELDOUT_LABELS_SHA256" ]]
done

JOB_ID=$(sbatch --parsable --export=ALL,CSD_Q17_CONFIG="$CONFIG" "$SCRIPT")
printf '%s\n' "$JOB_ID" > "$CSD_Q17_RUN_ROOT/SUBMITTED_JOB_ID"
echo "submitted array job $JOB_ID with tasks 0-3 (four H100 jobs)"
squeue -r -j "$JOB_ID" -o '%.18i %.12P %.24j %.2t %.10M %.10l %R'
