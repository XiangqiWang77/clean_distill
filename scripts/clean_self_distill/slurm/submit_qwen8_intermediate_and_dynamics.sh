#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT=/home/da839/clean_distill
CONFIG=${CSD_I8_CONFIG:-$CODE_ROOT/configs/clean_self_distill/qwen3_8b_intermediate_checkpoints.env}
A2_CONFIG=${CSD_A2_CONFIG:-$CODE_ROOT/configs/clean_self_distill/qwen3_8b_component_ablations64.env}
# shellcheck disable=SC1090
source "$CONFIG"
# shellcheck disable=SC1090
source "$A2_CONFIG"

[[ -x "$CSD_I8_PYTHON" ]]
[[ -d "$CSD_I8_MODEL_LOCAL_DIR" ]]
[[ -s "$CSD_I8_PREPARED_ROOT/heldout_queries.jsonl" ]]
[[ -s "$CSD_I8_PREPARED_ROOT/heldout_labels.sealed.jsonl" ]]
[[ "$(wc -l < "$CSD_I8_PREPARED_ROOT/heldout_queries.jsonl")" == 143 ]]
[[ "$(wc -l < "$CSD_I8_PRIVILEGED_JOURNAL")" == 64 ]]
[[ "$(wc -l < "$CSD_I8_TRSD_TRAIN_ROOT/episodes.jsonl")" == 64 ]]
[[ "$(sha256sum "$CSD_I8_PRIVILEGED64_CHECKPOINT/adapter_model.safetensors" | cut -d' ' -f1)" == "$CSD_I8_PRIVILEGED64_ADAPTER_SHA256" ]]
[[ "$(sha256sum "$CSD_I8_TRSD_TRAIN_ROOT/checkpoints/episode_0064/adapter_model.safetensors" | cut -d' ' -f1)" == "$CSD_I8_TRSD64_ADAPTER_SHA256" ]]
for episode in 16 32 48 64; do
  [[ -s "$CSD_I8_TRSD_TRAIN_ROOT/checkpoints/episode_$(printf '%04d' "$episode")/checkpoint_manifest.json" ]]
done

if [[ -s "$CSD_I8_RUN_ROOT/SUBMISSION.json" ]]; then
  echo "Already submitted: $CSD_I8_RUN_ROOT/SUBMISSION.json" >&2
  exit 2
fi
mkdir -p "$CSD_I8_RUN_ROOT"/{logs,eval,hf/hub,report} "$CSD_I8_SNAPSHOT_ROOT"
rsync -a "$CODE_ROOT/src/" "$CSD_I8_SNAPSHOT_ROOT/src/"
rsync -a "$CODE_ROOT/scripts/" "$CSD_I8_SNAPSHOT_ROOT/scripts/"
rsync -a "$CODE_ROOT/baselines/" "$CSD_I8_SNAPSHOT_ROOT/baselines/"
(
  cd "$CSD_I8_SNAPSHOT_ROOT"
  find src scripts baselines -type f -print0 | sort -z | xargs -0 sha256sum
) > "$CSD_I8_RUN_ROOT/code_snapshot.sha256"

submit() {
  local result
  result=$(sbatch --parsable "$@")
  result=${result%%;*}
  [[ "$result" =~ ^[0-9]+$ ]]
  printf '%s\n' "$result"
}

# The prior ablation submission was cancelled because it depended on the
# intentionally incomplete GPT-OSS logic array.  The ablation inputs are now
# independently complete, so submit it without that unrelated dependency.
ablation_job=$(submit \
  --array=0-5%6 \
  --export=ALL,CSD_A2_CONFIG="$A2_CONFIG",CSD_A2_CODE_ROOT_OVERRIDE="$CSD_I8_SNAPSHOT_ROOT" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/qwen8_component_ablation64_b200.slurm")
ablation_report_job=$(submit \
  --dependency="afterok:$ablation_job" \
  --export=ALL,CSD_A2_CONFIG="$A2_CONFIG",CSD_A2_CODE_ROOT_OVERRIDE="$CSD_I8_SNAPSHOT_ROOT" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/qwen8_component_ablation64_report.slurm")

common_exports="ALL,CSD_I8_CONFIG=$CONFIG"
replay_job=$(submit \
  --output="$CSD_I8_RUN_ROOT/logs/privileged_replay_%j.out" \
  --error="$CSD_I8_RUN_ROOT/logs/privileged_replay_%j.err" \
  --export="$common_exports" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/qwen8_privileged_replay.slurm")
dynamics_job=$(submit \
  --output="$CSD_I8_RUN_ROOT/logs/common_nll_%j.out" \
  --error="$CSD_I8_RUN_ROOT/logs/common_nll_%j.err" \
  --export="$common_exports" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/qwen8_common_eval_nll.slurm")

submit_eval() {
  local dependency=$1 method=$2 episode=$3 checkpoint=$4 tag=$5 throttle=$6
  local dependency_args=()
  [[ -z "$dependency" ]] || dependency_args=(--dependency="$dependency")
  submit \
    "${dependency_args[@]}" \
    --array="0-3%$throttle" \
    --job-name="q8-${tag}" \
    --output="$CSD_I8_RUN_ROOT/logs/${tag}_%A_%a.out" \
    --error="$CSD_I8_RUN_ROOT/logs/${tag}_%A_%a.err" \
    --export="$common_exports,CSD_I8_METHOD=$method,CSD_I8_EPISODE=$episode,CSD_I8_CHECKPOINT=$checkpoint,CSD_I8_EVAL_ROOT=$CSD_I8_RUN_ROOT/eval/$tag" \
    "$CODE_ROOT/scripts/clean_self_distill/slurm/qwen8_intermediate_eval.slurm"
}

# Six ablations, replay, dynamics, and four evaluation shards may overlap;
# 6 + 1 + 1 + 4 = the requested global maximum of twelve H100s.
trsd32_job=$(submit_eval "" trsd 32 \
  "$CSD_I8_TRSD_TRAIN_ROOT/checkpoints/episode_0032" trsd_32 4)
trsd48_job=$(submit_eval "afterok:$trsd32_job" trsd 48 \
  "$CSD_I8_TRSD_TRAIN_ROOT/checkpoints/episode_0048" trsd_48 4)

# Use the eighth slot for one P16 shard as soon as replay verification finishes
# while the two-shard TRSD chain is still active.  P32 then waits for both P16
# and TRSD48, after which its four shards plus dynamics and six ablations
# remain below the requested peak of twelve H100s.
priv16_job=$(submit_eval "afterok:$replay_job" privileged_sd 16 \
  "$CSD_I8_PRIVILEGED_REPLAY_ROOT/checkpoints/episode_0016" privileged_16 1)
priv32_job=$(submit_eval "afterok:$priv16_job:$trsd48_job" privileged_sd 32 \
  "$CSD_I8_PRIVILEGED_REPLAY_ROOT/checkpoints/episode_0032" privileged_32 4)
priv48_job=$(submit_eval "afterok:$priv32_job" privileged_sd 48 \
  "$CSD_I8_PRIVILEGED_REPLAY_ROOT/checkpoints/episode_0048" privileged_48 4)

# P16 starts at one shard while four TRSD shards occupy the other evaluation
# slots.  Once TRSD48 completes, this tiny CPU helper raises P16 to four.
priv16_raise_job=$(submit \
  --job-name=q8-raise-p16 \
  --account="$CSD_I8_ACCOUNT" \
  --partition="$CSD_I8_CPU_PARTITION" \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 \
  --dependency="afterok:$trsd48_job" \
  --output="$CSD_I8_RUN_ROOT/logs/raise_p16_%j.out" \
  --error="$CSD_I8_RUN_ROOT/logs/raise_p16_%j.err" \
  --wrap="scontrol update JobId=$priv16_job ArrayTaskThrottle=4 || true")

report_job=$(submit \
  --dependency="afterok:$priv48_job:$dynamics_job" \
  --output="$CSD_I8_RUN_ROOT/logs/report_%j.out" \
  --error="$CSD_I8_RUN_ROOT/logs/report_%j.err" \
  --export="$common_exports" \
  "$CODE_ROOT/scripts/clean_self_distill/slurm/qwen8_intermediate_report.slurm")

"$CSD_I8_PYTHON" - "$CSD_I8_RUN_ROOT/SUBMISSION.json" <<PY
import json
import sys
from pathlib import Path

payload = {
    "schema_version": "qwen3-8b-intermediate-submission-v1",
    "max_concurrent_h100": 12,
    "ablation_job": "$ablation_job",
    "ablation_report_job": "$ablation_report_job",
    "privileged_replay_job": "$replay_job",
    "common_eval_nll_job": "$dynamics_job",
    "trsd32_eval_job": "$trsd32_job",
    "trsd48_eval_job": "$trsd48_job",
    "privileged16_eval_job": "$priv16_job",
    "privileged16_raise_throttle_job": "$priv16_raise_job",
    "privileged32_eval_job": "$priv32_job",
    "privileged48_eval_job": "$priv48_job",
    "cross_model_report_job": "$report_job",
    "frozen_code_snapshot": "$CSD_I8_SNAPSHOT_ROOT",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

cp "$CSD_I8_RUN_ROOT/SUBMISSION.json" \
  "$CSD_A2_RUN_ROOT/RESUBMISSION_FROM_INTERMEDIATE_20260809.json"
printf 'ablation=%s replay=%s dynamics=%s trsd32=%s trsd48=%s p16=%s p32=%s p48=%s report=%s\n' \
  "$ablation_job" "$replay_job" "$dynamics_job" "$trsd32_job" "$trsd48_job" \
  "$priv16_job" "$priv32_job" "$priv48_job" "$report_job"
squeue -r -j "$ablation_job,$ablation_report_job,$replay_job,$dynamics_job,$trsd32_job,$trsd48_job,$priv16_job,$priv16_raise_job,$priv32_job,$priv48_job,$report_job" \
  -o '%.18i %.12P %.28j %.2t %.10M %.10l %R'
