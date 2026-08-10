#!/usr/bin/env bash
#SBATCH --job-name=srpo-dispatch
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=23:50:00
#SBATCH --requeue
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%j.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%j.err

set -Eeuo pipefail
: "${CSD_SRPO_CONFIG:?export CSD_SRPO_CONFIG}"
: "${CSD_SRPO_Q17_TRAIN_JOB:?export CSD_SRPO_Q17_TRAIN_JOB}"
: "${CSD_SRPO_Q8_TRAIN_JOB:?export CSD_SRPO_Q8_TRAIN_JOB}"
: "${CSD_SRPO_GO_TRAIN_JOB:?export CSD_SRPO_GO_TRAIN_JOB}"
# shellcheck disable=SC1090
source "$CSD_SRPO_CONFIG"

if [[ -s "$CSD_SRPO_RUN_ROOT/DISPATCHED_JOBS.json" ]]; then
  printf 'downstream jobs already dispatched\n'
  exit 0
fi

checkpoint_ready() {
  local model_key=$1
  [[ -s "$CSD_SRPO_RUN_ROOT/train/$model_key/checkpoints/episode_0016/checkpoint_manifest.json" ]]
}

train_failed() {
  local job_id=$1 state
  state=$(sacct -X -n -j "$job_id" --format=State --parsable2 \
    | sed '/^[[:space:]]*$/d' | head -n 1 | cut -d'|' -f1)
  [[ "$state" =~ ^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE) ]]
}

while ! checkpoint_ready q17 || ! checkpoint_ready q8 || ! checkpoint_ready go20; do
  for job_id in "$CSD_SRPO_Q17_TRAIN_JOB" "$CSD_SRPO_Q8_TRAIN_JOB" "$CSD_SRPO_GO_TRAIN_JOB"; do
    if train_failed "$job_id"; then
      printf 'training job %s failed before checkpoint 16\n' "$job_id" >&2
      exit 1
    fi
  done
  sleep 30
done

common_export="ALL,CSD_SRPO_CONFIG=$CSD_SRPO_CONFIG"

# While the three 2-GPU training jobs continue, cap checkpoint-16 evaluation
# at another 6 H100s: 2 Qwen shards per model plus one 2-GPU GPT-OSS shard.
q17_eval16=$(sbatch --parsable --array=0-3%2 \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q17" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_qwen_eval.slurm")
q8_eval16=$(sbatch --parsable --array=0-3%2 \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q8" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_qwen_eval.slurm")
go_eval16=$(sbatch --parsable --array=0-3%1 \
  --export="$common_export" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_gptoss_eval.slurm")

# Checkpoint-64 evaluation waits for all training and checkpoint-16 evaluation.
# It then expands to the full 12-H100 evaluation layout.
ready64="afterok:$CSD_SRPO_Q17_TRAIN_JOB:$CSD_SRPO_Q8_TRAIN_JOB:$CSD_SRPO_GO_TRAIN_JOB:$q17_eval16:$q8_eval16:$go_eval16"
q17_eval64=$(sbatch --parsable --array=4-7%4 --dependency="$ready64" \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q17" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_qwen_eval.slurm")
q8_eval64=$(sbatch --parsable --array=4-7%4 --dependency="$ready64" \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q8" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_qwen_eval.slurm")
go_eval64=$(sbatch --parsable --array=4-7%2 --dependency="$ready64" \
  --export="$common_export" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_gptoss_eval.slurm")

q17_score=$(sbatch --parsable --dependency="afterok:$q17_eval64" \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q17" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_score.slurm")
q8_score=$(sbatch --parsable --dependency="afterok:$q8_eval64" \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q8" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_score.slurm")
go_score=$(sbatch --parsable --dependency="afterok:$go_eval64" \
  --export="$common_export,CSD_SRPO_MODEL_KEY=go20" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_score.slurm")
final_report=$(sbatch --parsable \
  --dependency="afterok:$q17_score:$q8_score:$go_score" \
  --export="$common_export" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_final_report.slurm")

jq -n \
  --arg dispatched_at "$(date -Is)" \
  --arg q17_eval16 "$q17_eval16" --arg q8_eval16 "$q8_eval16" --arg go_eval16 "$go_eval16" \
  --arg q17_eval64 "$q17_eval64" --arg q8_eval64 "$q8_eval64" --arg go_eval64 "$go_eval64" \
  --arg q17_score "$q17_score" --arg q8_score "$q8_score" --arg go_score "$go_score" \
  --arg final_report "$final_report" \
  '{dispatched_at:$dispatched_at,
    eval16:{q17:$q17_eval16,q8:$q8_eval16,go20:$go_eval16},
    eval64:{q17:$q17_eval64,q8:$q8_eval64,go20:$go_eval64},
    score:{q17:$q17_score,q8:$q8_score,go20:$go_score},
    final_report:$final_report}' \
  > "$CSD_SRPO_RUN_ROOT/DISPATCHED_JOBS.json"
