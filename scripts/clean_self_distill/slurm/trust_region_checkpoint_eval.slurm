#!/usr/bin/env bash
#SBATCH --job-name=trsd-main-table
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-19%4
#SBATCH --requeue

# Five checkpoints (Base, TRSD-16/32/36, Privileged-64) x four query shards.
# Every task owns one checkpoint/shard pair, so at most four H100s are active.
# Prediction files are atomic and resumable; the final task also scores all
# completed shards, without a dependent job or polling process.
set -Eeuo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit as a 0-19 array}"
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"

export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$CSD_RUN_ROOT/tmp/trsd_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

CSD_EVAL_SHARDS=4
CSD_CHECKPOINTS=(0 16 32 36 64)
CSD_CHECKPOINT_INDEX=$((SLURM_ARRAY_TASK_ID / CSD_EVAL_SHARDS))
CSD_SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % CSD_EVAL_SHARDS))
CSD_CHECKPOINT_EPISODE=${CSD_CHECKPOINTS[$CSD_CHECKPOINT_INDEX]}
CSD_EPISODE_LABEL=$(printf '%04d' "$CSD_CHECKPOINT_EPISODE")
CSD_SHARD_LABEL=$(printf '%02d' "$CSD_SHARD_INDEX")
CSD_EVAL_MAX_TOKENS=${CSD_TRSD_EVAL_MAX_NEW_TOKENS:-10240}
CSD_DEST="$CSD_TIMEBOX_OUTPUT/trust_region_eval_${CSD_EVAL_MAX_TOKENS}"

CSD_METHOD=base
CSD_ADAPTER_ARGS=()
if (( CSD_CHECKPOINT_EPISODE > 0 )); then
  if (( CSD_CHECKPOINT_EPISODE == 64 )); then
    CSD_METHOD=privileged_sd
    CSD_CHECKPOINT=${CSD_PRIVILEGED_CHECKPOINT:-/home/da839/scratch_pi_mg269/da839/clean_distill/runs/csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/checkpoints/episode_0064}
  elif (( CSD_CHECKPOINT_EPISODE == 36 )); then
    CSD_METHOD=trsd
    CSD_CHECKPOINT="$CSD_TIMEBOX_OUTPUT/checkpoints/rolling_episode_${CSD_EPISODE_LABEL}"
  else
    CSD_METHOD=trsd
    CSD_CHECKPOINT="$CSD_TIMEBOX_OUTPUT/checkpoints/episode_${CSD_EPISODE_LABEL}"
  fi
  [[ -s "$CSD_CHECKPOINT/checkpoint_manifest.json" ]]
  CSD_ADAPTER_ARGS=(--adapter "$CSD_CHECKPOINT")
fi

CSD_OUTPUT="$CSD_DEST/predictions/episode_${CSD_EPISODE_LABEL}/shard_${CSD_SHARD_LABEL}.jsonl"
CSD_DONE="$CSD_DEST/status/episode_${CSD_EPISODE_LABEL}_shard_${CSD_SHARD_LABEL}.done"
CSD_RESOURCE_LOG="$CSD_DEST/resources/episode_${CSD_EPISODE_LABEL}_shard_${CSD_SHARD_LABEL}.tsv"
mkdir -p "$(dirname "$CSD_OUTPUT")" "$(dirname "$CSD_DONE")" "$(dirname "$CSD_RESOURCE_LOG")" "$CSD_DEST/scored"

if [[ ! -s "$CSD_DONE" ]]; then
  # Requeue with ten minutes left. The generator resumes from its atomically
  # committed query prefix, so an interrupted task does not repeat examples.
  ( sleep 20400; scontrol requeue "$SLURM_JOB_ID" ) &
  CSD_WATCHDOG=$!
  trap 'kill "$CSD_WATCHDOG" 2>/dev/null || true' EXIT

  printf 'attempt_start\t%s\t%s\n' "$SLURM_JOB_ID" "$(date -Is)" >> "$CSD_RESOURCE_LOG"
  /usr/bin/time -a -o "$CSD_RESOURCE_LOG" \
    -f 'process\telapsed_seconds=%e\tmax_rss_kib=%M' \
    "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
      --queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
      --output "$CSD_OUTPUT" \
      --model "$CSD_MODEL_LOCAL_DIR" \
      --model-id "$CSD_MODEL_ID" \
      --revision "$CSD_MODEL_REVISION" \
      --method "$CSD_METHOD" \
      --checkpoint-episode "$CSD_CHECKPOINT_EPISODE" \
      "${CSD_ADAPTER_ARGS[@]}" \
      --sample-count 1 \
      --max-new-tokens "$CSD_EVAL_MAX_TOKENS" \
      --attn-implementation sdpa \
      --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS" \
      --context-window "$CSD_CONTEXT_WINDOW_TOKENS" \
      --num-shards "$CSD_EVAL_SHARDS" \
      --shard-index "$CSD_SHARD_INDEX" \
      --seed "$CSD_RUN_SEED"

  kill "$CSD_WATCHDOG" 2>/dev/null || true
  wait "$CSD_WATCHDOG" 2>/dev/null || true
  trap - EXIT
  printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" > "$CSD_DONE"
fi

# Whichever task completes the array last performs the cheap offline scoring.
# flock prevents duplicate scoring when the final markers arrive together.
if [[ "$(find "$CSD_DEST/status" -maxdepth 1 -type f -name '*.done' | wc -l)" == 20 ]]; then
  (
    flock -n 9 || exit 0
    for CSD_SCORE_EPISODE in "${CSD_CHECKPOINTS[@]}"; do
      CSD_SCORE_LABEL=$(printf '%04d' "$CSD_SCORE_EPISODE")
      CSD_PREDICTION_ARGS=()
      for CSD_SCORE_SHARD in 0 1 2 3; do
        CSD_SCORE_SHARD_LABEL=$(printf '%02d' "$CSD_SCORE_SHARD")
        CSD_PREDICTION_ARGS+=(--predictions "$CSD_DEST/predictions/episode_${CSD_SCORE_LABEL}/shard_${CSD_SCORE_SHARD_LABEL}.jsonl")
      done
      "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
        "${CSD_PREDICTION_ARGS[@]}" \
        --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
        --sample-count 1 \
        --output "$CSD_DEST/scored/episode_${CSD_SCORE_LABEL}.jsonl"
    done
    printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" > "$CSD_DEST/status/scoring.done"
  ) 9>"$CSD_DEST/status/scoring.lock"
fi
