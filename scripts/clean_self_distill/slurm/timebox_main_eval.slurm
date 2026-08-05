#!/usr/bin/env bash
#SBATCH --job-name=csd-main-acc1
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --array=0-15%4
#SBATCH --requeue

# Standalone, restartable 12-hour main-table evaluation.  The 16 array tasks
# are four methods x four query shards, with at most four H100s active.
set -Eeuo pipefail
: "${SLURM_ARRAY_TASK_ID:?submit as the embedded 0-15%4 array}"
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
export TMPDIR="$CSD_RUN_ROOT/tmp/main_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

CSD_EVAL_SHARDS=4
CSD_METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / CSD_EVAL_SHARDS))
CSD_SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % CSD_EVAL_SHARDS))
CSD_SHARD_LABEL=$(printf '%02d' "$CSD_SHARD_INDEX")
CSD_CHECKPOINT_EPISODE=0
CSD_EXTRA_ARGS=()
case "$CSD_METHOD_INDEX" in
  0)
    CSD_METHOD=base
    ;;
  1)
    CSD_METHOD=csd_t
    CSD_PROPOSALS="$CSD_RUN_ROOT/timebox12h/heldout_probes/merged.jsonl"
    [[ -s "$CSD_PROPOSALS" ]]
    CSD_EXTRA_ARGS+=(--proposals "$CSD_PROPOSALS")
    ;;
  2)
    CSD_METHOD=clean_sd
    CSD_CHECKPOINT_EPISODE=64
    CSD_CHECKPOINT="$CSD_RUN_ROOT/timebox12h/clean/checkpoints/episode_0064"
    [[ -s "$CSD_CHECKPOINT/checkpoint_manifest.json" ]]
    CSD_EXTRA_ARGS+=(--adapter "$CSD_CHECKPOINT")
    ;;
  3)
    CSD_METHOD=privileged_sd
    CSD_CHECKPOINT_EPISODE=64
    CSD_CHECKPOINT="$CSD_RUN_ROOT/timebox12h/privileged/checkpoints/episode_0064"
    [[ -s "$CSD_CHECKPOINT/checkpoint_manifest.json" ]]
    CSD_EXTRA_ARGS+=(--adapter "$CSD_CHECKPOINT")
    ;;
  *)
    echo "Unexpected method index $CSD_METHOD_INDEX" >&2
    exit 2
    ;;
esac

CSD_DEST="$CSD_RUN_ROOT/timebox12h/main_eval"
CSD_OUTPUT="$CSD_DEST/predictions/$CSD_METHOD/shard_${CSD_SHARD_LABEL}.jsonl"
CSD_DONE="$CSD_DEST/status/${CSD_METHOD}_shard_${CSD_SHARD_LABEL}.done"
CSD_RESOURCE_LOG="$CSD_DEST/resources/${CSD_METHOD}_shard_${CSD_SHARD_LABEL}.tsv"
mkdir -p "$(dirname "$CSD_OUTPUT")" "$(dirname "$CSD_DONE")" "$(dirname "$CSD_RESOURCE_LOG")"
[[ ! -s "$CSD_DONE" ]] || exit 0

# Each completed response is atomically committed.  Requeue ten minutes before
# walltime and continue from the exact prediction prefix on the next allocation.
( sleep 10200; scontrol requeue "$SLURM_JOB_ID" ) &
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
    --sample-count 1 \
    --max-new-tokens 4096 \
    --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS" \
    --context-window "$CSD_CONTEXT_WINDOW_TOKENS" \
    --num-shards "$CSD_EVAL_SHARDS" \
    --shard-index "$CSD_SHARD_INDEX" \
    --seed "$CSD_RUN_SEED" \
    --ridge-lambda "$CSD_RIDGE_LAMBDA" \
    --residual-step-size "$CSD_RESIDUAL_STEP_SIZE" \
    --max-tokens-per-candidate "$CSD_MAX_TOKENS_PER_CANDIDATE" \
    --max-support-tokens "$CSD_MAX_SUPPORT_TOKENS" \
    --hard-negatives "$CSD_HARD_NEGATIVES" \
    --max-sequence-tokens "$CSD_TRAIN_MAX_SEQUENCE_TOKENS" \
    --reasoning-token-weight "$CSD_REASONING_TOKEN_WEIGHT" \
    --answer-token-weight "$CSD_ANSWER_TOKEN_WEIGHT" \
    --frontier-positive-weight "$CSD_FRONTIER_POSITIVE_WEIGHT" \
    --frontier-negative-weight "$CSD_FRONTIER_NEGATIVE_WEIGHT" \
    --frontier-max-tokens "$CSD_FRONTIER_MAX_TOKENS" \
    --frontier-negative-probability-floor "$CSD_FRONTIER_NEGATIVE_PROBABILITY_FLOOR" \
    --frontier-target-margin "$CSD_FRONTIER_TARGET_MARGIN" \
    --max-update-norm "$CSD_MAX_UPDATE_NORM" \
    "${CSD_EXTRA_ARGS[@]}"

kill "$CSD_WATCHDOG" 2>/dev/null || true
wait "$CSD_WATCHDOG" 2>/dev/null || true
CSD_WATCHDOG=
[[ -s "$CSD_OUTPUT" ]]
printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" > "$CSD_DONE"
