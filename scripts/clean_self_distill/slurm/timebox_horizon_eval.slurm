#!/usr/bin/env bash
#SBATCH --job-name=csd-horizon-acc1
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --array=0-23%4
#SBATCH --requeue

# Clean/Privileged x episodes 16/32/48 x four held-out query shards.
set -Eeuo pipefail
: "${SLURM_ARRAY_TASK_ID:?submit as the embedded 0-23%4 array}"
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
export TMPDIR="$CSD_RUN_ROOT/tmp/horizon_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

CSD_EVAL_SHARDS=4
CSD_CHECKPOINTS=(16 32 48)
CSD_TASKS_PER_BRANCH=$((${#CSD_CHECKPOINTS[@]} * CSD_EVAL_SHARDS))
CSD_BRANCH_INDEX=$((SLURM_ARRAY_TASK_ID / CSD_TASKS_PER_BRANCH))
CSD_BRANCH_OFFSET=$((SLURM_ARRAY_TASK_ID % CSD_TASKS_PER_BRANCH))
CSD_CHECKPOINT_INDEX=$((CSD_BRANCH_OFFSET / CSD_EVAL_SHARDS))
CSD_SHARD_INDEX=$((CSD_BRANCH_OFFSET % CSD_EVAL_SHARDS))
CSD_CHECKPOINT_EPISODE=${CSD_CHECKPOINTS[$CSD_CHECKPOINT_INDEX]}
CSD_EPISODE_LABEL=$(printf '%04d' "$CSD_CHECKPOINT_EPISODE")
CSD_SHARD_LABEL=$(printf '%02d' "$CSD_SHARD_INDEX")

case "$CSD_BRANCH_INDEX" in
  0)
    CSD_METHOD=clean_sd
    CSD_BRANCH_DIR=clean
    ;;
  1)
    CSD_METHOD=privileged_sd
    CSD_BRANCH_DIR=privileged
    ;;
  *)
    echo "Unexpected branch index $CSD_BRANCH_INDEX" >&2
    exit 2
    ;;
esac

CSD_CHECKPOINT="$CSD_RUN_ROOT/timebox12h/$CSD_BRANCH_DIR/checkpoints/episode_${CSD_EPISODE_LABEL}"
[[ -s "$CSD_CHECKPOINT/checkpoint_manifest.json" ]]
CSD_DEST="$CSD_RUN_ROOT/timebox12h/horizon_eval"
CSD_OUTPUT="$CSD_DEST/predictions/$CSD_METHOD/episode_${CSD_EPISODE_LABEL}/shard_${CSD_SHARD_LABEL}.jsonl"
CSD_DONE="$CSD_DEST/status/${CSD_METHOD}_episode_${CSD_EPISODE_LABEL}_shard_${CSD_SHARD_LABEL}.done"
CSD_RESOURCE_LOG="$CSD_DEST/resources/${CSD_METHOD}_episode_${CSD_EPISODE_LABEL}_shard_${CSD_SHARD_LABEL}.tsv"
mkdir -p "$(dirname "$CSD_OUTPUT")" "$(dirname "$CSD_DONE")" "$(dirname "$CSD_RESOURCE_LOG")"
[[ ! -s "$CSD_DONE" ]] || exit 0

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
    --adapter "$CSD_CHECKPOINT" \
    --method "$CSD_METHOD" \
    --checkpoint-episode "$CSD_CHECKPOINT_EPISODE" \
    --sample-count 1 \
    --max-new-tokens 4096 \
    --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS" \
    --context-window "$CSD_CONTEXT_WINDOW_TOKENS" \
    --num-shards "$CSD_EVAL_SHARDS" \
    --shard-index "$CSD_SHARD_INDEX" \
    --seed "$CSD_RUN_SEED"

kill "$CSD_WATCHDOG" 2>/dev/null || true
wait "$CSD_WATCHDOG" 2>/dev/null || true
CSD_WATCHDOG=
[[ -s "$CSD_OUTPUT" ]]
printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" > "$CSD_DONE"

