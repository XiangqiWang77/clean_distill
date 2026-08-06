#!/usr/bin/env bash
#SBATCH --job-name=proposed-priv-main
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --array=0-3%4
#SBATCH --requeue

# Four independent H100 shards for the final Self-Proposed Privileged-SD
# checkpoint.  Outputs are isolated from the archived fixed-prompt baseline.
set -Eeuo pipefail
: "${SLURM_ARRAY_TASK_ID:?submit as the embedded 0-3%4 array}"
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
export TMPDIR="$CSD_RUN_ROOT/tmp/proposed_privileged_main_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

CSD_EVAL_SHARDS=4
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < CSD_EVAL_SHARDS )) || {
  echo "Unexpected array task id $SLURM_ARRAY_TASK_ID (expected 0-3)" >&2
  exit 2
}
CSD_SHARD_INDEX=$SLURM_ARRAY_TASK_ID
CSD_SHARD_LABEL=$(printf '%02d' "$CSD_SHARD_INDEX")
CSD_CHECKPOINT_EPISODE=64
CSD_CHECKPOINT="$CSD_RUN_ROOT/timebox12h/proposed_privileged/checkpoints/episode_0064"
[[ -s "$CSD_CHECKPOINT/checkpoint_manifest.json" ]]
CSD_DEST="$CSD_RUN_ROOT/timebox12h/proposed_privileged_main_eval"
CSD_OUTPUT="$CSD_DEST/predictions/proposed_privileged_sd/shard_${CSD_SHARD_LABEL}.jsonl"
CSD_DONE="$CSD_DEST/status/proposed_privileged_sd_shard_${CSD_SHARD_LABEL}.done"
CSD_RESOURCE_LOG="$CSD_DEST/resources/proposed_privileged_sd_shard_${CSD_SHARD_LABEL}.tsv"
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
    --method proposed_privileged_sd \
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
