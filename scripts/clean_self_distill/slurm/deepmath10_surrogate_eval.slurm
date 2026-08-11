#!/usr/bin/env bash
#SBATCH --job-name=d10-surrogate
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=192G
#SBATCH --time=11:50:00
#SBATCH --array=0-1%2
#SBATCH --requeue

set -Eeuo pipefail
: "${CSD_D10_CONFIG:?export CSD_D10_CONFIG}"
: "${CSD_D10_EXEC_ROOT:?export CSD_D10_EXEC_ROOT}"
source "$CSD_D10_CONFIG"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_D10_CONDA_ENV"
cd "$CSD_D10_EXEC_ROOT"

export PYTHONPATH="$CSD_D10_TORCH_OVERLAY:$CSD_D10_EXEC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_D10_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_D10_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/d10_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_D10_RUN_ROOT/evidence" "$CSD_D10_RUN_ROOT/status"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100
[[ "$(sha256sum "$CSD_D10_DEEPMATH" | awk '{print $1}')" == "$CSD_D10_DEEPMATH_SHA256" ]]
[[ "$(sha256sum "$CSD_D10_MANIFEST" | awk '{print $1}')" == "$CSD_D10_MANIFEST_SHA256" ]]
[[ "$(jq -er '.total' "$CSD_D10_MANIFEST")" == "$CSD_D10_EXPECTED_ROWS" ]]
[[ -s "$CSD_D10_MODEL_LOCAL_DIR/config.json" ]]

watchdog_pid=
cleanup() {
  if [[ -n "$watchdog_pid" ]]; then
    kill "$watchdog_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
(
  sleep 41400
  scontrol requeue "$SLURM_JOB_ID"
) &
watchdog_pid=$!

output="$CSD_D10_RUN_ROOT/evidence/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").jsonl"
"$CSD_D10_PYTHON" scripts/clean_self_distill/33_deepmath10_surrogate_study.py \
  --deepmath "$CSD_D10_DEEPMATH" \
  --manifest "$CSD_D10_MANIFEST" \
  --manifest-sha256 "$CSD_D10_MANIFEST_SHA256" \
  --output "$output" \
  --model "$CSD_D10_MODEL_LOCAL_DIR" \
  --model-id "$CSD_D10_MODEL_ID" \
  --revision "$CSD_D10_MODEL_REVISION" \
  --num-shards "$CSD_D10_NUM_SHARDS" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --reference-token-cap "$CSD_D10_REFERENCE_TOKEN_CAP" \
  --max-prompt-tokens "$CSD_D10_MAX_PROMPT_TOKENS" \
  --context-window "$CSD_D10_CONTEXT_WINDOW" \
  --epsilon "$CSD_D10_PRIMARY_EPSILON" \
  --binary-search-steps "$CSD_D10_BINARY_SEARCH_STEPS" \
  --full-vocab-chunk-size "$CSD_D10_FULL_VOCAB_CHUNK"

expected=$(( (CSD_D10_EXPECTED_ROWS + CSD_D10_NUM_SHARDS - 1 - SLURM_ARRAY_TASK_ID) / CSD_D10_NUM_SHARDS ))
[[ "$(wc -l < "$output")" == "$expected" ]]
printf 'complete\njob_id=%s\nshard=%s\nrows=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$expected" \
  > "$CSD_D10_RUN_ROOT/status/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").complete"
