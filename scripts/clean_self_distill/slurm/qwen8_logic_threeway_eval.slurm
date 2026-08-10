#!/usr/bin/env bash
#SBATCH --job-name=q8-logic3
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=11:50:00
#SBATCH --array=0-1%2
#SBATCH --requeue
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-logic-threeway-10k-20260808/logs/%A_%a.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-logic-threeway-10k-20260808/logs/%A_%a.err

set -Eeuo pipefail
: "${CSD_LOGIC_CONFIG:?export CSD_LOGIC_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_LOGIC_CONFIG"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_LOGIC_CONDA_ENV"
cd "$CSD_LOGIC_CODE_ROOT"

export PYTHONPATH="$CSD_LOGIC_SATQUEST_DEPS:$CSD_LOGIC_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_LOGIC_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_LOGIC_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CSD_LOGIC_TMP="/tmp/q8l_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TMPDIR="$CSD_LOGIC_TMP"

mkdir -p "$CSD_LOGIC_TMP" "$HF_HUB_CACHE" "$CSD_LOGIC_RUN_ROOT/logs"
[[ "$CSD_LOGIC_NUM_SHARDS" == 2 ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[01]$ ]]
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100
[[ -s "$CSD_LOGIC_MODEL_LOCAL_DIR/model.safetensors.index.json" ]]
[[ -s "$CSD_LOGIC_DATA_MANIFEST" ]]
[[ "$(sha256sum "$CSD_LOGIC_DATA_ROOT/satquest/val.parquet" | awk '{print $1}')" == "$CSD_LOGIC_SATQUEST_EVAL_SHA256" ]]
[[ "$(sha256sum "$CSD_LOGIC_DATA_ROOT/logicskills/val.parquet" | awk '{print $1}')" == "$CSD_LOGIC_LOGICSKILLS_EVAL_SHA256" ]]

run_method() {
  local method=$1
  local adapter=${2:-}
  local prediction_dir="$CSD_LOGIC_RUN_ROOT/predictions/$method"
  local scored_dir="$CSD_LOGIC_RUN_ROOT/scored/$method"
  local predictions="$prediction_dir/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").jsonl"
  local scored="$scored_dir/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").jsonl"
  mkdir -p "$prediction_dir" "$scored_dir"
  local adapter_args=()
  [[ -z "$adapter" ]] || adapter_args=(--adapter "$adapter")
  "$CSD_LOGIC_PYTHON" scripts/clean_self_distill/15_logic_eval.py generate \
    --data-root "$CSD_LOGIC_DATA_ROOT" \
    --output "$predictions" \
    --model "$CSD_LOGIC_MODEL_LOCAL_DIR" \
    --model-id "$CSD_LOGIC_MODEL_ID" \
    --revision "$CSD_LOGIC_MODEL_REVISION" \
    --method "$method" \
    --num-shards "$CSD_LOGIC_NUM_SHARDS" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --batch-size "$CSD_LOGIC_BATCH_SIZE" \
    --max-new-tokens "$CSD_LOGIC_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$CSD_LOGIC_MAX_PROMPT_TOKENS" \
    "${adapter_args[@]}"
  [[ "$(wc -l < "$predictions")" == 2430 ]]
  "$CSD_LOGIC_PYTHON" scripts/clean_self_distill/15_logic_eval.py score \
    --data-root "$CSD_LOGIC_DATA_ROOT" \
    --predictions "$predictions" \
    --output "$scored" \
    --satquest-repo "$CSD_LOGIC_SATQUEST_REPO" \
    --satquest-deps "$CSD_LOGIC_SATQUEST_DEPS" \
    --logicskills-repo "$CSD_LOGIC_LOGICSKILLS_REPO" \
    --num-shards "$CSD_LOGIC_NUM_SHARDS" \
    --shard-index "$SLURM_ARRAY_TASK_ID"
  [[ "$(wc -l < "$scored")" == 2430 ]]
}

echo "logic shard=$SLURM_ARRAY_TASK_ID host=$(hostname) methods=base,privileged_sd_64,trsd_64"
run_method base
run_method privileged_sd_64 "$CSD_LOGIC_PRIVILEGED64_ADAPTER"
run_method trsd_64 "$CSD_LOGIC_TRSD64_ADAPTER"

printf 'complete\njob_id=%s\narray_task_id=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" \
  > "$CSD_LOGIC_RUN_ROOT/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").TASK_COMPLETE"

"$CSD_LOGIC_PYTHON" scripts/clean_self_distill/15_logic_eval.py report \
  --run-root "$CSD_LOGIC_RUN_ROOT" \
  --num-shards "$CSD_LOGIC_NUM_SHARDS"
