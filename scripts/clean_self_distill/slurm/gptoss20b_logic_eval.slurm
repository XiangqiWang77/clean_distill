#!/usr/bin/env bash
#SBATCH --job-name=go20-logic
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=240G
#SBATCH --time=11:50:00
#SBATCH --array=0-7%3
#SBATCH --requeue
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%A_%a-logic.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%A_%a-logic.err

set -Eeuo pipefail
: "${CSD_GO_CONFIG:?export CSD_GO_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_GO_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_GO_CONDA_ENV"
cd "$CSD_GO_CODE_ROOT"
export PYTHONPATH="$CSD_GO_SATQUEST_DEPS:$CSD_GO_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_GO_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_GO_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/go20_logic_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_GO_RUN_ROOT/logs"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 2 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100
[[ "$CSD_GO_LOGIC_NUM_SHARDS" == 8 ]]
[[ "$(sha256sum "$CSD_GO_LOGIC_DATA_ROOT/satquest/val.parquet" | awk '{print $1}')" == "$CSD_GO_SATQUEST_EVAL_SHA256" ]]
[[ "$(sha256sum "$CSD_GO_LOGIC_DATA_ROOT/logicskills/val.parquet" | awk '{print $1}')" == "$CSD_GO_LOGICSKILLS_EVAL_SHA256" ]]

watchdog_pid=
cleanup() {
  if [[ -n "$watchdog_pid" ]]; then
    kill "$watchdog_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
(
  sleep 39000
  scontrol requeue "$SLURM_JOB_ID"
) &
watchdog_pid=$!

expected_rows=$(( (4860 + CSD_GO_LOGIC_NUM_SHARDS - 1 - SLURM_ARRAY_TASK_ID) / CSD_GO_LOGIC_NUM_SHARDS ))
run_method() {
  local method=$1
  local adapter=${2:-}
  local predictions="$CSD_GO_RUN_ROOT/logic/predictions/$method/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").jsonl"
  local scored="$CSD_GO_RUN_ROOT/logic/scored/$method/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").jsonl"
  mkdir -p "$(dirname "$predictions")" "$(dirname "$scored")"
  if [[ -s "$scored" ]] && [[ "$(wc -l < "$scored")" == "$expected_rows" ]]; then
    return
  fi
  adapter_args=()
  [[ -z "$adapter" ]] || adapter_args=(--adapter "$adapter")
  "$CSD_GO_PYTHON" scripts/clean_self_distill/15_logic_eval.py generate \
    --data-root "$CSD_GO_LOGIC_DATA_ROOT" \
    --output "$predictions" \
    --model "$CSD_GO_MODEL_LOCAL_DIR" \
    --model-id "$CSD_GO_MODEL_ID" \
    --revision "$CSD_GO_MODEL_REVISION" \
    --method "$method" \
    --num-shards "$CSD_GO_LOGIC_NUM_SHARDS" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --batch-size "$CSD_GO_LOGIC_BATCH_SIZE" \
    --max-new-tokens "$CSD_GO_LOGIC_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$CSD_GO_LOGIC_MAX_PROMPT_TOKENS" \
    --tensor-parallel-size 2 \
    "${adapter_args[@]}"
  [[ "$(wc -l < "$predictions")" == "$expected_rows" ]]
  "$CSD_GO_PYTHON" scripts/clean_self_distill/15_logic_eval.py score \
    --data-root "$CSD_GO_LOGIC_DATA_ROOT" \
    --predictions "$predictions" \
    --output "$scored" \
    --satquest-repo "$CSD_GO_SATQUEST_REPO" \
    --satquest-deps "$CSD_GO_SATQUEST_DEPS" \
    --logicskills-repo "$CSD_GO_LOGICSKILLS_REPO" \
    --num-shards "$CSD_GO_LOGIC_NUM_SHARDS" \
    --shard-index "$SLURM_ARRAY_TASK_ID"
  [[ "$(wc -l < "$scored")" == "$expected_rows" ]]
}

run_method base
run_method privileged_sd_16 "$CSD_GO_RUN_ROOT/train/privileged_16/checkpoints/episode_0016"
run_method trsd_16 "$CSD_GO_RUN_ROOT/train/trsd_16/checkpoints/episode_0016"
run_method privileged_sd_64 "$CSD_GO_RUN_ROOT/train/privileged_64/checkpoints/episode_0064"
run_method trsd_64 "$CSD_GO_RUN_ROOT/train/trsd_64/checkpoints/episode_0064"

"$CSD_GO_PYTHON" scripts/clean_self_distill/15_logic_eval.py report \
  --run-root "$CSD_GO_RUN_ROOT/logic" \
  --num-shards "$CSD_GO_LOGIC_NUM_SHARDS" \
  --methods base,privileged_sd_16,trsd_16,privileged_sd_64,trsd_64 \
  --model-label "$CSD_GO_MODEL_LABEL"
printf 'complete\njob_id=%s\nshard=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" \
  > "$CSD_GO_RUN_ROOT/logic/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").TASK_COMPLETE"
