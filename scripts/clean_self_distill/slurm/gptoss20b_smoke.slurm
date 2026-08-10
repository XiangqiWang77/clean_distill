#!/usr/bin/env bash
#SBATCH --job-name=go20-smoke
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%j-smoke.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%j-smoke.err

set -Eeuo pipefail
: "${CSD_GO_CONFIG:?export CSD_GO_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_GO_CONFIG"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_GO_CONDA_ENV"
cd "$CSD_GO_CODE_ROOT"

export PYTHONPATH="$CSD_GO_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_GO_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_GO_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/go20_smoke_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_GO_RUN_ROOT/logs" "$CSD_GO_RUN_ROOT/smoke"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100
[[ "$(sha256sum "$CSD_GO_MODEL_LOCAL_DIR/config.json" | awk '{print $1}')" == "$CSD_GO_MODEL_CONFIG_SHA256" ]]
[[ "$(sha256sum "$CSD_GO_MODEL_LOCAL_DIR/model.safetensors.index.json" | awk '{print $1}')" == "$CSD_GO_MODEL_INDEX_SHA256" ]]

SMOKE_QUERY="$CSD_GO_RUN_ROOT/smoke/distill_query.jsonl"
SMOKE_HELDOUT="$CSD_GO_RUN_ROOT/smoke/heldout_query.jsonl"
sed -n '1p' "$CSD_GO_PREPARED_16_ROOT/distill_queries.jsonl" > "$SMOKE_QUERY"
sed -n '1p' "$CSD_GO_PREPARED_16_ROOT/heldout_queries.jsonl" > "$SMOKE_HELDOUT"
[[ "$(wc -l < "$SMOKE_QUERY")" == 1 ]]
[[ "$(wc -l < "$SMOKE_HELDOUT")" == 1 ]]

for branch in privileged clean; do
  output="$CSD_GO_RUN_ROOT/smoke/$branch"
  checkpoint="$output/checkpoints/episode_0001"
  if [[ ! -s "$checkpoint/checkpoint_manifest.json" ]]; then
    resume=()
    [[ ! -s "$output/run_manifest.json" ]] || resume=(--resume)
    args=()
    if [[ "$branch" == clean ]]; then
      args=(
        --trust-region-kl-budget "$CSD_GO_TRUST_REGION_KL_BUDGET"
        --trust-region-binary-search-steps "$CSD_GO_TRUST_REGION_BINARY_SEARCH_STEPS"
      )
    fi
    "$CSD_GO_PYTHON" scripts/clean_self_distill/04_persistent_train.py \
      --branch "$branch" \
      --queries "$SMOKE_QUERY" \
      --model "$CSD_GO_MODEL_LOCAL_DIR" \
      --model-id "$CSD_GO_MODEL_ID" \
      --revision "$CSD_GO_MODEL_REVISION" \
      --output-dir "$output" \
      --episodes 1 \
      --scientific-checkpoints 0,1 \
      --rolling-checkpoint-interval 1 \
      --max-sequence-tokens 2048 \
      --max-rollout-tokens 64 \
      --learning-rate "$CSD_GO_LEARNING_RATE" \
      --lora-rank "$CSD_GO_LORA_RANK" \
      --lora-alpha "$CSD_GO_LORA_ALPHA" \
      --seed "$CSD_GO_TRAIN_SEED" \
      --distill-token-chunk-size 32 \
      "${args[@]}" \
      "${resume[@]}"
  fi
  [[ -s "$checkpoint/checkpoint_manifest.json" ]]
done

SMOKE_PREDICTIONS="$CSD_GO_RUN_ROOT/smoke/vllm_adapter_predictions.jsonl"
if [[ ! -s "$SMOKE_PREDICTIONS" ]]; then
  "$CSD_GO_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$SMOKE_HELDOUT" \
    --output "$SMOKE_PREDICTIONS" \
    --model "$CSD_GO_MODEL_LOCAL_DIR" \
    --model-id "$CSD_GO_MODEL_ID" \
    --revision "$CSD_GO_MODEL_REVISION" \
    --method trsd \
    --checkpoint-episode 1 \
    --adapter "$CSD_GO_RUN_ROOT/smoke/clean/checkpoints/episode_0001" \
    --engine vllm \
    --batch-size 1 \
    --gpu-memory-utilization 0.88 \
    --max-lora-rank "$CSD_GO_LORA_RANK" \
    --max-new-tokens 32 \
    --max-prompt-tokens 2048 \
    --context-window 4096 \
    --num-shards 1 \
    --shard-index 0 \
    --sample-count 1 \
    --seed "$CSD_GO_EVAL_SEED"
fi
[[ "$(wc -l < "$SMOKE_PREDICTIONS")" == 1 ]]
printf 'complete\njob_id=%s\nmodel=%s\n' "$SLURM_JOB_ID" "$CSD_GO_MODEL_ID" > "$CSD_GO_RUN_ROOT/SMOKE_COMPLETE"
