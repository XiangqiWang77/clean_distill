#!/usr/bin/env bash
#SBATCH --job-name=trsd-long16
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_LONG_ROOT:?export CSD_LONG_ROOT}"
: "${CSD_SHORT_ROOT:?export CSD_SHORT_ROOT with shared mechanism traces}"
: "${CSD_PREPARED_ROOT:?export CSD_PREPARED_ROOT}"

# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"

export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_SCRATCH_ROOT/hf" HF_HUB_CACHE="$CSD_SCRATCH_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$CSD_LONG_ROOT/tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$CSD_LONG_ROOT/train" "$CSD_LONG_ROOT/eval" "$CSD_LONG_ROOT/heldout"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

"$CSD_TTT_PYTHON" scripts/clean_self_distill/08_short_empirical_report.py prepare-stream \
  --queries "$CSD_PREPARED_ROOT/distill_queries.jsonl" \
  --episodes 16 \
  --output "$CSD_LONG_ROOT/distill_queries_16.jsonl"

"$CSD_TTT_PYTHON" scripts/clean_self_distill/08_short_empirical_report.py prepare-subset \
  --queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
  --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
  --out-dir "$CSD_LONG_ROOT/heldout" \
  --query-name queries.jsonl \
  --label-name labels.jsonl \
  --per-source 6

CSD_RESUME=()
[[ ! -s "$CSD_LONG_ROOT/train/run_manifest.json" ]] || CSD_RESUME+=(--resume)
"$CSD_TTT_PYTHON" scripts/clean_self_distill/04_persistent_train.py \
  --branch clean \
  --queries "$CSD_LONG_ROOT/distill_queries_16.jsonl" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --output-dir "$CSD_LONG_ROOT/train" \
  --episodes 16 \
  --scientific-checkpoints 0,4,8,12,16 \
  --rolling-checkpoint-interval 4 \
  --max-sequence-tokens 10240 \
  --max-rollout-tokens 10240 \
  --learning-rate "$CSD_LEARNING_RATE" \
  --lora-rank "$CSD_LORA_RANK" \
  --lora-alpha "$CSD_LORA_ALPHA" \
  --seed "$CSD_RUN_SEED" \
  --distill-token-chunk-size "$CSD_DISTILL_TOKEN_CHUNK_SIZE" \
  --trust-region-kl-budget 0.004 \
  --trust-region-binary-search-steps "$CSD_TRUST_REGION_BINARY_SEARCH_STEPS" \
  --attn-implementation sdpa \
  "${CSD_RESUME[@]}"

if [[ "${CSD_TRAIN_ONLY:-0}" == 1 ]]; then
  printf 'completed_at=%s\njob_id=%s\nmode=train_only\n' \
    "$(date -Is)" "$SLURM_JOB_ID" > "$CSD_LONG_ROOT/TRAIN_COMPLETE"
  exit 0
fi

for CSD_EPISODE in 0 4 8 12 16; do
  CSD_LABEL=$(printf '%04d' "$CSD_EPISODE")
  CSD_PREDICTIONS="$CSD_LONG_ROOT/eval/episode_${CSD_LABEL}.jsonl"
  CSD_SCORED="$CSD_LONG_ROOT/eval/episode_${CSD_LABEL}.scored.jsonl"
  CSD_METHOD=base
  CSD_ADAPTER_ARGS=()
  if (( CSD_EPISODE > 0 )); then
    CSD_METHOD=trsd
    CSD_ADAPTER_ARGS=(--adapter "$CSD_LONG_ROOT/train/checkpoints/episode_${CSD_LABEL}")
  fi
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
    --queries "$CSD_LONG_ROOT/heldout/queries.jsonl" \
    --output "$CSD_PREDICTIONS" \
    --model "$CSD_MODEL_LOCAL_DIR" \
    --model-id "$CSD_MODEL_ID" \
    --revision "$CSD_MODEL_REVISION" \
    --method "$CSD_METHOD" \
    --checkpoint-episode "$CSD_EPISODE" \
    "${CSD_ADAPTER_ARGS[@]}" \
    --sample-count 1 \
    --max-new-tokens 10240 \
    --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS" \
    --context-window "$CSD_CONTEXT_WINDOW_TOKENS" \
    --attn-implementation sdpa \
    --seed "$CSD_RUN_SEED"
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    --predictions "$CSD_PREDICTIONS" \
    --labels "$CSD_LONG_ROOT/heldout/labels.jsonl" \
    --sample-count 1 \
    --output "$CSD_SCORED"
done

# Mechanism traces are collected by the parallel H100 array.  If they are not
# complete yet, the scientific curve remains safely scored and a CPU report job
# can render it later; this GPU job never polls.
printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" > "$CSD_LONG_ROOT/LONG_RUN_COMPLETE"
