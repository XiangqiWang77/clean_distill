#!/usr/bin/env bash
#SBATCH --job-name=trsd-train64
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_TRAIN64_ROOT:?export CSD_TRAIN64_ROOT}"
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
export TMPDIR="$CSD_TRAIN64_ROOT/tmp_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$CSD_TRAIN64_ROOT/train"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

# Bind the run to the first 64 rows of the preregistered, answer-free stream.
# This keeps the H100 job self-contained instead of depending on report code.
sed -n '1,64p' "$CSD_PREPARED_ROOT/distill_queries.jsonl" \
  > "$CSD_TRAIN64_ROOT/distill_queries_64.jsonl"
[[ "$(wc -l < "$CSD_TRAIN64_ROOT/distill_queries_64.jsonl")" == 64 ]]

CSD_RESUME=()
[[ ! -s "$CSD_TRAIN64_ROOT/train/run_manifest.json" ]] || CSD_RESUME+=(--resume)
"$CSD_TTT_PYTHON" scripts/clean_self_distill/04_persistent_train.py \
  --branch clean \
  --queries "$CSD_TRAIN64_ROOT/distill_queries_64.jsonl" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --output-dir "$CSD_TRAIN64_ROOT/train" \
  --episodes 64 \
  --scientific-checkpoints 0,16,32,48,64 \
  --rolling-checkpoint-interval 8 \
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

printf 'completed_at=%s\njob_id=%s\nbranch=%s\nepisodes=64\nepsilon=0.004\n' \
  "$(date -Is)" "$SLURM_JOB_ID" clean \
  > "$CSD_TRAIN64_ROOT/TRAIN_COMPLETE"
