#!/usr/bin/env bash
#SBATCH --job-name=trsd-mechanism
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=192G
#SBATCH --time=06:00:00

# One real, label-free development query.  This job only collects model/logit
# evidence; the dependent high-memory CPU job renders publication artifacts.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_TRSD_FINAL_CHECKPOINT:?export the adopted rolling-36 or completed final TRSD checkpoint}"
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
export TMPDIR="$CSD_RUN_ROOT/tmp/trsd_mechanism_${SLURM_JOB_ID}"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

CSD_MECHANISM_ROOT=${CSD_MECHANISM_ROOT:-$CSD_RUN_ROOT/trust_region_mechanism}
# Index 22 is the preregistered historical training episode (the residue of
# z^3 exp(1/z)); the physically separate sealed labels are not passed here.
CSD_MECHANISM_QUERIES=${CSD_MECHANISM_QUERIES:-$CSD_PREPARED_ROOT/distill_queries.jsonl}
CSD_MECHANISM_QUERY_INDEX=${CSD_MECHANISM_QUERY_INDEX:-22}
CSD_MECHANISM_MAX_NEW_TOKENS=${CSD_MECHANISM_MAX_NEW_TOKENS:-6144}
CSD_MECHANISM_OUTPUT="$CSD_MECHANISM_ROOT/raw/query_$(printf '%04d' "$CSD_MECHANISM_QUERY_INDEX").jsonl"
CSD_MECHANISM_DONE="$CSD_MECHANISM_ROOT/status/collect_query_$(printf '%04d' "$CSD_MECHANISM_QUERY_INDEX").done"
mkdir -p "$(dirname "$CSD_MECHANISM_OUTPUT")" "$(dirname "$CSD_MECHANISM_DONE")"

if [[ -s "$CSD_MECHANISM_DONE" ]]; then
  [[ -s "$CSD_MECHANISM_OUTPUT" ]]
  exit 0
fi
[[ -s "$CSD_MECHANISM_QUERIES" ]]
[[ -s "$CSD_TRSD_FINAL_CHECKPOINT/checkpoint_manifest.json" ]]

"$CSD_TTT_PYTHON" scripts/clean_self_distill/06_trust_region_mechanism.py \
  --queries "$CSD_MECHANISM_QUERIES" \
  --adapter "$CSD_TRSD_FINAL_CHECKPOINT" \
  --output "$CSD_MECHANISM_OUTPUT" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --query-index "$CSD_MECHANISM_QUERY_INDEX" \
  --max-new-tokens "$CSD_MECHANISM_MAX_NEW_TOKENS" \
  --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS" \
  --context-window "$CSD_CONTEXT_WINDOW_TOKENS" \
  --alpha-grid "0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0" \
  --epsilon-grid "0.001,0.002,0.004,0.008,0.016,0.032,0.08" \
  --selected-epsilon 0.008 \
  --training-epsilon "$CSD_TRUST_REGION_KL_BUDGET" \
  --binary-search-steps 8 \
  --full-vocab-chunk-size "$CSD_DISTILL_TOKEN_CHUNK_SIZE" \
  --seed "$CSD_RUN_SEED" \
  --attn-implementation sdpa

[[ -s "$CSD_MECHANISM_OUTPUT" ]]
CSD_MARKER_TMP="${CSD_MECHANISM_DONE}.tmp.$$"
printf 'completed_at=%s\njob_id=%s\nquery_index=%s\noutput=%s\n' \
  "$(date -Is)" "$SLURM_JOB_ID" "$CSD_MECHANISM_QUERY_INDEX" \
  "$CSD_MECHANISM_OUTPUT" > "$CSD_MARKER_TMP"
mv "$CSD_MARKER_TMP" "$CSD_MECHANISM_DONE"
