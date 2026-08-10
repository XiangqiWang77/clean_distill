#!/usr/bin/env bash
#SBATCH --job-name=trsd-short-study
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/logs/trsd-short-study-%j.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/logs/trsd-short-study-%j.err

# A single-H100, self-contained empirical run for the current TRSD method.
# It trains two persistent episodes, evaluates checkpoints 0/1/2 on a fixed
# balanced held-out pilot, collects three multi-query answer-free style controls,
# and renders the resulting tables and figures before exiting.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_SHORT_ROOT:?export CSD_SHORT_ROOT}"
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
export TMPDIR="$CSD_SHORT_ROOT/tmp_${SLURM_JOB_ID}"

mkdir -p "$CSD_SHORT_ROOT" "$TMPDIR" "$CSD_SCRATCH_ROOT/logs"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

CSD_PILOT_ROOT="$CSD_SHORT_ROOT/heldout_pilot"
CSD_TRAIN_ROOT="$CSD_SHORT_ROOT/train"
CSD_EVAL_ROOT="$CSD_SHORT_ROOT/eval"
CSD_MECHANISM_ROOT="$CSD_SHORT_ROOT/mechanism"
CSD_REPORT_ROOT="$CSD_SHORT_ROOT/report"
CSD_RESOURCE_LOG="$CSD_SHORT_ROOT/job_resources.tsv"
mkdir -p "$CSD_PILOT_ROOT" "$CSD_EVAL_ROOT" "$CSD_MECHANISM_ROOT" "$CSD_REPORT_ROOT"

"$CSD_TTT_PYTHON" scripts/clean_self_distill/08_short_empirical_report.py prepare-stream \
  --queries "$CSD_PREPARED_ROOT/distill_queries.jsonl" \
  --episodes 2 \
  --output "$CSD_SHORT_ROOT/distill_queries_2.jsonl"

"$CSD_TTT_PYTHON" scripts/clean_self_distill/08_short_empirical_report.py prepare-subset \
  --queries "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
  --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
  --out-dir "$CSD_PILOT_ROOT" \
  --query-name queries.jsonl \
  --label-name labels.jsonl \
  --per-source 2

CSD_RESUME=()
[[ ! -s "$CSD_TRAIN_ROOT/run_manifest.json" ]] || CSD_RESUME+=(--resume)
printf 'stage\tstart\t%s\n' "$(date -Is)" >> "$CSD_RESOURCE_LOG"
/usr/bin/time -a -o "$CSD_RESOURCE_LOG" -f 'train\telapsed_seconds=%e\tmax_rss_kib=%M' \
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/04_persistent_train.py \
    --branch clean \
    --queries "$CSD_SHORT_ROOT/distill_queries_2.jsonl" \
    --model "$CSD_MODEL_LOCAL_DIR" \
    --model-id "$CSD_MODEL_ID" \
    --revision "$CSD_MODEL_REVISION" \
    --output-dir "$CSD_TRAIN_ROOT" \
    --episodes 2 \
    --scientific-checkpoints 0,1,2 \
    --rolling-checkpoint-interval 1 \
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

for CSD_EPISODE in 0 1 2; do
  CSD_LABEL=$(printf '%04d' "$CSD_EPISODE")
  CSD_PREDICTIONS="$CSD_EVAL_ROOT/episode_${CSD_LABEL}.jsonl"
  CSD_SCORED="$CSD_EVAL_ROOT/episode_${CSD_LABEL}.scored.jsonl"
  CSD_METHOD=base
  CSD_ADAPTER_ARGS=()
  if (( CSD_EPISODE > 0 )); then
    CSD_METHOD=trsd
    CSD_ADAPTER_ARGS=(--adapter "$CSD_TRAIN_ROOT/checkpoints/episode_${CSD_LABEL}")
  fi
  /usr/bin/time -a -o "$CSD_RESOURCE_LOG" -f "eval_${CSD_LABEL}\\telapsed_seconds=%e\\tmax_rss_kib=%M" \
    "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
      --queries "$CSD_PILOT_ROOT/queries.jsonl" \
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
    --labels "$CSD_PILOT_ROOT/labels.jsonl" \
    --sample-count 1 \
    --output "$CSD_SCORED"
done

for CSD_QUERY_INDEX in 0 1 2; do
  CSD_QUERY_LABEL=$(printf '%02d' "$CSD_QUERY_INDEX")
  /usr/bin/time -a -o "$CSD_RESOURCE_LOG" -f "mechanism_${CSD_QUERY_LABEL}\\telapsed_seconds=%e\\tmax_rss_kib=%M" \
    "$CSD_TTT_PYTHON" scripts/clean_self_distill/06_trust_region_mechanism.py \
      --queries "$CSD_PREPARED_ROOT/dev_queries.jsonl" \
      --adapter "$CSD_TRAIN_ROOT/checkpoints/episode_0002" \
      --output "$CSD_MECHANISM_ROOT/query_${CSD_QUERY_LABEL}.jsonl" \
      --model "$CSD_MODEL_LOCAL_DIR" \
      --model-id "$CSD_MODEL_ID" \
      --revision "$CSD_MODEL_REVISION" \
      --query-index "$CSD_QUERY_INDEX" \
      --max-new-tokens 10240 \
      --max-prompt-tokens "$CSD_MAX_PROMPT_TOKENS" \
      --context-window "$CSD_CONTEXT_WINDOW_TOKENS" \
      --alpha-grid 0,0.25,0.5,0.75,1 \
      --epsilon-grid 0.001,0.002,0.004,0.008,0.016 \
      --selected-epsilon 0.004 \
      --training-epsilon 0.004 \
      --binary-search-steps "$CSD_TRUST_REGION_BINARY_SEARCH_STEPS" \
      --full-vocab-chunk-size "$CSD_DISTILL_TOKEN_CHUNK_SIZE" \
      --attn-implementation sdpa \
      --seed "$CSD_RUN_SEED"
done

"$CSD_TTT_PYTHON" scripts/clean_self_distill/08_short_empirical_report.py report \
  --episodes "$CSD_TRAIN_ROOT/episodes.jsonl" \
  --scored 0:"$CSD_EVAL_ROOT/episode_0000.scored.jsonl" \
  --scored 1:"$CSD_EVAL_ROOT/episode_0001.scored.jsonl" \
  --scored 2:"$CSD_EVAL_ROOT/episode_0002.scored.jsonl" \
  --mechanism "$CSD_MECHANISM_ROOT/query_00.jsonl" \
  --mechanism "$CSD_MECHANISM_ROOT/query_01.jsonl" \
  --mechanism "$CSD_MECHANISM_ROOT/query_02.jsonl" \
  --epsilon 0.004 \
  --out-dir "$CSD_REPORT_ROOT"

printf 'stage\tcomplete\t%s\n' "$(date -Is)" >> "$CSD_RESOURCE_LOG"
printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" > "$CSD_SHORT_ROOT/COMPLETE"
