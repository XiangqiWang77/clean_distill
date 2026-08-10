#!/usr/bin/env bash
#SBATCH --job-name=baseline-table
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${CSD_PREPARED_ROOT:?export CSD_PREPARED_ROOT}"
: "${BASELINE_RUN_ROOT:?export BASELINE_RUN_ROOT}"

# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"
export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

score_four_shards() {
  local prediction_root=$1
  local output=$2
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    --predictions "$prediction_root/shard_00.jsonl" \
    --predictions "$prediction_root/shard_01.jsonl" \
    --predictions "$prediction_root/shard_02.jsonl" \
    --predictions "$prediction_root/shard_03.jsonl" \
    --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
    --sample-count 1 \
    --output "$output"
}

mkdir -p "$BASELINE_RUN_ROOT/comparison"
score_four_shards \
  "$BASELINE_RUN_ROOT/eval/demopsd/predictions" \
  "$BASELINE_RUN_ROOT/eval/demopsd/scored.jsonl"
score_four_shards \
  "$BASELINE_RUN_ROOT/eval/grpo/predictions" \
  "$BASELINE_RUN_ROOT/eval/grpo/scored.jsonl"
score_four_shards \
  "$BASELINE_RUN_ROOT/eval/base_vllm/predictions" \
  "$BASELINE_RUN_ROOT/eval/base_vllm/scored.jsonl"
score_four_shards \
  "$BASELINE_RUN_ROOT/eval/privileged_16_vllm/predictions" \
  "$BASELINE_RUN_ROOT/eval/privileged_16_vllm/scored.jsonl"
score_four_shards \
  "$BASELINE_RUN_ROOT/eval/trsd_16_vllm/predictions" \
  "$BASELINE_RUN_ROOT/eval/trsd_16_vllm/scored.jsonl"
score_four_shards \
  "$BASELINE_RUN_ROOT/eval/privileged_64_vllm/predictions" \
  "$BASELINE_RUN_ROOT/eval/privileged_64_vllm/scored.jsonl"
score_four_shards \
  "$BASELINE_RUN_ROOT/eval/trsd_64_vllm/predictions" \
  "$BASELINE_RUN_ROOT/eval/trsd_64_vllm/scored.jsonl"

"$CSD_TTT_PYTHON" baselines/build_comparison_table.py \
  --entry "Base|0|$BASELINE_RUN_ROOT/eval/base_vllm/scored.jsonl" \
  --entry "Privilege-SD 16|16|$BASELINE_RUN_ROOT/eval/privileged_16_vllm/scored.jsonl" \
  --entry "TRSD 16|16|$BASELINE_RUN_ROOT/eval/trsd_16_vllm/scored.jsonl" \
  --entry "DemoPSD 16|16|$BASELINE_RUN_ROOT/eval/demopsd/scored.jsonl" \
  --entry "GRPO 16|16|$BASELINE_RUN_ROOT/eval/grpo/scored.jsonl" \
  --entry "Privilege-SD 64|64|$BASELINE_RUN_ROOT/eval/privileged_64_vllm/scored.jsonl" \
  --entry "TRSD 64|64|$BASELINE_RUN_ROOT/eval/trsd_64_vllm/scored.jsonl" \
  --output-dir "$BASELINE_RUN_ROOT/comparison"
printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" \
  > "$BASELINE_RUN_ROOT/comparison/COMPLETE.done"
