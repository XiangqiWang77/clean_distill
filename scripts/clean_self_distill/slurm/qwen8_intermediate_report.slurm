#!/usr/bin/env bash
#SBATCH --job-name=q8-int-table
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -Eeuo pipefail
: "${CSD_I8_CONFIG:?export CSD_I8_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_I8_CONFIG"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_I8_CONDA_ENV"
cd "$CSD_I8_SNAPSHOT_ROOT"
export PYTHONPATH="$CSD_I8_SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

score_checkpoint() {
  local tag=$1
  local root="$CSD_I8_RUN_ROOT/eval/$tag"
  for shard in 00 01 02 03; do
    [[ -s "$root/predictions/shard_${shard}.jsonl" ]]
    [[ -s "$root/status/shard_${shard}.done" ]]
  done
  "$CSD_I8_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    --predictions "$root/predictions/shard_00.jsonl" \
    --predictions "$root/predictions/shard_01.jsonl" \
    --predictions "$root/predictions/shard_02.jsonl" \
    --predictions "$root/predictions/shard_03.jsonl" \
    --labels "$CSD_I8_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
    --sample-count 1 \
    --output "$root/scored.jsonl"
  [[ "$(wc -l < "$root/scored.jsonl")" == 143 ]]
}

for tag in privileged_32 privileged_48 trsd_32 trsd_48; do
  score_checkpoint "$tag"
done

output="$CSD_I8_RUN_ROOT/report"
"$CSD_I8_PYTHON" "$CSD_I8_CODE_ROOT/scripts/clean_self_distill/27_cross_model_math_table.py" \
  --qwen8-existing-eval-root "$CSD_I8_QWEN8_EXISTING_EVAL_ROOT" \
  --qwen8-new-eval-root "$CSD_I8_RUN_ROOT/eval" \
  --qwen8-privileged-journal "$CSD_I8_PRIVILEGED_JOURNAL" \
  --qwen8-trsd-journal "$CSD_I8_TRSD_TRAIN_ROOT/episodes.jsonl" \
  --qwen17-root "$CSD_I8_QWEN17_ROOT" \
  --gptoss20-root "$CSD_I8_GPTOSS20_ROOT" \
  --output-dir "$output"

"$CSD_I8_PYTHON" \
  "$CSD_I8_CODE_ROOT/scripts/clean_self_distill/28_plot_math_training_dynamics.py" \
  --privileged-journal "$CSD_I8_PRIVILEGED_JOURNAL" \
  --trsd-journal "$CSD_I8_TRSD_TRAIN_ROOT/episodes.jsonl" \
  --deepmath-labels "$CSD_I8_PREPARED_ROOT/distill_labels.sealed.jsonl" \
  --accuracy "$output/qwen3_8b_intermediate_checkpoints.csv" \
  --output-dir "$output/math_training_dynamics"

[[ -s "$CSD_I8_DYNAMICS_ROOT/DYNAMICS_COMPLETE" ]]
[[ -s "$CSD_I8_DYNAMICS_ROOT/common_evaluation_nll_rebound.pdf" ]]
[[ -s "$output/math_training_dynamics/math_training_dynamics.pdf" ]]
[[ -s "$output/math_training_dynamics/math_training_dynamics.csv" ]]
[[ "$(jq -r '.accuracy_complete' "$output/math_training_dynamics/summary.json")" == true ]]
printf 'completed_at=%s\njob_id=%s\n' "$(date -Is)" "$SLURM_JOB_ID" \
  > "$output/REPORT_COMPLETE"
