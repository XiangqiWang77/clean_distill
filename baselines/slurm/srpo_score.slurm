#!/usr/bin/env bash
#SBATCH --job-name=srpo-score
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%j.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/srpo-three-model-20260809/logs/%x_%j.err

set -Eeuo pipefail
: "${CSD_SRPO_CONFIG:?export CSD_SRPO_CONFIG}"
: "${CSD_SRPO_MODEL_KEY:?export CSD_SRPO_MODEL_KEY}"
# shellcheck disable=SC1090
source "$CSD_SRPO_CONFIG"

case "$CSD_SRPO_MODEL_KEY" in
  q17) MODEL_LABEL=$CSD_SRPO_Q17_MODEL_LABEL; BASE=$CSD_SRPO_Q17_BASE ;;
  q8) MODEL_LABEL=$CSD_SRPO_Q8_MODEL_LABEL; BASE=$CSD_SRPO_Q8_BASE ;;
  go20) MODEL_LABEL=$CSD_SRPO_GO_MODEL_LABEL; BASE=$CSD_SRPO_GO_BASE ;;
  *) printf 'unsupported model key: %s\n' "$CSD_SRPO_MODEL_KEY" >&2; exit 2 ;;
esac

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_SRPO_CONDA_ENV"
cd "$CSD_SRPO_SNAPSHOT_ROOT"
export PYTHONPATH="$CSD_SRPO_SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for episode in 16 64; do
  eval_root="$CSD_SRPO_RUN_ROOT/eval/$CSD_SRPO_MODEL_KEY/episode_$(printf '%04d' "$episode")"
  predictions=()
  for shard in 0 1 2 3; do
    path="$eval_root/predictions/shard_$(printf '%02d' "$shard").jsonl"
    [[ -s "$path" ]]
    predictions+=(--predictions "$path")
  done
  "$CSD_SRPO_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    "${predictions[@]}" \
    --labels "$CSD_SRPO_HELDOUT_LABELS" \
    --output "$eval_root/scored.jsonl" \
    --sample-count 1
  [[ "$(wc -l < "$eval_root/scored.jsonl")" == 143 ]]
done

RESULTS="$CSD_SRPO_RUN_ROOT/results/$CSD_SRPO_MODEL_KEY"
"$CSD_SRPO_PYTHON" baselines/build_comparison_table.py \
  --entry "Base|0|$BASE" \
  --entry "SRPO 16|16|$CSD_SRPO_RUN_ROOT/eval/$CSD_SRPO_MODEL_KEY/episode_0016/scored.jsonl" \
  --entry "SRPO 64|64|$CSD_SRPO_RUN_ROOT/eval/$CSD_SRPO_MODEL_KEY/episode_0064/scored.jsonl" \
  --model-label "$MODEL_LABEL" \
  --output-dir "$RESULTS"
printf 'complete\njob_id=%s\nmodel=%s\n' \
  "$SLURM_JOB_ID" "$CSD_SRPO_MODEL_KEY" \
  > "$CSD_SRPO_RUN_ROOT/${CSD_SRPO_MODEL_KEY}.SCORE_COMPLETE"
