#!/usr/bin/env bash
#SBATCH --job-name=csd8b-report
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --mem=256G
#SBATCH --time=01:00:00

set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?The empirical submitter must export CSD_RUN_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
# shellcheck disable=SC1091
source "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/empirical_common.sh"
csd_assert_gpu 'RTX.*6000|PRO.*6000'

CSD_REPORT_ROOT="$CSD_RUN_ROOT/report"
CSD_SCORED_ROOT="$CSD_REPORT_ROOT/scored"
CSD_MARKER="$CSD_RUN_ROOT/status/report.done"
mkdir -p "$CSD_SCORED_ROOT" "$CSD_REPORT_ROOT/aux"
[[ -s "$CSD_RUN_ROOT/dev/dev_audit.json" ]]
if [[ -f "$CSD_MARKER" ]]; then
  [[ -s "$CSD_REPORT_ROOT/main_table.json" ]]
  [[ -s "$CSD_REPORT_ROOT/main_table.csv" ]]
  [[ -s "$CSD_REPORT_ROOT/dev_audit.json" ]]
  exit 0
fi

csd_score_shards() {
  local raw_directory=$1
  local scored_output=$2
  local shard_arguments=()
  local index label path
  for ((index=0; index<CSD_EVAL_NUM_SHARDS; index++)); do
    label=$(printf '%03d' "$index")
    path="$raw_directory/shard_${label}.jsonl"
    [[ -s "$path" ]]
    shard_arguments+=(--predictions "$path")
  done
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
    "${shard_arguments[@]}" \
    --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
    --output "$scored_output"
  [[ -s "$scored_output" ]]
}

# Offline labels enter for the first time here.  Every GPU generation stage
# consumed query-only files, except the explicitly HER=1 post-outcome control.
CSD_SHORT_METHODS=(base csd_t csd_t_correct_only csd_sd privileged_sd)
for method in "${CSD_SHORT_METHODS[@]}"; do
  csd_score_shards \
    "$CSD_RUN_ROOT/short/predictions/$method" \
    "$CSD_SCORED_ROOT/short_${method}.jsonl"
done

CSD_LONG_METHODS=(clean_sd privileged_sd)
CSD_CHECKPOINTS=(250 500 750 1000)
for method in "${CSD_LONG_METHODS[@]}"; do
  for episode in "${CSD_CHECKPOINTS[@]}"; do
    episode_label=$(printf '%04d' "$episode")
    csd_score_shards \
      "$CSD_RUN_ROOT/persistent_eval/predictions/$method/episode_${episode_label}" \
      "$CSD_SCORED_ROOT/long_${method}_${episode_label}.jsonl"
  done
done

CSD_PRE_ARGS=()
CSD_POST_ARGS=()
for ((index=0; index<CSD_EVAL_NUM_SHARDS; index++)); do
  label=$(printf '%03d' "$index")
  pre="$CSD_RUN_ROOT/mechanism/raw/pre_decision_privilege/shard_${label}.jsonl"
  post="$CSD_RUN_ROOT/mechanism/raw/post_outcome_privilege/shard_${label}.jsonl"
  [[ -s "$pre" && -s "$post" ]]
  CSD_PRE_ARGS+=(--pre-decision "$pre")
  CSD_POST_ARGS+=(--post-outcome "$post")
done

CSD_MECHANISM="$CSD_REPORT_ROOT/aux/mechanism.jsonl"
CSD_ABLATION="$CSD_REPORT_ROOT/aux/ablation.jsonl"
CSD_AUX_AUDIT="$CSD_REPORT_ROOT/aux/audit.json"
"$CSD_TTT_PYTHON" scripts/clean_self_distill/08_build_empirical_aux.py \
  --labels "$CSD_PREPARED_ROOT/heldout_labels.sealed.jsonl" \
  --base-scored "$CSD_SCORED_ROOT/short_base.jsonl" \
  --signed-scored "$CSD_SCORED_ROOT/short_csd_t.jsonl" \
  --correct-only-scored "$CSD_SCORED_ROOT/short_csd_t_correct_only.jsonl" \
  "${CSD_PRE_ARGS[@]}" \
  "${CSD_POST_ARGS[@]}" \
  --mechanism-output "$CSD_MECHANISM" \
  --ablation-output "$CSD_ABLATION" \
  --audit-output "$CSD_AUX_AUDIT"
[[ -s "$CSD_MECHANISM" && -s "$CSD_ABLATION" && -s "$CSD_AUX_AUDIT" ]]

CSD_CHECKPOINT_MANIFEST_ARGS=()
for method in "${CSD_LONG_METHODS[@]}"; do
  for episode in 0 "${CSD_CHECKPOINTS[@]}"; do
    episode_label=$(printf '%04d' "$episode")
    manifest="$CSD_RUN_ROOT/persistent/$method/checkpoints/episode_${episode_label}/checkpoint_manifest.json"
    [[ -s "$manifest" ]]
    CSD_CHECKPOINT_MANIFEST_ARGS+=("$manifest")
  done
done
CSD_HELDOUT_ARGS=("$CSD_SCORED_ROOT/short_base.jsonl")
for method in "${CSD_LONG_METHODS[@]}"; do
  for episode in "${CSD_CHECKPOINTS[@]}"; do
    episode_label=$(printf '%04d' "$episode")
    CSD_HELDOUT_ARGS+=("$CSD_SCORED_ROOT/long_${method}_${episode_label}.jsonl")
  done
done
CSD_SHORT_ARGS=(
  "$CSD_SCORED_ROOT/short_base.jsonl"
  "$CSD_SCORED_ROOT/short_privileged_sd.jsonl"
  "$CSD_SCORED_ROOT/short_csd_t.jsonl"
  "$CSD_SCORED_ROOT/short_csd_sd.jsonl"
)

"$CSD_TTT_PYTHON" scripts/clean_self_distill/report_persistent_metrics.py \
  --training-journal \
    "$CSD_RUN_ROOT/persistent/clean_sd/episodes.jsonl" \
    "$CSD_RUN_ROOT/persistent/privileged_sd/episodes.jsonl" \
  --checkpoint-manifest "${CSD_CHECKPOINT_MANIFEST_ARGS[@]}" \
  --heldout-scored "${CSD_HELDOUT_ARGS[@]}" \
  --short-term-scored "${CSD_SHORT_ARGS[@]}" \
  --mechanism "$CSD_MECHANISM" \
  --ablation "$CSD_ABLATION" \
  --output-json "$CSD_REPORT_ROOT/main_table.json" \
  --output-csv "$CSD_REPORT_ROOT/main_table.csv"

[[ -s "$CSD_REPORT_ROOT/main_table.json" && -s "$CSD_REPORT_ROOT/main_table.csv" ]]
cp "$CSD_RUN_ROOT/dev/dev_audit.json" "$CSD_REPORT_ROOT/.dev_audit.json.tmp.$$"
mv "$CSD_REPORT_ROOT/.dev_audit.json.tmp.$$" "$CSD_REPORT_ROOT/dev_audit.json"
csd_atomic_marker "$CSD_MARKER"
