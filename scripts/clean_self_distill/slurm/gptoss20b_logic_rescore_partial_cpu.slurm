#!/usr/bin/env bash
#SBATCH --job-name=go20-logic-score
#SBATCH --account=pi_mg269
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%j-logic-rescore.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/gptoss20b-fiveway-20260809/logs/%j-logic-rescore.err

set -Eeuo pipefail
: "${CSD_GO_CONFIG:?export CSD_GO_CONFIG}"
: "${CSD_GO_LOGIC_RESCORE_ROOT:?export CSD_GO_LOGIC_RESCORE_ROOT}"
: "${CSD_GO_LOGIC_SCORER_SHA256:?export CSD_GO_LOGIC_SCORER_SHA256}"
: "${CSD_GO_LOGIC_PARSER_SHA256:?export CSD_GO_LOGIC_PARSER_SHA256}"

# shellcheck disable=SC1090
source "$CSD_GO_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_GO_CONDA_ENV"
cd "$CSD_GO_CODE_ROOT"

export PYTHONPATH="$CSD_GO_SATQUEST_DEPS:$CSD_GO_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/tmp/go20_logic_rescore_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$CSD_GO_RUN_ROOT/logs" "$CSD_GO_LOGIC_RESCORE_ROOT/scored"

[[ "$(sha256sum scripts/clean_self_distill/15_logic_eval.py | awk '{print $1}')" == "$CSD_GO_LOGIC_SCORER_SHA256" ]]
[[ "$(sha256sum src/clean_self_distill/logic_evaluation.py | awk '{print $1}')" == "$CSD_GO_LOGIC_PARSER_SHA256" ]]
[[ "$CSD_GO_LOGIC_NUM_SHARDS" == 8 ]]
[[ "$(sha256sum "$CSD_GO_LOGIC_DATA_ROOT/satquest/val.parquet" | awk '{print $1}')" == "$CSD_GO_SATQUEST_EVAL_SHA256" ]]
[[ "$(sha256sum "$CSD_GO_LOGIC_DATA_ROOT/logicskills/val.parquet" | awk '{print $1}')" == "$CSD_GO_LOGICSKILLS_EVAL_SHA256" ]]

SHARDS=(0 1 2 3 5 6)
METHODS=(base privileged_sd_16 trsd_16 privileged_sd_64 trsd_64)
for shard in "${SHARDS[@]}"; do
  expected_rows=$(( (4860 + CSD_GO_LOGIC_NUM_SHARDS - 1 - shard) / CSD_GO_LOGIC_NUM_SHARDS ))
  shard_name=$(printf '%02d' "$shard")
  for method in "${METHODS[@]}"; do
    predictions="$CSD_GO_RUN_ROOT/logic/predictions/$method/shard_${shard_name}.jsonl"
    scored="$CSD_GO_LOGIC_RESCORE_ROOT/scored/$method/shard_${shard_name}.jsonl"
    [[ -s "$predictions" ]]
    [[ "$(wc -l < "$predictions")" == "$expected_rows" ]]
    mkdir -p "$(dirname "$scored")"
    if [[ -s "$scored" ]] && [[ "$(wc -l < "$scored")" == "$expected_rows" ]]; then
      continue
    fi
    "$CSD_GO_PYTHON" scripts/clean_self_distill/15_logic_eval.py score \
      --data-root "$CSD_GO_LOGIC_DATA_ROOT" \
      --predictions "$predictions" \
      --output "$scored" \
      --satquest-repo "$CSD_GO_SATQUEST_REPO" \
      --satquest-deps "$CSD_GO_SATQUEST_DEPS" \
      --logicskills-repo "$CSD_GO_LOGICSKILLS_REPO" \
      --num-shards "$CSD_GO_LOGIC_NUM_SHARDS" \
      --shard-index "$shard"
    [[ "$(wc -l < "$scored")" == "$expected_rows" ]]
  done
done

"$CSD_GO_PYTHON" scripts/clean_self_distill/15_logic_eval.py report \
  --run-root "$CSD_GO_LOGIC_RESCORE_ROOT" \
  --num-shards "$CSD_GO_LOGIC_NUM_SHARDS" \
  --shard-indices 0,1,2,3,5,6 \
  --methods base,privileged_sd_16,trsd_16,privileged_sd_64,trsd_64 \
  --model-label "$CSD_GO_MODEL_LABEL"

[[ -s "$CSD_GO_LOGIC_RESCORE_ROOT/reports/LOGIC_RESULTS.md" ]]
printf 'completed_at=%s\njob_id=%s\nsource_shards=0,1,2,3,5,6\nscorer_sha256=%s\nparser_sha256=%s\n' \
  "$(date -Is)" "$SLURM_JOB_ID" "$CSD_GO_LOGIC_SCORER_SHA256" "$CSD_GO_LOGIC_PARSER_SHA256" \
  > "$CSD_GO_LOGIC_RESCORE_ROOT/RESCORE_COMPLETE"
