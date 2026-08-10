#!/usr/bin/env bash
#SBATCH --job-name=source-valid
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%A_%a-extended-valid.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%A_%a-extended-valid.err

set -Eeuo pipefail
: "${CSD_PT_CONFIG:?export CSD_PT_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_PT_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_PT_CONDA_ENV"
cd "$CSD_PT_CODE_ROOT"

export PYTHONPATH="$CSD_PT_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_PT_SCRATCH_ROOT/hf" HF_HUB_CACHE="$CSD_PT_SCRATCH_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$CSD_PT_RUN_ROOT/tmp_extended_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_ROOT="$CSD_PT_RUN_ROOT/validity/${SLURM_ARRAY_JOB_ID}/extended_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$RUN_ROOT"

"$CSD_PT_PYTHON" -m pytest -q \
  tests/test_baseline_objectives.py tests/test_persistent_training.py

if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  "$CSD_PT_PYTHON" baselines/train.py \
    --method trsd_source \
    --privilege-source style_only_directive \
    --trajectory-projection \
    --queries "$CSD_PT_PREPARED_ROOT/distill_queries.jsonl" \
    --labels "$CSD_PT_TRAIN_LABELS" \
    --model "$CSD_PT_MODEL_LOCAL_DIR" \
    --model-id "$CSD_PT_MODEL_ID" \
    --revision "$CSD_PT_MODEL_REVISION" \
    --output-dir "$RUN_ROOT/train" \
    --episodes 1 \
    --group-size 1 \
    --max-sequence-tokens 2048 \
    --teacher-max-sequence-tokens 4096 \
    --max-rollout-tokens 512 \
    --privileged-max-tokens 256 \
    --disable-thinking \
    --token-chunk-size 16 \
    --projection-kl-budget 0.004 \
    --projection-binary-search-steps 6 \
    --attn-implementation sdpa
  ROW="$RUN_ROOT/train/episodes.jsonl"
else
  "$CSD_PT_PYTHON" scripts/clean_self_distill/08_short_empirical_report.py prepare-stream \
    --queries "$CSD_PT_PREPARED_ROOT/distill_queries.jsonl" \
    --episodes 1 \
    --output "$RUN_ROOT/query.jsonl"
  "$CSD_PT_PYTHON" scripts/clean_self_distill/04_persistent_train.py \
    --branch clean \
    --queries "$RUN_ROOT/query.jsonl" \
    --model "$CSD_PT_MODEL_LOCAL_DIR" \
    --model-id "$CSD_PT_MODEL_ID" \
    --revision "$CSD_PT_MODEL_REVISION" \
    --output-dir "$RUN_ROOT/train" \
    --episodes 1 \
    --scientific-checkpoints 0,1 \
    --rolling-checkpoint-interval 1 \
    --max-sequence-tokens 2048 \
    --max-rollout-tokens 512 \
    --distill-token-chunk-size 16 \
    --trust-region-kl-budget 0.004 \
    --trust-region-binary-search-steps 6 \
    --update-guard \
    --attn-implementation sdpa
  ROW="$RUN_ROOT/train/episodes.jsonl"
fi

"$CSD_PT_PYTHON" - "$ROW" <<'PY'
import json
import sys
from pathlib import Path

row = json.loads(Path(sys.argv[1]).read_text().splitlines()[0])
required_phases = {"rollout", "teacher", "target", "update"}
if set(row["phase_seconds"]) != required_phases:
    raise SystemExit("four-phase timing is incomplete")
if not all(float(row["phase_seconds"][key]) >= 0 for key in required_phases):
    raise SystemExit("phase time cannot be negative")
if "cuda_peak_memory_allocated_bytes" not in row["resource_usage"]:
    raise SystemExit("peak memory audit is missing")
PY

printf 'completed_at=%s\njob_id=%s\ntask=%s\n' \
  "$(date -Is)" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" > "$RUN_ROOT/VALID.done"
