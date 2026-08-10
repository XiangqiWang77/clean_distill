#!/usr/bin/env bash
#SBATCH --job-name=proj-valid
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%A_%a-valid.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%A_%a-valid.err

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
export TMPDIR="$CSD_PT_RUN_ROOT/tmp_valid_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

METHODS=(opsd opsd demopsd demopsd)
PROJECTIONS=(0 1 0 1)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
PROJECTED="${PROJECTIONS[$SLURM_ARRAY_TASK_ID]}"
VARIANT="${METHOD}_original"
PROJECTION_ARGS=()
METHOD_ID="baseline:${METHOD}"
if [[ "$METHOD" == opsd ]]; then
  METHOD_ID=baseline:opsd:official_generalized_jsd_fixed_teacher_v1
else
  METHOD_ID=baseline:demopsd:exact_full_vocab_v1
fi
if [[ "$PROJECTED" == 1 ]]; then
  VARIANT="${METHOD}_projected"
  METHOD_ID="${METHOD_ID}:trajectory_projection_v1"
  PROJECTION_ARGS=(--trajectory-projection)
fi
RUN_ROOT="$CSD_PT_RUN_ROOT/validity/${SLURM_ARRAY_JOB_ID}/$VARIANT"
mkdir -p "$TMPDIR" "$RUN_ROOT" "$CSD_PT_RUN_ROOT/logs"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null
[[ "$(sha256sum "$CSD_PT_PREPARED_ROOT/distill_queries.jsonl" | awk '{print $1}')" == "$CSD_PT_DISTILL_QUERIES_SHA256" ]]
[[ "$(sha256sum "$CSD_PT_TRAIN_LABELS" | awk '{print $1}')" == "$CSD_PT_TRAIN_LABELS_SHA256" ]]

"$CSD_PT_PYTHON" -m pytest -q \
  tests/test_baseline_objectives.py tests/test_persistent_training.py
"$CSD_PT_PYTHON" baselines/train.py \
  --method "$METHOD" \
  --queries "$CSD_PT_PREPARED_ROOT/distill_queries.jsonl" \
  --labels "$CSD_PT_TRAIN_LABELS" \
  --model "$CSD_PT_MODEL_LOCAL_DIR" \
  --model-id "$CSD_PT_MODEL_ID" \
  --revision "$CSD_PT_MODEL_REVISION" \
  --output-dir "$RUN_ROOT/train" \
  --episodes 1 \
  --group-size 2 \
  --generation-batch-size 2 \
  --max-sequence-tokens 2048 \
  --teacher-max-sequence-tokens 4096 \
  --max-rollout-tokens 512 \
  --privileged-max-tokens 256 \
  --disable-thinking \
  --token-chunk-size 16 \
  --projection-kl-budget "$CSD_PT_PROJECTION_KL_BUDGET" \
  --projection-binary-search-steps "$CSD_PT_PROJECTION_SEARCH_STEPS" \
  --attn-implementation sdpa \
  "${PROJECTION_ARGS[@]}"

CHECKPOINT="$RUN_ROOT/train/checkpoints/episode_0001"
"$CSD_PT_PYTHON" - "$CHECKPOINT" "$METHOD" "$METHOD_ID" "$CSD_PT_MODEL_ID" "$CSD_PT_MODEL_REVISION" <<'PY'
import json
import sys
from pathlib import Path

checkpoint, method, method_id, model_id, revision = sys.argv[1:]
manifest = json.loads((Path(checkpoint) / "checkpoint_manifest.json").read_text())
expected = {
    "schema_version": "clean-self-distill-persistent-checkpoint-v1",
    "branch": method,
    "checkpoint_episode": 1,
    "completed_episodes": 1,
    "method_id": method_id,
    "model_id": model_id,
    "model_revision": revision,
}
mismatches = {
    key: (manifest.get(key), value)
    for key, value in expected.items()
    if manifest.get(key) != value
}
if mismatches:
    raise SystemExit(f"checkpoint identity mismatch: {mismatches}")
row = json.loads((Path(checkpoint).parents[1] / "episodes.jsonl").read_text().splitlines()[0])
if bool(row["trajectory_projection"]) != method_id.endswith(":trajectory_projection_v1"):
    raise SystemExit("trajectory projection flag does not match method identity")
if method_id.endswith(":trajectory_projection_v1") and row["optimized_rollout_count"]:
    metrics = row.get("objective_metrics", {})
    required = {"projection_alpha", "raw_target_kl", "projected_target_kl"}
    if not required <= metrics.keys():
        raise SystemExit(f"projection metrics missing: {required - metrics.keys()}")
PY

printf 'completed_at=%s\njob_id=%s\nmethod=%s\nprojected=%s\nmethod_id=%s\n' \
  "$(date -Is)" "$SLURM_JOB_ID" "$METHOD" "$PROJECTED" "$METHOD_ID" > "$RUN_ROOT/VALID.done"
