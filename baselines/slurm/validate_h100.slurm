#!/usr/bin/env bash
#SBATCH --job-name=baseline-validity
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2

set -Eeuo pipefail
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
: "${BASELINE_VALIDATION_ROOT:?export BASELINE_VALIDATION_ROOT}"

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
export TMPDIR="$BASELINE_VALIDATION_ROOT/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

METHODS=(demopsd grpo)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
RUN_ROOT="$BASELINE_VALIDATION_ROOT/$METHOD"
mkdir -p "$TMPDIR" "$RUN_ROOT"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

"$CSD_TTT_PYTHON" -m pytest -q tests/test_baseline_objectives.py
"$CSD_TTT_PYTHON" baselines/train.py \
  --method "$METHOD" \
  --queries baselines/fixtures/smoke_queries.jsonl \
  --labels baselines/fixtures/smoke_labels.sealed.jsonl \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --output-dir "$RUN_ROOT/train" \
  --episodes 1 \
  --group-size 2 \
  --max-sequence-tokens 2048 \
  --teacher-max-sequence-tokens 4096 \
  --max-rollout-tokens 512 \
  --privileged-max-tokens 256 \
  --disable-thinking \
  --token-chunk-size 16 \
  --attn-implementation sdpa

CHECKPOINT="$RUN_ROOT/train/checkpoints/episode_0001"
[[ -s "$CHECKPOINT/checkpoint_manifest.json" ]]
"$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py generate \
  --queries baselines/fixtures/smoke_queries.jsonl \
  --output "$RUN_ROOT/predictions.jsonl" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --model-id "$CSD_MODEL_ID" \
  --revision "$CSD_MODEL_REVISION" \
  --adapter "$CHECKPOINT" \
  --method "$METHOD" \
  --checkpoint-episode 1 \
  --sample-count 1 \
  --max-new-tokens 512 \
  --max-prompt-tokens 512 \
  --context-window 2048 \
  --attn-implementation sdpa \
  --seed 0
"$CSD_TTT_PYTHON" scripts/clean_self_distill/05_heldout_eval.py score \
  --predictions "$RUN_ROOT/predictions.jsonl" \
  --labels baselines/fixtures/smoke_labels.sealed.jsonl \
  --sample-count 1 \
  --output "$RUN_ROOT/scored.jsonl"

printf 'completed_at=%s\njob_id=%s\nmethod=%s\n' \
  "$(date -Is)" "$SLURM_JOB_ID" "$METHOD" > "$RUN_ROOT/VALID.done"
