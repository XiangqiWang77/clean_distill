#!/usr/bin/env bash
#SBATCH --job-name=source64-drift
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%j-source-drift.out
#SBATCH --error=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/qwen3-8b-projection-tables-20260809/logs/%j-source-drift.err

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
export TMPDIR="$CSD_PT_RUN_ROOT/tmp_source_drift_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

BASE_ROOT=/home/da839/scratch_pi_mg269/da839/clean_distill/runs/csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/main_eval/predictions/base
METHOD_ARGS=()
for source in \
  verified_reference_solution \
  answer_free_reasoning_method \
  verifier_critique \
  execution_solver_feedback \
  equivalent_prompt_wrappers \
  style_only_directive \
  permuted_noninferable_context; do
  METHOD_ARGS+=(--method "$source=$CSD_PT_RUN_ROOT/table15/$source/train/checkpoints/episode_0064")
done
"$CSD_PT_PYTHON" scripts/clean_self_distill/17_fixed_prefix_policy_drift.py \
  --model "$CSD_PT_MODEL_LOCAL_DIR" \
  --queries "$CSD_PT_PREPARED_ROOT/heldout_queries.jsonl" \
  --base-prediction "$BASE_ROOT/shard_00.jsonl" \
  --base-prediction "$BASE_ROOT/shard_01.jsonl" \
  --base-prediction "$BASE_ROOT/shard_02.jsonl" \
  --base-prediction "$BASE_ROOT/shard_03.jsonl" \
  "${METHOD_ARGS[@]}" \
  --max-queries 32 \
  --response-tokens 512 \
  --stride 32 \
  --bootstrap-draws 5000 \
  --output-dir "$CSD_PT_RUN_ROOT/table15/drift"
