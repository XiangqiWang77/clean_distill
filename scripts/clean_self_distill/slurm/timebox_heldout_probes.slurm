#!/usr/bin/env bash
#SBATCH --job-name=heldout-probes-12h
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --requeue

set -Eeuo pipefail
: "${SLURM_ARRAY_TASK_ID:?submit as a two-shard array}"
: "${CSD_RUN_CONFIG:?export CSD_RUN_CONFIG}"
: "${CSD_ONLINE_CODE_ROOT:?export CSD_ONLINE_CODE_ROOT}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CSD_CONDA_ENV"
cd "$CSD_ONLINE_CODE_ROOT"
export PYTHONPATH="$CSD_TORCH_OVERLAY:$CSD_ONLINE_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CSD_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$CSD_RUN_ROOT/tmp/heldout_probe_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | rg -i H100 >/dev/null

CSD_SHARDS=2
CSD_LABEL=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
CSD_DEST="$CSD_RUN_ROOT/timebox12h/heldout_probes"
CSD_OUTPUT="$CSD_DEST/shard_${CSD_LABEL}.jsonl"
CSD_DONE="$CSD_DEST/shard_${CSD_LABEL}.done"
mkdir -p "$CSD_DEST"
[[ ! -s "$CSD_DONE" ]] || exit 0
"$CSD_TTT_PYTHON" scripts/clean_self_distill/slurm/launcher_support.py \
  repair-proposals --path "$CSD_OUTPUT"

# Requeue before the allocation expires; every completed query is already a
# durable JSONL prefix and the next allocation resumes it.
( sleep 10200; scontrol requeue "$SLURM_JOB_ID" ) &
CSD_WATCHDOG=$!
trap 'kill "$CSD_WATCHDOG" 2>/dev/null || true' EXIT

"$CSD_TTT_PYTHON" scripts/clean_self_distill/01_propose.py \
  --input "$CSD_PREPARED_ROOT/heldout_queries.jsonl" \
  --output "$CSD_OUTPUT" \
  --model "$CSD_MODEL_LOCAL_DIR" \
  --revision "$CSD_MODEL_REVISION" \
  --num-candidates "${CSD_TIMEBOX_NUM_CANDIDATES:-4}" \
  --proposal-oversample "${CSD_TIMEBOX_PROPOSAL_OVERSAMPLE:-2}" \
  --max-rounds "${CSD_TIMEBOX_PROPOSAL_ROUNDS:-2}" \
  --min-accepted-candidates "${CSD_TIMEBOX_MIN_CANDIDATES:-2}" \
  --max-new-tokens "${CSD_TIMEBOX_PROPOSAL_TOKENS:-768}" \
  --temperature 0.8 \
  --solver-temperature 0.3 \
  --verifier-temperature 0 \
  --stage-max-attempts 2 \
  --top-p 0.95 \
  --num-shards "$CSD_SHARDS" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --resume \
  --seed "$CSD_RUN_SEED"

kill "$CSD_WATCHDOG" 2>/dev/null || true
wait "$CSD_WATCHDOG" 2>/dev/null || true
CSD_WATCHDOG=
printf 'completed_at=%s\n' "$(date -Is)" > "$CSD_DONE"
