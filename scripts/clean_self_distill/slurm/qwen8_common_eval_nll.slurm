#!/usr/bin/env bash
#SBATCH --job-name=q8-common-nll
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=160G
#SBATCH --time=11:50:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

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
export HF_HOME="$CSD_I8_RUN_ROOT/hf" HF_HUB_CACHE="$CSD_I8_RUN_ROOT/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/q8_common_nll_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$CSD_I8_DYNAMICS_ROOT"

[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]]
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi H100
if [[ -s "$CSD_I8_DYNAMICS_ROOT/DYNAMICS_COMPLETE" ]] &&
   [[ -s "$CSD_I8_DYNAMICS_ROOT/common_evaluation_nll_rebound.pdf" ]]; then
  printf 'dynamics already complete; verified outputs and exiting\n'
  exit 0
fi
"$CSD_I8_PYTHON" - <<'PY'
import torch
assert str(torch.__version__).endswith("+cu126"), torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
print(f"dynamics_torch={torch.__version__} cuda={torch.version.cuda}")
PY

replay="$CSD_I8_DYNAMICS_ROOT/trsd_replay"
resume=()
[[ ! -s "$replay/run_manifest.json" ]] || resume=(--resume)
interrupt_sentinel="$TMPDIR/job_started"
touch "$interrupt_sentinel"
set +e
"$CSD_I8_PYTHON" scripts/clean_self_distill/26_common_eval_nll_dynamics.py \
  --queries "$CSD_I8_TRSD_QUERIES" \
  --privileged-journal "$CSD_I8_PRIVILEGED_JOURNAL" \
  --original-trsd-checkpoint "$CSD_I8_TRSD_TRAIN_ROOT/checkpoints/episode_0064" \
  --expected-trsd-adapter-sha256 "$CSD_I8_TRSD64_ADAPTER_SHA256" \
  --model "$CSD_I8_MODEL_LOCAL_DIR" \
  --model-id "$CSD_I8_MODEL_ID" \
  --revision "$CSD_I8_MODEL_REVISION" \
  --replay-dir "$replay" \
  --output-dir "$CSD_I8_DYNAMICS_ROOT" \
  "${resume[@]}"
rc=$?
set -e

# Persistent training handles USR1 only at an episode-safe boundary, writes a
# durable restart checkpoint, then records INTERRUPTED.json.  Requeue only for
# a marker created during this allocation; a stale marker plus an unrelated
# import/logic failure must remain visible instead of looping forever.
if (( rc != 0 )) &&
   [[ -s "$replay/INTERRUPTED.json" ]] &&
   [[ "$replay/INTERRUPTED.json" -nt "$interrupt_sentinel" ]] &&
   [[ ! -s "$CSD_I8_DYNAMICS_ROOT/DYNAMICS_COMPLETE" ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
(( rc == 0 ))

[[ -s "$CSD_I8_DYNAMICS_ROOT/DYNAMICS_COMPLETE" ]]
[[ -s "$CSD_I8_DYNAMICS_ROOT/common_evaluation_nll_rebound.pdf" ]]
