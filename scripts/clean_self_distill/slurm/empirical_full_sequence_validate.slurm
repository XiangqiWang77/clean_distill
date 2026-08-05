#!/usr/bin/env bash
#SBATCH --job-name=csd8b-fullseq
#SBATCH --account=pi_mg269
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --requeue

# Exact-cap Qwen3-8B forward/backward validation; deliberately not a smoke test.
set -Eeuo pipefail

: "${CSD_RUN_CONFIG:?The empirical submitter must export CSD_RUN_CONFIG}"
# shellcheck disable=SC1090
source "$CSD_RUN_CONFIG"
# shellcheck disable=SC1091
source "$CSD_REPO_ROOT/scripts/clean_self_distill/slurm/empirical_common.sh"
csd_assert_model_gpu

CSD_OUTPUT="$CSD_RUN_ROOT/validation/full_sequence_h100.json"
CSD_MARKER="$CSD_RUN_ROOT/status/full_sequence_h100.done"
CSD_VALIDATION_ARGS=(
  --queries "$CSD_PREPARED_ROOT/dev_queries.jsonl"
  --run-config "$CSD_RUN_CONFIG"
  --git-commit "$CSD_GIT_COMMIT"
  --code-tree-sha256 "$CSD_CODE_TREE_SHA256"
  --model "$CSD_MODEL_LOCAL_DIR"
  --model-id "$CSD_MODEL_ID"
  --revision "$CSD_MODEL_REVISION"
  --output "$CSD_OUTPUT"
  --max-sequence-tokens "$CSD_TRAIN_MAX_SEQUENCE_TOKENS"
  --distill-token-chunk-size "$CSD_DISTILL_TOKEN_CHUNK_SIZE"
  --distill-top-k "$CSD_DISTILL_TOP_K"
  --distill-temperature "$CSD_DISTILL_TEMPERATURE"
  --lora-rank "$CSD_LORA_RANK"
  --lora-alpha "$CSD_LORA_ALPHA"
  --expected-gpu-name-regex "$CSD_GPU_NAME_REGEX"
  --expected-gpu-capability-major "$CSD_GPU_CAPABILITY_MAJOR"
  --expected-gpu-capability-minor "$CSD_GPU_CAPABILITY_MINOR"
  --expected-gpu-arch-flag "$CSD_GPU_ARCH_FLAG"
  --minimum-reserved-headroom-bytes "$CSD_FULL_SEQUENCE_MIN_HEADROOM_BYTES"
)
if [[ -s "$CSD_MARKER" ]]; then
  [[ -s "$CSD_OUTPUT" ]]
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/10_full_sequence_memory_validation.py \
    "${CSD_VALIDATION_ARGS[@]}" --validate-existing
  exit 0
fi
mkdir -p "$(dirname "$CSD_OUTPUT")"

# The submitted code snapshot is intentionally read-only.  Route bytecode from
# both the explicit compile check and pytest imports to node-local storage so
# validation never attempts to create __pycache__ inside the immutable archive.
CSD_PYCACHE_ROOT="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/csd-pycache-${SLURM_JOB_ID}.XXXXXX")"
export PYTHONPYCACHEPREFIX="$CSD_PYCACHE_ROOT"
trap 'rm -rf -- "$CSD_PYCACHE_ROOT"' EXIT

# Run the complete code/protocol suite on this same high-memory H100 node.
# The immutable archive contains no .git directory, so enumerate its files
# directly; empirical_common.sh has already verified the full tree hash.
mapfile -t CSD_PYTHON_FILES < <(find . -type f -name '*.py' | sort)
(( ${#CSD_PYTHON_FILES[@]} > 0 ))
"$CSD_TTT_PYTHON" -m py_compile "${CSD_PYTHON_FILES[@]}"
mapfile -t CSD_SHELL_FILES < <(find . -type f \( -name '*.sh' -o -name '*.slurm' \) | sort)
(( ${#CSD_SHELL_FILES[@]} > 0 ))
for path in "${CSD_SHELL_FILES[@]}"; do
  bash -n "$path"
done
"$CSD_TTT_PYTHON" -m pytest -q -p no:cacheprovider

"$CSD_TTT_PYTHON" scripts/clean_self_distill/10_full_sequence_memory_validation.py \
  "${CSD_VALIDATION_ARGS[@]}"

[[ -s "$CSD_OUTPUT" ]]
"$CSD_TTT_PYTHON" scripts/clean_self_distill/10_full_sequence_memory_validation.py \
  "${CSD_VALIDATION_ARGS[@]}" --validate-existing
csd_atomic_marker "$CSD_MARKER"
