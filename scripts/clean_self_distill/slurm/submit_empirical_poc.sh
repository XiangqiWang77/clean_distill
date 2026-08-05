#!/usr/bin/env bash
# Submit the adopted Qwen3-8B empirical PoC as one strictly sequential DAG.
# Set DRY_RUN=1 to print the exact sbatch commands without submitting.
set -Eeuo pipefail
umask 027

CSD_SOURCE_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CSD_CONFIG_SOURCE="$CSD_SOURCE_REPO_ROOT/configs/clean_self_distill/empirical_poc.env"
# shellcheck disable=SC1090
source "$CSD_CONFIG_SOURCE"

CSD_GIT_COMMIT=$(git -C "$CSD_SOURCE_REPO_ROOT" rev-parse HEAD)
[[ "$CSD_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]]
if [[ -n "$(git -C "$CSD_SOURCE_REPO_ROOT" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Formal empirical submission requires a clean committed worktree" >&2
  exit 2
fi

# Fail closed if a legacy/smoke configuration is accidentally sourced.
[[ "$CSD_MODEL_ID" == Qwen/Qwen3-8B ]]
[[ "$CSD_MODEL_REVISION" == b968826d9c46dd6066d109eabc6255188de91218 ]]
[[ "$CSD_ACCOUNT" == pi_mg269 ]]
[[ "$CSD_GPU_PARTITION" == gpu_h100 ]]
[[ "$CSD_GPU_GRES" == gpu:h100:1 ]]
[[ "$CSD_GPU_NAME_REGEX" == H100 ]]
(( CSD_GPU_CAPABILITY_MAJOR == 9 ))
(( CSD_GPU_CAPABILITY_MINOR == 0 ))
[[ "$CSD_GPU_ARCH_FLAG" == sm_90 ]]
[[ "$CSD_PREP_PARTITION" == gpu_rtx6000 ]]
[[ "$CSD_PREP_GRES" == gpu:rtx_pro_6000_blackwell:1 ]]
[[ "$CSD_CONDA_ENV" == TTT ]]
[[ "$CSD_TTT_PYTHON" == /home/da839/.conda/envs/TTT/bin/python ]]
[[ -x "$CSD_TTT_PYTHON" ]]
[[ -d "$CSD_TORCH_OVERLAY/torch" ]]
[[ "$CSD_SCRATCH_ROOT" == /home/da839/scratch_pi_mg269/da839/clean_distill ]]
(( CSD_MAX_NEW_DOWNLOAD_BYTES <= 20000000000 ))
(( CSD_MAX_TASK_SCRATCH_BYTES <= 100000000000 ))
(( CSD_MAX_CONCURRENT_GPUS == 4 ))
(( CSD_DISTILL_EPISODES == 1000 ))
(( CSD_DEV_EPISODES == 200 ))
[[ "$CSD_SCIENTIFIC_CHECKPOINTS" == 0,250,500,750,1000 ]]
(( CSD_TRAIN_MAX_SEQUENCE_TOKENS == 16384 ))
(( CSD_EVAL_MAX_NEW_TOKENS == 32768 ))
(( CSD_CONTEXT_WINDOW_TOKENS == 40960 ))
(( CSD_EVAL_SAMPLES == 4 ))
[[ "$CSD_EVAL_TEMPERATURE" == 0.6 ]]
[[ "$CSD_EVAL_TOP_P" == 0.95 ]]
(( CSD_EVAL_TOP_K == 20 ))
[[ "$CSD_FRONTIER_TARGET_MARGIN" == 1.0 ]]
(( CSD_PROPOSAL_NUM_SHARDS == 36 ))
(( CSD_EVAL_NUM_SHARDS == 16 ))
[[ "$CSD_PREP_WALLTIME" == 01:00:00 ]]
[[ "$CSD_PROPOSAL_WALLTIME" == 03:00:00 ]]
[[ "$CSD_PERSISTENT_WALLTIME" == 03:00:00 ]]
[[ "$CSD_EVAL_WALLTIME" == 03:00:00 ]]
[[ "$CSD_REPORT_WALLTIME" == 01:00:00 ]]

case "$(readlink -f "$CSD_MODEL_LOCAL_DIR")" in
  "$(readlink -f "$CSD_SCRATCH_ROOT")"/*) ;;
  *) echo "Pinned model is outside da839 task scratch" >&2; exit 2 ;;
esac
for path in "$CSD_DEEPMATH_LOCAL_PATH" "$CSD_HELDOUT_LOCAL_PATH"; do
  case "$(readlink -f "$path")" in
    "$(readlink -f "$CSD_SCRATCH_ROOT")"/*) ;;
    *) echo "Dataset is outside da839 task scratch: $path" >&2; exit 2 ;;
  esac
done

CSD_RUN_ID=${RUN_ID:-empirical-poc-$(date +%Y%m%d-%H%M%S)}
CSD_DRY_RUN=${DRY_RUN:-0}
[[ "$CSD_RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "$CSD_DRY_RUN" == 0 || "$CSD_DRY_RUN" == 1 ]]
CSD_RUN_ROOT="$CSD_SCRATCH_ROOT/runs/$CSD_RUN_ID"
CSD_PREPARED_ROOT="$CSD_RUN_ROOT/prepared"
CSD_RUN_CONFIG="$CSD_RUN_ROOT/config/run.env"
CSD_JOBS_FILE="$CSD_RUN_ROOT/config/jobs.env"
CSD_PARTIAL_JOBS_FILE="$CSD_RUN_ROOT/config/jobs.partial.env"
mkdir -p "$CSD_RUN_ROOT"/{config,logs,status,tmp,hf}

csd_code_tree_hash() {
  local root=$1
  (
    cd "$root"
    tar --format=gnu --sort=name --mtime='UTC 1970-01-01' \
      --owner=0 --group=0 --numeric-owner \
      --exclude='./.csd-commit' --exclude='./.csd-tree-sha256' \
      -cf - . | sha256sum | awk '{print $1}'
  )
}

# Queue latency must not bind jobs to a mutable shared checkout.  Archive the
# exact clean commit into scratch, then pin a deterministic hash of every
# archived path (including modes and symlink targets) beside the commit marker.
CSD_CODE_ROOT="$CSD_RUN_ROOT/code"
if [[ -e "$CSD_CODE_ROOT" ]]; then
  [[ -d "$CSD_CODE_ROOT" ]]
  [[ "$(cat "$CSD_CODE_ROOT/.csd-commit")" == "$CSD_GIT_COMMIT" ]]
  CSD_CODE_TREE_SHA256=$(cat "$CSD_CODE_ROOT/.csd-tree-sha256")
  [[ "$CSD_CODE_TREE_SHA256" =~ ^[0-9a-f]{64}$ ]]
  [[ "$(csd_code_tree_hash "$CSD_CODE_ROOT")" == "$CSD_CODE_TREE_SHA256" ]]
else
  CSD_CODE_TEMP="$CSD_RUN_ROOT/.code.tmp.$$"
  mkdir "$CSD_CODE_TEMP"
  git -C "$CSD_SOURCE_REPO_ROOT" archive --format=tar "$CSD_GIT_COMMIT" | \
    tar -xf - -C "$CSD_CODE_TEMP"
  printf '%s\n' "$CSD_GIT_COMMIT" > "$CSD_CODE_TEMP/.csd-commit"
  : > "$CSD_CODE_TEMP/.csd-tree-sha256"
  chmod -R a-w "$CSD_CODE_TEMP"
  CSD_CODE_TREE_SHA256=$(csd_code_tree_hash "$CSD_CODE_TEMP")
  chmod u+w "$CSD_CODE_TEMP/.csd-tree-sha256"
  printf '%s\n' "$CSD_CODE_TREE_SHA256" > "$CSD_CODE_TEMP/.csd-tree-sha256"
  chmod 0444 "$CSD_CODE_TEMP/.csd-commit" "$CSD_CODE_TEMP/.csd-tree-sha256"
  mv "$CSD_CODE_TEMP" "$CSD_CODE_ROOT"
fi
CSD_REPO_ROOT="$CSD_CODE_ROOT"
CSD_CONFIG_SOURCE="$CSD_CODE_ROOT/configs/clean_self_distill/empirical_poc.env"

# Serialize every protocol variable into a shell-quoted, immutable run file.
# Formal jobs source this file only and verify the pinned archive marker/hash.
CSD_CONFIG_TMP="${CSD_RUN_CONFIG}.tmp.$$"
{
  mapfile -t CSD_CONFIG_NAMES < <(compgen -A variable CSD_ | sort -u)
  for name in "${CSD_CONFIG_NAMES[@]}"; do
    case "$name" in
      CSD_CODE_TEMP|CSD_CONFIG_NAMES|CSD_CONFIG_TMP|CSD_DRY_RUN|CSD_JOBS_FILE|CSD_PARTIAL_JOBS_FILE|CSD_SOURCE_REPO_ROOT) continue ;;
    esac
    if [[ -v "$name" && "$(declare -p "$name" 2>/dev/null)" != declare\ -a* ]]; then
      printf '%s=%q\n' "$name" "${!name}"
    fi
  done
} > "$CSD_CONFIG_TMP"
if [[ -f "$CSD_RUN_CONFIG" ]]; then
  if ! cmp -s "$CSD_CONFIG_TMP" "$CSD_RUN_CONFIG"; then
    rm -f "$CSD_CONFIG_TMP"
    echo "RUN_ID=$CSD_RUN_ID already has a different immutable configuration" >&2
    exit 2
  fi
  rm -f "$CSD_CONFIG_TMP"
else
  mv "$CSD_CONFIG_TMP" "$CSD_RUN_CONFIG"
  chmod 0440 "$CSD_RUN_CONFIG"
fi
if [[ -e "$CSD_JOBS_FILE" ]]; then
  echo "RUN_ID=$CSD_RUN_ID already has a submitted DAG in $CSD_JOBS_FILE" >&2
  echo "The jobs are restart-safe; use Slurm requeue rather than duplicating the DAG." >&2
  exit 2
fi
if [[ -e "$CSD_PARTIAL_JOBS_FILE" ]]; then
  echo "A prior launcher stopped after submitting part of the DAG: $CSD_PARTIAL_JOBS_FILE" >&2
  echo "Inspect those recorded job IDs before deciding whether any new submission is safe." >&2
  exit 2
fi

csd_submit() {
  local dry_id=$1
  shift
  if [[ "$CSD_DRY_RUN" == 1 ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    printf '%s\n' "$dry_id"
    return
  fi
  local result
  result=$("$@")
  result=${result%%;*}
  [[ "$result" =~ ^[0-9]+$ ]] || {
    echo "Unexpected sbatch output: $result" >&2
    return 1
  }
  printf '%s\n' "$result"
}

CSD_EXPORT="ALL,CSD_RUN_CONFIG=$CSD_RUN_CONFIG"
CSD_SLURM_ROOT="$CSD_REPO_ROOT/scripts/clean_self_distill/slurm"
if [[ "$CSD_DRY_RUN" == 0 ]]; then
  {
    printf 'CSD_RUN_ID=%q\n' "$CSD_RUN_ID"
    printf 'CSD_RUN_ROOT=%q\n' "$CSD_RUN_ROOT"
    printf 'CSD_GIT_COMMIT=%q\n' "$CSD_GIT_COMMIT"
    printf 'CSD_CODE_TREE_SHA256=%q\n' "$CSD_CODE_TREE_SHA256"
  } > "$CSD_PARTIAL_JOBS_FILE"
fi

# GPU phases are deliberately non-overlapping.  Combined with each array's
# throttle, the complete experiment can never consume more than four H100s.
CSD_PREP_JOB_ID=$(csd_submit 900001 \
  sbatch --parsable \
  --account "$CSD_ACCOUNT" --partition "$CSD_PREP_PARTITION" \
  --time "$CSD_PREP_WALLTIME" --export "$CSD_EXPORT" \
  --output "$CSD_RUN_ROOT/logs/prep-%j.out" \
  --error "$CSD_RUN_ROOT/logs/prep-%j.err" \
  "$CSD_SLURM_ROOT/empirical_prep.slurm")
[[ "$CSD_DRY_RUN" == 1 ]] || printf 'CSD_PREP_JOB_ID=%q\n' "$CSD_PREP_JOB_ID" >> "$CSD_PARTIAL_JOBS_FILE"

CSD_PROPOSAL_JOB_ID=$(csd_submit 900002 \
  sbatch --parsable --dependency "afterok:$CSD_PREP_JOB_ID" \
  --account "$CSD_ACCOUNT" --partition "$CSD_GPU_PARTITION" \
  --gres "$CSD_GPU_GRES" \
  --time "$CSD_PROPOSAL_WALLTIME" \
  --array "0-$((CSD_PROPOSAL_NUM_SHARDS - 1))%$CSD_MAX_CONCURRENT_GPUS" \
  --export "$CSD_EXPORT" \
  --output "$CSD_RUN_ROOT/logs/propose-%A_%a.out" \
  --error "$CSD_RUN_ROOT/logs/propose-%A_%a.err" \
  "$CSD_SLURM_ROOT/empirical_propose.slurm")
[[ "$CSD_DRY_RUN" == 1 ]] || printf 'CSD_PROPOSAL_JOB_ID=%q\n' "$CSD_PROPOSAL_JOB_ID" >> "$CSD_PARTIAL_JOBS_FILE"

CSD_MERGE_JOB_ID=$(csd_submit 900003 \
  sbatch --parsable --dependency "afterok:$CSD_PROPOSAL_JOB_ID" \
  --account "$CSD_ACCOUNT" --partition "$CSD_PREP_PARTITION" \
  --time "$CSD_PREP_WALLTIME" --export "$CSD_EXPORT" \
  --output "$CSD_RUN_ROOT/logs/merge-%j.out" \
  --error "$CSD_RUN_ROOT/logs/merge-%j.err" \
  "$CSD_SLURM_ROOT/empirical_merge.slurm")
[[ "$CSD_DRY_RUN" == 1 ]] || printf 'CSD_MERGE_JOB_ID=%q\n' "$CSD_MERGE_JOB_ID" >> "$CSD_PARTIAL_JOBS_FILE"

CSD_DEV_AUDIT_JOB_ID=$(csd_submit 900004 \
  sbatch --parsable --dependency "afterok:$CSD_MERGE_JOB_ID" \
  --account "$CSD_ACCOUNT" --partition "$CSD_PREP_PARTITION" \
  --time "$CSD_PREP_WALLTIME" --export "$CSD_EXPORT" \
  --output "$CSD_RUN_ROOT/logs/dev-audit-%j.out" \
  --error "$CSD_RUN_ROOT/logs/dev-audit-%j.err" \
  "$CSD_SLURM_ROOT/empirical_dev_audit.slurm")
[[ "$CSD_DRY_RUN" == 1 ]] || printf 'CSD_DEV_AUDIT_JOB_ID=%q\n' "$CSD_DEV_AUDIT_JOB_ID" >> "$CSD_PARTIAL_JOBS_FILE"

CSD_SHORT_TASKS=$((5 * CSD_EVAL_NUM_SHARDS))
CSD_SHORT_JOB_ID=$(csd_submit 900005 \
  sbatch --parsable --dependency "afterok:$CSD_DEV_AUDIT_JOB_ID" \
  --account "$CSD_ACCOUNT" --partition "$CSD_GPU_PARTITION" \
  --gres "$CSD_GPU_GRES" \
  --time "$CSD_EVAL_WALLTIME" \
  --array "0-$((CSD_SHORT_TASKS - 1))%$CSD_MAX_CONCURRENT_GPUS" \
  --export "$CSD_EXPORT" \
  --output "$CSD_RUN_ROOT/logs/short-%A_%a.out" \
  --error "$CSD_RUN_ROOT/logs/short-%A_%a.err" \
  "$CSD_SLURM_ROOT/empirical_short.slurm")
[[ "$CSD_DRY_RUN" == 1 ]] || printf 'CSD_SHORT_JOB_ID=%q\n' "$CSD_SHORT_JOB_ID" >> "$CSD_PARTIAL_JOBS_FILE"

CSD_PERSISTENT_JOB_ID=$(csd_submit 900006 \
  sbatch --parsable --dependency "afterok:$CSD_SHORT_JOB_ID" \
  --account "$CSD_ACCOUNT" --partition "$CSD_GPU_PARTITION" \
  --gres "$CSD_GPU_GRES" \
  --time "$CSD_PERSISTENT_WALLTIME" \
  --array "0-1%$CSD_MAX_CONCURRENT_GPUS" \
  --export "$CSD_EXPORT" \
  --output "$CSD_RUN_ROOT/logs/persistent-%A_%a.out" \
  --error "$CSD_RUN_ROOT/logs/persistent-%A_%a.err" \
  "$CSD_SLURM_ROOT/empirical_persistent.slurm")
[[ "$CSD_DRY_RUN" == 1 ]] || printf 'CSD_PERSISTENT_JOB_ID=%q\n' "$CSD_PERSISTENT_JOB_ID" >> "$CSD_PARTIAL_JOBS_FILE"

CSD_LONG_EVAL_TASKS=$((2 * 4 * CSD_EVAL_NUM_SHARDS))
CSD_LONG_EVAL_JOB_ID=$(csd_submit 900007 \
  sbatch --parsable --dependency "afterok:$CSD_PERSISTENT_JOB_ID" \
  --account "$CSD_ACCOUNT" --partition "$CSD_GPU_PARTITION" \
  --gres "$CSD_GPU_GRES" \
  --time "$CSD_EVAL_WALLTIME" \
  --array "0-$((CSD_LONG_EVAL_TASKS - 1))%$CSD_MAX_CONCURRENT_GPUS" \
  --export "$CSD_EXPORT" \
  --output "$CSD_RUN_ROOT/logs/long-eval-%A_%a.out" \
  --error "$CSD_RUN_ROOT/logs/long-eval-%A_%a.err" \
  "$CSD_SLURM_ROOT/empirical_eval_persistent.slurm")
[[ "$CSD_DRY_RUN" == 1 ]] || printf 'CSD_LONG_EVAL_JOB_ID=%q\n' "$CSD_LONG_EVAL_JOB_ID" >> "$CSD_PARTIAL_JOBS_FILE"

CSD_MECHANISM_TASKS=$((2 * CSD_EVAL_NUM_SHARDS))
CSD_MECHANISM_JOB_ID=$(csd_submit 900008 \
  sbatch --parsable --dependency "afterok:$CSD_LONG_EVAL_JOB_ID" \
  --account "$CSD_ACCOUNT" --partition "$CSD_GPU_PARTITION" \
  --gres "$CSD_GPU_GRES" \
  --time "$CSD_EVAL_WALLTIME" \
  --array "0-$((CSD_MECHANISM_TASKS - 1))%$CSD_MAX_CONCURRENT_GPUS" \
  --export "$CSD_EXPORT" \
  --output "$CSD_RUN_ROOT/logs/mechanism-%A_%a.out" \
  --error "$CSD_RUN_ROOT/logs/mechanism-%A_%a.err" \
  "$CSD_SLURM_ROOT/empirical_mechanism.slurm")
[[ "$CSD_DRY_RUN" == 1 ]] || printf 'CSD_MECHANISM_JOB_ID=%q\n' "$CSD_MECHANISM_JOB_ID" >> "$CSD_PARTIAL_JOBS_FILE"

CSD_REPORT_JOB_ID=$(csd_submit 900009 \
  sbatch --parsable --dependency "afterok:$CSD_MECHANISM_JOB_ID" \
  --account "$CSD_ACCOUNT" --partition "$CSD_PREP_PARTITION" \
  --time "$CSD_REPORT_WALLTIME" --export "$CSD_EXPORT" \
  --output "$CSD_RUN_ROOT/logs/report-%j.out" \
  --error "$CSD_RUN_ROOT/logs/report-%j.err" \
  "$CSD_SLURM_ROOT/empirical_report.slurm")
if [[ "$CSD_DRY_RUN" == 0 ]]; then
  {
    printf 'CSD_REPORT_JOB_ID=%q\n' "$CSD_REPORT_JOB_ID"
    printf 'CSD_GPU_PARTITION=%q\n' "$CSD_GPU_PARTITION"
    printf 'CSD_GPU_GRES=%q\n' "$CSD_GPU_GRES"
    printf 'CSD_GPU_NAME_REGEX=%q\n' "$CSD_GPU_NAME_REGEX"
    printf 'CSD_GPU_CAPABILITY_MAJOR=%q\n' "$CSD_GPU_CAPABILITY_MAJOR"
    printf 'CSD_GPU_CAPABILITY_MINOR=%q\n' "$CSD_GPU_CAPABILITY_MINOR"
    printf 'CSD_GPU_ARCH_FLAG=%q\n' "$CSD_GPU_ARCH_FLAG"
    printf 'CSD_MAX_CONCURRENT_GPUS=%q\n' "$CSD_MAX_CONCURRENT_GPUS"
    printf 'CSD_DAG_ORDER=%q\n' 'prep>proposal>merge>dev_audit>short>persistent>long_eval>mechanism>report'
    printf 'CSD_SUBMITTED_AT=%q\n' "$(date -Is)"
  } >> "$CSD_PARTIAL_JOBS_FILE"
  mv "$CSD_PARTIAL_JOBS_FILE" "$CSD_JOBS_FILE"
  chmod 0440 "$CSD_JOBS_FILE"
else
  CSD_JOBS_FILE='DRY_RUN_NOT_WRITTEN'
fi

printf 'run_id=%s\nrun_root=%s\njob_file=%s\n' \
  "$CSD_RUN_ID" "$CSD_RUN_ROOT" "$CSD_JOBS_FILE"
printf 'prep=%s proposal=%s merge=%s dev_audit=%s short=%s persistent=%s long_eval=%s mechanism=%s report=%s\n' \
  "$CSD_PREP_JOB_ID" "$CSD_PROPOSAL_JOB_ID" "$CSD_MERGE_JOB_ID" \
  "$CSD_DEV_AUDIT_JOB_ID" "$CSD_SHORT_JOB_ID" "$CSD_PERSISTENT_JOB_ID" "$CSD_LONG_EVAL_JOB_ID" \
  "$CSD_MECHANISM_JOB_ID" "$CSD_REPORT_JOB_ID"
