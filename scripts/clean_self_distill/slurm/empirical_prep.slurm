#!/usr/bin/env bash
#SBATCH --job-name=csd8b-prep
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

CSD_MARKER="$CSD_RUN_ROOT/status/prep.done"
if [[ -f "$CSD_MARKER" ]]; then
  [[ -s "$CSD_PREPARED_ROOT/manifest.json" ]]
  exit 0
fi

if [[ ! -s "$CSD_PREPARED_ROOT/manifest.json" ]]; then
  # The Python preparer publishes the directory atomically.  A complete
  # manifest without the tiny stage marker therefore means an earlier node
  # finished the expensive hash/stream pass and died between publication and
  # marker creation; validate and adopt it instead of refusing to overwrite.
  [[ ! -e "$CSD_PREPARED_ROOT" ]]
  "$CSD_TTT_PYTHON" scripts/clean_self_distill/prepare_empirical_data.py \
    --deepmath "$CSD_DEEPMATH_LOCAL_PATH" \
    --heldout "$CSD_HELDOUT_LOCAL_PATH" \
    --model-dir "$CSD_MODEL_LOCAL_DIR" \
    --task-root "$CSD_SCRATCH_ROOT" \
    --output-dir "$CSD_PREPARED_ROOT" \
    --deepmath-revision "$CSD_DEEPMATH_REVISION" \
    --heldout-revision "$CSD_HELDOUT_REVISION" \
    --max-new-download-bytes "$CSD_MAX_NEW_DOWNLOAD_BYTES" \
    --max-task-bytes "$CSD_MAX_TASK_SCRATCH_BYTES" \
    --batch-size 256 \
    --full-asset-hash
fi

"$CSD_TTT_PYTHON" - "$CSD_PREPARED_ROOT/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert manifest["counts"] == {
    "distill": 1000,
    "dev": 200,
    "heldout": 143,
    "heldout_by_source": {"amc23": 83, "aime24": 30, "aime25": 30},
    "proposal_queries": 1343,
}
assert manifest["overlap_audit"]["passed"] is True
assert manifest["data_firewall"]["query_files_physically_exclude_targets"] is True
assert manifest["model"] == {
    "id": "Qwen/Qwen3-8B",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
}
assert manifest["inputs"]["deepmath"]["revision"] == "33d7de919af5b03257ff92c30303fddf9afdda4a"
assert manifest["inputs"]["deepmath"]["sha256"] == "611d3030a2a74eaea9514ab732fc33aa6a35d668c7f804c383772939b159f2a0"
assert manifest["inputs"]["heldout"]["revision"] == "33d7de919af5b03257ff92c30303fddf9afdda4a"
assert manifest["inputs"]["heldout"]["sha256"] == "42e7c50d0511fb52680ae6fc6cbfc46ff6c361771378dd5c6d228acb61be1cbf"
PY

csd_atomic_marker "$CSD_MARKER"
