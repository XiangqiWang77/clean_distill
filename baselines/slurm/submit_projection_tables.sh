#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/da839/clean_distill
CONFIG=${CSD_PT_CONFIG:-$ROOT/configs/clean_self_distill/qwen3_8b_projection_tables.env}
VALIDITY_JOB=${1:?usage: submit_projection_tables.sh VALIDITY_JOB_ID}
# shellcheck disable=SC1090
source "$CONFIG"
cd "$ROOT"
mkdir -p "$CSD_PT_RUN_ROOT/logs"

export_arg="ALL,CSD_PT_CONFIG=$CONFIG"
train14=$(sbatch --parsable --dependency="afterok:$VALIDITY_JOB" --export="$export_arg" \
  baselines/slurm/train_projection64.slurm)
math14=$(sbatch --parsable --dependency="afterok:$train14" --export="$export_arg" \
  baselines/slurm/eval_projection64_math.slurm)
logic14=$(sbatch --parsable --dependency="afterok:$math14" --export="$export_arg" \
  baselines/slurm/eval_projection64_logic.slurm)
score14=$(sbatch --parsable --dependency="afterok:$math14" --export="$export_arg" \
  baselines/slurm/score_projection64_math.slurm)
drift14=$(sbatch --parsable --dependency="afterok:$logic14" --export="$export_arg" \
  baselines/slurm/drift_projection64.slurm)

cost16=$(sbatch --parsable --dependency="afterok:$drift14" --export="$export_arg" \
  baselines/slurm/train_cost64.slurm)
source15=$(sbatch --parsable --dependency="afterok:$cost16" --export="$export_arg" \
  baselines/slurm/train_privilege_sources64.slurm)
source_eval=$(sbatch --parsable --dependency="afterok:$source15" --export="$export_arg" \
  baselines/slurm/eval_privilege_sources64_math.slurm)
source_score=$(sbatch --parsable --dependency="afterok:$source_eval" --export="$export_arg" \
  baselines/slurm/score_privilege_sources64_math.slurm)
source_drift=$(sbatch --parsable --dependency="afterok:$source_eval" --export="$export_arg" \
  baselines/slurm/drift_privilege_sources64.slurm)
source_wrapper=$(sbatch --parsable --dependency="afterok:$source_eval" --export="$export_arg" \
  baselines/slurm/wrapper_privilege_sources64.slurm)

collect=$(sbatch --parsable \
  --dependency="afterok:$score14:$drift14:$cost16:$source_score:$source_drift:$source_wrapper" \
  --export="$export_arg" baselines/slurm/collect_projection_tables.slurm)

"$CSD_PT_PYTHON" - "$CSD_PT_RUN_ROOT/SUBMISSION.json" <<PY
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

jobs = {
    "validity": "$VALIDITY_JOB",
    "table14_train": "$train14",
    "table14_math_eval": "$math14",
    "table14_logic_eval": "$logic14",
    "table14_math_score": "$score14",
    "table14_drift": "$drift14",
    "table16_cost_train": "$cost16",
    "table15_source_train": "$source15",
    "table15_math_eval": "$source_eval",
    "table15_math_score": "$source_score",
    "table15_drift": "$source_drift",
    "table15_wrapper_variance": "$source_wrapper",
    "collect": "$collect",
}
Path(sys.argv[1]).write_text(json.dumps({
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "gpu_concurrency_cap": 2,
    "jobs": jobs,
}, indent=2) + "\n")
print(json.dumps(jobs, indent=2))
PY
