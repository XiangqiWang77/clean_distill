"""Static guarantees for the two-phase, no-dependency time-box evaluation."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/clean_self_distill/slurm/submit_timebox_main_eval.sh"
EVAL = ROOT / "scripts/clean_self_distill/slurm/timebox_main_eval.slurm"


def test_timebox_eval_scripts_are_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SUBMIT), str(EVAL)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_submitter_exposes_nonoverlapping_early_and_final_phases() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    evaluation = EVAL.read_text(encoding="utf-8")

    assert "CSD_ARRAY=0-7%2" in submit
    assert "CSD_ARRAY=8-15%4" in submit
    assert "CSD_ARRAY=0-15%4" in submit
    assert 'sbatch --parsable --export=ALL --array="$CSD_ARRAY"' in submit
    assert "csd_merge_probes\n    CSD_ARRAY=0-7%2" in submit
    assert "csd_require_final_checkpoints\n    CSD_ARRAY=8-15%4" in submit
    assert "CSD_METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / CSD_EVAL_SHARDS))" in evaluation

