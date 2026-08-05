"""Static guarantees for the two-phase, no-dependency time-box evaluation."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/clean_self_distill/slurm/submit_timebox_main_eval.sh"
EVAL = ROOT / "scripts/clean_self_distill/slurm/timebox_main_eval.slurm"
HORIZON_SUBMIT = (
    ROOT / "scripts/clean_self_distill/slurm/submit_timebox_horizon_eval.sh"
)
HORIZON_EVAL = ROOT / "scripts/clean_self_distill/slurm/timebox_horizon_eval.slurm"
HORIZON_SCORE = ROOT / "scripts/clean_self_distill/score_timebox_horizon.sh"


def test_timebox_eval_scripts_are_valid_bash() -> None:
    result = subprocess.run(
        [
            "bash",
            "-n",
            str(SUBMIT),
            str(EVAL),
            str(HORIZON_SUBMIT),
            str(HORIZON_EVAL),
            str(HORIZON_SCORE),
        ],
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


def test_horizon_array_covers_both_branches_and_three_intermediate_checkpoints() -> None:
    submit = HORIZON_SUBMIT.read_text(encoding="utf-8")
    evaluation = HORIZON_EVAL.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-23%4" in evaluation
    assert "CSD_CHECKPOINTS=(16 32 48)" in evaluation
    assert "CSD_METHOD=clean_sd" in evaluation
    assert "CSD_METHOD=privileged_sd" in evaluation
    assert "--sample-count 1" in evaluation
    assert "--max-new-tokens 4096" in evaluation
    assert 'CSD_DEST="$CSD_RUN_ROOT/timebox12h/horizon_eval"' in evaluation
    assert "for CSD_BRANCH in clean privileged" in submit
    assert "for CSD_EPISODE in 0016 0032 0048" in submit
    assert "--dependency" not in submit
