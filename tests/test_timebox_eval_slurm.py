from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_h100_table_contains_only_reported_methods() -> None:
    script = (ROOT / "scripts/clean_self_distill/slurm/trust_region_checkpoint_eval.slurm").read_text()
    assert "CSD_METHOD=trsd" in script
    assert "CSD_METHOD=privileged_sd" in script
    assert "CSD_METHOD=base" in script
    assert "#SBATCH --gres=gpu:h100:1" in script
    assert "#SBATCH --array=0-19%4" in script

