"""Static and resume guards for the formal empirical accelerator contract."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from src.clean_self_distill.train_eval import _validate_resume_runtime


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "clean_self_distill" / "empirical_poc.env"
SLURM = ROOT / "scripts" / "clean_self_distill" / "slurm"


def _config_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def test_formal_run_pins_one_h100_per_task_and_four_way_concurrency():
    values = _config_values()
    assert values["CSD_GPU_PARTITION"] == "gpu_h100"
    assert values["CSD_GPU_GRES"] == "gpu:h100:1"
    assert values["CSD_GPU_NAME_REGEX"] == "H100"
    assert values["CSD_GPU_CAPABILITY_MAJOR"] == "9"
    assert values["CSD_GPU_CAPABILITY_MINOR"] == "0"
    assert values["CSD_GPU_ARCH_FLAG"] == "sm_90"
    assert values["CSD_MAX_CONCURRENT_GPUS"] == "4"
    assert "CSD_MAX_CONCURRENT_B200" not in values


def test_every_formal_model_job_has_matching_typed_h100_guards():
    names = (
        "empirical_propose.slurm",
        "empirical_short.slurm",
        "empirical_persistent.slurm",
        "empirical_eval_persistent.slurm",
        "empirical_mechanism.slurm",
    )
    for name in names:
        text = (SLURM / name).read_text(encoding="utf-8")
        assert "#SBATCH --partition=gpu_h100" in text
        assert "#SBATCH --gres=gpu:h100:1" in text
        assert "csd_assert_model_gpu" in text
        assert "gpu:b200" not in text.casefold()

    common = (SLURM / "empirical_common.sh").read_text(encoding="utf-8")
    assert "csd_assert_gpu()" in common
    assert "csd_assert_model_gpu()" in common
    assert 'csd_assert_gpu "$CSD_GPU_NAME_REGEX"' in common
    assert 'torch.cuda.device_count() == 1' in common
    assert 'CSD_GPU_CAPABILITY_MAJOR' in common
    assert 'CSD_GPU_CAPABILITY_MINOR' in common
    assert 'CSD_GPU_ARCH_FLAG' in common

    for name in (
        "empirical_prep.slurm",
        "empirical_merge.slurm",
        "empirical_dev_audit.slurm",
        "empirical_report.slurm",
    ):
        text = (SLURM / name).read_text(encoding="utf-8")
        assert "csd_assert_gpu 'RTX.*6000|PRO.*6000'" in text
        assert "csd_assert_model_gpu" not in text


def test_launcher_overrides_every_model_job_and_records_hardware_contract():
    text = (SLURM / "submit_empirical_poc.sh").read_text(encoding="utf-8")
    assert text.count('--gres "$CSD_GPU_GRES"') == 5
    assert text.count("%$CSD_MAX_CONCURRENT_GPUS") == 5
    assert "CSD_MAX_CONCURRENT_B200" not in text
    for key in (
        "CSD_GPU_PARTITION",
        "CSD_GPU_GRES",
        "CSD_GPU_NAME_REGEX",
        "CSD_GPU_CAPABILITY_MAJOR",
        "CSD_GPU_CAPABILITY_MINOR",
        "CSD_GPU_ARCH_FLAG",
        "CSD_MAX_CONCURRENT_GPUS",
    ):
        assert f"printf '{key}=%q" in text


def _formal_h100_runtime() -> dict:
    overlay = "/home/da839/scratch_pi_mg269/da839/mfspd/pydeps-cu128"
    return {
        "python_executable": "/home/da839/.conda/envs/TTT/bin/python",
        "conda_prefix": "/home/da839/.conda/envs/TTT",
        "torch_overlay": overlay,
        "torch": "2.9.1+cu128",
        "torch_module_path": f"{overlay}/torch/__init__.py",
        "torch_arch_flags": ["sm_90", "sm_100"],
        "cuda_runtime": "12.8",
        "model": "Qwen/Qwen3-8B",
        "requested_model_revision": "b" * 40,
        "resolved_model_revision": "b" * 40,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "slurm_array_task_id": "3",
        "gpu_count": 1,
        "gpus": [
            {"name": "NVIDIA H100 80GB HBM3", "capability": [9, 0]}
        ],
        "slurm_job_partition": "gpu_h100",
        "requested_gpu_partition": "gpu_h100",
        "requested_gpu_gres": "gpu:h100:1",
        "expected_gpu_name_regex": "H100",
        "expected_gpu_capability": [9, 0],
        "expected_gpu_arch_flag": "sm_90",
    }


def test_formal_resume_accepts_h100_and_rejects_hardware_drift():
    runtime = _formal_h100_runtime()
    args = SimpleNamespace(
        model="Qwen/Qwen3-8B", revision="b" * 40, shard_index=3
    )
    environment = {
        "CSD_GPU_PARTITION": "gpu_h100",
        "CSD_GPU_GRES": "gpu:h100:1",
        "CSD_GPU_NAME_REGEX": "H100",
        "CSD_GPU_CAPABILITY_MAJOR": "9",
        "CSD_GPU_CAPABILITY_MINOR": "0",
        "CSD_GPU_ARCH_FLAG": "sm_90",
    }
    with mock.patch.dict(os.environ, environment, clear=False):
        _validate_resume_runtime(runtime, copy.deepcopy(runtime), args, context="row")
        drifted = copy.deepcopy(runtime)
        drifted["gpus"][0] = {"name": "NVIDIA B200", "capability": [10, 0]}
        with pytest.raises(ValueError, match="GPU identities disagree"):
            _validate_resume_runtime(runtime, drifted, args, context="row")
