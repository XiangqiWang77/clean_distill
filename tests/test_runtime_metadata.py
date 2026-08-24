"""Focused tests for reproducibility-critical runtime provenance."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from src.clean_self_distill.runtime import collect_runtime_metadata


class RuntimeMetadataTest(unittest.TestCase):
    def test_records_interpreter_overlay_torch_and_slurm_allocation(self):
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: "NVIDIA H100 80GB HBM3",
            get_device_properties=lambda index: SimpleNamespace(total_memory=1234),
            get_device_capability=lambda index: (9, 0),
        )
        fake_torch = SimpleNamespace(
            __version__="2.9.1+cu128",
            __file__="/shared/cu128/torch/__init__.py",
            version=SimpleNamespace(cuda="12.8"),
            cuda=fake_cuda,
            _C=SimpleNamespace(
                _cuda_getArchFlags=lambda: "sm_80 sm_90 sm_100 sm_120"
            ),
        )
        environment = {
            "CONDA_PREFIX": "/workspace/.conda/envs/trsd",
            "CSD_TORCH_OVERLAY": "/shared/cu128",
            "SLURM_JOB_ID": "12345_1",
            "SLURM_JOB_PARTITION": "accelerator",
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "CSD_GPU_PARTITION": "accelerator",
            "CSD_GPU_GRES": "gpu:1",
            "CSD_GPU_NAME_REGEX": "GPU",
            "CSD_GPU_CAPABILITY_MAJOR": "9",
            "CSD_GPU_CAPABILITY_MINOR": "0",
            "CSD_GPU_ARCH_FLAG": "sm_90",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.dict(sys.modules, {"torch": fake_torch}),
            mock.patch(
                "src.clean_self_distill.runtime.socket.gethostname",
                return_value="accelerator-node",
            ),
            mock.patch(
                "src.clean_self_distill.runtime.platform.platform",
                return_value="Linux-test",
            ),
            mock.patch(
                "src.clean_self_distill.runtime.subprocess.check_output",
                side_effect=["a" * 40 + "\n", ""],
            ),
        ):
            metadata = collect_runtime_metadata(
                model_path="Qwen/Qwen3-4B", revision="b" * 40
            )

        self.assertEqual(metadata["hostname"], "accelerator-node")
        self.assertEqual(metadata["python_executable"], sys.executable)
        self.assertEqual(metadata["torch_overlay"], "/shared/cu128")
        self.assertEqual(metadata["torch"], "2.9.1+cu128")
        self.assertEqual(
            metadata["torch_module_path"],
            "/shared/cu128/torch/__init__.py",
        )
        self.assertIn("sm_100", metadata["torch_arch_flags"])
        self.assertEqual(metadata["slurm_array_job_id"], "12345")
        self.assertEqual(metadata["slurm_array_task_id"], "1")
        self.assertEqual(metadata["slurm_job_partition"], "accelerator")
        self.assertEqual(metadata["requested_gpu_gres"], "gpu:1")
        self.assertEqual(metadata["expected_gpu_capability"], [9, 0])
        self.assertEqual(metadata["expected_gpu_arch_flag"], "sm_90")
        self.assertEqual(metadata["gpus"][0]["capability"], [9, 0])


if __name__ == "__main__":
    unittest.main()
