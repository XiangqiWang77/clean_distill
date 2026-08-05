"""Fail-closed provenance contracts for the exact-16k H100 gate."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


gate = importlib.import_module(
    "scripts.clean_self_distill.10_full_sequence_memory_validation"
)
ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path):
    queries = tmp_path / "dev_queries.jsonl"
    queries.write_text('{"query_id":"q","problem":"p"}\n', encoding="utf-8")
    run_config = tmp_path / "run.env"
    run_config.write_text("CSD_RUN_ID=test\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    args = SimpleNamespace(
        queries=str(queries),
        run_config=str(run_config),
        git_commit="a" * 40,
        code_tree_sha256="b" * 64,
        model=str(model),
        model_id="Qwen/Qwen3-8B",
        revision="c" * 40,
        max_sequence_tokens=16_384,
        distill_token_chunk_size=128,
        distill_top_k=64,
        distill_temperature=1.0,
        lora_rank=8,
        lora_alpha=16,
        max_grad_norm=1.0,
        dtype="bfloat16",
        device_map="auto",
        expected_gpu_name_regex="H100",
        expected_gpu_capability_major=9,
        expected_gpu_capability_minor=0,
        expected_gpu_arch_flag="sm_90",
        minimum_reserved_headroom_bytes=1_000_000_000,
    )
    runtime = {
        "git_commit": args.git_commit,
        "code_tree_sha256": args.code_tree_sha256,
        "requested_model_revision": args.revision,
        "resolved_model_revision": args.revision,
        "model_use_cache": False,
        "gradient_checkpointing_enabled": True,
        "gradient_checkpointing_use_reentrant": False,
        "gpu_count": 1,
        "torch_arch_flags": ["sm_90"],
        "gpus": [
            {
                "name": "NVIDIA H100 80GB HBM3",
                "capability": [9, 0],
                "total_memory_bytes": 85_000_000_000,
            }
        ],
    }
    branch = {
        "sequence_tokens": 16_384,
        "distilled_positions": 16_383,
        "chunk_size": 128,
        "max_projected_chunk_tokens": 128,
        "loss": 0.2,
        "mean_teacher_student_kl": 0.2,
        "gradient_norm_before_clip": 0.3,
        "optimizer_changed_trainable_parameter": True,
        "peak_memory_allocated_bytes": 60_000_000_000,
        "peak_memory_reserved_bytes": 70_000_000_000,
    }
    payload = {
        "schema_version": gate.SCHEMA_VERSION,
        "status": "complete",
        "target_fields_loaded": False,
        "binding": gate._validation_binding(args),
        "runtime": runtime,
        "device_placement": {
            "parameter_devices": ["cuda:0"],
            "hf_device_maps": [{"": "0"}],
            "cpu_or_disk_offload": False,
        },
        "branches": [
            {**branch, "branch": "clean"},
            {**branch, "branch": "privileged"},
        ],
    }
    return args, payload


def test_gate_accepts_only_fully_bound_exact_full_sequence_artifact(tmp_path: Path):
    args, payload = _fixture(tmp_path)
    gate._validate_artifact(payload, args)

    mutated = copy.deepcopy(payload)
    mutated["branches"][0]["sequence_tokens"] = 8192
    with pytest.raises(ValueError, match="sequence length"):
        gate._validate_artifact(mutated, args)

    mutated = copy.deepcopy(payload)
    mutated["runtime"]["gradient_checkpointing_enabled"] = False
    with pytest.raises(ValueError, match="checkpointing"):
        gate._validate_artifact(mutated, args)

    mutated = copy.deepcopy(payload)
    mutated["device_placement"]["parameter_devices"] = ["cuda:0", "cpu"]
    with pytest.raises(ValueError, match="offload"):
        gate._validate_artifact(mutated, args)

    mutated = copy.deepcopy(payload)
    mutated["branches"][1]["peak_memory_reserved_bytes"] = 84_500_000_000
    with pytest.raises(ValueError, match="headroom"):
        gate._validate_artifact(mutated, args)


def test_gate_binding_detects_immutable_config_and_query_drift(tmp_path: Path):
    args, payload = _fixture(tmp_path)
    Path(args.run_config).write_text("CSD_RUN_ID=changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        gate._validate_artifact(payload, args)


def test_slurm_revalidates_marker_with_the_same_strict_program():
    text = (
        ROOT
        / "scripts/clean_self_distill/slurm/empirical_full_sequence_validate.slurm"
    ).read_text(encoding="utf-8")
    assert text.count("10_full_sequence_memory_validation.py") == 3
    assert text.count("--validate-existing") == 2
    assert '--run-config "$CSD_RUN_CONFIG"' in text
    assert '--code-tree-sha256 "$CSD_CODE_TREE_SHA256"' in text
    assert "CSD_FULL_SEQUENCE_MIN_HEADROOM_BYTES" in text
