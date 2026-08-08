import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.clean_self_distill.persistent import (
    PersistentConfig,
    _traced_mean_teacher_kl,
    load_persistent_inputs,
)


def _config(branch: str = "clean") -> PersistentConfig:
    return PersistentConfig(
        branch=branch,
        variant="trust_region",
        model="/model",
        model_id="Qwen/Qwen3-8B",
        revision="revision",
        episodes=1,
        scientific_checkpoints=(0, 1),
    )


def test_only_trust_region_variant_is_accepted() -> None:
    config = _config()
    config.validate()
    assert config.method_id == "trsd:exponential_teacher_projection"
    with pytest.raises(ValueError, match="Unknown variant"):
        PersistentConfig(
            **{**config.__dict__, "variant": "removed_method"}
        ).validate()


def test_query_stream_needs_no_auxiliary_training_dataset(tmp_path: Path) -> None:
    problem = "Compute 1+1."
    row = {
        "query_id": "q0",
        "problem": problem,
        "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        "source": "deepmath",
    }
    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    queries, hashes = load_persistent_inputs(path, episodes=1)
    assert queries == [row]
    assert set(hashes) == {"query_manifest_sha256", "teacher_signal_sha256"}


def test_exponential_path_starts_at_student_distribution() -> None:
    student = torch.tensor([[[2.0, 0.0, -1.0]]])
    privileged = torch.tensor([[[-1.0, 2.0, 0.0]]])
    zero = _traced_mean_teacher_kl(student, privileged, 0.0)
    moved = _traced_mean_teacher_kl(student, privileged, 0.5)
    assert torch.allclose(zero, torch.zeros_like(zero), atol=1e-7)
    assert torch.all(moved > 0)
