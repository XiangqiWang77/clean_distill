import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.clean_self_distill.persistent import (
    LGSD_DISTILLATION_KL_DIRECTION,
    PersistentConfig,
    _projected_teacher_logits,
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
    assert config.student_kl_direction == "forward"
    assert config.method_id == "lgsd:geometric_kl_ball_projection:forward_kl_v1"
    assert config.distillation_kl_direction == LGSD_DISTILLATION_KL_DIRECTION
    assert config.identity_payload()["student_kl_direction"] == "forward"
    assert "update_guard" not in config.identity_payload()
    guarded = PersistentConfig(**{**config.__dict__, "update_guard": True})
    guarded.validate()
    assert guarded.method_id.endswith(":update_guard_v1")
    assert guarded.identity_payload()["update_guard"] is True
    with pytest.raises(ValueError, match="defined only for LGSD"):
        PersistentConfig(
            **{**config.__dict__, "branch": "privileged", "update_guard": True}
        ).validate()

    legacy = PersistentConfig(
        **{**config.__dict__, "student_kl_direction": "reverse"}
    )
    legacy.validate()
    assert legacy.method_id == "trsd:exponential_teacher_projection"
    assert "student_kl_direction" not in legacy.identity_payload()
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


def test_exponential_logits_are_the_normalized_geometric_mixture() -> None:
    student = torch.tensor([[[1.7, -0.3, 0.1]]], dtype=torch.float64)
    privileged = torch.tensor([[[-0.2, 1.4, 0.6]]], dtype=torch.float64)
    alpha = 0.37
    projected = _projected_teacher_logits(
        student, privileged, alpha, path="exponential"
    )
    p = torch.softmax(student, dim=-1)
    q = torch.softmax(privileged, dim=-1)
    geometric = p.pow(1.0 - alpha) * q.pow(alpha)
    geometric = geometric / geometric.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(torch.softmax(projected, dim=-1), geometric.float())


def test_reverse_kl_has_exact_anchor_rewrite_but_forward_kl_does_not() -> None:
    alpha = 0.41
    p = torch.tensor([0.72, 0.19, 0.09], dtype=torch.float64)
    privileged = torch.tensor([0.08, 0.31, 0.61], dtype=torch.float64)
    policy = torch.tensor([0.24, 0.51, 0.25], dtype=torch.float64)
    unnormalized = p.pow(1.0 - alpha) * privileged.pow(alpha)
    normalizer = unnormalized.sum()
    projected = unnormalized / normalizer

    def kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.sum(left * (left.log() - right.log()))

    reverse_projected = kl(policy, projected)
    reverse_anchor = (
        (1.0 - alpha) * kl(policy, p)
        + alpha * kl(policy, privileged)
        + normalizer.log()
    )
    torch.testing.assert_close(reverse_projected, reverse_anchor)

    forward_projected = kl(projected, policy)
    forward_anchor = (
        (1.0 - alpha) * kl(p, policy)
        + alpha * kl(privileged, policy)
    )
    assert not torch.isclose(forward_projected, forward_anchor, atol=1e-8)
