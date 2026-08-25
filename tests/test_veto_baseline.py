import pytest
import torch

from src.clean_self_distill.persistent import (
    VETO_DISTILLATION_KL_DIRECTION,
    PersistentConfig,
)
from src.clean_self_distill.streaming_distill import (
    _same_prefix_distillation_terms,
)
from src.clean_self_distill.veto import (
    VETO_TARGET_VERSION,
    scheduled_veto_beta,
    veto_target_logits,
)


def _veto_config(**overrides) -> PersistentConfig:
    values = {
        "branch": "veto",
        "variant": "adaptive_target_reformulation",
        "model": "/model",
        "model_id": "Qwen/Qwen3-8B",
        "revision": "revision",
        "episodes": 64,
        "scientific_checkpoints": (0, 16, 64),
    }
    values.update(overrides)
    return PersistentConfig(**values)


def test_veto_linear_schedule_matches_released_step_semantics() -> None:
    assert scheduled_veto_beta(step=0, total_steps=10) == pytest.approx(0.8)
    assert scheduled_veto_beta(step=5, total_steps=10) == pytest.approx(0.4)
    # The denominator is total_steps, so the final update approaches rather
    # than reaches beta_end.
    assert scheduled_veto_beta(step=9, total_steps=10) == pytest.approx(0.08)
    assert scheduled_veto_beta(
        step=9,
        total_steps=10,
        beta_start=0.3,
        beta_end=0.1,
        schedule="const",
    ) == pytest.approx(0.3)


def test_veto_target_is_normalized_teacher_student_product() -> None:
    student_logits = torch.tensor([[[1.3, -0.4, 0.2]]], dtype=torch.float64)
    teacher_logits = torch.tensor([[[-0.2, 1.1, 0.5]]], dtype=torch.float64)
    beta = 0.6
    target_logits = veto_target_logits(student_logits, teacher_logits, beta)

    student = torch.softmax(student_logits, dim=-1)
    teacher = torch.softmax(teacher_logits, dim=-1)
    expected = teacher * student.pow(beta)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(
        torch.softmax(target_logits, dim=-1), expected.float()
    )


def test_veto_beta_zero_recovers_raw_teacher() -> None:
    student_logits = torch.tensor([[[3.0, -2.0, 0.1]]])
    teacher_logits = torch.tensor([[[-0.7, 0.2, 1.4]]])
    target_logits = veto_target_logits(student_logits, teacher_logits, beta=0.0)
    torch.testing.assert_close(
        torch.softmax(target_logits, dim=-1),
        torch.softmax(teacher_logits, dim=-1),
    )


def test_veto_target_is_detached_and_fitted_with_forward_kl() -> None:
    student_logits = torch.tensor(
        [[[0.5, -0.3, 0.8]]], dtype=torch.float64, requires_grad=True
    )
    teacher_logits = torch.tensor(
        [[[-0.4, 1.2, 0.1]]], dtype=torch.float64, requires_grad=True
    )
    target_logits = veto_target_logits(student_logits, teacher_logits, beta=0.8)
    assert not target_logits.requires_grad

    loss, _ = _same_prefix_distillation_terms(
        student_logits,
        target_logits,
        top_k=3,
        temperature=1.0,
        token_clip=0.0,
        kl_direction="forward",
    )
    loss.backward()
    expected_gradient = (
        torch.softmax(student_logits.detach(), dim=-1)
        - torch.softmax(target_logits, dim=-1).to(student_logits.dtype)
    )
    torch.testing.assert_close(student_logits.grad, expected_gradient)
    assert teacher_logits.grad is None


def test_veto_has_distinct_fail_closed_run_identity() -> None:
    config = _veto_config()
    config.validate()
    assert config.method_id == "veto:adaptive_target_reformulation:forward_kl_v1"
    assert config.distillation_kl_direction == VETO_DISTILLATION_KL_DIRECTION
    payload = config.identity_payload()
    assert payload["target_reformulation"] == VETO_TARGET_VERSION
    assert payload["veto_beta_start"] == pytest.approx(0.8)
    assert payload["veto_beta_end"] == pytest.approx(0.0)
    assert payload["veto_beta_schedule"] == "linear"
    assert "trust_region_kl_budget" not in payload

    with pytest.raises(ValueError, match="forward KL"):
        _veto_config(student_kl_direction="reverse").validate()
    with pytest.raises(ValueError, match="distill_temperature=1"):
        _veto_config(distill_temperature=2.0).validate()
    with pytest.raises(ValueError, match="requires variant"):
        _veto_config(variant="trust_region").validate()


def test_veto_only_options_cannot_silently_change_lgsd_identity() -> None:
    lgsd = PersistentConfig(
        branch="clean",
        variant="trust_region",
        model="/model",
        model_id="Qwen/Qwen3-8B",
        revision="revision",
        episodes=1,
        scientific_checkpoints=(0, 1),
    )
    lgsd.validate()
    assert "veto_beta_start" not in lgsd.identity_payload()
    with pytest.raises(ValueError, match="defined only for Veto"):
        PersistentConfig(
            **{**lgsd.__dict__, "veto_beta_start": 0.7}
        ).validate()
