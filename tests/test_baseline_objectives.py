from __future__ import annotations

import math

import pytest
import torch

from baselines.objectives import (
    demopsd_reverse_kl,
    exponential_target_projection,
    grpo_group_advantages,
    grpo_token_loss,
    opsd_generalized_jsd,
    srpo_entropy_weights,
    srpo_route_masks,
    srpo_topk_jsd,
)
from baselines.train import _sample_group_tokens


def test_demopsd_identical_ema_policies_reduce_to_teacher() -> None:
    logits = torch.tensor([[[1.0, 0.0, -1.0]]])
    student = torch.tensor([[[0.2, 0.1, -0.4]]], requires_grad=True)
    loss, metrics = demopsd_reverse_kl(
        student, logits, logits, beta=50.0, alpha_max=0.15
    )
    assert torch.equal(metrics.jsd, torch.zeros_like(metrics.jsd))
    assert torch.equal(metrics.alpha, torch.zeros_like(metrics.alpha))
    assert torch.allclose(
        metrics.target_log_probs,
        torch.log_softmax(logits, dim=-1),
        atol=1e-7,
    )
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_demopsd_disagreement_is_bounded_and_target_is_normalized() -> None:
    student = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
    teacher = torch.tensor([[[12.0, -12.0, -12.0]]])
    reference = torch.tensor([[[-12.0, 12.0, -12.0]]])
    loss, metrics = demopsd_reverse_kl(
        student, teacher, reference, beta=100.0, alpha_max=0.15
    )
    assert 0 < float(metrics.alpha.item()) <= 0.15 + 1e-6
    assert torch.allclose(
        metrics.target_log_probs.exp().sum(dim=-1),
        torch.ones(1, 1),
        atol=1e-6,
    )
    assert torch.isfinite(loss)


def test_opsd_jsd_is_zero_for_identical_policies_and_backpropagates() -> None:
    student = torch.tensor([[[1.0, 0.0, -1.0]]], requires_grad=True)
    loss, metrics = opsd_generalized_jsd(
        student,
        student.detach().clone(),
        beta=0.5,
        element_clip=0.05,
    )
    assert torch.allclose(metrics.per_token_jsd, torch.zeros(1, 1), atol=1e-7)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_exponential_projection_enforces_one_trajectory_level_cap() -> None:
    student = torch.zeros(1, 3, 4)
    raw = torch.tensor(
        [[[12.0, -12.0, -12.0, -12.0]] * 3],
        dtype=torch.float32,
    )
    projected, metrics = exponential_target_projection(
        student,
        raw,
        kl_budget=0.02,
        binary_search_steps=14,
    )
    assert 0.0 < metrics.alpha < 1.0
    assert metrics.raw_target_kl > 0.02
    assert metrics.projected_target_kl <= 0.02 + 1e-6
    assert torch.allclose(
        projected,
        metrics.alpha * raw,
        atol=1e-7,
    )


def test_grpo_group_advantages_are_centered_and_constant_group_is_zero() -> None:
    advantages = grpo_group_advantages(torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(
        advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-6
    )
    assert torch.equal(
        grpo_group_advantages(torch.ones(4)), torch.zeros(4)
    )


def test_grpo_objective_has_positive_kl_and_finite_policy_gradient() -> None:
    current = torch.tensor([-0.8, -1.1], requires_grad=True)
    old = torch.tensor([-1.0, -1.0])
    reference = torch.tensor([-1.2, -0.7])
    loss, metrics = grpo_token_loss(
        current,
        old,
        reference,
        advantage=1.0,
        clip_epsilon=0.2,
        kl_coefficient=0.04,
    )
    assert (metrics["per_token_kl"] >= 0).all()
    loss.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_srpo_routing_matches_incorrect_with_teacher_rule() -> None:
    correct = torch.tensor([1, 0, 0, 1], dtype=torch.bool)
    available = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    sdpo, grpo = srpo_route_masks(correct, available)
    assert torch.equal(sdpo, torch.tensor([False, True, False, False]))
    assert torch.equal(grpo, ~sdpo)


def test_srpo_entropy_weights_have_mean_one_and_prefer_low_entropy() -> None:
    teacher = torch.tensor(
        [
            [[12.0, -12.0, -12.0, -12.0]],
            [[0.0, 0.0, 0.0, 0.0]],
        ]
    )
    entropy, weights = srpo_entropy_weights(teacher, beta=1.0)
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-6)
    assert entropy[0] < entropy[1]
    assert weights[0] > weights[1]


def test_srpo_topk_tail_jsd_is_zero_for_identical_policies() -> None:
    student = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]], requires_grad=True)
    loss, metrics = srpo_topk_jsd(
        student,
        student.detach().clone(),
        top_k=2,
        entropy_beta=1.0,
        jsd_alpha=0.5,
    )
    assert torch.allclose(metrics.per_token_jsd, torch.zeros(1, 1), atol=1e-7)
    assert (metrics.teacher_topk_mass < 1.0).all()
    assert (metrics.student_topk_mass < 1.0).all()
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_srpo_topk_tail_jsd_supports_split_student_and_teacher_devices() -> None:
    student = torch.tensor(
        [[[2.0, 1.0, 0.0, -1.0]]], device="cuda:0", requires_grad=True
    )
    teacher = student.detach().to("cuda:1")
    loss, metrics = srpo_topk_jsd(
        student,
        teacher,
        top_k=2,
        entropy_beta=1.0,
        jsd_alpha=0.5,
    )
    assert loss.device == student.device
    assert metrics.per_token_jsd.device == student.device
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_srpo_group_sampling_preserves_independent_seed_determinism() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])

    def draw() -> torch.Tensor:
        generators = []
        for seed in (17, 29):
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(seed)
            generators.append(generator)
        return _sample_group_tokens(
            logits,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            generators=generators,
        )

    first = draw()
    second = draw()
    assert first.shape == (2,)
    assert torch.equal(first, second)


def test_grpo_supports_srpo_asymmetric_high_clip() -> None:
    current = torch.tensor([math.log(1.4)], requires_grad=True)
    old = torch.zeros(1)
    reference = torch.zeros(1)
    _, metrics = grpo_token_loss(
        current,
        old,
        reference,
        advantage=1.0,
        clip_epsilon=0.2,
        clip_epsilon_high=0.28,
        kl_coefficient=0.0,
    )
    assert torch.allclose(metrics["surrogate"], torch.tensor([1.28]), atol=1e-6)
