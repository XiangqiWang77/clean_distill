"""Small, independently testable distillation and outcome-RL objectives.

The equations implemented here are deliberately separated from model loading
and Slurm orchestration.  This makes the signs, detach boundaries, and
normalizations testable on CPU before an 8B model is allocated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DemoPSDMetrics:
    """Detached per-token diagnostics for the DemoPSD target."""

    jsd: torch.Tensor
    alpha: torch.Tensor
    target_log_probs: torch.Tensor
    per_token_kl: torch.Tensor


@dataclass(frozen=True)
class OPSDMetrics:
    """Detached per-token diagnostics for the official generalized-JSD loss."""

    per_token_jsd: torch.Tensor
    student_entropy: torch.Tensor
    target_entropy: torch.Tensor


@dataclass(frozen=True)
class SRPOMetrics:
    """Detached per-token diagnostics for entropy-weighted routed SDPO."""

    per_token_jsd: torch.Tensor
    teacher_entropy: torch.Tensor
    raw_entropy_weight: torch.Tensor
    normalized_entropy_weight: torch.Tensor
    teacher_topk_mass: torch.Tensor
    student_topk_mass: torch.Tensor


@dataclass(frozen=True)
class ProjectionMetrics:
    """Diagnostics for a student-centered exponential target projection."""

    alpha: float
    raw_target_kl: float
    projected_target_kl: float


def demopsd_reverse_kl(
    student_logits: torch.Tensor,
    privileged_teacher_logits: torch.Tensor,
    ema_student_logits: torch.Tensor,
    *,
    beta: float,
    alpha_max: float = 0.15,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, DemoPSDMetrics]:
    """Return ``KL(student || stopgrad(geometric EMA target))``.

    ``privileged_teacher_logits`` and ``ema_student_logits`` must come from the
    same frozen EMA policy under privileged and ordinary contexts,
    respectively.  The target follows DemoPSD equations (7)--(12): JSD gates a
    reverse-KL barycenter, i.e. a geometric mixture in log-probability space.
    The complete vocabulary is used; this is an exact version of the stated
    objective rather than an undocumented guess at the paper's top-k kernel.
    """

    if student_logits.shape != privileged_teacher_logits.shape:
        raise ValueError("student and privileged teacher logits must have one shape")
    if student_logits.shape != ema_student_logits.shape:
        raise ValueError("student and EMA reference logits must have one shape")
    if student_logits.ndim < 2:
        raise ValueError("logits must include token and vocabulary dimensions")
    if not math.isfinite(float(beta)) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if not math.isfinite(float(alpha_max)) or not 0 <= alpha_max <= 1:
        raise ValueError("alpha_max must lie in [0, 1]")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(
            privileged_teacher_logits.detach().float() / temperature, dim=-1
        )
        reference_log_probs = F.log_softmax(
            ema_student_logits.detach().float() / temperature, dim=-1
        )
        mixture_log_probs = torch.logaddexp(
            teacher_log_probs, reference_log_probs
        ) - math.log(2.0)
        jsd = 0.5 * (
            (
                teacher_log_probs.exp()
                * (teacher_log_probs - mixture_log_probs)
            ).sum(dim=-1)
            + (
                reference_log_probs.exp()
                * (reference_log_probs - mixture_log_probs)
            ).sum(dim=-1)
        )
        # Floating-point cancellation can produce a tiny negative JSD.
        jsd = jsd.clamp_min(0.0)
        alpha = (torch.sigmoid(float(beta) * jsd) - 0.5) * (
            2.0 * float(alpha_max)
        )
        unnormalized_target = (
            (1.0 - alpha.unsqueeze(-1)) * teacher_log_probs
            + alpha.unsqueeze(-1) * reference_log_probs
        )
        target_log_probs = unnormalized_target - torch.logsumexp(
            unnormalized_target, dim=-1, keepdim=True
        )

    per_token_kl = (
        student_log_probs.exp() * (student_log_probs - target_log_probs)
    ).sum(dim=-1)
    loss = per_token_kl.mean() * (float(temperature) ** 2)
    return loss, DemoPSDMetrics(
        jsd=jsd.detach(),
        alpha=alpha.detach(),
        target_log_probs=target_log_probs.detach(),
        per_token_kl=per_token_kl.detach(),
    )


def opsd_generalized_jsd(
    student_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
    element_clip: float = 0.05,
) -> tuple[torch.Tensor, OPSDMetrics]:
    """Return the full-vocabulary generalized JSD used by OPSD.

    The implementation follows the released OPSD trainer: ``beta`` weights
    the teacher/target term, and the optional clip is applied to individual
    vocabulary contributions before summation.  The target is always detached.
    """

    if student_logits.shape != target_logits.shape or student_logits.ndim < 2:
        raise ValueError("student and target logits must have one valid shape")
    if not math.isfinite(float(beta)) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(element_clip)) or element_clip < 0:
        raise ValueError("element_clip must be finite and nonnegative")

    student_log_probs = F.log_softmax(
        student_logits.float() / float(temperature), dim=-1
    )
    with torch.no_grad():
        target_log_probs = F.log_softmax(
            target_logits.detach().float() / float(temperature), dim=-1
        )
        target_probs = target_log_probs.exp()
        target_entropy = -(target_probs * target_log_probs).sum(dim=-1)
    student_probs = student_log_probs.exp()
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1)

    if beta == 0.0:
        contributions = student_probs * (student_log_probs - target_log_probs)
    elif beta == 1.0:
        contributions = target_probs * (target_log_probs - student_log_probs)
    else:
        log_mixture = torch.logaddexp(
            student_log_probs + math.log1p(-float(beta)),
            target_log_probs + math.log(float(beta)),
        )
        contributions = (
            (1.0 - float(beta))
            * student_probs
            * (student_log_probs - log_mixture)
            + float(beta)
            * target_probs
            * (target_log_probs - log_mixture)
        )
    if element_clip > 0:
        contributions = contributions.clamp(max=float(element_clip))
    per_token_jsd = contributions.sum(dim=-1)
    loss = per_token_jsd.mean() * (float(temperature) ** 2)
    return loss, OPSDMetrics(
        per_token_jsd=per_token_jsd.detach(),
        student_entropy=student_entropy.detach(),
        target_entropy=target_entropy.detach(),
    )


def srpo_route_masks(
    correct: torch.Tensor,
    teacher_available: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the paper's disjoint SDPO and GRPO sample-routing masks."""

    if correct.ndim != 1 or correct.numel() == 0:
        raise ValueError("correct must be a non-empty rank-one tensor")
    if teacher_available.shape != correct.shape:
        raise ValueError("teacher_available and correct must have one shape")
    if correct.dtype != torch.bool and not torch.all((correct == 0) | (correct == 1)):
        raise ValueError("correct must contain only binary values")
    if teacher_available.dtype != torch.bool and not torch.all(
        (teacher_available == 0) | (teacher_available == 1)
    ):
        raise ValueError("teacher_available must contain only binary values")
    sdpo_mask = (~correct.bool()) & teacher_available.bool()
    return sdpo_mask, ~sdpo_mask


def srpo_entropy_weights(
    teacher_logits: torch.Tensor,
    *,
    beta: float = 1.0,
    normalizer: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-vocabulary teacher entropy and normalized SRPO weights."""

    if teacher_logits.ndim < 2:
        raise ValueError("teacher logits must include token and vocabulary dimensions")
    if teacher_logits.shape[-1] < 2:
        raise ValueError("teacher vocabulary must contain at least two entries")
    if not math.isfinite(float(beta)) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(teacher_logits.detach().float(), dim=-1)
        teacher_probs = teacher_log_probs.exp()
        entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
        raw_weight = torch.exp(-float(beta) * entropy)
        denominator = raw_weight.mean() if normalizer is None else torch.as_tensor(
            normalizer, dtype=raw_weight.dtype, device=raw_weight.device
        )
        if denominator.numel() != 1 or not bool(torch.isfinite(denominator).item()):
            raise ValueError("normalizer must be one finite scalar")
        if float(denominator.item()) <= 0:
            raise ValueError("normalizer must be positive")
        normalized = raw_weight / denominator
    return entropy, normalized


def srpo_topk_jsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    top_k: int = 100,
    entropy_beta: float = 1.0,
    jsd_alpha: float = 0.5,
    weight_normalizer: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, SRPOMetrics]:
    """Return entropy-weighted SDPO JSD on teacher top-k plus a tail bucket.

    Indices are selected by the detached EMA teacher. Both policies retain
    their mass on those indices and aggregate all remaining vocabulary
    probability into one tail event, as in the released SDPO implementation.
    """

    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("student and teacher logits must have one valid shape")
    vocabulary_size = int(student_logits.shape[-1])
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not 1 <= top_k < vocabulary_size
    ):
        raise ValueError("top_k must be an integer in [1, vocabulary_size)")
    if not math.isfinite(float(jsd_alpha)) or not 0.0 < jsd_alpha < 1.0:
        raise ValueError("jsd_alpha must lie strictly between zero and one")

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(teacher_logits.detach().float(), dim=-1)
        top_indices = teacher_log_probs.topk(top_k, dim=-1).indices
        teacher_selected_log = teacher_log_probs.gather(-1, top_indices)
        teacher_selected_prob = teacher_selected_log.exp()
        teacher_topk_mass = teacher_selected_prob.sum(dim=-1)
        teacher_tail = (1.0 - teacher_topk_mass).clamp_min(
            torch.finfo(teacher_selected_prob.dtype).tiny
        )
        teacher_reduced_log = torch.cat(
            (teacher_selected_log, teacher_tail.log().unsqueeze(-1)), dim=-1
        )
        teacher_reduced_log = teacher_reduced_log - torch.logsumexp(
            teacher_reduced_log, dim=-1, keepdim=True
        )
        teacher_entropy, normalized_weight = srpo_entropy_weights(
            teacher_logits,
            beta=entropy_beta,
            normalizer=weight_normalizer,
        )
        raw_weight = torch.exp(-float(entropy_beta) * teacher_entropy)

    student_selected_log = student_log_probs.gather(-1, top_indices)
    student_selected_prob = student_selected_log.exp()
    student_topk_mass = student_selected_prob.sum(dim=-1)
    student_tail = (1.0 - student_topk_mass).clamp_min(
        torch.finfo(student_selected_prob.dtype).tiny
    )
    student_reduced_log = torch.cat(
        (student_selected_log, student_tail.log().unsqueeze(-1)), dim=-1
    )
    student_reduced_log = student_reduced_log - torch.logsumexp(
        student_reduced_log, dim=-1, keepdim=True
    )
    log_mixture = torch.logaddexp(
        student_reduced_log + math.log1p(-float(jsd_alpha)),
        teacher_reduced_log + math.log(float(jsd_alpha)),
    )
    per_token_jsd = (
        (1.0 - float(jsd_alpha))
        * student_reduced_log.exp()
        * (student_reduced_log - log_mixture)
        + float(jsd_alpha)
        * teacher_reduced_log.exp()
        * (teacher_reduced_log - log_mixture)
    ).sum(dim=-1)
    loss = (normalized_weight * per_token_jsd).mean()
    return loss, SRPOMetrics(
        per_token_jsd=per_token_jsd.detach(),
        teacher_entropy=teacher_entropy.detach(),
        raw_entropy_weight=raw_weight.detach(),
        normalized_entropy_weight=normalized_weight.detach(),
        teacher_topk_mass=teacher_topk_mass.detach(),
        student_topk_mass=student_topk_mass.detach(),
    )


def _target_to_student_kl(
    student_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    target_log_probs = F.log_softmax(target_logits.float(), dim=-1)
    target_probs = target_log_probs.exp()
    return (target_probs * (target_log_probs - student_log_probs)).sum(dim=-1)


def exponential_target_projection(
    student_logits: torch.Tensor,
    raw_target_logits: torch.Tensor,
    *,
    kl_budget: float,
    binary_search_steps: int = 6,
) -> tuple[torch.Tensor, ProjectionMetrics]:
    """Project any raw target into a trajectory-mean ``KL(target||student)`` cap.

    The projected logits are the exponential-geodesic interpolation
    ``(1-alpha) * student_logits + alpha * raw_target_logits``.  A single
    trajectory-level ``alpha`` is shared by every token position.
    """

    if student_logits.shape != raw_target_logits.shape or student_logits.ndim < 2:
        raise ValueError("student and raw target logits must have one valid shape")
    if not math.isfinite(float(kl_budget)) or kl_budget <= 0:
        raise ValueError("kl_budget must be finite and positive")
    if (
        not isinstance(binary_search_steps, int)
        or isinstance(binary_search_steps, bool)
        or binary_search_steps <= 0
    ):
        raise ValueError("binary_search_steps must be a positive integer")

    student = student_logits.detach().float()
    raw = raw_target_logits.detach().float()

    def mean_kl(alpha: float) -> float:
        projected = (1.0 - alpha) * student + alpha * raw
        return float(_target_to_student_kl(student, projected).mean().item())

    raw_kl = mean_kl(1.0)
    if raw_kl <= float(kl_budget):
        alpha = 1.0
        achieved = raw_kl
    else:
        low, high = 0.0, 1.0
        for _ in range(binary_search_steps):
            mid = (low + high) / 2.0
            if mean_kl(mid) <= float(kl_budget):
                low = mid
            else:
                high = mid
        alpha = low
        achieved = mean_kl(alpha)
    projected = ((1.0 - alpha) * student + alpha * raw).detach()
    return projected, ProjectionMetrics(
        alpha=alpha,
        raw_target_kl=raw_kl,
        projected_target_kl=achieved,
    )


def grpo_group_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize outcome rewards within one prompt group.

    Population standard deviation (``unbiased=False``) matches the definition
    in DemoPSD equation (3) and avoids an undefined value for a group of one.
    A constant-reward group correctly yields zero advantage and no policy
    gradient.
    """

    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a non-empty rank-one tensor")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    values = rewards.float()
    return (values - values.mean()) / (values.std(unbiased=False) + float(eps))


def grpo_token_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantage: torch.Tensor | float,
    *,
    clip_epsilon: float = 0.2,
    clip_epsilon_high: float | None = None,
    kl_coefficient: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the token-mean clipped GRPO loss from DeepSeekMath eqs. (3)--(4)."""

    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current and old log probabilities must have one shape")
    if current_log_probs.shape != reference_log_probs.shape:
        raise ValueError("current and reference log probabilities must have one shape")
    if current_log_probs.numel() == 0:
        raise ValueError("log probabilities must not be empty")
    if not math.isfinite(float(clip_epsilon)) or clip_epsilon < 0:
        raise ValueError("clip_epsilon must be finite and nonnegative")
    high = clip_epsilon if clip_epsilon_high is None else clip_epsilon_high
    if not math.isfinite(float(high)) or high < 0:
        raise ValueError("clip_epsilon_high must be finite and nonnegative")
    if not math.isfinite(float(kl_coefficient)) or kl_coefficient < 0:
        raise ValueError("kl_coefficient must be finite and nonnegative")

    current = current_log_probs.float()
    old = old_log_probs.detach().float()
    reference = reference_log_probs.detach().float()
    advantage_tensor = torch.as_tensor(
        advantage, dtype=current.dtype, device=current.device
    ).detach()
    ratio = torch.exp(current - old)
    unclipped = ratio * advantage_tensor
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + float(high)) * advantage_tensor
    surrogate = torch.minimum(unclipped, clipped)

    # Schulman's k3 estimator used by DeepSeekMath equation (4).
    log_reference_ratio = reference - current
    per_token_kl = (
        torch.exp(log_reference_ratio) - log_reference_ratio - 1.0
    )
    loss = (-surrogate + float(kl_coefficient) * per_token_kl).mean()
    return loss, {
        "ratio": ratio.detach(),
        "surrogate": surrogate.detach(),
        "per_token_kl": per_token_kl.detach(),
    }


__all__ = [
    "DemoPSDMetrics",
    "OPSDMetrics",
    "ProjectionMetrics",
    "SRPOMetrics",
    "demopsd_reverse_kl",
    "exponential_target_projection",
    "grpo_group_advantages",
    "grpo_token_loss",
    "opsd_generalized_jsd",
    "srpo_entropy_weights",
    "srpo_route_masks",
    "srpo_topk_jsd",
]
