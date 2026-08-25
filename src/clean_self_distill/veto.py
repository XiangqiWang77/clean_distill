"""Formula-faithful Veto target reformulation for matched baselines.

Veto replaces the raw teacher distribution with a product-of-experts target

    Q_beta(y | x) proportional to P_T(y | x) P_S(y | x) ** beta,

then fits the detached target with forward KL.  The implementation here is an
independent transcription of the published equation; no source from the
authors' repository is vendored.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


VETO_PAPER_URL = "https://aclanthology.org/2026.findings-acl.2094/"
VETO_REFERENCE_REPOSITORY = "https://github.com/jjun-0824/Veto"
VETO_REFERENCE_COMMIT = "0ff04a0de21e93bb7e13beaa55d37fd6975dd70e"
VETO_TARGET_VERSION = "teacher_student_product_of_experts_v1"
VETO_SCHEDULES = frozenset({"linear", "const"})


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def scheduled_veto_beta(
    *,
    step: int,
    total_steps: int,
    beta_start: float = 0.8,
    beta_end: float = 0.0,
    schedule: str = "linear",
) -> float:
    """Return Veto's global-step beta, matching the public implementation.

    Training steps are indexed from zero.  For the linear schedule, progress
    is ``step / total_steps`` rather than ``step / (total_steps - 1)``.  Thus a
    finite run approaches ``beta_end`` without using it on its final update,
    exactly as in the authors' released code.
    """
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if (
        not isinstance(total_steps, int)
        or isinstance(total_steps, bool)
        or total_steps <= 0
    ):
        raise ValueError("total_steps must be a positive integer")
    if schedule not in VETO_SCHEDULES:
        raise ValueError(f"Unknown Veto beta schedule {schedule!r}")
    start = _finite_nonnegative("beta_start", beta_start)
    end = _finite_nonnegative("beta_end", beta_end)
    if schedule == "const":
        return start
    progress = min(max(step / float(total_steps), 0.0), 1.0)
    return start + (end - start) * progress


def veto_target_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Construct detached logits for ``Q ∝ P_T P_S**beta``.

    Both inputs must describe the same token positions and vocabulary.  The
    returned values are normalized log probabilities and never retain an
    autograd connection to either input, as required for a fixed KD target.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have equal shapes")
    if student_logits.ndim < 1 or int(student_logits.shape[-1]) <= 0:
        raise ValueError("Veto logits must have a nonempty vocabulary axis")
    coefficient = _finite_nonnegative("beta", beta)
    with torch.no_grad():
        student_log_probs = F.log_softmax(student_logits.detach().float(), dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits.detach().float(), dim=-1)
        target = teacher_log_probs + coefficient * student_log_probs
        return target - torch.logsumexp(target, dim=-1, keepdim=True)


__all__ = [
    "VETO_PAPER_URL",
    "VETO_REFERENCE_COMMIT",
    "VETO_REFERENCE_REPOSITORY",
    "VETO_SCHEDULES",
    "VETO_TARGET_VERSION",
    "scheduled_veto_beta",
    "veto_target_logits",
]
