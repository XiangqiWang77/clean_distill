"""Memory-bounded same-prefix distillation over token chunks.

The student backbone may still be evaluated once over the full sequence, but
the vocabulary-sized student projection, teacher logits, and KL tensors are
materialized for at most ``chunk_size`` token positions at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F


def _same_prefix_distillation_terms(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    token_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact reverse KL ``KL(student || stopgrad(teacher))``.

    Sequence chunking already bounds the vocabulary tensors, so the formal
    TRSD objective is evaluated over the complete vocabulary without another
    top-k approximation. ``top_k`` remains a validated API field for run-schema
    compatibility with earlier checkpoints.
    """
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    if top_k <= 0:
        raise ValueError("distill_top_k must be positive")
    scaled_teacher = teacher_logits.float() / temperature
    scaled_student = student_logits.float() / temperature
    teacher_full_log_probs = F.log_softmax(scaled_teacher, dim=-1)
    student_full_log_probs = F.log_softmax(scaled_student, dim=-1)
    student_probs = student_full_log_probs.exp()
    per_token_kl = (
        student_probs * (student_full_log_probs - teacher_full_log_probs)
    ).sum(dim=-1)
    optimization_terms = (
        per_token_kl.clamp(max=token_clip) if token_clip > 0 else per_token_kl
    )
    return optimization_terms.mean() * (temperature**2), per_token_kl


ProjectStudent = Callable[[torch.Tensor], torch.Tensor]
TeacherForChunk = Callable[
    [torch.Tensor, torch.Tensor, int, int], torch.Tensor
]
ChunkObserver = Callable[
    [
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    None,
]


@dataclass(frozen=True)
class StreamingDistillationResult:
    """Scalar pre-update measurements from one streamed distillation pass."""

    loss: float
    mean_kl: float
    student_logprob_sum: float
    student_normalized_logprob: float
    teacher_logprob_sum: float
    teacher_normalized_logprob: float
    token_count: int
    max_chunk_tokens: int


def _realized_logprob_sum(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Sum realized-token log probabilities without a full log-softmax tensor."""
    with torch.no_grad():
        float_logits = logits.detach().float()
        label_ids = labels.to(device=logits.device, dtype=torch.long)
        selected_logits = float_logits.gather(
            -1, label_ids.unsqueeze(-1)
        ).squeeze(-1)
        selected_log_probs = selected_logits - torch.logsumexp(
            float_logits, dim=-1
        )
        return float(selected_log_probs.sum().item())


def stream_distillation_chunks(
    student_hidden: torch.Tensor,
    labels: torch.Tensor,
    project_student: ProjectStudent,
    teacher_for_chunk: TeacherForChunk,
    chunk_size: int,
    top_k: int,
    temperature: float,
    token_clip: float,
    backward: bool,
    observer: Optional[ChunkObserver] = None,
) -> StreamingDistillationResult:
    """Compute same-prefix distillation without full-sequence vocabulary logits.

    When ``backward`` is true, each token-count-weighted chunk is differentiated
    only with respect to a detached hidden-state leaf.  The resulting full
    hidden gradient is then propagated through ``student_hidden`` exactly once.
    This is crucial with gradient checkpointing: directly backpropagating every
    chunk through the shared decoder would recompute the 16k backbone once per
    chunk.  The execution contract therefore requires a frozen output
    projection; formal callers validate that invariant before entering here.
    The teacher is always evaluated as a fixed target.

    The optional observer runs after the chunk's backward call and receives
    only detached tensors.  This ordering lets diagnostics consume exact
    per-token values without extending the lifetime of the chunk loss graph.
    """
    if student_hidden.ndim != 3 or int(student_hidden.shape[0]) != 1:
        raise ValueError("student_hidden must have shape [1, L, H]")
    token_count = int(student_hidden.shape[1])
    if token_count <= 0:
        raise ValueError("student_hidden must contain at least one token")
    if labels.ndim != 2 or tuple(labels.shape) != (1, token_count):
        raise ValueError("labels must have shape [1, L] matching student_hidden")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    if not isinstance(token_clip, (int, float)) or token_clip < 0:
        raise ValueError("token_clip must be nonnegative")
    if not torch.isfinite(torch.tensor(float(temperature))):
        raise ValueError("distillation temperature must be finite")
    if not torch.isfinite(torch.tensor(float(token_clip))):
        raise ValueError("token_clip must be finite")
    if backward and not student_hidden.requires_grad:
        raise ValueError("backward=True requires differentiable student_hidden")

    total_loss = 0.0
    total_kl = 0.0
    student_logprob_sum = 0.0
    teacher_logprob_sum = 0.0
    max_chunk_tokens = 0
    hidden_gradient = torch.empty_like(student_hidden) if backward else None

    for start in range(0, token_count, chunk_size):
        stop = min(start + chunk_size, token_count)
        chunk_tokens = stop - start
        max_chunk_tokens = max(max_chunk_tokens, chunk_tokens)
        hidden_chunk = student_hidden[:, start:stop]
        labels_chunk = labels[:, start:stop]
        projection_hidden = hidden_chunk.detach()
        if backward:
            projection_hidden.requires_grad_(True)
        with torch.set_grad_enabled(backward):
            student_logits = project_student(projection_hidden)
        if (
            student_logits.ndim != 3
            or int(student_logits.shape[0]) != 1
            or int(student_logits.shape[1]) != chunk_tokens
            or int(student_logits.shape[2]) <= 0
        ):
            raise ValueError(
                "project_student must return logits with shape [1, chunk_tokens, V]"
            )

        # A same-prefix teacher is a fixed target.  Detaching both inputs also
        # prevents a callback that re-enables grad from reconnecting the target
        # to the student's shared backbone graph.
        with torch.no_grad():
            teacher_logits = teacher_for_chunk(
                student_logits.detach(), projection_hidden.detach(), start, stop
            )
        if teacher_logits.shape != student_logits.shape:
            raise ValueError(
                "teacher_for_chunk must return logits with exactly the student "
                "logits shape"
            )
        teacher_logits = teacher_logits.detach()

        chunk_loss, per_token_kl = _same_prefix_distillation_terms(
            student_logits,
            teacher_logits,
            top_k=top_k,
            temperature=float(temperature),
            token_clip=float(token_clip),
        )
        weighted_chunk_loss = chunk_loss * (chunk_tokens / token_count)

        # Reduce the objective and KL before releasing the chunk graph.
        total_loss += float(weighted_chunk_loss.detach().item())
        total_kl += float(per_token_kl.detach().float().sum().item())
        observed_kl = per_token_kl.detach()
        if backward:
            (chunk_hidden_gradient,) = torch.autograd.grad(
                weighted_chunk_loss,
                projection_hidden,
                retain_graph=False,
                create_graph=False,
            )
            assert hidden_gradient is not None
            hidden_gradient[:, start:stop].copy_(chunk_hidden_gradient)
            del chunk_hidden_gradient
        del weighted_chunk_loss, chunk_loss, per_token_kl

        # These detached metrics run only after the allocation-heavy KL graph
        # has been released.  gather-minus-logsumexp avoids full log-softmax.
        student_logprob_sum += _realized_logprob_sum(student_logits, labels_chunk)
        teacher_logprob_sum += _realized_logprob_sum(teacher_logits, labels_chunk)
        if observer is not None:
            observed_student = student_logits.detach()
            observed_teacher = teacher_logits.detach()
            observed_labels = labels_chunk.detach()
            observer(
                start,
                stop,
                observed_student,
                observed_teacher,
                observed_kl,
                observed_labels,
            )
            del observed_student, observed_teacher, observed_labels
        del observed_kl, student_logits, teacher_logits
        del projection_hidden, hidden_chunk, labels_chunk

    if backward:
        assert hidden_gradient is not None
        # One and only one traversal of the checkpointed shared backbone.
        student_hidden.backward(hidden_gradient)
        del hidden_gradient

    return StreamingDistillationResult(
        loss=total_loss,
        mean_kl=total_kl / token_count,
        student_logprob_sum=student_logprob_sum,
        student_normalized_logprob=student_logprob_sum / token_count,
        teacher_logprob_sum=teacher_logprob_sum,
        teacher_normalized_logprob=teacher_logprob_sum / token_count,
        token_count=token_count,
        max_chunk_tokens=max_chunk_tokens,
    )


__all__ = ["StreamingDistillationResult", "stream_distillation_chunks"]
