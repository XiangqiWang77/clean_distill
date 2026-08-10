"""Post-hoc, label-free diagnostics for the TRSD exponential trust region.

The functions in this module never update a model.  They score one fixed,
on-policy response under its ordinary prompt and under several answer-free
pre-decision reasoning prompts.  Vocabulary-sized tensors are bounded along
the token axis; all reported KL values nevertheless use the complete model
vocabulary (there is no top-k approximation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .persistent import (
    STYLE_TASK_PARTITION_VERSION,
    _decoded_token,
    _token_partition,
    build_privileged_prompt,
)
from .runtime import project_logits, render_chat


MECHANISM_SCHEMA_VERSION = "trsd-posthoc-trust-region-mechanism-v1"
WRAPPER_SET_VERSION = "answer-free-predecision-paraphrases-v1"
TOKEN_SHIFT_DEFINITION = "projected_minus_normal_realized_token_logprob"
WRAPPER_IDS = ("neutral", "terse", "verbose")


class TrustRegionMechanismError(ValueError):
    """Raised when a post-hoc mechanism artifact violates its contract."""


def parse_float_grid(
    value: str | Sequence[float],
    *,
    name: str,
    minimum: float,
    maximum: float,
    require_endpoints: bool = False,
) -> tuple[float, ...]:
    """Parse a strictly increasing, finite grid with stable deduplication."""
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        try:
            values = [float(piece) for piece in pieces]
        except ValueError as exc:
            raise TrustRegionMechanismError(f"{name} contains a non-number") from exc
    else:
        values = [float(item) for item in value]
    if not values:
        raise TrustRegionMechanismError(f"{name} must not be empty")
    if any(
        not math.isfinite(item) or item < minimum or item > maximum
        for item in values
    ):
        raise TrustRegionMechanismError(
            f"{name} values must be finite and in [{minimum}, {maximum}]"
        )
    if any(right <= left for left, right in zip(values, values[1:])):
        raise TrustRegionMechanismError(f"{name} must be strictly increasing")
    if require_endpoints and (
        not math.isclose(values[0], minimum, abs_tol=1e-12)
        or not math.isclose(values[-1], maximum, abs_tol=1e-12)
    ):
        raise TrustRegionMechanismError(
            f"{name} must include endpoints {minimum} and {maximum}"
        )
    return tuple(values)


def build_privileged_prompt_wrapper(tokenizer, problem: str, wrapper_id: str) -> str:
    """Render one of three answer-free, semantically matched teacher prompts."""
    if wrapper_id == "neutral":
        # The neutral condition is byte-for-byte the training-time prompt.
        return build_privileged_prompt(tokenizer, problem)
    instructions = {
        "terse": (
            "Private reasoning-method instruction for the teacher: use only the "
            "problem statement. Split the task into explicit subgoals, track "
            "constraints and invariants, check boundary cases, and when possible "
            "verify the route independently."
        ),
        "verbose": (
            "Private reasoning-method instruction for the teacher: rely exclusively "
            "on the stated problem, with no answer, solution, outcome, or external "
            "feedback. Organize the reasoning as explicit subgoals; keep all "
            "constraints and invariants visible; examine relevant boundary cases; "
            "and, whenever feasible, validate the selected route by an independent "
            "alternative calculation or argument."
        ),
    }
    if wrapper_id not in instructions:
        raise TrustRegionMechanismError(
            f"Unknown wrapper {wrapper_id!r}; choose from {WRAPPER_IDS}"
        )
    messages = [
        {"role": "system", "content": instructions[wrapper_id]},
        {
            "role": "user",
            "content": (
                f"{problem.strip()}\n\nPlease reason step by step, and put your final "
                "answer within \\boxed{}."
            ),
        },
    ]
    return render_chat(tokenizer, messages, add_generation_prompt=True)


def population_variance(values: Sequence[float]) -> float:
    """Return population variance using accurate scalar summation."""
    numbers = [float(item) for item in values]
    if not numbers or any(not math.isfinite(item) for item in numbers):
        raise TrustRegionMechanismError(
            "population_variance requires at least one finite value"
        )
    mean = math.fsum(numbers) / len(numbers)
    return math.fsum((item - mean) ** 2 for item in numbers) / len(numbers)


def partition_shift_summary(
    categories: Sequence[str], logratios: Sequence[float]
) -> dict[str, Any]:
    """Reduce signed and absolute realized-token shifts by fixed token class."""
    if len(categories) != len(logratios) or not categories:
        raise TrustRegionMechanismError(
            "partition reducer requires equally sized, nonempty inputs"
        )
    result: dict[str, Any] = {
        "partition_version": STYLE_TASK_PARTITION_VERSION,
        "shift_definition": TOKEN_SHIFT_DEFINITION,
    }
    for partition in ("style", "task", "other"):
        selected = [
            float(shift)
            for category, shift in zip(categories, logratios)
            if category == partition
        ]
        if any(not math.isfinite(item) for item in selected):
            raise TrustRegionMechanismError("token logratio shifts must be finite")
        count = len(selected)
        signed = math.fsum(selected)
        absolute = math.fsum(abs(item) for item in selected)
        result[partition] = {
            "token_count": count,
            "signed_logratio_sum": signed,
            "mean_signed_logratio": signed / count if count else None,
            "abs_logratio_sum": absolute,
            "mean_abs_logratio": absolute / count if count else None,
        }
    if sum(result[key]["token_count"] for key in ("style", "task", "other")) != len(
        categories
    ):
        raise TrustRegionMechanismError("unknown token category in partition reducer")
    return result


def _summary_from_vectors(
    *,
    alpha: float,
    categories: Sequence[str],
    logratios: Sequence[float],
    kls: Sequence[float],
) -> dict[str, Any]:
    if len(logratios) != len(kls) or len(logratios) != len(categories):
        raise TrustRegionMechanismError("projection vectors have inconsistent lengths")
    count = len(logratios)
    if count <= 0:
        raise TrustRegionMechanismError("cannot summarize an empty rollout")
    trajectory = math.fsum(float(item) for item in logratios)
    kl_sum = math.fsum(float(item) for item in kls)
    shifts = partition_shift_summary(categories, logratios)
    task_gain = shifts["task"]["mean_signed_logratio"]
    style_shift = shifts["style"]["mean_abs_logratio"]
    return {
        "alpha": float(alpha),
        "token_count": count,
        "trajectory_logratio": trajectory,
        "normalized_logratio": trajectory / count,
        "kl_sum": kl_sum,
        "mean_kl": kl_sum / count,
        "task_logprob_gain": task_gain,
        "style_abs_logprob_shift": style_shift,
        "partition_shifts": shifts,
    }


def exact_projection_chunk(
    student_logits: torch.Tensor,
    privileged_logits: torch.Tensor,
    labels: torch.Tensor,
    alphas: Sequence[float],
) -> dict[float, dict[str, torch.Tensor]]:
    """Compute exact full-vocabulary exponential projections for one token chunk.

    ``q_alpha`` is represented by logits ``(1-alpha) z_s + alpha z_priv``.
    The returned KL direction is ``KL(q_alpha || p_student)`` and the realized
    logratio is ``log q_alpha(y_t) - log p_student(y_t)``.
    """
    if student_logits.shape != privileged_logits.shape or student_logits.ndim != 3:
        raise TrustRegionMechanismError(
            "student and privileged logits must have identical [B,T,V] shapes"
        )
    if student_logits.shape[0] != 1:
        raise TrustRegionMechanismError("mechanism scorer supports batch size one")
    if labels.shape != student_logits.shape[:2]:
        raise TrustRegionMechanismError("labels must match the [B,T] logit prefix")
    normalized_alphas = parse_float_grid(
        alphas,
        name="projection alphas",
        minimum=0.0,
        maximum=1.0,
    )
    student = student_logits.detach().float()
    privileged = privileged_logits.detach().float()
    label_ids = labels.to(student.device, dtype=torch.long)
    student_logz = torch.logsumexp(student, dim=-1)
    student_selected = student.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1)
    student_selected = student_selected - student_logz
    result: dict[float, dict[str, torch.Tensor]] = {}
    for alpha in normalized_alphas:
        projected = torch.lerp(student, privileged, float(alpha))
        projected_logz = torch.logsumexp(projected, dim=-1)
        projected_selected = projected.gather(
            -1, label_ids.unsqueeze(-1)
        ).squeeze(-1) - projected_logz
        projected_probability = F.softmax(projected, dim=-1)
        log_density_ratio = (
            projected - student + student_logz.unsqueeze(-1) - projected_logz.unsqueeze(-1)
        )
        per_token_kl = torch.sum(
            projected_probability * log_density_ratio, dim=-1
        ).clamp_min(0.0)
        result[float(alpha)] = {
            "student_logprob": student_selected.detach(),
            "projected_logprob": projected_selected.detach(),
            "logratio": (projected_selected - student_selected).detach(),
            "kl": per_token_kl.detach(),
        }
        del projected, projected_probability, log_density_ratio
    return result


@dataclass
class ProjectionEvaluation:
    summaries: dict[float, dict[str, Any]]
    traces: dict[float, dict[str, list[float]]]


@torch.inference_mode()
def evaluate_projection_alphas(
    *,
    model,
    student_hidden: torch.Tensor,
    privileged_hidden: torch.Tensor,
    labels: torch.Tensor,
    categories: Sequence[str],
    alphas: Sequence[float],
    chunk_size: int,
    capture_trace: bool = False,
) -> ProjectionEvaluation:
    """Evaluate alphas with full-vocabulary KL while bounding token chunks."""
    if student_hidden.shape != privileged_hidden.shape or student_hidden.ndim != 3:
        raise TrustRegionMechanismError("student/privileged hidden states must match")
    if student_hidden.shape[0] != 1 or labels.shape != student_hidden.shape[:2]:
        raise TrustRegionMechanismError("hidden states and labels must be batch-one aligned")
    token_count = int(student_hidden.shape[1])
    if token_count != len(categories) or token_count <= 0:
        raise TrustRegionMechanismError("token categories must cover the rollout")
    if chunk_size <= 0:
        raise TrustRegionMechanismError("chunk_size must be positive")
    normalized_alphas = parse_float_grid(
        sorted(set(float(item) for item in alphas)),
        name="projection alphas",
        minimum=0.0,
        maximum=1.0,
    )
    vectors = {
        alpha: {"logratio": [], "kl": [], "student_logprob": [], "projected_logprob": []}
        for alpha in normalized_alphas
    }
    max_projected_chunk_tokens = 0
    for start in range(0, token_count, int(chunk_size)):
        stop = min(start + int(chunk_size), token_count)
        max_projected_chunk_tokens = max(max_projected_chunk_tokens, stop - start)
        student_logits = project_logits(model, student_hidden[:, start:stop])
        privileged_logits = project_logits(model, privileged_hidden[:, start:stop])
        chunk = exact_projection_chunk(
            student_logits,
            privileged_logits,
            labels[:, start:stop],
            normalized_alphas,
        )
        for alpha in normalized_alphas:
            values = chunk[alpha]
            vectors[alpha]["logratio"].extend(
                float(item) for item in values["logratio"].cpu().reshape(-1).tolist()
            )
            vectors[alpha]["kl"].extend(
                float(item) for item in values["kl"].cpu().reshape(-1).tolist()
            )
            if capture_trace:
                vectors[alpha]["student_logprob"].extend(
                    float(item)
                    for item in values["student_logprob"].cpu().reshape(-1).tolist()
                )
                vectors[alpha]["projected_logprob"].extend(
                    float(item)
                    for item in values["projected_logprob"].cpu().reshape(-1).tolist()
                )
        del student_logits, privileged_logits, chunk
    summaries: dict[float, dict[str, Any]] = {}
    traces: dict[float, dict[str, list[float]]] = {}
    for alpha in normalized_alphas:
        summary = _summary_from_vectors(
            alpha=alpha,
            categories=categories,
            logratios=vectors[alpha]["logratio"],
            kls=vectors[alpha]["kl"],
        )
        summary["full_vocabulary_exact"] = True
        summary["max_projected_chunk_tokens"] = max_projected_chunk_tokens
        summaries[alpha] = summary
        if capture_trace:
            traces[alpha] = vectors[alpha]
    return ProjectionEvaluation(summaries=summaries, traces=traces)


def solve_epsilon_alphas(
    *,
    model,
    student_hidden: torch.Tensor,
    privileged_hidden: torch.Tensor,
    labels: torch.Tensor,
    categories: Sequence[str],
    alpha_evaluation: ProjectionEvaluation,
    epsilon_grid: Sequence[float],
    chunk_size: int,
    binary_search_steps: int,
) -> dict[float, float]:
    """Find the largest feasible alpha for every epsilon using exact KL passes."""
    epsilons = parse_float_grid(
        epsilon_grid,
        name="epsilon grid",
        minimum=0.0,
        maximum=float("inf"),
    )
    if epsilons[0] <= 0:
        raise TrustRegionMechanismError("epsilon grid values must be positive")
    if binary_search_steps <= 0:
        raise TrustRegionMechanismError("binary_search_steps must be positive")
    curve = sorted(
        (alpha, row["mean_kl"])
        for alpha, row in alpha_evaluation.summaries.items()
    )
    if not curve or not math.isclose(curve[0][0], 0.0, abs_tol=1e-12) or not math.isclose(
        curve[-1][0], 1.0, abs_tol=1e-12
    ):
        raise TrustRegionMechanismError("alpha calibration must include 0 and 1")
    for (_, left), (_, right) in zip(curve, curve[1:]):
        if right + 1e-7 < left:
            raise TrustRegionMechanismError("exact KL curve is unexpectedly non-monotone")

    bounds: dict[float, list[float]] = {}
    raw_kl = float(curve[-1][1])
    for epsilon in epsilons:
        if raw_kl <= epsilon:
            bounds[epsilon] = [1.0, 1.0]
            continue
        low, high = 0.0, 1.0
        for (left_alpha, left_kl), (right_alpha, _right_kl) in zip(
            curve, curve[1:]
        ):
            if left_kl <= epsilon:
                low, high = left_alpha, right_alpha
            else:
                break
        bounds[epsilon] = [low, high]

    for _ in range(int(binary_search_steps)):
        active = {
            epsilon: (bound[0] + bound[1]) / 2.0
            for epsilon, bound in bounds.items()
            if bound[1] - bound[0] > 1e-12
        }
        if not active:
            break
        unique_midpoints = sorted(set(active.values()))
        evaluated = evaluate_projection_alphas(
            model=model,
            student_hidden=student_hidden,
            privileged_hidden=privileged_hidden,
            labels=labels,
            categories=categories,
            alphas=unique_midpoints,
            chunk_size=chunk_size,
            capture_trace=False,
        )
        for epsilon, midpoint in active.items():
            value = float(evaluated.summaries[midpoint]["mean_kl"])
            if value <= epsilon:
                bounds[epsilon][0] = midpoint
            else:
                bounds[epsilon][1] = midpoint
    return {epsilon: bound[0] for epsilon, bound in bounds.items()}


def token_categories(tokenizer, response_ids: torch.Tensor) -> tuple[list[str], list[str]]:
    """Return the frozen style/task/other category and decoded token text."""
    ids = response_ids.detach().cpu().reshape(-1).tolist()
    texts = [_decoded_token(tokenizer, int(token_id)) for token_id in ids]
    return ([_token_partition(text) for text in texts], texts)


def summarize_wrapper_robustness(
    wrappers: Sequence[Mapping[str, Any]], *, selected_epsilon: float
) -> dict[str, Any]:
    """Reduce trajectory- and position-level wording sensitivity across wrappers."""
    if [str(row.get("wrapper_id")) for row in wrappers] != list(WRAPPER_IDS):
        raise TrustRegionMechanismError(
            "robustness reducer requires neutral/terse/verbose in canonical order"
        )

    raw_normalized = [float(row["raw"]["normalized_logratio"]) for row in wrappers]
    raw_kl = [float(row["raw"]["mean_kl"]) for row in wrappers]
    trace_lengths = {len(row["token_trace"]) for row in wrappers}
    if len(trace_lengths) != 1 or not trace_lengths or next(iter(trace_lengths)) <= 0:
        raise TrustRegionMechanismError("wrapper token traces are not aligned")

    raw_position_variances = [
        population_variance(
            [
                float(wrapper["token_trace"][position]["raw_surrogate_logratio"])
                for wrapper in wrappers
            ]
        )
        for position in range(next(iter(trace_lengths)))
    ]
    raw_trajectory_variance = population_variance(raw_normalized)
    raw_position_variance_mean = math.fsum(raw_position_variances) / len(
        raw_position_variances
    )
    epsilon_values = [
        float(item["epsilon"]) for item in wrappers[0]["epsilon_sweep"]
    ]
    epsilon_rows: list[dict[str, Any]] = []
    for epsilon in epsilon_values:
        matched = []
        for wrapper in wrappers:
            candidates = [
                item
                for item in wrapper["epsilon_sweep"]
                if math.isclose(float(item["epsilon"]), epsilon, abs_tol=1e-12)
            ]
            if len(candidates) != 1:
                raise TrustRegionMechanismError("wrapper epsilon sweeps are not aligned")
            matched.append(candidates[0])
        normalized = [float(item["normalized_logratio"]) for item in matched]
        achieved = [float(item["achieved_mean_kl"]) for item in matched]
        alphas = [float(item["alpha"]) for item in matched]
        projected_position_variances = []
        for position in range(next(iter(trace_lengths))):
            values = []
            for wrapper in wrappers:
                projections = wrapper["token_trace"][position]["epsilon_projections"]
                candidate = [
                    item
                    for item in projections
                    if math.isclose(float(item["epsilon"]), epsilon, abs_tol=1e-12)
                ]
                if len(candidate) != 1:
                    raise TrustRegionMechanismError(
                        "token traces do not cover every epsilon"
                    )
                values.append(float(candidate[0]["projected_surrogate_logratio"]))
            projected_position_variances.append(population_variance(values))
        trajectory_variance = population_variance(normalized)
        position_variance_mean = math.fsum(projected_position_variances) / len(
            projected_position_variances
        )
        epsilon_rows.append(
            {
                "epsilon": epsilon,
                "normalized_logratio_variance": trajectory_variance,
                "achieved_mean_kl_variance": population_variance(achieved),
                "alpha_variance": population_variance(alphas),
                "per_position_logratio_variance_mean": position_variance_mean,
                "raw_to_projected_normalized_logratio_variance_ratio": (
                    trajectory_variance / raw_trajectory_variance
                    if raw_trajectory_variance > 0
                    else None
                ),
                "raw_to_projected_position_variance_ratio": (
                    position_variance_mean / raw_position_variance_mean
                    if raw_position_variance_mean > 0
                    else None
                ),
                "constraint_active_wrapper_count": sum(
                    bool(item["constraint_active"]) for item in matched
                ),
            }
        )
    selected = [
        row
        for row in epsilon_rows
        if math.isclose(float(row["epsilon"]), selected_epsilon, abs_tol=1e-12)
    ]
    if len(selected) != 1:
        raise TrustRegionMechanismError("selected epsilon is missing from robustness grid")
    return {
        "wrapper_set_version": WRAPPER_SET_VERSION,
        "raw": {
            "normalized_logratio_variance": raw_trajectory_variance,
            "mean_kl_variance": population_variance(raw_kl),
            "per_position_logratio_variance_mean": raw_position_variance_mean,
        },
        "epsilon_sweep": epsilon_rows,
        "selected_epsilon": selected[0],
        "variance_definition": "population_variance_across_three_answer_free_wrappers",
    }
