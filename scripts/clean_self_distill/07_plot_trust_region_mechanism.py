#!/usr/bin/env python3
"""Build a fail-closed empirical figure from one TRSD post-hoc trajectory.

The input is the one-row JSONL emitted by ``06_trust_region_mechanism.py``.
This reporter never loads a model and never generates text.  It verifies the
sealed problem binding, recomputes every plotted aggregate from token-level
rows, and refuses incomplete wrapper or epsilon coverage.  Token positions are
not treated as independent query replicates: the sole interval in the figure
is a paired circular moving-block bootstrap over the observed trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from src.clean_self_distill.heldout import load_sealed_labels
from src.clean_self_distill.trust_region_mechanism import (
    MECHANISM_SCHEMA_VERSION,
    TOKEN_SHIFT_DEFINITION,
    WRAPPER_IDS,
    WRAPPER_SET_VERSION,
)


class MechanismFigureError(ValueError):
    """Raised when the real artifact cannot support every requested panel."""


def _text(value: Any, context: str) -> str:
    result = str(value).strip() if value is not None else ""
    if not result:
        raise MechanismFigureError(f"{context} must be a non-empty string")
    return result


def _number(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise MechanismFigureError(f"{context} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MechanismFigureError(f"{context} must be numeric") from exc
    if not math.isfinite(result):
        raise MechanismFigureError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise MechanismFigureError(f"{context} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise MechanismFigureError(f"{context} must be <= {maximum}")
    return result


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise MechanismFigureError(f"{context} must be an integer, not boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MechanismFigureError(f"{context} must be an integer") from exc
    if isinstance(value, float) and value != result:
        raise MechanismFigureError(f"{context} must be an exact integer")
    if minimum is not None and result < minimum:
        raise MechanismFigureError(f"{context} must be >= {minimum}")
    return result


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MechanismFigureError(f"{context} must be a JSON object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise MechanismFigureError(f"{context} must be a JSON list")
    return value


def _close(
    left: float,
    right: float,
    *,
    absolute: float = 1e-7,
    relative: float = 2e-5,
) -> bool:
    return abs(left - right) <= absolute + relative * max(abs(left), abs(right))


def _one_row_jsonl(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise MechanismFigureError(f"Missing post-hoc JSONL: {target}")
    lines = [line.strip() for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise MechanismFigureError(
            f"{target} must contain exactly one completed query row, found {len(lines)}"
        )
    try:
        row = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise MechanismFigureError(f"Invalid JSON in {target}: {exc}") from exc
    if not isinstance(row, dict):
        raise MechanismFigureError(f"{target} row must be a JSON object")
    return row


def _find_epsilon(rows: Sequence[Mapping[str, Any]], epsilon: float, context: str) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if _close(_number(row.get("epsilon"), f"{context}.epsilon"), epsilon)
    ]
    if len(matches) != 1:
        raise MechanismFigureError(
            f"{context} must contain exactly one row for epsilon={epsilon}"
        )
    return matches[0]


def _population_variance(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise MechanismFigureError("Variance input must be a non-empty finite vector")
    return float(np.var(array, ddof=0))


def _partition_metrics(
    categories: Sequence[str], shifts: Sequence[float]
) -> tuple[float, float]:
    task = [
        float(shift)
        for category, shift in zip(categories, shifts)
        if category == "task"
    ]
    style = [
        abs(float(shift))
        for category, shift in zip(categories, shifts)
        if category == "style"
    ]
    if not task or not style:
        raise MechanismFigureError(
            "The selected trajectory must contain both task and style token positions"
        )
    return float(np.mean(task)), float(np.mean(style))


def _validate_artifact(
    row: Mapping[str, Any], labels: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    if row.get("schema_version") != MECHANISM_SCHEMA_VERSION:
        raise MechanismFigureError(
            f"schema_version must be {MECHANISM_SCHEMA_VERSION!r}"
        )
    if row.get("record_type") != "query_mechanism":
        raise MechanismFigureError("record_type must be 'query_mechanism'")
    if row.get("labels_loaded") is not False or row.get("label_paths_accepted_by_cli") is not False:
        raise MechanismFigureError("Collector must be label-free")
    query_id = _text(row.get("query_id"), "query_id")
    if query_id not in labels:
        raise MechanismFigureError("Mechanism query is absent from the sealed labels")
    problem_sha256 = _text(row.get("problem_sha256"), "problem_sha256")
    if problem_sha256 != labels[query_id]["problem_sha256"]:
        raise MechanismFigureError("Mechanism problem hash disagrees with sealed labels")
    if row.get("full_vocabulary_kl") is not True:
        raise MechanismFigureError("Figure requires exact full-vocabulary KL")
    checkpoint = _mapping(row.get("checkpoint_validation"), "checkpoint_validation")
    checkpoint_episode = _integer(
        row.get("checkpoint_episode"), "checkpoint_episode", minimum=1
    )
    selection_status = _text(
        checkpoint.get("selection_status"),
        "checkpoint_validation.selection_status",
    )
    checkpoint_type = _text(
        checkpoint.get("checkpoint_type"),
        "checkpoint_validation.checkpoint_type",
    )
    method_id = _text(
        checkpoint.get("method_id"), "checkpoint_validation.method_id"
    )
    variant = _text(
        checkpoint.get("variant"), "checkpoint_validation.variant"
    )
    checkpoint_manifest_sha256 = _text(
        checkpoint.get("checkpoint_manifest_sha256"),
        "checkpoint_validation.checkpoint_manifest_sha256",
    )
    run_manifest_sha256 = _text(
        checkpoint.get("run_manifest_sha256"),
        "checkpoint_validation.run_manifest_sha256",
    )
    for value, context in (
        (checkpoint_manifest_sha256, "checkpoint manifest hash"),
        (run_manifest_sha256, "run manifest hash"),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise MechanismFigureError(f"{context} must be lowercase SHA-256")
    completed_final = (
        selection_status == "completed_scientific_final"
        and checkpoint_type == "scientific"
        and method_id == "trsd:exponential_teacher_projection"
        and variant == "trust_region"
    )
    historical_endpoint = (
        selection_status == "historical_latest_rolling_endpoint"
        and checkpoint_type == "rolling"
        and checkpoint_episode == 36
        and method_id.endswith(":trust_region")
    )
    if not completed_final and not historical_endpoint:
        raise MechanismFigureError(
            "Checkpoint must be either a completed scientific TRSD final or the "
            "explicitly audited latest rolling episode-36 historical endpoint"
        )
    complete_sha256 = checkpoint.get("complete_sha256")
    if completed_final:
        complete_hash = _text(
            complete_sha256, "checkpoint_validation.complete_sha256"
        )
        if len(complete_hash) != 64 or any(
            character not in "0123456789abcdef" for character in complete_hash
        ):
            raise MechanismFigureError("complete manifest hash must be lowercase SHA-256")
    elif complete_sha256 is not None:
        # A historical run directory may contain a later/partial COMPLETE file;
        # its presence does not promote a rolling checkpoint to a scientific
        # final.  Preserve the explicit selection_status and validate only the
        # reported content hash here.
        complete_hash = _text(
            complete_sha256, "checkpoint_validation.complete_sha256"
        )
        if len(complete_hash) != 64 or any(
            character not in "0123456789abcdef" for character in complete_hash
        ):
            raise MechanismFigureError("complete manifest hash must be lowercase SHA-256")
    _text(row.get("checkpoint_sha256"), "checkpoint_sha256")
    _text(row.get("run_identity_sha256"), "run_identity_sha256")
    generated_tokens = _integer(row.get("generated_tokens"), "generated_tokens", minimum=8)
    response_ids = _sequence(row.get("response_token_ids"), "response_token_ids")
    if len(response_ids) != generated_tokens:
        raise MechanismFigureError("response_token_ids length disagrees with generated_tokens")
    for index, token_id in enumerate(response_ids):
        _integer(token_id, f"response_token_ids[{index}]", minimum=0)

    alpha_grid = [
        _number(value, f"alpha_grid[{index}]", minimum=0.0, maximum=1.0)
        for index, value in enumerate(_sequence(row.get("alpha_grid"), "alpha_grid"))
    ]
    epsilon_grid = [
        _number(value, f"epsilon_grid[{index}]", minimum=0.0)
        for index, value in enumerate(_sequence(row.get("epsilon_grid"), "epsilon_grid"))
    ]
    if (
        len(alpha_grid) < 3
        or not _close(alpha_grid[0], 0.0)
        or not _close(alpha_grid[-1], 1.0)
        or any(right <= left for left, right in zip(alpha_grid, alpha_grid[1:]))
    ):
        raise MechanismFigureError("alpha_grid must be increasing and include 0 and 1")
    if len(epsilon_grid) < 4 or any(
        right <= left for left, right in zip(epsilon_grid, epsilon_grid[1:])
    ):
        raise MechanismFigureError("epsilon_grid must contain at least four increasing budgets")
    selected_epsilon = _number(row.get("selected_epsilon"), "selected_epsilon", minimum=0.0)
    training_epsilon = _number(row.get("training_epsilon"), "training_epsilon", minimum=0.0)
    if row.get("selected_epsilon_role") != "posthoc_stress_test_not_training_budget":
        raise MechanismFigureError("selected epsilon must be explicitly marked post-hoc")
    if row.get("training_epsilon_role") != "checkpoint_training_budget":
        raise MechanismFigureError("training epsilon role is missing")
    if not any(_close(selected_epsilon, value) for value in epsilon_grid):
        raise MechanismFigureError("selected_epsilon is absent from epsilon_grid")
    if not any(_close(training_epsilon, value) for value in epsilon_grid):
        raise MechanismFigureError("training_epsilon is absent from epsilon_grid")

    wrappers_raw = _sequence(row.get("wrappers"), "wrappers")
    wrapper_ids = [
        _text(_mapping(value, f"wrappers[{index}]").get("wrapper_id"), f"wrappers[{index}].wrapper_id")
        for index, value in enumerate(wrappers_raw)
    ]
    if tuple(wrapper_ids) != tuple(WRAPPER_IDS):
        raise MechanismFigureError(
            f"wrappers must occur once in canonical order {tuple(WRAPPER_IDS)!r}"
        )

    wrappers: list[dict[str, Any]] = []
    shared_identity: list[tuple[int, int, str, str]] | None = None
    for wrapper_index, wrapper_value in enumerate(wrappers_raw):
        wrapper = dict(_mapping(wrapper_value, f"wrappers[{wrapper_index}]"))
        wrapper_id = wrapper_ids[wrapper_index]
        context = f"wrapper[{wrapper_id}]"
        if wrapper.get("wrapper_set_version") != WRAPPER_SET_VERSION:
            raise MechanismFigureError(f"{context} has the wrong wrapper_set_version")
        _text(wrapper.get("prompt_sha256"), f"{context}.prompt_sha256")
        _integer(wrapper.get("prompt_tokens"), f"{context}.prompt_tokens", minimum=1)
        raw_summary = _mapping(wrapper.get("raw"), f"{context}.raw")
        raw_alpha = _number(raw_summary.get("alpha"), f"{context}.raw.alpha", minimum=0.0, maximum=1.0)
        if not _close(raw_alpha, 1.0):
            raise MechanismFigureError(f"{context}.raw must be alpha=1")
        if raw_summary.get("projection") != "unconstrained_privileged_surrogate":
            raise MechanismFigureError(f"{context}.raw projection identity is missing")
        if _integer(raw_summary.get("token_count"), f"{context}.raw.token_count") != generated_tokens:
            raise MechanismFigureError(f"{context}.raw token count mismatch")

        alpha_sweep = [
            _mapping(value, f"{context}.alpha_sweep[{index}]")
            for index, value in enumerate(_sequence(wrapper.get("alpha_sweep"), f"{context}.alpha_sweep"))
        ]
        alpha_values = [
            _number(value.get("alpha"), f"{context}.alpha_sweep.alpha", minimum=0.0, maximum=1.0)
            for value in alpha_sweep
        ]
        if len(alpha_values) != len(alpha_grid) or any(
            not _close(left, right) for left, right in zip(alpha_values, alpha_grid)
        ):
            raise MechanismFigureError(f"{context}.alpha_sweep does not cover alpha_grid")

        epsilon_sweep = [
            dict(_mapping(value, f"{context}.epsilon_sweep[{index}]"))
            for index, value in enumerate(_sequence(wrapper.get("epsilon_sweep"), f"{context}.epsilon_sweep"))
        ]
        epsilon_values = [
            _number(value.get("epsilon"), f"{context}.epsilon_sweep.epsilon", minimum=0.0)
            for value in epsilon_sweep
        ]
        if len(epsilon_values) != len(epsilon_grid) or any(
            not _close(left, right) for left, right in zip(epsilon_values, epsilon_grid)
        ):
            raise MechanismFigureError(f"{context}.epsilon_sweep does not cover epsilon_grid")

        token_trace = [
            dict(_mapping(value, f"{context}.token_trace[{index}]"))
            for index, value in enumerate(_sequence(wrapper.get("token_trace"), f"{context}.token_trace"))
        ]
        if len(token_trace) != generated_tokens:
            raise MechanismFigureError(f"{context}.token_trace length mismatch")
        identities: list[tuple[int, int, str, str]] = []
        categories: list[str] = []
        raw_kls: list[float] = []
        raw_shifts: list[float] = []
        per_epsilon_kls: dict[float, list[float]] = {value: [] for value in epsilon_grid}
        per_epsilon_shifts: dict[float, list[float]] = {value: [] for value in epsilon_grid}
        per_epsilon_alpha: dict[float, list[float]] = {value: [] for value in epsilon_grid}
        for position, token in enumerate(token_trace):
            token_context = f"{context}.token_trace[{position}]"
            actual_position = _integer(token.get("position"), f"{token_context}.position", minimum=0)
            if actual_position != position:
                raise MechanismFigureError(f"{context} token positions must be contiguous from zero")
            normalized = _number(token.get("normalized_position"), f"{token_context}.normalized_position", minimum=0.0, maximum=1.0)
            expected_normalized = position / (generated_tokens - 1)
            if not _close(normalized, expected_normalized, absolute=1e-9, relative=1e-7):
                raise MechanismFigureError(f"{token_context}.normalized_position mismatch")
            token_id = _integer(token.get("token_id"), f"{token_context}.token_id", minimum=0)
            if token_id != int(response_ids[position]):
                raise MechanismFigureError(f"{token_context}.token_id disagrees with response")
            token_text = str(token.get("token_text", ""))
            category = _text(token.get("token_category"), f"{token_context}.token_category")
            if category not in {"task", "style", "other"}:
                raise MechanismFigureError(f"{token_context} has unknown token category")
            identities.append((position, token_id, token_text, category))
            categories.append(category)
            raw_kls.append(_number(token.get("raw_kl"), f"{token_context}.raw_kl", minimum=0.0))
            raw_shifts.append(_number(token.get("raw_surrogate_logratio"), f"{token_context}.raw_surrogate_logratio"))
            projections = [
                _mapping(value, f"{token_context}.epsilon_projections[{index}]")
                for index, value in enumerate(_sequence(token.get("epsilon_projections"), f"{token_context}.epsilon_projections"))
            ]
            projection_eps = [
                _number(value.get("epsilon"), f"{token_context}.epsilon", minimum=0.0)
                for value in projections
            ]
            if len(projection_eps) != len(epsilon_grid) or any(
                not _close(left, right) for left, right in zip(projection_eps, epsilon_grid)
            ):
                raise MechanismFigureError(f"{token_context} does not cover epsilon_grid")
            for epsilon, projection in zip(epsilon_grid, projections):
                per_epsilon_alpha[epsilon].append(
                    _number(projection.get("alpha"), f"{token_context}.alpha", minimum=0.0, maximum=1.0)
                )
                per_epsilon_kls[epsilon].append(
                    _number(projection.get("projected_kl"), f"{token_context}.projected_kl", minimum=0.0)
                )
                per_epsilon_shifts[epsilon].append(
                    _number(projection.get("projected_surrogate_logratio"), f"{token_context}.projected_surrogate_logratio")
                )
            selected_projection = _find_epsilon(projections, selected_epsilon, token_context)
            for flat_key, nested_key in (
                ("projected_alpha", "alpha"),
                ("projected_kl", "projected_kl"),
                ("projected_surrogate_logratio", "projected_surrogate_logratio"),
            ):
                flat = _number(token.get(flat_key), f"{token_context}.{flat_key}")
                nested = _number(selected_projection.get(nested_key), f"{token_context}.{nested_key}")
                if not _close(flat, nested):
                    raise MechanismFigureError(f"{token_context}.{flat_key} is not selected-epsilon data")
        if shared_identity is None:
            shared_identity = identities
        elif identities != shared_identity:
            raise MechanismFigureError("Wrapper token traces are not position-aligned")
        raw_mean_kl = float(np.mean(raw_kls))
        if not _close(raw_mean_kl, _number(raw_summary.get("mean_kl"), f"{context}.raw.mean_kl")):
            raise MechanismFigureError(f"{context}.raw mean KL disagrees with token trace")
        raw_task_gain, raw_style_shift = _partition_metrics(categories, raw_shifts)
        if not _close(raw_task_gain, _number(raw_summary.get("task_logprob_gain"), f"{context}.raw.task_logprob_gain")):
            raise MechanismFigureError(f"{context}.raw task gain disagrees with token trace")
        if not _close(raw_style_shift, _number(raw_summary.get("style_abs_logprob_shift"), f"{context}.raw.style_abs_logprob_shift")):
            raise MechanismFigureError(f"{context}.raw style shift disagrees with token trace")
        previous_alpha = -1.0
        previous_kl = -1.0
        for epsilon, epsilon_row in zip(epsilon_grid, epsilon_sweep):
            alphas = per_epsilon_alpha[epsilon]
            if max(alphas) - min(alphas) > 1e-8:
                raise MechanismFigureError(f"{context} alpha changes within epsilon={epsilon}")
            alpha = float(alphas[0])
            achieved = float(np.mean(per_epsilon_kls[epsilon]))
            declared_alpha = _number(epsilon_row.get("alpha"), f"{context}.epsilon[{epsilon}].alpha", minimum=0.0, maximum=1.0)
            declared_achieved = _number(epsilon_row.get("achieved_mean_kl"), f"{context}.epsilon[{epsilon}].achieved_mean_kl", minimum=0.0)
            if not _close(alpha, declared_alpha) or not _close(achieved, declared_achieved):
                raise MechanismFigureError(f"{context} epsilon={epsilon} summary disagrees with token trace")
            if achieved > epsilon + max(1e-7, 2e-5 * epsilon):
                raise MechanismFigureError(f"{context} epsilon={epsilon} violates its KL budget")
            task_gain, style_shift = _partition_metrics(categories, per_epsilon_shifts[epsilon])
            if not _close(task_gain, _number(epsilon_row.get("task_logprob_gain"), f"{context}.epsilon[{epsilon}].task_logprob_gain")):
                raise MechanismFigureError(f"{context} epsilon={epsilon} task gain mismatch")
            if not _close(style_shift, _number(epsilon_row.get("style_abs_logprob_shift"), f"{context}.epsilon[{epsilon}].style_abs_logprob_shift")):
                raise MechanismFigureError(f"{context} epsilon={epsilon} style shift mismatch")
            if alpha + 1e-7 < previous_alpha or achieved + 1e-7 < previous_kl:
                raise MechanismFigureError(f"{context} epsilon calibration is non-monotone")
            previous_alpha, previous_kl = alpha, achieved
        wrapper["raw"] = dict(raw_summary)
        wrapper["epsilon_sweep"] = epsilon_sweep
        wrapper["token_trace"] = token_trace
        wrappers.append(wrapper)

    robustness = _mapping(row.get("wrapper_robustness"), "wrapper_robustness")
    if robustness.get("wrapper_set_version") != WRAPPER_SET_VERSION:
        raise MechanismFigureError("wrapper_robustness version mismatch")
    if robustness.get("variance_definition") != "population_variance_across_three_answer_free_wrappers":
        raise MechanismFigureError("wrapper robustness variance definition mismatch")
    return {
        "row": dict(row),
        "query_id": query_id,
        "problem_sha256": problem_sha256,
        "generated_tokens": generated_tokens,
        "selected_epsilon": selected_epsilon,
        "training_epsilon": training_epsilon,
        "epsilon_grid": epsilon_grid,
        "wrappers": wrappers,
        "selection_status": selection_status,
        "checkpoint_type": checkpoint_type,
        "method_id": method_id,
        "variant": variant,
    }


def _moving_block_ci(
    raw_variance: Sequence[float],
    projected_variance: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float, int]:
    raw = np.asarray(raw_variance, dtype=float)
    projected = np.asarray(projected_variance, dtype=float)
    if raw.shape != projected.shape or raw.ndim != 1 or len(raw) < 8:
        raise MechanismFigureError("Paired wrapper variances require at least eight positions")
    if replicates < 100:
        raise MechanismFigureError("At least 100 bootstrap replicates are required")
    difference = raw - projected
    observed = float(np.mean(difference))
    block_length = max(2, int(math.ceil(math.sqrt(len(difference)))))
    blocks = int(math.ceil(len(difference) / block_length))
    offsets = np.arange(block_length)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        starts = rng.integers(0, len(difference), size=blocks)
        positions = ((starts[:, None] + offsets[None, :]) % len(difference)).ravel()
        estimates[index] = float(np.mean(difference[positions[: len(difference)]]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return observed, float(low), float(high), block_length


def _build_tables(
    artifact: Mapping[str, Any], *, bootstrap_replicates: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    wrappers = {wrapper["wrapper_id"]: wrapper for wrapper in artifact["wrappers"]}
    neutral = wrappers["neutral"]
    token_count = int(artifact["generated_tokens"])
    selected_epsilon = float(artifact["selected_epsilon"])
    raw_prefix_sum = 0.0
    projected_prefix_sum = 0.0
    token_rows: list[dict[str, Any]] = []
    raw_variances: list[float] = []
    projected_variances: list[float] = []
    for position in range(token_count):
        traces = {name: wrappers[name]["token_trace"][position] for name in WRAPPER_IDS}
        raw_scores = [float(traces[name]["raw_surrogate_logratio"]) for name in WRAPPER_IDS]
        projected_scores = [
            float(traces[name]["projected_surrogate_logratio"]) for name in WRAPPER_IDS
        ]
        raw_variance = _population_variance(raw_scores)
        projected_variance = _population_variance(projected_scores)
        raw_variances.append(raw_variance)
        projected_variances.append(projected_variance)
        raw_kl = float(neutral["token_trace"][position]["raw_kl"])
        projected_kl = float(neutral["token_trace"][position]["projected_kl"])
        raw_prefix_sum += raw_kl
        projected_prefix_sum += projected_kl
        token_rows.append(
            {
                "query_id": artifact["query_id"],
                "checkpoint_episode": artifact["row"]["checkpoint_episode"],
                "position": position,
                "normalized_position": neutral["token_trace"][position]["normalized_position"],
                "token_id": neutral["token_trace"][position]["token_id"],
                "token_text": neutral["token_trace"][position]["token_text"],
                "token_category": neutral["token_trace"][position]["token_category"],
                "neutral_raw_kl": raw_kl,
                "neutral_projected_kl": projected_kl,
                "neutral_raw_prefix_mean_kl": raw_prefix_sum / (position + 1),
                "neutral_projected_prefix_mean_kl": projected_prefix_sum / (position + 1),
                "selected_epsilon": selected_epsilon,
                "neutral_selected_alpha": neutral["token_trace"][position]["projected_alpha"],
                **{
                    f"{name}_raw_surrogate_logratio": traces[name]["raw_surrogate_logratio"]
                    for name in WRAPPER_IDS
                },
                **{
                    f"{name}_projected_surrogate_logratio": traces[name]["projected_surrogate_logratio"]
                    for name in WRAPPER_IDS
                },
                "raw_across_wrapper_variance": raw_variance,
                "projected_across_wrapper_variance": projected_variance,
                "variance_reduction_raw_minus_projected": raw_variance - projected_variance,
            }
        )

    epsilon_rows: list[dict[str, Any]] = []
    for name in WRAPPER_IDS:
        wrapper = wrappers[name]
        for epsilon_row in wrapper["epsilon_sweep"]:
            epsilon = float(epsilon_row["epsilon"])
            epsilon_rows.append(
                {
                    "query_id": artifact["query_id"],
                    "checkpoint_episode": artifact["row"]["checkpoint_episode"],
                    "wrapper_id": name,
                    "epsilon": epsilon,
                    "alpha": epsilon_row["alpha"],
                    "achieved_mean_kl": epsilon_row["achieved_mean_kl"],
                    "epsilon_slack": epsilon_row["epsilon_slack"],
                    "constraint_active": epsilon_row["constraint_active"],
                    "task_logprob_gain": epsilon_row["task_logprob_gain"],
                    "style_abs_logprob_shift": epsilon_row["style_abs_logprob_shift"],
                    "normalized_surrogate_logratio": epsilon_row["normalized_logratio"],
                    "is_training_budget": epsilon_row["is_training_budget"],
                    "is_posthoc_stress_test": epsilon_row["is_posthoc_stress_test"],
                    "shift_definition": TOKEN_SHIFT_DEFINITION,
                }
            )

    observed, low, high, block_length = _moving_block_ci(
        raw_variances,
        projected_variances,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    neutral_selected = _find_epsilon(
        neutral["epsilon_sweep"], selected_epsilon, "neutral epsilon sweep"
    )
    summary_rows = [
        {
            "schema_version": "trsd-posthoc-mechanism-figure-summary-v1",
            "query_id": artifact["query_id"],
            "problem_sha256": artifact["problem_sha256"],
            "checkpoint_episode": artifact["row"]["checkpoint_episode"],
            "checkpoint_sha256": artifact["row"]["checkpoint_sha256"],
            "selection_status": artifact["selection_status"],
            "checkpoint_type": artifact["checkpoint_type"],
            "method_id": artifact["method_id"],
            "variant": artifact["variant"],
            "generated_tokens": token_count,
            "selected_epsilon": selected_epsilon,
            "training_epsilon": artifact["training_epsilon"],
            "neutral_raw_mean_kl": neutral["raw"]["mean_kl"],
            "neutral_selected_projected_mean_kl": neutral_selected["achieved_mean_kl"],
            "neutral_selected_alpha": neutral_selected["alpha"],
            "neutral_selected_constraint_active": neutral_selected["constraint_active"],
            "mean_raw_across_wrapper_variance": float(np.mean(raw_variances)),
            "mean_projected_across_wrapper_variance": float(np.mean(projected_variances)),
            "paired_variance_reduction": observed,
            "paired_variance_reduction_ci_low": low,
            "paired_variance_reduction_ci_high": high,
            "bootstrap_unit": "paired_circular_token_block",
            "bootstrap_block_length": block_length,
            "bootstrap_replicates": bootstrap_replicates,
            "query_replicates": 1,
        }
    ]
    return token_rows, epsilon_rows, summary_rows


def _configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _panel(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = min(max(1, window), len(values))
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    body = (cumulative[window:] - cumulative[:-window]) / window
    prefix = np.asarray([values[: index + 1].mean() for index in range(window - 1)])
    return np.concatenate([prefix, body])


def _plot(
    artifact: Mapping[str, Any],
    token_rows: Sequence[Mapping[str, Any]],
    epsilon_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> plt.Figure:
    _configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.87, wspace=0.30, hspace=0.40)
    positions = np.asarray([int(row["position"]) for row in token_rows])
    raw_kl = np.asarray([float(row["neutral_raw_kl"]) for row in token_rows])
    projected_kl = np.asarray([float(row["neutral_projected_kl"]) for row in token_rows])
    selected_epsilon = float(artifact["selected_epsilon"])
    training_epsilon = float(artifact["training_epsilon"])
    positive_kl = np.concatenate([raw_kl[raw_kl > 0], projected_kl[projected_kl > 0]])
    kl_floor = (
        max(1e-12, float(np.quantile(positive_kl, 0.01) * 0.2))
        if len(positive_kl)
        else 1e-12
    )
    window = max(5, min(101, 2 * int(math.sqrt(len(positions))) + 1))
    stride = max(1, len(positions) // 1600)

    axis = axes[0, 0]
    axis.scatter(
        positions[::stride],
        np.maximum(raw_kl[::stride], kl_floor),
        s=4,
        color="#D55E00",
        alpha=0.14,
        linewidth=0,
        rasterized=True,
    )
    axis.scatter(
        positions[::stride],
        np.maximum(projected_kl[::stride], kl_floor),
        s=4,
        color="#0072B2",
        alpha=0.18,
        linewidth=0,
        rasterized=True,
    )
    axis.plot(
        positions,
        np.maximum(_rolling_mean(raw_kl, window), kl_floor),
        color="#D55E00",
        linewidth=1.25,
        label=f"raw surrogate ({window}-token mean)",
    )
    axis.plot(
        positions,
        np.maximum(_rolling_mean(projected_kl, window), kl_floor),
        color="#0072B2",
        linewidth=1.35,
        label=f"projected ({window}-token mean)",
    )
    axis.axhline(selected_epsilon, color="#111827", linestyle="--", linewidth=0.9, label="post-hoc ε")
    axis.set_yscale("log")
    axis.set_xlabel("Token position")
    axis.set_ylabel(r"Per-token $D_{KL}(q\Vert p)$")
    axis.set_title("Local surrogate divergence (neutral wrapper)")
    axis.grid(axis="y", which="both", color="#E5E7EB", linewidth=0.55)
    axis.legend(frameon=False, loc="upper right")
    _panel(axis, "a")

    axis = axes[0, 1]
    raw_prefix = np.asarray([float(row["neutral_raw_prefix_mean_kl"]) for row in token_rows])
    projected_prefix = np.asarray(
        [float(row["neutral_projected_prefix_mean_kl"]) for row in token_rows]
    )
    axis.plot(positions, raw_prefix, color="#D55E00", linewidth=1.15, label="raw prefix mean")
    axis.plot(positions, projected_prefix, color="#0072B2", linewidth=1.35, label="projected prefix mean")
    axis.axhline(selected_epsilon, color="#111827", linestyle="--", linewidth=0.9, label=f"post-hoc ε={selected_epsilon:g}")
    axis.axhline(training_epsilon, color="#7C3AED", linestyle=":", linewidth=1.0, label=f"training ε={training_epsilon:g}")
    axis.scatter([positions[-1]], [projected_prefix[-1]], s=36, color="#0072B2", edgecolor="white", zorder=4)
    axis.set_xlabel("Prefix endpoint")
    axis.set_ylabel("Cumulative mean KL")
    axis.set_title("Global budget calibration along the trajectory")
    axis.grid(color="#E5E7EB", linewidth=0.6)
    axis.legend(frameon=False, loc="upper right")
    inset = axis.inset_axes([0.11, 0.12, 0.41, 0.38])
    wrapper_colors = {"neutral": "#0072B2", "terse": "#009E73", "verbose": "#CC79A7"}
    achieved_floor = 1e-12
    for wrapper_id in WRAPPER_IDS:
        rows = sorted(
            [row for row in epsilon_rows if row["wrapper_id"] == wrapper_id],
            key=lambda value: float(value["epsilon"]),
        )
        inset.plot(
            [float(row["epsilon"]) for row in rows],
            [max(float(row["achieved_mean_kl"]), achieved_floor) for row in rows],
            marker="o",
            markersize=2.5,
            linewidth=0.8,
            color=wrapper_colors[wrapper_id],
        )
    epsilon_grid = np.asarray(artifact["epsilon_grid"], dtype=float)
    inset.plot(epsilon_grid, epsilon_grid, color="#6B7280", linestyle="--", linewidth=0.7)
    inset.set_xscale("log")
    inset.set_yscale("log")
    inset.set_xlabel("budget ε", fontsize=6.4)
    inset.set_ylabel("achieved KL", fontsize=6.4)
    inset.set_title("all wrappers", fontsize=6.8)
    inset.tick_params(labelsize=5.8)
    inset.spines["top"].set_visible(True)
    inset.spines["right"].set_visible(True)
    _panel(axis, "b")

    axis = axes[1, 0]
    raw_variance = np.asarray([float(row["raw_across_wrapper_variance"]) for row in token_rows])
    projected_variance = np.asarray(
        [float(row["projected_across_wrapper_variance"]) for row in token_rows]
    )
    positive_variance = np.concatenate(
        [raw_variance[raw_variance > 0], projected_variance[projected_variance > 0]]
    )
    variance_floor = (
        max(1e-16, float(np.quantile(positive_variance, 0.01) * 0.15))
        if len(positive_variance)
        else 1e-16
    )
    plot_raw = np.maximum(raw_variance, variance_floor)
    plot_projected = np.maximum(projected_variance, variance_floor)
    axis.scatter(
        plot_raw,
        plot_projected,
        s=7,
        alpha=0.22,
        color="#0072B2",
        edgecolor="none",
        rasterized=True,
    )
    lower = min(float(plot_raw.min()), float(plot_projected.min()))
    upper = max(float(plot_raw.max()), float(plot_projected.max()))
    axis.plot([lower, upper], [lower, upper], color="#111827", linestyle="--", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Raw across-wrapper variance")
    axis.set_ylabel("Projected across-wrapper variance")
    axis.set_title("Prompt sensitivity paired at each token")
    axis.grid(which="both", color="#E5E7EB", linewidth=0.5)
    axis.text(
        0.03,
        0.97,
        (
            f"mean(raw − projected) = {float(summary['paired_variance_reduction']):.3g}\n"
            f"95% paired block CI "
            f"[{float(summary['paired_variance_reduction_ci_low']):.3g}, "
            f"{float(summary['paired_variance_reduction_ci_high']):.3g}]\n"
            f"circular block = {summary['bootstrap_block_length']} tokens"
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92, "pad": 3},
    )
    axis.text(
        0.98,
        0.03,
        "exact zeros shown at visual floor",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#6B7280",
    )
    _panel(axis, "c")

    axis = axes[1, 1]
    by_wrapper = {
        wrapper_id: sorted(
            [row for row in epsilon_rows if row["wrapper_id"] == wrapper_id],
            key=lambda value: float(value["epsilon"]),
        )
        for wrapper_id in WRAPPER_IDS
    }
    neutral_rows = by_wrapper["neutral"]
    eps = np.asarray([float(row["epsilon"]) for row in neutral_rows])
    neutral_style = np.asarray([float(row["style_abs_logprob_shift"]) for row in neutral_rows])
    neutral_task = np.asarray([float(row["task_logprob_gain"]) for row in neutral_rows])
    neutral_alpha = np.asarray([float(row["alpha"]) for row in neutral_rows])
    style_all = np.asarray(
        [[float(row["style_abs_logprob_shift"]) for row in by_wrapper[name]] for name in WRAPPER_IDS]
    )
    task_all = np.asarray(
        [[float(row["task_logprob_gain"]) for row in by_wrapper[name]] for name in WRAPPER_IDS]
    )
    axis.errorbar(
        neutral_style,
        neutral_task,
        xerr=np.vstack([neutral_style - style_all.min(axis=0), style_all.max(axis=0) - neutral_style]),
        yerr=np.vstack([neutral_task - task_all.min(axis=0), task_all.max(axis=0) - neutral_task]),
        fmt="none",
        ecolor="#9CA3AF",
        elinewidth=0.8,
        capsize=2,
        zorder=1,
    )
    axis.plot(neutral_style, neutral_task, color="#9CA3AF", linewidth=0.9, zorder=1)
    points = axis.scatter(
        neutral_style,
        neutral_task,
        c=eps,
        norm=LogNorm(vmin=float(eps.min()), vmax=float(eps.max())),
        cmap="viridis",
        s=32 + 70 * neutral_alpha,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    selected_index = int(np.argmin(np.abs(eps - selected_epsilon)))
    training_index = int(np.argmin(np.abs(eps - training_epsilon)))
    axis.scatter(
        [neutral_style[selected_index]],
        [neutral_task[selected_index]],
        marker="*",
        s=170,
        facecolor="none",
        edgecolor="#111827",
        linewidth=1.0,
        zorder=4,
        label="post-hoc ε",
    )
    axis.scatter(
        [neutral_style[training_index]],
        [neutral_task[training_index]],
        marker="s",
        s=85,
        facecolor="none",
        edgecolor="#7C3AED",
        linewidth=1.1,
        zorder=4,
        label="training ε",
    )
    for index in sorted({0, selected_index, training_index, len(eps) - 1}):
        axis.annotate(
            f"ε={eps[index]:g}, α={neutral_alpha[index]:.2f}",
            (neutral_style[index], neutral_task[index]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    axis.axhline(0, color="#9CA3AF", linewidth=0.7)
    axis.set_xlabel("Style-token |log-prob shift| (↓)")
    axis.set_ylabel("Task-token log-prob gain (↑)")
    axis.set_title("Task/style response across KL budgets")
    axis.grid(color="#E5E7EB", linewidth=0.6)
    axis.legend(frameon=False, loc="best")
    colorbar = fig.colorbar(points, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("KL budget ε", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    _panel(axis, "d")

    fig.suptitle(
        "Trust-region hidden assumptions on one observed trajectory",
        fontsize=13.5,
        fontweight="bold",
        x=0.08,
        ha="left",
        y=0.98,
    )
    endpoint_description = (
        "latest loadable TRSD endpoint (historical rolling episode 36)"
        if artifact["selection_status"] == "historical_latest_rolling_endpoint"
        else "completed scientific final TRSD checkpoint"
    )
    fig.text(
        0.08,
        0.932,
        (
            f"{endpoint_description}; "
            f"query index {artifact['row']['query_index']}; {artifact['generated_tokens']:,} tokens. "
            "Neutral/terse/verbose prompts are answer-free; error ranges in (d) span wrappers."
        ),
        fontsize=8.1,
        color="#4B5563",
        ha="left",
    )
    fig.text(
        0.08,
        0.012,
        (
            "Post-hoc ε is a stress-test projection, not the training budget. "
            "The bootstrap in (c) quantifies within-trajectory paired sensitivity only (one query replicate)."
        ),
        fontsize=7.5,
        color="#4B5563",
        ha="left",
    )
    return fig


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise MechanismFigureError(f"Refusing to write empty CSV {path.name}")
    fields = list(rows[0].keys())
    if any(list(row.keys()) != fields for row in rows):
        raise MechanismFigureError(f"CSV {path.name} rows have inconsistent fields")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(
    *,
    posthoc_path: str | Path,
    labels_path: str | Path,
    output_prefix: str | Path,
    bootstrap_replicates: int,
    seed: int,
) -> list[Path]:
    labels = load_sealed_labels(labels_path)
    artifact = _validate_artifact(_one_row_jsonl(posthoc_path), labels)
    token_rows, epsilon_rows, summary_rows = _build_tables(
        artifact,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trsd-mechanism-plot-", dir=prefix.parent) as temporary:
        staging = Path(temporary)
        staged_prefix = staging / prefix.name
        _write_csv(staged_prefix.with_suffix(".csv"), token_rows)
        _write_csv(staging / f"{prefix.name}_epsilon_sweep.csv", epsilon_rows)
        _write_csv(staging / f"{prefix.name}_summary.csv", summary_rows)
        figure = _plot(artifact, token_rows, epsilon_rows, summary_rows[0])
        for suffix in ("png", "pdf"):
            path = staged_prefix.with_suffix(f".{suffix}")
            metadata: dict[str, Any] = {
                "Creator": "TRSD fail-closed mechanism figure builder"
            }
            if suffix == "pdf":
                metadata["CreationDate"] = None
            figure.savefig(
                path,
                format=suffix,
                dpi=320,
                bbox_inches="tight",
                pad_inches=0.04,
                metadata=metadata,
            )
        plt.close(figure)
        published: list[Path] = []
        for source in sorted(staging.iterdir()):
            destination = prefix.parent / source.name
            os.replace(source, destination)
            published.append(destination)
    return published


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--posthoc", required=True)
    root.add_argument("--labels", required=True)
    root.add_argument("--output-prefix", required=True)
    root.add_argument("--bootstrap-replicates", type=int, default=10_000)
    root.add_argument("--seed", type=int, default=20260807)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        outputs = build(
            posthoc_path=args.posthoc,
            labels_path=args.labels,
            output_prefix=args.output_prefix,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    except (MechanismFigureError, ValueError, OSError) as exc:
        raise SystemExit(f"Refusing mechanism figure build: {exc}") from exc
    print(json.dumps({"outputs": [str(path) for path in outputs]}, sort_keys=True))


if __name__ == "__main__":
    main()
