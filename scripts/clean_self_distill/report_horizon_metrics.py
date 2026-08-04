#!/usr/bin/env python3
"""Derive pre-registered short/long/hindsight evidence from a validated PoC report.

This script never runs a model and never selects a subset using CSD outcomes.
Target-horizon strata are defined only by the Base response length.  The
distillation horizon is the maximum contiguous prefix in any one rollout; the
lengths of independently generated rollouts are deliberately never summed and
reported as a causal horizon.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "clean-self-distill-horizon-report-v1"
METHODS = ("Base", "Privileged Control", "CSD-T", "CSD-SD")
WINDOWS = ((0, 512), (512, 1024), (1024, 2048), (2048, 4096))
WINDOW_METRICS = (
    "pre_update_mean_teacher_student_kl",
    "pre_update_teacher_student_top1_agreement",
    "pre_update_mean_teacher_base_ridge_shift_l2",
)


class HorizonReportError(ValueError):
    """Raised when a supposedly validated report violates the horizon contract."""


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise HorizonReportError(f"{context}.{key} is required")
    return mapping[key]


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HorizonReportError(f"{context} must be an object")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HorizonReportError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HorizonReportError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise HorizonReportError(f"{context} is outside its finite numeric range")
    return result


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise HorizonReportError(f"{context} must be boolean")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HorizonReportError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HorizonReportError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    raise HorizonReportError(f"{path}:{line_number} is blank")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise HorizonReportError(f"{path}:{line_number} is not an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HorizonReportError(f"Cannot read valid JSONL from {path}: {exc}") from exc
    if not rows:
        raise HorizonReportError(f"{path} is empty")
    ids = [str(_required(row, "query_id", f"{path} row {i}")).strip() for i, row in enumerate(rows, 1)]
    if any(not query_id for query_id in ids) or len(ids) != len(set(ids)):
        raise HorizonReportError(f"{path} has empty or duplicate query IDs")
    return rows


def _nearest_rank(values: Sequence[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(math.ceil(probability * len(ordered)) - 1, 0)
    return ordered[index]


def _exact_paired_binomial_p(wrong_to_correct: int, correct_to_wrong: int) -> float | None:
    discordant = wrong_to_correct + correct_to_wrong
    if discordant == 0:
        return None
    tail = min(wrong_to_correct, correct_to_wrong)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _audit_metrics(values: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    counts = [value.get("audit_counts") for value in values]
    if method == "Base":
        if any(count is not None for count in counts):
            raise HorizonReportError("Base must not declare teacher audit counts")
        return {"HER": None, "CPP": None, "HFS": None, "audit_totals": None}
    if any(not isinstance(count, Mapping) for count in counts):
        raise HorizonReportError(f"{method} has missing raw audit counts")
    keys = (
        "teacher_context_events",
        "forbidden_context_events",
        "comparison_events",
        "compared_token_positions",
        "same_prefix_positions",
    )
    totals = {
        key: sum(
            _integer(count[key], f"{method}.audit_counts.{key}")  # type: ignore[index]
            for count in counts
        )
        for key in keys
    }
    if totals["forbidden_context_events"] > totals["teacher_context_events"]:
        raise HorizonReportError(f"{method} forbidden events exceed teacher events")
    if totals["same_prefix_positions"] > totals["compared_token_positions"]:
        raise HorizonReportError(f"{method} same-prefix positions exceed compared positions")
    her = (
        totals["forbidden_context_events"] / totals["teacher_context_events"]
        if totals["teacher_context_events"]
        else 0.0
    )
    cpp = (
        totals["same_prefix_positions"] / totals["compared_token_positions"]
        if totals["compared_token_positions"]
        else 0.0
    )
    return {"HER": her, "CPP": cpp, "HFS": (1.0 - her) * cpp, "audit_totals": totals}


def _nll_summary(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any] | None:
    if method == "CSD-T":
        values = [
            _number(
                _required(_mapping(row["task1_artifact"], "task1_artifact"), "target_answer_nll_gain", "task1_artifact"),
                "task1_artifact.target_answer_nll_gain",
            )
            for row in rows
        ]
    elif method == "CSD-SD":
        values = [
            _number(
                _required(_mapping(row["task2_artifact"], "task2_artifact"), "distilled_target_answer_nll_gain", "task2_artifact"),
                "task2_artifact.distilled_target_answer_nll_gain",
            )
            for row in rows
        ]
    else:
        return None
    return {
        "n": len(values),
        "mean_gain": fmean(values),
        "median_gain": median(values),
        "positive_count": sum(value > 0.0 for value in values),
        "positive_rate": fmean(float(value > 0.0) for value in values),
    }


def _condition_summary(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "accuracy": None,
            "gain_vs_base_pp": None,
            "wrong_to_correct": 0,
            "correct_to_wrong": 0,
            "paired_exact_p": None,
            "HER": None,
            "CPP": None,
            "HFS": None,
            "HFAG_pp": None,
            "audit_totals": None,
            "truncation_count": 0,
            "mean_generated_tokens": None,
            "target_answer_nll_gain": None,
        }
    conditions = [
        _mapping(_required(row, "conditions", "row"), "row.conditions") for row in rows
    ]
    method_values = [
        _mapping(_required(condition, method, "row.conditions"), f"row.conditions.{method}")
        for condition in conditions
    ]
    base_values = [
        _mapping(_required(condition, "Base", "row.conditions"), "row.conditions.Base")
        for condition in conditions
    ]
    correct = [_number(_required(value, "correct", method), f"{method}.correct", minimum=0.0) for value in method_values]
    base_correct = [_number(_required(value, "correct", "Base"), "Base.correct", minimum=0.0) for value in base_values]
    if any(value not in {0.0, 1.0} for value in [*correct, *base_correct]):
        raise HorizonReportError("Formal Acc@1 rows must contain binary correctness")
    accuracy = fmean(correct)
    base_accuracy = fmean(base_correct)
    wrong_to_correct = sum(base == 0.0 and value == 1.0 for base, value in zip(base_correct, correct))
    correct_to_wrong = sum(base == 1.0 and value == 0.0 for base, value in zip(base_correct, correct))
    audit = _audit_metrics(method_values, method)
    generated_tokens = [
        _number(_required(value, "generated_tokens", method), f"{method}.generated_tokens", minimum=0.0)
        for value in method_values
    ]
    truncated = [
        _boolean(_required(value, "truncated", method), f"{method}.truncated")
        for value in method_values
    ]
    gain_pp = None if method == "Base" else 100.0 * (accuracy - base_accuracy)
    hfs = audit["HFS"]
    return {
        "n": len(rows),
        "accuracy": accuracy,
        "gain_vs_base_pp": gain_pp,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "paired_exact_p": (
            None
            if method == "Base"
            else _exact_paired_binomial_p(wrong_to_correct, correct_to_wrong)
        ),
        **audit,
        "HFAG_pp": None if gain_pp is None or hfs is None else hfs * gain_pp,
        "truncation_count": sum(truncated),
        "mean_generated_tokens": fmean(generated_tokens),
        "target_answer_nll_gain": _nll_summary(rows, method),
    }


def _target_horizon(row: Mapping[str, Any], split_tokens: int) -> str:
    conditions = _mapping(_required(row, "conditions", "row"), "row.conditions")
    base = _mapping(_required(conditions, "Base", "row.conditions"), "row.conditions.Base")
    tokens = _number(_required(base, "generated_tokens", "Base"), "Base.generated_tokens", minimum=0.0)
    truncated = _boolean(_required(base, "truncated", "Base"), "Base.truncated")
    return "long" if truncated or tokens > split_tokens else "short"


def _performance(rows: Sequence[Mapping[str, Any]], split_tokens: int) -> list[dict[str, Any]]:
    dataset_scopes = {
        "overall": list(rows),
        "amc23": [row for row in rows if row.get("source") == "amc23"],
        "aime": [row for row in rows if row.get("source") in {"aime24", "aime25"}],
    }
    stage = {
        "Base": "direct_reference",
        "Privileged Control": "immediate_auxiliary_context",
        "CSD-T": "immediate_temporary_teacher",
        "CSD-SD": "post_teacher_retained_student",
    }
    output: list[dict[str, Any]] = []
    for dataset_scope, scoped_rows in dataset_scopes.items():
        horizon_scopes = {
            "all": scoped_rows,
            "short": [row for row in scoped_rows if _target_horizon(row, split_tokens) == "short"],
            "long": [row for row in scoped_rows if _target_horizon(row, split_tokens) == "long"],
        }
        for target_horizon, horizon_rows in horizon_scopes.items():
            for method in METHODS:
                output.append(
                    {
                        "dataset_scope": dataset_scope,
                        "target_horizon": target_horizon,
                        "target_horizon_definition": (
                            "all"
                            if target_horizon == "all"
                            else f"Base output {'<=' if target_horizon == 'short' else '>'}{split_tokens} tokens"
                        ),
                        "method": method,
                        "performance_stage": stage[method],
                        **_condition_summary(horizon_rows, method),
                    }
                )
    return output


def _temporal_retention(performance: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare the active temporary teacher with the post-destruction student."""
    output: dict[str, Any] = {}
    for scope in ("overall", "amc23", "aime"):
        indexed = {
            str(row["method"]): row
            for row in performance
            if row["dataset_scope"] == scope and row["target_horizon"] == "all"
        }
        if set(indexed) != set(METHODS):
            raise HorizonReportError(f"Missing all-horizon temporal rows for {scope}")
        base = indexed["Base"]["accuracy"]
        privileged = indexed["Privileged Control"]["accuracy"]
        teacher = indexed["CSD-T"]["accuracy"]
        student = indexed["CSD-SD"]["accuracy"]
        if base is None:
            output[scope] = {
                "n": 0,
                "immediate_privileged_accuracy": None,
                "immediate_csd_t_accuracy": None,
                "post_teacher_csd_sd_accuracy": None,
                "teacher_gain_pp": None,
                "student_retained_gain_pp": None,
                "teacher_gain_retention": None,
            }
            continue
        teacher_gain = float(teacher) - float(base)
        student_gain = float(student) - float(base)
        output[scope] = {
            "n": indexed["Base"]["n"],
            "immediate_privileged_accuracy": privileged,
            "immediate_csd_t_accuracy": teacher,
            "post_teacher_csd_sd_accuracy": student,
            "teacher_gain_pp": 100.0 * teacher_gain,
            "student_retained_gain_pp": 100.0 * student_gain,
            "teacher_gain_retention": (
                student_gain / teacher_gain if teacher_gain > 0.0 else None
            ),
            "interpretation": (
                "CSD-SD is evaluated after teacher destruction; Privileged has no persistent student update"
            ),
        }
    return output


def _validate_window(
    window: Any,
    *,
    expected_start: int,
    expected_end: int,
    prefix_tokens: int,
    context: str,
) -> dict[str, Any]:
    item = _mapping(window, context)
    token_count = max(min(prefix_tokens, expected_end) - expected_start, 0)
    if (
        _required(item, "start_token", context) != expected_start
        or _required(item, "end_token", context) != expected_end
        or _required(item, "measurement_point", context) != "pre_update"
        or _integer(_required(item, "token_count", context), f"{context}.token_count") != token_count
    ):
        raise HorizonReportError(f"{context} does not match its pre-registered window")
    normalized = {
        "start_token": expected_start,
        "end_token": expected_end,
        "token_count": token_count,
    }
    for key in WINDOW_METRICS:
        value = _required(item, key, context)
        if token_count == 0:
            if value is not None:
                raise HorizonReportError(f"{context}.{key} must be null for an empty window")
            normalized[key] = None
            continue
        number = _number(value, f"{context}.{key}")
        if key.endswith("top1_agreement") and not 0.0 <= number <= 1.0:
            raise HorizonReportError(f"{context}.{key} must be in [0,1]")
        if not key.endswith("top1_agreement") and number < -1e-6:
            raise HorizonReportError(f"{context}.{key} must be nonnegative")
        normalized[key] = max(number, 0.0)
    return normalized


def _horizon_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    eval_horizon: int,
    long_prefix_threshold: int,
    late_window_start: int,
    min_heldout_late_queries: int,
) -> dict[str, Any]:
    ready_rows: list[Mapping[str, Any]] = []
    prefix_lengths: list[int] = []
    compared_positions = 0
    qualified_queries = 0
    late_query_ids: set[str] = set()
    heldout_late_query_ids: set[str] = set()
    threshold_query_ids: set[str] = set()
    complete_query_ids: set[str] = set()
    window_sums: dict[tuple[int, int], dict[str, float]] = {
        window: {key: 0.0 for key in WINDOW_METRICS} for window in WINDOWS
    }
    window_counts: Counter[tuple[int, int]] = Counter()

    for row_index, row in enumerate(rows):
        context = f"merged row {row_index + 1} ({row.get('query_id')})"
        task2 = _mapping(_required(row, "task2_artifact", context), f"{context}.task2_artifact")
        no_op = _boolean(_required(row, "specialization_no_op", context), f"{context}.specialization_no_op")
        config = _mapping(_required(task2, "distillation_config", context), f"{context}.distillation_config")
        cap = _integer(_required(config, "prefix_max_new_tokens", context), f"{context}.prefix_max_new_tokens", minimum=1)
        minimum = _integer(_required(config, "long_horizon_min_tokens", context), f"{context}.long_horizon_min_tokens", minimum=1)
        if cap < long_prefix_threshold or minimum < long_prefix_threshold or cap < minimum:
            raise HorizonReportError(
                f"{context} does not provide the pre-registered {long_prefix_threshold}-token opportunity"
            )
        trace = _required(task2, "distillation_trace", context)
        if not isinstance(trace, list):
            raise HorizonReportError(f"{context}.distillation_trace must be a list")
        completed = _integer(_required(task2, "distillation_steps_completed", context), f"{context}.steps")
        if no_op:
            if trace or completed != 0:
                raise HorizonReportError(f"{context} no-op must have an empty trace")
            continue
        if not trace or completed != len(trace):
            raise HorizonReportError(f"{context} ready row has incomplete distillation trace")
        ready_rows.append(row)
        query_id = str(row["query_id"])
        query_qualified = False
        for step_index, raw_step in enumerate(trace):
            step_context = f"{context}.distillation_trace[{step_index}]"
            step = _mapping(raw_step, step_context)
            prefix_tokens = _integer(_required(step, "prefix_tokens", step_context), f"{step_context}.prefix_tokens", minimum=1)
            if prefix_tokens > cap:
                raise HorizonReportError(f"{step_context} exceeds its configured cap")
            truncated = _boolean(_required(step, "prefix_truncated", step_context), f"{step_context}.prefix_truncated")
            complete = _boolean(_required(step, "trajectory_complete", step_context), f"{step_context}.trajectory_complete")
            qualified = _boolean(_required(step, "long_horizon_qualified", step_context), f"{step_context}.long_horizon_qualified")
            expected_qualified = complete or prefix_tokens >= minimum
            if qualified is not expected_qualified or (complete and truncated):
                raise HorizonReportError(f"{step_context} has inconsistent completion/qualification")
            if truncated != (prefix_tokens == cap and not complete):
                raise HorizonReportError(f"{step_context}.prefix_truncated disagrees with cap/EOS")
            query_qualified = query_qualified or qualified
            prefix_lengths.append(prefix_tokens)
            compared_positions += prefix_tokens
            if prefix_tokens > late_window_start:
                late_query_ids.add(query_id)
                if row.get("source") in {"aime24", "aime25"}:
                    heldout_late_query_ids.add(query_id)
            if prefix_tokens >= minimum:
                threshold_query_ids.add(query_id)
            if complete:
                complete_query_ids.add(query_id)
            windows = _required(step, "horizon_windows", step_context)
            if not isinstance(windows, list) or len(windows) != len(WINDOWS):
                raise HorizonReportError(f"{step_context}.horizon_windows must contain four windows")
            for window_index, ((start, end), raw_window) in enumerate(zip(WINDOWS, windows)):
                window = _validate_window(
                    raw_window,
                    expected_start=start,
                    expected_end=end,
                    prefix_tokens=prefix_tokens,
                    context=f"{step_context}.horizon_windows[{window_index}]",
                )
                count = window["token_count"]
                window_counts[(start, end)] += count
                for key in WINDOW_METRICS:
                    if count:
                        window_sums[(start, end)][key] += count * float(window[key])
        if not query_qualified or task2.get("long_horizon_qualified") is not True:
            raise HorizonReportError(f"{context} has no qualified long-horizon trajectory")
        qualified_queries += 1

    window_metrics = []
    for start, end in WINDOWS:
        count = window_counts[(start, end)]
        window_metrics.append(
            {
                "start_token": start,
                "end_token": end,
                "sampled_positions": count,
                "measurement_point": "pre_update",
                **{
                    key: (window_sums[(start, end)][key] / count if count else None)
                    for key in WINDOW_METRICS
                },
            }
        )

    ready_count = len(ready_rows)
    heldout_ready_count = sum(row.get("source") in {"aime24", "aime25"} for row in ready_rows)
    heldout_late_count = len(heldout_late_query_ids)
    configuration_pass = True  # Every row was checked above, including no-ops.
    qualification_pass = ready_count > 0 and qualified_queries == ready_count
    heldout_late_pass = heldout_late_count >= min_heldout_late_queries
    return {
        "definition": {
            "H_train": "maximum contiguous same-prefix tokens in one rollout; independent rollouts are not summed",
            "H_eval": "maximum final evaluation decode tokens",
            "long_prefix_threshold_tokens": long_prefix_threshold,
            "late_window_start_tokens": late_window_start,
            "window_metrics_measurement_point": "pre_update",
            "scope_limit": "query-local post-teacher retention, not cross-query continual learning",
        },
        "ready_queries": ready_count,
        "heldout_ready_queries": heldout_ready_count,
        "sampled_rollouts": len(prefix_lengths),
        "sampled_positions_across_independent_rollouts": compared_positions,
        "H_train_tokens": max(prefix_lengths, default=0),
        "H_eval_tokens": eval_horizon,
        "H_train_over_H_eval": (max(prefix_lengths) / eval_horizon if prefix_lengths else 0.0),
        "prefix_length_p50": _nearest_rank(prefix_lengths, 0.50),
        "prefix_length_p90": _nearest_rank(prefix_lengths, 0.90),
        "qualified_query_count": qualified_queries,
        "qualified_query_rate": (qualified_queries / ready_count if ready_count else None),
        "threshold_reached_query_count": len(threshold_query_ids),
        "naturally_completed_query_count": len(complete_query_ids),
        "late_coverage_query_count": len(late_query_ids),
        "late_coverage_query_rate": (len(late_query_ids) / ready_count if ready_count else None),
        "heldout_late_coverage_query_count": heldout_late_count,
        "window_metrics": window_metrics,
        "claim_gate": {
            "formal_configuration_pass": configuration_pass,
            "ready_query_qualification_pass": qualification_pass,
            "heldout_late_count_required": min_heldout_late_queries,
            "heldout_late_count_observed": heldout_late_count,
            "heldout_late_coverage_pass": heldout_late_pass,
            "long_horizon_evidence_pass": configuration_pass and qualification_pass and heldout_late_pass,
            "interpretation": (
                "pre-update late-prefix signal coverage plus post-teacher final performance are reportable"
                if configuration_pass and qualification_pass and heldout_late_pass
                else "long-prefix evidence is insufficient; do not claim late-prefix stability"
            ),
        },
    }


def build_horizon_report(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    split_tokens: int,
    long_prefix_threshold: int,
    late_window_start: int,
    min_heldout_late_queries: int,
) -> dict[str, Any]:
    validation = _mapping(_required(summary, "validation", "summary"), "summary.validation")
    if validation.get("status") != "passed":
        raise HorizonReportError("Core report validation did not pass")
    expected_rows = _integer(_required(validation, "unique_query_count", "summary.validation"), "summary.validation.unique_query_count", minimum=1)
    if len(rows) != expected_rows:
        raise HorizonReportError(f"Merged rows={len(rows)} disagree with validated count={expected_rows}")
    for name, value in (
        ("split_tokens", split_tokens),
        ("long_prefix_threshold", long_prefix_threshold),
        ("late_window_start", late_window_start),
        ("min_heldout_late_queries", min_heldout_late_queries),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise HorizonReportError(f"{name} must be a positive integer")
    eval_horizon = _integer(_required(summary, "max_tokens", "summary"), "summary.max_tokens", minimum=1)
    if not (late_window_start < long_prefix_threshold <= eval_horizon):
        raise HorizonReportError("Require late_window_start < long_prefix_threshold <= H_eval")
    sources = Counter(str(row.get("source", "")) for row in rows)
    performance = _performance(rows, split_tokens)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_counts": dict(sorted(sources.items())),
        "pre_registration": {
            "target_horizon_split_tokens": split_tokens,
            "target_horizon_subset_source": "Base generated-token count only",
            "long_prefix_threshold_tokens": long_prefix_threshold,
            "late_window_start_tokens": late_window_start,
            "min_heldout_late_queries": min_heldout_late_queries,
        },
        "performance": performance,
        "temporal_retention": _temporal_retention(performance),
        "distillation_horizon": _horizon_evidence(
            rows,
            eval_horizon=eval_horizon,
            long_prefix_threshold=long_prefix_threshold,
            late_window_start=late_window_start,
            min_heldout_late_queries=min_heldout_late_queries,
        ),
    }


def _csv_text(performance: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "dataset_scope",
        "target_horizon",
        "target_horizon_definition",
        "method",
        "performance_stage",
        "n",
        "accuracy",
        "gain_vs_base_pp",
        "wrong_to_correct",
        "correct_to_wrong",
        "paired_exact_p",
        "HER",
        "CPP",
        "HFS",
        "HFAG_pp",
        "truncation_count",
        "mean_generated_tokens",
        "mean_target_nll_gain",
        "median_target_nll_gain",
        "positive_target_nll_rate",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in performance:
        flattened = dict(row)
        nll = row.get("target_answer_nll_gain")
        flattened.update(
            {
                "mean_target_nll_gain": (
                    nll.get("mean_gain") if isinstance(nll, Mapping) else None
                ),
                "median_target_nll_gain": (
                    nll.get("median_gain") if isinstance(nll, Mapping) else None
                ),
                "positive_target_nll_rate": (
                    nll.get("positive_rate") if isinstance(nll, Mapping) else None
                ),
            }
        )
        writer.writerow({field: flattened.get(field) for field in fields})
    return output.getvalue()


def _markdown_text(report: Mapping[str, Any]) -> str:
    rows = [
        row
        for row in report["performance"]
        if row["dataset_scope"] in {"overall", "aime"}
        and row["target_horizon"] in {"all", "long"}
    ]
    lines = [
        "# Short/Long/Hindsight Results",
        "",
        "Short/long target strata are fixed by Base output length; CSD outcomes never define the subset.",
        "",
        "| Scope | Target horizon | Method | Stage | N | Acc@1 | Gain (pp) | W→C | C→W | HER | CPP | HFS | HFAG (pp) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(key: str, digits: int = 4) -> str:
            value = row.get(key)
            return "N/A" if value is None else f"{float(value):.{digits}f}"

        lines.append(
            "| {dataset_scope} | {target_horizon} | {method} | {performance_stage} | {n} | {acc} | {gain} | {w2c} | {c2w} | {her} | {cpp} | {hfs} | {hfag} |".format(
                **row,
                acc=fmt("accuracy"),
                gain=fmt("gain_vs_base_pp", 2),
                w2c=row["wrong_to_correct"],
                c2w=row["correct_to_wrong"],
                her=fmt("HER"),
                cpp=fmt("CPP"),
                hfs=fmt("HFS"),
                hfag=fmt("HFAG_pp", 2),
            )
        )
    horizon = report["distillation_horizon"]
    gate = horizon["claim_gate"]
    lines.extend(
        [
            "",
            "## Distillation horizon audit",
            "",
            f"`H_train={horizon['H_train_tokens']}` contiguous tokens; `H_eval={horizon['H_eval_tokens']}`; sampled positions across independent rollouts={horizon['sampled_positions_across_independent_rollouts']} (not a causal horizon).",
            f"Held-out late-prefix coverage: {gate['heldout_late_count_observed']}/{gate['heldout_late_count_required']} required; long-horizon evidence pass={str(gate['long_horizon_evidence_pass']).lower()}.",
            "Window KL/top-1/ridge-shift diagnostics are measured pre-update; post-teacher retention is established separately by final CSD-SD evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def generate_horizon_report(
    report_dir: str | Path,
    *,
    split_tokens: int = 2048,
    long_prefix_threshold: int = 4096,
    late_window_start: int = 2048,
    min_heldout_late_queries: int = 10,
) -> dict[str, Any]:
    root = Path(report_dir)
    summary = _load_json(root / "summary.json")
    rows = _load_jsonl(root / "merged_per_query.jsonl")
    report = build_horizon_report(
        rows,
        summary,
        split_tokens=split_tokens,
        long_prefix_threshold=long_prefix_threshold,
        late_window_start=late_window_start,
        min_heldout_late_queries=min_heldout_late_queries,
    )
    _atomic_write(
        root / "horizon_results.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _atomic_write(root / "horizon_results.csv", _csv_text(report["performance"]))
    _atomic_write(root / "horizon_results.md", _markdown_text(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--short-long-split-tokens", type=int, default=2048)
    parser.add_argument("--long-prefix-threshold", type=int, default=4096)
    parser.add_argument("--late-window-start", type=int, default=2048)
    parser.add_argument("--min-heldout-late-queries", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = generate_horizon_report(
            args.report_dir,
            split_tokens=args.short_long_split_tokens,
            long_prefix_threshold=args.long_prefix_threshold,
            late_window_start=args.late_window_start,
            min_heldout_late_queries=args.min_heldout_late_queries,
        )
    except HorizonReportError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
