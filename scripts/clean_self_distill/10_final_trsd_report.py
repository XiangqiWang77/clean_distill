#!/usr/bin/env python3
"""Build the final matched TRSD/privileged style and held-out report.

The reporter is deliberately model-free.  It consumes five currently matched
held-out JSONL files, an optional historical TRSD-16/Base pair for the appendix,
and the two 64-episode training journals. It writes paper-ready CSV, Markdown,
PNG, PDF, and JSON artifacts. The historical appendix is excluded from current
inference. Missing observations are reported as N/A; no metric is imputed.

The primary style statistic is the token-normalized absolute difference
between the distillation target's and the pre-update student's realized-token
log probabilities.  The style/task partition is the versioned heuristic stored
in each journal row.  It is a distributional diagnostic, not a claim that every
token assigned to a partition is semantically pure.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_SOURCES = {"amc23": 83, "aime24": 30, "aime25": 30}
SOURCE_ORDER = ("combined", "amc23", "aime24", "aime25")
SOURCE_LABELS = {
    "combined": "Combined",
    "amc23": "AMC23",
    "aime24": "AIME24",
    "aime25": "AIME25",
}
METHOD_ORDER = (
    "base",
    "privileged_16",
    "trsd_16",
    "privileged_64",
    "trsd_64",
)
MATCHED_INFERENCE_METHODS = (
    "base",
    "privileged_16",
    "trsd_16",
    "privileged_64",
    "trsd_64",
)
HISTORICAL_TRSD16_LABEL = "TRSD 16† (historical)"
METHOD_LABELS = {
    "base": "Base",
    "privileged_16": "Privilege-SD 16",
    "trsd_16": "TRSD 16",
    "privileged_64": "Privilege-SD 64",
    "trsd_64": "TRSD 64",
}
COLORS = {
    "base": "#64748B",
    "privileged_16": "#F59E0B",
    "trsd_16": "#2A9D8F",
    "privileged_64": "#C2410C",
    "trsd_64": "#0F766E",
    "privileged": "#C2410C",
    "trsd": "#0F766E",
}
HELDOUT_BOOTSTRAP_REPLICATES = 10_000
HELDOUT_BOOTSTRAP_SEED = 20260808


class ReportError(RuntimeError):
    """Raised when an input cannot support a matched scientific comparison."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReportError(f"Input does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReportError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ReportError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    if not rows:
        raise ReportError(f"Input is empty: {path}")
    return rows


def finite_float(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ReportError(f"{context} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ReportError(f"{context} is not finite")
    return number


def integer(value: Any, context: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ReportError(f"{context} is not an integer: {value!r}") from error
    return result


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    if not rows:
        raise ReportError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "N/A" if row.get(key) is None else row.get(key)
                    for key in fields
                }
            )
    temporary.replace(path)


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ReportError("Cannot take a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def ci95(values: Sequence[float]) -> tuple[float, float]:
    return quantile(values, 0.025), quantile(values, 0.975)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def load_journal(path: Path, *, expected_episodes: int, name: str) -> list[dict[str, Any]]:
    raw_rows = read_jsonl(path)
    # Drop the long response text and token-id vectors immediately.  The final
    # reporter is designed for a modest CPU node and never needs either field.
    rows = [
        {
            key: row.get(key)
            for key in (
                "episode",
                "stream_index",
                "query_id",
                "problem_sha256",
                "response_tokens",
                "optimizer_step",
                "episode_seconds",
                "style_task_error",
                "trust_region_alpha",
                "trust_region_achieved_kl",
                "mean_teacher_student_kl",
                "resource_usage",
            )
        }
        for row in raw_rows
    ]
    del raw_rows
    if len(rows) != expected_episodes:
        raise ReportError(
            f"{name} has {len(rows)} episodes; expected exactly {expected_episodes}"
        )
    episodes = [integer(row.get("episode"), f"{name}.episode") for row in rows]
    if sorted(episodes) != list(range(1, expected_episodes + 1)):
        raise ReportError(f"{name} episode indices are not exactly 1..{expected_episodes}")
    rows = sorted(rows, key=lambda row: int(row["episode"]))
    query_ids = [str(row.get("query_id", "")).strip() for row in rows]
    if any(not query_id for query_id in query_ids) or len(set(query_ids)) != len(query_ids):
        raise ReportError(f"{name} has missing or duplicate query IDs")

    partition_version: str | None = None
    for row in rows:
        episode = int(row["episode"])
        style = row.get("style_task_error")
        if not isinstance(style, dict):
            raise ReportError(f"{name} episode {episode} lacks style_task_error")
        version = str(style.get("partition_version", ""))
        if not version:
            raise ReportError(f"{name} episode {episode} lacks partition_version")
        if partition_version is None:
            partition_version = version
        elif version != partition_version:
            raise ReportError(f"{name} mixes style/task partition versions")
        for partition in ("style", "task", "other"):
            total = finite_float(
                style.get(f"{partition}_abs_error_sum"),
                f"{name}.{episode}.{partition}_abs_error_sum",
            )
            count = integer(
                style.get(f"{partition}_token_count"),
                f"{name}.{episode}.{partition}_token_count",
            )
            if total < 0.0 or count < 0:
                raise ReportError(f"{name} episode {episode} contains a negative metric")
        if not isinstance(row.get("optimizer_step"), bool):
            raise ReportError(f"{name} episode {episode} lacks optimizer_step bool")
    return rows


def match_journals(
    privileged: Sequence[Mapping[str, Any]], trsd: Sequence[Mapping[str, Any]]
) -> None:
    if len(privileged) != len(trsd):
        raise ReportError("Journal episode counts differ")
    for left, right in zip(privileged, trsd):
        episode = int(left["episode"])
        for field in ("episode", "stream_index", "query_id", "problem_sha256"):
            if str(left.get(field, "")) != str(right.get(field, "")):
                raise ReportError(
                    f"Journals are not paired at episode {episode}: {field} differs"
                )


def episode_style_row(row: Mapping[str, Any], method: str) -> dict[str, Any]:
    style = row["style_task_error"]
    style_sum = finite_float(style["style_abs_error_sum"], "style sum")
    style_count = integer(style["style_token_count"], "style count")
    task_sum = finite_float(style["task_abs_error_sum"], "task sum")
    task_count = integer(style["task_token_count"], "task count")
    other_sum = finite_float(style["other_abs_error_sum"], "other sum")
    other_count = integer(style["other_token_count"], "other count")
    style_mean = safe_ratio(style_sum, style_count)
    task_mean = safe_ratio(task_sum, task_count)
    other_mean = safe_ratio(other_sum, other_count)
    alpha_value = row.get("trust_region_alpha")
    kl_value = row.get("trust_region_achieved_kl")
    resource = row.get("resource_usage")
    resource = resource if isinstance(resource, Mapping) else {}

    def resource_gib(field: str) -> float | None:
        value = resource.get(field)
        if value is None:
            return None
        return finite_float(value, field) / (1024.0**3)

    return {
        "method": method,
        "episode": int(row["episode"]),
        "query_id": str(row["query_id"]),
        "problem_sha256": str(row.get("problem_sha256", "")),
        "response_tokens": int(row["response_tokens"]),
        "optimizer_step": bool(row["optimizer_step"]),
        "episode_seconds": finite_float(row.get("episode_seconds", 0.0), "episode_seconds"),
        "style_error_sum": style_sum,
        "style_token_count": style_count,
        "style_error_per_token": style_mean,
        "task_error_sum": task_sum,
        "task_token_count": task_count,
        "task_error_per_token": task_mean,
        "other_error_sum": other_sum,
        "other_token_count": other_count,
        "other_error_per_token": other_mean,
        "psr": (
            None
            if style_mean is None or task_mean is None or task_mean <= 0.0
            else style_mean / task_mean
        ),
        "alpha": None if alpha_value is None else finite_float(alpha_value, "alpha"),
        "achieved_kl": None if kl_value is None else finite_float(kl_value, "achieved KL"),
        "mean_teacher_student_kl": finite_float(
            row.get("mean_teacher_student_kl"), "mean_teacher_student_kl"
        ),
        "gpu_peak_allocated_gib": resource_gib("cuda_peak_memory_allocated_bytes"),
        "gpu_peak_delta_gib": resource_gib("cuda_peak_memory_delta_bytes"),
        "gpu_peak_reserved_gib": resource_gib("cuda_peak_memory_reserved_bytes"),
        "process_peak_rss_gib": resource_gib("process_peak_rss_bytes"),
        "partition_version": style["partition_version"],
        "error_definition": style.get("error_definition", ""),
    }


def aggregate_style(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    style_sum = sum(float(row["style_error_sum"]) for row in rows)
    style_count = sum(int(row["style_token_count"]) for row in rows)
    task_sum = sum(float(row["task_error_sum"]) for row in rows)
    task_count = sum(int(row["task_token_count"]) for row in rows)
    other_sum = sum(float(row["other_error_sum"]) for row in rows)
    other_count = sum(int(row["other_token_count"]) for row in rows)
    style_mean = safe_ratio(style_sum, style_count)
    task_mean = safe_ratio(task_sum, task_count)
    other_mean = safe_ratio(other_sum, other_count)
    alphas = [float(row["alpha"]) for row in rows if row["alpha"] is not None]
    achieved = [float(row["achieved_kl"]) for row in rows if row["achieved_kl"] is not None]

    def maximum_optional(field: str) -> float | None:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return max(values) if values else None

    return {
        "episodes": len(rows),
        "response_tokens": sum(int(row["response_tokens"]) for row in rows),
        "optimizer_steps": sum(bool(row["optimizer_step"]) for row in rows),
        "no_op_episodes": sum(not bool(row["optimizer_step"]) for row in rows),
        "training_hours": sum(float(row["episode_seconds"]) for row in rows) / 3600.0,
        "style_error_sum": style_sum,
        "style_token_count": style_count,
        "style_error_per_token": style_mean,
        "task_error_sum": task_sum,
        "task_token_count": task_count,
        "task_error_per_token": task_mean,
        "other_error_sum": other_sum,
        "other_token_count": other_count,
        "other_error_per_token": other_mean,
        "psr": (
            None
            if style_mean is None or task_mean is None or task_mean <= 0.0
            else style_mean / task_mean
        ),
        "mean_alpha": statistics.fmean(alphas) if alphas else None,
        "mean_achieved_kl": statistics.fmean(achieved) if achieved else None,
        "constraint_activation_rate": (
            sum(value < 1.0 - 1e-12 for value in alphas) / len(alphas)
            if alphas
            else None
        ),
        "mean_teacher_student_kl": statistics.fmean(
            float(row["mean_teacher_student_kl"]) for row in rows
        ),
        "max_gpu_peak_allocated_gib": maximum_optional("gpu_peak_allocated_gib"),
        "max_gpu_peak_delta_gib": maximum_optional("gpu_peak_delta_gib"),
        "max_gpu_peak_reserved_gib": maximum_optional("gpu_peak_reserved_gib"),
        "max_process_peak_rss_gib": maximum_optional("process_peak_rss_gib"),
    }


def paired_bootstrap(
    privileged: Sequence[Mapping[str, Any]],
    trsd: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    if replicates < 100:
        raise ReportError("At least 100 bootstrap replicates are required")
    generator = random.Random(seed)
    count = len(privileged)
    result: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        indices = [generator.randrange(count) for _ in range(count)]
        p = aggregate_style([privileged[index] for index in indices])
        t = aggregate_style([trsd[index] for index in indices])
        for metric in ("style_error_per_token", "task_error_per_token", "psr"):
            p_value = p[metric]
            t_value = t[metric]
            if p_value is None or t_value is None:
                continue
            p_number = float(p_value)
            t_number = float(t_value)
            result[f"privileged_{metric}"].append(p_number)
            result[f"trsd_{metric}"].append(t_number)
            result[f"delta_{metric}"].append(t_number - p_number)
            if p_number > 0.0:
                result[f"ratio_{metric}"].append(t_number / p_number)
    return result


def load_scored(path: Path, *, name: str, expected_total: int) -> list[dict[str, Any]]:
    raw_rows = read_jsonl(path)
    # Evaluation responses can contain 10,240 tokens each.  Retain only the
    # scalar fields used below so loading five methods does not duplicate those
    # large strings in memory.
    rows = [
        {
            key: row.get(key)
            for key in (
                "profile",
                "query_id",
                "problem_sha256",
                "source",
                "sample_index",
                "correct",
                "truncated",
                "generated_tokens",
                "behavioral_diagnostics",
                "resource_usage",
                "parsed_answer",
                "max_new_tokens",
                "checkpoint_episode",
                "evaluation_prompt_version",
            )
        }
        for row in raw_rows
        if row.get("profile", "acc1") == "acc1"
    ]
    del raw_rows
    if len(rows) != expected_total:
        raise ReportError(f"{name} has {len(rows)} Acc@1 rows; expected {expected_total}")
    query_ids = [str(row.get("query_id", "")).strip() for row in rows]
    if any(not value for value in query_ids) or len(set(query_ids)) != len(query_ids):
        raise ReportError(f"{name} has missing or duplicate query IDs")
    source_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        source_counts[str(row.get("source", ""))] += 1
        correct = finite_float(row.get("correct"), f"{name}.correct")
        if correct not in (0.0, 1.0):
            raise ReportError(f"{name} correctness must be binary")
        row["correct"] = int(correct)
        if not isinstance(row.get("truncated"), bool):
            raise ReportError(f"{name} lacks a boolean completion flag")
    if expected_total == sum(EXPECTED_SOURCES.values()) and dict(source_counts) != EXPECTED_SOURCES:
        raise ReportError(
            f"{name} source counts {dict(source_counts)} do not equal {EXPECTED_SOURCES}"
        )
    return rows


def validate_historical_trsd16(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed unless the T16 artifact has the declared historical signature."""
    episodes = {row.get("checkpoint_episode") for row in rows}
    if episodes != {16}:
        raise ReportError(
            f"Historical TRSD-16 must contain only checkpoint episode 16, found {episodes}"
        )
    prompt_versions = {
        ""
        if row.get("evaluation_prompt_version") is None
        else str(row.get("evaluation_prompt_version", "")).strip()
        for row in rows
    }
    if prompt_versions != {""}:
        raise ReportError(
            "Historical TRSD-16 unexpectedly declares an evaluation prompt version; "
            "do not label a current-protocol artifact historical"
        )


def match_scored(method_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    base = {str(row["query_id"]): row for row in method_rows["base"]}
    for method, rows in method_rows.items():
        current = {str(row["query_id"]): row for row in rows}
        if set(current) != set(base):
            raise ReportError(f"{method} held-out query coverage differs from Base")
        for query_id, base_row in base.items():
            row = current[query_id]
            for field in ("problem_sha256", "source", "sample_index"):
                if str(row.get(field, "")) != str(base_row.get(field, "")):
                    raise ReportError(f"{method}/{query_id}: {field} differs from Base")


def aggregate_scored(method: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    groups: dict[str, Sequence[Mapping[str, Any]]] = {"combined": rows, **by_source}
    output: list[dict[str, Any]] = []
    for source in SOURCE_ORDER:
        values = list(groups.get(source, []))
        if not values:
            raise ReportError(f"{method} lacks source {source}")
        strict_correct_count = sum(strict_correct(row) for row in values)
        truncated_count = sum(bool(row["truncated"]) for row in values)
        response_tokens = [int(row.get("generated_tokens", 0)) for row in values]
        diagnostics = [
            row.get("behavioral_diagnostics")
            if isinstance(row.get("behavioral_diagnostics"), dict)
            else {}
            for row in values
        ]
        resource = [
            row.get("resource_usage") if isinstance(row.get("resource_usage"), dict) else {}
            for row in values
        ]
        seconds = [
            finite_float(item["generation_seconds"], "generation_seconds")
            for item in resource
            if item.get("generation_seconds") is not None
        ]
        peaks = [
            finite_float(item["cuda_peak_memory_allocated_bytes"], "peak memory")
            for item in resource
            if item.get("cuda_peak_memory_allocated_bytes") is not None
        ]
        hedge_total = sum(int(item.get("hedging_token_count", 0)) for item in diagnostics)
        ref_total = sum(bool(item.get("fabricated_reference_hallucination", False)) for item in diagnostics)
        entropy = [
            finite_float(item["mean_entropy"], "mean_entropy")
            for item in diagnostics
            if item.get("mean_entropy") is not None
        ]
        token_total = sum(response_tokens)
        output.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "comparison_status": "current_protocol_matched",
                "dataset": source,
                "strict_correct": strict_correct_count,
                "n": len(values),
                "strict_acc1": strict_correct_count / len(values),
                "strict_percent": 100.0 * strict_correct_count / len(values),
                "truncated_count": truncated_count,
                "truncation_rate": truncated_count / len(values),
                "mean_generated_tokens": statistics.fmean(response_tokens),
                "hedging_tokens_per_1k": 1000.0 * hedge_total / token_total if token_total else None,
                "fabricated_reference_rate": ref_total / len(values),
                "mean_entropy": statistics.fmean(entropy) if entropy else None,
                "mean_generation_seconds": statistics.fmean(seconds) if seconds else None,
                "peak_gpu_allocated_gib": max(peaks) / (1024.0**3) if peaks else None,
            }
        )
    return output


def paired_transitions(
    base_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    base = {str(row["query_id"]): row for row in base_rows}
    current = {str(row["query_id"]): row for row in rows}
    pairs = [(base[key], current[key]) for key in sorted(base)]
    wrong_to_correct = sum(not strict_correct(a) and strict_correct(b) for a, b in pairs)
    correct_to_wrong = sum(strict_correct(a) and not strict_correct(b) for a, b in pairs)
    return {
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "n": len(pairs),
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "discordant_pairs": wrong_to_correct + correct_to_wrong,
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided_p(
            wrong_to_correct, correct_to_wrong
        ),
        "correct_to_correct": sum(strict_correct(a) and strict_correct(b) for a, b in pairs),
        "wrong_to_wrong": sum(not strict_correct(a) and not strict_correct(b) for a, b in pairs),
        "parsed_answer_changes": sum(
            str(a.get("parsed_answer", "")) != str(b.get("parsed_answer", ""))
            for a, b in pairs
        ),
    }


def strict_correct(row: Mapping[str, Any]) -> int:
    """Return the primary all-query outcome, counting unfinished responses wrong."""
    correct = finite_float(row.get("correct"), "strict correctness")
    if correct not in (0.0, 1.0):
        raise ReportError("Strict correctness must be binary")
    return int(correct == 1.0 and not bool(row["truncated"]))


def exact_mcnemar_two_sided_p(wrong_to_correct: int, correct_to_wrong: int) -> float:
    """Exact two-sided McNemar p-value via Binomial(n_discordant, 0.5).

    This is the conventional doubled smaller-tail exact test.  It is exact and
    deterministic, uses no asymptotic chi-square approximation, and returns one
    when there are no discordant pairs.
    """
    wrong_to_correct = integer(wrong_to_correct, "wrong_to_correct")
    correct_to_wrong = integer(correct_to_wrong, "correct_to_wrong")
    if wrong_to_correct < 0 or correct_to_wrong < 0:
        raise ReportError("McNemar discordant counts must be non-negative")
    discordant = wrong_to_correct + correct_to_wrong
    if discordant == 0:
        return 1.0
    smaller = min(wrong_to_correct, correct_to_wrong)
    lower_tail_numerator = sum(
        math.comb(discordant, successes) for successes in range(smaller + 1)
    )
    return min(1.0, 2.0 * lower_tail_numerator / float(2**discordant))


def heldout_paired_bootstrap(
    method_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    dataset: str,
    replicates: int,
    seed: int,
) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    """Bootstrap strict accuracies and paired deltas on a shared query resample."""
    if dataset not in SOURCE_ORDER:
        raise ReportError(f"Unknown held-out dataset: {dataset}")
    if replicates < 100:
        raise ReportError("At least 100 held-out bootstrap replicates are required")

    by_method = {
        method: {
            str(row["query_id"]): row
            for row in rows
            if dataset == "combined" or str(row["source"]) == dataset
        }
        for method, rows in method_rows.items()
    }
    query_ids = sorted(by_method["base"])
    if not query_ids:
        raise ReportError(f"No held-out rows for dataset {dataset}")
    for method, rows in by_method.items():
        if set(rows) != set(query_ids):
            raise ReportError(f"{method} query coverage differs from Base in {dataset}")

    outcomes = {
        method: [strict_correct(rows[query_id]) for query_id in query_ids]
        for method, rows in by_method.items()
    }
    bootstrap: dict[str, list[float]] = defaultdict(list)
    generator = random.Random(seed)
    n_queries = len(query_ids)
    for _ in range(replicates):
        indices = [generator.randrange(n_queries) for _ in range(n_queries)]
        accuracies = {
            method: sum(values[index] for index in indices) / n_queries
            for method, values in outcomes.items()
        }
        for method in MATCHED_INFERENCE_METHODS:
            bootstrap[f"{method}_accuracy"].append(accuracies[method])
            bootstrap[f"{method}_delta_vs_base"].append(
                accuracies[method] - accuracies["base"]
            )
    return dict(bootstrap), outcomes


def build_heldout_robustness(
    method_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    replicates: int = HELDOUT_BOOTSTRAP_REPLICATES,
    seed: int = HELDOUT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Return per-dataset strict accuracy inference under paired query sampling."""
    output: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(SOURCE_ORDER):
        bootstrap, outcomes = heldout_paired_bootstrap(
            method_rows,
            dataset=dataset,
            replicates=replicates,
            seed=seed + dataset_index,
        )
        n_queries = len(outcomes["base"])
        base_accuracy = sum(outcomes["base"]) / n_queries
        for method in MATCHED_INFERENCE_METHODS:
            accuracy = sum(outcomes[method]) / n_queries
            accuracy_low, accuracy_high = ci95(bootstrap[f"{method}_accuracy"])
            delta = accuracy - base_accuracy
            delta_low, delta_high = ci95(bootstrap[f"{method}_delta_vs_base"])
            if method == "base":
                wrong_to_correct = None
                correct_to_wrong = None
                discordant = None
                exact_p = None
            else:
                pairs = zip(outcomes["base"], outcomes[method])
                paired_values = list(pairs)
                wrong_to_correct = sum(
                    base_value == 0 and method_value == 1
                    for base_value, method_value in paired_values
                )
                correct_to_wrong = sum(
                    base_value == 1 and method_value == 0
                    for base_value, method_value in paired_values
                )
                discordant = wrong_to_correct + correct_to_wrong
                exact_p = exact_mcnemar_two_sided_p(
                    wrong_to_correct, correct_to_wrong
                )
            output.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "dataset": dataset,
                    "n": n_queries,
                    "strict_correct": sum(outcomes[method]),
                    "strict_accuracy": accuracy,
                    "strict_accuracy_percent": 100.0 * accuracy,
                    "strict_accuracy_bootstrap_ci_low": accuracy_low,
                    "strict_accuracy_bootstrap_ci_high": accuracy_high,
                    "base_strict_accuracy": base_accuracy,
                    "delta_vs_base": delta,
                    "delta_vs_base_percentage_points": 100.0 * delta,
                    "delta_bootstrap_ci_low": delta_low,
                    "delta_bootstrap_ci_high": delta_high,
                    "wrong_to_correct": wrong_to_correct,
                    "correct_to_wrong": correct_to_wrong,
                    "discordant_pairs": discordant,
                    "mcnemar_exact_two_sided_p": exact_p,
                    "bootstrap_unit": "paired_query",
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed + dataset_index,
                    "estimand": "strict_acc1_truncated_is_wrong",
                }
            )
    return output


def build_historical_trsd16_reference(
    trsd_rows: Sequence[Mapping[str, Any]],
    historical_base_rows: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Build point estimates within the historical T16 evaluation protocol.

    No confidence interval or hypothesis test is produced here.  In particular,
    this function never compares the historical TRSD-16 row with the current
    explicit-budget Base evaluation.
    """
    trsd = {
        str(row["dataset"]): row
        for row in aggregate_scored("trsd_16", trsd_rows)
    }
    base = (
        {
            str(row["dataset"]): row
            for row in aggregate_scored("historical_base", historical_base_rows)
        }
        if historical_base_rows is not None
        else {}
    )
    output: list[dict[str, Any]] = []
    for dataset in SOURCE_ORDER:
        trsd_row = trsd[dataset]
        base_row = base.get(dataset)
        base_accuracy = (
            None
            if base_row is None
            else float(base_row["strict_acc1"])
        )
        trsd_accuracy = float(trsd_row["strict_acc1"])
        output.append(
            {
                "method": "trsd_16",
                "method_label": HISTORICAL_TRSD16_LABEL,
                "dataset": dataset,
                "n": int(trsd_row["n"]),
                "historical_base_strict_correct": (
                    None
                    if base_row is None
                    else int(base_row["strict_correct"])
                ),
                "historical_base_strict_accuracy": base_accuracy,
                "trsd16_strict_correct": int(trsd_row["strict_correct"]),
                "trsd16_strict_accuracy": trsd_accuracy,
                "strict_delta_vs_historical_base": (
                    None if base_accuracy is None else trsd_accuracy - base_accuracy
                ),
                "comparison_protocol": "historical_10240_token_no_explicit_eval_prompt_version",
                "checkpoint_protocol": "historical_pre_exact_reverse_kl",
                "inference_status": "point_estimate_only_not_compared_to_current_base",
            }
        )
    return output


def mechanism_summary(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ReportError(f"Mechanism CSV is empty: {path}")
    required_wrappers = {"neutral", "terse", "verbose"}
    by_projection: dict[str, set[tuple[str, str]]] = {}
    for projection in ("raw_privileged_surrogate", "trsd_projected"):
        keys = {
            (str(row.get("query_id", "")), str(row.get("wrapper", "")))
            for row in rows
            if row.get("projection") == projection
        }
        if any(not query_id or wrapper not in required_wrappers for query_id, wrapper in keys):
            raise ReportError(f"Mechanism CSV has invalid {projection} query/wrapper keys")
        if len(keys) != sum(row.get("projection") == projection for row in rows):
            raise ReportError(f"Mechanism CSV has duplicate {projection} query/wrapper rows")
        by_projection[projection] = keys
    if by_projection["raw_privileged_surrogate"] != by_projection["trsd_projected"]:
        raise ReportError("Mechanism raw/projected rows are not exactly same-prefix paired")
    queries = {query_id for query_id, _ in by_projection["raw_privileged_surrogate"]}
    if any(
        {wrapper for query_id, wrapper in by_projection["raw_privileged_surrogate"] if query_id == query}
        != required_wrappers
        for query in queries
    ):
        raise ReportError("Every mechanism query must contain neutral/terse/verbose wrappers")
    if any(str(row.get("answer_free", "")).strip().casefold() not in {"1", "true"} for row in rows):
        raise ReportError("Mechanism pilot must be answer-free")
    output: list[dict[str, Any]] = []
    for projection in ("raw_privileged_surrogate", "trsd_projected"):
        subset = [row for row in rows if row.get("projection") == projection]
        if not subset:
            raise ReportError(f"Mechanism CSV lacks projection {projection}")
        output.append(
            {
                "projection": projection,
                "n_query_wrappers": len(subset),
                "n_queries": len({row["query_id"] for row in subset}),
                "style_abs_logprob_shift": statistics.fmean(
                    finite_float(row["style_abs_logprob_shift"], "mechanism style")
                    for row in subset
                ),
                "task_logprob_gain": statistics.fmean(
                    finite_float(row["task_logprob_gain"], "mechanism task")
                    for row in subset
                ),
                "mean_alpha": statistics.fmean(
                    finite_float(row["alpha"], "mechanism alpha") for row in subset
                ),
                "mean_achieved_kl": statistics.fmean(
                    finite_float(row["achieved_mean_kl"], "mechanism KL") for row in subset
                ),
            }
        )
    return output


def epsilon_sensitivity(path: Path | None) -> list[dict[str, Any]]:
    """Load the label-free one-episode development epsilon sweep."""
    if path is None:
        return []
    if not path.is_file():
        raise ReportError(f"Epsilon sensitivity CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ReportError(f"Epsilon sensitivity CSV is empty: {path}")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows, 1):
        selected_text = str(raw.get("is_selected", "")).strip().casefold()
        if selected_text not in {"true", "false"}:
            raise ReportError(f"Epsilon row {index} has invalid is_selected")
        row = {
            "epsilon": finite_float(raw.get("epsilon"), f"epsilon row {index}"),
            "mean_alpha": finite_float(raw.get("mean_alpha"), f"alpha row {index}"),
            "achieved_mean_kl": finite_float(
                raw.get("achieved_mean_kl"), f"achieved KL row {index}"
            ),
            "active_wrappers": integer(
                raw.get("active_wrappers"), f"active wrappers row {index}"
            ),
            "task_token_gain": finite_float(
                raw.get("task_token_gain"), f"task gain row {index}"
            ),
            "task_gain_vs_raw": finite_float(
                raw.get("task_gain_vs_raw"), f"task gain ratio row {index}"
            ),
            "style_abs_shift": finite_float(
                raw.get("style_abs_shift"), f"style shift row {index}"
            ),
            "style_retention_vs_raw": finite_float(
                raw.get("style_retention_vs_raw"), f"style retention row {index}"
            ),
            "prompt_variance_retention": finite_float(
                raw.get("prompt_variance_retention"),
                f"prompt variance retention row {index}",
            ),
            "is_selected": selected_text == "true",
        }
        if not 0.0 < row["epsilon"] or not 0.0 <= row["mean_alpha"] <= 1.0:
            raise ReportError(f"Epsilon row {index} is outside its valid range")
        if row["achieved_mean_kl"] < 0.0 or row["active_wrappers"] < 0:
            raise ReportError(f"Epsilon row {index} contains a negative diagnostic")
        rows.append(row)
    if len({row["epsilon"] for row in rows}) != len(rows):
        raise ReportError("Epsilon sensitivity CSV contains duplicate budgets")
    if sum(bool(row["is_selected"]) for row in rows) != 1:
        raise ReportError("Epsilon sensitivity CSV must mark exactly one selected budget")
    return sorted(rows, key=lambda row: float(row["epsilon"]))


def configure_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    return plt


def save_figure(figure: Any, root: Path, stem: str) -> None:
    figure.savefig(root / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(root / f"{stem}.pdf", bbox_inches="tight")


def plot_style(
    root: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Sequence[float]],
    mechanism: Sequence[Mapping[str, Any]],
) -> None:
    plt = configure_plotting()
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), constrained_layout=True)
    specs = (
        ("style_error_per_token", "Normalized style drift ↓", "|Δ log p| / style token"),
        (
            "task_error_per_token",
            "Task-token target movement",
            "|Δ log p| / task token (not accuracy)",
        ),
        ("psr", "Style / task ratio ↓", "PSR"),
    )
    for axis, (metric, title, ylabel) in zip(axes, specs):
        values = [float(summaries[name][metric]) for name in ("privileged", "trsd")]
        intervals = [
            ci95(bootstrap[f"{name}_{metric}"])
            for name in ("privileged", "trsd")
        ]
        errors = [
            [value - interval[0] for value, interval in zip(values, intervals)],
            [interval[1] - value for value, interval in zip(values, intervals)],
        ]
        bars = axis.bar(
            [0, 1],
            values,
            color=[COLORS["privileged"], COLORS["trsd"]],
            width=0.62,
            yerr=errors,
            capsize=4,
        )
        axis.set_xticks([0, 1], ["Privilege-SD\nraw target", "TRSD\nprojected target"])
        axis.set_title(title, fontweight="semibold")
        axis.set_ylabel(ylabel)
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=5, fontsize=8)
    if mechanism:
        raw, projected = mechanism
        reduction = 1.0 - float(projected["style_abs_logprob_shift"]) / float(raw["style_abs_logprob_shift"])
        figure.text(
            0.5,
            -0.02,
            f"Separate same-prefix pilot: projected style shift was {100*reduction:.1f}% lower "
            f"({raw['n_queries']} queries × neutral/terse/verbose; descriptive).",
            ha="center",
            fontsize=8.5,
            color="#475569",
        )
    figure.suptitle(
        "Trajectory-level trust-region projection controls the distillation target",
        fontsize=12,
        fontweight="semibold",
    )
    save_figure(figure, root, "matched64_style_task_psr")
    plt.close(figure)


def plot_epsilon(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    plt = configure_plotting()
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), constrained_layout=True)
    epsilon = [float(row["epsilon"]) for row in rows]
    selected = next(index for index, row in enumerate(rows) if row["is_selected"])

    axes[0].plot(epsilon, [float(row["mean_alpha"]) for row in rows], marker="o")
    axes[0].scatter(
        [epsilon[selected]], [float(rows[selected]["mean_alpha"])],
        s=75, color=COLORS["trsd"], zorder=3, label="Selected on dev",
    )
    axes[0].set_title("Auto-adaptive projection", fontweight="semibold")
    axes[0].set_ylabel("Mean projection α")
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].legend(fontsize=8)

    achieved = [float(row["achieved_mean_kl"]) for row in rows]
    axes[1].plot(epsilon, achieved, marker="o", color=COLORS["trsd"], label="Achieved KL")
    axes[1].plot(epsilon, epsilon, linestyle="--", color="#94A3B8", label="Budget ε")
    axes[1].set_title("Trajectory KL budget", fontweight="semibold")
    axes[1].set_ylabel("Mean KL")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        epsilon,
        [float(row["style_retention_vs_raw"]) for row in rows],
        marker="o",
        label="Style-shift retention ↓",
        color=COLORS["privileged"],
    )
    axes[2].plot(
        epsilon,
        [float(row["task_gain_vs_raw"]) for row in rows],
        marker="s",
        label="Task-token gain / raw",
        color=COLORS["trsd"],
    )
    axes[2].plot(
        epsilon,
        [float(row["prompt_variance_retention"]) for row in rows],
        marker="^",
        label="Prompt-variance retention ↓",
        color="#7C3AED",
    )
    axes[2].axhline(1.0, linestyle="--", color="#94A3B8", linewidth=1)
    axes[2].set_title("Label-free signal trade-off", fontweight="semibold")
    axes[2].set_ylabel("Ratio to raw surrogate")
    axes[2].legend(fontsize=7.5)

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("KL budget ε")
        axis.set_xticks(epsilon, [f"{value:g}" for value in epsilon], rotation=25)
    figure.suptitle(
        "One-episode development sensitivity (no held-out labels)",
        fontsize=12,
        fontweight="semibold",
    )
    save_figure(figure, root, "epsilon_dev_sensitivity")
    plt.close(figure)


def plot_heldout(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    plt = configure_plotting()
    lookup = {(str(row["method"]), str(row["dataset"])): row for row in rows}
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), constrained_layout=True)
    width = 0.16
    x = list(range(len(SOURCE_ORDER)))
    for method_index, method in enumerate(METHOD_ORDER):
        offset = (method_index - (len(METHOD_ORDER) - 1) / 2.0) * width
        values = [
            float(lookup[(method, source)]["strict_percent"])
            for source in SOURCE_ORDER
        ]
        axes[0, 0].bar(
            [position + offset for position in x],
            values,
            width * 0.92,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0, 0].set_xticks(x, [SOURCE_LABELS[source] for source in SOURCE_ORDER])
    axes[0, 0].set_ylabel("Accuracy (%) ↑")
    axes[0, 0].set_title("Strict Acc@1 (unfinished = wrong)", fontweight="semibold")
    axes[0, 0].legend(fontsize=8, ncol=2)

    combined = {method: lookup[(method, "combined")] for method in METHOD_ORDER}
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    colors = [COLORS[method] for method in METHOD_ORDER]
    base_strict = float(combined["base"]["strict_percent"])
    delta_values = [
        float(combined[method]["strict_percent"]) - base_strict
        for method in METHOD_ORDER
    ]
    delta_bars = axes[0, 1].bar(labels, delta_values, color=colors, width=0.68)
    axes[0, 1].axhline(0.0, color="#64748B", linewidth=1)
    axes[0, 1].set_title("Combined strict Δ vs Base (pp)", fontweight="semibold")
    axes[0, 1].tick_params(axis="x", rotation=14)
    axes[0, 1].bar_label(
        delta_bars,
        labels=[f"{value:+.2f}" for value in delta_values],
        padding=3,
        fontsize=8,
    )
    panels = (
        (axes[1, 0], "hedging_tokens_per_1k", 1.0, "Hedging tokens / 1k (diagnostic)"),
        (axes[1, 1], "fabricated_reference_rate", 100.0, "Fabricated-reference rate (%) ↓"),
    )
    for axis, field, scale, title in panels:
        values = [
            math.nan
            if combined[method][field] is None
            else float(combined[method][field]) * scale
            for method in METHOD_ORDER
        ]
        bars = axis.bar(labels, values, color=colors, width=0.68)
        axis.set_title(title, fontweight="semibold")
        axis.tick_params(axis="x", rotation=14)
        axis.bar_label(
            bars,
            labels=["N/A" if math.isnan(value) else f"{value:.2f}" for value in values],
            padding=3,
            fontsize=8,
        )
    figure.suptitle(
        "Final-checkpoint accuracy and response-shape diagnostics",
        fontsize=12,
        fontweight="semibold",
    )
    save_figure(figure, root, "heldout_accuracy_behavior")
    plt.close(figure)


def fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def accuracy_cell(row: Mapping[str, Any]) -> str:
    """Format the only paper-facing accuracy: strict Acc@1."""
    return (
        f"{float(row['strict_percent']):.2f}% "
        f"({row['strict_correct']}/{row['n']})"
    )


def heldout_robustness_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the primary all-query inference table and its statistical contract."""
    lines = [
        "## Paired held-out robustness (primary estimand)",
        "",
        "Every unfinished response is counted wrong. Confidence intervals are "
        "paired-query percentile bootstrap intervals; McNemar p-values are exact "
        "two-sided Binomial(discordant, 0.5) tests.",
        "",
        "| Dataset | Method | Strict Acc@1 [95% CI] | Δ vs Base [95% CI] | W→C / C→W | Exact p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        method = str(row["method"])
        accuracy = (
            f"{100*float(row['strict_accuracy']):.2f}% "
            f"[{100*float(row['strict_accuracy_bootstrap_ci_low']):.2f}, "
            f"{100*float(row['strict_accuracy_bootstrap_ci_high']):.2f}]"
        )
        if method == "base":
            delta = "—"
            transitions = "—"
            exact_p = "—"
        else:
            delta = (
                f"{100*float(row['delta_vs_base']):+.2f} pp "
                f"[{100*float(row['delta_bootstrap_ci_low']):+.2f}, "
                f"{100*float(row['delta_bootstrap_ci_high']):+.2f}]"
            )
            transitions = f"{row['wrong_to_correct']} / {row['correct_to_wrong']}"
            exact_p = f"{float(row['mcnemar_exact_two_sided_p']):.4g}"
        lines.append(
            f"| {SOURCE_LABELS[str(row['dataset'])]} | {METHOD_LABELS[method]} | "
            f"{accuracy} | {delta} | {transitions} | {exact_p} |"
        )
    lines.extend(
        [
            "",
            f"Bootstrap protocol: `{HELDOUT_BOOTSTRAP_REPLICATES:,}` resamples, "
            f"fixed seed `{HELDOUT_BOOTSTRAP_SEED}` for Combined and deterministic "
            "source-specific offsets.",
            "",
            "The Combined comparison is the primary aggregate. Source-specific tests "
            "are reported transparently as exploratory and their p-values are not "
            "multiplicity-adjusted.",
            f"{METHOD_LABELS['trsd_16']} in this table is the newly matched reverse-KL "
            "checkpoint under the current explicit-budget prompt. The separately marked "
            f"{HISTORICAL_TRSD16_LABEL} appendix row is excluded from current inference.",
        ]
    )
    return "\n".join(lines)


def historical_reference_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the non-inferential historical T16 comparison."""
    lines = [
        "## Historical TRSD-16 reference (point estimates only)",
        "",
        "| Dataset | Historical Base strict Acc@1 | TRSD 16† strict Acc@1 | Δ within historical protocol |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        base_accuracy = row["historical_base_strict_accuracy"]
        delta = row["strict_delta_vs_historical_base"]
        base_text = (
            "N/A"
            if base_accuracy is None
            else f"{100*float(base_accuracy):.2f}% "
            f"({row['historical_base_strict_correct']}/{row['n']})"
        )
        delta_text = "N/A" if delta is None else f"{100*float(delta):+.2f} pp"
        lines.append(
            f"| {SOURCE_LABELS[str(row['dataset'])]} | {base_text} | "
            f"{100*float(row['trsd16_strict_accuracy']):.2f}% "
            f"({row['trsd16_strict_correct']}/{row['n']}) | {delta_text} |"
        )
    lines.extend(
        [
            "",
            "† Historical TRSD-16 used a 10,240-token evaluation artifact without an "
            "explicit evaluation-prompt-version field, and its checkpoint predates the "
            "exact reverse-KL implementation. It is not directly compared with current Base/P16/P64/T64.",
        ]
    )
    return "\n".join(lines)


def report_markdown(
    heldout: Sequence[Mapping[str, Any]],
    heldout_robustness: Sequence[Mapping[str, Any]],
    historical_reference: Sequence[Mapping[str, Any]],
    style: Mapping[str, Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    mechanism: Sequence[Mapping[str, Any]],
    epsilon_rows: Sequence[Mapping[str, Any]],
    partition_version: str,
    error_definition: str,
) -> str:
    lookup = {(row["method"], row["dataset"]): row for row in heldout}
    lines = [
        "# Final TRSD matched report",
        "",
        "## Held-out 10k-budget accuracy and behavior",
        "",
        "The only reported accuracy is strict Acc@1: every unfinished or truncated "
        "response is wrong, with all 143 queries retained. Exact numerators and "
        "denominators are included.",
        "",
        "| Method | Combined strict Acc@1 | AMC23 | AIME24 | AIME25 | Hedge/1k | Fabricated ref. | Sec/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        combined = lookup[(method, "combined")]
        lines.append(
            f"| {METHOD_LABELS[method]} | {accuracy_cell(combined)} | "
            + " | ".join(
                accuracy_cell(lookup[(method, source)])
                for source in ('amc23', 'aime24', 'aime25')
            )
            + " | "
            f"{fmt(combined['hedging_tokens_per_1k'], 2)} | "
            f"{100*combined['fabricated_reference_rate']:.2f}% | "
            f"{fmt(combined['mean_generation_seconds'], 1)} |"
        )

    lines.extend(
        [
            "",
            heldout_robustness_markdown(heldout_robustness),
            "",
            historical_reference_markdown(historical_reference),
        ]
    )
    lines.extend(
        [
            "",
            "## Matched 64-episode target diagnostics",
            "",
            "| Distillation target | Style/token | Task/token† | PSR | Effective α | Target KL | Constraint active | Steps/no-op | Train h | Peak alloc / Δ GiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("privileged", "trsd"):
        row = style[method]
        label = "Raw privileged" if method == "privileged" else "TRSD projected"
        activation = (
            "N/A"
            if row["constraint_activation_rate"] is None
            else f"{100*float(row['constraint_activation_rate']):.2f}%"
        )
        lines.append(
            f"| {label} | {fmt(row['style_error_per_token'], 5)} | "
            f"{fmt(row['task_error_per_token'], 5)} | {fmt(row['psr'], 3)} | "
            f"{fmt(row['effective_alpha'], 4)} | {fmt(row['target_student_kl'], 6)} | "
            f"{activation} | "
            f"{row['optimizer_steps']}/{row['no_op_episodes']} | "
            f"{row['training_hours']:.2f} | "
            f"{fmt(row['max_gpu_peak_allocated_gib'], 2)} / "
            f"{fmt(row['max_gpu_peak_delta_gib'], 2)} |"
        )
    lines.extend(
        [
            "",
            "† Task/token is absolute realized-token target movement. It is not an "
            "accuracy or signed task-improvement measure. Privilege-SD64 did not record "
            "training-memory telemetry, so its memory entry is N/A.",
        ]
    )

    if mechanism:
        lines.extend(
            [
                "",
                "## Same-prefix mechanism pilot (descriptive)",
                "",
                "| Target | Queries × wrappers | Style shift | Signed task-token gain | α | Target KL |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in mechanism:
            label = (
                "Raw privileged"
                if row["projection"] == "raw_privileged_surrogate"
                else "TRSD projected"
            )
            lines.append(
                f"| {label} | {row['n_queries']} × "
                f"{int(row['n_query_wrappers']) // int(row['n_queries'])} | "
                f"{fmt(row['style_abs_logprob_shift'], 6)} | "
                f"{fmt(row['task_logprob_gain'], 6)} | "
                f"{fmt(row['mean_alpha'], 4)} | {fmt(row['mean_achieved_kl'], 6)} |"
            )

    if epsilon_rows:
        lines.extend(
            [
                "",
                "## One-episode development epsilon sensitivity",
                "",
                "This label-free mechanism sweep selects the training budget on a "
                "development episode; it does not use held-out correctness.",
                "",
                "| ε | Mean α | Achieved KL | Active wrappers | Task gain / raw† | Style retention ↓ | Prompt-variance retention ↓ | Selected |",
                "|---:|---:|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in epsilon_rows:
            lines.append(
                f"| {fmt(row['epsilon'], 3)} | {fmt(row['mean_alpha'], 4)} | "
                f"{fmt(row['achieved_mean_kl'], 6)} | {row['active_wrappers']} | "
                f"{fmt(row['task_gain_vs_raw'], 3)} | "
                f"{fmt(row['style_retention_vs_raw'], 3)} | "
                f"{fmt(row['prompt_variance_retention'], 3)} | "
                f"{'✓' if row['is_selected'] else ''} |"
            )

    effect = {str(row["metric"]): row for row in effects}
    style_effect = effect["style_error_per_token"]
    task_effect = effect["task_error_per_token"]
    psr_effect = effect["psr"]
    style_reduction = -float(style_effect["relative_change"])
    lines.extend(["", "## Evidence-bounded interpretation", ""])
    if style_reduction > 0.0:
        lines.append(
            f"- On the matched 64-query stream, TRSD reduced normalized style-target "
            f"movement by **{100*style_reduction:.1f}%** relative to the raw privileged "
            f"target (paired episode bootstrap delta 95% CI "
            f"[{style_effect['delta_ci_low']:.5f}, {style_effect['delta_ci_high']:.5f}])."
        )
    else:
        lines.append(
            f"- On this stream, TRSD did not reduce the primary normalized style metric "
            f"(relative change {100*style_effect['relative_change']:+.1f}%)."
        )
    lines.append(
        f"- Normalized task-target movement changed by "
        f"{100*task_effect['relative_change']:+.1f}% and PSR changed by "
        f"{100*psr_effect['relative_change']:+.1f}%; these are distributional diagnostics, "
        "not correctness guarantees."
    )
    if mechanism:
        raw, projected = mechanism
        reduction = 1.0 - float(projected["style_abs_logprob_shift"]) / float(raw["style_abs_logprob_shift"])
        lines.append(
            f"- The separate same-prefix mechanism pilot ({raw['n_queries']} queries, "
            f"neutral/terse/verbose wrappers) observed a {100*reduction:.1f}% reduction "
            f"in style log-probability shift: {raw['style_abs_logprob_shift']:.5f} → "
            f"{projected['style_abs_logprob_shift']:.5f}. It is descriptive because n is small."
        )
    lines.extend(
        [
            "",
            "## Metric contract",
            "",
            "- Every method receives a fixed 10,240-token generation opportunity and no "
            "continuation. Strict Acc@1 is the sole accuracy estimand: unfinished or "
            "truncated responses are wrong and the Combined denominator is always 143.",
            f"- Partition version: `{partition_version}`.",
            f"- Error definition: `{error_definition}`.",
            "- Style words: accordingly, alternatively, answer, clearly, consequently, "
            "finally, first, hence, however, indeed, next, note, now, perhaps, second, "
            "similarly, step, suppose, therefore, thus, verify, we.",
            "- Task tokens are tokens containing a digit, mathematical operator/bracket, "
            "or a versioned mathematical LaTeX command (e.g. frac, sqrt, boxed, sum, mod).",
            "- The 64 episodes are paired by episode, query ID, and problem hash. Their "
            "generated trajectories need not contain identical tokens, so the journal-level "
            "comparison is query-paired; the optional mechanism pilot is the same-prefix check.",
            "- Accuracy evaluation is unprivileged. Hindsight exposure must be audited from "
            "the training journals separately; this reporter never reads target labels during training.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scored", type=Path, required=True)
    parser.add_argument("--privileged16-scored", type=Path, required=True)
    parser.add_argument("--trsd16-scored", type=Path, required=True)
    parser.add_argument(
        "--historical-trsd16-scored",
        type=Path,
        required=True,
        help="Historical pre-reverse-KL TRSD-16 scored artifact for the appendix",
    )
    parser.add_argument(
        "--historical-base-scored",
        type=Path,
        required=True,
        help="Base scored under the same protocol as --historical-trsd16-scored",
    )
    parser.add_argument("--privileged64-scored", type=Path, required=True)
    parser.add_argument("--trsd64-scored", type=Path, required=True)
    parser.add_argument("--privileged64-journal", type=Path, required=True)
    parser.add_argument("--trsd64-journal", type=Path, required=True)
    parser.add_argument("--mechanism-csv", type=Path)
    parser.add_argument(
        "--epsilon-sensitivity-csv",
        type=Path,
        help="Label-free one-episode development epsilon sweep",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-heldout", type=int, default=143)
    parser.add_argument("--expected-episodes", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    args = parser.parse_args()

    journals_raw = {
        "privileged": load_journal(
            args.privileged64_journal,
            expected_episodes=args.expected_episodes,
            name="Privilege-SD 64",
        ),
        "trsd": load_journal(
            args.trsd64_journal,
            expected_episodes=args.expected_episodes,
            name="TRSD 64",
        ),
    }
    match_journals(journals_raw["privileged"], journals_raw["trsd"])
    journal_rows = {
        method: [episode_style_row(row, method) for row in rows]
        for method, rows in journals_raw.items()
    }
    style_summaries = {
        method: aggregate_style(rows) for method, rows in journal_rows.items()
    }
    style_summaries["privileged"]["effective_alpha"] = 1.0
    style_summaries["privileged"]["target_student_kl"] = style_summaries[
        "privileged"
    ]["mean_teacher_student_kl"]
    style_summaries["trsd"]["effective_alpha"] = style_summaries["trsd"][
        "mean_alpha"
    ]
    style_summaries["trsd"]["target_student_kl"] = style_summaries["trsd"][
        "mean_achieved_kl"
    ]
    bootstrap = paired_bootstrap(
        journal_rows["privileged"],
        journal_rows["trsd"],
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )

    style_summary_rows: list[dict[str, Any]] = []
    for method in ("privileged", "trsd"):
        row = {"method": method, **style_summaries[method]}
        for metric in ("style_error_per_token", "task_error_per_token", "psr"):
            low, high = ci95(bootstrap[f"{method}_{metric}"])
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        style_summary_rows.append(row)

    effect_rows: list[dict[str, Any]] = []
    for metric in ("style_error_per_token", "task_error_per_token", "psr"):
        p_value = float(style_summaries["privileged"][metric])
        t_value = float(style_summaries["trsd"][metric])
        delta_low, delta_high = ci95(bootstrap[f"delta_{metric}"])
        ratio_low, ratio_high = ci95(bootstrap[f"ratio_{metric}"])
        effect_rows.append(
            {
                "metric": metric,
                "privileged": p_value,
                "trsd": t_value,
                "delta_trsd_minus_privileged": t_value - p_value,
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
                "ratio_trsd_over_privileged": t_value / p_value,
                "ratio_ci_low": ratio_low,
                "ratio_ci_high": ratio_high,
                "relative_change": t_value / p_value - 1.0,
            }
        )

    scored = {
        "base": load_scored(args.base_scored, name="Base", expected_total=args.expected_heldout),
        "privileged_16": load_scored(
            args.privileged16_scored,
            name="Privilege-SD 16",
            expected_total=args.expected_heldout,
        ),
        "trsd_16": load_scored(
            args.trsd16_scored,
            name="TRSD 16",
            expected_total=args.expected_heldout,
        ),
        "privileged_64": load_scored(
            args.privileged64_scored,
            name="Privilege-SD 64",
            expected_total=args.expected_heldout,
        ),
        "trsd_64": load_scored(
            args.trsd64_scored,
            name="TRSD 64",
            expected_total=args.expected_heldout,
        ),
    }
    matched_scored = {method: scored[method] for method in MATCHED_INFERENCE_METHODS}
    match_scored(matched_scored)
    historical_trsd16 = load_scored(
        args.historical_trsd16_scored,
        name="Historical TRSD 16 appendix",
        expected_total=args.expected_heldout,
    )
    validate_historical_trsd16(historical_trsd16)
    historical_base = load_scored(
        args.historical_base_scored,
        name="Historical Base for TRSD 16",
        expected_total=args.expected_heldout,
    )
    match_scored({"base": historical_base, "trsd_16": historical_trsd16})
    historical_reference = build_historical_trsd16_reference(
        historical_trsd16, historical_base
    )
    heldout_rows = [
        row
        for method in METHOD_ORDER
        for row in aggregate_scored(method, scored[method])
    ]
    heldout_robustness = build_heldout_robustness(matched_scored)
    transition_rows = [
        paired_transitions(scored["base"], scored[method], method)
        for method in MATCHED_INFERENCE_METHODS
        if method != "base"
    ]
    mechanism = mechanism_summary(args.mechanism_csv)
    epsilon_rows = epsilon_sensitivity(args.epsilon_sensitivity_csv)

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    episode_fields = list(journal_rows["privileged"][0])
    write_csv(
        root / "matched64_episode_style_metrics.csv",
        [*journal_rows["privileged"], *journal_rows["trsd"]],
        episode_fields,
    )
    write_csv(root / "matched64_style_summary.csv", style_summary_rows, list(style_summary_rows[0]))
    write_csv(root / "matched64_paired_effects.csv", effect_rows, list(effect_rows[0]))
    write_csv(root / "heldout_main_table.csv", heldout_rows, list(heldout_rows[0]))
    write_csv(
        root / "heldout_robustness.csv",
        heldout_robustness,
        list(heldout_robustness[0]),
    )
    write_csv(
        root / "historical_trsd16_reference.csv",
        historical_reference,
        list(historical_reference[0]),
    )
    write_csv(root / "heldout_paired_transitions.csv", transition_rows, list(transition_rows[0]))
    if mechanism:
        write_csv(root / "same_prefix_mechanism_summary.csv", mechanism, list(mechanism[0]))
    if epsilon_rows:
        write_csv(root / "epsilon_dev_sensitivity.csv", epsilon_rows, list(epsilon_rows[0]))

    partition_version = str(journal_rows["privileged"][0]["partition_version"])
    error_definition = str(journal_rows["privileged"][0]["error_definition"])
    if str(journal_rows["trsd"][0]["partition_version"]) != partition_version:
        raise ReportError("Privilege-SD and TRSD use different partition versions")
    summary = {
        "schema_version": "trsd-final-matched-style-report-v5",
        "protocol": {
            "heldout_queries": args.expected_heldout,
            "matched_training_episodes": args.expected_episodes,
            "style_bootstrap_unit": "paired_episode_query",
            "style_bootstrap_replicates": args.bootstrap_replicates,
            "style_bootstrap_seed": args.bootstrap_seed,
            "heldout_primary_estimand": "strict_acc1_truncated_is_wrong",
            "heldout_bootstrap_unit": "paired_query",
            "heldout_bootstrap_replicates": HELDOUT_BOOTSTRAP_REPLICATES,
            "heldout_bootstrap_seed": HELDOUT_BOOTSTRAP_SEED,
            "heldout_test": "exact_two_sided_mcnemar_binomial",
            "matched_inference_methods": list(MATCHED_INFERENCE_METHODS),
            "historical_trsd16_status": "appendix_point_estimate_only_not_current_protocol_matched",
            "partition_version": partition_version,
            "error_definition": error_definition,
        },
        "style_summary": style_summaries,
        "paired_effects": effect_rows,
        "heldout": heldout_rows,
        "heldout_robustness": heldout_robustness,
        "historical_trsd16_reference": historical_reference,
        "paired_transitions": transition_rows,
        "same_prefix_mechanism": mechanism,
        "epsilon_dev_sensitivity": epsilon_rows,
    }
    atomic_text(root / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(
        root / "heldout_robustness.json",
        json.dumps(
            {
                "schema_version": "trsd-heldout-robustness-v2",
                "estimand": "strict_acc1_truncated_is_wrong",
                "bootstrap_unit": "paired_query",
                "bootstrap_replicates": HELDOUT_BOOTSTRAP_REPLICATES,
                "bootstrap_seed": HELDOUT_BOOTSTRAP_SEED,
                "mcnemar_test": "exact_two_sided_binomial_on_discordant_pairs",
                "matched_inference_methods": list(MATCHED_INFERENCE_METHODS),
                "excluded_from_current_inference": ["historical_trsd_16_appendix"],
                "historical_trsd16_reference": historical_reference,
                "rows": heldout_robustness,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    atomic_text(
        root / "HELDOUT_ROBUSTNESS.md",
        "# Held-out robustness\n\n"
        + heldout_robustness_markdown(heldout_robustness)
        + "\n\n"
        + historical_reference_markdown(historical_reference)
        + "\n",
    )
    atomic_text(
        root / "FINAL_RESULTS.md",
        report_markdown(
            heldout_rows,
            heldout_robustness,
            historical_reference,
            style_summaries,
            effect_rows,
            mechanism,
            epsilon_rows,
            partition_version,
            error_definition,
        ),
    )
    plot_style(root, style_summaries, bootstrap, mechanism)
    gc.collect()
    plot_heldout(root, heldout_rows)
    gc.collect()
    plot_epsilon(root, epsilon_rows)
    atomic_text(root / "REPORT_COMPLETE", "complete\n")


if __name__ == "__main__":
    try:
        main()
    except ReportError as error:
        raise SystemExit(f"Refusing final report build: {error}") from error
