#!/usr/bin/env python3
"""Build the final matched TRSD/privileged style and held-out report.

The reporter is deliberately model-free.  It consumes four already-scored
held-out JSONL files and the two 64-episode training journals, validates that
their query identities agree, and writes paper-ready CSV, Markdown, PNG, PDF,
and JSON artifacts.  Missing observations are reported as N/A; no metric is
imputed.

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
METHOD_ORDER = ("base", "privileged_16", "privileged_64", "trsd_64")
METHOD_LABELS = {
    "base": "Base",
    "privileged_16": "Privilege-SD 16",
    "privileged_64": "Privilege-SD 64",
    "trsd_64": "TRSD 64",
}
COLORS = {
    "base": "#64748B",
    "privileged_16": "#F59E0B",
    "privileged_64": "#C2410C",
    "trsd_64": "#0F766E",
    "privileged": "#C2410C",
    "trsd": "#0F766E",
}


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
    # scalar fields used below so loading four methods does not duplicate those
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
                "generated_tokens",
                "behavioral_diagnostics",
                "resource_usage",
                "parsed_answer",
                "max_new_tokens",
                "checkpoint_episode",
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
    if expected_total == sum(EXPECTED_SOURCES.values()) and dict(source_counts) != EXPECTED_SOURCES:
        raise ReportError(
            f"{name} source counts {dict(source_counts)} do not equal {EXPECTED_SOURCES}"
        )
    return rows


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
        correct = sum(int(float(row["correct"])) for row in values)
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
                "dataset": source,
                "correct": correct,
                "n": len(values),
                "acc1": correct / len(values),
                "acc1_percent": 100.0 * correct / len(values),
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
    return {
        "method": method,
        "n": len(pairs),
        "wrong_to_correct": sum(not bool(a["correct"]) and bool(b["correct"]) for a, b in pairs),
        "correct_to_wrong": sum(bool(a["correct"]) and not bool(b["correct"]) for a, b in pairs),
        "correct_to_correct": sum(bool(a["correct"]) and bool(b["correct"]) for a, b in pairs),
        "wrong_to_wrong": sum(not bool(a["correct"]) and not bool(b["correct"]) for a, b in pairs),
        "parsed_answer_changes": sum(
            str(a.get("parsed_answer", "")) != str(b.get("parsed_answer", ""))
            for a, b in pairs
        ),
    }


def mechanism_summary(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ReportError(f"Mechanism CSV is empty: {path}")
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
        ("task_error_per_token", "Normalized task signal", "|Δ log p| / task token"),
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


def plot_heldout(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    plt = configure_plotting()
    lookup = {(str(row["method"]), str(row["dataset"])): row for row in rows}
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), constrained_layout=True)
    width = 0.19
    x = list(range(len(SOURCE_ORDER)))
    for method_index, method in enumerate(METHOD_ORDER):
        offset = (method_index - 1.5) * width
        values = [float(lookup[(method, source)]["acc1_percent"]) for source in SOURCE_ORDER]
        axes[0, 0].bar(
            [position + offset for position in x],
            values,
            width * 0.92,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0, 0].set_xticks(x, [SOURCE_LABELS[source] for source in SOURCE_ORDER])
    axes[0, 0].set_ylabel("10k-budget Acc@1 (%) ↑")
    axes[0, 0].set_title("Unprivileged held-out performance", fontweight="semibold")
    axes[0, 0].legend(fontsize=8, ncol=2)

    combined = {method: lookup[(method, "combined")] for method in METHOD_ORDER}
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    colors = [COLORS[method] for method in METHOD_ORDER]
    panels = (
        (axes[0, 1], "mean_generated_tokens", 1.0, "Mean response tokens"),
        (axes[1, 0], "hedging_tokens_per_1k", 1.0, "Hedging tokens / 1k (diagnostic)"),
        (axes[1, 1], "fabricated_reference_rate", 100.0, "Fabricated-reference rate (%) ↓"),
    )
    for axis, field, scale, title in panels:
        values = [float(combined[method][field] or 0.0) * scale for method in METHOD_ORDER]
        bars = axis.bar(labels, values, color=colors, width=0.68)
        axis.set_title(title, fontweight="semibold")
        axis.tick_params(axis="x", rotation=14)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=8)
    figure.suptitle(
        "Final-checkpoint accuracy and response-shape diagnostics",
        fontsize=12,
        fontweight="semibold",
    )
    save_figure(figure, root, "heldout_accuracy_behavior")
    plt.close(figure)


def fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def report_markdown(
    heldout: Sequence[Mapping[str, Any]],
    style: Mapping[str, Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    mechanism: Sequence[Mapping[str, Any]],
    partition_version: str,
    error_definition: str,
) -> str:
    lookup = {(row["method"], row["dataset"]): row for row in heldout}
    lines = [
        "# Final TRSD matched report",
        "",
        "## Held-out 10k-budget Acc@1 and behavior",
        "",
        "| Method | Combined | AMC23 | AIME24 | AIME25 | Hedge/1k | Fabricated ref. | Sec/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        combined = lookup[(method, "combined")]
        lines.append(
            f"| {METHOD_LABELS[method]} | {combined['acc1_percent']:.2f}% "
            f"({combined['correct']}/{combined['n']}) | "
            f"{lookup[(method, 'amc23')]['acc1_percent']:.2f}% | "
            f"{lookup[(method, 'aime24')]['acc1_percent']:.2f}% | "
            f"{lookup[(method, 'aime25')]['acc1_percent']:.2f}% | "
            f"{fmt(combined['hedging_tokens_per_1k'], 2)} | "
            f"{100*combined['fabricated_reference_rate']:.2f}% | "
            f"{fmt(combined['mean_generation_seconds'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Matched 64-episode target diagnostics",
            "",
            "| Distillation target | Style | Task | PSR | α | Achieved KL | Steps/no-op | Train h |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("privileged", "trsd"):
        row = style[method]
        label = "Raw privileged" if method == "privileged" else "TRSD projected"
        lines.append(
            f"| {label} | {fmt(row['style_error_per_token'], 5)} | "
            f"{fmt(row['task_error_per_token'], 5)} | {fmt(row['psr'], 3)} | "
            f"{fmt(row['mean_alpha'], 4)} | {fmt(row['mean_achieved_kl'], 6)} | "
            f"{row['optimizer_steps']}/{row['no_op_episodes']} | "
            f"{row['training_hours']:.2f} |"
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
            "- 10k-budget Acc@1 counts a query iff a correct boxed answer appears within "
            "the fixed 10,240-token generation opportunity; all 143 queries remain in "
            "the denominator, with no continuation, filtering, or reweighting.",
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
    parser.add_argument("--privileged64-scored", type=Path, required=True)
    parser.add_argument("--trsd64-scored", type=Path, required=True)
    parser.add_argument("--privileged64-journal", type=Path, required=True)
    parser.add_argument("--trsd64-journal", type=Path, required=True)
    parser.add_argument("--mechanism-csv", type=Path)
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
    match_scored(scored)
    heldout_rows = [
        row
        for method in METHOD_ORDER
        for row in aggregate_scored(method, scored[method])
    ]
    transition_rows = [
        paired_transitions(scored["base"], scored[method], method)
        for method in METHOD_ORDER
        if method != "base"
    ]
    mechanism = mechanism_summary(args.mechanism_csv)

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
    write_csv(root / "heldout_paired_transitions.csv", transition_rows, list(transition_rows[0]))
    if mechanism:
        write_csv(root / "same_prefix_mechanism_summary.csv", mechanism, list(mechanism[0]))

    partition_version = str(journal_rows["privileged"][0]["partition_version"])
    error_definition = str(journal_rows["privileged"][0]["error_definition"])
    if str(journal_rows["trsd"][0]["partition_version"]) != partition_version:
        raise ReportError("Privilege-SD and TRSD use different partition versions")
    summary = {
        "schema_version": "trsd-final-matched-style-report-v1",
        "protocol": {
            "heldout_queries": args.expected_heldout,
            "matched_training_episodes": args.expected_episodes,
            "bootstrap_unit": "paired_episode_query",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "partition_version": partition_version,
            "error_definition": error_definition,
        },
        "style_summary": style_summaries,
        "paired_effects": effect_rows,
        "heldout": heldout_rows,
        "paired_transitions": transition_rows,
        "same_prefix_mechanism": mechanism,
    }
    atomic_text(root / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(
        root / "FINAL_RESULTS.md",
        report_markdown(
            heldout_rows,
            style_summaries,
            effect_rows,
            mechanism,
            partition_version,
            error_definition,
        ),
    )
    plot_style(root, style_summaries, bootstrap, mechanism)
    gc.collect()
    plot_heldout(root, heldout_rows)
    atomic_text(root / "REPORT_COMPLETE", "complete\n")


if __name__ == "__main__":
    try:
        main()
    except ReportError as error:
        raise SystemExit(f"Refusing final report build: {error}") from error
