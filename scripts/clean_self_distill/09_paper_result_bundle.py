#!/usr/bin/env python3
"""Render the full held-out TRSD paper-result bundle from scored artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASETS = ("combined", "amc23", "aime24", "aime25")
LABELS = {"combined": "Combined", "amc23": "AMC23", "aime24": "AIME24", "aime25": "AIME25"}
EXPECTED_SOURCE_COUNTS = {"amc23": 83, "aime24": 30, "aime25": 30}
METHODS = ("base", "privileged_sd", "trsd")
METHOD_LABELS = {
    "base": "Base",
    "privileged_sd": "Privileged-SD",
    "trsd": "TRSD",
}
COLORS = {
    "base": "#6B7280",
    "privileged_sd": "#D97706",
    "trsd": "#0F766E",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in rows if row.get("profile") == "acc1"]
    if not rows:
        raise ValueError(f"No Acc@1 rows in {path}")
    return rows


def validate_full_artifact(name: str, rows: list[dict[str, Any]]) -> None:
    query_ids = [str(row.get("query_id", "")) for row in rows]
    if any(not query_id for query_id in query_ids):
        raise ValueError(f"{name} contains an Acc@1 row without query_id")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError(f"{name} contains duplicate Acc@1 query_id values")
    actual_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        actual_counts[str(row.get("source", ""))] += 1
        if int(row.get("max_new_tokens", -1)) != 10240:
            raise ValueError(f"{name} is not uniformly evaluated at 10,240 tokens")
    if dict(actual_counts) != EXPECTED_SOURCE_COUNTS:
        raise ValueError(
            f"{name} has source counts {dict(actual_counts)}; "
            f"expected {EXPECTED_SOURCE_COUNTS}"
        )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    groups = {"combined": rows, **by_source}
    result: list[dict[str, Any]] = []
    for dataset in DATASETS:
        values = groups.get(dataset, [])
        if not values:
            raise ValueError(f"{method} lacks dataset {dataset}")
        correct = sum(int(row["correct"]) for row in values)
        generation_seconds = [float(row["resource_usage"]["generation_seconds"]) for row in values]
        peak_memory = [float(row["resource_usage"]["cuda_peak_memory_allocated_bytes"]) for row in values]
        result.append(
            {
                "method": method,
                "dataset": dataset,
                "correct": correct,
                "n": len(values),
                "acc1": correct / len(values),
                "acc1_percent": 100.0 * correct / len(values),
                "truncated": sum(bool(row["truncated"]) for row in values),
                "truncation_rate": sum(bool(row["truncated"]) for row in values) / len(values),
                "mean_generated_tokens": sum(float(row["generated_tokens"]) for row in values) / len(values),
                "mean_generation_seconds": sum(generation_seconds) / len(generation_seconds),
                "peak_gpu_allocated_gib": max(peak_memory) / (1024.0**3),
            }
        )
    return result


def transitions(base: list[dict[str, Any]], comparator: list[dict[str, Any]], method: str) -> dict[str, Any]:
    base_by_id = {str(row["query_id"]): row for row in base}
    comp_by_id = {str(row["query_id"]): row for row in comparator}
    if set(base_by_id) != set(comp_by_id):
        raise ValueError(f"{method} query coverage differs from Base")
    paired = [(base_by_id[key], comp_by_id[key]) for key in sorted(base_by_id)]
    mismatched = [
        str(a["query_id"])
        for a, b in paired
        if str(a.get("problem_sha256", "")) != str(b.get("problem_sha256", ""))
        or str(a.get("source", "")) != str(b.get("source", ""))
    ]
    if mismatched:
        raise ValueError(f"{method} disagrees with Base metadata for {mismatched[:5]}")
    return {
        "method": method,
        "n": len(paired),
        "wrong_to_correct": sum(int(a["correct"]) == 0 and int(b["correct"]) == 1 for a, b in paired),
        "correct_to_wrong": sum(int(a["correct"]) == 1 and int(b["correct"]) == 0 for a, b in paired),
        "correct_to_correct": sum(int(a["correct"]) == 1 and int(b["correct"]) == 1 for a, b in paired),
        "wrong_to_wrong": sum(int(a["correct"]) == 0 and int(b["correct"]) == 0 for a, b in paired),
        "parsed_answer_changes": sum(str(a.get("parsed_answer", "")) != str(b.get("parsed_answer", "")) for a, b in paired),
    }


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
            "grid.alpha": 0.45,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    return plt


def save(figure: Any, root: Path, name: str) -> None:
    figure.savefig(root / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(root / f"{name}.pdf", bbox_inches="tight")


def plot_main(
    plt: Any,
    rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    root: Path,
) -> None:
    lookup = {(row["method"], row["dataset"]): row for row in rows}
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), constrained_layout=True)
    centers = list(range(len(DATASETS)))
    width = 0.26
    for index, method in enumerate(METHODS):
        offset = (index - 1.0) * width
        values = [lookup[(method, dataset)]["acc1_percent"] for dataset in DATASETS]
        bars = axes[0].bar(
            [value + offset for value in centers],
            values,
            width * 0.92,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[0].bar_label(
            bars,
            labels=[f"{value:.1f}" for value in values],
            padding=2,
            fontsize=7.5,
        )
    axes[0].set_xticks(centers, [LABELS[value] for value in DATASETS])
    axes[0].set_ylabel("Acc@1 (%) ↑")
    maximum_accuracy = max(
        float(lookup[(method, dataset)]["acc1_percent"])
        for method in METHODS
        for dataset in DATASETS
    )
    axes[0].set_ylim(0, min(100.0, max(85.0, maximum_accuracy + 8.0)))
    axes[0].set_title("Held-out mathematical reasoning")
    axes[0].legend(loc="upper right")

    names = ("Wrong→correct", "Correct→wrong", "Answer changed")
    transition_lookup = {row["method"]: row for row in transition_rows}
    transition_centers = list(range(len(names)))
    transition_methods = ("privileged_sd", "trsd")
    transition_width = 0.34
    for index, method in enumerate(transition_methods):
        transition = transition_lookup[method]
        values = (
            transition["wrong_to_correct"],
            transition["correct_to_wrong"],
            transition["parsed_answer_changes"],
        )
        offset = (index - 0.5) * transition_width
        bars = axes[1].bar(
            [value + offset for value in transition_centers],
            values,
            transition_width * 0.92,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[1].bar_label(bars, padding=3, fontsize=8)
    axes[1].set_xticks(transition_centers, names)
    axes[1].set_ylabel("Queries")
    axes[1].set_title("Paired behavioral transitions vs Base")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].legend(loc="upper left")
    figure.suptitle("10,240-token full held-out evaluation (143 queries)", fontsize=12, fontweight="semibold")
    save(figure, root, "heldout_main_performance")
    plt.close(figure)


def plot_operational(plt: Any, rows: list[dict[str, Any]], root: Path) -> None:
    combined = {row["method"]: row for row in rows if row["dataset"] == "combined"}
    labels = tuple(METHOD_LABELS[method] for method in METHODS)
    specs = (
        ("truncation_rate", "Truncation rate (%) ↓", 100.0),
        ("mean_generated_tokens", "Mean generated tokens ↓", 1.0),
        ("mean_generation_seconds", "Seconds / query ↓", 1.0),
        ("peak_gpu_allocated_gib", "Peak GPU allocated (GiB) ↓", 1.0),
    )
    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.65), constrained_layout=True)
    for axis, (field, title, scale) in zip(axes, specs):
        values = [float(combined[method][field]) * scale for method in METHODS]
        bars = axis.bar(
            labels,
            values,
            color=[COLORS[method] for method in METHODS],
            width=0.68,
        )
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=8)
        axis.set_title(title, fontsize=9.5)
        axis.tick_params(axis="x", rotation=15)
    figure.suptitle("Operational profile under the identical generation budget", fontsize=12, fontweight="semibold")
    save(figure, root, "heldout_operational_profile")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--privileged", type=Path, required=True)
    parser.add_argument("--trsd", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = read_jsonl(args.base)
    privileged = read_jsonl(args.privileged)
    trsd = read_jsonl(args.trsd)
    validate_full_artifact("Base", base)
    validate_full_artifact("Privileged-SD", privileged)
    validate_full_artifact("TRSD", trsd)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    rows = (
        aggregate("base", base)
        + aggregate("privileged_sd", privileged)
        + aggregate("trsd", trsd)
    )
    paired = [
        transitions(base, privileged, "privileged_sd"),
        transitions(base, trsd, "trsd"),
    ]
    write_csv(root / "heldout_main_table.csv", rows)
    write_csv(root / "paired_transitions.csv", paired)

    summary = {
        "schema_version": "trsd-paper-result-bundle-v2",
        "protocol": {"queries": 143, "max_new_tokens": 10240, "sample_count": 1},
        "heldout": rows,
        "paired_transitions": paired,
        "current_trsd_full_143_status": "measured",
        "pilot_report": str(args.pilot_report.resolve()),
    }
    atomic_text(root / "main_table_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lookup = {(row["method"], row["dataset"]): row for row in rows}
    markdown = [
        "# TRSD empirical result bundle\n",
        "## Full 10,240-token held-out table\n",
        "| Method | Combined | Δ vs Base | AMC23 | AIME24 | AIME25 | Trunc. | Sec/query | Peak GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base_combined = lookup[("base", "combined")]["acc1_percent"]
    for method in METHODS:
        label = METHOD_LABELS[method]
        combined = lookup[(method, "combined")]
        delta = combined["acc1_percent"] - base_combined
        markdown.append(
            f"| {label} | {combined['acc1_percent']:.2f}% | "
            f"{delta:+.2f} pp | "
            f"{lookup[(method, 'amc23')]['acc1_percent']:.2f}% | "
            f"{lookup[(method, 'aime24')]['acc1_percent']:.2f}% | "
            f"{lookup[(method, 'aime25')]['acc1_percent']:.2f}% | "
            f"{100 * combined['truncation_rate']:.2f}% | "
            f"{combined['mean_generation_seconds']:.1f} | {combined['peak_gpu_allocated_gib']:.2f} |"
        )
    markdown.extend(
        [
            "\n## Paired changes\n",
            *[
                f"{METHOD_LABELS[row['method']]} produces {row['wrong_to_correct']} "
                f"wrong→correct and {row['correct_to_wrong']} correct→wrong "
                "transitions relative to Base.\n"
                for row in paired
            ],
            "The short pilot and mechanism diagnostics remain separate supporting "
            "evidence; the TRSD row above is the full 143-query final-checkpoint evaluation.\n",
        ]
    )
    atomic_text(root / "PAPER_RESULTS.md", "\n".join(markdown))
    plt = configure_plotting()
    plot_main(plt, rows, paired, root)
    plot_operational(plt, rows, root)


if __name__ == "__main__":
    main()
