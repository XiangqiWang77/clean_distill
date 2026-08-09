#!/usr/bin/env python3
"""Plot exact teacher--student KL against optimizer step.

The persistent runs currently perform one optimizer update for every recorded
episode.  This script verifies that invariant and constructs the x-axis from
the cumulative number of successful optimizer steps instead of reusing the
episode index.  Additional methods (for example a teacher-forcing baseline)
can be supplied as ``--run LABEL=episodes.jsonl`` without changing the plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = (
    (
        "Privilege-SD",
        Path(
            "/home/da839/scratch_pi_mg269/da839/clean_distill/runs/"
            "csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/"
            "episodes.jsonl"
        ),
    ),
    (
        "TRSD",
        Path(
            "/home/da839/scratch_pi_mg269/da839/clean_distill/runs/"
            "reverse-kl-matched64-20260807/trsd/train/episodes.jsonl"
        ),
    ),
)

COLORS = {
    "Privilege-SD": "#B97925",
    "Teacher forcing": "#4169A1",
    "TF": "#4169A1",
    "TRSD": "#D5533D",
}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=/path/to/episodes.jsonl")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run label and path must both be non-empty")
    return label.strip(), Path(raw_path).expanduser()


def load_trace(label: str, path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: journal does not exist: {path}")
    rows: list[dict[str, Any]] = []
    optimizer_step = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("optimizer_step"), bool):
                raise ValueError(f"{path}:{line_number}: optimizer_step must be boolean")
            if row["optimizer_step"]:
                optimizer_step += 1
                value = float(row["mean_teacher_student_kl"])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{path}:{line_number}: invalid KL {value!r}")
                rows.append(
                    {
                        "method": label,
                        "training_step": optimizer_step,
                        "journal_episode": int(row["episode"]),
                        "teacher_student_reverse_kl": value,
                        "query_id": str(row.get("query_id", "")),
                    }
                )
    if not rows:
        raise ValueError(f"{label}: no successful optimizer steps in {path}")
    return rows


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = float(values[start : index + 1].mean())
    return result


def write_csv(path: Path, traces: list[list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "method",
        "training_step",
        "journal_episode",
        "teacher_student_reverse_kl",
        "query_id",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for trace in traces:
            writer.writerows(trace)


def plot(path: Path, traces: list[list[dict[str, Any]]], window: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    fallback = plt.get_cmap("tab10")
    for index, trace in enumerate(traces):
        label = str(trace[0]["method"])
        steps = np.asarray([row["training_step"] for row in trace], dtype=int)
        values = np.asarray(
            [row["teacher_student_reverse_kl"] for row in trace], dtype=float
        )
        color = COLORS.get(label, fallback(index % 10))
        axis.plot(steps, values, color=color, alpha=0.20, linewidth=0.9)
        axis.plot(
            steps,
            moving_average(values, window),
            color=color,
            linewidth=2.35,
            label=f"{label} ({window}-step mean)",
        )

    means = {
        str(trace[0]["method"]): float(
            np.mean([row["teacher_student_reverse_kl"] for row in trace])
        )
        for trace in traces
    }
    if {"Privilege-SD", "TRSD"} <= means.keys():
        reduction = 100.0 * (1.0 - means["TRSD"] / means["Privilege-SD"])
        axis.text(
            0.985,
            0.78,
            f"Mean KL: {means['Privilege-SD']:.4f} → {means['TRSD']:.4f}\n"
            f"TRSD reduction: {reduction:.1f}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=COLORS["TRSD"],
            fontweight="bold",
        )

    axis.set_xlabel("Training step")
    axis.set_ylabel(r"Teacher--student KL  $D_{KL}(\pi_s\,\|\,q)$")
    axis.set_title(
        "Teacher--student KL across training steps", loc="left", fontweight="bold"
    )
    axis.grid(axis="both", color="#D8DDE3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.set_xlim(left=1)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        default=[],
        help="LABEL=episodes.jsonl; repeat to add curves (for example TF)",
    )
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/training_step_teacher_student_kl.png"),
    )
    args = parser.parse_args()
    if args.window <= 0:
        raise ValueError("--window must be positive")
    run_specs = args.run or list(DEFAULT_RUNS)
    traces = [load_trace(label, path) for label, path in run_specs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output.with_suffix(".csv"), traces)
    plot(args.output, traces, args.window)
    print(
        json.dumps(
            {
                "figure": str(args.output),
                "pdf": str(args.output.with_suffix('.pdf')),
                "csv": str(args.output.with_suffix('.csv')),
                "methods": {
                    trace[0]["method"]: len(trace) for trace in traces
                },
                "x_axis": "cumulative successful optimizer steps",
                "rolling_window": args.window,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
