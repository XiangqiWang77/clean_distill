#!/usr/bin/env python3
"""Plot same-prefix prompt sensitivity across multiple mechanism queries.

Each input contains one student trajectory scored under the neutral, terse,
and verbose answer-free wrappers.  For every fixed token position, this script
computes the population variance of the three realized-token log-probability
shifts before and after TRSD projection.  Large JSON artifacts are streamed
through ``jq`` so the CPU reporting step does not load all token traces into
Python memory at once.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from statistics import pvariance
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WRAPPERS = ("neutral", "terse", "verbose")
QUERY_COLORS = ("#2563A6", "#2A9D8F", "#8B5CF6")
LOWER = "#1D7EA8"
HIGHER = "#D97706"
TIED = "#9CA3AF"
INK = "#1F2937"
MUTED = "#667085"
GRID = "#DDE3EA"


class PromptSensitivityError(ValueError):
    """Raised when mechanism inputs cannot support the claimed plot."""


def _jq(path: Path, expression: str) -> subprocess.Popen[str]:
    if not path.is_file():
        raise PromptSensitivityError(f"Missing mechanism artifact: {path}")
    return subprocess.Popen(
        ["jq", "-c", expression, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def _metadata(path: Path) -> dict[str, Any]:
    expression = (
        "{query_id,query_index,generated_tokens,selected_epsilon,"
        "full_vocabulary_kl,labels_loaded,label_paths_accepted_by_cli,"
        "wrappers:[.wrappers[].wrapper_id]}"
    )
    completed = subprocess.run(
        ["jq", "-c", expression, str(path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise PromptSensitivityError(
            f"jq metadata extraction failed for {path}: {completed.stderr.strip()}"
        )
    try:
        row = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PromptSensitivityError(f"Invalid metadata JSON for {path}") from exc
    if row.get("wrappers") != list(WRAPPERS):
        raise PromptSensitivityError(
            f"{path} must contain wrappers in order {WRAPPERS!r}"
        )
    if row.get("full_vocabulary_kl") is not True:
        raise PromptSensitivityError(f"{path} does not contain full-vocabulary KL")
    if row.get("labels_loaded") is not False or row.get("label_paths_accepted_by_cli") is not False:
        raise PromptSensitivityError(f"{path} mechanism collection was not label-free")
    if int(row.get("generated_tokens", 0)) <= 0:
        raise PromptSensitivityError(f"{path} has no generated tokens")
    return row


def _token_rows(path: Path, query_order: int) -> Iterable[dict[str, Any]]:
    metadata = _metadata(path)
    expression = (
        "[.wrappers[].token_trace] as $t | "
        "range(0; .generated_tokens) as $i | "
        "[$i,$t[0][$i].token_id,$t[0][$i].token_text,$t[0][$i].token_category,"
        "$t[0][$i].raw_surrogate_logratio,$t[1][$i].raw_surrogate_logratio,"
        "$t[2][$i].raw_surrogate_logratio,"
        "$t[0][$i].projected_surrogate_logratio,"
        "$t[1][$i].projected_surrogate_logratio,"
        "$t[2][$i].projected_surrogate_logratio]"
    )
    process = _jq(path, expression)
    assert process.stdout is not None
    expected_position = 0
    for line in process.stdout:
        values = json.loads(line)
        if len(values) != 10:
            raise PromptSensitivityError(f"Malformed token row in {path}")
        position = int(values[0])
        if position != expected_position:
            raise PromptSensitivityError(
                f"Non-contiguous token positions in {path}: {position} != {expected_position}"
            )
        expected_position += 1
        raw = tuple(float(value) for value in values[4:7])
        projected = tuple(float(value) for value in values[7:10])
        if not all(math.isfinite(value) for value in (*raw, *projected)):
            raise PromptSensitivityError(f"Non-finite token shift in {path} at {position}")
        raw_variance = float(pvariance(raw))
        projected_variance = float(pvariance(projected))
        scale = max(raw_variance, projected_variance)
        tolerance = 1e-15 + 1e-9 * scale
        relation = (
            "lower"
            if projected_variance < raw_variance - tolerance
            else "higher"
            if projected_variance > raw_variance + tolerance
            else "tied"
        )
        yield {
            "query_order": query_order,
            "query_index": int(metadata["query_index"]),
            "query_id": str(metadata["query_id"]),
            "epsilon": float(metadata["selected_epsilon"]),
            "position": position,
            "token_id": int(values[1]),
            "token_text": str(values[2]),
            "token_category": str(values[3]),
            "neutral_raw_shift": raw[0],
            "terse_raw_shift": raw[1],
            "verbose_raw_shift": raw[2],
            "neutral_projected_shift": projected[0],
            "terse_projected_shift": projected[1],
            "verbose_projected_shift": projected[2],
            "raw_across_wrapper_variance": raw_variance,
            "projected_across_wrapper_variance": projected_variance,
            "relation_to_identity": relation,
        }
    _, stderr = process.communicate()
    if process.returncode != 0:
        raise PromptSensitivityError(
            f"jq token extraction failed for {path}: {stderr.strip()}"
        )
    if expected_position != int(metadata["generated_tokens"]):
        raise PromptSensitivityError(
            f"{path} declared {metadata['generated_tokens']} tokens but yielded {expected_position}"
        )


def _summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    groups = [(f"query_{index + 1}", [row for row in rows if row["query_order"] == index])
              for index in sorted({int(row["query_order"]) for row in rows})]
    groups.append(("all_queries", list(rows)))
    for label, group in groups:
        raw = np.asarray([float(row["raw_across_wrapper_variance"]) for row in group])
        projected = np.asarray(
            [float(row["projected_across_wrapper_variance"]) for row in group]
        )
        counts = {
            relation: sum(row["relation_to_identity"] == relation for row in group)
            for relation in ("lower", "tied", "higher")
        }
        raw_mean = float(raw.mean())
        projected_mean = float(projected.mean())
        summaries.append(
            {
                "scope": label,
                "n_queries": len({row["query_id"] for row in group}),
                "n_tokens": len(group),
                "mean_raw_variance": raw_mean,
                "mean_projected_variance": projected_mean,
                "mean_variance_reduction_pct": 100.0 * (1.0 - projected_mean / raw_mean),
                "lower_count": counts["lower"],
                "tied_count": counts["tied"],
                "higher_count": counts["higher"],
                "lower_pct": 100.0 * counts["lower"] / len(group),
                "tied_pct": 100.0 * counts["tied"] / len(group),
                "higher_pct": 100.0 * counts["higher"] / len(group),
                "query_replicates": len({row["query_id"] for row in group}),
                "inference_scope": "descriptive_same_prefix_case_study",
            }
        )
    return summaries


def _example(rows: Sequence[dict[str, Any]], token: str) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["token_category"] == "style"
        and str(row["token_text"]).strip().casefold() == token.casefold()
        and row["relation_to_identity"] == "lower"
    ]
    if not matches:
        raise PromptSensitivityError(f"No variance-reducing style token {token!r} found")
    return max(
        matches,
        key=lambda row: float(row["raw_across_wrapper_variance"])
        - float(row["projected_across_wrapper_variance"]),
    )


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#AAB4BF",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot(
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    example: dict[str, Any],
) -> plt.Figure:
    _configure()
    fig = plt.figure(figsize=(13.8, 6.35))
    grid = fig.add_gridspec(1, 2, width_ratios=(0.82, 1.55), wspace=0.27)
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.15, top=0.83)
    fig.suptitle(
        "Does an answer-free prompt paraphrase change the distillation target?",
        x=0.065,
        y=0.965,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.905,
        "Same student prefix and token; only the neutral / terse / verbose privileged prompt changes.",
        ha="left",
        color=MUTED,
        fontsize=9.4,
    )

    axis = fig.add_subplot(grid[0, 0])
    wrapper_labels = ("Neutral", "Terse", "Verbose")
    raw = np.asarray([float(example[f"{name}_raw_shift"]) for name in WRAPPERS])
    projected = np.asarray(
        [float(example[f"{name}_projected_shift"]) for name in WRAPPERS]
    )
    y = np.arange(3)[::-1]
    for index, (raw_value, projected_value) in enumerate(zip(raw, projected)):
        axis.plot(
            [raw_value, projected_value],
            [y[index], y[index]],
            color="#CBD5E1",
            linewidth=2.1,
            zorder=1,
        )
    axis.scatter(raw, y, s=86, color=HIGHER, edgecolor="white", linewidth=0.8, label="Raw target", zorder=3)
    axis.scatter(projected, y, s=86, color=LOWER, edgecolor="white", linewidth=0.8, label="TRSD target", zorder=3)
    axis.axvline(0, color="#64748B", linestyle="--", linewidth=0.9)
    axis.set_yticks(y, wrapper_labels)
    axis.set_xlabel(r"Realized-token shift  $\log q_t(y_t)-\log p_t(y_t)$")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.set_title(
        f"How one point is built: style token “{str(example['token_text']).strip()}”",
        loc="left",
        pad=12,
    )
    axis.text(
        0.02,
        0.92,
        "negative shift = target downweights this token",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=8.2,
    )
    raw_variance = float(example["raw_across_wrapper_variance"])
    projected_variance = float(example["projected_across_wrapper_variance"])
    reduction = 100.0 * (1.0 - projected_variance / raw_variance)
    axis.text(
        0.03,
        0.08,
        (
            f"variance: {raw_variance:.2e}  →  {projected_variance:.2e}\n"
            f"{reduction:.1f}% lower prompt sensitivity\n"
            f"one point at ({raw_variance:.2e}, {projected_variance:.2e}) →"
        ),
        transform=axis.transAxes,
        fontsize=9,
        color=INK,
        bbox={"boxstyle": "round,pad=0.45", "fc": "#F8FAFC", "ec": GRID},
    )
    axis.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.30))
    axis.text(-0.16, 1.08, "a", transform=axis.transAxes, fontsize=14, fontweight="bold")

    axis = fig.add_subplot(grid[0, 1])
    raw_variances = np.asarray([float(row["raw_across_wrapper_variance"]) for row in rows])
    projected_variances = np.asarray(
        [float(row["projected_across_wrapper_variance"]) for row in rows]
    )
    positive = np.concatenate(
        [raw_variances[raw_variances > 0], projected_variances[projected_variances > 0]]
    )
    floor = max(1e-13, float(np.quantile(positive, 0.003) * 0.35))
    maximum = float(max(raw_variances.max(), projected_variances.max()) * 1.35)
    x = np.maximum(raw_variances, floor)
    y_values = np.maximum(projected_variances, floor)
    line = np.geomspace(floor, maximum, 400)
    axis.fill_between(line, floor, line, color="#E7F4F8", alpha=0.85, zorder=0)
    axis.fill_between(line, line, maximum, color="#FFF4E5", alpha=0.82, zorder=0)
    for relation, color, label, size, alpha in (
        ("tied", TIED, "unchanged / numerical tie", 5, 0.13),
        ("lower", LOWER, "lower after TRSD", 6, 0.16),
        ("higher", HIGHER, "higher after TRSD", 7, 0.22),
    ):
        mask = np.asarray([row["relation_to_identity"] == relation for row in rows])
        axis.scatter(
            x[mask],
            y_values[mask],
            s=size,
            color=color,
            alpha=alpha,
            edgecolor="none",
            rasterized=True,
            label=label,
            zorder=2,
        )
    axis.plot(line, line, color=INK, linestyle="--", linewidth=1.15, zorder=3, label=r"no change: $y=x$")
    query_summaries = [row for row in summaries if row["scope"] != "all_queries"]
    query_label_offsets = ((-88, 23), (-91, -20), (10, 6))
    for index, summary in enumerate(query_summaries):
        qx = float(summary["mean_raw_variance"])
        qy = float(summary["mean_projected_variance"])
        axis.scatter(
            [qx],
            [qy],
            marker="D",
            s=88,
            facecolor=QUERY_COLORS[index % len(QUERY_COLORS)],
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
        axis.annotate(
            f"Q{index + 1}: −{float(summary['mean_variance_reduction_pct']):.1f}%",
            (qx, qy),
            xytext=query_label_offsets[index],
            textcoords="offset points",
            fontsize=8.2,
            color=QUERY_COLORS[index % len(QUERY_COLORS)],
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-",
                "color": QUERY_COLORS[index % len(QUERY_COLORS)],
                "lw": 0.7,
                "alpha": 0.85,
            },
        )
    axis.scatter(
        [max(raw_variance, floor)],
        [max(projected_variance, floor)],
        marker="*",
        s=185,
        facecolor="#F2C94C",
        edgecolor=INK,
        linewidth=0.9,
        zorder=6,
        label=f"example in (a): {str(example['token_text']).strip()}",
    )
    aggregate = next(row for row in summaries if row["scope"] == "all_queries")
    axis.text(
        0.035,
        0.955,
        (
            f"mean variance  −{float(aggregate['mean_variance_reduction_pct']):.1f}%\n"
            f"below / tied / above: {float(aggregate['lower_pct']):.1f}% / "
            f"{float(aggregate['tied_pct']):.1f}% / {float(aggregate['higher_pct']):.1f}%"
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=9.2,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.42", "fc": "white", "ec": GRID, "alpha": 0.94},
        zorder=7,
    )
    axis.text(
        0.98,
        0.04,
        "below line = less prompt-sensitive",
        transform=axis.transAxes,
        ha="right",
        color=LOWER,
        fontweight="bold",
        fontsize=8.6,
    )
    axis.text(
        0.98,
        0.865,
        "above line = more prompt-sensitive",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color=HIGHER,
        fontweight="bold",
        fontsize=8.6,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(floor, maximum)
    axis.set_ylim(floor, maximum)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Raw target variance across 3 prompts")
    axis.set_ylabel("TRSD target variance across 3 prompts")
    axis.set_title(
        f"Every token is paired before vs. after projection ({len(rows):,} tokens)",
        loc="left",
        pad=12,
    )
    axis.grid(which="major", color=GRID, linewidth=0.65)
    handles, labels = axis.get_legend_handles_labels()
    order = [1, 2, 0, 3, 4]
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        framealpha=0.93,
        fontsize=7.6,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.085),
    )
    axis.text(-0.12, 1.08, "b", transform=axis.transAxes, fontsize=14, fontweight="bold")

    fig.text(
        0.065,
        0.045,
        (
            "Descriptive same-prefix case study: 3 DeepMath queries × 3 answer-free prompt wrappers; "
            "token points are paired observations, not independent query replicates. Exact zeros are displayed at a visual floor."
        ),
        ha="left",
        color=MUTED,
        fontsize=8.5,
    )
    return fig


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise PromptSensitivityError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def build(inputs: Sequence[Path], output_prefix: Path, example_token: str) -> list[Path]:
    if len(inputs) < 2:
        raise PromptSensitivityError("At least two mechanism queries are required")
    rows = [row for index, path in enumerate(inputs) for row in _token_rows(path, index)]
    if len({row["query_id"] for row in rows}) != len(inputs):
        raise PromptSensitivityError("Mechanism inputs must have distinct query IDs")
    epsilons = {float(row["epsilon"]) for row in rows}
    if len(epsilons) != 1:
        raise PromptSensitivityError(f"All inputs must share one epsilon, found {epsilons}")
    summaries = _summarize(rows)
    example = _example(rows, example_token)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prompt-sensitivity-", dir=output_prefix.parent) as temp:
        stage = Path(temp)
        staged_prefix = stage / output_prefix.name
        _write_csv(staged_prefix.with_suffix(".csv"), rows)
        _write_csv(stage / f"{output_prefix.name}_summary.csv", summaries)
        _write_csv(stage / f"{output_prefix.name}_example.csv", [example])
        figure = _plot(rows, summaries, example)
        for suffix in ("png", "pdf", "svg"):
            path = staged_prefix.with_suffix(f".{suffix}")
            kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.04}
            if suffix == "png":
                kwargs["dpi"] = 300
            if suffix == "pdf":
                kwargs["metadata"] = {"Creator": "TRSD prompt-sensitivity reporter", "CreationDate": None}
            figure.savefig(path, format=suffix, **kwargs)
            if suffix == "svg":
                _normalize_svg(path)
        plt.close(figure)
        outputs: list[Path] = []
        for source in sorted(stage.iterdir()):
            destination = output_prefix.parent / source.name
            os.replace(source, destination)
            outputs.append(destination)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posthoc", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--example-token", default="Therefore")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        outputs = build(args.posthoc, args.output_prefix, args.example_token)
    except (OSError, PromptSensitivityError, subprocess.SubprocessError, ValueError) as exc:
        raise SystemExit(f"Refusing prompt-sensitivity plot: {exc}") from exc
    print(json.dumps({"outputs": [str(path) for path in outputs]}, sort_keys=True))


if __name__ == "__main__":
    main()
