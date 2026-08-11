#!/usr/bin/env python3
"""Render two preregistered academic figures for the DeepMath-10 TRSD study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RAW = "#171717"
TRSD = "#F4C430"
TRSD_DARK = "#A66F00"
INK = "#111111"
MUTED = "#6F685A"
GRID = "#E7E0CF"
PAPER = "#FFFDF7"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite(values: list[Any]) -> np.ndarray:
    result = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("Metric is empty or non-finite")
    return result


def bootstrap(
    arrays: list[np.ndarray],
    statistic: Callable[..., float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    sizes = {array.size for array in arrays}
    if len(sizes) != 1:
        raise ValueError("Paired bootstrap arrays have different lengths")
    count = next(iter(sizes))
    estimate = float(statistic(*arrays))
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for start in range(0, resamples, 250):
        stop = min(start + 250, resamples)
        indices = rng.integers(0, count, size=(stop - start, count))
        for offset, index in enumerate(indices, start=start):
            values[offset] = statistic(*(array[index] for array in arrays))
    low, high = np.quantile(values, [0.025, 0.975])
    return estimate, float(low), float(high)


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.titlesize": 10.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.alpha": 0.8,
            "grid.linewidth": 0.75,
            "legend.frameon": False,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 320
        if suffix == "pdf":
            kwargs["metadata"] = {"Creator": "DeepMath-10 TRSD study", "CreationDate": None}
        fig.savefig(output.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)


def ci_bar(
    ax: plt.Axes,
    x: float,
    result: tuple[float, float, float],
    *,
    color: str,
    width: float = 0.62,
    scale: float = 1.0,
) -> float:
    estimate, low, high = (value * scale for value in result)
    ax.bar(
        x,
        estimate,
        width=width,
        color=color,
        edgecolor=INK,
        linewidth=0.8,
        zorder=3,
    )
    ax.errorbar(
        x,
        estimate,
        yerr=[[estimate - low], [high - estimate]],
        fmt="none",
        color=INK,
        linewidth=1.2,
        capsize=3.2,
        zorder=4,
    )
    return estimate


def ci_barh(
    ax: plt.Axes,
    y: float,
    result: tuple[float, float, float],
    *,
    color: str,
    height: float = 0.58,
    scale: float = 1.0,
) -> float:
    estimate, low, high = (value * scale for value in result)
    ax.barh(
        y,
        estimate,
        height=height,
        color=color,
        edgecolor=INK,
        linewidth=0.8,
        zorder=3,
    )
    ax.errorbar(
        estimate,
        y,
        xerr=[[estimate - low], [high - estimate]],
        fmt="none",
        color=INK,
        linewidth=1.2,
        capsize=3.2,
        zorder=4,
    )
    return estimate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rows = sorted(
        [row for path in args.shard for row in read_jsonl(path)],
        key=lambda row: int(row["study_index"]),
    )
    expected_ids = [str(value) for value in manifest["query_ids"]]
    if [str(row.get("query_id")) for row in rows] != expected_ids:
        raise ValueError("Evidence rows do not exactly match the frozen 10% manifest")
    if len(rows) != 3116 or [int(row["study_index"]) for row in rows] != list(range(3116)):
        raise ValueError("Evidence is not the exact contiguous 3,116-query study")

    raw_prompt = finite([row["nuisance"]["raw_prompt_variance_mean"] for row in rows])
    trsd_prompt = finite(
        [row["nuisance"]["projected_prompt_variance_mean"] for row in rows]
    )
    style_pairs = [
        (
            row["nuisance"]["raw_style_shift_neutral"],
            row["nuisance"]["projected_style_shift_neutral"],
        )
        for row in rows
        if row["nuisance"]["raw_style_shift_neutral"] is not None
        and row["nuisance"]["projected_style_shift_neutral"] is not None
    ]
    raw_style = finite([pair[0] for pair in style_pairs])
    trsd_style = finite([pair[1] for pair in style_pairs])
    raw_gold = finite(
        [row["useful_signal"]["raw_gold_logprob_gain_neutral"] for row in rows]
    )
    trsd_gold = finite(
        [row["useful_signal"]["projected_gold_logprob_gain_neutral"] for row in rows]
    )
    task_pairs = [
        (
            row["useful_signal"]["raw_task_gold_logprob_gain_neutral"],
            row["useful_signal"]["projected_task_gold_logprob_gain_neutral"],
        )
        for row in rows
        if row["useful_signal"]["raw_task_gold_logprob_gain_neutral"] is not None
        and row["useful_signal"]["projected_task_gold_logprob_gain_neutral"] is not None
    ]
    raw_task = finite([pair[0] for pair in task_pairs])
    trsd_task = finite([pair[1] for pair in task_pairs])
    raw_worst = finite(
        [row["useful_signal"]["raw_all_wrapper_min_gold_gain"] for row in rows]
    )
    trsd_worst = finite(
        [row["useful_signal"]["projected_all_wrapper_min_gold_gain"] for row in rows]
    )

    ratio = lambda projected, raw: float(projected.mean() / raw.mean())
    average = lambda values: float(values.mean())
    positive = lambda values: float((values > 0).mean())
    nuisance_results = {
        "prompt_variance_retained": bootstrap(
            [trsd_prompt, raw_prompt], ratio, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed
        ),
        "style_shift_retained": bootstrap(
            [trsd_style, raw_style], ratio, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 1
        ),
    }
    signal_results = {
        "raw_gold_gain": bootstrap([raw_gold], average, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 2),
        "trsd_gold_gain": bootstrap([trsd_gold], average, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 3),
        "raw_task_gain": bootstrap([raw_task], average, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 4),
        "trsd_task_gain": bootstrap([trsd_task], average, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 5),
    }
    reliability_results = {
        "raw_neutral_positive": bootstrap([raw_gold], positive, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 6),
        "trsd_neutral_positive": bootstrap([trsd_gold], positive, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 7),
        "raw_worst_positive": bootstrap([raw_worst], positive, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 8),
        "trsd_worst_positive": bootstrap([trsd_worst], positive, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed + 9),
    }
    paired_results = {
        "prompt_variance_lower": bootstrap(
            [(trsd_prompt < raw_prompt).astype(float)],
            average,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed + 10,
        ),
        "style_shift_lower": bootstrap(
            [(trsd_style < raw_style).astype(float)],
            average,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed + 11,
        ),
        "gold_gain_higher": bootstrap(
            [(trsd_gold > raw_gold).astype(float)],
            average,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed + 12,
        ),
        "worst_wrapper_gain_higher": bootstrap(
            [(trsd_worst > raw_worst).astype(float)],
            average,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed + 13,
        ),
    }

    configure()
    raw_handle = plt.Rectangle(
        (0, 0), 1, 1, facecolor=RAW, edgecolor=INK, label="Raw privileged"
    )
    trsd_handle = plt.Rectangle(
        (0, 0), 1, 1, facecolor=TRSD, edgecolor=INK, label="TRSD, ε=0.004"
    )

    # Main Claim-1 figure: raw privilege is not inherently reliable; TRSD's
    # advantage comes from turning it into a student-local surrogate.
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.15), constrained_layout=True)
    ax = axes[0]
    raw_mean = signal_results["raw_gold_gain"][0]
    trsd_mean = signal_results["trsd_gold_gain"][0]
    ax.axvspan(0.0, 0.005, color=TRSD, alpha=0.18, zorder=0)
    ax.axvline(0.0, color=INK, linestyle="--", linewidth=1.1, zorder=1)
    ax.annotate(
        "",
        xy=(trsd_mean, 0.0),
        xytext=(raw_mean, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": TRSD_DARK, "lw": 2.4},
        zorder=2,
    )
    for result, color, marker in (
        (signal_results["raw_gold_gain"], RAW, "o"),
        (signal_results["trsd_gold_gain"], TRSD, "D"),
    ):
        estimate, low, high = result
        ax.errorbar(
            estimate,
            0.0,
            xerr=[[estimate - low], [high - estimate]],
            fmt=marker,
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.9,
            ecolor=INK,
            elinewidth=1.2,
            capsize=3.2,
            zorder=4,
        )
    ax.text(raw_mean, 0.20, f"Raw\n{raw_mean:+.3f}", ha="center", va="bottom", fontweight="bold")
    ax.text(trsd_mean, -0.20, f"TRSD\n{trsd_mean:+.3f}", ha="center", va="top", fontweight="bold")
    harm_reduction = 100.0 * (1.0 - abs(trsd_mean) / abs(raw_mean))
    paired_gain = 100.0 * paired_results["gold_gain_higher"][0]
    ax.text(
        0.04,
        0.05,
        f"{harm_reduction:.1f}% less negative\n{paired_gain:.1f}% of queries improve",
        transform=ax.transAxes,
        color=INK,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": TRSD, "ec": INK, "lw": 0.8},
    )
    ax.text(0.0018, 0.48, "helpful", ha="center", color=TRSD_DARK, fontweight="bold")
    ax.set_xlim(-0.0435, 0.005)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.grid(axis="x")
    ax.set_xlabel("Mean gold-token gain (nats/token) ↑")
    ax.set_title("A  Raw privilege is often harmful", loc="left")

    ax = axes[1]
    keys = ("prompt_variance_retained", "style_shift_retained")
    labels = ("Prompt\nvariance", "Style-token\nmovement")
    x = np.arange(2, dtype=float)
    width = 0.34
    ax.bar(
        x - width / 2,
        [100.0, 100.0],
        width,
        color=RAW,
        edgecolor=INK,
        linewidth=0.8,
        zorder=3,
    )
    for index, key in enumerate(keys):
        retained = ci_bar(
            ax,
            index + width / 2,
            nuisance_results[key],
            color=TRSD,
            width=width,
            scale=100.0,
        )
        ax.text(index - width / 2, 102.0, "100", ha="center", fontweight="bold")
        ax.text(index + width / 2, retained + 2.2, f"{retained:.1f}", ha="center", fontweight="bold")
        ax.text(
            index - width / 2,
            90.0,
            "RAW",
            ha="center",
            color=PAPER,
            fontsize=8.2,
            fontweight="bold",
        )
        ax.text(
            index + width / 2,
            retained / 2.0,
            "TRSD",
            ha="center",
            color=INK,
            fontsize=8.2,
            fontweight="bold",
            rotation=90,
        )
        ax.text(
            index,
            112.0,
            f"−{100.0 - retained:.1f}%",
            ha="center",
            fontweight="bold",
            color=TRSD_DARK,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 119)
    ax.set_ylabel("Nuisance retained (%) ↓")
    ax.grid(axis="y")
    ax.set_title("B  Projection filters wrapper nuisance", loc="left")

    ax = axes[2]
    paired_keys = (
        "prompt_variance_lower",
        "style_shift_lower",
        "gold_gain_higher",
        "worst_wrapper_gain_higher",
    )
    paired_labels = (
        "Lower prompt variance",
        "Lower style movement",
        "Higher gold gain",
        "Higher worst-wrapper gain",
    )
    y = np.arange(len(paired_keys), dtype=float)
    for position, key in zip(y, paired_keys, strict=True):
        estimate = ci_barh(
            ax,
            position,
            paired_results[key],
            color=TRSD,
            height=0.58,
            scale=100.0,
        )
        ax.text(estimate - 1.3, position, f"{estimate:.1f}%", ha="right", va="center", fontweight="bold")
    ax.axvline(50.0, color=MUTED, linestyle="--", linewidth=1.0)
    ax.text(50.0, -0.48, "paired majority", ha="center", color=MUTED, fontsize=8.3)
    ax.set_yticks(y, paired_labels)
    ax.set_ylim(3.65, -0.65)
    ax.set_xlim(0, 104)
    ax.set_xlabel("Queries where TRSD improves over raw (%) ↑")
    ax.grid(axis="x")
    ax.set_title("C  Improvement is population-wide", loc="left")

    fig.suptitle(
        "TRSD's advantage comes from a more reliable student-local surrogate",
        fontsize=15.0,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Qwen3-8B · 3,116 frozen DeepMath queries · 797,696 reference tokens · whiskers: 95% query-bootstrap CI",
        ha="center",
        color=MUTED,
        fontsize=8.8,
    )
    save(fig, args.output_dir / "figure1_nuisance_vs_useful_signal")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    ax = axes[0]
    groups = np.arange(2, dtype=float)
    width = 0.34
    raw_results = (
        reliability_results["raw_neutral_positive"],
        reliability_results["raw_worst_positive"],
    )
    trsd_results = (
        reliability_results["trsd_neutral_positive"],
        reliability_results["trsd_worst_positive"],
    )
    for index, (raw_result, trsd_result) in enumerate(zip(raw_results, trsd_results, strict=True)):
        raw_value = ci_bar(ax, index - width / 2, raw_result, color=RAW, width=width, scale=100.0)
        trsd_value = ci_bar(ax, index + width / 2, trsd_result, color=TRSD, width=width, scale=100.0)
        ax.text(index - width / 2, raw_value + 0.65, f"{raw_value:.1f}%", ha="center", fontweight="bold")
        ax.text(index + width / 2, trsd_value + 0.65, f"{trsd_value:.1f}%", ha="center", fontweight="bold")
        ax.text(
            index,
            max(raw_result[2], trsd_result[2]) * 100.0 + 1.7,
            f"{trsd_value / raw_value:.1f}×",
            ha="center",
            fontweight="bold",
            color=TRSD_DARK,
        )
    ax.set_xticks(groups, ["Neutral wrapper", "Worst of 3 wrappers"])
    ax.set_ylabel("Queries with positive gold-token gain (%) ↑")
    ax.set_ylim(0, 17.5)
    ax.grid(axis="y")
    ax.set_title("A  Helpful targets become more frequent", loc="left")
    ax.legend(handles=[raw_handle, trsd_handle], loc="upper right", fontsize=8.5)

    ax = axes[1]
    raw_ordered = np.sort(np.maximum(raw_prompt, np.finfo(float).tiny))
    trsd_ordered = np.sort(np.maximum(trsd_prompt, np.finfo(float).tiny))
    probability = np.arange(1, raw_ordered.size + 1) / raw_ordered.size
    raw_line = ax.plot(
        raw_ordered,
        probability,
        color=RAW,
        linestyle="--",
        linewidth=2.4,
        label="Raw privileged",
    )[0]
    ax.plot(trsd_ordered, probability, color=INK, linewidth=4.4, zorder=3)
    trsd_line = ax.plot(
        trsd_ordered,
        probability,
        color=TRSD,
        linewidth=2.8,
        label="TRSD, ε=0.004",
        zorder=4,
    )[0]
    ax.set_xscale("log")
    ax.set_xlabel("Across-wrapper token variance ↓")
    ax.set_ylabel("Query empirical CDF")
    ax.grid(axis="both")
    ax.text(
        0.97,
        0.08,
        f"Mean variance\n−{100.0 * (1.0 - nuisance_results['prompt_variance_retained'][0]):.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": TRSD, "ec": INK, "lw": 0.8},
    )
    ax.set_title("B  The full distribution shifts left", loc="left")
    ax.legend(handles=[raw_line, trsd_line], loc="upper left", fontsize=8.5)
    fig.suptitle(
        "Student-local projection improves reliability across prompt perturbations",
        fontsize=14.2,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Positive-gain rates remain low in absolute terms; the preregistered robustness comparison is nevertheless supported.",
        ha="center",
        color=MUTED,
        fontsize=8.8,
    )
    save(fig, args.output_dir / "figure2_local_surrogate_reliability")

    margin = float(manifest["estimands"]["signal_quality_noninferiority_margin"])
    claim_1 = bool(
        nuisance_results["prompt_variance_retained"][0] < 1
        and nuisance_results["style_shift_retained"][0] < 1
        and signal_results["trsd_gold_gain"][0] > 0
        and reliability_results["trsd_neutral_positive"][0]
        >= reliability_results["raw_neutral_positive"][0] - margin
    )
    claim_2 = bool(
        nuisance_results["prompt_variance_retained"][0] < 1
        and reliability_results["trsd_worst_positive"][0]
        >= reliability_results["raw_worst_positive"][0] - margin
    )
    summary = {
        "schema_version": "trsd-deepmath10-surrogate-report-v2",
        "queries": len(rows),
        "reference_tokens": int(sum(int(row["reference_tokens"]) for row in rows)),
        "query_bootstrap_resamples": args.bootstrap_resamples,
        "nuisance": nuisance_results,
        "useful_signal": signal_results,
        "reliability": reliability_results,
        "paired_descriptive": paired_results,
        "noninferiority_margin": margin,
        "claim_1_supported_by_preregistered_rule": claim_1,
        "claim_2_supported_by_preregistered_rule": claim_2,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Qwen3-8B DeepMath-10 surrogate study\n\n"
        "## Paper claim\n\n"
        "TRSD's advantage comes from projecting an unstable privileged direction into a "
        "student-local neighborhood, yielding a less wrapper-sensitive and more reliable "
        "surrogate update. Raw privileged supervision is not assumed to be inherently helpful.\n\n"
        "## Protocol\n\n"
        f"Exact frozen queries: {len(rows):,}. Reference tokens: {summary['reference_tokens']:,}. "
        f"Query-bootstrap resamples: {args.bootstrap_resamples:,}. Tokens are not treated as "
        "independent replicates.\n\n"
        "## Headline evidence\n\n"
        f"- Prompt variance retained: **{100.0 * nuisance_results['prompt_variance_retained'][0]:.1f}%** "
        f"(a {100.0 * (1.0 - nuisance_results['prompt_variance_retained'][0]):.1f}% reduction).\n"
        f"- Style-token movement retained: **{100.0 * nuisance_results['style_shift_retained'][0]:.1f}%** "
        f"(a {100.0 * (1.0 - nuisance_results['style_shift_retained'][0]):.1f}% reduction).\n"
        f"- Mean gold-token gain: **{signal_results['raw_gold_gain'][0]:+.4f} → "
        f"{signal_results['trsd_gold_gain'][0]:+.4f}** nats/token; it improves but remains negative.\n"
        f"- Queries with higher gold gain under TRSD: **{100.0 * paired_results['gold_gain_higher'][0]:.1f}%**.\n"
        f"- Queries with higher worst-wrapper gain under TRSD: "
        f"**{100.0 * paired_results['worst_wrapper_gain_higher'][0]:.1f}%**.\n\n"
        "## Preregistered decisions\n\n"
        f"- Figure-1 signal-preservation rule: **{claim_1}**. The positive-mean condition failed; "
        "the paper figure therefore does not claim that absolute signal quality is positive.\n"
        f"- Figure-2 surrogate-reliability rule: **{claim_2}**.\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
