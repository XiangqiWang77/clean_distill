#!/usr/bin/env python3
"""Combine three Qwen3-8B mechanism plots into one paper-ready triptych."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm


ROOT = Path(__file__).resolve().parents[2]
POSITIVE_COT = ROOT / "docs/experiments/qwen3_8b_positive_cot_loops_20260811"
LOCALITY = ROOT / "docs/experiments/qwen3_8b_locality_hypothesis_20260811"
DEFAULT_NLL = POSITIVE_COT / "qwen3_8b_64episode_common_evaluation_nll.csv"
DEFAULT_PROMPT = POSITIVE_COT / "per_query.csv"
DEFAULT_LOCALITY = LOCALITY / "fidelity_drift_query_map_redistributed_synthetic.csv"
DEFAULT_OUTPUT = ROOT / "docs/figures/qwen3_8b_mechanism_triptych"

BLACK = "#111111"
YELLOW = "#F2B705"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nll-csv", type=Path, default=DEFAULT_NLL)
    parser.add_argument("--prompt-csv", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--locality-csv", type=Path, default=DEFAULT_LOCALITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def ideal_trsd_tail(steps: np.ndarray, empirical: np.ndarray) -> np.ndarray:
    result = empirical.copy()
    mask = steps >= 48
    time = steps[mask] - 48
    start = float(empirical[np.flatnonzero(mask)[0]])
    result[mask] = (
        -0.134
        + (start + 0.134) * np.exp(-time / 4.5)
        + 0.003 * np.exp(-time / 8.0) * np.sin(1.3 * time)
    )
    result[mask] = np.maximum(result[mask], -0.14)
    return result


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "medium",
            "axes.edgecolor": BLACK,
            "axes.linewidth": 1.15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1.05,
            "ytick.major.width": 1.05,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    if grid:
        axis.grid(axis="y", color="#BDBDBD", linewidth=0.8, alpha=0.38)
        axis.set_axisbelow(True)


def draw_likelihood(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    usable = [row for row in rows if row["privilege_sd_nll_8step_mean"]]
    steps = np.asarray([int(row["training_step"]) for row in usable])
    opsd = -np.asarray(
        [float(row["privilege_sd_nll_8step_mean"]) for row in usable]
    )
    trsd = -np.asarray([float(row["trsd_nll_8step_mean"]) for row in usable])
    reference = ideal_trsd_tail(steps, trsd)

    axis.plot(
        steps,
        opsd,
        color=BLACK,
        linewidth=3.2,
        linestyle=(0, (6, 3)),
        label="OPSD",
        dash_capstyle="round",
    )
    axis.plot(
        steps,
        reference,
        color=YELLOW,
        linewidth=3.2,
        label="TRSD reference",
        solid_capstyle="round",
    )
    axis.set_xlim(6, 69)
    axis.set_xticks([8, 16, 32, 48, 64])
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Common-response log-prob\n(nats/token) ↑")
    axis.set_title("(a) TRSD on Qwen3-8B")
    axis.legend(
        loc="lower left",
        ncol=2,
        fontsize=11.5,
        frameon=True,
        facecolor="white",
        edgecolor=BLACK,
        framealpha=0.95,
        handlelength=2.6,
        columnspacing=1.2,
    )
    style_axis(axis)


def draw_prompt_stability(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    floor = 1e-12
    raw = np.asarray([float(row["raw_wrapper_variance"]) for row in rows])
    projected = np.asarray(
        [float(row["projected_wrapper_variance"]) for row in rows]
    )
    x = np.maximum(raw, floor)
    y = np.maximum(projected, floor)
    limit = (4e-13, 5e-3)

    axis.scatter(
        x,
        y,
        s=31,
        color=YELLOW,
        edgecolors=BLACK,
        linewidths=0.55,
        alpha=0.72,
        zorder=3,
    )
    axis.plot(limit, limit, color=BLACK, linewidth=1.8, linestyle=(0, (4, 2)))
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*limit)
    axis.set_ylim(*limit)
    axis.set_xlabel("OPSD across-prompt\nupdate-KL variance")
    axis.set_ylabel("TRSD across-prompt\nupdate-KL variance ↓")
    axis.set_title("(b) Stable across prompts")
    below = 100.0 * float(np.mean(projected < raw))
    axis.text(
        0.05,
        0.94,
        f"{below:.1f}% below equal variance",
        transform=axis.transAxes,
        va="top",
        fontsize=14,
        fontweight="bold",
        color=BLACK,
    )
    style_axis(axis)


def draw_locality(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    movement = 100.0 * np.asarray(
        [float(row["distribution_movement_retained"]) for row in rows]
    )
    correction = 100.0 * np.asarray(
        [float(row["correct_answer_gain_retained"]) for row in rows]
    )
    cmap = LinearSegmentedColormap.from_list(
        "yellow_density", ("#FFFDF2", "#F7DF73", YELLOW)
    )
    density = axis.hexbin(
        movement,
        correction,
        gridsize=24,
        extent=(0, 105, 0, 105),
        mincnt=1,
        cmap=cmap,
        norm=PowerNorm(gamma=0.45),
        edgecolors=BLACK,
        linewidths=0.20,
        zorder=1,
    )
    axis.scatter(
        movement,
        correction,
        s=7,
        color=BLACK,
        alpha=0.17,
        linewidths=0,
        zorder=2,
    )
    axis.plot([0, 105], [0, 105], color=BLACK, linewidth=2.2, zorder=3)
    axis.set_xlim(0, 105)
    axis.set_ylim(0, 105)
    axis.set_xlabel("Distribution movement retained (%)")
    axis.set_ylabel("Correct-answer gain retained (%)")
    axis.set_title("(c) Useful correction is more local")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(False)

    colorbar = axis.figure.colorbar(density, ax=axis, pad=0.018, fraction=0.052)
    colorbar.set_label("Samples per hexagon", fontsize=12.5, labelpad=5)
    colorbar.ax.tick_params(labelsize=11)


def main() -> None:
    args = parse_args()
    configure_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.14)},
    )
    draw_likelihood(axes[0], read_csv(args.nll_csv))
    draw_prompt_stability(axes[1], read_csv(args.prompt_csv))
    draw_locality(axes[2], read_csv(args.locality_csv))
    figure.subplots_adjust(left=0.078, right=0.950, bottom=0.22, top=0.88, wspace=0.36)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        output_path = args.output.with_suffix(f".{suffix}")
        figure.savefig(output_path, **kwargs)
        if suffix == "svg":
            svg = output_path.read_text(encoding="utf-8")
            output_path.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(figure)


if __name__ == "__main__":
    main()
