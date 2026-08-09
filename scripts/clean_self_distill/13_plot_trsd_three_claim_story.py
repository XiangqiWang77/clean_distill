#!/usr/bin/env python3
"""Render the three-claim TRSD paper story from the released evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE = "#737B87"
PRIV = "#3B6FB6"
TRSD = "#D5533D"
TEAL = "#168C8C"
GOLD = "#E6A23C"
INK = "#1F2933"
MUTED = "#65717E"
GRID = "#DDE3E8"
BG = "#F7F9FB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--episode-csv", type=Path, required=True)
    parser.add_argument("--stage", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    return next(row for row in rows if row[key] == value)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
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
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def normalize_text_file(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def save_all(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=260)
    fig.savefig(output_dir / f"{stem}.pdf")
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(svg_path)
    normalize_text_file(svg_path)
    plt.close(fig)


def load_bundle(bundle: Path, episode_csv: Path) -> dict[str, Any]:
    tables = bundle / "tables"
    accuracy = read_csv(tables / "main_accuracy.csv")
    style = read_csv(tables / "trust_region_target_summary.csv")
    paired = read_csv(tables / "trsd64_vs_privileged64.csv")
    efficiency = read_csv(tables / "evaluation_efficiency.csv")
    completion = read_csv(tables / "completion_behavior_diagnostics.csv")
    same_prefix = read_csv(tables / "same_prefix_pilot.csv")
    epsilon = read_csv(tables / "epsilon_sensitivity.csv")
    episodes = read_csv(episode_csv)
    trsd_episodes = [row for row in episodes if row["method"] == "trsd"]
    if len(trsd_episodes) != 64:
        raise ValueError(f"Expected 64 TRSD episode rows, found {len(trsd_episodes)}")
    return {
        "accuracy": accuracy,
        "style": style,
        "paired": paired,
        "efficiency": efficiency,
        "completion": completion,
        "same_prefix": same_prefix,
        "epsilon": epsilon,
        "episodes": trsd_episodes,
    }


def combined_accuracy(data: dict[str, Any], method: str) -> float:
    rows = data["accuracy"]
    row = next(row for row in rows if row["method"] == method and row["dataset"] == "combined")
    return f(row, "strict_acc1_percent")


def figure_three_claims(data: dict[str, Any], output_dir: Path) -> None:
    raw = by(data["style"], "target", "raw_privileged")
    projected = by(data["style"], "target", "trsd_projected")
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), gridspec_kw={"wspace": 0.32})
    fig.suptitle(
        "TRSD controls drift early and converts it into long-horizon performance",
        fontsize=17,
        fontweight="bold",
        x=0.5,
        y=1.02,
    )

    ax = axes[0]
    panel_label(ax, "A")
    metrics = ["Target KL", "Style drift", "Task movement"]
    raw_values = np.array(
        [f(raw, "target_student_kl"), f(raw, "style_error_per_token"), f(raw, "task_error_per_token")]
    )
    projected_values = np.array(
        [
            f(projected, "target_student_kl"),
            f(projected, "style_error_per_token"),
            f(projected, "task_error_per_token"),
        ]
    )
    retained = 100.0 * projected_values / raw_values
    x = np.arange(3)
    ax.bar(x - 0.19, [100, 100, 100], 0.38, color=PRIV, label="Privileged target")
    ax.bar(x + 0.19, retained, 0.38, color=TRSD, label="TRSD target")
    for index, value in enumerate(retained):
        ax.text(index + 0.19, value + 3, f"−{100-value:.1f}%", ha="center", fontweight="bold", color=TRSD)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Movement retained (%)")
    ax.set_ylim(0, 118)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Claim 1 · Drift control", loc="left")
    ax.text(
        0.02,
        0.96,
        "ε = 0.004 active on 63/64 episodes",
        transform=ax.transAxes,
        va="top",
        color=TEAL,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.02, 0.86), fontsize=9)

    ax = axes[1]
    panel_label(ax, "B")
    names = ["Base", "Privilege-SD\n16", "TRSD\n16"]
    values = [
        combined_accuracy(data, "base"),
        combined_accuracy(data, "privileged_16"),
        combined_accuracy(data, "trsd_16"),
    ]
    bars = ax.bar(names, values, color=[BASE, PRIV, TRSD], width=0.66)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.1, f"{value:.2f}%", ha="center", fontweight="bold")
    ax.axhline(values[0], color=BASE, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.annotate(
        "Preserves Base accuracy\n16/16 updates",
        xy=(2, values[2]),
        xytext=(1.25, 67),
        arrowprops={"arrowstyle": "->", "color": TRSD, "lw": 1.5},
        color=TRSD,
        fontweight="bold",
        ha="center",
    )
    ax.set_ylim(0, 76)
    ax.set_ylabel("Strict Acc@1 (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Claim 2 · Short-term performance", loc="left")

    ax = axes[2]
    panel_label(ax, "C")
    names = ["Base", "Privilege-SD\n64", "TRSD\n64"]
    values = [
        combined_accuracy(data, "base"),
        combined_accuracy(data, "privileged_64"),
        combined_accuracy(data, "trsd_64"),
    ]
    bars = ax.bar(names, values, color=[BASE, PRIV, TRSD], width=0.66)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.1, f"{value:.2f}%", ha="center", fontweight="bold")
    ax.plot([1, 2], [80.0, 80.0], color=INK, linewidth=1.2)
    ax.text(1.5, 81.0, "+8.39 pp", ha="center", fontweight="bold", color=TRSD)
    ax.text(
        0.04,
        0.14,
        "16 W→C vs 4 C→W\n11 completion rescues",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=TEAL,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": GRID, "alpha": 0.92},
    )
    ax.set_ylim(0, 86)
    ax.set_ylabel("Strict Acc@1 (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Claim 3 · Long-term performance", loc="left")

    fig.text(
        0.5,
        -0.035,
        "Qwen3-8B · DeepMath self-distillation · 143 held-out AMC23/AIME24/AIME25 problems · equal episode horizons",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    save_all(fig, output_dir, "fig1_three_claim_story")


def draw_method_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.62, 0.22, 0.22, "Student anchor\n$\\pi_\\theta$", BASE),
        (0.02, 0.16, 0.22, 0.22, "Privileged teacher\n$\\tau$", PRIV),
        (0.38, 0.39, 0.26, 0.28, "Adaptive projection\n$q_\\alpha \\propto \\pi^{1-\\alpha}\\tau^\\alpha$", TEAL),
        (0.77, 0.39, 0.21, 0.28, "Student update\n$\\mathrm{KL}(q_\\alpha\\|\\pi) \\leq \\epsilon$", TRSD),
    ]
    for x, y, w, h, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=color + "18",
            edgecolor=color,
            linewidth=1.8,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontweight="bold")
    for start, end in [((0.24, 0.73), (0.38, 0.57)), ((0.24, 0.27), (0.38, 0.47)), ((0.64, 0.53), (0.77, 0.53))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=INK, lw=1.3))
    ax.text(0.51, 0.25, "α chosen per trajectory", color=TEAL, ha="center", fontweight="bold")
    ax.text(0.51, 0.16, "ε = 0.004", color=MUTED, ha="center")


def figure_drift_mechanism(data: dict[str, Any], output_dir: Path) -> None:
    rows = data["episodes"]
    episode = np.array([int(row["episode"]) for row in rows])
    raw_kl = np.array([f(row, "mean_teacher_student_kl") for row in rows])
    achieved_kl = np.array([f(row, "achieved_kl") for row in rows])
    alpha = np.array([f(row, "alpha") for row in rows])
    same_raw = by(data["same_prefix"], "projection", "raw_privileged_surrogate")
    same_trsd = by(data["same_prefix"], "projection", "trsd_projected")

    fig = plt.figure(figsize=(15.8, 9.3))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)
    fig.suptitle(
        "Claim 1 · The trust region actively controls privileged-teacher drift",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    ax = fig.add_subplot(grid[0, 0])
    panel_label(ax, "A")
    draw_method_schematic(ax)
    ax.set_title("From privileged direction to student-centered target", loc="left")

    ax = fig.add_subplot(grid[0, 1])
    panel_label(ax, "B")
    cmap = LinearSegmentedColormap.from_list("alpha", ["#BBDCDC", TEAL, "#075B5B"])
    scatter = ax.scatter(raw_kl, achieved_kl, c=alpha, cmap=cmap, s=56, edgecolor="white", linewidth=0.6)
    ax.axhline(0.004, color=TRSD, linestyle="--", linewidth=1.5, label="KL budget ε=0.004")
    low = min(raw_kl.min(), achieved_kl.min()) * 0.94
    high = max(raw_kl.max(), achieved_kl.max()) * 1.04
    ax.plot([low, high], [low, high], color=BASE, linestyle=":", linewidth=1.2, label="Unprojected identity")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Raw privileged direction KL")
    ax.set_ylabel("Projected target KL")
    ax.set_title("63/64 trajectories activate the constraint", loc="left")
    ax.grid(color=GRID, linewidth=0.7)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label("Projection α")

    ax = fig.add_subplot(grid[1, 0])
    panel_label(ax, "C")
    ax.plot(episode, achieved_kl, color=TRSD, linewidth=1.6, label="Achieved KL")
    ax.axhline(0.004, color=TRSD, linestyle="--", linewidth=1.1, alpha=0.75)
    ax.fill_between(episode, achieved_kl, 0.004, color=TRSD, alpha=0.08)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Projected target KL", color=TRSD)
    ax.tick_params(axis="y", labelcolor=TRSD)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax2 = ax.twinx()
    ax2.plot(episode, alpha, color=TEAL, linewidth=1.2, alpha=0.9, label="α")
    ax2.set_ylabel("Adaptive projection α", color=TEAL)
    ax2.tick_params(axis="y", labelcolor=TEAL)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, 1.05)
    ax.set_title("Projection adapts throughout all 64 episodes", loc="left")
    ax.text(
        0.02,
        0.94,
        f"mean α = {alpha.mean():.3f}\nmean KL = {achieved_kl.mean():.4f}",
        transform=ax.transAxes,
        va="top",
        color=INK,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID},
    )

    ax = fig.add_subplot(grid[1, 1])
    panel_label(ax, "D")
    style_retention = f(same_trsd, "style_abs_logprob_shift") / f(same_raw, "style_abs_logprob_shift")
    task_gain_ratio = f(same_trsd, "task_logprob_gain") / f(same_raw, "task_logprob_gain")
    x = np.arange(2)
    values = [100 * style_retention, 100 * task_gain_ratio]
    bars = ax.bar(x, values, color=[TEAL, GOLD], width=0.58)
    ax.axhline(100, color=BASE, linestyle="--", linewidth=1.2, label="Raw privileged = 100%")
    labels = [f"{100*style_retention:.1f}% retained", f"{task_gain_ratio:.2f}×"]
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 14, label, ha="center", fontweight="bold")
    ax.set_xticks(x, ["Style shift ↓", "Signed task gain ↑"])
    ax.set_ylabel("Relative to raw privileged target (%)")
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Controlled-prefix check reproduces the mechanism", loc="left")
    ax.text(0.98, 0.05, "3 queries × 3 wrappers", transform=ax.transAxes, ha="right", color=MUTED)

    save_all(fig, output_dir, "fig2_drift_mechanism")


def figure_performance_anatomy(data: dict[str, Any], output_dir: Path) -> None:
    fig = plt.figure(figsize=(15.8, 9.2))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)
    fig.suptitle(
        "Claims 2–3 · Stable at 16 episodes, decisive separation at 64",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    ax = fig.add_subplot(grid[0, 0])
    panel_label(ax, "A")
    horizons = [16, 64]
    privilege = [combined_accuracy(data, "privileged_16"), combined_accuracy(data, "privileged_64")]
    trsd = [combined_accuracy(data, "trsd_16"), combined_accuracy(data, "trsd_64")]
    base = combined_accuracy(data, "base")
    ax.plot(horizons, privilege, "o-", color=PRIV, lw=2.4, ms=8, label="Privilege-SD")
    ax.plot(horizons, trsd, "o-", color=TRSD, lw=2.8, ms=8, label="TRSD")
    ax.axhline(base, color=BASE, linestyle="--", lw=1.4, label=f"Base {base:.2f}%")
    for x, y in zip(horizons, privilege):
        ax.text(x, y + 1.1, f"{y:.2f}%", color=PRIV, ha="center", fontweight="bold")
    for x, y in zip(horizons, trsd):
        ax.text(x, y - 2.8, f"{y:.2f}%", color=TRSD, ha="center", fontweight="bold")
    ax.annotate(
        "+8.39 pp",
        xy=(64, trsd[-1]),
        xytext=(51, 76),
        arrowprops={"arrowstyle": "->", "color": TRSD, "lw": 1.4},
        color=TRSD,
        fontweight="bold",
    )
    ax.set_xticks(horizons)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Strict Acc@1 (%)")
    ax.set_ylim(47, 80)
    ax.grid(color=GRID, linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("The long-horizon gain emerges after early stability", loc="left")

    ax = fig.add_subplot(grid[0, 1])
    panel_label(ax, "B")
    order = ["amc23", "aime24", "aime25", "combined"]
    labels = ["AMC23", "AIME24", "AIME25", "Combined"]
    paired = {row["dataset"]: row for row in data["paired"]}
    delta = np.array([f(paired[key], "delta_percentage_points") for key in order])
    lo = np.array([100 * f(paired[key], "delta_ci_low") for key in order])
    hi = np.array([100 * f(paired[key], "delta_ci_high") for key in order])
    y = np.arange(len(order))
    ax.errorbar(
        delta,
        y,
        xerr=np.vstack([delta - lo, hi - delta]),
        fmt="o",
        color=TRSD,
        ecolor="#E59A8D",
        elinewidth=2.0,
        capsize=4,
        markersize=8,
    )
    ax.axvline(0, color=BASE, linewidth=1.1)
    for xv, yv in zip(delta, y):
        ax.text(xv + 0.55, yv, f"{xv:+.2f}", va="center", color=TRSD, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("TRSD-64 − Privilege-SD64 (percentage points)")
    ax.set_xlim(min(-8, lo.min() - 2), max(18, hi.max() + 2))
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_title("The 64-episode gain spans all three benchmarks", loc="left")

    ax = fig.add_subplot(grid[1, 0])
    panel_label(ax, "C")
    matrix = np.array([[37, 16], [4, 86]])
    image = ax.imshow(matrix, cmap=LinearSegmentedColormap.from_list("paired", ["#F4F6F8", "#F6C9BF", TRSD]))
    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            color = "white" if value > 60 else INK
            tag = "W→C" if (row, col) == (0, 1) else "C→W" if (row, col) == (1, 0) else ""
            ax.text(col, row, f"{value}\n{tag}", ha="center", va="center", fontsize=16, fontweight="bold", color=color)
    ax.set_xticks([0, 1], ["TRSD wrong", "TRSD correct"])
    ax.set_yticks([0, 1], ["P64 wrong", "P64 correct"])
    ax.set_title("Paired outcomes: 16 gains against 4 losses", loc="left")
    ax.text(0.5, 2.02, "Net +12 solved queries · McNemar p=0.0118", ha="center", transform=ax.transData, color=TEAL, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Queries")

    ax = fig.add_subplot(grid[1, 1])
    panel_label(ax, "D")
    methods = ["base", "privileged_16", "privileged_64", "trsd_64"]
    labels = {"base": "Base = T16", "privileged_16": "P16", "privileged_64": "P64", "trsd_64": "T64"}
    colors = {"base": BASE, "privileged_16": PRIV, "trsd_16": "#E58D7E", "privileged_64": PRIV, "trsd_64": TRSD}
    completion = {row["method"]: row for row in data["completion"]}
    for method in methods:
        x = 100 * f(completion[method], "budget_cap_hit_rate")
        yv = combined_accuracy(data, method)
        size = 55 + f(completion[method], "mean_generated_tokens") / 35
        ax.scatter(x, yv, s=size, color=colors[method], alpha=0.9, edgecolor="white", linewidth=0.8)
        offset = (0.55, 0.55) if method != "trsd_64" else (-3.8, 1.0)
        ax.text(x + offset[0], yv + offset[1], labels[method], color=colors[method], fontweight="bold")
    ax.annotate(
        "Best accuracy + fewest cap hits",
        xy=(100 * f(completion["trsd_64"], "budget_cap_hit_rate"), combined_accuracy(data, "trsd_64")),
        xytext=(28, 75.2),
        arrowprops={"arrowstyle": "->", "color": TRSD, "lw": 1.4},
        color=TRSD,
        fontweight="bold",
    )
    ax.set_xlabel("Evaluation budget-cap hit rate (%)  ← better")
    ax.set_ylabel("Strict Acc@1 (%)  ↑ better")
    ax.set_xlim(13, 50)
    ax.set_ylim(49, 78)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_title("Completion control drives the accuracy frontier", loc="left")
    ax.text(0.98, 0.04, "Bubble area ∝ tokens/query", transform=ax.transAxes, ha="right", color=MUTED)

    save_all(fig, output_dir, "fig3_performance_anatomy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_figure_guide(root: Path) -> None:
    guide = """# TRSD figure story

The paper is organized around three claims: **drift control**, **short-term performance**, and **long-term performance**. The visual sequence follows the same causal narrative: constrain the privileged update, preserve the early student, then accumulate a stronger long-horizon policy.

## Figure 1 — Three-claim overview

![Three-claim overview](fig1_three_claim_story.png)

Use this as the main teaser or first experiment figure. Panel A shows that projection retains 26.36% of raw target KL and 60.12% of measured style movement, with the constraint active on 63/64 episodes. Panel B shows short-term stability: TRSD-16 matches Base at 53.85% after completing 16/16 updates. Panel C delivers the endpoint: at the equal 64-episode horizon, TRSD-64 reaches 71.33%, leading Privilege-SD64 by 8.39 points and Base by 17.48 points.

**Caption.** *TRSD controls privileged-teacher drift early and converts the controlled updates into long-horizon performance. At 16 episodes, TRSD preserves the Qwen3-8B base accuracy. At the matched 64-episode horizon, TRSD reaches 71.33% strict Acc@1, 8.39 points above Privilege-SD64 and 17.48 points above Base.*

## Figure 2 — Drift mechanism

![Drift mechanism](fig2_drift_mechanism.png)

Use this in the method analysis. Panel A ties the exponential projection directly to the student-centered KL budget. Panels B–C show that the trust region is operational and adaptive on the complete 64-episode trajectory. Panel D reproduces the mechanism with identical prefixes: style shift contracts to 54.0% of the raw target while signed task-token gain rises 4.85×.

**Caption.** *The trajectory-level trust region actively projects the privileged direction. The KL constraint activates on 63/64 episodes, mean projection strength is α=0.560, and the projected target remains near ε=0.004 throughout training. A controlled-prefix test reproduces lower style shift together with larger signed task-token gain.*

## Figure 3 — Performance anatomy

![Performance anatomy](fig3_performance_anatomy.png)

Use this as the main result analysis. Panel A connects the 16- and 64-episode checkpoints. Panel B shows positive T64−P64 gains on AMC23, AIME24, and AIME25. Panel C exposes the paired transition matrix: 16 wrong-to-correct moves against 4 correct-to-wrong moves. Panel D shows the accuracy/completion frontier; TRSD-64 combines the highest accuracy with the lowest cap-hit rate.

**Caption.** *TRSD is stable at 16 episodes and separates at 64. The long-horizon gain spans all three benchmarks and is strongly paired: 16 P64 errors become correct under T64, compared with 4 reverse transitions. Eleven of the sixteen favorable transitions are completion rescues, placing T64 at the best accuracy–completion point.*

Each figure is available as PNG for GitHub, vector PDF for papers, and editable SVG for slides.
"""
    (root / "figures" / "FIGURE_GUIDE.md").write_text(guide, encoding="utf-8")


def update_manifests(root: Path, episode_csv: Path) -> None:
    evidence_dir = root / "evidence" / "mechanism"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    destination = evidence_dir / "matched64_episode_style_metrics.csv"
    if episode_csv.resolve() != destination.resolve():
        shutil.copy2(episode_csv, destination)
    normalize_text_file(destination)

    evidence_manifest = root / "evidence" / "MANIFEST.json"
    if evidence_manifest.is_file():
        payload = json.loads(evidence_manifest.read_text(encoding="utf-8"))
        files = [entry for entry in payload.get("files", []) if entry.get("path") != "mechanism/matched64_episode_style_metrics.csv"]
        files.append(
            {
                "path": "mechanism/matched64_episode_style_metrics.csv",
                "rows": len(read_csv(destination)),
                "sha256": sha256(destination),
            }
        )
        payload["files"] = files
        payload["created_utc"] = datetime.now(timezone.utc).isoformat()
        evidence_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = root / "MANIFEST.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["output_files"] = sorted(
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and "logs" not in path.relative_to(root).parts
        )
        payload["created_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render(root: Path, episode_csv: Path) -> None:
    data = load_bundle(root, episode_csv)
    configure_style()
    output_dir = root / "figures"
    figure_three_claims(data, output_dir)
    figure_drift_mechanism(data, output_dir)
    figure_performance_anatomy(data, output_dir)
    write_figure_guide(root)
    update_manifests(root, episode_csv)


def main() -> None:
    args = parse_args()
    render(args.bundle, args.episode_csv)
    if args.stage is not None and args.stage.resolve() != args.bundle.resolve():
        stage_figures = args.stage / "figures"
        stage_figures.mkdir(parents=True, exist_ok=True)
        for source in (args.bundle / "figures").iterdir():
            if source.is_file():
                shutil.copy2(source, stage_figures / source.name)
        update_manifests(args.stage, args.episode_csv)


if __name__ == "__main__":
    main()
