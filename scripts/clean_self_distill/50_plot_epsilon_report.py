#!/usr/bin/env python3
"""Render the compact figure embedded by epsilon.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    blue = "#3166A3"
    red = "#C44E52"
    green = "#3A8D5D"
    purple = "#7A5195"
    orange = "#D97745"
    slate = "#5B6F8A"
    gray = "#687386"

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.75))
    figure.subplots_adjust(left=0.055, right=0.99, top=0.76, bottom=0.24, wspace=0.30)

    # Panel A: held-out performance.  Epsilon=.004 is intentionally not joined
    # to the matched-seed line because its existing generation uses another seed.
    ax = axes[0]
    eps = np.array([0.001, 0.002, 0.003, 0.006, 0.008, 0.016])
    accuracy = np.array([67.83, 67.83, 66.43, 66.43, 69.93, 67.13])
    ax.axhspan(accuracy.min(), accuracy.max(), color=blue, alpha=0.08, zorder=0)
    ax.plot(eps, accuracy, color=blue, marker="o", linewidth=2.0, markersize=5.5, label="TRSD, matched seed")
    ax.axhline(62.94, color=red, linestyle=(0, (4, 3)), linewidth=1.7, label="OPSD + policy KL, β=1")
    ax.scatter([0.004], [73.43], marker="D", s=50, facecolors="white", edgecolors=gray, linewidth=1.6, zorder=5, label="ε=.004 canonical†")
    ax.annotate("3.50-pp band", xy=(0.011, 68.1), color=blue, fontsize=8, ha="center")
    ax.set_title("A. Held-out Math accuracy")
    ax.set_ylabel("Math-Verify Accuracy@1 (%)")
    ax.set_ylim(60, 75.5)
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.7, alpha=0.75)
    ax.legend(loc="lower right", fontsize=7.4, handlelength=2.3)

    # Panel B: projection behavior on the 64 training trajectories.
    ax = axes[1]
    eps_full = np.array([0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.016])
    mean_alpha = 100 * np.array([0.275390625, 0.388427734375, 0.48291015625, 0.5595703125, 0.677734375, 0.78662109375, 0.956298828125])
    active = 100 * np.array([64, 64, 64, 63, 60, 55, 20]) / 64
    ax.plot(eps_full, mean_alpha, color=green, marker="o", linewidth=2.0, markersize=5, label="Mean α")
    ax.plot(eps_full, active, color=purple, marker="s", linewidth=2.0, markersize=4.8, label="Constraint active")
    ax.axvline(0.004, color=gray, linestyle=(0, (2, 2)), linewidth=1.2)
    ax.annotate("ε=.004\n98.4% active", xy=(0.004, 98.4), xytext=(0.0051, 83), fontsize=7.8, color=gray, arrowprops={"arrowstyle": "-", "color": gray, "lw": 0.8})
    ax.set_title("B. Per-trajectory adaptation")
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(20, 104)
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.7, alpha=0.75)
    ax.legend(loc="lower right", fontsize=7.8)

    # Panel C: independent projection-mechanism probe.
    ax = axes[2]
    mech_eps = np.array([0.001, 0.002, 0.004, 0.008, 0.016])
    style = np.array([31.6144, 46.4342, 69.3037, 99.4590, 100.0])
    prompt_variance = np.array([9.6932, 20.9202, 46.6657, 98.0336, 100.0])
    ax.plot(mech_eps, style, color=orange, marker="o", linewidth=2.0, markersize=5, label="Style retained")
    ax.plot(mech_eps, prompt_variance, color=slate, marker="s", linewidth=2.0, markersize=4.8, label="Prompt variance retained")
    ax.axhline(70, color=orange, linewidth=0.9, linestyle=(0, (2, 2)), alpha=0.65)
    ax.axhline(50, color=slate, linewidth=0.9, linestyle=(0, (2, 2)), alpha=0.65)
    ax.axvline(0.004, color=gray, linestyle=(0, (2, 2)), linewidth=1.2)
    ax.scatter([0.004, 0.004], [69.3037, 46.6657], s=85, facecolors="none", edgecolors="#222222", linewidth=1.1, zorder=5)
    ax.annotate("selected knee", xy=(0.004, 69.3037), xytext=(0.00115, 86), fontsize=7.8, color=gray, arrowprops={"arrowstyle": "->", "color": gray, "lw": 0.8})
    ax.set_title("C. Target-retention knee")
    ax.set_ylabel("Retention vs. OPSD target (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#D6DCE5", linewidth=0.7, alpha=0.75)
    ax.legend(loc="lower right", fontsize=7.3)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.00085, 0.0185)
        ax.set_xlabel("Global KL radius ε")
    for ax in axes[:2]:
        ax.set_xticks(eps_full)
        ax.set_xticklabels([".001", ".002", ".003", ".004", ".006", ".008", ".016"], fontsize=7.5)
    axes[2].set_xticks(mech_eps)
    axes[2].set_xticklabels([".001", ".002", ".004", ".008", ".016"])

    figure.suptitle("TRSD epsilon controls locality over a broad performance plateau", x=0.055, ha="left", y=0.96, fontsize=16, fontweight="bold", color="#202733")
    figure.text(0.055, 0.885, "Qwen3-8B, 64 DeepMath episodes · ordinary-prompt Math-143 evaluation", ha="left", fontsize=9.5, color=gray)
    figure.text(0.055, 0.055, "† The canonical ε=.004 point uses a different evaluation seed and is not included in the matched-sweep band. Panel C is a one-trajectory, three-wrapper mechanism probe.", ha="left", fontsize=7.7, color=gray)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
