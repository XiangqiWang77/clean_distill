#!/usr/bin/env python3
"""Build non-toy empirical figures for Trust-Region Clean SD vs privileged baseline."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "artifacts" / "figures"

SUPPORT_SUMMARY_PATH = FIG_DIR / "teacher_support_summary_trust_region_full_run.csv"
BRANCH_STYLE_SUMMARY_PATH = FIG_DIR / "token_style_drift_heatmap" / "token_style_drift_branch_summary.csv"
TRUST_TOKEN_STYLE_PATH = FIG_DIR / "token_style_drift_heatmap" / "trust_region_clean_per_episode_token_style_metrics.csv"
PRIV_TOKEN_STYLE_PATH = FIG_DIR / "token_style_drift_heatmap" / "privileged_per_episode_token_style_metrics.csv"
TOKEN_SHIFT_PATH = FIG_DIR / "token_style_drift_heatmap" / "token_signature_top40_trust_vs_privilege.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def as_float(value: str | None, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def normalize_name(name: str) -> str:
    if "trust-region" in name.lower():
        return "Trust-Region Clean"
    if "privileged" in name.lower():
        return "Privileged"
    return name


def build_main_summary_figure(rows: list[dict[str, str]]) -> None:
    branch_metrics: dict[str, dict[str, float]] = {}
    for row in rows:
        branch = normalize_name(row["branch"])
        branch_metrics[branch] = {
            "support_tokens": as_float(row.get("mean_support_tokens")),
            "frontier_corrective_tokens_selected": as_float(row.get("mean_frontier_corrective_tokens_selected")),
            "frontier_wrong_tokens_selected": as_float(row.get("mean_frontier_wrong_tokens_selected")),
            "support_other_tokens": as_float(row.get("mean_support_other_tokens")),
            "frontier_ratio_corrective": as_float(row.get("mean_frontier_ratio_corrective")),
            "frontier_ratio_wrong": as_float(row.get("mean_frontier_ratio_wrong")),
            "mean_frontier_margin_gain": as_float(row.get("mean_frontier_margin_gain_mean")),
            "db_crossing_rate": as_float(row.get("mean_db_crossing_rate")),
            "target_margin_attainment": as_float(row.get("mean_target_margin_attainment")),
            "mean_candidate_count": as_float(row.get("mean_candidate_count")),
            "style_abs_error_mean": as_float(row.get("mean_style_abs_error_mean")),
            "task_abs_error_mean": as_float(row.get("mean_task_abs_error_mean")),
            "mean_teacher_student_kl": as_float(row.get("mean_teacher_student_kl")),
            "mean_episode_seconds": as_float(row.get("mean_episode_seconds")),
        }

    branches = ["Trust-Region Clean", "Privileged"]
    x = np.arange(len(branches))
    width = 0.35

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.tight_layout()

    # Support composition.
    supp = np.array([[
        branch_metrics[b]["support_other_tokens"],
        branch_metrics[b]["frontier_corrective_tokens_selected"],
        branch_metrics[b]["frontier_wrong_tokens_selected"],
    ] for b in branches], dtype=float)
    totals = supp.sum(axis=1)
    bottom = np.zeros(len(branches))
    colors = ["#8ecae6", "#219ebc", "#ffb703"]
    labels = ["Other support", "Corrective frontier", "Wrong frontier"]
    for i, lab in enumerate(labels):
        vals = supp[:, i] / np.maximum(totals, 1.0)
        ax[0, 0].bar(x, vals, width=0.45, bottom=bottom, label=lab, color=colors[i])
        bottom += vals
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels(branches)
    ax[0, 0].set_title("Support composition in adaptation signals")
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_ylabel("Fraction")
    ax[0, 0].grid(axis="y", alpha=0.3)
    ax[0, 0].legend(loc="upper right", frameon=False)

    # Frontier ratio and margin.
    rat = np.array([
        [branch_metrics[b]["frontier_ratio_corrective"] for b in branches],
        [branch_metrics[b]["frontier_ratio_wrong"] for b in branches],
    ])
    at = np.array([
        branch_metrics[b]["mean_candidate_count"] for b in branches], dtype=float
    )
    ax[0, 1].bar(x - width / 2, rat[0], width=width, label="Corrective ratio")
    ax[0, 1].bar(x + width / 2, rat[1], width=width, label="Wrong ratio")
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels(branches)
    ax[0, 1].set_ylabel("Token ratio")
    ax[0, 1].set_title("Frontier ratio split")
    ax[0, 1].grid(axis="y", alpha=0.3)

    ax2 = ax[0, 1].twinx()
    ax2.plot(x, at, color="black", marker="o", linestyle="--", label="Mean candidates")
    ax2.set_ylabel("Mean candidate count")

    lines, labels1 = ax[0, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax[0, 1].legend(lines + lines2, labels1 + labels2, loc="upper right", frameon=False)

    # Distillation quality (cleanliness) panel.
    width = 0.35
    y = np.array([
        [branch_metrics[b]["mean_teacher_student_kl"] for b in branches],
        [branch_metrics[b]["style_abs_error_mean"] for b in branches],
        [branch_metrics[b]["task_abs_error_mean"] for b in branches],
    ])
    x2 = np.arange(len(branches))
    labels = ["KL", "Mean style abs error", "Mean task abs error"]
    colors = ["#219ebc", "#ff6b6b", "#06d6a0"]
    ax[1, 0].bar(x2 - width, y[0], width, label=labels[0], color=colors[0])
    ax[1, 0].bar(x2, y[1], width, label=labels[1], color=colors[1])
    ax[1, 0].bar(x2 + width, y[2], width, label=labels[2], color=colors[2])
    ax[1, 0].set_xticks(x2)
    ax[1, 0].set_xticklabels(branches)
    ax[1, 0].set_title("Teacher shift and cleanliness")
    ax[1, 0].grid(axis="y", alpha=0.3)
    ax[1, 0].legend(frameon=False, ncol=3)

    ax2 = ax[1, 0].twinx()
    sec = np.array([branch_metrics[b]["mean_frontier_margin_gain"] for b in branches], dtype=float)
    ax2.plot(x2, sec, color="#333333", marker="D", linestyle="-", label="Mean frontier margin")
    ax2.set_ylabel("Margin gain")
    lines, labels1 = ax[1, 0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax[1, 0].legend(lines + lines2, labels1 + labels2, loc="upper left", frameon=False)

    # Runtime/crossing panel.
    y_runtime = np.array([branch_metrics[b]["mean_episode_seconds"] for b in branches], dtype=float)
    y_cross = np.array([branch_metrics[b]["db_crossing_rate"] for b in branches], dtype=float)
    y_target = np.array([branch_metrics[b]["target_margin_attainment"] for b in branches], dtype=float)
    ax[1, 1].bar(x - width / 2, y_runtime, width=width, color="#8ecae6", label="Episode time (s)")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(branches)
    ax[1, 1].set_title("Efficiency and boundary behaviour")
    ax[1, 1].set_ylabel("Mean seconds")
    ax[1, 1].grid(axis="y", alpha=0.3)

    ax2 = ax[1, 1].twinx()
    ax2.plot(x + width / 2, y_cross, color="#e63946", marker="o", linestyle="-", label="Decision crossing")
    ax2.plot(x + width / 2, y_target, color="#8338ec", marker="s", linestyle="--", label="Target attainment")
    ax2.set_ylabel("Rate")

    lines, labels1 = ax[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax[1, 1].legend(lines + lines2, labels1 + labels2, loc="upper left", frameon=False)

    for a in ax.flat:
        for spine in [a.spines["top"], a.spines["right"]]:
            spine.set_visible(False)

    out = FIG_DIR / "trust_region_vs_privileged_summary.png"
    fig.suptitle("Trust-Region Clean SD vs privileged SD (summary metrics)", fontsize=14)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")


def build_style_drift_figure(style_rows: list[dict[str, str]],
                           trust_rows: list[dict[str, str]],
                           priv_rows: list[dict[str, str]]) -> None:
    # Branch mean style summary.
    bdata: dict[str, dict[str, float]] = {}
    for row in style_rows:
        branch = row["branch"]
        bdata[branch] = {
            "style_abs_error_mean": as_float(row.get("style_abs_error_mean")),
            "task_abs_error_mean": as_float(row.get("task_abs_error_mean")),
            "style_task_ratio": as_float(row.get("style_task_ratio")),
            "mean_teacher_student_kl": as_float(row.get("mean_teacher_student_kl")),
            "style_token_frac": as_float(row.get("style_token_frac")),
            "task_token_frac": as_float(row.get("task_token_frac")),
            "other_token_frac": as_float(row.get("other_token_frac")),
            "episode_seconds": as_float(row.get("episode_seconds")),
        }

    branches = ["Trust-Region Clean", "Privileged"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.tight_layout()

    # Style-task error and KL.
    x = np.arange(len(branches))
    width = 0.28
    style_err = [bdata[b]["style_abs_error_mean"] for b in branches]
    task_err = [bdata[b]["task_abs_error_mean"] for b in branches]
    ratio = [bdata[b]["style_task_ratio"] for b in branches]

    ax[0, 0].bar(x - width, style_err, width, label="Style abs error", color="#ffb703")
    ax[0, 0].bar(x, task_err, width, label="Task abs error", color="#06d6a0")
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels(branches)
    ax[0, 0].set_title("Per-token error means")
    ax[0, 0].set_ylabel("Mean abs error")
    ax[0, 0].grid(axis="y", alpha=0.3)
    ax2 = ax[0, 0].twinx()
    ax2.plot(x + width, ratio, marker="D", color="#333333", label="Style/Task ratio")
    ax2.set_ylabel("Style/Task error ratio")
    lines1, labels1 = ax[0, 0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax[0, 0].legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    # Token fractions.
    style_frac = [bdata[b]["style_token_frac"] for b in branches]
    task_frac = [bdata[b]["task_token_frac"] for b in branches]
    other_frac = [bdata[b]["other_token_frac"] for b in branches]
    arr = np.array([style_frac, task_frac, other_frac]).T
    bottom = np.zeros(len(branches))
    colors = ["#ffb703", "#3a86ff", "#d9d9d9"]
    labels = ["Style", "Task", "Other"]
    for i, lab in enumerate(labels):
        ax[0, 1].bar(x, arr[:, i], width=0.55, bottom=bottom, color=colors[i], label=lab)
        bottom += arr[:, i]
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels(branches)
    ax[0, 1].set_title("Token-type composition of distillation states")
    ax[0, 1].set_ylabel("Token fraction")
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].grid(axis="y", alpha=0.3)
    ax[0, 1].legend(loc="lower right", frameon=False)

    # Per-episode drift (where available).
    def plot_series(rows: list[dict[str, str]], branch: str, axis):
        if not rows:
            return
        rows2 = sorted(rows, key=lambda r: as_float(r.get("episode"), 0.0))
        episodes = [as_float(r.get("episode"), 0.0) for r in rows2]
        axis.plot(episodes, [as_float(r.get("style_abs_error_mean")) for r in rows2], marker="o", label=f"{branch}: style")
        axis.plot(episodes, [as_float(r.get("task_abs_error_mean")) for r in rows2], marker="s", linestyle="--", label=f"{branch}: task")

    ax[1, 0].set_title("Per-episode per-token error (available samples)")
    plot_series(trust_rows, "Trust", ax[1, 0])
    plot_series(priv_rows, "Priv", ax[1, 0])
    ax[1, 0].set_xlabel("Episode")
    ax[1, 0].set_ylabel("Mean abs error")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend(loc="upper right", fontsize=8, frameon=False)

    # Efficiency by branch.
    sec = [bdata[b]["episode_seconds"] for b in branches]
    kl = [bdata[b]["mean_teacher_student_kl"] for b in branches]
    ax[1, 1].bar(x, sec, width=0.35, color="#8ecae6", label="Episode seconds")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(branches)
    ax[1, 1].set_title("Runtime / KL tradeoff")
    ax[1, 1].set_ylabel("Mean seconds")
    ax[1, 1].grid(axis="y", alpha=0.3)
    ax2 = ax[1, 1].twinx()
    ax2.plot(x + width / 2, kl, color="#d62828", marker="D", linestyle="--", label="Mean KL")
    ax2.set_ylabel("Mean KL")
    lines1, labels1 = ax[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax[1, 1].legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")

    out = FIG_DIR / "trust_region_style_drift_summary.png"
    fig.suptitle("Style drift and behavioral decomposition", fontsize=14)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")


def build_token_signature_figure(token_rows: list[dict[str, str]]) -> None:
    if not token_rows:
        return
    parsed = []
    for r in token_rows:
        token = r.get("token", "")
        td = as_float(r.get("trust_region_mean_delta"))
        pdv = as_float(r.get("privileged_mean_delta"))
        diff = as_float(r.get("delta_difference"))
        parsed.append((token.strip(), td, pdv, diff))

    parsed.sort(key=lambda x: abs(x[3]), reverse=True)
    parsed = parsed[:20]

    tokens = [x[0] for x in parsed]
    trust = np.array([x[1] for x in parsed], dtype=float)
    priv = np.array([x[2] for x in parsed], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    fig.tight_layout()
    idx = np.arange(len(tokens))

    # Signed logit-shift comparison.
    axes[0].barh(idx - 0.18, trust, 0.36, label="Trust-Region", color="#0077b6")
    axes[0].barh(idx + 0.18, priv, 0.36, label="Privileged", color="#e76f51")
    axes[0].set_yticks(idx)
    axes[0].set_yticklabels([str(t)[:32] for t in tokens], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#333333", linewidth=1)
    axes[0].set_title("Per-token signed mean logit shift")
    axes[0].set_xlabel("Mean delta")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="x", alpha=0.2)

    # Magnitude comparison.
    magnitude = np.array([abs(v) for v in trust])
    axes[1].barh(idx, magnitude, color="#2a9d8f", height=0.45)
    axes[1].set_yticks(idx)
    axes[1].set_yticklabels([str(t)[:32] for t in tokens], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title("|Trust-Region shift| (top tokens)")
    axes[1].set_xlabel("Absolute mean delta")
    axes[1].grid(axis="x", alpha=0.2)

    out = FIG_DIR / "token_signature_top40_trust_vs_privilege.png"
    fig.suptitle("Token-drift signatures: what gets up-weighted/down-weighted", fontsize=14)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")


def main() -> None:
    support_rows = read_csv(SUPPORT_SUMMARY_PATH)
    if not support_rows:
        raise SystemExit(f"No data in {SUPPORT_SUMMARY_PATH}")
    build_main_summary_figure(support_rows)

    style_rows = read_csv(BRANCH_STYLE_SUMMARY_PATH)
    trust_rows = read_csv(TRUST_TOKEN_STYLE_PATH)
    priv_rows = read_csv(PRIV_TOKEN_STYLE_PATH)
    if style_rows:
        build_style_drift_figure(style_rows, trust_rows, priv_rows)

    token_rows = read_csv(TOKEN_SHIFT_PATH)
    if token_rows:
        build_token_signature_figure(token_rows)


if __name__ == "__main__":
    main()
