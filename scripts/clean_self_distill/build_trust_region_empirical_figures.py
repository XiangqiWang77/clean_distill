#!/usr/bin/env python3
"""Build publication-style figures for Trust-Region vs Privileged SD."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIG_DIR = ROOT / "artifacts" / "figures"
DEFAULT_SUPPORT_SUMMARY_PATH = DEFAULT_FIG_DIR / "teacher_support_summary_trust_region_full_run.csv"
DEFAULT_BRANCH_STYLE_SUMMARY_PATH = DEFAULT_FIG_DIR / "token_style_drift_heatmap" / "token_style_drift_branch_summary.csv"
DEFAULT_TRUST_TOKEN_STYLE_PATH = DEFAULT_FIG_DIR / "token_style_drift_heatmap" / "trust_region_clean_per_episode_token_style_metrics.csv"
DEFAULT_PRIV_TOKEN_STYLE_PATH = DEFAULT_FIG_DIR / "token_style_drift_heatmap" / "privileged_per_episode_token_style_metrics.csv"
DEFAULT_TOKEN_SHIFT_PATH = DEFAULT_FIG_DIR / "token_style_drift_heatmap" / "token_signature_top40_trust_vs_privilege.csv"


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


def normalize_branch_name(name: str) -> str:
    low = str(name).lower()
    if "trust" in low or low == "clean" or "trust-region" in low:
        return "Trust-Region Clean"
    if "priv" in low:
        return "Privileged"
    return name


def build_main_summary_figure(rows: list[dict[str, str]], fig_dir: Path) -> None:
    if not rows:
        return

    branch_metrics: dict[str, dict[str, float]] = {}
    for row in rows:
        branch = normalize_branch_name(row.get("branch", ""))
        branch_metrics[branch] = {
            "mean_support_tokens": as_float(row.get("mean_support_tokens")),
            "mean_frontier_corrective": as_float(row.get("mean_frontier_corrective_tokens_selected")),
            "mean_frontier_wrong": as_float(row.get("mean_frontier_wrong_tokens_selected")),
            "mean_support_other": as_float(row.get("mean_support_other_tokens")),
            "frontier_ratio_corrective": as_float(row.get("mean_frontier_ratio_corrective")),
            "frontier_ratio_wrong": as_float(row.get("mean_frontier_ratio_wrong")),
            "frontier_margin_gain": as_float(row.get("mean_frontier_margin_gain_mean")),
            "db_crossing_rate": as_float(row.get("mean_db_crossing_rate")),
            "db_regression_rate": as_float(row.get("mean_db_regression_rate")),
            "target_margin_attainment": as_float(row.get("mean_target_margin_attainment")),
            "mean_candidate_count": as_float(row.get("mean_candidate_count")),
            "style_abs_error_mean": as_float(row.get("mean_style_abs_error_mean")),
            "task_abs_error_mean": as_float(row.get("mean_task_abs_error_mean")),
            "kl": as_float(row.get("mean_teacher_student_kl")),
            "mean_episode_seconds": as_float(row.get("mean_episode_seconds")),
            "n": as_float(row.get("n_episodes"), 0.0),
        }

    branches = ["Trust-Region Clean", "Privileged"]
    x = np.arange(len(branches))
    width = 0.34

    fig = plt.figure(figsize=(15, 10), dpi=220)
    gs = fig.add_gridspec(2, 2)

    # Support composition.
    ax0 = fig.add_subplot(gs[0, 0])
    support = np.array([
        [
            branch_metrics[b]["mean_support_other"],
            branch_metrics[b]["mean_frontier_corrective"],
            branch_metrics[b]["mean_frontier_wrong"],
        ]
        for b in branches
    ], dtype=float)
    denom = support.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    support_pct = support / denom * 100

    bottoms = np.zeros(len(branches))
    colors = ["#6C757D", "#0077B6", "#F77F00"]
    labels = ["Other", "Corrective frontier", "Wrong frontier"]
    for i, label in enumerate(labels):
        ax0.bar(x, support_pct[:, i], width=0.55, bottom=bottoms, label=label, color=colors[i])
        bottoms += support_pct[:, i]
    ax0.set_title("Support split by adaptation signal")
    ax0.set_xticks(x)
    ax0.set_xticklabels(branches)
    ax0.set_ylabel("Token share (%)")
    ax0.set_ylim(0, 100)
    ax0.grid(alpha=0.2, axis="y")
    ax0.legend(frameon=False)

    # Corrective vs wrong frontier and sample counts.
    ax1 = fig.add_subplot(gs[0, 1])
    corr = [branch_metrics[b]["frontier_ratio_corrective"] for b in branches]
    wrong = [branch_metrics[b]["frontier_ratio_wrong"] for b in branches]
    candidates = [branch_metrics[b]["mean_candidate_count"] for b in branches]
    n_eps = [branch_metrics[b]["n"] for b in branches]
    ax1.bar(x - width / 2, corr, width=width, color="#2A9D8F", label="Corrective ratio")
    ax1.bar(x + width / 2, wrong, width=width, color="#E76F51", label="Wrong ratio")
    ax1.set_ylabel("Token frontier ratio")
    ax1.set_title("Signal composition and frontier counts")
    ax1.set_xticks(x)
    ax1.set_xticklabels(branches)
    ax1.grid(alpha=0.2, axis="y")

    ax1_t = ax1.twinx()
    ax1_t.plot(x, candidates, marker="o", color="#264653", label="Mean frontier-support candidates", linestyle="--")
    ax1_t.set_ylabel("Mean candidate count")
    ax1_t.plot(x, [v / max(v or 1.0, 1.0) for v in n_eps], marker="s", color="#457B9D", alpha=0.0)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    # Cleanliness: KL + task/style effect.
    ax2 = fig.add_subplot(gs[1, 0])
    bars = np.array([
        [branch_metrics[b]["kl"] for b in branches],
        [branch_metrics[b]["style_abs_error_mean"] for b in branches],
        [branch_metrics[b]["task_abs_error_mean"] for b in branches],
    ])
    colors2 = ["#1D3557", "#F4A261", "#2A9D8F"]
    labels2 = ["KL", "Style mean abs error", "Task mean abs error"]
    xpos = np.arange(len(branches))
    for i, lab in enumerate(labels2):
        ax2.bar(xpos + (i - 1) * width, bars[i], width=width, label=lab, color=colors2[i])
    ax2.set_title("Cleanliness and task signal quality")
    ax2.set_xticks(x)
    ax2.set_xticklabels(branches)
    ax2.set_ylabel("Mean score")
    ax2.grid(alpha=0.2, axis="y")

    ax2_t = ax2.twinx()
    ax2_t.plot(x, [branch_metrics[b]["frontier_margin_gain"] for b in branches],
               marker="D", linestyle="-", color="#E63946", label="Frontier margin gain")
    ax2_t.set_ylabel("Mean frontier gain")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_t.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    # Runtime + boundary stability.
    ax3 = fig.add_subplot(gs[1, 1])
    sec = [branch_metrics[b]["mean_episode_seconds"] for b in branches]
    cross = [branch_metrics[b]["db_crossing_rate"] for b in branches]
    regress = [branch_metrics[b]["db_regression_rate"] for b in branches]
    target = [branch_metrics[b]["target_margin_attainment"] for b in branches]

    ax3.bar(x - width / 2, sec, width=width, color="#8ECAE6", label="Episode sec")
    ax3.set_title("Runtime and boundary behavior")
    ax3.set_xticks(x)
    ax3.set_xticklabels(branches)
    ax3.set_ylabel("Seconds")
    ax3.grid(alpha=0.2, axis="y")

    ax3_t = ax3.twinx()
    ax3_t.plot(x, cross, marker="o", color="#E76F51", label="Decision crossing")
    ax3_t.plot(x + width / 4, regress, marker="s", color="#6D6875", linestyle="--", label="Regression")
    ax3_t.plot(x + width / 2, target, marker="^", color="#2A9D8F", linestyle=":", label="Target attainment")
    ax3_t.set_ylabel("Rate")

    l1, lb1 = ax3.get_legend_handles_labels()
    l2, lb2 = ax3_t.get_legend_handles_labels()
    ax3.legend(l1 + l2, lb1 + lb2, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.suptitle("Trust-Region Clean SD vs Privileged SD", fontsize=15, y=1.02)
    fig.savefig(fig_dir / "trust_region_vs_privileged_summary.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_style_drift_figure(
    branch_rows: list[dict[str, str]],
    trust_rows: list[dict[str, str]],
    priv_rows: list[dict[str, str]],
    fig_dir: Path,
) -> None:
    if not branch_rows:
        return

    branch: dict[str, dict[str, float]] = {}
    for row in branch_rows:
        b = row["branch"]
        branch[b] = {
            "style_abs_error_mean": as_float(row.get("style_abs_error_mean")),
            "task_abs_error_mean": as_float(row.get("task_abs_error_mean")),
            "style_task_ratio": as_float(row.get("style_task_ratio")),
            "style_token_frac": as_float(row.get("style_token_frac")),
            "task_token_frac": as_float(row.get("task_token_frac")),
            "other_token_frac": as_float(row.get("other_token_frac")),
            "kl": as_float(row.get("mean_teacher_student_kl")),
            "episode_seconds": as_float(row.get("episode_seconds")),
        }

    branches = ["Trust-Region Clean", "Privileged"]
    x = np.arange(len(branches))

    fig, ax = plt.subplots(2, 2, figsize=(15, 10), dpi=220)

    # Error decomposition.
    style = [branch[b]["style_abs_error_mean"] for b in branches]
    task = [branch[b]["task_abs_error_mean"] for b in branches]
    ratio = [branch[b]["style_task_ratio"] for b in branches]

    ax[0, 0].bar(x - 0.18, style, width=0.36, label="Style abs error", color="#E9C46A")
    ax[0, 0].bar(x + 0.18, task, width=0.36, label="Task abs error", color="#2A9D8F")
    ax[0, 0].set_title("Token-level error decomposition")
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels(branches)
    ax[0, 0].set_ylabel("Mean abs error")
    ax[0, 0].grid(alpha=0.2, axis="y")

    ax0_t = ax[0, 0].twinx()
    ax0_t.plot(x, ratio, marker="D", color="#264653", label="Style/Task ratio")
    ax0_t.set_ylabel("Ratio")
    lines1, labs1 = ax[0, 0].get_legend_handles_labels()
    lines2, labs2 = ax0_t.get_legend_handles_labels()
    ax[0, 0].legend(lines1 + lines2, labs1 + labs2, frameon=False)

    # Token-type fractions.
    style_frac = [branch[b]["style_token_frac"] for b in branches]
    task_frac = [branch[b]["task_token_frac"] for b in branches]
    other_frac = [branch[b]["other_token_frac"] for b in branches]
    arr = np.array([style_frac, task_frac, other_frac]).T
    base = np.zeros(len(branches))
    for i, lab, c in zip(range(3), ["Style", "Task", "Other"], ["#E9C46A", "#2A9D8F", "#ADB5BD"]):
        ax[0, 1].bar(x, arr[:, i], width=0.55, bottom=base, label=lab, color=c)
        base += arr[:, i]
    ax[0, 1].set_title("Composition of compared tokens")
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels(branches)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_ylabel("Fraction")
    ax[0, 1].legend(loc="lower right", frameon=False)

    # Per-episode trajectories.
    def add_series(rows: list[dict[str, str]], label_prefix: str, axis):
        if not rows:
            return
        rows2 = sorted(rows, key=lambda r: as_float(r.get("episode"), 0.0))
        episodes = [as_float(r.get("episode")) for r in rows2]
        style_vals = [as_float(r.get("style_abs_error_mean")) for r in rows2]
        task_vals = [as_float(r.get("task_abs_error_mean")) for r in rows2]
        axis.plot(episodes, style_vals, marker="o", linewidth=1.1, alpha=0.8, label=f"{label_prefix}: style")
        axis.plot(episodes, task_vals, marker="s", linestyle="--", linewidth=1.1, alpha=0.8, label=f"{label_prefix}: task")

    add_series(trust_rows, "Trust", ax[1, 0])
    add_series(priv_rows, "Priv", ax[1, 0])
    ax[1, 0].set_title("Per-episode mean token error")
    ax[1, 0].set_xlabel("Episode")
    ax[1, 0].set_ylabel("Mean abs error")
    ax[1, 0].grid(alpha=0.2)
    ax[1, 0].legend(fontsize=8, frameon=False, loc="upper right")

    # Runtime-KL trade-off.
    sec = [branch[b]["episode_seconds"] for b in branches]
    kl = [branch[b]["kl"] for b in branches]
    ax[1, 1].bar(x, sec, width=0.35, color="#8ECAE6")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(branches)
    ax[1, 1].set_title("Runtime vs distillation KL")
    ax[1, 1].set_ylabel("Mean seconds")
    ax[1, 1].grid(alpha=0.2, axis="y")
    ax1t = ax[1, 1].twinx()
    ax1t.plot(x, kl, marker="D", linestyle="-", color="#264653", label="Mean KL")
    ax1t.set_ylabel("Mean KL")
    l1, lb1 = ax[1, 1].get_legend_handles_labels()
    l2, lb2 = ax1t.get_legend_handles_labels()
    ax[1, 1].legend(l1 + l2, lb1 + lb2, frameon=False)

    fig.tight_layout()
    fig.suptitle("Trust vs Privileged token-behavior profile", fontsize=15, y=1.02)
    fig.savefig(fig_dir / "trust_region_style_drift_summary.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_token_signature_figure(token_rows: list[dict[str, str]], fig_dir: Path, max_tokens: int) -> None:
    if not token_rows:
        return

    parsed = []
    for row in token_rows:
        token = str(row.get("token", "")).strip()
        if not token:
            continue
        trust = as_float(row.get("trust_region_mean_delta"))
        trust_abs = as_float(row.get("trust_region_mean_abs_delta"))
        priv = as_float(row.get("privileged_estimated_mean_delta"))
        alpha = as_float(row.get("projection_alpha_weighted_mean"), 1.0)
        parsed.append(
            {
                "token": token,
                "trust": trust,
                "trust_abs": trust_abs,
                "priv": priv,
                "priv_abs": as_float(row.get("privileged_estimated_mean_abs_delta")),
                "alpha": alpha,
                "shrink": trust - priv,
            }
        )

    parsed.sort(key=lambda x: abs(x["trust_abs"]), reverse=True)
    if max_tokens > 0 and len(parsed) > max_tokens:
        parsed = parsed[:max_tokens]

    tokens = [x["token"] for x in parsed]
    trust = np.array([x["trust"] for x in parsed], dtype=float)
    trust_abs = np.array([x["trust_abs"] for x in parsed], dtype=float)
    priv = np.array([x["priv"] for x in parsed], dtype=float)
    priv_abs = np.array([x["priv_abs"] for x in parsed], dtype=float)
    alpha = np.array([x["alpha"] for x in parsed], dtype=float)
    shrink = np.array([x["shrink"] for x in parsed], dtype=float)

    idx = np.arange(len(tokens))
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), dpi=220)

    axes[0].barh(idx - 0.22, trust, 0.22, color="#2A9D8F", label="Trust-Region (closed form)")
    axes[0].barh(idx + 0.22, priv, 0.22, color="#E76F51", alpha=0.9, label="Privileged (estimated)")
    axes[0].set_yticks(idx)
    axes[0].set_yticklabels(tokens, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#264653", lw=1)
    axes[0].set_title("Per-token mean logit-shift: trust-region closes privileged")
    axes[0].set_xlabel("Mean signed shift")
    axes[0].legend(loc="lower right", frameon=False)

    # Magnitude and shrinkage.
    axes[1].barh(idx - 0.22, trust_abs, 0.2, color="#52B788", label="|Trust|")
    axes[1].barh(idx + 0.22, priv_abs, 0.2, color="#FFD166", alpha=0.9, label="|Estimated privileged|")
    axes[1].set_yticks(idx)
    axes[1].set_yticklabels(tokens, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title("Per-token shift magnitude (closed form attenuation)")
    axes[1].set_xlabel("Mean absolute shift")
    axes[1].legend(loc="lower right", frameon=False)

    # Optional shrink overlay as tiny alpha line on top panel.
    # Keep compact: place a secondary strip in bottom area using twinx.
    ax2 = axes[1].twiny()
    ax2.scatter(np.abs(shrink), idx, c=alpha, cmap="viridis", s=8, alpha=0.6, label="Trust shrink")
    ax2.set_xlabel("|Closed-Form Shrink|")
    ax2.set_xlim(left=0)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=alpha.min(), vmax=alpha.max() or 1.0)), ax=axes[1], pad=0.01)
    cbar.set_label("Projection alpha (mean)")

    for axis in axes:
        axis.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.suptitle(
        "Token signature heatmap: trust projection trims high-magnitude privileged shifts",
        fontsize=15,
        y=1.02,
    )
    fig.savefig(fig_dir / "token_signature_top40_trust_vs_privilege.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_token_signature_heatmap(token_rows: list[dict[str, str]], fig_dir: Path, max_tokens: int) -> None:
    if not token_rows:
        return

    parsed = []
    for row in token_rows:
        token = str(row.get("token", "")).strip()
        if not token:
            continue
        trust_delta = as_float(row.get("trust_region_mean_delta"))
        priv_delta = as_float(row.get("privileged_estimated_mean_delta"))
        trust_abs = as_float(row.get("trust_region_mean_abs_delta"))
        alpha = as_float(row.get("projection_alpha_weighted_mean"), 1.0)
        parsed.append(
            {
                "token": token,
                "trust": trust_delta,
                "priv": priv_delta,
                "diff": trust_delta - priv_delta,
                "trust_abs": trust_abs,
                "alpha": alpha,
            }
        )

    parsed.sort(key=lambda x: abs(x["trust_abs"]), reverse=True)
    if max_tokens > 0 and len(parsed) > max_tokens:
        parsed = parsed[:max_tokens]

    if not parsed:
        return

    tokens = [x["token"] for x in parsed]
    matrix = np.array(
        [
            [x["trust"] for x in parsed],
            [x["priv"] for x in parsed],
            [x["diff"] for x in parsed],
        ],
        dtype=float,
    ).T

    vlim = max(1e-6, float(np.max(np.abs(matrix))))
    alphas = np.array([x["alpha"] for x in parsed], dtype=float)
    shrink = np.abs(matrix[:, 0] - matrix[:, 1])
    shrink = shrink / (shrink.max() if shrink.max() > 0 else 1.0)

    fig = plt.figure(
        figsize=(14, max(6.0, min(len(tokens) * 0.12, 16.0))),
        dpi=180,
    )
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 1, 1.2], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vlim, vmax=vlim)
    ax0.set_title("Token-shift heatmap: Trust vs Estimated Privileged")
    ax0.set_yticks(range(len(tokens)))
    ax0.set_yticklabels(tokens, fontsize=8)
    ax0.set_xticks([0, 1, 2])
    ax0.set_xticklabels(["Trust", "Priv-est", "Trust-Priv"], rotation=20)
    ax0.set_ylabel("Token")
    ax0.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax0, pad=0.01)
    cbar.set_label("Mean logit-shift")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.barh(range(len(tokens)), shrink, color="#2A9D8F", alpha=0.75)
    ax1.set_yticks(range(len(tokens)))
    ax1.set_yticklabels(tokens, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.0)
    ax1.set_title("Normalized trust-priv diff")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(alphas, range(len(tokens)), color="#E76F51", s=14, alpha=0.85)
    ax2.set_yticks(range(len(tokens)))
    ax2.set_yticklabels(tokens, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.0)
    ax2.set_title("Projection α")
    ax2.set_xlabel("")

    fig.tight_layout()
    fig.savefig(fig_dir / "token_signature_heatmap_trust_vs_privilege.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-summary-path", type=Path, default=DEFAULT_SUPPORT_SUMMARY_PATH)
    parser.add_argument("--branch-style-summary-path", type=Path, default=DEFAULT_BRANCH_STYLE_SUMMARY_PATH)
    parser.add_argument("--trust-style-path", type=Path, default=DEFAULT_TRUST_TOKEN_STYLE_PATH)
    parser.add_argument("--priv-style-path", type=Path, default=DEFAULT_PRIV_TOKEN_STYLE_PATH)
    parser.add_argument("--token-shift-path", type=Path, default=DEFAULT_TOKEN_SHIFT_PATH)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--max-token-vocab", type=int, default=120)
    parser.add_argument("--skip-heatmap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    support_rows = read_csv(args.support_summary_path)
    if not support_rows:
        raise SystemExit(f"No data in {args.support_summary_path}")

    build_main_summary_figure(support_rows, args.fig_dir)

    branch_rows = read_csv(args.branch_style_summary_path)
    trust_rows = read_csv(args.trust_style_path)
    priv_rows = read_csv(args.priv_style_path)
    if branch_rows:
        build_style_drift_figure(branch_rows, trust_rows, priv_rows, args.fig_dir)

    token_rows = read_csv(args.token_shift_path)
    if token_rows:
        build_token_signature_figure(token_rows, args.fig_dir, args.max_token_vocab)
        if not args.skip_heatmap:
            build_token_signature_heatmap(token_rows, args.fig_dir, args.max_token_vocab)


if __name__ == "__main__":
    main()
