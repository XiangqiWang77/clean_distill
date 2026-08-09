#!/usr/bin/env python3
"""Join the lexical style diagnostic with vocabulary-free policy drift."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "raw_privileged": "Privilege-SD 64",
    "trsd_projected": "TRSD 64",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ci(mean: str, low: str, high: str, digits: int) -> str:
    return f"{float(mean):.{digits}f} [{float(low):.{digits}f}, {float(high):.{digits}f}]"


def paired_policy_delta(
    path: Path, *, seed: int = 20260809, draws: int = 10000
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    paired: dict[str, dict[str, float]] = {}
    for row in read_csv(path):
        paired.setdefault(row["query_id"], {})[row["method"]] = float(
            row["normalized_jsd"]
        )
    differences = [
        values["TRSD 64"] - values["Privilege-SD 64"]
        for values in paired.values()
        if {"TRSD 64", "Privilege-SD 64"} <= values.keys()
    ]
    if not differences:
        return None
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(draws)
    )
    return {
        "mean": sum(differences) / len(differences),
        "ci_low": means[int(0.025 * draws)],
        "ci_high": means[min(draws - 1, int(0.975 * draws))],
        "positive": sum(value > 0 for value in differences),
        "pairs": len(differences),
    }


def plot_metrics(
    path: Path,
    rows: list[dict[str, Any]],
    paired_delta: dict[str, Any] | None,
) -> None:
    if any(row["anchored_policy_jsd"] == "pending" for row in rows):
        return
    colors = ["#B97925", "#D5533D"]
    labels = [str(row["method"]) for row in rows]
    x = np.arange(len(rows))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)

    lexical = np.asarray([row["lexicon_style_logprob_shift"] for row in rows])
    lexical_low = np.asarray([row["lexicon_style_ci_low"] for row in rows])
    lexical_high = np.asarray([row["lexicon_style_ci_high"] for row in rows])
    bars = axes[0].bar(
        x,
        lexical,
        color=colors,
        width=0.62,
        yerr=np.vstack([lexical - lexical_low, lexical_high - lexical]),
        capsize=4,
    )
    axes[0].bar_label(bars, labels=[f"{value:.4f}" for value in lexical], padding=5)
    reduction = 100.0 * (1.0 - lexical[1] / lexical[0])
    axes[0].text(
        0.5,
        0.92,
        f"TRSD: {reduction:.1f}% less marker transfer",
        transform=axes[0].transAxes,
        ha="center",
        color=colors[1],
        fontweight="bold",
    )
    axes[0].set_ylabel(r"Mean $|\Delta\log p|$ / style-marker token ↓")
    axes[0].set_title("A  Lexicon-conditioned diagnostic", loc="left", fontweight="bold")

    policy = np.asarray([float(row["anchored_policy_jsd"]) for row in rows])
    policy_low = np.asarray([float(row["anchored_policy_jsd_ci_low"]) for row in rows])
    policy_high = np.asarray([float(row["anchored_policy_jsd_ci_high"]) for row in rows])
    bars = axes[1].bar(
        x,
        policy,
        color=colors,
        width=0.62,
        yerr=np.vstack([policy - policy_low, policy_high - policy]),
        capsize=4,
    )
    axes[1].bar_label(bars, labels=[f"{value:.4f}" for value in policy], padding=5)
    if paired_delta:
        delta_text = (
            f"Paired Δ = +{paired_delta['mean']:.4f} "
            f"[{paired_delta['ci_low']:.4f}, {paired_delta['ci_high']:.4f}]"
        )
        axes[1].text(
            0.5,
            0.92,
            delta_text,
            transform=axes[1].transAxes,
            ha="center",
            color=colors[1],
            fontweight="bold",
        )
    axes[1].set_ylabel("Normalized anchored JS divergence [0, 1]")
    axes[1].set_title("B  Lexicon-free policy drift", loc="left", fontweight="bold")

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#D8DDE3", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
    figure.suptitle(
        "TRSD redirects adaptation away from privileged style markers",
        fontsize=14,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lexical",
        type=Path,
        default=Path(
            "docs/experiments/trsd_table_report_20260808/tables/"
            "trust_region_target_summary.csv"
        ),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("docs/figures/anchored_policy_jsd_summary.csv"),
    )
    parser.add_argument(
        "--policy-per-query",
        type=Path,
        default=Path("docs/figures/anchored_policy_jsd_per_query.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("docs/results/drift_metrics_side_by_side.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/results/drift_metrics_side_by_side.md"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("docs/figures/style_and_policy_drift.png"),
    )
    args = parser.parse_args()

    lexical = {
        LABELS[row["target"]]: row
        for row in read_csv(args.lexical)
        if row["target"] in LABELS
    }
    policy = (
        {row["method"]: row for row in read_csv(args.policy)}
        if args.policy.is_file()
        else {}
    )
    methods = ("Privilege-SD 64", "TRSD 64")
    rows: list[dict[str, Any]] = []
    for method in methods:
        old = lexical[method]
        new = policy.get(method)
        rows.append(
            {
                "method": method,
                "optimizer_steps": int(old["optimizer_steps"]),
                "lexicon_style_logprob_shift": float(old["style_error_per_token"]),
                "lexicon_style_ci_low": float(old["style_error_per_token_ci_low"]),
                "lexicon_style_ci_high": float(old["style_error_per_token_ci_high"]),
                "anchored_policy_jsd": (
                    float(new["normalized_jsd_mean"]) if new else "pending"
                ),
                "anchored_policy_jsd_ci_low": (
                    float(new["normalized_jsd_ci_low"]) if new else "pending"
                ),
                "anchored_policy_jsd_ci_high": (
                    float(new["normalized_jsd_ci_high"]) if new else "pending"
                ),
                "entropy_retention": (
                    float(new["entropy_retention_mean"]) if new else "pending"
                ),
                "anchor_queries": int(new["queries"]) if new else "pending",
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Drift metrics: lexical diagnostic and lexicon-free policy drift",
        "",
        "| Method | Training steps | Lexicon-conditioned style drift ↓ | AP-JSD | Entropy retention |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, row in zip(methods, rows):
        old = lexical[method]
        lexical_cell = ci(
            old["style_error_per_token"],
            old["style_error_per_token_ci_low"],
            old["style_error_per_token_ci_high"],
            4,
        )
        if policy.get(method):
            new = policy[method]
            js_cell = ci(
                new["normalized_jsd_mean"],
                new["normalized_jsd_ci_low"],
                new["normalized_jsd_ci_high"],
                5,
            )
            entropy_cell = ci(
                new["entropy_retention_mean"],
                new["entropy_retention_ci_low"],
                new["entropy_retention_ci_high"],
                3,
            )
        else:
            js_cell = "pending fixed-prefix evaluation"
            entropy_cell = "pending fixed-prefix evaluation"
        markdown.append(
            f"| {method} | {row['optimizer_steps']} | {lexical_cell} | "
            f"{js_cell} | {entropy_cell} |"
        )
    delta = paired_policy_delta(args.policy_per_query)
    if delta:
        markdown.append("")
        markdown.append(
            f"- Paired AP-JSD difference (TRSD64 − Privilege-SD64): "
            f"{delta['mean']:+.5f} [{delta['ci_low']:+.5f}, "
            f"{delta['ci_high']:+.5f}], with {delta['positive']}/{delta['pairs']} "
            "query-level differences positive."
        )
    markdown.extend(
        [
            "",
            "- Lexicon-conditioned style drift is the original realized-token "
            "absolute log-probability shift on the frozen style-word partition.",
            "- AP-JSD is mean full-vocabulary Jensen-Shannon divergence from the "
            "Base policy on identical ordinary-context anchor prefixes, normalized "
            "by log(2) to [0, 1].",
            "- Intervals are query/episode bootstrap 95% confidence intervals.",
            "- Joint reading: TRSD transfers fewer predefined style markers while "
            "moving the overall policy farther from Base; its effect is selective "
            "rather than globally conservative.",
            "",
        ]
    )
    args.output_md.write_text("\n".join(markdown), encoding="utf-8")
    plot_metrics(args.output_figure, rows, delta)
    print(args.output_md)


if __name__ == "__main__":
    main()
