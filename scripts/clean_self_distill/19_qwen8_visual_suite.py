#!/usr/bin/env python3
"""Build an evidence-linked Qwen3-8B visual suite for the TRSD paper.

The training journals contain one aggregate observation per trajectory.  Each
trajectory is split into fixed-size token chunks for loss computation and
backward passes, so the detailed training coordinate is cumulative chunk
microsteps.  Points mark observed trajectory boundaries.
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter


PRIV = "#3B6FB6"
TRSD = "#D5533D"
BASE = "#737B87"
TEAL = "#168C8C"
GOLD = "#E6A23C"
INK = "#1F2933"
MUTED = "#65717E"
GRID = "#DDE3E8"

DEFAULT_PRIVILEGED = Path(
    "/home/da839/scratch_pi_mg269/da839/clean_distill/runs/"
    "csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/episodes.jsonl"
)
DEFAULT_TRSD = Path(
    "/home/da839/scratch_pi_mg269/da839/clean_distill/runs/"
    "reverse-kl-matched64-20260807/trsd/train/episodes.jsonl"
)
DEFAULT_BUNDLE = Path("docs/experiments/trsd_table_report_20260808")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--privileged-journal", type=Path, default=DEFAULT_PRIVILEGED)
    parser.add_argument("--trsd-journal", type=Path, default=DEFAULT_TRSD)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--apjsd",
        type=Path,
        default=Path("docs/figures/anchored_policy_jsd_per_query.csv"),
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("docs/figures/qwen3_8b")
    )
    parser.add_argument(
        "--table-dir", type=Path, default=Path("docs/results/qwen3_8b_visual_suite")
    )
    return parser.parse_args()


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
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty evidence table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_journal(label: str, path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing journal {path}")
    rows: list[dict[str, Any]] = []
    cumulative = 0
    update = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("optimizer_step") is not True:
                continue
            update += 1
            chunk_size = int(raw["distill_token_chunk_size"])
            response_tokens = int(raw["response_tokens"])
            microsteps = math.ceil(response_tokens / chunk_size)
            cumulative += microsteps
            row = {
                "method": label,
                "trajectory": int(raw["episode"]),
                "optimizer_update": update,
                "training_step": cumulative,
                "trajectory_microsteps": microsteps,
                "chunk_size_tokens": chunk_size,
                "response_tokens": response_tokens,
                "distillation_loss": float(raw["distillation_loss"]),
                "teacher_student_reverse_kl": float(raw["mean_teacher_student_kl"]),
                "student_token_nll": -float(raw["student_normalized_logprob"]),
                "episode_seconds": float(raw["episode_seconds"]),
                "projection_alpha": raw.get("trust_region_alpha", ""),
                "projected_target_kl": raw.get("trust_region_achieved_kl", ""),
                "kl_budget": raw.get("trust_region_kl_budget", ""),
                "query_id": str(raw.get("query_id", "")),
            }
            numeric = [
                row["distillation_loss"],
                row["teacher_student_reverse_kl"],
                row["student_token_nll"],
                row["episode_seconds"],
            ]
            if chunk_size <= 0 or response_tokens <= 0 or not all(
                math.isfinite(float(value)) for value in numeric
            ):
                raise ValueError(f"{path}:{line_number}: invalid training metric")
            rows.append(row)
    if len(rows) != 64:
        raise ValueError(f"{label}: expected 64 successful trajectories, found {len(rows)}")
    chunk_sizes = {int(row["chunk_size_tokens"]) for row in rows}
    if chunk_sizes != {128}:
        raise ValueError(f"{label}: expected 128-token chunks, found {chunk_sizes}")
    return rows


def rolling(values: np.ndarray, window: int = 5) -> np.ndarray:
    return np.asarray(
        [values[max(0, index - window + 1) : index + 1].mean() for index in range(len(values))]
    )


def save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf")
    fig.savefig(directory / f"{stem}.png", dpi=220)
    plt.close(fig)


def panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"{label}  {title}", loc="left")
    ax.grid(color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))


def plot_training_process(
    traces: dict[str, list[dict[str, Any]]], directory: Path
) -> None:
    colors = {"Privilege-SD 64": PRIV, "TRSD 64": TRSD}
    fig, axes = plt.subplots(3, 2, figsize=(13.4, 10.4), constrained_layout=True)
    fig.suptitle(
        "Qwen3-8B training process · 64 trajectories, thousands of chunk-level steps",
        fontsize=17,
        fontweight="bold",
    )

    specs = [
        ("distillation_loss", "A", "Distillation loss", "Trajectory-aggregate loss ↓"),
        (
            "teacher_student_reverse_kl",
            "B",
            "Teacher–student divergence",
            r"Reverse KL  $D_{KL}(\pi_s\,\|\,q)$ ↓",
        ),
        ("student_token_nll", "C", "Student likelihood on the rollout", "Student token NLL ↓"),
        ("response_tokens", "D", "Training-response length", "Response tokens"),
        ("episode_seconds", "E", "Wall-clock cost per trajectory", "Seconds"),
    ]
    for ax, (key, letter, title, ylabel) in zip(axes.flat[:5], specs):
        for label, rows in traces.items():
            x = np.asarray([row["training_step"] for row in rows], dtype=float)
            y = np.asarray([row[key] for row in rows], dtype=float)
            color = colors[label]
            ax.scatter(x, y, s=15, color=color, alpha=0.24, edgecolor="none")
            ax.plot(x, rolling(y), color=color, linewidth=2.25, label=f"{label} · 5-trajectory mean")
        panel(ax, letter, title)
        ax.set_xlabel("Training step (128-token distillation chunk)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        if key != "student_token_nll":
            ax.set_ylim(bottom=0)

    loss_ax = axes.flat[0]
    annotations = []
    for label, rows in traces.items():
        values = np.asarray([row["distillation_loss"] for row in rows])
        change = 100 * (values[-16:].mean() / values[:16].mean() - 1)
        annotations.append(f"{label}: {change:+.1f}% (first→last 16)")
    loss_ax.text(
        0.98,
        0.95,
        "\n".join(annotations),
        transform=loss_ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID, "alpha": 0.92},
    )
    axes.flat[0].legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes.flat[5]
    trsd = traces["TRSD 64"]
    x = np.asarray([row["training_step"] for row in trsd], dtype=float)
    achieved = np.asarray([float(row["projected_target_kl"]) for row in trsd])
    alpha = np.asarray([float(row["projection_alpha"]) for row in trsd])
    budget = float(trsd[0]["kl_budget"])
    ax.scatter(x, achieved, s=15, color=TRSD, alpha=0.3, edgecolor="none")
    ax.plot(x, rolling(achieved), color=TRSD, linewidth=2.25, label="Projected KL")
    ax.axhline(budget, color=TRSD, linestyle="--", linewidth=1.2, label=f"Budget ε={budget:.3f}")
    ax.set_xlabel("Training step (128-token distillation chunk)")
    ax.set_ylabel("Projected target KL", color=TRSD)
    ax.tick_params(axis="y", labelcolor=TRSD)
    ax.set_xlim(left=0)
    panel(ax, "F", "TRSD trust-region activity")
    twin = ax.twinx()
    twin.plot(x, rolling(alpha), color=TEAL, linewidth=1.8, label="Projection α")
    twin.set_ylabel("Projection α", color=TEAL)
    twin.tick_params(axis="y", labelcolor=TEAL)
    twin.spines["right"].set_visible(True)
    twin.set_ylim(0, 1.0)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=9, loc="lower right")

    totals = ", ".join(
        f"{label}: {rows[-1]['training_step']:,} steps"
        for label, rows in traces.items()
    )
    fig.text(
        0.5,
        -0.012,
        f"{totals}. Each point is a trajectory aggregate at its exact cumulative chunk boundary.",
        ha="center",
        color=MUTED,
        fontsize=9.2,
    )
    save(fig, directory, "fig1_training_process_microsteps")


def load_apjsd(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        entry = grouped.setdefault(row["query_id"], {"query_id": row["query_id"], "source": row["source"]})
        entry[row["method"]] = float(row["normalized_jsd"])
    rows = [
        {
            "query_id": value["query_id"],
            "source": value["source"],
            "privileged_sd_64_apjsd": value["Privilege-SD 64"],
            "trsd_64_apjsd": value["TRSD 64"],
            "paired_delta": value["TRSD 64"] - value["Privilege-SD 64"],
        }
        for value in grouped.values()
        if {"Privilege-SD 64", "TRSD 64"} <= value.keys()
    ]
    if len(rows) != 32:
        raise ValueError(f"Expected 32 paired AP-JSD anchors, found {len(rows)}")
    return rows


def plot_apjsd_scatter(rows: list[dict[str, Any]], directory: Path) -> None:
    colors = {"amc23": TEAL, "aime24": GOLD, "aime25": TRSD}
    labels = {"amc23": "AMC23", "aime24": "AIME24", "aime25": "AIME25"}
    fig, ax = plt.subplots(figsize=(7.6, 6.8), constrained_layout=True)
    for source in colors:
        subset = [row for row in rows if row["source"] == source]
        ax.scatter(
            [row["privileged_sd_64_apjsd"] for row in subset],
            [row["trsd_64_apjsd"] for row in subset],
            s=62,
            color=colors[source],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
            label=labels[source],
        )
    maximum = max(max(row["privileged_sd_64_apjsd"], row["trsd_64_apjsd"]) for row in rows) * 1.07
    ax.plot([0, maximum], [0, maximum], color=BASE, linestyle="--", linewidth=1.3, label="Equal policy drift")
    positive = sum(row["paired_delta"] > 0 for row in rows)
    mean_delta = float(np.mean([row["paired_delta"] for row in rows]))
    ax.text(
        0.04,
        0.95,
        f"{positive}/{len(rows)} anchors above identity\nMean paired Δ = {mean_delta:+.4f}",
        transform=ax.transAxes,
        va="top",
        color=TRSD,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID, "alpha": 0.94},
    )
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, maximum)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Privilege-SD 64 · AP-JSD from Base")
    ax.set_ylabel("TRSD 64 · AP-JSD from Base")
    ax.set_title("Qwen3-8B policy movement is selective, not globally smaller", loc="left")
    ax.grid(color=GRID, linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    save(fig, directory, "fig2_paired_apjsd_scatter")


def plot_loss_violin(traces: dict[str, list[dict[str, Any]]], directory: Path) -> None:
    labels = list(traces)
    colors = [PRIV, TRSD]
    values = [np.asarray([row["distillation_loss"] for row in traces[label]]) for label in labels]
    fig, ax = plt.subplots(figsize=(7.7, 6.1), constrained_layout=True)
    violin = ax.violinplot(values, positions=[1, 2], widths=0.72, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
    rng = np.random.default_rng(20260809)
    for position, (data, color) in enumerate(zip(values, colors), 1):
        ax.scatter(
            position + rng.normal(0, 0.055, len(data)),
            data,
            s=15,
            color=color,
            alpha=0.45,
            edgecolor="none",
        )
        q1, median, q3 = np.quantile(data, [0.25, 0.5, 0.75])
        ax.vlines(position, q1, q3, color=INK, linewidth=5)
        ax.scatter(position, median, color="white", edgecolor=INK, s=45, zorder=5)
        ax.text(position, data.max() * 1.04, f"median {median:.4f}", ha="center", color=color, fontweight="bold")
    ax.set_xticks([1, 2], labels)
    ax.set_ylabel("Trajectory-aggregate distillation loss")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_title("Qwen3-8B · TRSD concentrates training at lower loss", loc="left")
    save(fig, directory, "fig3_distillation_loss_violin")


def load_eval(bundle: Path) -> list[dict[str, Any]]:
    evidence = bundle / "evidence" / "evaluation"
    specs = [
        ("Privilege-SD 16", "privileged_sd_16.scored.jsonl"),
        ("TRSD 16", "trsd_16.scored.jsonl"),
        ("Privilege-SD 64", "privileged_sd_64.scored.jsonl"),
        ("TRSD 64", "trsd_64.scored.jsonl"),
    ]
    rows: list[dict[str, Any]] = []
    for label, filename in specs:
        path = evidence / filename
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(
                    {
                        "method": label,
                        "query_id": row["query_id"],
                        "source": row["source"],
                        "strict_correct": int(bool(row["correct"])),
                        "generated_tokens": int(row["generated_tokens"]),
                        "budget_cap_hit": int(bool(row["truncated"])),
                    }
                )
                count += 1
        if count != 143:
            raise ValueError(f"{label}: expected 143 evaluation rows, found {count}")
    return rows


def plot_response_box(eval_rows: list[dict[str, Any]], directory: Path) -> None:
    methods = ["Privilege-SD 16", "TRSD 16", "Privilege-SD 64", "TRSD 64"]
    colors = ["#7DA2D4", "#E58D7E", PRIV, TRSD]
    data = [np.asarray([row["generated_tokens"] for row in eval_rows if row["method"] == method]) for method in methods]
    fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
    boxes = ax.boxplot(
        data,
        positions=np.arange(1, 5),
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.8},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
    )
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.36)
        box.set_edgecolor(color)
    rng = np.random.default_rng(20260809)
    for position, (values, color, method) in enumerate(zip(data, colors, methods), 1):
        ax.scatter(
            position + rng.normal(0, 0.065, len(values)),
            values,
            s=10,
            color=color,
            alpha=0.25,
            edgecolor="none",
        )
        cap_hits = sum(row["budget_cap_hit"] for row in eval_rows if row["method"] == method)
        ax.text(position, 10480, f"{cap_hits} cap hits", ha="center", va="bottom", color=color, fontweight="bold")
    ax.axhline(10240, color=BASE, linestyle="--", linewidth=1.2, label="10,240-token evaluation cap")
    ax.set_xticks(np.arange(1, 5), ["P16", "T16", "P64", "T64"])
    ax.set_ylabel("Generated tokens per evaluation query")
    ax.set_ylim(0, 11200)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Qwen3-8B · Long-horizon TRSD improves completion control", loc="left")
    save(fig, directory, "fig4_evaluation_length_boxplot")


def plot_transition_donut(bundle: Path, directory: Path) -> list[dict[str, Any]]:
    row = read_csv(bundle / "tables" / "transition_anatomy.csv")[0]
    wins = int(row["wrong_to_correct"])
    losses = int(row["correct_to_wrong"])
    rescues = int(row["p64_cap_hit_to_t64_correct"])
    other_wins = wins - rescues
    evidence = [
        {"transition": "P64 wrong → T64 correct", "count": wins},
        {"transition": "P64 correct → T64 wrong", "count": losses},
        {"transition": "Completion rescue within favorable transitions", "count": rescues},
        {"transition": "Other favorable transitions", "count": other_wins},
    ]
    fig, ax = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    ax.pie(
        [wins, losses],
        radius=1.0,
        colors=[TRSD, BASE],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.28, "edgecolor": "white"},
        labels=[f"TRSD wins\n{wins}", f"Privilege-SD wins\n{losses}"],
        labeldistance=1.12,
        textprops={"fontweight": "bold"},
    )
    ax.pie(
        [rescues, other_wins, losses],
        radius=0.68,
        colors=[TEAL, GOLD, "#BCC3CB"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.27, "edgecolor": "white"},
    )
    ax.text(0, 0.04, "+12", ha="center", va="center", fontsize=26, color=TRSD, fontweight="bold")
    ax.text(0, -0.16, "net solved", ha="center", va="center", color=MUTED)
    ax.text(
        0,
        -1.25,
        f"Inner ring · {rescues} completion rescues · {other_wins} other TRSD wins · {losses} reverse transitions",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    ax.set_title("Qwen3-8B · Paired P64→T64 outcome anatomy", loc="left")
    save(fig, directory, "fig5_transition_donut")
    return evidence


def plot_accuracy_heatmap(bundle: Path, directory: Path) -> list[dict[str, Any]]:
    table = read_csv(bundle / "tables" / "main_accuracy.csv")
    methods = ["privileged_16", "trsd_16", "privileged_64", "trsd_64"]
    method_labels = ["Privilege-SD 16", "TRSD 16", "Privilege-SD 64", "TRSD 64"]
    datasets = ["amc23", "aime24", "aime25", "combined"]
    dataset_labels = ["AMC23", "AIME24", "AIME25", "Combined"]
    lookup = {(row["method"], row["dataset"]): row for row in table}
    matrix = np.asarray(
        [[float(lookup[(method, dataset)]["delta_vs_base_percentage_points"]) for dataset in datasets] for method in methods]
    )
    evidence = [
        {
            "method": method_label,
            "dataset": dataset_label,
            "delta_vs_base_percentage_points": matrix[row_index, column_index],
            "strict_acc1_percent": float(lookup[(method, dataset)]["strict_acc1_percent"]),
        }
        for row_index, (method, method_label) in enumerate(zip(methods, method_labels))
        for column_index, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels))
    ]
    limit = max(abs(matrix.min()), abs(matrix.max()))
    fig, ax = plt.subplots(figsize=(9.5, 5.7), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if abs(value) > limit * 0.55 else INK
            ax.text(column_index, row_index, f"{value:+.2f} pp", ha="center", va="center", color=color, fontweight="bold")
    ax.set_xticks(np.arange(4), dataset_labels)
    ax.set_yticks(np.arange(4), method_labels)
    ax.set_title("Qwen3-8B · Accuracy movement relative to Base", loc="left")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("Strict Acc@1 change (percentage points)")
    save(fig, directory, "fig6_accuracy_delta_heatmap")
    return evidence


def write_guide(
    directory: Path,
    traces: dict[str, list[dict[str, Any]]],
    apjsd: list[dict[str, Any]],
) -> None:
    positive = sum(row["paired_delta"] > 0 for row in apjsd)
    guide = f"""# Qwen3-8B visual suite

This suite separates **training dynamics**, **policy movement**, **distributional behavior**, **paired outcome mechanism**, and **benchmark generalization**. Every figure is backed by a machine-readable CSV in `docs/results/qwen3_8b_visual_suite/`.

## Figure 1 — Training process over detailed training steps

![Training process](fig1_training_process_microsteps.png)

The x-axis is cumulative 128-token distillation chunks: **{traces['Privilege-SD 64'][-1]['training_step']:,} steps** for Privilege-SD and **{traces['TRSD 64'][-1]['training_step']:,} steps** for TRSD, across 64 outer-loop trajectories each. Every marker is a trajectory aggregate placed at its exact end-of-trajectory microstep boundary. The loss panel shows the main optimization result: TRSD's last-16 mean is 17.1% below its first-16 mean, while Privilege-SD is essentially flat (+1.1%). The remaining panels connect this descent to divergence, likelihood, response length, runtime, and active trust-region projection.

**Caption.** *Qwen3-8B training dynamics over cumulative 128-token distillation microsteps. Thin markers are trajectory aggregates at exact microstep boundaries and solid curves are five-trajectory moving means. TRSD spans 3,413 chunk-level steps and reduces mean loss by 17.1% from the first to last trajectory quartile, while maintaining the projected target near ε=0.004.*

## Figure 2 — Paired AP-JSD scatter

![Paired AP-JSD](fig2_paired_apjsd_scatter.png)

TRSD lies above the identity line on **{positive}/{len(apjsd)}** fixed-prefix anchors. Read jointly with the lower predefined-vocabulary drift: TRSD changes the policy more broadly while transferring fewer privileged style markers. This is selective adaptation, not blanket conservatism.

## Figure 3 — Loss violin

![Loss violin](fig3_distillation_loss_violin.png)

The violin exposes the complete 64-trajectory loss distribution rather than only a mean curve. TRSD's density is lower and tighter, showing that the training-process result is distributional rather than driven by a few late points.

## Figure 4 — Evaluation-length boxplot

![Length boxplot](fig4_evaluation_length_boxplot.png)

The 143-query distributions show the completion mechanism. At 64 episodes, TRSD cuts evaluation cap hits from 43 to 25 while raising strict Acc@1 from 62.94% to 71.33%.

## Figure 5 — Paired-transition donut

![Transition donut](fig5_transition_donut.png)

Among the 20 queries where P64 and T64 disagree, TRSD wins 16 and loses 4. Eleven favorable transitions are completion rescues, tying the aggregate accuracy gain to an interpretable query-level mechanism.

## Figure 6 — Accuracy heatmap

![Accuracy heatmap](fig6_accuracy_delta_heatmap.png)

The heatmap reports percentage-point change from Base on every dataset and horizon. The short horizon remains close to Base; the 64-episode TRSD branch is positive on AMC23, AIME24, AIME25, and Combined.
"""
    (directory / "FIGURE_GUIDE.md").write_text(guide, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    privileged = load_journal("Privilege-SD 64", args.privileged_journal)
    trsd = load_journal("TRSD 64", args.trsd_journal)
    traces = {"Privilege-SD 64": privileged, "TRSD 64": trsd}
    if privileged[-1]["training_step"] != 1932 or trsd[-1]["training_step"] != 3413:
        raise ValueError(
            "Unexpected microstep totals: "
            f"Privilege-SD={privileged[-1]['training_step']}, TRSD={trsd[-1]['training_step']}"
        )
    training_rows = [row for rows in traces.values() for row in rows]
    write_csv(args.table_dir / "training_process.csv", training_rows)
    plot_training_process(traces, args.figure_dir)

    apjsd = load_apjsd(args.apjsd)
    write_csv(args.table_dir / "paired_apjsd.csv", apjsd)
    plot_apjsd_scatter(apjsd, args.figure_dir)
    plot_loss_violin(traces, args.figure_dir)

    eval_rows = load_eval(args.bundle)
    write_csv(args.table_dir / "evaluation_per_query.csv", eval_rows)
    plot_response_box(eval_rows, args.figure_dir)
    transitions = plot_transition_donut(args.bundle, args.figure_dir)
    write_csv(args.table_dir / "transition_composition.csv", transitions)
    accuracy = plot_accuracy_heatmap(args.bundle, args.figure_dir)
    write_csv(args.table_dir / "accuracy_delta.csv", accuracy)
    write_guide(args.figure_dir, traces, apjsd)
    print(
        json.dumps(
            {
                "figures": str(args.figure_dir),
                "tables": str(args.table_dir),
                "qwen3_8b_training_steps": {
                    "Privilege-SD 64": privileged[-1]["training_step"],
                    "TRSD 64": trsd[-1]["training_step"],
                },
                "outer_trajectories_per_method": 64,
                "observation_unit": "trajectory aggregate at exact cumulative chunk boundary",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
