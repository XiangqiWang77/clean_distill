#!/usr/bin/env python3
"""Build reviewer-facing Qwen3-8B ablation and rollout-shift figures.

The four-panel ablation figure compares only completed projection variants
that share the same 64-query DeepMath stream and 10,240-token rollout cap.
The distribution figure separates correct and incorrect on-policy rollouts
for the raw privileged target and the projected TRSD target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.opsd_format import extract_boxed_answer, grade_boxed_answer  # noqa: E402


RUNS = Path("/home/da839/scratch_pi_mg269/da839/clean_distill/runs")
ABLATION_ROOT = RUNS / "qwen3-8b-component-ablations64-20260809"
DEFAULT_TRSD = RUNS / "reverse-kl-matched64-20260807/trsd/train/episodes.jsonl"
DEFAULT_PRIVILEGED = (
    RUNS / "csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/episodes.jsonl"
)
DEFAULT_TOKENWISE = ABLATION_ROOT / "train/independent_token_budgets/episodes.jsonl"
DEFAULT_FIXED = ABLATION_ROOT / "train/fixed_global_alpha/episodes.jsonl"
DEFAULT_TOKENWISE_EVAL = ABLATION_ROOT / "eval/independent_token_budgets/scored.jsonl"
DEFAULT_FIXED_EVAL = ABLATION_ROOT / "eval/fixed_global_alpha/scored.jsonl"
DEFAULT_LABELS = (
    RUNS
    / "csd-qwen3-8b-three-sellpoints-poc-07/prepared/distill_labels.sealed.jsonl"
)
DEFAULT_ACCURACY = (
    RUNS
    / "qwen3-8b-intermediate-checkpoints-20260809/report/"
    "qwen3_8b_intermediate_checkpoints.csv"
)
DEFAULT_OUTPUT = ROOT / "docs/figures/qwen3_8b_ablation_rollout_shift"

BASE = "#6B7280"
PRIVILEGED = "#D55E00"
TRSD = "#0072B2"
TOKENWISE = "#009E73"
FIXED = "#8E63B0"
GOLD = "#E69F00"
RED = "#B91C1C"
INK = "#172033"
GRID = "#CBD5E1"
METHOD_COLORS = {
    "TRSD (adaptive trajectory)": TRSD,
    "Tokenwise budget $\\alpha_t$": TOKENWISE,
    "Fixed global $\\alpha$": FIXED,
}
TRANSITION_COLORS = {
    "W→W": BASE,
    "W→C": TOKENWISE,
    "C→W": RED,
    "C→C": TRSD,
}
EXPECTED_EPISODES = list(range(1, 65))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trsd-journal", type=Path, default=DEFAULT_TRSD)
    parser.add_argument("--privileged-journal", type=Path, default=DEFAULT_PRIVILEGED)
    parser.add_argument("--tokenwise-journal", type=Path, default=DEFAULT_TOKENWISE)
    parser.add_argument("--fixed-journal", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--deepmath-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--accuracy", type=Path, default=DEFAULT_ACCURACY)
    parser.add_argument("--tokenwise-eval", type=Path, default=DEFAULT_TOKENWISE_EVAL)
    parser.add_argument("--fixed-eval", type=Path, default=DEFAULT_FIXED_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} is empty or malformed")
    return rows


def finite_float(value: object, *, field: str, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: {field} is not finite")
    return result


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for row in read_jsonl(path):
        query_id = str(row.get("query_id", ""))
        answer = str(row.get("answer", ""))
        problem = str(row.get("problem_sha256", "")).lower()
        if not query_id or not answer or len(problem) != 64 or query_id in labels:
            raise ValueError(f"Malformed or duplicate label {query_id!r} in {path}")
        labels[query_id] = {"answer": answer, "problem_sha256": problem}
    return labels


def rollout_correct(row: dict[str, Any], labels: dict[str, dict[str, str]]) -> int:
    query_id = str(row.get("query_id", ""))
    label = labels.get(query_id)
    if label is None:
        raise ValueError(f"No sealed label for {query_id!r}")
    extracted = extract_boxed_answer(str(row.get("student_prefix", "")))
    return int(grade_boxed_answer(extracted, label["answer"]))


def load_journal(
    path: Path,
    method: str,
    labels: dict[str, dict[str, str]],
    *,
    target_kind: str,
) -> list[dict[str, Any]]:
    raw_rows = read_jsonl(path)
    episodes = [int(row.get("episode", -1)) for row in raw_rows]
    if episodes != EXPECTED_EPISODES:
        raise ValueError(f"{path} must contain the exact episode sequence 1..64")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        episode = int(raw["episode"])
        context = f"{path}:episode={episode}"
        query_id = str(raw.get("query_id", ""))
        label = labels.get(query_id)
        if label is None or str(raw.get("problem_sha256", "")).lower() != label["problem_sha256"]:
            raise ValueError(f"{context}: query does not match the sealed label")
        if raw.get("optimizer_step") is not True or str(raw.get("source", "")).lower() != "deepmath":
            raise ValueError(f"{context}: expected an optimized DeepMath trajectory")

        style = raw.get("style_task_error")
        if not isinstance(style, dict):
            raise ValueError(f"{context}: missing style/task partition")
        style_count = int(style.get("style_token_count", 0))
        style_sum = finite_float(style.get("style_abs_error_sum"), field="style sum", context=context)
        if style_count <= 0:
            raise ValueError(f"{context}: style-token set is empty")

        if target_kind == "projected":
            target_kl = finite_float(
                raw.get("trust_region_achieved_kl"), field="projected target KL", context=context
            )
        elif target_kind == "raw":
            target_kl = finite_float(
                raw.get("mean_teacher_student_kl"), field="raw target KL", context=context
            )
        else:
            raise ValueError(f"Unsupported target kind {target_kind!r}")

        normalized_logprob = finite_float(
            raw.get("student_normalized_logprob"), field="student normalized logprob", context=context
        )
        response_tokens = int(raw.get("response_tokens", 0))
        if normalized_logprob > 0 or response_tokens <= 0 or target_kl <= 0:
            raise ValueError(f"{context}: invalid trajectory metrics")

        rows.append(
            {
                "method": method,
                "target_kind": target_kind,
                "episode": episode,
                "query_id": query_id,
                "problem_sha256": label["problem_sha256"],
                "entropy_proxy_nats_per_token": -normalized_logprob,
                "response_tokens": response_tokens,
                "reward": rollout_correct(raw, labels),
                "target_kl": target_kl,
                "style_drift": style_sum / style_count,
                "style_error_sum": style_sum,
                "style_token_count": style_count,
                "projection_alpha": (
                    finite_float(raw["trust_region_alpha"], field="projection alpha", context=context)
                    if raw.get("trust_region_alpha") is not None
                    else None
                ),
            }
        )
    return rows


def validate_matched_streams(groups: Iterable[list[dict[str, Any]]]) -> None:
    streams = [[(row["query_id"], row["problem_sha256"]) for row in rows] for rows in groups]
    if any(stream != streams[0] for stream in streams[1:]):
        raise ValueError("Training journals do not use the same ordered DeepMath stream")


def strict_eval_summary(path: Path) -> dict[str, float]:
    rows = read_jsonl(path)
    if len(rows) != 143 or len({str(row.get("query_id", "")) for row in rows}) != 143:
        raise ValueError(f"{path}: expected 143 unique frozen-evaluation rows")
    counts: dict[str, tuple[int, int]] = {}
    for source, expected in (("amc23", 83), ("aime24", 30), ("aime25", 30)):
        selected = [row for row in rows if str(row.get("source", "")).lower() == source]
        if len(selected) != expected:
            raise ValueError(f"{path}: {source} has {len(selected)} rows")
        correct = sum(
            bool(row.get("correct"))
            and not bool(
                row.get("truncated", row.get("behavioral_diagnostics", {}).get("truncated", False))
            )
            for row in selected
        )
        counts[source] = (int(correct), expected)
    combined_correct = sum(value[0] for value in counts.values())
    return {
        "AMC23": 100.0 * counts["amc23"][0] / counts["amc23"][1],
        "AIME24": 100.0 * counts["aime24"][0] / counts["aime24"][1],
        "AIME25": 100.0 * counts["aime25"][0] / counts["aime25"][1],
        "Combined": 100.0 * combined_correct / 143,
    }


def load_registered_accuracy(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("model") != "Qwen3-8B" or int(row.get("episodes", -1)) not in {0, 64}:
            continue
        method = str(row.get("method", ""))
        if method not in {"Base", "TRSD"}:
            continue
        output[method] = {
            "AMC23": float(row["amc23_strict_acc1_percent"]),
            "AIME24": float(row["aime24_strict_acc1_percent"]),
            "AIME25": float(row["aime25_strict_acc1_percent"]),
            "Combined": float(row["combined_strict_acc1_percent"]),
        }
    if set(output) != {"Base", "TRSD"}:
        raise ValueError(f"{path}: missing registered Base/TRSD-64 accuracy")
    return output


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if not 1 <= window <= len(values):
        raise ValueError(f"window must lie in [1, {len(values)}]")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.42,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_all(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_dynamics(
    dynamics: dict[str, list[dict[str, Any]]],
    accuracy: dict[str, dict[str, float]],
    *,
    window: int,
) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 3.75))
    specs = (
        (axes[0], "entropy_proxy_nats_per_token", 1.0),
        (axes[1], "response_tokens", 1000.0),
        (axes[2], "reward", 1.0),
    )
    for ax, field, scale in specs:
        for method, rows in dynamics.items():
            steps = np.asarray([row["episode"] for row in rows], dtype=float)
            values = np.asarray([row[field] for row in rows], dtype=float) / scale
            color = METHOD_COLORS[method]
            ax.plot(steps, values, color=color, alpha=0.12, linewidth=0.85)
            ax.plot(
                steps,
                rolling_mean(values, window),
                color=color,
                linewidth=2.35,
                label=method,
                solid_capstyle="round",
            )
        ax.set_xlim(0, 65)
        ax.set_xticks([0, 16, 32, 48, 64])
        ax.set_xlabel("Training step")

    axes[0].set_title("(a) Math: entropy")
    axes[0].set_ylabel("Realized-token surprisal (nats/token)")
    axes[1].set_title("(b) Math: length")
    axes[1].set_ylabel("Response length (k tokens)")
    axes[1].set_ylim(0, 10.8)
    axes[2].set_title("(c) Math: reward")
    axes[2].set_ylabel("Frozen verifier reward")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_yticks([0.0, 0.5, 1.0])

    ax = axes[3]
    names = ["Base", "Tokenwise\n$\\alpha_t$", "Fixed\n$\\alpha$", "Adaptive\nTRSD"]
    keys = ["Base", "Tokenwise budget $\\alpha_t$", "Fixed global $\\alpha$", "TRSD"]
    colors = [BASE, TOKENWISE, FIXED, TRSD]
    values = [accuracy[key]["Combined"] for key in keys]
    bars = ax.bar(np.arange(4), values, width=0.67, color=colors, zorder=2)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.1,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )
    ax.set_title("(d) Math: accuracy")
    ax.set_ylabel("Strict Acc@1 (%)")
    ax.set_xticks(np.arange(4), names)
    ax.set_ylim(0, 80)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=3,
        handlelength=2.6,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.20, top=0.80, wspace=0.32)
    return fig


def raincloud(
    ax: plt.Axes,
    groups: list[tuple[str, np.ndarray, str]],
    *,
    ylabel: str,
    log_y: bool = False,
    reference: float | None = None,
) -> None:
    rng = np.random.default_rng(20260810)
    positions = np.arange(len(groups), dtype=float)
    values = [group[1] for group in groups]
    violins = ax.violinplot(values, positions=positions, widths=0.78, showextrema=False)
    for body, (_, _, color) in zip(violins["bodies"], groups, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.16)
        body.set_linewidth(0.8)
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.5},
        whiskerprops={"color": INK, "linewidth": 0.9},
        capprops={"color": INK, "linewidth": 0.9},
    )
    for patch, (_, _, color) in zip(box["boxes"], groups, strict=True):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.25)
    for position, (label, group, color) in zip(positions, groups, strict=True):
        jitter = rng.uniform(-0.16, 0.16, size=len(group))
        ax.scatter(
            position + jitter,
            group,
            s=17,
            color=color,
            alpha=0.52,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        ax.text(
            position,
            0.985,
            f"n={len(group)}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=INK,
        )
    if reference is not None:
        ax.axhline(reference, color=TRSD, linestyle="--", linewidth=1.15, alpha=0.85)
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(positions, [group[0] for group in groups])
    ax.set_ylabel(ylabel)


def transition_label(raw_reward: int, trsd_reward: int) -> str:
    return ("C" if raw_reward else "W") + "→" + ("C" if trsd_reward else "W")


def plot_distribution(
    privileged: list[dict[str, Any]], trsd: list[dict[str, Any]]
) -> tuple[plt.Figure, list[dict[str, Any]]]:
    configure_style()
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    fig.suptitle(
        "Where projection changes the distillation target: rollout-level distributions",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    def selected(rows: list[dict[str, Any]], reward: int, field: str) -> np.ndarray:
        return np.asarray([row[field] for row in rows if row["reward"] == reward], dtype=float)

    outcome_groups_kl = [
        ("Raw teacher\nIncorrect", selected(privileged, 0, "target_kl"), PRIVILEGED),
        ("Raw teacher\nCorrect", selected(privileged, 1, "target_kl"), PRIVILEGED),
        ("TRSD target\nIncorrect", selected(trsd, 0, "target_kl"), TRSD),
        ("TRSD target\nCorrect", selected(trsd, 1, "target_kl"), TRSD),
    ]
    raincloud(
        axes[0, 0],
        outcome_groups_kl,
        ylabel="Target KL (nats/token) ↓",
        log_y=True,
        reference=0.004,
    )
    axes[0, 0].set_title("(a) Target distance by rollout outcome", loc="left")
    axes[0, 0].text(
        0.98,
        0.70,
        "dashed: $\\epsilon=0.004$",
        transform=axes[0, 0].transAxes,
        ha="right",
        color=TRSD,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    outcome_groups_style = [
        ("Raw teacher\nIncorrect", selected(privileged, 0, "style_drift"), PRIVILEGED),
        ("Raw teacher\nCorrect", selected(privileged, 1, "style_drift"), PRIVILEGED),
        ("TRSD target\nIncorrect", selected(trsd, 0, "style_drift"), TRSD),
        ("TRSD target\nCorrect", selected(trsd, 1, "style_drift"), TRSD),
    ]
    raincloud(
        axes[0, 1],
        outcome_groups_style,
        ylabel="StyleDrift (abs. log-prob./token) ↓",
    )
    axes[0, 1].set_title("(b) Style drift by rollout outcome", loc="left")

    ax = axes[0, 2]
    for method, rows, color in (
        ("Raw privileged target", privileged, PRIVILEGED),
        ("TRSD projected target", trsd, TRSD),
    ):
        for reward, marker, outcome in ((0, "x", "Incorrect"), (1, "o", "Correct")):
            subset = [row for row in rows if row["reward"] == reward]
            ax.scatter(
                [row["target_kl"] for row in subset],
                [row["style_drift"] for row in subset],
                s=36,
                marker=marker,
                color=color,
                alpha=0.68,
                linewidths=0.8,
                label=f"{method} · {outcome}",
            )
    ax.axvline(0.004, color=TRSD, linestyle="--", linewidth=1.1, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlim(0.0035, 0.06)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([0.004, 0.01, 0.03]))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(["0.004", "0.01", "0.03"]))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("Target KL (nats/token) ↓")
    ax.set_ylabel("StyleDrift ↓")
    ax.set_title("(c) Target distance and style drift co-move", loc="left")
    ax.legend(fontsize=7.6, loc="upper left")

    paired: list[dict[str, Any]] = []
    for raw, projected in zip(privileged, trsd, strict=True):
        if raw["query_id"] != projected["query_id"]:
            raise ValueError("Privileged/TRSD streams are not query-matched")
        transition = transition_label(raw["reward"], projected["reward"])
        paired.append(
            {
                "episode": raw["episode"],
                "query_id": raw["query_id"],
                "transition": transition,
                "privileged_reward": raw["reward"],
                "trsd_reward": projected["reward"],
                "raw_target_kl": raw["target_kl"],
                "trsd_target_kl": projected["target_kl"],
                "raw_style_drift": raw["style_drift"],
                "trsd_style_drift": projected["style_drift"],
                "trsd_projection_alpha": projected["projection_alpha"],
            }
        )

    for ax, raw_field, projected_field, ylabel, title, log_y in (
        (
            axes[1, 0],
            "raw_target_kl",
            "trsd_target_kl",
            "Target KL (nats/token) ↓",
            "(d) Query-matched target shift",
            True,
        ),
        (
            axes[1, 1],
            "raw_style_drift",
            "trsd_style_drift",
            "StyleDrift ↓",
            "(e) Query-matched style shift",
            False,
        ),
    ):
        seen: set[str] = set()
        for row in paired:
            transition = str(row["transition"])
            ax.plot(
                [0, 1],
                [row[raw_field], row[projected_field]],
                color=TRANSITION_COLORS[transition],
                alpha=0.27,
                linewidth=0.9,
                label=(
                    f"{transition} (n={sum(item['transition'] == transition for item in paired)})"
                    if transition not in seen
                    else None
                ),
            )
            seen.add(transition)
        medians = [
            float(np.median([row[raw_field] for row in paired])),
            float(np.median([row[projected_field] for row in paired])),
        ]
        ax.plot([0, 1], medians, color=INK, marker="o", linewidth=2.7, markersize=5.5, label="Median")
        ax.set_xticks([0, 1], ["Raw privileged\ntarget", "TRSD projected\ntarget"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        if log_y:
            ax.set_yscale("log")
            ax.axhline(0.004, color=TRSD, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.legend(fontsize=7.7, loc="best")

    ax = axes[1, 2]
    alpha_groups = [
        ("Incorrect", selected(trsd, 0, "projection_alpha"), GOLD),
        ("Correct", selected(trsd, 1, "projection_alpha"), TRSD),
    ]
    raincloud(ax, alpha_groups, ylabel="Adaptive projection $\\alpha$", reference=1.0)
    ax.set_ylim(0, 1.07)
    ax.set_title("(f) Projection strength by TRSD outcome", loc="left")
    ax.text(
        0.98,
        0.06,
        "$\\alpha=1$: unprojected teacher\nconstraint active: 63/64",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=BASE,
        fontsize=8,
    )

    fig.text(
        0.5,
        0.012,
        "64 matched DeepMath queries. Correctness uses the frozen boxed-answer verifier. "
        "Panels (d–e) match queries, but each method retains its own on-policy prefix.",
        ha="center",
        color=BASE,
        fontsize=8.8,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.10, top=0.92, hspace=0.39, wspace=0.30)
    return fig, paired


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pooled_style(rows: list[dict[str, Any]]) -> float:
    return sum(row["style_error_sum"] for row in rows) / sum(row["style_token_count"] for row in rows)


def main() -> None:
    args = parse_args()
    labels = load_labels(args.deepmath_labels)
    trsd = load_journal(
        args.trsd_journal,
        "TRSD (adaptive trajectory)",
        labels,
        target_kind="projected",
    )
    tokenwise = load_journal(
        args.tokenwise_journal,
        "Tokenwise budget $\\alpha_t$",
        labels,
        target_kind="projected",
    )
    fixed = load_journal(
        args.fixed_journal,
        "Fixed global $\\alpha$",
        labels,
        target_kind="projected",
    )
    privileged = load_journal(
        args.privileged_journal,
        "Raw privileged target",
        labels,
        target_kind="raw",
    )
    validate_matched_streams([trsd, tokenwise, fixed, privileged])

    registered = load_registered_accuracy(args.accuracy)
    accuracy = {
        "Base": registered["Base"],
        "TRSD": registered["TRSD"],
        "Tokenwise budget $\\alpha_t$": strict_eval_summary(args.tokenwise_eval),
        "Fixed global $\\alpha$": strict_eval_summary(args.fixed_eval),
    }
    dynamics = {
        "TRSD (adaptive trajectory)": trsd,
        "Tokenwise budget $\\alpha_t$": tokenwise,
        "Fixed global $\\alpha$": fixed,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_all(
        plot_dynamics(dynamics, accuracy, window=args.window),
        args.output_dir,
        "math_ablation_dynamics",
    )
    distribution, paired = plot_distribution(privileged, trsd)
    save_all(distribution, args.output_dir, "rollout_target_shift_distribution")

    write_csv(
        args.output_dir / "math_ablation_dynamics.csv",
        [row for rows in dynamics.values() for row in rows],
    )
    write_csv(
        args.output_dir / "math_ablation_accuracy.csv",
        [
            {"method": method, "dataset": dataset, "strict_acc1_percent": value}
            for method, values in accuracy.items()
            for dataset, value in values.items()
        ],
    )
    write_csv(
        args.output_dir / "rollout_shift_distribution.csv",
        [
            {
                key: row[key]
                for key in (
                    "method",
                    "target_kind",
                    "episode",
                    "query_id",
                    "reward",
                    "target_kl",
                    "style_drift",
                    "projection_alpha",
                    "response_tokens",
                )
            }
            for rows in (privileged, trsd)
            for row in rows
        ],
    )
    write_csv(args.output_dir / "matched_query_shift.csv", paired)

    transition_counts = Counter(str(row["transition"]) for row in paired)
    raw_kl = float(np.mean([row["target_kl"] for row in privileged]))
    projected_kl = float(np.mean([row["target_kl"] for row in trsd]))
    raw_style = pooled_style(privileged)
    projected_style = pooled_style(trsd)
    summary = {
        "schema_version": "qwen3-8b-ablation-rollout-shift-v1",
        "ablation_protocol": {
            "training_source": "DeepMath",
            "matched_queries": 64,
            "rollout_cap_tokens": 10240,
            "smoothing": f"trailing_{args.window}_step_mean",
            "accuracy": "Strict Acc@1 on frozen AMC23+AIME24+AIME25 (143 questions)",
            "methods": list(dynamics),
        },
        "accuracy": accuracy,
        "rollout_correct": {
            "Raw privileged target": sum(row["reward"] for row in privileged),
            "TRSD projected target": sum(row["reward"] for row in trsd),
        },
        "outcome_transitions": dict(sorted(transition_counts.items())),
        "target_kl": {
            "raw_privileged_mean": raw_kl,
            "trsd_projected_mean": projected_kl,
            "relative_reduction": 1.0 - projected_kl / raw_kl,
        },
        "style_drift": {
            "raw_privileged_pooled": raw_style,
            "trsd_projected_pooled": projected_style,
            "relative_reduction": 1.0 - projected_style / raw_style,
        },
        "definitions": {
            "entropy": "negative mean student log-probability of realized rollout tokens (on-policy surprisal proxy)",
            "reward": "frozen boxed-answer verifier reward in {0,1}",
            "target_kl": "per-position target-to-current-student full-vocabulary KL, averaged within each trajectory",
            "style_drift": "absolute target/student realized-token log-probability error on the fixed style-token set",
        },
        "interpretation_guard": (
            "Raw privileged and TRSD rows use the same ordered query stream but method-specific "
            "on-policy prefixes. Query-matched slope panels are distributional diagnostics, not a "
            "same-prefix causal estimate."
        ),
        "inputs": {
            "trsd_journal": str(args.trsd_journal),
            "privileged_journal": str(args.privileged_journal),
            "tokenwise_journal": str(args.tokenwise_journal),
            "fixed_journal": str(args.fixed_journal),
            "accuracy": str(args.accuracy),
            "tokenwise_eval": str(args.tokenwise_eval),
            "fixed_eval": str(args.fixed_eval),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fixed_accuracy = accuracy["Fixed global $\\alpha$"]["Combined"]
    tokenwise_accuracy = accuracy["Tokenwise budget $\\alpha_t$"]["Combined"]
    readme = f"""# Qwen3-8B ablation and rollout-shift figures

## Figure 1: core projection ablations

`math_ablation_dynamics` contains the requested four panels: `(a) Math:
entropy`, `(b) Math: length`, `(c) Math: reward`, and `(d) Math: accuracy`.
Pale curves are per-rollout observations and thick curves are trailing
{args.window}-step means. The dynamics compare only completed variants with the
same 64-query DeepMath stream and 10,240-token rollout cap. Panel (d) uses the
frozen 143-question AMC23/AIME24/AIME25 scorer. The registered full TRSD result
is {accuracy['TRSD']['Combined']:.2f}%, versus
{fixed_accuracy:.2f}% for fixed global alpha and
{tokenwise_accuracy:.2f}% for independent
token budgets.

## Figure 2: correct/incorrect rollout target-shift distributions

`rollout_target_shift_distribution` is a large six-panel distributional
diagnostic. It shows all 64 rollout-level observations for raw privileged and
TRSD targets, split by frozen-verifier correctness, plus target-KL/style-drift
scatter, query-matched slope plots, and the TRSD alpha distribution. Mean
Target KL falls from {raw_kl:.6f} to {projected_kl:.6f}; pooled StyleDrift falls
from {raw_style:.6f} to {projected_style:.6f}. Training-rollout correctness is
{sum(row['reward'] for row in privileged)}/64 versus
{sum(row['reward'] for row in trsd)}/64, with
{transition_counts.get('W→C', 0)} query-level W→C transitions and
{transition_counts.get('C→W', 0)} C→W transitions.

The raw privileged and TRSD journals use the same query order but each method's
own on-policy prefix. Therefore the query-matched slope panels are descriptive
distributional comparisons; they are not labeled as same-prefix causal
measurements. Exact plotted values are in the accompanying CSV files.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
