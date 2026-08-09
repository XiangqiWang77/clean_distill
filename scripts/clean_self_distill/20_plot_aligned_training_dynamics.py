#!/usr/bin/env python3
"""Plot evidence-backed Qwen3-8B training dynamics on matched optimizer steps.

The source journals contain one aggregate observation after each optimizer
update.  Privileged-SD and TRSD therefore share the exact x-axis 1..64 here;
we do not expand response-token chunks into synthetic within-update steps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIVILEGED = Path(
    "/home/da839/scratch_pi_mg269/da839/clean_distill/runs/"
    "csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/episodes.jsonl"
)
DEFAULT_TRSD = Path(
    "/home/da839/scratch_pi_mg269/da839/clean_distill/runs/"
    "reverse-kl-matched64-20260807/trsd/train/episodes.jsonl"
)
DEFAULT_ACCURACY = (
    ROOT
    / "docs/experiments/trsd_table_report_20260808/tables/main_accuracy.csv"
)
DEFAULT_FIGURE_DIR = ROOT / "docs/figures/qwen3_8b_training_dynamics"
DEFAULT_RESULT_DIR = ROOT / "docs/results/qwen3_8b_training_dynamics"

COLORS = {
    "Privileged-SD": "#2878B5",
    "TRSD": "#D9534F",
    "Base": "#626262",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--privileged", type=Path, default=DEFAULT_PRIVILEGED)
    parser.add_argument("--trsd", type=Path, default=DEFAULT_TRSD)
    parser.add_argument("--accuracy", type=Path, default=DEFAULT_ACCURACY)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--window", type=int, default=8)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.20,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def finite_float(value: object, name: str, line_no: int) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name} at JSONL line {line_no}: {value}")
    return result


def load_journal(path: Path, method: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not raw.get("optimizer_step", False):
                continue
            step = int(raw["episode"])
            partition = raw["style_task_error"]
            style_count = finite_float(
                partition["style_token_count"], "style_count", line_no
            )
            task_count = finite_float(
                partition["task_token_count"], "task_count", line_no
            )
            if style_count <= 0 or task_count <= 0:
                raise ValueError(f"Invalid token count at {path}:{line_no}")
            rows.append(
                {
                    "method": method,
                    "training_step": step,
                    "query_id": str(raw["query_id"]),
                    "distillation_loss": finite_float(
                        raw["distillation_loss"], "distillation_loss", line_no
                    ),
                    "student_nll": -finite_float(
                        raw["student_normalized_logprob"],
                        "student_normalized_logprob",
                        line_no,
                    ),
                    "teacher_student_kl": finite_float(
                        raw["mean_teacher_student_kl"],
                        "mean_teacher_student_kl",
                        line_no,
                    ),
                    "response_tokens": finite_float(
                        raw["response_tokens"], "response_tokens", line_no
                    ),
                    "style_error_sum": finite_float(
                        partition["style_abs_error_sum"],
                        "style_abs_error_sum",
                        line_no,
                    ),
                    "style_error_count": style_count,
                    "style_error_per_token": finite_float(
                        partition["style_abs_error_sum"],
                        "style_abs_error_sum",
                        line_no,
                    )
                    / style_count,
                    "task_error_sum": finite_float(
                        partition["task_abs_error_sum"],
                        "task_abs_error_sum",
                        line_no,
                    ),
                    "task_error_count": task_count,
                    "task_error_per_token": finite_float(
                        partition["task_abs_error_sum"],
                        "task_abs_error_sum",
                        line_no,
                    )
                    / task_count,
                }
            )
    rows.sort(key=lambda row: int(row["training_step"]))
    expected = list(range(1, 65))
    observed = [int(row["training_step"]) for row in rows]
    if observed != expected:
        raise ValueError(f"{method} must contain exactly matched steps 1..64; got {observed}")
    return rows


def validate_pair(privileged: list[dict[str, object]], trsd: list[dict[str, object]]) -> None:
    privileged_ids = [row["query_id"] for row in privileged]
    trsd_ids = [row["query_id"] for row in trsd]
    if privileged_ids != trsd_ids:
        mismatch = next(
            index + 1
            for index, (left, right) in enumerate(zip(privileged_ids, trsd_ids))
            if left != right
        )
        raise ValueError(f"Matched journals diverge at training step {mismatch}")


def values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    result = np.full(series.shape, np.nan, dtype=np.float64)
    if window <= 0 or window > len(series):
        raise ValueError(f"Window must be in [1, {len(series)}], got {window}")
    result[window - 1 :] = np.convolve(series, np.ones(window) / window, mode="valid")
    return result


def causal_mean_sem(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return descriptive local mean and SEM over preceding observations.

    The band describes within-trajectory update variability. It is intentionally
    not presented as a multi-seed confidence interval.
    """
    means = np.empty(series.shape, dtype=np.float64)
    sems = np.zeros(series.shape, dtype=np.float64)
    for index in range(len(series)):
        local = series[max(0, index - window + 1) : index + 1]
        means[index] = float(np.mean(local))
        if len(local) > 1:
            sems[index] = float(np.std(local, ddof=1) / np.sqrt(len(local)))
    return means, sems


def cumulative_token_microsteps(rows: list[dict[str, object]]) -> np.ndarray:
    response_tokens = values(rows, "response_tokens")
    return np.cumsum(np.ceil(response_tokens / 128.0).astype(np.int64))


def cumulative_ratio(
    rows: list[dict[str, object]], numerator: str, denominator: str
) -> np.ndarray:
    numerators = values(rows, numerator)
    denominators = values(rows, denominator)
    return np.cumsum(numerators) / np.cumsum(denominators)


def weighted_ratio(
    rows: Iterable[dict[str, object]], numerator: str, denominator: str
) -> float:
    subset = list(rows)
    return sum(float(row[numerator]) for row in subset) / sum(
        float(row[denominator]) for row in subset
    )


def phase_rows(
    by_method: dict[str, list[dict[str, object]]], window: int
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for method, rows in by_method.items():
        initial_loss = float(np.mean(values(rows[:window], "distillation_loss")))
        for start in range(0, 64, window):
            block = rows[start : start + window]
            result.append(
                {
                    "method": method,
                    "step_start": start + 1,
                    "step_end": start + len(block),
                    "distillation_loss": float(
                        np.mean(values(block, "distillation_loss"))
                    ),
                    "normalized_objective": float(
                        np.mean(values(block, "distillation_loss")) / initial_loss
                    ),
                    "student_nll": float(np.mean(values(block, "student_nll"))),
                    "teacher_student_kl": float(
                        np.mean(values(block, "teacher_student_kl"))
                    ),
                    "style_error_per_token": weighted_ratio(
                        block, "style_error_sum", "style_error_count"
                    ),
                    "task_error_per_token": weighted_ratio(
                        block, "task_error_sum", "task_error_count"
                    ),
                    "mean_response_tokens": float(
                        np.mean(values(block, "response_tokens"))
                    ),
                }
            )
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        output_path = figure_dir / f"{stem}.{extension}"
        fig.savefig(
            output_path,
            bbox_inches="tight",
            **kwargs,
        )
        if extension == "svg":
            svg_text = output_path.read_text()
            output_path.write_text(
                "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
            )
    plt.close(fig)


def add_step_axis(ax: plt.Axes) -> None:
    ax.set_xlim(1, 64)
    ax.set_xticks([1, 8, 16, 32, 48, 64])
    ax.set_xlabel("Training step (matched optimizer update)")


def plot_objective(
    by_method: dict[str, list[dict[str, object]]], window: int, figure_dir: Path
) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(7.35, 4.55))
    summary: dict[str, float] = {}
    steps = np.arange(1, 65)
    y_lower, y_upper = 0.55, 1.32
    clipped_any = False
    for method, rows in by_method.items():
        raw = values(rows, "distillation_loss")
        baseline = float(np.mean(raw[:window]))
        normalized = raw / baseline
        smooth = rolling_mean(normalized, window)
        color = COLORS[method]
        inside = (normalized >= y_lower) & (normalized <= y_upper)
        ax.scatter(
            steps[inside], normalized[inside], s=13, alpha=0.20,
            color=color, edgecolors="none"
        )
        high = normalized > y_upper
        low = normalized < y_lower
        clipped_any = clipped_any or bool(np.any(high) or np.any(low))
        ax.scatter(
            steps[high], np.full(np.count_nonzero(high), y_upper - 0.012),
            marker="^", s=22, alpha=0.34, color=color, edgecolors="none"
        )
        ax.scatter(
            steps[low], np.full(np.count_nonzero(low), y_lower + 0.012),
            marker="v", s=22, alpha=0.34, color=color, edgecolors="none"
        )
        ax.plot(steps, smooth, lw=2.6, color=color, label=f"{method} · {window}-step mean")
        final_value = float(np.mean(normalized[-window:]))
        summary[f"{method}_final_relative_objective"] = final_value
        summary[f"{method}_objective_reduction_percent"] = 100.0 * (1.0 - final_value)
        ax.annotate(
            f"{final_value:.2f}× early loss",
            xy=(64, final_value),
            xytext=(-8, 10 if method == "TRSD" else -18),
            textcoords="offset points",
            ha="right",
            color=color,
            fontsize=9.3,
            fontweight="bold",
        )
    ax.axhline(1.0, color="#9A9A9A", lw=1.0, ls="--")
    ax.text(1.6, 1.015, "steps 1–8 baseline", color="#777777", fontsize=8.8)
    if clipped_any:
        ax.text(
            0.01, 0.025, "triangles mark raw-update values outside the display range",
            transform=ax.transAxes, color="#777777", fontsize=8.3
        )
    ax.set_ylim(y_lower, y_upper)
    ax.set_ylabel("Within-method normalized objective\n(steps 1–8 = 1.0)")
    add_step_axis(ax)
    ax.set_title("TRSD keeps improving after raw-teacher fitting plateaus", loc="left")
    ax.legend(loc="upper right")
    save_figure(fig, figure_dir, "fig1_objective_convergence")
    return summary


def plot_nll_rebound(
    by_method: dict[str, list[dict[str, object]]],
    phases: list[dict[str, object]],
    window: int,
    figure_dir: Path,
) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(7.35, 4.55))
    steps = np.arange(1, 65)
    summary: dict[str, float] = {}
    ax.axvspan(49, 64, color="#E6B75A", alpha=0.11, lw=0)
    for method, rows in by_method.items():
        raw = values(rows, "student_nll")
        smooth = rolling_mean(raw, window)
        color = COLORS[method]
        ax.scatter(steps, raw, s=13, alpha=0.18, color=color, edgecolors="none")
        ax.plot(steps, smooth, lw=2.6, color=color, label=f"{method} · {window}-step mean")
        method_phases = [row for row in phases if row["method"] == method]
        best = min(method_phases, key=lambda row: float(row["student_nll"]))
        final = method_phases[-1]
        rebound = 100.0 * (
            float(final["student_nll"]) / float(best["student_nll"]) - 1.0
        )
        summary[f"{method}_best_block_start"] = float(best["step_start"])
        summary[f"{method}_best_block_end"] = float(best["step_end"])
        summary[f"{method}_best_block_nll"] = float(best["student_nll"])
        summary[f"{method}_final_block_nll"] = float(final["student_nll"])
        summary[f"{method}_late_nll_rebound_percent"] = rebound
        center = 0.5 * (float(best["step_start"]) + float(best["step_end"]))
        ax.scatter([center, 60.5], [best["student_nll"], final["student_nll"]], s=42, color=color, zorder=5)
        ax.annotate(
            "",
            xy=(60.5, float(final["student_nll"])),
            xytext=(center, float(best["student_nll"])),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.3, "alpha": 0.62},
        )
        y_position = 0.76 if method == "Privileged-SD" else 0.68
        ax.text(
            0.985, y_position, f"{method}: +{rebound:.1f}% from best block",
            transform=ax.transAxes, ha="right", color=color,
            fontsize=9.3, fontweight="bold"
        )
    ax.set_ylim(0.05, 0.255)
    ax.set_ylabel("Student token NLL on training rollouts ↓")
    add_step_axis(ax)
    ax.set_title("Privileged-SD shows the sharper late-stage regression", loc="left")
    ax.legend(loc="lower left")
    save_figure(fig, figure_dir, "fig2_late_stage_nll_rebound")
    return summary


def plot_style_drift(
    by_method: dict[str, list[dict[str, object]]], figure_dir: Path
) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(7.35, 4.55))
    steps = np.arange(1, 65)
    final_values: dict[str, float] = {}
    for method, rows in by_method.items():
        raw = values(rows, "style_error_per_token")
        cumulative = cumulative_ratio(rows, "style_error_sum", "style_error_count")
        final_values[method] = float(cumulative[-1])
        color = COLORS[method]
        ax.scatter(steps, raw, s=12, alpha=0.13, color=color, edgecolors="none")
        ax.plot(steps, cumulative, lw=2.7, color=color, label=f"{method} · cumulative")
        ax.annotate(
            f"{cumulative[-1]:.3f}",
            xy=(64, cumulative[-1]),
            xytext=(-7, 8 if method == "Privileged-SD" else -16),
            textcoords="offset points",
            ha="right",
            color=color,
            fontsize=9.4,
            fontweight="bold",
        )
    reduction = 100.0 * (
        1.0 - final_values["TRSD"] / final_values["Privileged-SD"]
    )
    ax.text(
        0.985,
        0.79,
        f"TRSD: {reduction:.1f}% less\nmarker transfer at step 64",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=COLORS["TRSD"],
        fontweight="bold",
    )
    ax.set_ylabel("Vocabulary style drift ↓\n(absolute log-probability error/token)")
    add_step_axis(ax)
    ax.set_title("Raw privileged style transfer accumulates across training", loc="left")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.68))
    save_figure(fig, figure_dir, "fig3_vocabulary_style_drift")
    return {
        "Privileged-SD_final_style_drift": final_values["Privileged-SD"],
        "TRSD_final_style_drift": final_values["TRSD"],
        "TRSD_style_drift_reduction_percent": reduction,
    }


def plot_phase_heatmap(
    phases: list[dict[str, object]], window: int, figure_dir: Path
) -> list[dict[str, object]]:
    privileged = [row for row in phases if row["method"] == "Privileged-SD"]
    trsd = [row for row in phases if row["method"] == "TRSD"]
    row_specs = [
        ("normalized_objective", "Objective change"),
        ("student_nll", "Student NLL"),
        ("style_error_per_token", "Style drift"),
        ("task_error_per_token", "Task-token error"),
    ]
    matrix = np.zeros((len(row_specs), len(privileged)), dtype=np.float64)
    table: list[dict[str, object]] = []
    for row_index, (key, label) in enumerate(row_specs):
        for column, (p_row, t_row) in enumerate(zip(privileged, trsd)):
            p_value = float(p_row[key])
            t_value = float(t_row[key])
            difference = 100.0 * (t_value - p_value) / p_value
            matrix[row_index, column] = difference
            table.append(
                {
                    "metric": key,
                    "step_start": p_row["step_start"],
                    "step_end": p_row["step_end"],
                    "privileged_value": p_value,
                    "trsd_value": t_value,
                    "trsd_relative_difference_percent": difference,
                }
            )
    bound = max(25.0, float(np.ceil(np.max(np.abs(matrix)) / 5.0) * 5.0))
    norm = mpl.colors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    fig, ax = plt.subplots(figsize=(8.25, 3.65))
    image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row_index, column]
            text_color = "white" if abs(value) > 0.58 * bound else "#202020"
            ax.text(
                column,
                row_index,
                f"{value:+.0f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.2,
                fontweight="bold",
            )
    ax.set_xticks(range(len(privileged)))
    ax.set_xticklabels(
        [f"{row['step_start']}–{row['step_end']}" for row in privileged]
    )
    ax.set_yticks(range(len(row_specs)))
    ax.set_yticklabels([label for _, label in row_specs])
    ax.set_xlabel(f"Training-step window ({window} matched updates)")
    ax.grid(False)
    ax.set_title("TRSD advantage emerges across the full training horizon", loc="left")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.025)
    colorbar.set_label("TRSD relative to Privileged-SD (%)\nblue = lower / better")
    save_figure(fig, figure_dir, "fig4_phase_difference_heatmap")
    return table


def plot_compute_indexed_metric(
    by_method: dict[str, list[dict[str, object]]],
    *,
    metric: str,
    window: int,
    figure_dir: Path,
    stem: str,
    title: str,
    ylabel: str,
    normalize_early: bool = False,
    ylim: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Plot a metric against actual cumulative 128-token training chunks."""
    fig, ax = plt.subplots(figsize=(7.35, 4.55))
    summary: dict[str, float] = {}
    for method, rows in by_method.items():
        x = cumulative_token_microsteps(rows)
        raw = values(rows, metric)
        if normalize_early:
            raw = raw / float(np.mean(raw[:window]))
        local_mean, local_sem = causal_mean_sem(raw, window)
        color = COLORS[method]
        ax.scatter(x, raw, s=12, alpha=0.16, color=color, edgecolors="none")
        ax.fill_between(
            x,
            local_mean - local_sem,
            local_mean + local_sem,
            color=color,
            alpha=0.17,
            linewidth=0,
        )
        ax.plot(
            x,
            local_mean,
            color=color,
            lw=2.5,
            label=f"{method} · mean ± local SEM",
        )
        ax.scatter([x[-1]], [local_mean[-1]], s=38, color=color, zorder=5)
        ax.annotate(
            f"{int(x[-1]):,} microsteps",
            xy=(x[-1], local_mean[-1]),
            xytext=(-6, 9 if method == "TRSD" else -17),
            textcoords="offset points",
            ha="right",
            color=color,
            fontsize=9.1,
            fontweight="bold",
        )
        summary[f"{method}_terminal_microstep"] = int(x[-1])
        summary[f"{method}_terminal_local_mean"] = float(local_mean[-1])
        summary[f"{method}_terminal_local_sem"] = float(local_sem[-1])
    if normalize_early:
        ax.axhline(1.0, color="#929292", lw=1.0, ls="--")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(left=0)
    ax.set_xlabel("Cumulative token-update microstep (128 response tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="best", fontsize=9.2)
    ax.text(
        0.012,
        0.018,
        f"shadow: causal {window}-update SEM within each trajectory",
        transform=ax.transAxes,
        color="#727272",
        fontsize=8.3,
    )
    save_figure(fig, figure_dir, stem)
    return summary


def load_accuracy(path: Path) -> tuple[float, dict[str, dict[int, float]], list[dict[str, object]]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    wanted = {
        "base": ("Base", 0),
        "privileged_16": ("Privileged-SD", 16),
        "trsd_16": ("TRSD", 16),
        "privileged_64": ("Privileged-SD", 64),
        "trsd_64": ("TRSD", 64),
    }
    parsed: dict[str, tuple[str, int, float, int, int]] = {}
    for row in rows:
        method = row["method"]
        if row["dataset"] != "combined" or method not in wanted:
            continue
        family, step = wanted[method]
        parsed[method] = (
            family,
            step,
            float(row["strict_acc1_percent"]),
            int(row["strict_correct"]),
            int(row["n"]),
        )
    missing = sorted(set(wanted) - set(parsed))
    if missing:
        raise ValueError(f"Missing Combined accuracy rows: {missing}")
    base = parsed["base"][2]
    curves = {
        "Privileged-SD": {
            16: parsed["privileged_16"][2],
            64: parsed["privileged_64"][2],
        },
        "TRSD": {
            16: parsed["trsd_16"][2],
            64: parsed["trsd_64"][2],
        },
    }
    table: list[dict[str, object]] = []
    for source_name in wanted:
        family, step, accuracy, correct, total = parsed[source_name]
        table.append(
            {
                "source_method": source_name,
                "method": family,
                "training_step": step,
                "correct": correct,
                "total": total,
                "strict_accuracy_pct": accuracy,
            }
        )
    return base, curves, table


def plot_checkpoint_accuracy(
    base: float,
    curves: dict[str, dict[int, float]],
    figure_dir: Path,
) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(7.35, 4.55))
    x = np.asarray([16, 64])
    ax.axhline(base, color=COLORS["Base"], lw=1.5, ls=(0, (4, 3)), label=f"Base · {base:.2f}%")
    for method, checkpoint_values in curves.items():
        y = np.asarray([checkpoint_values[16], checkpoint_values[64]])
        color = COLORS[method]
        ax.plot(x, y, color=color, lw=2.8, ls=(0, (4, 2)), marker="o", ms=8, label=method)
        for step, accuracy in zip(x, y):
            if method == "Privileged-SD":
                offset = (0, 11) if step == 16 else (0, -18)
            else:
                offset = (0, -25) if step == 16 else (0, 11)
            ax.annotate(
                f"{accuracy:.2f}%",
                xy=(step, accuracy),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                color=color,
                fontweight="bold",
                fontsize=9.5,
            )
    short_edge = curves["Privileged-SD"][16] - curves["TRSD"][16]
    long_edge = curves["TRSD"][64] - curves["Privileged-SD"][64]
    ax.text(
        16,
        75.1,
        f"short horizon\nP-SD +{short_edge:.2f} pp",
        ha="center",
        va="top",
        color=COLORS["Privileged-SD"],
        fontsize=9.5,
    )
    ax.text(
        64,
        75.1,
        f"long horizon\nTRSD +{long_edge:.2f} pp",
        ha="center",
        va="top",
        color=COLORS["TRSD"],
        fontsize=9.5,
    )
    ax.set_ylim(50.0, 76.0)
    ax.set_xlim(8, 72)
    ax.set_xticks([16, 64])
    ax.set_xlabel("Training step (evaluated checkpoints only)")
    ax.set_ylabel("Strict Acc@1 on AIME + AMC (%) ↑")
    ax.set_title("Privileged-SD leads early; TRSD wins the long horizon", loc="left")
    ax.text(45, 59.8, "Privileged-SD", color=COLORS["Privileged-SD"], fontweight="bold")
    ax.text(47, 67.7, "TRSD", color=COLORS["TRSD"], fontweight="bold")
    ax.text(70.3, base + 0.22, f"Base · {base:.2f}%", color=COLORS["Base"], ha="right")
    save_figure(fig, figure_dir, "fig5_checkpoint_accuracy")
    return {
        "base_accuracy_pct": base,
        "Privileged-SD_step16_accuracy_pct": curves["Privileged-SD"][16],
        "TRSD_step16_accuracy_pct": curves["TRSD"][16],
        "Privileged-SD_step64_accuracy_pct": curves["Privileged-SD"][64],
        "TRSD_step64_accuracy_pct": curves["TRSD"][64],
        "Privileged-SD_short_horizon_edge_pp": short_edge,
        "TRSD_long_horizon_edge_pp": long_edge,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_readme(
    result_dir: Path,
    objective: dict[str, float],
    rebound: dict[str, float],
    style: dict[str, float],
    accuracy: dict[str, float],
    compute_indexed: dict[str, dict[str, float]],
) -> None:
    text = f"""# Qwen3-8B aligned training dynamics

All trajectories use the same 64 optimizer updates. Each plotted dot is one
post-update observation from the journal; the curves are causal 8-update means
or prefix-cumulative means. The checkpoint-performance figure contains only the
two checkpoints that were actually evaluated (16 and 64).

## Figures

1. `fig1_objective_convergence`: within-method normalized optimization. TRSD
   reduces its objective by {objective['TRSD_objective_reduction_percent']:.1f}%
   from the first to final 8-step window, versus
   {objective['Privileged-SD_objective_reduction_percent']:.1f}% for
   Privileged-SD.
2. `fig2_late_stage_nll_rebound`: the Privileged-SD student NLL rebounds
   {rebound['Privileged-SD_late_nll_rebound_percent']:.1f}% from its best
   8-step block to steps 57–64; TRSD rebounds
   {rebound['TRSD_late_nll_rebound_percent']:.1f}%.
3. `fig3_vocabulary_style_drift`: the original vocabulary-based diagnostic is
   preserved and shown throughout training. TRSD finishes with
   {style['TRSD_style_drift_reduction_percent']:.1f}% less privileged-marker
   transfer.
4. `fig4_phase_difference_heatmap`: eight matched training windows summarize
   where TRSD's advantage appears across objective, student NLL, style drift,
   and task-token error.
5. `fig5_checkpoint_accuracy`: Privileged-SD leads by
   {accuracy['Privileged-SD_short_horizon_edge_pp']:.2f} pp at step 16, but
   TRSD leads by {accuracy['TRSD_long_horizon_edge_pp']:.2f} pp at step 64.
6. `fig6_compute_indexed_loss`, `fig7_compute_indexed_nll`, and
   `fig8_compute_indexed_style_drift`: the same process is indexed by actual
   cumulative 128-token chunks, reaching
   {compute_indexed['loss']['Privileged-SD_terminal_microstep']:,} microsteps
   for Privileged-SD and
   {compute_indexed['loss']['TRSD_terminal_microstep']:,} for TRSD. Shading is
   the causal 8-update local SEM within each trajectory, not a multi-seed CI.

## Paper-level interpretation

The matched process supports a two-timescale story. Raw privileged distillation
has the short-horizon performance edge, but its fitting signal plateaus and its
student NLL rebounds sharply late in training while privileged-style markers
continue to accumulate. TRSD is deliberately slower at step 16, continues to
improve across the 64-step horizon, and converts that stability into the best
long-horizon strict accuracy.

We operationalize *collapse-like late regression* as the NLL rebound plus
accumulating style drift. Strict accuracy itself does not collapse:
Privileged-SD rises from {accuracy['Privileged-SD_step16_accuracy_pct']:.2f}%
to {accuracy['Privileged-SD_step64_accuracy_pct']:.2f}%, while TRSD rises from
{accuracy['TRSD_step16_accuracy_pct']:.2f}% to
{accuracy['TRSD_step64_accuracy_pct']:.2f}% and overtakes it.

## Reproduction

```bash
python scripts/clean_self_distill/20_plot_aligned_training_dynamics.py
```

PNG, PDF, and SVG versions are in `docs/figures/qwen3_8b_training_dynamics/`.
The exact plotted values and phase-level comparisons are stored beside this
README. Older token-chunk visualizations remain untouched, but these aligned
figures are the appropriate evidence for optimizer-step claims.
"""
    (result_dir / "README.md").write_text(text)


def main() -> None:
    args = parse_args()
    configure_style()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    privileged = load_journal(args.privileged, "Privileged-SD")
    trsd = load_journal(args.trsd, "TRSD")
    validate_pair(privileged, trsd)
    by_method = {"Privileged-SD": privileged, "TRSD": trsd}

    aligned_rows: list[dict[str, object]] = []
    for method_rows in by_method.values():
        for row in method_rows:
            aligned_rows.append({key: value for key, value in row.items() if key != "query_id"})
    write_csv(args.result_dir / "aligned_training_metrics.csv", aligned_rows)

    phases = phase_rows(by_method, args.window)
    write_csv(args.result_dir / "window_summary.csv", phases)

    objective_summary = plot_objective(by_method, args.window, args.figure_dir)
    rebound_summary = plot_nll_rebound(
        by_method, phases, args.window, args.figure_dir
    )
    style_summary = plot_style_drift(by_method, args.figure_dir)
    heatmap_rows = plot_phase_heatmap(phases, args.window, args.figure_dir)
    write_csv(args.result_dir / "phase_difference_heatmap.csv", heatmap_rows)

    compute_indexed = {
        "loss": plot_compute_indexed_metric(
            by_method,
            metric="distillation_loss",
            window=args.window,
            figure_dir=args.figure_dir,
            stem="fig6_compute_indexed_loss",
            title="TRSD converges steadily over thousands of token updates",
            ylabel="Within-method normalized objective ↓",
            normalize_early=True,
            ylim=(0.55, 1.42),
        ),
        "nll": plot_compute_indexed_metric(
            by_method,
            metric="student_nll",
            window=args.window,
            figure_dir=args.figure_dir,
            stem="fig7_compute_indexed_nll",
            title="Privileged-SD regresses after its early NLL minimum",
            ylabel="Student token NLL on training rollouts ↓",
            ylim=(0.05, 0.255),
        ),
        "style_drift": plot_compute_indexed_metric(
            by_method,
            metric="style_error_per_token",
            window=args.window,
            figure_dir=args.figure_dir,
            stem="fig8_compute_indexed_style_drift",
            title="TRSD limits privileged-style drift throughout training",
            ylabel="Vocabulary style drift per token ↓",
            ylim=(0.025, 0.24),
        ),
    }

    base, curves, accuracy_rows = load_accuracy(args.accuracy)
    write_csv(args.result_dir / "checkpoint_accuracy.csv", accuracy_rows)
    accuracy_summary = plot_checkpoint_accuracy(base, curves, args.figure_dir)

    summary: dict[str, object] = {
        "axis_definition": "matched optimizer updates 1..64",
        "smoothing": f"causal {args.window}-update mean",
        "objective": objective_summary,
        "student_nll": rebound_summary,
        "style_drift": style_summary,
        "checkpoint_accuracy": accuracy_summary,
        "compute_indexed": compute_indexed,
        "source_sha256": {
            "privileged_journal": sha256(args.privileged),
            "trsd_journal": sha256(args.trsd),
            "accuracy_table": sha256(args.accuracy),
        },
    }
    (args.result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    build_readme(
        args.result_dir,
        objective_summary,
        rebound_summary,
        style_summary,
        accuracy_summary,
        compute_indexed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
