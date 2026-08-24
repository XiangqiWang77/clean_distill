#!/usr/bin/env python3
"""Build reviewer-safe LGSD tables and figures from saved, lightweight outputs.

This script never loads a model or imports torch.  It only reads saved JSONL/CSV
evaluation records, performs paired bootstrap resampling, and renders figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


YELLOW = "#F2C94C"
BLUE = "#2F80ED"
BLACK = "#161616"
GRAY = "#767676"
LIGHT = "#F4F1E8"
WHITE = "#FFFFFF"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap(
    arrays: dict[str, np.ndarray], *, replicates: int, seed: int
) -> dict[str, tuple[float, float]]:
    """Bootstrap aligned pair-level statistics in bounded-memory batches."""
    names = list(arrays)
    n = len(arrays[names[0]])
    if any(len(values) != n for values in arrays.values()):
        raise ValueError("Bootstrap arrays are not aligned")
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(replicates, dtype=np.float64) for name in names}
    offset = 0
    while offset < replicates:
        width = min(500, replicates - offset)
        indices = rng.integers(0, n, size=(width, n), endpoint=False)
        for name, values in arrays.items():
            draws[name][offset : offset + width] = values[indices].mean(axis=1)
        offset += width
    return {name: percentile_ci(values) for name, values in draws.items()}


def style_axes(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(BLACK)
    axis.spines["bottom"].set_color(BLACK)
    axis.tick_params(colors=BLACK, labelsize=9)
    axis.grid(axis="y", color="#D8D4CA", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)


def bootstrap_arena(
    arena_run_root: Path,
    alpha_story_path: Path,
    output_dir: Path,
    arena_doc_dir: Path,
    *,
    replicates: int,
    seed: int,
) -> list[dict]:
    story = json.loads(alpha_story_path.read_text(encoding="utf-8"))
    metadata = {row["method"]: row for row in story["rows"]}
    score_paths = {
        "Base": arena_run_root / "preference_eval/scores/base/episode_0000.jsonl",
        "LGSD-Small": arena_run_root / "preference_eval/scores/lgsd_small/episode_1000.jsonl",
        "LGSD-Medium": arena_run_root / "preference_eval/scores/lgsd_medium/episode_1000.jsonl",
        "LGSD-Large": arena_run_root / "preference_eval/scores/lgsd_large/episode_1000.jsonl",
        "LGSD-High": arena_run_root / "preference_eval/scores/lgsd_b200_r0040/episode_1000.jsonl",
        "OPSD": arena_run_root / "preference_eval/scores/opsd/episode_1000.jsonl",
    }
    rows_by_method: dict[str, dict[str, dict]] = {}
    for method, path in score_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        keyed = {str(row["query_id"]): row for row in rows}
        if len(keyed) != 600:
            raise ValueError(f"{method}: expected 600 unique pairs, found {len(keyed)}")
        rows_by_method[method] = keyed

    ids = sorted(rows_by_method["Base"])
    if any(sorted(rows) != ids for rows in rows_by_method.values()):
        raise ValueError("Arena score files do not contain the same pair IDs")

    margins = {
        method: np.asarray([rows_by_method[method][qid]["preference_margin"] for qid in ids])
        for method in score_paths
    }
    correct = {method: (values > 0).astype(np.float64) for method, values in margins.items()}
    base = margins["Base"]
    opsd = margins["OPSD"]
    bootstrap_inputs: dict[str, np.ndarray] = {}
    for method in score_paths:
        bootstrap_inputs[f"margin::{method}"] = margins[method]
        bootstrap_inputs[f"acc::{method}"] = correct[method]
        bootstrap_inputs[f"gain::{method}"] = margins[method] - base
        bootstrap_inputs[f"delta_opsd::{method}"] = margins[method] - opsd
    cis = paired_bootstrap(bootstrap_inputs, replicates=replicates, seed=seed)

    order = ["Base", "LGSD-Small", "LGSD-Medium", "LGSD-Large", "LGSD-High", "OPSD"]
    table: list[dict] = []
    for method in order:
        info = metadata[method]
        gain = margins[method] - base
        delta = margins[method] - opsd
        margin_ci = cis[f"margin::{method}"]
        gain_ci = cis[f"gain::{method}"]
        delta_ci = cis[f"delta_opsd::{method}"]
        acc_ci = cis[f"acc::{method}"]
        table.append(
            {
                "method": method,
                "radius": info["radius"],
                "mean_alpha_token": "" if method == "Base" else float(info["mean_alpha_token"]),
                "target_kl_raw": float(info["target_kl_raw"]),
                "pref_margin": float(margins[method].mean()),
                "pref_margin_ci_low": margin_ci[0],
                "pref_margin_ci_high": margin_ci[1],
                "pref_gain_vs_base": float(gain.mean()),
                "pref_gain_ci_low": gain_ci[0],
                "pref_gain_ci_high": gain_ci[1],
                "delta_pref_gain_vs_opsd": float(delta.mean()),
                "delta_opsd_ci_low": delta_ci[0],
                "delta_opsd_ci_high": delta_ci[1],
                "pref_accuracy": float(correct[method].mean()),
                "pref_acc_ci_low": acc_ci[0],
                "pref_acc_ci_high": acc_ci[1],
                "pair_count": len(ids),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
            }
        )

    fields = list(table[0])
    write_csv(output_dir / "table_arena_preference_uncertainty.csv", table, fields)
    write_csv(arena_doc_dir / "arena_preference_main_table.csv", table, fields)
    render_arena_figure(table, arena_doc_dir)
    render_arena_table(table, arena_doc_dir)
    write_arena_latex(table, arena_doc_dir / "arena_preference_main_table.tex")
    return table


def fmt_ci(value: float, low: float, high: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def render_arena_figure(table: list[dict], output_dir: Path) -> None:
    lgsd = [row for row in table if row["method"].startswith("LGSD")]
    opsd = next(row for row in table if row["method"] == "OPSD")
    x = np.asarray([float(row["mean_alpha_token"]) for row in lgsd])
    y = np.asarray([row["pref_gain_vs_base"] for row in lgsd])
    ylow = y - np.asarray([row["pref_gain_ci_low"] for row in lgsd])
    yhigh = np.asarray([row["pref_gain_ci_high"] for row in lgsd]) - y

    fig, axis = plt.subplots(figsize=(6.6, 6.2), facecolor=WHITE)
    axis.set_facecolor(WHITE)
    axis.axhline(0, color=BLACK, linewidth=1.1, linestyle=(0, (4, 3)), label="Frozen Base")
    axis.plot(x, y, color=YELLOW, linewidth=3.0, zorder=2)
    axis.errorbar(
        x,
        y,
        yerr=np.vstack([ylow, yhigh]),
        fmt="o",
        color=BLACK,
        ecolor=BLACK,
        markerfacecolor=YELLOW,
        markeredgecolor=BLACK,
        markersize=8,
        capsize=4,
        linewidth=1.25,
        zorder=3,
        label="LGSD radius sweep",
    )
    opsd_y = opsd["pref_gain_vs_base"]
    axis.errorbar(
        [1.0],
        [opsd_y],
        yerr=[[opsd_y - opsd["pref_gain_ci_low"]], [opsd["pref_gain_ci_high"] - opsd_y]],
        fmt="D",
        color=BLACK,
        ecolor=BLUE,
        markerfacecolor=BLUE,
        markeredgecolor=BLACK,
        markersize=8,
        capsize=4,
        linewidth=1.4,
        zorder=4,
        label="OPSD (raw proposal)",
    )
    labels = ["Small", "Medium", "Large", "High"]
    offsets = [(0, -19), (0, 10), (0, 10), (0, -19)]
    for xi, yi, label, offset in zip(x, y, labels, offsets):
        axis.annotate(label, (xi, yi), xytext=offset, textcoords="offset points", ha="center", fontsize=9)
    axis.annotate("OPSD", (1.0, opsd_y), xytext=(0, -19), textcoords="offset points", ha="center", fontsize=9)
    axis.set_xlim(0.27, 1.06)
    axis.set_ylim(-0.015, 0.205)
    axis.set_xlabel(r"Token-weighted mean projection $\alpha$", fontsize=11, color=BLACK)
    axis.set_ylabel("PrefGain vs frozen Base (mean log-prob/token)", fontsize=11, color=BLACK)
    fig.text(0.105, 0.965, "Held-out human-preference likelihood after 1,000 episodes", fontsize=14, weight="bold", color=BLACK)
    fig.text(0.105, 0.928, "600 LMArena pairs · paired 95% bootstrap CIs · one training seed", fontsize=9.5, color=GRAY)
    style_axes(axis)
    axis.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90), pad=1.4)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig2_locality_tradeoff.{suffix}", dpi=260, bbox_inches="tight")
    plt.close(fig)


def render_arena_table(table: list[dict], output_dir: Path) -> None:
    headers = ["Method", r"Mean $\alpha$", "Target KL/raw", "PrefMargin [95% CI]", "PrefGain [95% CI]", r"$\Delta$ vs OPSD [95% CI]", "PrefAcc [95% CI]"]
    body = []
    for row in table:
        alpha = "—" if row["method"] == "Base" else f"{float(row['mean_alpha_token']):.3f}"
        body.append(
            [
                row["method"],
                alpha,
                f"{row['target_kl_raw']:.3f}",
                fmt_ci(row["pref_margin"], row["pref_margin_ci_low"], row["pref_margin_ci_high"]),
                fmt_ci(row["pref_gain_vs_base"], row["pref_gain_ci_low"], row["pref_gain_ci_high"]),
                fmt_ci(row["delta_pref_gain_vs_opsd"], row["delta_opsd_ci_low"], row["delta_opsd_ci_high"]),
                fmt_ci(row["pref_accuracy"], row["pref_acc_ci_low"], row["pref_acc_ci_high"]),
            ]
        )

    fig, axis = plt.subplots(figsize=(13.6, 4.3), facecolor=WHITE)
    axis.axis("off")
    table_artist = axis.table(
        cellText=body,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.13, 0.08, 0.11, 0.18, 0.18, 0.18, 0.16],
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(7.7)
    table_artist.scale(1.0, 1.75)
    for (row_index, col_index), cell in table_artist.get_celld().items():
        cell.set_edgecolor(WHITE)
        if row_index == 0:
            cell.set_facecolor(BLACK)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_weight("bold")
        else:
            method = body[row_index - 1][0]
            cell.set_facecolor("#FFF8D8" if method == "LGSD-Large" else (LIGHT if row_index % 2 == 0 else WHITE))
            if col_index == 0:
                cell.get_text().set_ha("left")
                if method == "LGSD-Large":
                    cell.get_text().set_weight("bold")
    axis.text(0.0, 1.08, "Held-out preference likelihood at 1K", transform=axis.transAxes, fontsize=17, weight="bold", color=BLACK)
    axis.text(
        0.0,
        1.015,
        "Absolute likelihood metrics with paired uncertainty; normalized gain and 'surplus' are intentionally omitted.",
        transform=axis.transAxes,
        fontsize=10.5,
        color=GRAY,
    )
    fig.tight_layout(pad=1.1)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"table1_alpha_preference.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_arena_latex(table: list[dict], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Mean $\alpha$ & Target KL/raw & PrefMargin & PrefGain & $\Delta$ vs OPSD & PrefAcc \\",
        r"\midrule",
    ]
    for row in table:
        alpha = "--" if row["method"] == "Base" else f"{float(row['mean_alpha_token']):.3f}"
        values = [
            tex_escape(row["method"]),
            alpha,
            f"{row['target_kl_raw']:.3f}",
            fmt_ci(row["pref_margin"], row["pref_margin_ci_low"], row["pref_margin_ci_high"]),
            fmt_ci(row["pref_gain_vs_base"], row["pref_gain_ci_low"], row["pref_gain_ci_high"]),
            fmt_ci(row["delta_pref_gain_vs_opsd"], row["delta_opsd_ci_low"], row["delta_opsd_ci_high"]),
            fmt_ci(row["pref_accuracy"], row["pref_acc_ci_low"], row["pref_acc_ci_high"]),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def build_completion_analysis(math_per_query: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    records = read_jsonl(math_per_query)
    selected = [
        row
        for row in records
        if row.get("model") == "Qwen3-8B"
        and row.get("method") in {"OPSD", "TRSD"}
        and int(row.get("episodes", -1)) == 64
        and row.get("run_tag") == "canonical"
        and row.get("section") == "main"
    ]
    keyed: dict[str, dict[str, dict]] = {"OPSD": {}, "TRSD": {}}
    for row in selected:
        keyed[row["method"]][row["query_id"]] = row
    ids = sorted(keyed["OPSD"])
    if len(ids) != 143 or sorted(keyed["TRSD"]) != ids:
        raise ValueError("Expected 143 paired Qwen3-8B episode-64 records")

    method_rows: list[dict] = []
    for method, label in [("OPSD", "OPSD"), ("TRSD", "LGSD")]:
        rows = [keyed[method][qid] for qid in ids]
        strict = np.asarray([bool(row["math_verify_strict_correct"]) for row in rows])
        loose = np.asarray([bool(row["math_verify_correct"]) for row in rows])
        cap = np.asarray([bool(row["truncated"]) for row in rows])
        completed = ~cap
        method_rows.append(
            {
                "method": label,
                "n": len(rows),
                "parse_correct": int(loose.sum()),
                "strict_correct": int(strict.sum()),
                "strict_accuracy": float(strict.mean()),
                "cap_hits": int(cap.sum()),
                "cap_hit_rate": float(cap.mean()),
                "completed_n": int(completed.sum()),
                "strict_correct_given_completed": int((strict & completed).sum()),
                "conditional_accuracy_method_specific": float(strict[completed].mean()),
            }
        )

    opsd = keyed["OPSD"]
    lgsd = keyed["TRSD"]
    opsd_strict = np.asarray([bool(opsd[qid]["math_verify_strict_correct"]) for qid in ids])
    lgsd_strict = np.asarray([bool(lgsd[qid]["math_verify_strict_correct"]) for qid in ids])
    opsd_cap = np.asarray([bool(opsd[qid]["truncated"]) for qid in ids])
    lgsd_cap = np.asarray([bool(lgsd[qid]["truncated"]) for qid in ids])
    favorable = (~opsd_strict) & lgsd_strict
    unfavorable = opsd_strict & (~lgsd_strict)
    common_completed = (~opsd_cap) & (~lgsd_cap)
    paired_rows = [
        {
            "subset": "All paired problems",
            "n": len(ids),
            "lgsd_correct_opsd_wrong": int(favorable.sum()),
            "opsd_correct_lgsd_wrong": int(unfavorable.sum()),
            "favorable_from_opsd_cap_hit": int((favorable & opsd_cap).sum()),
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(int(favorable.sum()), int(unfavorable.sum())),
        },
        {
            "subset": "Both methods completed",
            "n": int(common_completed.sum()),
            "lgsd_correct_opsd_wrong": int((favorable & common_completed).sum()),
            "opsd_correct_lgsd_wrong": int((unfavorable & common_completed).sum()),
            "favorable_from_opsd_cap_hit": 0,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                int((favorable & common_completed).sum()), int((unfavorable & common_completed).sum())
            ),
        },
    ]
    write_csv(output_dir / "table_math_completion_by_method.csv", method_rows, list(method_rows[0]))
    write_csv(output_dir / "table_math_paired_transitions.csv", paired_rows, list(paired_rows[0]))
    render_completion_figure(method_rows, paired_rows, output_dir)
    return method_rows, paired_rows


def render_completion_figure(method_rows: list[dict], paired_rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), facecolor=WHITE, gridspec_kw={"width_ratios": [1.15, 1.0]})
    for axis in axes:
        axis.set_facecolor(WHITE)
    labels = [row["method"] for row in method_rows]
    strict = np.asarray([row["strict_correct"] for row in method_rows])
    cap = np.asarray([row["cap_hits"] for row in method_rows])
    other = 143 - strict - cap
    x = np.arange(2)
    axes[0].bar(x, strict, color=[BLUE, YELLOW], edgecolor=BLACK, linewidth=0.8, label="Strict correct")
    axes[0].bar(x, cap, bottom=strict, color="#444444", edgecolor=BLACK, linewidth=0.8, label="Hit 10,240 cap")
    axes[0].bar(x, other, bottom=strict + cap, color="#D9D6CE", edgecolor=BLACK, linewidth=0.8, label="Other incorrect")
    for index, value in enumerate(strict):
        axes[0].text(index, value / 2, f"{value}/143\n{100*value/143:.1f}%", ha="center", va="center", fontsize=10, weight="bold")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Problems")
    axes[0].set_title("A  Strict accuracy includes completion", loc="left", fontsize=12, weight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(axes[0])

    all_pairs, completed_pairs = paired_rows
    categories = ["All 143\npaired", "Both completed\n(n=97)"]
    favorable = [all_pairs["lgsd_correct_opsd_wrong"], completed_pairs["lgsd_correct_opsd_wrong"]]
    unfavorable = [all_pairs["opsd_correct_lgsd_wrong"], completed_pairs["opsd_correct_lgsd_wrong"]]
    positions = np.arange(2)
    width = 0.33
    axes[1].bar(positions - width / 2, favorable, width, color=YELLOW, edgecolor=BLACK, label="LGSD fixes OPSD")
    axes[1].bar(positions + width / 2, unfavorable, width, color=BLUE, edgecolor=BLACK, label="OPSD fixes LGSD")
    for i, (fav, unfav) in enumerate(zip(favorable, unfavorable)):
        axes[1].text(i - width / 2, fav + 0.4, str(fav), ha="center", fontsize=10, weight="bold")
        axes[1].text(i + width / 2, unfav + 0.4, str(unfav), ha="center", fontsize=10, weight="bold")
    axes[1].annotate("11/16 follow\nan OPSD cap hit", xy=(-width / 2, 11), xytext=(0.25, 13.0), arrowprops={"arrowstyle": "->", "color": BLACK}, fontsize=9)
    axes[1].set_xticks(positions, categories)
    axes[1].set_ylim(0, 19)
    axes[1].set_ylabel("Discordant problems")
    axes[1].set_title("B  Paired transition anatomy", loc="left", fontsize=12, weight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    style_axes(axes[1])
    fig.suptitle("Qwen3-8B at episode 64: completion explains most, not all, of the gain", x=0.04, ha="left", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=1.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_math_completion_anatomy.{suffix}", dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_gptoss_analysis(trace_path: Path, output_dir: Path) -> list[dict]:
    with trace_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    base_row = next(row for row in rows if row["method"] == "base")
    base_acc = float(base_row["accuracy"])
    table: list[dict] = []
    series: dict[str, list[tuple[int, float]]] = {}
    for method, label in [("opsd", "OPSD"), ("lgsd", "LGSD")]:
        method_rows = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["episode"]))
        points = [(0, base_acc)] + [(int(row["episode"]), float(row["accuracy"])) for row in method_rows]
        series[label] = points
        episodes = np.asarray([point[0] for point in points], dtype=float)
        accuracy = np.asarray([point[1] for point in points], dtype=float)
        auc = float(np.trapezoid(accuracy, episodes) / (episodes[-1] - episodes[0]))
        best_index = int(np.argmax(accuracy))
        endpoint = method_rows[-1]
        table.append(
            {
                "method": label,
                "best_episode": int(episodes[best_index]),
                "best_accuracy": float(accuracy[best_index]),
                "episode64_accuracy": float(endpoint["accuracy"]),
                "episode64_delta_vs_base": float(endpoint["accuracy"]) - base_acc,
                "normalized_auc_0_64": auc,
                "episode64_kl_to_base": float(endpoint["reverse_kl_to_base_nats"]),
                "episode64_cap_hit_rate": float(endpoint["cap_hit_rate"]),
                "training_seed_count": 1,
                "decode_seed_for_trace": int(endpoint["decode_seed"]),
            }
        )
    write_csv(output_dir / "table_gptoss_checkpoint_selection.csv", table, list(table[0]))
    render_gptoss_figure(series, base_acc, output_dir)
    return table


def render_gptoss_figure(series: dict[str, list[tuple[int, float]]], base_acc: float, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 5.3), facecolor=WHITE)
    axis.set_facecolor(WHITE)
    for label, color, marker in [("OPSD", BLUE, "s"), ("LGSD", YELLOW, "o")]:
        points = series[label]
        x = [point[0] for point in points]
        y = [100 * point[1] for point in points]
        axis.plot(x, y, color=color, marker=marker, markeredgecolor=BLACK, linewidth=2.7, markersize=7, label=label)
    axis.axhline(100 * base_acc, color=BLACK, linestyle=(0, (4, 3)), linewidth=1.2, label="Frozen Base")
    axis.set_xticks([0, 16, 32, 48, 64])
    axis.set_ylim(10, 84)
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Strict accuracy (%)")
    fig.text(0.115, 0.965, "GPT-OSS-20B: LGSD delays collapse but does not prevent it", fontsize=13.5, weight="bold", color=BLACK)
    fig.text(0.115, 0.925, "One training seed · fixed 143-problem probe · checkpoint selection must be reported", fontsize=9, color=GRAY)
    style_axes(axis)
    axis.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(rect=(0, 0, 1, 0.89), pad=1.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig_gptoss_checkpoint_selection.{suffix}", dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_overlap_audit(arena_run_root: Path, output_dir: Path) -> dict:
    train = read_jsonl(arena_run_root / "train/lgsd_large/episodes.jsonl")
    heldout = read_jsonl(arena_run_root / "preference_eval/data/heldout_preference_pairs.jsonl")
    preparation = json.loads(
        (arena_run_root / "preference_eval/data/MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    train_hashes = {row["problem_sha256"] for row in train}
    heldout_hashes = {row["prompt_sha256"] for row in heldout}
    result = {
        "train_rows": len(train),
        "train_unique_exact_sha256": len(train_hashes),
        "heldout_rows": len(heldout),
        "heldout_unique_exact_sha256": len(heldout_hashes),
        "exact_text_hash_overlap": len(train_hashes & heldout_hashes),
        "excluded_train_and_audit_normalized_prompts": preparation[
            "leakage_audit"
        ]["excluded_normalized_prompt_count"],
        "normalized_prompt_overlap_after_exclusion": preparation[
            "leakage_audit"
        ]["heldout_overlap_count"],
        "semantic_or_template_overlap_audited": False,
    }
    (output_dir / "arena_exact_overlap_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena-run-root", type=Path, required=True)
    parser.add_argument("--math-per-query", type=Path, required=True)
    parser.add_argument("--gptoss-checkpoint-trace", type=Path, required=True)
    parser.add_argument(
        "--alpha-story",
        type=Path,
        default=Path("docs/experiments/qwen3_8b_arena_preference_20260818/alpha_preference_story.json"),
    )
    parser.add_argument(
        "--arena-doc-dir",
        type=Path,
        default=Path("docs/experiments/qwen3_8b_arena_preference_20260818"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument(
        "--stage",
        choices=("all", "arena", "math", "gptoss", "overlap", "arena-table", "arena-figure"),
        default="all",
        help="Run one lightweight stage at a time on memory-constrained hosts.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "arena"}:
        bootstrap_arena(
            args.arena_run_root,
            args.alpha_story,
            args.output_dir,
            args.arena_doc_dir,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
    if args.stage in {"arena-table", "arena-figure"}:
        with (args.output_dir / "table_arena_preference_uncertainty.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            saved = list(csv.DictReader(handle))
        numeric = {
            "target_kl_raw",
            "pref_margin",
            "pref_margin_ci_low",
            "pref_margin_ci_high",
            "pref_gain_vs_base",
            "pref_gain_ci_low",
            "pref_gain_ci_high",
            "delta_pref_gain_vs_opsd",
            "delta_opsd_ci_low",
            "delta_opsd_ci_high",
            "pref_accuracy",
            "pref_acc_ci_low",
            "pref_acc_ci_high",
        }
        for row in saved:
            for key in numeric:
                row[key] = float(row[key])
        if args.stage == "arena-table":
            render_arena_table(saved, args.arena_doc_dir)
            write_arena_latex(saved, args.arena_doc_dir / "arena_preference_main_table.tex")
        else:
            for row in saved:
                if row["mean_alpha_token"]:
                    row["mean_alpha_token"] = float(row["mean_alpha_token"])
            render_arena_figure(saved, args.arena_doc_dir)
    if args.stage in {"all", "math"}:
        build_completion_analysis(args.math_per_query, args.output_dir)
    if args.stage in {"all", "gptoss"}:
        build_gptoss_analysis(args.gptoss_checkpoint_trace, args.output_dir)
    if args.stage in {"all", "overlap"}:
        build_overlap_audit(args.arena_run_root, args.output_dir)


if __name__ == "__main__":
    main()
