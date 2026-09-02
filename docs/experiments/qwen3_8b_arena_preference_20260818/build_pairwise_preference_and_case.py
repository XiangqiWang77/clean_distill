#!/usr/bin/env python3
"""Build pair-level human-preference diagnostics and one full LGSD/OPSD case.

The figures deliberately use only the 600 saved human-voted response pairs and
their teacher-forced preference margins.  They contain no KL quantities and do
not load a model.  The case bundle copies the two saved generation trajectories
verbatim and records hashes of every source file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


BLACK = "#161616"
GRAY = "#706F6A"
LIGHT_GRAY = "#E8E5DD"
WHITE = "#FFFFFF"
PAPER = "#FFFEFA"
YELLOW = "#F2C94C"
DARK_YELLOW = "#9A6700"
BLUE = "#2F80ED"
DARK_BLUE = "#24557A"

METHOD_SPECS = (
    ("Base", "base/episode_0000.jsonl"),
    ("LGSD-Small", "lgsd_small/episode_1000.jsonl"),
    ("LGSD-Medium", "lgsd_medium/episode_1000.jsonl"),
    ("LGSD-Large", "lgsd_large/episode_1000.jsonl"),
    ("LGSD-High", "lgsd_b200_r0040/episode_1000.jsonl"),
    ("OPSD", "opsd/episode_1000.jsonl"),
)
TRAINED_METHODS = tuple(method for method, _ in METHOD_SPECS if method != "Base")
VIOLIN_SPECS = (
    ("Base", "base_margin", "Base"),
    ("LGSD-Small", "lgsd_small_margin", "LGSD-Small\nα = .339"),
    ("LGSD-Medium", "lgsd_medium_margin", "LGSD-Medium\nα = .485"),
    ("LGSD-Large", "lgsd_large_margin", "LGSD-Large\nα = .678"),
    ("LGSD-High", "lgsd_high_margin", "LGSD-High\nα = .772"),
    ("OPSD", "opsd_margin", "OPSD\nα = 1.000"),
)
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "preference_gain", (DARK_BLUE, WHITE, YELLOW)
)
SHARE_CMAP = LinearSegmentedColormap.from_list(
    "winner_share", (WHITE, "#FFF2B8", YELLOW, DARK_YELLOW)
)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def load_arena_scores(
    arena_run_root: Path,
) -> tuple[list[str], dict[str, dict[str, dict]], dict[str, Path]]:
    score_root = arena_run_root / "preference_eval" / "scores"
    paths = {method: score_root / suffix for method, suffix in METHOD_SPECS}
    keyed: dict[str, dict[str, dict]] = {}
    for method, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        method_rows: dict[str, dict] = {}
        for row in read_jsonl(path):
            query_id = str(row.get("query_id", ""))
            if not query_id or query_id in method_rows:
                raise ValueError(f"{path} has a missing or duplicate query_id")
            margin = finite_number(
                row.get("preference_margin"), context=f"{method}:{query_id}:margin"
            )
            preferred = finite_number(
                row.get("preferred_mean_logprob"),
                context=f"{method}:{query_id}:preferred_mean_logprob",
            )
            rejected = finite_number(
                row.get("rejected_mean_logprob"),
                context=f"{method}:{query_id}:rejected_mean_logprob",
            )
            if not math.isclose(margin, preferred - rejected, abs_tol=1e-9):
                raise ValueError(f"{method}:{query_id} has an inconsistent margin")
            method_rows[query_id] = row
        if len(method_rows) != 600:
            raise ValueError(f"{method}: expected 600 pairs, found {len(method_rows)}")
        keyed[method] = method_rows

    ids = sorted(
        keyed["Base"], key=lambda qid: int(keyed["Base"][qid]["global_query_index"])
    )
    reference = set(ids)
    if any(set(rows) != reference for rows in keyed.values()):
        raise ValueError("Arena score files do not contain the same 600 pair IDs")
    for method, rows in keyed.items():
        method_order = sorted(
            rows, key=lambda qid: int(rows[qid]["global_query_index"])
        )
        if method_order != ids:
            raise ValueError(f"{method}: global pair order disagrees with Base")
    return ids, keyed, paths


def collect_pair_diagnostics(
    ids: list[str], keyed: dict[str, dict[str, dict]]
) -> tuple[list[dict], list[dict], list[dict], dict]:
    margins = {
        method: np.asarray(
            [finite_number(keyed[method][qid]["preference_margin"], context=qid) for qid in ids],
            dtype=np.float64,
        )
        for method, _ in METHOD_SPECS
    }
    base = margins["Base"]
    gains = {method: margins[method] - base for method in TRAINED_METHODS}
    trained_matrix = np.column_stack([margins[method] for method in TRAINED_METHODS])
    winner_indices = np.argmax(trained_matrix, axis=1)

    # Equal-count bins are assigned after a stable Base-only sort.  No trained
    # method outcome is used to decide the columns of either heatmap.
    base_order = np.argsort(base, kind="stable")
    deciles = np.empty(len(ids), dtype=np.int64)
    for decile, indices in enumerate(np.array_split(base_order, 10), 1):
        if len(indices) != 60:
            raise ValueError("Expected exactly 60 pairs in every Base-margin decile")
        deciles[indices] = decile

    pair_rows: list[dict] = []
    for index, query_id in enumerate(ids):
        base_row = keyed["Base"][query_id]
        domains = base_row.get("domains")
        if not isinstance(domains, list) or not domains:
            raise ValueError(f"{query_id}: missing domains")
        row = {
            "query_id": query_id,
            "global_query_index": int(base_row["global_query_index"]),
            "base_margin_decile": int(deciles[index]),
            "domains": ";".join(str(value) for value in domains),
            "prompt_token_count": int(base_row["prompt_token_count"]),
            "preferred_token_count": int(base_row["preferred_token_count"]),
            "rejected_token_count": int(base_row["rejected_token_count"]),
            "base_margin": float(base[index]),
            "base_correct": bool(base[index] > 0),
            "pair_best_method": TRAINED_METHODS[int(winner_indices[index])],
        }
        for method in TRAINED_METHODS:
            stem = method.lower().replace("lgsd-", "lgsd_")
            row[f"{stem}_margin"] = float(margins[method][index])
            row[f"{stem}_gain_vs_base"] = float(gains[method][index])
            row[f"{stem}_correct"] = bool(margins[method][index] > 0)
        row["lgsd_large_minus_opsd"] = float(
            margins["LGSD-Large"][index] - margins["OPSD"][index]
        )
        pair_rows.append(row)

    decile_rows: list[dict] = []
    for decile in range(1, 11):
        mask = deciles == decile
        base_values = base[mask]
        for method_index, method in enumerate(TRAINED_METHODS):
            values = gains[method][mask]
            wins = int(np.sum(winner_indices[mask] == method_index))
            decile_rows.append(
                {
                    "base_margin_decile": decile,
                    "pair_count": int(mask.sum()),
                    "base_margin_min": float(base_values.min()),
                    "base_margin_max": float(base_values.max()),
                    "base_margin_mean": float(base_values.mean()),
                    "method": method,
                    "mean_gain_vs_base": float(values.mean()),
                    "median_gain_vs_base": float(np.median(values)),
                    "best_method_count": wins,
                    "best_method_share": wins / int(mask.sum()),
                }
            )

    transition_rows: list[dict] = []
    transition_summary: dict[str, dict] = {}
    base_correct = base > 0
    for method in ("LGSD-Large", "OPSD"):
        final_correct = margins[method] > 0
        matrix = np.asarray(
            [
                [np.sum(~base_correct & ~final_correct), np.sum(~base_correct & final_correct)],
                [np.sum(base_correct & ~final_correct), np.sum(base_correct & final_correct)],
            ],
            dtype=np.int64,
        )
        transition_summary[method] = {
            "matrix_rows_base_wrong_right_columns_final_wrong_right": matrix.tolist(),
            "base_errors_corrected": int(matrix[0, 1]),
            "base_correct_pairs_broken": int(matrix[1, 0]),
            "net_correct_pair_change": int(matrix[0, 1] - matrix[1, 0]),
            "final_correct_pairs": int(final_correct.sum()),
            "final_preference_accuracy": float(final_correct.mean()),
        }
        for base_index, base_label in enumerate(("Base wrong", "Base correct")):
            row_total = int(matrix[base_index].sum())
            for final_index, final_label in enumerate(("Final wrong", "Final correct")):
                transition_rows.append(
                    {
                        "method": method,
                        "base_state": base_label,
                        "final_state": final_label,
                        "count": int(matrix[base_index, final_index]),
                        "row_share": float(matrix[base_index, final_index] / row_total),
                    }
                )

    large_gain = gains["LGSD-Large"]
    opsd_gain = gains["OPSD"]
    difference = large_gain - opsd_gain
    method_summary: dict[str, dict] = {}
    for method_index, method in enumerate(TRAINED_METHODS):
        values = gains[method]
        method_summary[method] = {
            "mean_gain_vs_base": float(values.mean()),
            "median_gain_vs_base": float(np.median(values)),
            "pairs_with_positive_gain": int(np.sum(values > 0)),
            "pair_best_count": int(np.sum(winner_indices == method_index)),
            "pair_best_share": float(np.mean(winner_indices == method_index)),
            "preference_accuracy": float(np.mean(margins[method] > 0)),
        }
    summary = {
        "schema_version": "arena-pairwise-preference-landscape-v1",
        "pair_count": len(ids),
        "metric": "per-pair mean_logprob(human_preferred)-mean_logprob(human_rejected)",
        "contains_kl_metric": False,
        "decile_assignment": "stable equal-count sort by frozen-Base margin; 60 pairs per decile",
        "base": {
            "mean_margin": float(base.mean()),
            "median_margin": float(np.median(base)),
            "correct_pairs": int(base_correct.sum()),
            "preference_accuracy": float(base_correct.mean()),
        },
        "methods": method_summary,
        "lgsd_large_vs_opsd": {
            "mean_margin_difference": float(difference.mean()),
            "median_margin_difference": float(np.median(difference)),
            "large_higher_pairs": int(np.sum(difference > 0)),
            "opsd_higher_pairs": int(np.sum(difference < 0)),
            "ties": int(np.sum(difference == 0)),
            "pair_gain_pearson_r": float(np.corrcoef(large_gain, opsd_gain)[0, 1]),
            "both_gain": int(np.sum((large_gain > 0) & (opsd_gain > 0))),
            "both_lose": int(np.sum((large_gain < 0) & (opsd_gain < 0))),
            "large_only_gains": int(np.sum((large_gain > 0) & (opsd_gain < 0))),
            "opsd_only_gains": int(np.sum((large_gain < 0) & (opsd_gain > 0))),
        },
        "preference_correctness_transitions": transition_summary,
    }
    return pair_rows, decile_rows, transition_rows, summary


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(BLACK)
    axis.tick_params(colors=BLACK, labelsize=9)
    axis.set_facecolor(WHITE)
    axis.grid(color=LIGHT_GRAY, linewidth=0.65, alpha=0.7)
    axis.set_axisbelow(True)


def paired_mean_intervals(
    arrays: list[np.ndarray], *, replicates: int = 10_000, seed: int = 20260824
) -> list[tuple[float, float]]:
    """Return paired percentile-bootstrap CIs using bounded-memory batches."""

    pair_count = len(arrays[0])
    if any(len(values) != pair_count for values in arrays):
        raise ValueError("Preference-margin arrays are not pair-aligned")
    rng = np.random.default_rng(seed)
    draws = np.empty((len(arrays), replicates), dtype=np.float64)
    offset = 0
    while offset < replicates:
        width = min(500, replicates - offset)
        indices = rng.integers(
            0, pair_count, size=(width, pair_count), endpoint=False
        )
        for method_index, values in enumerate(arrays):
            draws[method_index, offset : offset + width] = values[indices].mean(axis=1)
        offset += width
    return [tuple(np.quantile(values, (0.025, 0.975))) for values in draws]


def render_margin_violin(pair_rows: list[dict], output_dir: Path) -> None:
    """Render raw PrefMargin rainclouds and a magnified absolute-mean panel."""

    # The monotone display transform preserves every raw margin and its sign,
    # while resolving the dense center despite a few large-magnitude outliers.
    display_scale = 0.25
    raw_values = [
        np.asarray([float(row[field]) for row in pair_rows], dtype=np.float64)
        for _, field, _ in VIOLIN_SPECS
    ]
    display_values = [np.arcsinh(values / display_scale) for values in raw_values]
    positions = np.arange(len(VIOLIN_SPECS), 0, -1)
    intervals = paired_mean_intervals(raw_values)
    means = [float(values.mean()) for values in raw_values]

    fig = plt.figure(figsize=(7.4, 7.4), facecolor=PAPER)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(3.15, 1.38),
        wspace=0.13,
        left=0.18,
        right=0.975,
        bottom=0.16,
        top=0.82,
    )
    ax = fig.add_subplot(grid[0, 0])
    bx = fig.add_subplot(grid[0, 1], sharey=ax)
    style_axis(ax)
    style_axis(bx)
    ax.grid(axis="y", visible=False)
    bx.grid(axis="y", visible=False)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.7, alpha=0.78)
    bx.grid(axis="x", color=LIGHT_GRAY, linewidth=0.7, alpha=0.78)

    violins = ax.violinplot(
        display_values,
        positions=positions,
        orientation="horizontal",
        widths=0.72,
        points=260,
        bw_method=0.22,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    method_colors = (
        "#D8D5CC",
        "#FFF0B7",
        "#FFE58A",
        "#FFD34E",
        "#F0BF32",
        "#BFD5F1",
    )
    for body, position, color in zip(
        violins["bodies"], positions, method_colors
    ):
        # Keep one density half: cloud above, raw observations below.
        for path in body.get_paths():
            path.vertices[:, 1] = np.maximum(path.vertices[:, 1], position)
        body.set_facecolor(color)
        body.set_edgecolor(BLACK)
        body.set_linewidth(0.85)
        body.set_alpha(0.66)

    rng = np.random.default_rng(20260824)
    for position, values, shown in zip(positions, raw_values, display_values):
        aligned = values > 0
        rain_y = position - 0.07 - rng.uniform(0.0, 0.28, len(values))
        ax.scatter(
            shown[~aligned],
            rain_y[~aligned],
            s=7.0,
            c=BLUE,
            alpha=0.38,
            edgecolors="none",
            rasterized=True,
            zorder=3,
        )
        ax.scatter(
            shown[aligned],
            rain_y[aligned],
            s=8.0,
            c=YELLOW,
            alpha=0.55,
            edgecolors=BLACK,
            linewidths=0.14,
            rasterized=True,
            zorder=4,
        )
        q25, median, q75 = np.arcsinh(
            np.quantile(values, (0.25, 0.5, 0.75)) / display_scale
        )
        ax.plot(
            [q25, q75],
            [position, position],
            color=BLACK,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=5,
        )
        ax.scatter(
            [median],
            [position],
            s=27,
            c=WHITE,
            edgecolors=BLACK,
            linewidths=0.9,
            zorder=6,
        )

    ax.axvline(0, color=BLACK, linewidth=1.15, linestyle=(0, (5, 3)), zorder=2)
    raw_ticks = np.asarray([-15, -5, -1, -0.2, 0, 0.2, 1, 5, 10])
    ax.set_xticks(
        np.arcsinh(raw_ticks / display_scale),
        [f"{value:g}" for value in raw_ticks],
    )
    raw_lower = min(float(values.min()) for values in raw_values)
    raw_upper = max(float(values.max()) for values in raw_values)
    ax.set_xlim(
        np.arcsinh((raw_lower - 1.0) / display_scale),
        np.arcsinh((raw_upper + 1.0) / display_scale),
    )
    ax.set_ylim(0.48, len(positions) + 0.48)
    ax.set_yticks(positions, [label for _, _, label in VIOLIN_SPECS])
    ax.tick_params(axis="y", length=0, pad=8, labelsize=8.5)
    ax.set_xlabel("Raw PrefMargin  (symmetric asinh display)", labelpad=8)
    ax.set_title(
        "A  Full pair-level distributions",
        loc="left",
        fontsize=10.5,
        weight="bold",
    )

    bx.axvline(0, color=BLACK, linewidth=1.0, linestyle=(0, (5, 3)), zorder=2)
    for position, mean, interval, color in zip(
        positions, means, intervals, method_colors
    ):
        low, high = interval
        bx.plot(
            [low, high],
            [position, position],
            color=GRAY,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=3,
        )
        bx.scatter(
            [mean],
            [position],
            marker="D",
            s=38,
            c=color,
            edgecolors=BLACK,
            linewidths=0.8,
            zorder=4,
        )
        bx.text(
            0.335,
            position,
            f"{mean:.3f}".lstrip("0"),
            ha="right",
            va="center",
            fontsize=8.4,
            weight="bold" if math.isclose(mean, max(means)) else "normal",
            color=BLACK,
        )
    bx.set_xlim(-0.055, 0.345)
    bx.set_xticks([0.0, 0.1, 0.2, 0.3], ["0", ".1", ".2", ".3"])
    bx.tick_params(axis="y", left=False, labelleft=False)
    bx.set_xlabel("Mean PrefMargin", labelpad=8)
    bx.set_title("B  Mean [95% CI]", loc="left", fontsize=10.5, weight="bold")

    legend = (
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=YELLOW,
            markeredgecolor=BLACK,
            markeredgewidth=0.4,
            markersize=6,
            label="PrefMargin > 0",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=BLUE,
            markeredgecolor="none",
            markersize=6,
            label="PrefMargin ≤ 0",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=BLACK,
            markerfacecolor=WHITE,
            markeredgecolor=BLACK,
            linewidth=3,
            markersize=5,
            label="median + IQR",
        ),
    )
    fig.legend(
        handles=legend,
        frameon=False,
        fontsize=7.8,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=3,
        columnspacing=1.6,
        handletextpad=0.55,
    )

    fig.suptitle(
        "Small mean gaps sit inside broad PrefMargin distributions",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=15.0,
        weight="bold",
        color=BLACK,
    )
    fig.text(
        0.075,
        0.918,
        "PrefMargin = mean log p(human-preferred) − mean log p(rejected)",
        fontsize=8.8,
        color=GRAY,
    )
    fig.text(
        0.075,
        0.893,
        "Rainclouds retain every saved pair; the right panel magnifies "
        "absolute mean PrefMargin with paired uncertainty",
        fontsize=8.3,
        color=GRAY,
    )
    fig.text(
        0.075,
        0.018,
        "Only panel A uses a monotone asinh display to retain outliers; "
        "panel B is linear in raw PrefMargin.",
        fontsize=7.8,
        color=GRAY,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"fig11_human_preference_margin_violin.{suffix}",
            dpi=260,
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


def render_pair_scatter(pair_rows: list[dict], summary: dict, output_dir: Path) -> None:
    opsd = np.asarray([row["opsd_gain_vs_base"] for row in pair_rows])
    large = np.asarray([row["lgsd_large_gain_vs_base"] for row in pair_rows])
    difference = large - opsd
    large_wins = difference > 0

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7), facecolor=PAPER)
    ax, bx = axes
    style_axis(ax)
    style_axis(bx)

    ax.scatter(
        opsd[~large_wins],
        large[~large_wins],
        s=24,
        c=BLUE,
        alpha=0.48,
        edgecolors="none",
        label="OPSD higher (269)",
        rasterized=True,
    )
    ax.scatter(
        opsd[large_wins],
        large[large_wins],
        s=28,
        c=YELLOW,
        alpha=0.72,
        edgecolors=BLACK,
        linewidths=0.25,
        label="LGSD-Large higher (331)",
        rasterized=True,
    )
    joint_min = float(min(opsd.min(), large.min()))
    joint_max = float(max(opsd.max(), large.max()))
    ax.plot(
        [joint_min, joint_max],
        [joint_min, joint_max],
        color=BLACK,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        label="Equal pair gain",
    )
    ticks = [-3, -1, -0.3, -0.1, 0, 0.1, 0.3, 1, 3, 8]
    ax.set_xscale("symlog", linthresh=0.08, linscale=0.9)
    ax.set_yscale("symlog", linthresh=0.08, linscale=0.9)
    ax.set_xlim(-4.0, 8.8)
    ax.set_ylim(-4.0, 8.8)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(value) for value in ticks])
    ax.set_yticklabels([str(value) for value in ticks])
    ax.set_xlabel("OPSD pair gain vs Base")
    ax.set_ylabel("LGSD-Large pair gain vs Base")
    ax.set_title("A  Same-pair changes mostly co-move", loc="left", fontsize=12, weight="bold")
    pair_stats = summary["lgsd_large_vs_opsd"]
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {pair_stats['pair_gain_pearson_r']:.3f}\n"
        f"LGSD-Large higher on {pair_stats['large_higher_pairs']}/600 pairs",
        transform=ax.transAxes,
        va="top",
        fontsize=9.2,
        color=BLACK,
        bbox={"facecolor": WHITE, "edgecolor": LIGHT_GRAY, "boxstyle": "round,pad=0.35", "alpha": 0.92},
    )
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")

    order = np.argsort(difference, kind="stable")
    percentile = 100.0 * (np.arange(len(difference)) + 0.5) / len(difference)
    bx.scatter(
        difference[order],
        percentile,
        c=np.where(difference[order] > 0, YELLOW, BLUE),
        s=20,
        alpha=0.74,
        edgecolors=np.where(difference[order] > 0, BLACK, "none"),
        linewidths=0.2,
        rasterized=True,
    )
    bx.axvline(0, color=BLACK, linewidth=1.2, linestyle=(0, (4, 3)))
    bx.axhline(50, color=LIGHT_GRAY, linewidth=0.9)
    bx.set_xscale("symlog", linthresh=0.04, linscale=0.9)
    diff_ticks = [-5, -1, -0.3, -0.1, 0, 0.1, 0.3, 1, 4]
    bx.set_xticks(diff_ticks)
    bx.set_xticklabels([str(value) for value in diff_ticks])
    bx.set_ylim(0, 100)
    bx.set_xlabel("LGSD-Large margin − OPSD margin (same pair)")
    bx.set_ylabel("Pairs at or below value (%)")
    bx.set_title("B  The advantage is heterogeneous", loc="left", fontsize=12, weight="bold")
    bx.text(
        0.04,
        0.96,
        f"mean = {pair_stats['mean_margin_difference']:+.3f}\n"
        f"median = {pair_stats['median_margin_difference']:+.3f}\n"
        f"positive = {100 * pair_stats['large_higher_pairs'] / 600:.1f}%",
        transform=bx.transAxes,
        va="top",
        fontsize=9.2,
        bbox={"facecolor": WHITE, "edgecolor": LIGHT_GRAY, "boxstyle": "round,pad=0.35", "alpha": 0.92},
    )

    fig.suptitle(
        "Human-preference means hide substantial pair-level variation",
        x=0.055,
        ha="left",
        fontsize=16,
        weight="bold",
        color=BLACK,
    )
    fig.text(
        0.055,
        0.925,
        "600 held-out human-voted pairs · episode 1,000 · every dot is one matched pair · symmetric-log axes retain all outliers",
        fontsize=9.5,
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), pad=1.5, w_pad=2.2)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"fig8_human_preference_pair_scatter.{suffix}",
            dpi=260,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


def annotated_heatmap(
    axis,
    values: np.ndarray,
    *,
    cmap,
    norm,
    formats: list[list[str]],
    xlabels: list[str],
    ylabels: list[str],
) -> None:
    image = axis.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    axis.set_xticks(np.arange(len(xlabels)), xlabels)
    axis.set_yticks(np.arange(len(ylabels)), ylabels)
    axis.tick_params(length=0, labelsize=8.5)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            rgba = cmap(norm(values[row, column]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(
                column,
                row,
                formats[row][column],
                ha="center",
                va="center",
                fontsize=7.6,
                color=WHITE if luminance < 0.46 else BLACK,
                weight="bold" if abs(values[row, column]) > 0.5 else "normal",
            )
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def render_decile_heatmaps(
    decile_rows: list[dict], summary: dict, output_dir: Path
) -> None:
    keyed = {
        (str(row["method"]), int(row["base_margin_decile"])): row
        for row in decile_rows
    }
    mean_gain = np.asarray(
        [
            [keyed[(method, decile)]["mean_gain_vs_base"] for decile in range(1, 11)]
            for method in TRAINED_METHODS
        ],
        dtype=float,
    )
    winner_share = np.asarray(
        [
            [keyed[(method, decile)]["best_method_share"] for decile in range(1, 11)]
            for method in TRAINED_METHODS
        ],
        dtype=float,
    )
    winner_totals = {
        method: int(summary["methods"][method]["pair_best_count"])
        for method in TRAINED_METHODS
    }
    xlabels = ["D1\nlowest"] + [f"D{value}" for value in range(2, 10)] + ["D10\nhighest"]
    method_labels = [method.replace("LGSD-", "LGSD–") for method in TRAINED_METHODS]
    winner_labels = [
        f"{method.replace('LGSD-', 'LGSD–')}  ({winner_totals[method]}/600)"
        for method in TRAINED_METHODS
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13.4, 7.8), facecolor=PAPER)
    fig.subplots_adjust(left=0.15, right=0.91, top=0.86, bottom=0.10, hspace=0.48)
    gain_norm = TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)
    image_a = annotated_heatmap(
        axes[0],
        mean_gain,
        cmap=GAIN_CMAP,
        norm=gain_norm,
        formats=[[f"{value:+.2f}" for value in row] for row in mean_gain],
        xlabels=xlabels,
        ylabels=method_labels,
    )
    axes[0].set_title(
        "A  Mean margin change vs Base within each frozen-Base difficulty decile",
        loc="left",
        fontsize=12,
        weight="bold",
        pad=12,
    )
    axes[0].set_xlabel("Pairs sorted only by frozen-Base preference margin (60 per column)", labelpad=8)
    color_a = fig.colorbar(image_a, ax=axes[0], fraction=0.022, pad=0.018)
    color_a.set_label("Mean pair gain", fontsize=9)
    color_a.ax.tick_params(labelsize=8)

    share_norm = Normalize(vmin=0.0, vmax=0.75)
    image_b = annotated_heatmap(
        axes[1],
        winner_share,
        cmap=SHARE_CMAP,
        norm=share_norm,
        formats=[[f"{100 * value:.0f}%" for value in row] for row in winner_share],
        xlabels=xlabels,
        ylabels=winner_labels,
    )
    axes[1].set_title(
        "B  Which trained method gives the largest margin on each pair?",
        loc="left",
        fontsize=12,
        weight="bold",
        pad=12,
    )
    axes[1].set_xlabel("Share within the same frozen-Base margin decile", labelpad=8)
    color_b = fig.colorbar(image_b, ax=axes[1], fraction=0.022, pad=0.018)
    color_b.set_label("Pair-best share", fontsize=9)
    color_b.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    color_b.ax.tick_params(labelsize=8)

    fig.suptitle(
        "The aggregate peak is concentrated, not a universal per-pair optimum",
        x=0.055,
        ha="left",
        fontsize=16,
        weight="bold",
        color=BLACK,
    )
    fig.text(
        0.055,
        0.91,
        "Low-margin pairs lose margin on average; high-margin pairs drive most of the positive mean. LGSD-Large is pair-best for 203/600, not all 600.",
        fontsize=9.5,
        color=GRAY,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"fig9_human_preference_decile_heatmaps.{suffix}",
            dpi=260,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


def render_transition_heatmaps(summary: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.6), facecolor=PAPER)
    fig.subplots_adjust(left=0.11, right=0.94, top=0.71, bottom=0.15, wspace=0.40)
    for axis, method in zip(axes, ("LGSD-Large", "OPSD")):
        info = summary["preference_correctness_transitions"][method]
        counts = np.asarray(
            info["matrix_rows_base_wrong_right_columns_final_wrong_right"], dtype=int
        )
        shares = counts / counts.sum(axis=1, keepdims=True)
        image = axis.imshow(shares, cmap=SHARE_CMAP, norm=Normalize(0, 1))
        axis.set_xticks([0, 1], [f"{method}\nwrong", f"{method}\ncorrect"])
        axis.set_yticks([0, 1], ["Base wrong\n(n=272)", "Base correct\n(n=328)"])
        axis.tick_params(length=0, labelsize=9)
        labels = [["stays wrong", "fixed"], ["broken", "stays correct"]]
        for row in range(2):
            for column in range(2):
                rgba = SHARE_CMAP(float(shares[row, column]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(
                    column,
                    row,
                    f"{counts[row, column]}\n{100 * shares[row, column]:.1f}%\n{labels[row][column]}",
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    color=WHITE if luminance < 0.46 else BLACK,
                    weight="bold" if row != column else "normal",
                )
        for spine in axis.spines.values():
            spine.set_visible(False)
        net = info["net_correct_pair_change"]
        axis.set_title(
            f"{method}: {info['base_errors_corrected']} fixed, "
            f"{info['base_correct_pairs_broken']} broken (net {net:+d})",
            fontsize=10.5,
            weight="bold",
            pad=11,
        )
    colorbar = fig.colorbar(
        image, ax=axes, fraction=0.028, pad=0.035, label="Row share"
    )
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    fig.suptitle(
        "A higher mean margin does not imply more correctly ranked human pairs",
        x=0.055,
        ha="left",
        fontsize=15,
        weight="bold",
        color=BLACK,
    )
    fig.text(
        0.055,
        0.825,
        "Frozen Base: 328/600 correct (54.7%) · LGSD-Large: 327/600 (54.5%) · OPSD: 337/600 (56.2%)",
        fontsize=9.5,
        color=GRAY,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"fig10_human_preference_transition_heatmaps.{suffix}",
            dpi=260,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


def build_preference_bundle(arena_run_root: Path, output_dir: Path) -> dict:
    ids, keyed, paths = load_arena_scores(arena_run_root)
    pair_rows, decile_rows, transition_rows, summary = collect_pair_diagnostics(ids, keyed)
    summary["inputs"] = {
        method: {
            "logical_path": f"preference_eval/scores/{path.parent.name}/{path.name}",
            "sha256": sha256_file(path),
        }
        for method, path in paths.items()
    }
    pair_fields = list(pair_rows[0])
    write_csv(output_dir / "human_preference_pair_diagnostics.csv", pair_rows, pair_fields)
    write_csv(
        output_dir / "human_preference_decile_diagnostics.csv",
        decile_rows,
        list(decile_rows[0]),
    )
    write_csv(
        output_dir / "human_preference_correctness_transitions.csv",
        transition_rows,
        list(transition_rows[0]),
    )
    (output_dir / "human_preference_pair_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_pair_scatter(pair_rows, summary, output_dir)
    render_decile_heatmaps(decile_rows, summary, output_dir)
    render_transition_heatmaps(summary, output_dir)
    render_margin_violin(pair_rows, output_dir)
    return summary


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_saved_pair_rows(output_dir: Path) -> list[dict]:
    rows = read_csv_rows(output_dir / "human_preference_pair_diagnostics.csv")
    for row in rows:
        for key in (
            "base_margin",
            "lgsd_small_margin",
            "lgsd_medium_margin",
            "lgsd_large_margin",
            "lgsd_high_margin",
            "opsd_margin",
            "opsd_gain_vs_base",
            "lgsd_large_gain_vs_base",
        ):
            row[key] = float(row[key])
    return rows


def load_saved_decile_rows(output_dir: Path) -> list[dict]:
    rows = read_csv_rows(output_dir / "human_preference_decile_diagnostics.csv")
    for row in rows:
        row["base_margin_decile"] = int(row["base_margin_decile"])
        row["mean_gain_vs_base"] = float(row["mean_gain_vs_base"])
        row["best_method_share"] = float(row["best_method_share"])
    return rows


def load_saved_summary(output_dir: Path) -> dict:
    return json.loads(
        (output_dir / "human_preference_pair_summary.json").read_text(
            encoding="utf-8"
        )
    )


def find_unique(path: Path, query_id: str) -> dict:
    matches = [row for row in read_jsonl(path) if row.get("query_id") == query_id]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one row for {query_id}, found {len(matches)}")
    return matches[0]


def markdown_fence(value: str) -> str:
    longest = 0
    current = 0
    for character in value:
        if character == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return "`" * max(4, longest + 1)


def build_case_bundle(
    *,
    query_id: str,
    queries_path: Path,
    labels_path: Path,
    lgsd_scores_path: Path,
    opsd_scores_path: Path,
    output_dir: Path,
) -> dict:
    query = find_unique(queries_path, query_id)
    label = find_unique(labels_path, query_id)
    lgsd = find_unique(lgsd_scores_path, query_id)
    opsd = find_unique(opsd_scores_path, query_id)
    if lgsd.get("method") != "trsd" or opsd.get("method") != "privileged_sd":
        raise ValueError("The saved method identities do not match LGSD/TRSD and OPSD")
    if float(lgsd.get("correct", 0)) != 1.0 or float(opsd.get("correct", 1)) != 0.0:
        raise ValueError("Requested case is not LGSD-correct / OPSD-wrong")
    if label.get("answer") != lgsd.get("parsed_answer"):
        raise ValueError("LGSD parsed answer does not match the sealed answer")
    if query.get("problem_sha256") != label.get("problem_sha256"):
        raise ValueError("Query and label hashes disagree")

    methods: dict[str, dict] = {}
    for display, row in (("LGSD", lgsd), ("OPSD", opsd)):
        response = str(row["response"])
        methods[display] = {
            "logged_method": row["method"],
            "checkpoint_episode": int(row["checkpoint_episode"]),
            "parsed_answer": str(row["parsed_answer"]),
            "correct": bool(float(row["correct"])),
            "generated_tokens": int(row["generated_tokens"]),
            "max_new_tokens": int(row["max_new_tokens"]),
            "truncated": bool(row["truncated"]),
            "seed": int(row["seed"]),
            "temperature": float(row["temperature"]),
            "top_p": float(row["top_p"]),
            "top_k": int(row["top_k"]),
            "response_sha256": sha256_text(response),
            "response": response,
        }
    bundle = {
        "schema_version": "lgsd-win-opsd-loss-full-trajectory-v1",
        "query_id": query_id,
        "source": query["source"],
        "problem_sha256": query["problem_sha256"],
        "problem": query["problem"],
        "expected_answer": str(label["answer"]),
        "selection": "The exact case already summarized in fig7_math_case; both outputs finish below the shared cap.",
        "responses_exactly_copied_from_saved_records": True,
        "methods": methods,
        "provenance": {
            "queries": {"name": queries_path.name, "sha256": sha256_file(queries_path)},
            "labels": {"name": labels_path.name, "sha256": sha256_file(labels_path)},
            "lgsd_scores": {"name": lgsd_scores_path.name, "sha256": sha256_file(lgsd_scores_path)},
            "opsd_scores": {"name": opsd_scores_path.name, "sha256": sha256_file(opsd_scores_path)},
        },
    }
    json_path = output_dir / "case_lgsd_win_opsd_lose_full.json"
    json_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Full LGSD-win / OPSD-loss reasoning trajectories",
        "",
        "This is the unabridged saved output for the same episode-64 Qwen3-8B case shown in `fig7_math_case`. "
        "The companion JSON copies both source strings exactly. The readable blocks below preserve all text, including each `<think>` block and final answer, while removing line-end spaces.",
        "",
        "## Problem",
        "",
        str(query["problem"]),
        "",
        f"Ground-truth answer: `{label['answer']}`",
        f"Query ID: `{query_id}`",
        "",
        "## Outcome",
        "",
        "| Display name | Logged method | Parsed answer | Correct | Generated tokens | Truncated |",
        "|:--|:--|--:|:--:|--:|:--:|",
        f"| LGSD | `{lgsd['method']}` | {lgsd['parsed_answer']} | yes | {lgsd['generated_tokens']:,} | {str(bool(lgsd['truncated'])).lower()} |",
        f"| OPSD | `{opsd['method']}` | {opsd['parsed_answer']} | no | {opsd['generated_tokens']:,} | {str(bool(opsd['truncated'])).lower()} |",
        "",
        "The decisive divergence occurs while expanding "
        "`(x^2+x+1)(mx+n)+x+2`: LGSD retains the remainder's extra `+x`, so the linear coefficient is `m+n+1`; "
        "OPSD writes `m+n`, leading to the wrong polynomial and answer 35.",
        "",
    ]
    for display in ("LGSD", "OPSD"):
        response = methods[display]["response"]
        rendered_response = "\n".join(line.rstrip() for line in response.splitlines())
        fence = markdown_fence(rendered_response)
        lines.extend(
            [
                f"## {display}: complete saved response",
                "",
                f"Source response SHA-256: `{methods[display]['response_sha256']}`",
                "",
                f"{fence}text",
                rendered_response,
                fence,
                "",
            ]
        )
    lines.extend(
        [
            "## Provenance",
            "",
            "The companion JSON stores the same raw responses plus decoding metadata and source-file hashes: "
            "[`case_lgsd_win_opsd_lose_full.json`](case_lgsd_win_opsd_lose_full.json).",
            "",
        ]
    )
    (output_dir / "case_lgsd_win_opsd_lose_full.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena-run-root", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--lgsd-scores", type=Path)
    parser.add_argument("--opsd-scores", type=Path)
    parser.add_argument(
        "--query-id",
        default="amc23:a8c44eb2e7e442bb08f61ab859496b716d1c73f9dd6221dc0f89bbebf05c4be8",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "preference",
            "preference-data",
            "scatter",
            "violin",
            "decile",
            "transition",
            "case",
        ),
        default="all",
        help="Use the split preference stages on memory-constrained hosts.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"all", "preference", "preference-data"}:
        if args.arena_run_root is None:
            parser.error("--arena-run-root is required for the preference stage")
        if args.stage == "preference-data":
            ids, keyed, paths = load_arena_scores(args.arena_run_root)
            pair_rows, decile_rows, transition_rows, summary = collect_pair_diagnostics(
                ids, keyed
            )
            summary["inputs"] = {
                method: {
                    "logical_path": f"preference_eval/scores/{path.parent.name}/{path.name}",
                    "sha256": sha256_file(path),
                }
                for method, path in paths.items()
            }
            write_csv(
                args.output_dir / "human_preference_pair_diagnostics.csv",
                pair_rows,
                list(pair_rows[0]),
            )
            write_csv(
                args.output_dir / "human_preference_decile_diagnostics.csv",
                decile_rows,
                list(decile_rows[0]),
            )
            write_csv(
                args.output_dir / "human_preference_correctness_transitions.csv",
                transition_rows,
                list(transition_rows[0]),
            )
            (args.output_dir / "human_preference_pair_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            build_preference_bundle(args.arena_run_root, args.output_dir)
    if args.stage == "scatter":
        render_pair_scatter(
            load_saved_pair_rows(args.output_dir),
            load_saved_summary(args.output_dir),
            args.output_dir,
        )
    if args.stage == "violin":
        render_margin_violin(load_saved_pair_rows(args.output_dir), args.output_dir)
    if args.stage == "decile":
        render_decile_heatmaps(
            load_saved_decile_rows(args.output_dir),
            load_saved_summary(args.output_dir),
            args.output_dir,
        )
    if args.stage == "transition":
        render_transition_heatmaps(load_saved_summary(args.output_dir), args.output_dir)
    if args.stage in {"all", "case"}:
        missing = [
            name
            for name, value in (
                ("--queries", args.queries),
                ("--labels", args.labels),
                ("--lgsd-scores", args.lgsd_scores),
                ("--opsd-scores", args.opsd_scores),
            )
            if value is None
        ]
        if missing:
            parser.error(f"{' '.join(missing)} required for the case stage")
        build_case_bundle(
            query_id=args.query_id,
            queries_path=args.queries,
            labels_path=args.labels,
            lgsd_scores_path=args.lgsd_scores,
            opsd_scores_path=args.opsd_scores,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
