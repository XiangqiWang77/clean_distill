#!/usr/bin/env python3
"""Plot support/context heatmaps for Clean (trust-region) vs Privileged teachers.

The script reads episode journals from time-box runs and outputs:
  1. Per-episode support component heatmaps for each branch.
  2. Aggregate teacher-context-source incidence heatmap.
  3. A compact CSV summary with branch-level means.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _load_rows(journal_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with journal_path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _extract_support_row(row: dict[str, Any], branch_label: str) -> dict[str, Any]:
    episode = int(_safe_float(row.get("episode")))
    rid = row.get("query_id", "")
    style = row.get("style_task_error", {})
    ridge = row.get("ridge_metrics", {})

    support_tokens = _safe_float(ridge.get("support_tokens"))
    frontier_corrective = _safe_float(ridge.get("frontier_corrective_tokens_selected"))
    frontier_wrong = _safe_float(ridge.get("frontier_wrong_tokens_selected"))
    answer_tokens = _safe_float(ridge.get("answer_tokens_selected"))
    applicable = _safe_float(1 if ridge.get("applicable", False) else 0)
    no_op = _safe_float(1 if ridge.get("specialization_no_op", False) else 0)
    ridge_loss_seconds = _safe_float(ridge.get("specialization_seconds"))
    frontier_margin = _safe_float(ridge.get("frontier_margin_gain_mean"))
    frontier_target_margin_attainment = _safe_float(
        ridge.get("frontier_target_margin_attainment_rate")
    )
    db_cross_rate = _safe_float(ridge.get("decision_boundary_crossing_rate"))
    db_reg_rate = _safe_float(ridge.get("decision_boundary_regression_rate"))
    candidate_count = _safe_float(ridge.get("candidate_count"))
    support_other = max(0.0, support_tokens - frontier_corrective - frontier_wrong)
    style_mean = _safe_ratio(
        _safe_float(style.get("style_abs_error_sum")),
        _safe_float(style.get("style_token_count", 0.0)),
    )
    task_mean = _safe_ratio(
        _safe_float(style.get("task_abs_error_sum")),
        _safe_float(style.get("task_token_count", 0.0)),
    )
    logit_ratio = _safe_float(row.get("teacher_student_normalized_logratio"))
    kl = _safe_float(row.get("mean_teacher_student_kl"))
    episode_seconds = _safe_float(row.get("episode_seconds"))

    return {
        "branch": branch_label,
        "episode": episode,
        "query_id": rid,
        "support_tokens": support_tokens,
        "frontier_corrective_tokens_selected": frontier_corrective,
        "frontier_wrong_tokens_selected": frontier_wrong,
        "support_other_tokens": support_other,
        "answer_tokens_selected": answer_tokens,
        "frontier_ratio_corrective": _safe_ratio(
            frontier_corrective, support_tokens
        ),
        "frontier_ratio_wrong": _safe_ratio(frontier_wrong, support_tokens),
        "frontier_margin_gain_mean": frontier_margin,
        "target_margin_attainment": frontier_target_margin_attainment,
        "db_crossing_rate": db_cross_rate,
        "db_regression_rate": db_reg_rate,
        "candidate_count": candidate_count,
        "applicable": applicable,
        "no_op": no_op,
        "specialization_seconds": ridge_loss_seconds,
        "style_abs_error_mean": style_mean,
        "task_abs_error_mean": task_mean,
        "logit_ratio": logit_ratio,
        "teacher_student_kl": kl,
        "episode_seconds": episode_seconds,
    }


def _build_branch_rows(path: Path, branch_label: str) -> list[dict[str, Any]]:
    rows = []
    for row in _load_rows(path):
        rows.append(_extract_support_row(row, branch_label))
    rows.sort(key=lambda r: r["episode"])
    return rows


def _matrix_from_rows(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(columns)), dtype=float)
    return np.array([[r.get(col, 0.0) for col in columns] for r in rows], dtype=float)


def _source_matrix(
    clean_raw_rows: list[dict[str, Any]],
    priv_raw_rows: list[dict[str, Any]],
) -> tuple[list[str], np.ndarray]:
    """Return a 2xS matrix of source incidence frequencies.

    Each cell is the fraction of episodes in a branch where the source appears
    in teacher_context_sources.
    """
    source_to_index: dict[str, int] = {}
    for row in clean_raw_rows:
        for source in row.get("teacher_context_sources", []) or []:
            source = str(source).strip()
            if source and source not in source_to_index:
                source_to_index[source] = len(source_to_index)
    for row in priv_raw_rows:
        for source in row.get("teacher_context_sources", []) or []:
            source = str(source).strip()
            if source and source not in source_to_index:
                source_to_index[source] = len(source_to_index)

    source_names = [None] * len(source_to_index)
    for source, index in source_to_index.items():
        source_names[index] = source

    if not source_names:
        return [], np.zeros((2, 0), dtype=float)

    clean_matrix = np.zeros((1, len(source_names)), dtype=float)
    priv_matrix = np.zeros((1, len(source_names)), dtype=float)

    if clean_raw_rows:
        for row in clean_raw_rows:
            row_sources = {str(source).strip() for source in row.get("teacher_context_sources", []) or []}
            for source in row_sources:
                idx = source_to_index[source]
                clean_matrix[0, idx] += 1.0
        clean_matrix /= max(1, len(clean_raw_rows))
    if priv_raw_rows:
        for row in priv_raw_rows:
            row_sources = {str(source).strip() for source in row.get("teacher_context_sources", []) or []}
            for source in row_sources:
                idx = source_to_index[source]
                priv_matrix[0, idx] += 1.0
        priv_matrix /= max(1, len(priv_raw_rows))

    return source_names, np.vstack([clean_matrix, priv_matrix])


def _plot_support_heatmaps(
    clean_rows: list[dict[str, Any]],
    priv_rows: list[dict[str, Any]],
    clean_raw_rows: list[dict[str, Any]],
    priv_raw_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    support_columns = [
        "support_tokens",
        "frontier_corrective_tokens_selected",
        "frontier_wrong_tokens_selected",
        "support_other_tokens",
        "frontier_ratio_corrective",
        "frontier_ratio_wrong",
        "frontier_margin_gain_mean",
        "db_crossing_rate",
        "db_regression_rate",
        "target_margin_attainment",
        "candidate_count",
        "specialization_seconds",
        "style_abs_error_mean",
        "task_abs_error_mean",
        "logit_ratio",
        "teacher_student_kl",
    ]
    clean_matrix = _matrix_from_rows(clean_rows, support_columns)
    priv_matrix = _matrix_from_rows(priv_rows, support_columns)

    if clean_matrix.ndim == 1:
        clean_matrix = clean_matrix[:, None]
    if priv_matrix.ndim == 1:
        priv_matrix = priv_matrix[:, None]

    # normalize columns independently for readability per branch.
    def _norm(a: np.ndarray) -> np.ndarray:
        if a.size == 0:
            return a
        col_max = np.nanmax(np.abs(a), axis=0, keepdims=True)
        col_max[col_max == 0] = 1.0
        return a / col_max

    clean_norm = _norm(clean_matrix)
    priv_norm = _norm(priv_matrix)
    vmin, vmax = -1.0, 1.0

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        gridspec_kw={"width_ratios": [6, 1], "height_ratios": [1, 1]},
    )

    # Support heatmaps.
    im0 = axes[0, 0].imshow(clean_norm, aspect="auto", interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Trust-Region/ Clean support traces")
    axes[0, 0].set_xlabel("Support component")
    axes[0, 0].set_ylabel("Episode")
    axes[0, 0].set_yticks(range(len(clean_rows)))
    axes[0, 0].set_yticklabels([str(r["episode"]) for r in clean_rows])
    axes[0, 0].set_xticks(range(len(support_columns)))
    axes[0, 0].set_xticklabels(support_columns, rotation=45, ha="right")

    im1 = axes[1, 0].imshow(priv_norm, aspect="auto", interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Privileged support traces")
    axes[1, 0].set_xlabel("Support component")
    axes[1, 0].set_ylabel("Episode")
    axes[1, 0].set_yticks(range(len(priv_rows)))
    axes[1, 0].set_yticklabels([str(r["episode"]) for r in priv_rows])
    axes[1, 0].set_xticks(range(len(support_columns)))
    axes[1, 0].set_xticklabels(support_columns, rotation=45, ha="right")

    # context source heatmaps (right column)
    source_names, source_matrix = _source_matrix(clean_raw_rows, priv_raw_rows)
    im2 = axes[0, 1].imshow(source_matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=1)
    axes[0, 1].set_title("Context source incidence")
    if source_names:
        axes[0, 1].set_xticks(range(len(source_names)))
        axes[0, 1].set_xticklabels(source_names, rotation=75, ha="left")
    else:
        axes[0, 1].set_xticks([])
        axes[0, 1].set_xticklabels([])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_yticklabels(["trust-region", "privileged"])

    # hide unused axes area in lower-right
    axes[1, 1].axis("off")
    cbar = fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of episodes containing source")

    fig.tight_layout()
    out = output_dir / "teacher_support_heatmap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return [out]


def _write_summary_csv(clean_rows: list[dict[str, Any]], priv_rows: list[dict[str, Any]], output: Path) -> None:
    metric_columns = [
        "support_tokens",
        "frontier_corrective_tokens_selected",
        "frontier_wrong_tokens_selected",
        "support_other_tokens",
        "frontier_ratio_corrective",
        "frontier_ratio_wrong",
        "frontier_margin_gain_mean",
        "db_crossing_rate",
        "db_regression_rate",
        "target_margin_attainment",
        "candidate_count",
        "style_abs_error_mean",
        "task_abs_error_mean",
        "teacher_student_kl",
        "episode_seconds",
    ]
    output_columns = [
        "branch",
        "n_episodes",
        *(f"mean_{name}" for name in metric_columns),
    ]

    def summarize(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {"branch": name, "n_episodes": 0, **{f"mean_{c}": 0.0 for c in metric_columns}}
        means = {"branch": name, "n_episodes": n}
        for metric in metric_columns:
            means[f"mean_{metric}"] = float(np.mean([r.get(metric, 0.0) for r in rows]))
        return means

    clean_summary = summarize(clean_rows, "trust-region-clean")
    priv_summary = summarize(priv_rows, "privileged")

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        for row in (clean_summary, priv_summary):
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-episodes", type=Path, required=True)
    parser.add_argument("--priv-episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # load raw journals to preserve context source incidence
    clean_raw = _load_rows(args.clean_episodes)
    priv_raw = _load_rows(args.priv_episodes)
    if not clean_raw and not priv_raw:
        raise SystemExit("both branch episode journals are empty")

    clean_rows = _build_branch_rows(args.clean_episodes, "clean")
    priv_rows = _build_branch_rows(args.priv_episodes, "privileged")

    outputs = _plot_support_heatmaps(
        clean_rows,
        priv_rows,
        clean_raw,
        priv_raw,
        args.output_dir,
    )
    _write_summary_csv(clean_rows, priv_rows, args.output_dir / "teacher_support_summary.csv")

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
