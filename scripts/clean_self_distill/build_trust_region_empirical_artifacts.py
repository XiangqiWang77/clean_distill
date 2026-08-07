#!/usr/bin/env python3
"""Rebuild empirical CSV artifacts and plots for trust-region vs privileged runs.

This utility reads raw JSONL episode journals and writes the exact input tables
consumed by plotting scripts, then regenerates figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ART_DIR = ROOT / "artifacts" / "figures"
TOKEN_DIR = ART_DIR / "token_style_drift_heatmap"

SUPPORT_SUMMARY_PATH = ART_DIR / "teacher_support_summary_trust_region_full_run.csv"
BRANCH_STYLE_SUMMARY_PATH = TOKEN_DIR / "token_style_drift_branch_summary.csv"
TRUST_TOKEN_STYLE_PATH = TOKEN_DIR / "trust_region_clean_per_episode_token_style_metrics.csv"
PRIV_TOKEN_STYLE_PATH = TOKEN_DIR / "privileged_per_episode_token_style_metrics.csv"
TOKEN_SHIFT_PATH = TOKEN_DIR / "token_signature_top40_trust_vs_privilege.csv"
MAX_ALPHA_GUARD = 1e-12


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_ratio(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path} is not valid JSONL at line {index}") from exc
    return rows


def _extract_support_row(row: dict[str, Any], branch: str) -> dict[str, Any]:
    ridge = row.get("ridge_metrics") if isinstance(row.get("ridge_metrics"), dict) else {}
    style = row.get("style_task_error") if isinstance(row.get("style_task_error"), dict) else {}

    support_tokens = _as_float(ridge.get("support_tokens", 0.0))
    frontier_corrective = _as_float(ridge.get("frontier_corrective_tokens_selected", 0.0))
    frontier_wrong = _as_float(ridge.get("frontier_wrong_tokens_selected", 0.0))
    support_other = max(0.0, support_tokens - frontier_corrective - frontier_wrong)

    comparable = _as_float(ridge.get("frontier_comparable_count", 0.0))
    attainment = _as_float(ridge.get("frontier_target_margin_attainment_count", 0.0))

    return {
        "branch": branch,
        "query_id": str(row.get("query_id", "")),
        "episode": _as_int(row.get("episode", 0)),
        "support_tokens": support_tokens,
        "frontier_corrective_tokens_selected": frontier_corrective,
        "frontier_wrong_tokens_selected": frontier_wrong,
        "support_other_tokens": support_other,
        "frontier_ratio_corrective": _safe_ratio(frontier_corrective, support_tokens),
        "frontier_ratio_wrong": _safe_ratio(frontier_wrong, support_tokens),
        "frontier_margin_gain_mean": _as_float(ridge.get("frontier_margin_gain_mean", 0.0)),
        "db_crossing_rate": _as_float(ridge.get("decision_boundary_crossing_rate", 0.0)),
        "db_regression_rate": _as_float(ridge.get("decision_boundary_regression_rate", 0.0)),
        "target_margin_attainment": _safe_ratio(attainment, comparable),
        "candidate_count": _as_float(ridge.get("candidate_count", 0.0)),
        "specialization_seconds": _as_float(ridge.get("specialization_seconds", 0.0)),
        "style_abs_error_mean": _safe_ratio(
            _as_float(style.get("style_abs_error_sum", 0.0)),
            _as_float(style.get("style_token_count", 0.0)),
        ),
        "task_abs_error_mean": _safe_ratio(
            _as_float(style.get("task_abs_error_sum", 0.0)),
            _as_float(style.get("task_token_count", 0.0)),
        ),
        "other_abs_error_mean": _safe_ratio(
            _as_float(style.get("other_abs_error_sum", 0.0)),
            _as_float(style.get("other_token_count", 0.0)),
        ),
        "mean_teacher_student_kl": _as_float(row.get("mean_teacher_student_kl", 0.0)),
        "episode_seconds": _as_float(row.get("episode_seconds", 0.0)),
        "applicable": bool(ridge.get("applicable", True)),
    }


def _extract_style_row(row: dict[str, Any], branch: str) -> dict[str, Any]:
    style = row.get("style_task_error") if isinstance(row.get("style_task_error"), dict) else {}
    style_abs_sum = _as_float(style.get("style_abs_error_sum", 0.0))
    task_abs_sum = _as_float(style.get("task_abs_error_sum", 0.0))
    other_abs_sum = _as_float(style.get("other_abs_error_sum", 0.0))
    style_tokens = _as_float(style.get("style_token_count", 0.0))
    task_tokens = _as_float(style.get("task_token_count", 0.0))
    other_tokens = _as_float(style.get("other_token_count", 0.0))
    total_tokens = style_tokens + task_tokens + other_tokens

    return {
        "branch": branch,
        "query_id": str(row.get("query_id", "")),
        "episode": _as_int(row.get("episode", 0)),
        "style_abs_error_sum": style_abs_sum,
        "task_abs_error_sum": task_abs_sum,
        "other_abs_error_sum": other_abs_sum,
        "style_token_count": style_tokens,
        "task_token_count": task_tokens,
        "other_token_count": other_tokens,
        "style_token_frac": _safe_ratio(style_tokens, total_tokens),
        "task_token_frac": _safe_ratio(task_tokens, total_tokens),
        "other_token_frac": _safe_ratio(other_tokens, total_tokens),
        "style_abs_error_mean": _safe_ratio(style_abs_sum, style_tokens),
        "task_abs_error_mean": _safe_ratio(task_abs_sum, task_tokens),
        "other_abs_error_mean": _safe_ratio(other_abs_sum, other_tokens),
        "style_task_ratio": _safe_ratio(style_abs_sum, task_abs_sum),
        "mean_teacher_student_kl": _as_float(row.get("mean_teacher_student_kl", 0.0)),
        "episode_seconds": _as_float(row.get("episode_seconds", 0.0)),
    }


def _select_intersection_rows(
    trust_rows: list[dict[str, Any]],
    priv_rows: list[dict[str, Any]],
    require_match: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    if not trust_rows or not priv_rows:
        return trust_rows, priv_rows, {str(r.get("query_id", "")) for r in trust_rows + priv_rows}

    trust_by_q: dict[str, dict[str, Any]] = {}
    for row in trust_rows:
        trust_by_q[str(row.get("query_id", ""))] = row
    priv_by_q: dict[str, dict[str, Any]] = {}
    for row in priv_rows:
        priv_by_q[str(row.get("query_id", ""))] = row

    intersection = set(trust_by_q) & set(priv_by_q)
    if not require_match:
        return trust_rows, priv_rows, set(trust_by_q) | set(priv_by_q)

    return (
        [trust_by_q[q] for q in intersection],
        [priv_by_q[q] for q in intersection],
        intersection,
    )


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def _build_support_summary(rows: list[dict[str, Any]], branch: str) -> dict[str, float]:
    if not rows:
        return {
            "branch": branch,
            "n_episodes": 0,
            "mean_support_tokens": 0.0,
            "mean_frontier_corrective_tokens_selected": 0.0,
            "mean_frontier_wrong_tokens_selected": 0.0,
            "mean_support_other_tokens": 0.0,
            "mean_frontier_ratio_corrective": 0.0,
            "mean_frontier_ratio_wrong": 0.0,
            "mean_frontier_margin_gain_mean": 0.0,
            "mean_db_crossing_rate": 0.0,
            "mean_db_regression_rate": 0.0,
            "mean_target_margin_attainment": 0.0,
            "mean_candidate_count": 0.0,
            "mean_style_abs_error_mean": 0.0,
            "mean_task_abs_error_mean": 0.0,
            "mean_teacher_student_kl": 0.0,
            "mean_episode_seconds": 0.0,
        }

    metrics = [
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
        "mean_teacher_student_kl",
        "episode_seconds",
    ]
    out = {"branch": branch, "n_episodes": len(rows)}
    for metric in metrics:
        out[f"mean_{metric}"] = _mean([_as_float(r.get(metric, 0.0)) for r in rows])
    return out


def _build_style_branch_summary(rows: list[dict[str, Any]], branch: str) -> dict[str, float]:
    if not rows:
        return {
            "branch": branch,
            "n_episodes": 0,
            "style_token_count": 0.0,
            "task_token_count": 0.0,
            "other_token_count": 0.0,
            "style_token_frac": 0.0,
            "task_token_frac": 0.0,
            "other_token_frac": 0.0,
            "style_abs_error_mean": 0.0,
            "task_abs_error_mean": 0.0,
            "other_abs_error_mean": 0.0,
            "style_abs_error_sum": 0.0,
            "task_abs_error_sum": 0.0,
            "other_abs_error_sum": 0.0,
            "style_task_ratio": 0.0,
            "mean_teacher_student_kl": 0.0,
            "episode_seconds": 0.0,
        }

    totals = {key: 0.0 for key in (
        "style_abs_error_sum",
        "task_abs_error_sum",
        "other_abs_error_sum",
        "style_token_count",
        "task_token_count",
        "other_token_count",
    )}
    for r in rows:
        for k in totals:
            totals[k] += _as_float(r.get(k, 0.0))

    total_tokens = totals["style_token_count"] + totals["task_token_count"] + totals["other_token_count"]
    return {
        "branch": branch,
        "n_episodes": len(rows),
        "style_token_count": totals["style_token_count"],
        "task_token_count": totals["task_token_count"],
        "other_token_count": totals["other_token_count"],
        "style_token_frac": _safe_ratio(totals["style_token_count"], total_tokens),
        "task_token_frac": _safe_ratio(totals["task_token_count"], total_tokens),
        "other_token_frac": _safe_ratio(totals["other_token_count"], total_tokens),
        "style_abs_error_mean": _safe_ratio(totals["style_abs_error_sum"], totals["style_token_count"]),
        "task_abs_error_mean": _safe_ratio(totals["task_abs_error_sum"], totals["task_token_count"]),
        "other_abs_error_mean": _safe_ratio(totals["other_abs_error_sum"], totals["other_token_count"]),
        "style_abs_error_sum": totals["style_abs_error_sum"],
        "task_abs_error_sum": totals["task_abs_error_sum"],
        "other_abs_error_sum": totals["other_abs_error_sum"],
        "style_task_ratio": _safe_ratio(totals["style_abs_error_sum"], totals["task_abs_error_sum"]),
        "mean_teacher_student_kl": _mean([_as_float(r.get("mean_teacher_student_kl", 0.0)) for r in rows]),
        "episode_seconds": _mean([_as_float(r.get("episode_seconds", 0.0)) for r in rows]),
    }


def _token_signature_from_trust(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agg: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "signed_sum": 0.0,
            "abs_sum": 0.0,
            "estimated_priv_signed_sum": 0.0,
            "estimated_priv_abs_sum": 0.0,
            "alpha_sum": 0.0,
            "alpha_count": 0.0,
        }
    )
    for row in episode_rows:
        alpha = _as_float(row.get("trust_region_alpha", 1.0))
        if alpha <= 0:
            alpha = MAX_ALPHA_GUARD
        for token_row in row.get("token_level_signal") or []:
            token = str(token_row.get("token", "")).strip()
            if not token:
                continue
            d = agg[token]
            cnt = _as_float(token_row.get("count", 0.0))
            signed = _as_float(token_row.get("total_signed_delta", 0.0))
            abs_sum = _as_float(token_row.get("total_abs_delta", 0.0))
            d["count"] += cnt
            d["signed_sum"] += signed
            d["abs_sum"] += abs_sum
            d["alpha_sum"] += alpha * cnt
            d["alpha_count"] += cnt
            d["estimated_priv_signed_sum"] += signed / alpha
            d["estimated_priv_abs_sum"] += abs_sum / alpha

    rows: list[dict[str, Any]] = []
    for token, metrics in agg.items():
        count = metrics["count"]
        signed_sum = metrics["signed_sum"]
        abs_sum = metrics["abs_sum"]
        trust_mean_delta = _safe_ratio(signed_sum, count)
        trust_mean_abs_delta = _safe_ratio(abs_sum, count)
        estimated_priv_mean_delta = _safe_ratio(metrics["estimated_priv_signed_sum"], count)
        estimated_priv_mean_abs_delta = _safe_ratio(metrics["estimated_priv_abs_sum"], count)
        rows.append({
            "token": token,
            "trust_region_mean_delta": trust_mean_delta,
            "trust_region_mean_abs_delta": trust_mean_abs_delta,
            "trust_region_count": count,
            "trust_region_total_abs_delta": abs_sum,
            "trust_region_total_signed_delta": signed_sum,
            "privileged_estimated_mean_delta": estimated_priv_mean_delta,
            "privileged_estimated_mean_abs_delta": estimated_priv_mean_abs_delta,
            "privileged_estimated_total_signed_delta": metrics["estimated_priv_signed_sum"],
            "privileged_estimated_total_abs_delta": metrics["estimated_priv_abs_sum"],
            "projection_alpha_weighted_mean": _safe_ratio(metrics["alpha_sum"], metrics["alpha_count"]),
            "estimated_shrink_ratio": _safe_ratio(
                abs(trust_mean_delta),
                abs(estimated_priv_mean_delta),
            ),
            "delta_difference": trust_mean_delta - estimated_priv_mean_delta,
            "token_level_logged": "true",
        })

    rows.sort(key=lambda x: abs(_as_float(x["trust_region_total_abs_delta"])), reverse=True)
    return rows


def _run_plot_scripts(artifact_dir: Path, max_token_vocab: int = 120) -> None:
    
    # Build support heatmaps + summary.
    subprocess.run([
        "python",
        str(ROOT / "scripts" / "clean_self_distill" / "plot_teacher_support_heatmap.py"),
        "--clean-episodes",
        str(TRUST_EPISODES),
        "--priv-episodes",
        str(PRIV_EPISODES),
        "--output-dir",
        str(artifact_dir / "teacher_support_heatmaps"),
    ], check=True)

    # Build trust-region empirical figures.
    subprocess.run([
        "python",
        str(ROOT / "scripts" / "clean_self_distill" / "build_trust_region_empirical_figures.py"),
        "--support-summary-path",
        str(SUPPORT_SUMMARY_PATH),
        "--branch-style-summary-path",
        str(BRANCH_STYLE_SUMMARY_PATH),
        "--trust-style-path",
        str(TRUST_TOKEN_STYLE_PATH),
        "--priv-style-path",
        str(PRIV_TOKEN_STYLE_PATH),
        "--token-shift-path",
        str(TOKEN_SHIFT_PATH),
        "--max-token-vocab",
        str(max_token_vocab),
        "--skip-heatmap",
        "--fig-dir",
        str(ART_DIR),
    ], check=True)

    # Heatmap step is run in a dedicated process to keep peak memory bounded.
    subprocess.run([
        "python",
        "-c",
        (
            "from scripts.clean_self_distill.build_trust_region_empirical_figures "
            "import read_csv, build_token_signature_heatmap; "
            "from pathlib import Path; "
            "rows = read_csv(Path('"
            + str(TOKEN_SHIFT_PATH).replace("'", "\\'")
            + "')); "
            "build_token_signature_heatmap(rows, Path('"
            + str(ART_DIR).replace("'", "\\'")
            + "'), "
            + str(max_token_vocab)
            + ")"
        ),
    ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trust-episodes", type=Path, required=True)
    parser.add_argument("--priv-episodes", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=ART_DIR)
    parser.add_argument("--max-token-heatmap-vocab", type=int, default=120)
    parser.add_argument("--no-intersection-only", action="store_true")
    parser.add_argument("--max-token-signature", type=int, default=200)
    args = parser.parse_args()

    global TRUST_EPISODES, PRIV_EPISODES
    TRUST_EPISODES = args.trust_episodes
    PRIV_EPISODES = args.priv_episodes

    trust_raw = _read_rows(TRUST_EPISODES)
    priv_raw = _read_rows(PRIV_EPISODES)
    if not trust_raw:
        raise SystemExit(f"No rows in trust episodes: {TRUST_EPISODES}")
    if not priv_raw:
        raise SystemExit(f"No rows in privileged episodes: {PRIV_EPISODES}")

    trust_rows, priv_rows, paired_queries = _select_intersection_rows(
        trust_raw,
        priv_raw,
        require_match=not args.no_intersection_only,
    )

    artifact_dir = args.artifact_dir
    if str(artifact_dir) != str(ART_DIR):
        artifact_dir.mkdir(parents=True, exist_ok=True)

    trust_support_rows = [_extract_support_row(r, "trust-region-clean") for r in trust_rows]
    priv_support_rows = [_extract_support_row(r, "privileged") for r in priv_rows]
    trust_style_rows = [_extract_style_row(r, "Trust-Region Clean") for r in trust_rows]
    priv_style_rows = [_extract_style_row(r, "Privileged") for r in priv_rows]

    trust_summary = _build_support_summary(trust_support_rows, "trust-region-clean")
    priv_summary = _build_support_summary(priv_support_rows, "privileged")
    _write_csv(SUPPORT_SUMMARY_PATH, [trust_summary, priv_summary])

    trust_branch_style = _build_style_branch_summary(trust_style_rows, "Trust-Region Clean")
    priv_branch_style = _build_style_branch_summary(priv_style_rows, "Privileged")
    _write_csv(BRANCH_STYLE_SUMMARY_PATH, [trust_branch_style, priv_branch_style])

    _write_csv(TRUST_TOKEN_STYLE_PATH, [
        {
            "branch": "Trust-Region Clean",
            **{k: v for k, v in row.items() if k not in {"query_id", "style_abs_error_sum", "task_abs_error_sum", "other_abs_error_sum", "style_token_count", "task_token_count", "other_token_count"}},
        }
        for row in trust_style_rows
    ])

    _write_csv(PRIV_TOKEN_STYLE_PATH, [
        {
            "branch": "Privileged",
            **{k: v for k, v in row.items() if k not in {"query_id", "style_abs_error_sum", "task_abs_error_sum", "other_abs_error_sum", "style_token_count", "task_token_count", "other_token_count"}},
        }
        for row in priv_style_rows
    ])

    token_rows = _token_signature_from_trust(trust_rows)
    if args.max_token_signature > 0 and len(token_rows) > args.max_token_signature:
        token_rows = token_rows[: args.max_token_signature]
    _write_csv(TOKEN_SHIFT_PATH, [
        {
            "token": row["token"],
            "trust_region_mean_delta": row["trust_region_mean_delta"],
            "trust_region_mean_abs_delta": row["trust_region_mean_abs_delta"],
            "trust_region_count": row["trust_region_count"],
            "trust_region_total_abs_delta": row["trust_region_total_abs_delta"],
            "trust_region_total_signed_delta": row["trust_region_total_signed_delta"],
            "privileged_estimated_mean_delta": row["privileged_estimated_mean_delta"],
            "privileged_estimated_mean_abs_delta": row["privileged_estimated_mean_abs_delta"],
            "privileged_estimated_total_signed_delta": row[
                "privileged_estimated_total_signed_delta"
            ],
            "privileged_estimated_total_abs_delta": row[
                "privileged_estimated_total_abs_delta"
            ],
            "projection_alpha_weighted_mean": row["projection_alpha_weighted_mean"],
            "estimated_shrink_ratio": row["estimated_shrink_ratio"],
            "delta_difference": row["delta_difference"],
            "token_level_logged": row["token_level_logged"],
        }
        for row in token_rows
    ])

    # Keep provenance files for a tiny sanity check.
    metadata = {
        "trust_episodes": str(TRUST_EPISODES),
        "privileged_episodes": str(PRIV_EPISODES),
        "trust_episodes_rows": len(trust_rows),
        "privileged_episodes_rows": len(priv_rows),
        "paired_query_count": len(paired_queries),
        "intersection_only": not args.no_intersection_only,
    }
    (artifact_dir / "trust_region_empirical_build_meta.txt").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Regenerate figures.
    _run_plot_scripts(artifact_dir, max_token_vocab=args.max_token_heatmap_vocab)
    print("Wrote:")
    for path in [
        SUPPORT_SUMMARY_PATH,
        BRANCH_STYLE_SUMMARY_PATH,
        TRUST_TOKEN_STYLE_PATH,
        PRIV_TOKEN_STYLE_PATH,
        TOKEN_SHIFT_PATH,
    ]:
        print(path)


if __name__ == "__main__":
    main()
