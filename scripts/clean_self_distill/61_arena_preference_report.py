#!/usr/bin/env python3
"""Render the complete Arena human-preference likelihood figure suite.

Inputs are existing human-vote pair log-probabilities, training journals, the
fixed-prefix policy-KL audit, and the response-level StyleDistance audit.  No
generated-response judge score or Bradley--Terry estimate is accepted.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.clean_self_distill.arena_preference import (
    METHOD_ORDER,
    ArenaPreferenceError,
    align_score_rows,
    load_preference_pairs,
    read_jsonl,
    summarize_score_rows,
    validate_score_row,
)


SCHEMA_VERSION = "arena-human-preference-complete-report-v1"
CONDITIONS = (
    ("lgsd_small", "LGSD-Small"),
    ("lgsd_medium", "LGSD-Medium"),
    ("lgsd_large", "LGSD-Large"),
    ("opsd", "OPSD"),
)
TAG_BY_METHOD = {label: tag for tag, label in CONDITIONS}
BLACK = "#111111"
YELLOW = "#FACC15"
PALE_YELLOW = "#FDE68A"
DARK_YELLOW = "#CA8A04"
PAPER_WHITE = "#FFFDF2"
METHOD_COLORS = {
    "Base": BLACK,
    "LGSD-Small": PALE_YELLOW,
    "LGSD-Medium": YELLOW,
    "LGSD-Large": DARK_YELLOW,
    "OPSD": BLACK,
}
METHOD_MARKERS = {
    "LGSD-Small": "o",
    "LGSD-Medium": "s",
    "LGSD-Large": "^",
    "OPSD": "D",
}
METHOD_LINESTYLES = {
    "LGSD-Small": ":",
    "LGSD-Medium": "-",
    "LGSD-Large": "--",
    "OPSD": "-.",
}
PREFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "preference_black_yellow", (BLACK, PAPER_WHITE, YELLOW)
)
FIGURE_DPI = 200
FIGURE_FORMATS = ("png", "pdf")


class PreferenceReportError(ArenaPreferenceError):
    """Raised when completed inputs cannot form a matched report."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_input_path(path: Path, *, run_root: Path) -> str:
    """Describe an input without publishing a workstation-specific prefix."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(run_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreferenceReportError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PreferenceReportError(f"{path} must contain one JSON object")
    return value


def _finite(value: object, *, context: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreferenceReportError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise PreferenceReportError(f"{context} must be finite")
    return result


def _stable_seed(*parts: object) -> int:
    value = "\0".join(str(part) for part in parts)
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise PreferenceReportError("bootstrap values must be a finite vector")
    point = float(array.mean())
    if array.size == 1 or resamples <= 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=float)
    batch = min(512, resamples)
    for start in range(0, resamples, batch):
        stop = min(start + batch, resamples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        samples[start:stop] = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return point, float(low), float(high)


def _bootstrap_ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float] | None:
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise PreferenceReportError("ratio bootstrap vectors must align")
    denominator_mean = float(denominator.mean())
    if denominator_mean <= 0.0:
        return None
    rng = np.random.default_rng(seed)
    ratios: list[np.ndarray] = []
    batch = min(512, resamples)
    positive = 0
    total = 0
    for start in range(0, resamples, batch):
        stop = min(start + batch, resamples)
        indices = rng.integers(
            0, numerator.size, size=(stop - start, numerator.size)
        )
        numerator_means = numerator[indices].mean(axis=1)
        denominator_means = denominator[indices].mean(axis=1)
        valid = denominator_means > 0.0
        positive += int(valid.sum())
        total += len(valid)
        if valid.any():
            ratios.append(numerator_means[valid] / denominator_means[valid])
    if total == 0 or positive / total < 0.95:
        return None
    values = np.concatenate(ratios)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _score_paths(score_root: Path) -> tuple[Path, list[int], dict[tuple[str, int], Path]]:
    base = score_root / "base" / "episode_0000.jsonl"
    if not base.is_file():
        raise PreferenceReportError(f"missing Base score file: {base}")
    discovered: dict[tuple[str, int], Path] = {}
    checkpoint_sets: list[set[int]] = []
    for tag, method in CONDITIONS:
        method_dir = score_root / tag
        paths = sorted(method_dir.glob("episode_*.jsonl"))
        checkpoints: set[int] = set()
        for path in paths:
            match = re.fullmatch(r"episode_(\d+)\.jsonl", path.name)
            if match is None:
                continue
            checkpoint = int(match.group(1))
            checkpoints.add(checkpoint)
            discovered[(method, checkpoint)] = path
        if not checkpoints:
            raise PreferenceReportError(f"no score files for {method} under {method_dir}")
        checkpoint_sets.append(checkpoints)
    common = set.intersection(*checkpoint_sets)
    union = set.union(*checkpoint_sets)
    if common != union:
        raise PreferenceReportError(
            f"methods have different scored checkpoints: {[sorted(x) for x in checkpoint_sets]}"
        )
    return base, sorted(common), discovered


def _load_score_file(path: Path) -> list[dict[str, Any]]:
    rows = [
        validate_score_row(row, row_number=index)
        for index, row in enumerate(read_jsonl(path), 1)
    ]
    summarize_score_rows(rows)
    return rows


def _number(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _training_prefix_metrics(
    journal: Path, checkpoints: Sequence[int]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    requested = set(checkpoints)
    maximum = max(requested)
    snapshots: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    target_sum = 0.0
    target_tokens = 0
    achieved_sum = 0.0
    achieved_tokens = 0
    raw_sum = 0.0
    raw_tokens = 0
    alpha_sum = 0.0
    alpha_tokens = 0
    update_values: list[float] = []
    response_tokens = 0
    cap_hits = 0
    radius: float | None = None
    expected_episode = 1
    with journal.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw_row = json.loads(line)
            if not isinstance(raw_row, Mapping):
                raise PreferenceReportError(f"{journal}:{line_number} is not an object")
            episode = raw_row.get("episode")
            if episode != expected_episode:
                raise PreferenceReportError(
                    f"{journal}:{line_number} expected episode {expected_episode}"
                )
            expected_episode += 1
            tokens = raw_row.get("response_tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                raise PreferenceReportError(
                    f"{journal}:{line_number} has invalid response_tokens"
                )
            target = _number(raw_row, "mean_teacher_student_kl")
            if target is None or target < -1e-6:
                raise PreferenceReportError(
                    f"{journal}:{line_number} has invalid target KL"
                )
            target = max(0.0, target)
            target_sum += target * tokens
            target_tokens += tokens
            achieved = _number(raw_row, "trust_region_achieved_kl")
            if achieved is not None:
                achieved_sum += max(0.0, achieved) * tokens
                achieved_tokens += tokens
            raw_target = _number(raw_row, "trust_region_raw_kl")
            if raw_target is not None:
                raw_sum += max(0.0, raw_target) * tokens
                raw_tokens += tokens
            alpha = _number(raw_row, "trust_region_alpha")
            if alpha is not None:
                alpha_sum += alpha * tokens
                alpha_tokens += tokens
            update = _number(raw_row, "relative_update_norm")
            if update is not None:
                update_values.append(update)
            budget = _number(raw_row, "trust_region_kl_budget")
            if budget is not None:
                if radius is None:
                    radius = budget
                elif not math.isclose(radius, budget, rel_tol=0.0, abs_tol=1e-12):
                    raise PreferenceReportError(f"{journal} changes its KL radius")
            response_tokens += tokens
            cap_hits += int(bool(raw_row.get("rollout_cap_hit")))
            rows.append(dict(raw_row))
            if episode in requested:
                snapshots[episode] = {
                    "checkpoint": episode,
                    "target_kl": target_sum / target_tokens,
                    "achieved_forward_kl": (
                        achieved_sum / achieved_tokens if achieved_tokens else None
                    ),
                    "raw_forward_kl": raw_sum / raw_tokens if raw_tokens else None,
                    "mean_alpha": alpha_sum / alpha_tokens if alpha_tokens else 1.0,
                    "radius": radius,
                    "mean_response_tokens": response_tokens / episode,
                    "rollout_cap_fraction": cap_hits / episode,
                    "relative_update_norm_mean": (
                        math.fsum(update_values) / len(update_values)
                        if update_values
                        else None
                    ),
                    "response_tokens": response_tokens,
                }
            if episode >= maximum:
                break
    missing = sorted(requested - set(snapshots))
    if missing:
        raise PreferenceReportError(f"{journal} is missing checkpoints {missing}")
    return snapshots, rows


def _training_metrics(
    run_root: Path, checkpoints: Sequence[int]
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    snapshots: dict[tuple[str, int], dict[str, Any]] = {}
    journals: dict[str, list[dict[str, Any]]] = {}
    for tag, method in CONDITIONS:
        path = run_root / "train" / tag / "episodes.jsonl"
        method_snapshots, rows = _training_prefix_metrics(path, checkpoints)
        journals[method] = rows
        for checkpoint, values in method_snapshots.items():
            snapshots[(method, checkpoint)] = values
    for checkpoint in checkpoints:
        raw = snapshots[("OPSD", checkpoint)]["target_kl"]
        if raw <= 0:
            raise PreferenceReportError(
                f"OPSD target KL must be positive at checkpoint {checkpoint}"
            )
        for _, method in CONDITIONS:
            snapshots[(method, checkpoint)]["target_kl_raw"] = (
                snapshots[(method, checkpoint)]["target_kl"] / raw
            )
    return snapshots, journals


def _style_metrics(path: Path | None, final_checkpoint: int) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _read_json(path)
    results = payload.get("results")
    if not isinstance(results, list):
        raise PreferenceReportError(f"{path} lacks StyleDistance results")
    by_method: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, Mapping):
            raise PreferenceReportError(f"{path} has a non-object StyleDistance row")
        method = str(item.get("method", ""))
        episode = item.get("episode")
        if method not in METHOD_ORDER:
            raise PreferenceReportError(f"{path} has unknown method {method!r}")
        expected_episode = 0 if method == "Base" else final_checkpoint
        if episode != expected_episode:
            raise PreferenceReportError(
                f"{path} StyleDistance for {method} is at episode {episode}, "
                f"expected {expected_episode}"
            )
        by_method[method] = {
            "style_distance": _finite(
                item.get("style_distance"), context=f"{method} StyleDistance", nonnegative=True
            ),
            "style_n": int(item.get("n", 0)),
        }
    if set(by_method) != set(METHOD_ORDER):
        raise PreferenceReportError(f"{path} does not cover all methods")
    raw = by_method["OPSD"]["style_distance"]
    if raw <= 0:
        raise PreferenceReportError("OPSD StyleDistance must be positive")
    for method in METHOD_ORDER:
        by_method[method]["style_raw"] = by_method[method]["style_distance"] / raw
    return by_method


def _movement_metrics(
    movement_root: Path | None, final_checkpoint: int
) -> dict[str, dict[str, Any]]:
    if movement_root is None:
        return {}
    by_method: dict[str, dict[str, Any]] = {
        "Base": {"policy_kl": 0.0, "movement_n": None}
    }
    for tag, method in CONDITIONS:
        path = movement_root / tag / "final.json"
        payload = _read_json(path)
        if payload.get("method") != method:
            raise PreferenceReportError(f"{path} method identity mismatch")
        adapter = str(payload.get("adapter", ""))
        if f"episode_{final_checkpoint:04d}" not in adapter:
            raise PreferenceReportError(f"{path} is not checkpoint {final_checkpoint}")
        summary = payload.get("summary")
        if not isinstance(summary, Mapping):
            raise PreferenceReportError(f"{path} lacks a movement summary")
        by_method[method] = {
            "policy_kl": _finite(
                summary.get("policy_kl"), context=f"{method} policy KL", nonnegative=True
            ),
            "movement_n": int(summary.get("query_count", 0)),
        }
    raw = by_method["OPSD"]["policy_kl"]
    if raw <= 0:
        raise PreferenceReportError("OPSD policy KL must be positive")
    for method in METHOD_ORDER:
        by_method[method]["policy_kl_raw"] = by_method[method]["policy_kl"] / raw
    return by_method


def _preference_metrics(
    base_rows: Sequence[Mapping[str, Any]],
    score_rows: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    checkpoints: Sequence[int],
    *,
    resamples: int,
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], dict[str, np.ndarray]],
]:
    base_summary = summarize_score_rows(base_rows)
    metrics: dict[tuple[str, int], dict[str, Any]] = {}
    details: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for checkpoint in checkpoints:
        for _, method in CONDITIONS:
            aligned = align_score_rows(base_rows, score_rows[(method, checkpoint)])
            base_margin = np.asarray(
                [float(base["preference_margin"]) for base, _ in aligned]
            )
            candidate_margin = np.asarray(
                [float(candidate["preference_margin"]) for _, candidate in aligned]
            )
            base_preferred = np.asarray(
                [float(base["preferred_mean_logprob"]) for base, _ in aligned]
            )
            candidate_preferred = np.asarray(
                [float(candidate["preferred_mean_logprob"]) for _, candidate in aligned]
            )
            base_rejected = np.asarray(
                [float(base["rejected_mean_logprob"]) for base, _ in aligned]
            )
            candidate_rejected = np.asarray(
                [float(candidate["rejected_mean_logprob"]) for _, candidate in aligned]
            )
            accuracy = np.asarray(
                [float(bool(candidate["preference_correct"])) for _, candidate in aligned]
            )
            gain = candidate_margin - base_margin
            preferred_delta = candidate_preferred - base_preferred
            rejected_delta = candidate_rejected - base_rejected
            margin_ci = _bootstrap_mean_ci(
                candidate_margin,
                seed=_stable_seed(method, checkpoint, "margin"),
                resamples=resamples,
            )
            gain_ci = _bootstrap_mean_ci(
                gain,
                seed=_stable_seed(method, checkpoint, "gain"),
                resamples=resamples,
            )
            acc_ci = _bootstrap_mean_ci(
                accuracy,
                seed=_stable_seed(method, checkpoint, "accuracy"),
                resamples=resamples,
            )
            preferred_ci = _bootstrap_mean_ci(
                preferred_delta,
                seed=_stable_seed(method, checkpoint, "preferred-delta"),
                resamples=resamples,
            )
            rejected_ci = _bootstrap_mean_ci(
                rejected_delta,
                seed=_stable_seed(method, checkpoint, "rejected-delta"),
                resamples=resamples,
            )
            summary = summarize_score_rows(score_rows[(method, checkpoint)])
            metrics[(method, checkpoint)] = {
                **summary,
                "pref_margin_ci_low": margin_ci[1],
                "pref_margin_ci_high": margin_ci[2],
                "pref_gain": gain_ci[0],
                "pref_gain_ci_low": gain_ci[1],
                "pref_gain_ci_high": gain_ci[2],
                "pref_acc_ci_low": acc_ci[1],
                "pref_acc_ci_high": acc_ci[2],
                "preferred_logprob_delta": preferred_ci[0],
                "preferred_logprob_delta_ci_low": preferred_ci[1],
                "preferred_logprob_delta_ci_high": preferred_ci[2],
                "rejected_logprob_delta": rejected_ci[0],
                "rejected_logprob_delta_ci_low": rejected_ci[1],
                "rejected_logprob_delta_ci_high": rejected_ci[2],
                "base_preference_margin": base_summary["preference_margin"],
                "base_preference_accuracy": base_summary["preference_accuracy"],
            }
            details[(method, checkpoint)] = {
                "gain": gain,
                "margin": candidate_margin,
                "accuracy": accuracy,
                "preferred_delta": preferred_delta,
                "rejected_delta": rejected_delta,
            }

        opsd_gain = details[("OPSD", checkpoint)]["gain"]
        opsd_mean = float(opsd_gain.mean())
        for _, method in CONDITIONS:
            gain = details[(method, checkpoint)]["gain"]
            metric = metrics[(method, checkpoint)]
            if abs(opsd_mean) <= 1e-12:
                metric["pref_gain_raw"] = None
                metric["pref_gain_raw_ci_low"] = None
                metric["pref_gain_raw_ci_high"] = None
            elif method == "OPSD":
                metric["pref_gain_raw"] = 1.0
                metric["pref_gain_raw_ci_low"] = 1.0
                metric["pref_gain_raw_ci_high"] = 1.0
            else:
                metric["pref_gain_raw"] = float(gain.mean()) / opsd_mean
                ratio_ci = _bootstrap_ratio_ci(
                    gain,
                    opsd_gain,
                    seed=_stable_seed(method, checkpoint, "gain-raw"),
                    resamples=resamples,
                )
                metric["pref_gain_raw_ci_low"] = ratio_ci[0] if ratio_ci else None
                metric["pref_gain_raw_ci_high"] = ratio_ci[1] if ratio_ci else None
            metric["raw_preference_gain_positive"] = opsd_mean > 0.0
    return metrics, details


def _domain_metrics(
    base_rows: Sequence[Mapping[str, Any]],
    score_rows: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    *,
    checkpoint: int,
    resamples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, method in CONDITIONS:
        aligned = align_score_rows(base_rows, score_rows[(method, checkpoint)])
        by_domain: dict[str, list[float]] = {}
        for base, candidate in aligned:
            gain = float(candidate["preference_margin"]) - float(
                base["preference_margin"]
            )
            for domain in candidate["domains"]:
                by_domain.setdefault(str(domain), []).append(gain)
        for domain, values in sorted(by_domain.items()):
            point, low, high = _bootstrap_mean_ci(
                values,
                seed=_stable_seed(method, checkpoint, domain),
                resamples=resamples,
            )
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "method": method,
                    "domain": domain,
                    "n": len(values),
                    "pref_gain": point,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return rows


def _rolling_training_metrics(
    journals: Mapping[str, Sequence[Mapping[str, Any]]], *, window: int
) -> list[dict[str, Any]]:
    if window <= 0:
        raise PreferenceReportError("rolling window must be positive")
    windowed: dict[tuple[str, int], dict[str, Any]] = {}
    for method, rows in journals.items():
        for start in range(0, len(rows), window):
            subset = rows[start : start + window]
            if not subset:
                continue
            token_total = sum(int(row["response_tokens"]) for row in subset)

            def weighted(key: str) -> float | None:
                numerator = 0.0
                denominator = 0
                for row in subset:
                    value = _number(row, key)
                    tokens = int(row["response_tokens"])
                    if value is not None:
                        numerator += value * tokens
                        denominator += tokens
                return numerator / denominator if denominator else None

            update_values = [
                value
                for row in subset
                if (value := _number(row, "relative_update_norm")) is not None
            ]
            item = {
                "method": method,
                "episode_start": int(subset[0]["episode"]),
                "episode_end": int(subset[-1]["episode"]),
                "episode_mid": 0.5
                * (int(subset[0]["episode"]) + int(subset[-1]["episode"])),
                "n_episodes": len(subset),
                "response_tokens": token_total,
                "mean_alpha": weighted("trust_region_alpha"),
                "target_kl": weighted("mean_teacher_student_kl"),
                "relative_update_norm": (
                    math.fsum(update_values) / len(update_values)
                    if update_values
                    else None
                ),
                "rollout_cap_fraction": math.fsum(
                    float(bool(row.get("rollout_cap_hit"))) for row in subset
                )
                / len(subset),
            }
            if method == "OPSD":
                item["mean_alpha"] = 1.0
            windowed[(method, int(item["episode_end"]))] = item
    result: list[dict[str, Any]] = []
    for (method, episode_end), item in sorted(
        windowed.items(), key=lambda pair: (pair[0][1], METHOD_ORDER.index(pair[0][0]))
    ):
        opsd = windowed.get(("OPSD", episode_end))
        if opsd is None or not opsd["target_kl"]:
            raise PreferenceReportError(
                f"missing OPSD rolling target KL at episode {episode_end}"
            )
        item = dict(item)
        item["target_kl_raw"] = float(item["target_kl"]) / float(opsd["target_kl"])
        result.append(item)
    return result


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (dict, list))
                        else ""
                        if value is None
                        else value
                    )
                    for key, value in row.items()
                }
            )
        handle.seek(0)
        return handle.read()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _checkpoint_label(checkpoint: int) -> str:
    if checkpoint >= 1000 and checkpoint % 1000 == 0:
        return f"{checkpoint // 1000}K"
    return str(checkpoint)


def _errorbar(metric: Mapping[str, Any], key: str) -> tuple[float, list[list[float]]] | None:
    value = metric.get(key)
    low = metric.get(f"{key}_ci_low")
    high = metric.get(f"{key}_ci_high")
    if any(item is None for item in (value, low, high)):
        return None
    point = float(value)
    return point, [[point - float(low)], [float(high) - point]]


def _save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    if "png" in FIGURE_FORMATS:
        fig.savefig(
            output_dir / f"{stem}.png", dpi=FIGURE_DPI, bbox_inches="tight"
        )
    if "pdf" in FIGURE_FORMATS:
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.clear()
    plt.close(fig)
    gc.collect()


def _plot_preference_dynamics(
    metrics: Mapping[tuple[str, int], Mapping[str, Any]],
    checkpoints: Sequence[int],
    *,
    base_accuracy: float,
    output_dir: Path,
) -> None:
    positions = np.arange(len(checkpoints))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5))
    for method in (label for _, label in CONDITIONS):
        values = np.asarray([metrics[(method, cp)]["pref_gain"] for cp in checkpoints])
        lows = np.asarray(
            [metrics[(method, cp)]["pref_gain_ci_low"] for cp in checkpoints]
        )
        highs = np.asarray(
            [metrics[(method, cp)]["pref_gain_ci_high"] for cp in checkpoints]
        )
        axes[0].plot(
            positions,
            values,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            markeredgecolor=BLACK,
            markeredgewidth=0.8,
            linewidth=2.1,
            markersize=6.5,
            label=method,
        )
        axes[0].fill_between(positions, lows, highs, color=METHOD_COLORS[method], alpha=0.13)
        acc = np.asarray(
            [metrics[(method, cp)]["preference_accuracy"] for cp in checkpoints]
        )
        acc_low = np.asarray(
            [metrics[(method, cp)]["pref_acc_ci_low"] for cp in checkpoints]
        )
        acc_high = np.asarray(
            [metrics[(method, cp)]["pref_acc_ci_high"] for cp in checkpoints]
        )
        axes[1].plot(
            positions,
            acc,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            markeredgecolor=BLACK,
            markeredgewidth=0.8,
            linewidth=2.1,
            markersize=6.5,
            label=method,
        )
        axes[1].fill_between(positions, acc_low, acc_high, color=METHOD_COLORS[method], alpha=0.13)
    axes[0].axhline(0.0, color=BLACK, linestyle="--", linewidth=1.2)
    axes[1].axhline(
        base_accuracy,
        color=METHOD_COLORS["Base"],
        linestyle="--",
        linewidth=1.3,
        label=f"Base ({base_accuracy:.3f})",
    )
    axes[0].set_title("Human-preference gain")
    axes[0].set_ylabel(r"PrefGain $\uparrow$ (nats/token)")
    axes[1].set_title("Human-preference accuracy")
    axes[1].set_ylabel(r"PrefAcc $\uparrow$")
    for axis in axes:
        axis.set_xticks(positions, [_checkpoint_label(cp) for cp in checkpoints])
        axis.set_xlabel("Training checkpoint")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8.5, ncol=2)
    fig.suptitle("Held-out LMArena human-vote likelihood (paired 95% bootstrap CI)", fontsize=13)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig1_preference_dynamics")


def _plot_locality_tradeoff(
    rows: Mapping[str, Mapping[str, Any]], *, output_dir: Path
) -> None:
    x_specs = (
        ("target_kl_raw", "Target KL / OPSD"),
        ("policy_kl_raw", "Policy KL / OPSD"),
        ("style_raw", "StyleDistance / OPSD"),
    )
    available = [spec for spec in x_specs if all(rows[m].get(spec[0]) is not None for _, m in CONDITIONS)]
    fig, axes = plt.subplots(1, len(available), figsize=(5.0 * len(available), 4.5), squeeze=False)
    for axis, (key, label) in zip(axes[0], available, strict=True):
        points: list[tuple[float, float]] = []
        for _, method in CONDITIONS:
            x = float(rows[method][key])
            y_value = rows[method].get("pref_gain_raw")
            if y_value is None:
                continue
            y = float(y_value)
            points.append((x, y))
            error = _errorbar(rows[method], "pref_gain_raw")
            yerr = error[1] if error is not None else None
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                markeredgecolor=BLACK,
                markeredgewidth=0.8,
                markersize=8,
                capsize=3,
                linestyle="none",
            )
            offset = (6, 5) if method != "OPSD" else (-42, -15)
            axis.annotate(method.replace("LGSD-", ""), (x, y), xytext=offset, textcoords="offset points", fontsize=8.5)
        extent = max([1.05, *(max(x, y) for x, y in points)])
        minimum = min([0.0, *(min(x, y) for x, y in points)])
        padding = 0.06 * (extent - minimum or 1.0)
        axis.plot(
            [minimum - padding, extent + padding],
            [minimum - padding, extent + padding],
            color=BLACK,
            linestyle="--",
            linewidth=1.2,
            label="equal retention",
        )
        axis.fill_between(
            [minimum - padding, extent + padding],
            [minimum - padding, extent + padding],
            [extent + padding, extent + padding],
            color=YELLOW,
            alpha=0.10,
        )
        axis.set_xlim(minimum - padding, extent + padding)
        axis.set_ylim(minimum - padding, extent + padding)
        axis.set_xlabel(rf"{label} $\downarrow$")
        axis.set_ylabel(r"PrefGain / OPSD $\uparrow$")
        axis.set_title(label.split(" / ")[0])
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    if available:
        axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Final-checkpoint utility retained versus movement retained", fontsize=13)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig2_locality_tradeoff")


def _plot_logprob_decomposition(
    final_rows: Mapping[str, Mapping[str, Any]],
    details: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    *,
    checkpoint: int,
    output_dir: Path,
) -> None:
    methods = [label for _, label in CONDITIONS]
    positions = np.arange(len(methods))
    width = 0.36
    preferred = np.asarray([final_rows[m]["preferred_logprob_delta"] for m in methods])
    rejected = np.asarray([final_rows[m]["rejected_logprob_delta"] for m in methods])
    preferred_err = np.asarray(
        [
            [
                final_rows[m]["preferred_logprob_delta"]
                - final_rows[m]["preferred_logprob_delta_ci_low"],
                final_rows[m]["preferred_logprob_delta_ci_high"]
                - final_rows[m]["preferred_logprob_delta"],
            ]
            for m in methods
        ]
    ).T
    rejected_err = np.asarray(
        [
            [
                final_rows[m]["rejected_logprob_delta"]
                - final_rows[m]["rejected_logprob_delta_ci_low"],
                final_rows[m]["rejected_logprob_delta_ci_high"]
                - final_rows[m]["rejected_logprob_delta"],
            ]
            for m in methods
        ]
    ).T

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    axes[0].bar(
        positions - width / 2,
        preferred,
        width,
        yerr=preferred_err,
        capsize=3,
        color=YELLOW,
        edgecolor=BLACK,
        linewidth=0.7,
        label=r"$\Delta s(y^+)$",
    )
    axes[0].bar(
        positions + width / 2,
        rejected,
        width,
        yerr=rejected_err,
        capsize=3,
        color=BLACK,
        label=r"$\Delta s(y^-)$",
    )
    axes[0].axhline(0.0, color=BLACK, linewidth=1.0)
    axes[0].set_xticks(positions, [m.replace("LGSD-", "") for m in methods])
    axes[0].set_ylabel("Change from Base (nats/token)")
    axes[0].set_title("Where does PrefGain come from?")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].spines[["top", "right"]].set_visible(False)

    for method in methods:
        values = np.sort(details[(method, checkpoint)]["gain"])
        y = np.arange(1, len(values) + 1) / len(values)
        axes[1].plot(
            values,
            y,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.0,
            label=method,
        )
    axes[1].axvline(0.0, color=BLACK, linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("Per-pair preference-margin gain (nats/token)")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_title("Pair-level heterogeneity")
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].grid(alpha=0.18)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Log-probability decomposition at {_checkpoint_label(checkpoint)}", fontsize=13)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig3_logprob_decomposition")


def _plot_domain_heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_count: int,
    max_domains: int,
    output_dir: Path,
) -> None:
    methods = [label for _, label in CONDITIONS]
    counts: dict[str, int] = {}
    for row in rows:
        if row["method"] == "OPSD":
            counts[str(row["domain"])] = int(row["n"])
    domains = [
        domain
        for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_count
    ][:max_domains]
    if not domains:
        domains = [domain for domain, _ in sorted(counts.items(), key=lambda item: -item[1])[:max_domains]]
    lookup = {(str(row["method"]), str(row["domain"])): float(row["pref_gain"]) for row in rows}
    matrix = np.asarray([[lookup[(method, domain)] for domain in domains] for method in methods])
    limit = float(np.max(np.abs(matrix))) or 1e-6
    fig, axis = plt.subplots(figsize=(max(9.0, 1.05 * len(domains)), 4.5))
    image = axis.imshow(
        matrix,
        cmap=PREFERENCE_CMAP,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axis.set_yticks(np.arange(len(methods)), [m.replace("LGSD-", "") for m in methods])
    axis.set_xticks(
        np.arange(len(domains)),
        [f"{domain.replace('_', ' ')}\n(n={counts[domain]})" for domain in domains],
        rotation=32,
        ha="right",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < -0.48 * limit else BLACK,
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("PrefGain (nats/token)")
    axis.set_title("Final-checkpoint preference gain by overlapping LMArena domain tag")
    axis.set_xlabel("Domain slice (multi-label)")
    axis.set_ylabel("Method")
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig4_domain_heatmap")


def _plot_training_diagnostics(
    rows: Sequence[Mapping[str, Any]], *, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for _, method in CONDITIONS:
        subset = [row for row in rows if row["method"] == method]
        x = np.asarray([row["episode_mid"] for row in subset])
        alpha = np.asarray([row["mean_alpha"] for row in subset], dtype=float)
        target = np.asarray([row["target_kl_raw"] for row in subset], dtype=float)
        update = np.asarray([row["relative_update_norm"] for row in subset], dtype=float)
        kwargs = {
            "color": METHOD_COLORS[method],
            "linestyle": METHOD_LINESTYLES[method],
            "linewidth": 1.9,
            "label": method,
        }
        axes[0].plot(x, alpha, **kwargs)
        axes[1].plot(x, target, **kwargs)
        axes[2].plot(x, update, **kwargs)
    axes[0].set_ylabel(r"Mean projection $\alpha$")
    axes[0].set_title("Projection strength")
    axes[1].set_ylabel("Window Target KL / OPSD")
    axes[1].set_title("Target distance retained")
    axes[2].set_ylabel("Relative update norm")
    axes[2].set_title("Optimizer movement")
    for axis in axes:
        axis.set_xlabel("Training episode")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[2].legend(frameon=False, fontsize=8.2)
    fig.suptitle("Training diagnostics (response-token-weighted rolling windows)", fontsize=13)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig5_training_diagnostics")


def _plot_metric_profile(
    final_rows: Mapping[str, Mapping[str, Any]], *, output_dir: Path
) -> None:
    candidates = (
        ("pref_gain_raw", "Preference gain"),
        ("target_kl_raw", "Target KL"),
        ("policy_kl_raw", "Policy KL"),
        ("style_raw", "StyleDistance"),
        ("prompt_variance_raw", "Prompt variance"),
    )
    metrics = [
        item
        for item in candidates
        if all(final_rows[method].get(item[0]) is not None for _, method in CONDITIONS)
    ]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(metrics))
    offsets = np.linspace(-0.24, 0.24, len(CONDITIONS))
    for offset, (_, method) in zip(offsets, CONDITIONS, strict=True):
        values = [float(final_rows[method][key]) for key, _ in metrics]
        axis.scatter(
            values,
            y + offset,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            edgecolors=BLACK,
            linewidths=0.8,
            s=58,
            label=method,
            zorder=3,
        )
    axis.axvline(1.0, color=BLACK, linestyle="--", linewidth=1.1)
    axis.axvline(0.0, color=BLACK, linewidth=0.8, alpha=0.45)
    axis.set_yticks(y, [label for _, label in metrics])
    axis.invert_yaxis()
    axis.set_xlabel("Fraction of matched OPSD (OPSD = 1)")
    axis.set_title("Final-checkpoint utility and movement profile")
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig6_metric_profile")


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "--"
    number = float(value)
    if number != 0.0 and abs(number) < 10.0**-digits:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _main_table_rows(
    metrics: Mapping[tuple[str, int], Mapping[str, Any]],
    training: Mapping[tuple[str, int], Mapping[str, Any]],
    checkpoints: Sequence[int],
    final_rows: Mapping[str, Mapping[str, Any]],
    *,
    base_accuracy: float,
) -> list[dict[str, Any]]:
    final = checkpoints[-1]
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        if method == "Base":
            row: dict[str, Any] = {
                "method": method,
                "radius": "0",
                "mean_alpha": None,
                "target_kl_raw": 0.0,
                "pref_acc_final": base_accuracy,
                "policy_kl_raw": 0.0 if "policy_kl_raw" in final_rows[method] else None,
                "style_raw": 0.0 if "style_raw" in final_rows[method] else None,
            }
            for checkpoint in checkpoints:
                row[f"pref_gain_raw_{checkpoint}"] = 0.0
        else:
            training_row = training[(method, final)]
            radius = "Raw" if method == "OPSD" else f"{float(training_row['radius']):.3g}"
            row = {
                "method": method,
                "radius": radius,
                "mean_alpha": training_row["mean_alpha"],
                "target_kl_raw": training_row["target_kl_raw"],
                "pref_acc_final": metrics[(method, final)]["preference_accuracy"],
                "policy_kl_raw": final_rows[method].get("policy_kl_raw"),
                "style_raw": final_rows[method].get("style_raw"),
            }
            for checkpoint in checkpoints:
                row[f"pref_gain_raw_{checkpoint}"] = metrics[(method, checkpoint)].get(
                    "pref_gain_raw"
                )
        rows.append(row)
    return rows


def _main_table_markdown(rows: Sequence[Mapping[str, Any]], checkpoints: Sequence[int]) -> str:
    headers = ["Method", "Radius", "Mean α", "Target KL/raw ↓"]
    headers.extend(f"PrefGain/raw @{_checkpoint_label(cp)} ↑" for cp in checkpoints)
    headers.extend([f"PrefAcc @{_checkpoint_label(checkpoints[-1])} ↑", "Policy/raw ↓", "Style/raw ↓"])
    lines = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|" + "---:|" * len(checkpoints) + "---:|---:|---:|"]
    for row in rows:
        values = [
            str(row["method"]),
            str(row["radius"]),
            _fmt(row["mean_alpha"]),
            _fmt(row["target_kl_raw"]),
        ]
        values.extend(_fmt(row[f"pref_gain_raw_{cp}"]) for cp in checkpoints)
        values.extend(
            [
                _fmt(row["pref_acc_final"]),
                _fmt(row["policy_kl_raw"]),
                _fmt(row["style_raw"]),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _main_table_latex(rows: Sequence[Mapping[str, Any]], checkpoints: Sequence[int]) -> str:
    columns = "l" + "r" * (6 + len(checkpoints))
    headers = ["Method", "Radius", r"Mean $\alpha$", r"Target KL/raw $\downarrow$"]
    headers.extend(rf"PrefGain/raw @{_checkpoint_label(cp)} $\uparrow$" for cp in checkpoints)
    headers.extend([rf"PrefAcc @{_checkpoint_label(checkpoints[-1])} $\uparrow$", r"Policy/raw $\downarrow$", r"Style/raw $\downarrow$"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Held-out human-preference likelihood and locality diagnostics. Raw denotes matched OPSD at the same checkpoint.}",
        r"\label{tab:arena-preference-locality}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = [
            str(row["method"]),
            str(row["radius"]),
            _fmt(row["mean_alpha"]),
            _fmt(row["target_kl_raw"]),
        ]
        values.extend(_fmt(row[f"pref_gain_raw_{cp}"]) for cp in checkpoints)
        values.extend([_fmt(row["pref_acc_final"]), _fmt(row["policy_kl_raw"]), _fmt(row["style_raw"])])
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _report_markdown(
    *,
    checkpoints: Sequence[int],
    pair_count: int,
    base_metrics: Mapping[str, Any],
    final_rows: Mapping[str, Mapping[str, Any]],
    table: str,
    style_present: bool,
    movement_present: bool,
) -> str:
    final = checkpoints[-1]
    candidates = [
        (method, final_rows[method].get("locality_excess"), final_rows[method].get("locality_excess_ci_low"))
        for method in ("LGSD-Small", "LGSD-Medium", "LGSD-Large")
    ]
    best_method, best_excess, best_low = max(
        candidates, key=lambda item: float("-inf") if item[1] is None else float(item[1])
    )
    raw_positive = bool(final_rows["OPSD"].get("raw_preference_gain_positive"))
    if not raw_positive:
        headline = (
            "The matched OPSD preference gain is non-positive, so normalized "
            "PrefGain retention is reported algebraically but is not interpreted "
            "as retained utility."
        )
    elif best_excess is not None and best_low is not None and float(best_low) > 0:
        headline = (
            f"At {_checkpoint_label(final)}, {best_method} has positive locality "
            f"excess {float(best_excess):+.4f} nats/token, with a paired 95% "
            f"bootstrap lower bound of {float(best_low):+.4f}."
        )
    elif best_excess is not None:
        headline = (
            f"At {_checkpoint_label(final)}, the largest point-estimate locality "
            f"excess is {best_method} at {float(best_excess):+.4f} nats/token; "
            "its paired confidence interval does not establish a positive effect."
        )
    else:
        headline = "The final locality criterion is not estimable from these inputs."

    scope_parts = [f"preference N={pair_count}"]
    if movement_present:
        scope_parts.append("fixed-prefix policy movement N=128")
    if style_present:
        scope_parts.append("StyleDistance N=600 on the separate Arena-Hard split")
    scope = "; ".join(scope_parts)
    return f"""# Qwen3-8B Arena human-preference likelihood

{headline}

This report uses one matched training seed and checkpoints {', '.join(_checkpoint_label(cp) for cp in checkpoints)}; {scope}. `PrefGain` is a paired change from the frozen Base on existing human-voted responses. It is **not** a generated-response Arena win rate. No external LLM judge or Bradley--Terry model is used.

## Main table

{table}

Base preference margin is {float(base_metrics['preference_margin']):+.5f} nats/token and Base preference accuracy is {float(base_metrics['preference_accuracy']):.3f} on N={pair_count} pairs.

## Complete figure suite

![Preference dynamics](fig1_preference_dynamics.png)

*Figure 1: PrefGain and PrefAcc across every completed checkpoint. Bands are query-paired 95% bootstrap intervals.*

![Locality tradeoff](fig2_locality_tradeoff.png)

*Figure 2: Preference gain retained versus target, learned-policy, and response-style movement retained at the final checkpoint. Points above the dashed line retain a larger fraction of preference gain than movement.*

![Log-probability decomposition](fig3_logprob_decomposition.png)

*Figure 3: Change in preferred and rejected response log-probability, plus the full pair-level distribution of preference-margin gain.*

![Domain heatmap](fig4_domain_heatmap.png)

*Figure 4: Multi-domain slice audit on overlapping source tags. Each cell is a pair-macro PrefGain; prompts may appear in more than one column.*

![Training diagnostics](fig5_training_diagnostics.png)

*Figure 5: Projection strength, target distance retained, and optimizer movement during matched training.*

![Metric profile](fig6_metric_profile.png)

*Figure 6: Final normalized profile. OPSD is one for every raw-normalized metric; lower movement with comparable preference gain is the desired signature.*

## Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| The evaluation uses existing human preferences without an LLM judge. | Pair manifest, score identities, and per-pair teacher-forced log-probabilities. | Supported |
| LGSD retains more preference gain than target movement. | Figure 2 and paired locality-excess CI at {_checkpoint_label(final)}. | {'Supported' if raw_positive and best_low is not None and float(best_low) > 0 else 'Needs stronger evidence'} |
| The result persists through long multi-domain training. | Completed checkpoints: {', '.join(_checkpoint_label(cp) for cp in checkpoints)}. | {'Supported' if final >= 20_000 else 'Needs 5K/20K checkpoints'} |
| The metric is an Arena win rate. | No newly generated responses are compared. | Not claimed |

## Limitations

- The current run has one seed; confidence intervals quantify held-out pair uncertainty, not training-seed uncertainty.
- Preference likelihood measures ranking of recorded responses under teacher forcing, not open-ended generation quality.
- Target KL uses the response-token-weighted pre-update distillation KL and is normalized to matched OPSD.
- Policy KL and StyleDistance use separate, explicitly reported audit sets; they are aggregate locality diagnostics rather than per-pair causal mediators.
"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_preference_pairs(args.pairs)
    base_path, checkpoints, score_paths = _score_paths(args.score_root)
    base_rows = _load_score_file(base_path)
    if len(base_rows) != len(pairs):
        raise PreferenceReportError("Base score coverage differs from pair file")
    base_metrics = summarize_score_rows(base_rows)
    score_rows = {key: _load_score_file(path) for key, path in score_paths.items()}
    if any(len(rows) != len(pairs) for rows in score_rows.values()):
        raise PreferenceReportError("a trained score file has incomplete pair coverage")

    preference, details = _preference_metrics(
        base_rows, score_rows, checkpoints, resamples=args.bootstrap_resamples
    )
    training, journals = _training_metrics(args.run_root, checkpoints)
    final = checkpoints[-1]
    style = _style_metrics(args.style_summary, final)
    movement = _movement_metrics(args.movement_root, final)

    long_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        long_rows.append(
            {
                "checkpoint": checkpoint,
                "method": "Base",
                "pair_count": len(pairs),
                "preference_margin": base_metrics["preference_margin"],
                "preference_accuracy": base_metrics["preference_accuracy"],
                "pref_gain": 0.0,
                "pref_gain_raw": 0.0,
                "target_kl": 0.0,
                "target_kl_raw": 0.0,
            }
        )
        for _, method in CONDITIONS:
            item = {
                "checkpoint": checkpoint,
                "method": method,
                **training[(method, checkpoint)],
                **preference[(method, checkpoint)],
            }
            target_ratio = float(item["target_kl_raw"])
            opsd_gain = details[("OPSD", checkpoint)]["gain"]
            excess_values = details[(method, checkpoint)]["gain"] - target_ratio * opsd_gain
            excess = _bootstrap_mean_ci(
                excess_values,
                seed=_stable_seed(method, checkpoint, "locality-excess"),
                resamples=args.bootstrap_resamples,
            )
            item["locality_excess"] = excess[0]
            item["locality_excess_ci_low"] = excess[1]
            item["locality_excess_ci_high"] = excess[2]
            item["criterion_point"] = bool(
                item["raw_preference_gain_positive"]
                and item.get("pref_gain_raw") is not None
                and float(item["pref_gain_raw"]) > target_ratio
            )
            item["criterion_ci_supported"] = bool(
                item["raw_preference_gain_positive"] and excess[1] > 0.0
            )
            if checkpoint == final:
                item.update(style.get(method, {}))
                item.update(movement.get(method, {}))
            long_rows.append(item)

    final_rows: dict[str, dict[str, Any]] = {
        "Base": next(
            dict(row)
            for row in long_rows
            if row["method"] == "Base" and row["checkpoint"] == final
        )
    }
    final_rows["Base"].update(style.get("Base", {}))
    final_rows["Base"].update(movement.get("Base", {}))
    for _, method in CONDITIONS:
        final_rows[method] = next(
            dict(row)
            for row in long_rows
            if row["method"] == method and row["checkpoint"] == final
        )

    domains = _domain_metrics(
        base_rows,
        score_rows,
        checkpoint=final,
        resamples=args.bootstrap_resamples,
    )
    rolling = _rolling_training_metrics(journals, window=args.rolling_window)
    table_rows = _main_table_rows(
        preference,
        training,
        checkpoints,
        final_rows,
        base_accuracy=float(base_metrics["preference_accuracy"]),
    )

    _write_text(args.output_dir / "arena_preference_metrics_long.csv", _csv_text(long_rows))
    _write_text(args.output_dir / "arena_preference_domain_slices.csv", _csv_text(domains))
    _write_text(args.output_dir / "arena_training_diagnostics.csv", _csv_text(rolling))
    _write_text(args.output_dir / "arena_preference_main_table.csv", _csv_text(table_rows))
    _write_text(
        args.output_dir / "arena_preference_main_table.tex",
        _main_table_latex(table_rows, checkpoints),
    )

    _plot_preference_dynamics(
        preference,
        checkpoints,
        base_accuracy=float(base_metrics["preference_accuracy"]),
        output_dir=args.output_dir,
    )
    _plot_locality_tradeoff(final_rows, output_dir=args.output_dir)
    _plot_logprob_decomposition(
        final_rows, details, checkpoint=final, output_dir=args.output_dir
    )
    _plot_domain_heatmap(
        domains,
        min_count=args.min_domain_count,
        max_domains=args.max_domains,
        output_dir=args.output_dir,
    )
    _plot_training_diagnostics(rolling, output_dir=args.output_dir)
    _plot_metric_profile(final_rows, output_dir=args.output_dir)

    markdown_table = _main_table_markdown(table_rows, checkpoints)
    report = _report_markdown(
        checkpoints=checkpoints,
        pair_count=len(pairs),
        base_metrics=base_metrics,
        final_rows=final_rows,
        table=markdown_table,
        style_present=bool(style),
        movement_present=bool(movement),
    )
    _write_text(args.output_dir / "README.md", report)

    machine_rows = [
        {
            key: value
            for key, value in row.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        for row in long_rows
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "external_llm_judge_used": False,
        "bradley_terry_used": False,
        "evaluation_name": "human-preference likelihood evaluation",
        "checkpoints": checkpoints,
        "pair_count": len(pairs),
        "bootstrap_resamples": args.bootstrap_resamples,
        "base": base_metrics,
        "metrics": machine_rows,
        "inputs": {
            "path_basis": "run_root_relative_or_basename",
            "pairs": {
                "path": _portable_input_path(args.pairs, run_root=args.run_root),
                "sha256": _file_sha256(args.pairs),
            },
            "base_scores": {
                "path": _portable_input_path(base_path, run_root=args.run_root),
                "sha256": _file_sha256(base_path),
            },
            "score_files": {
                f"{method}@{checkpoint}": {
                    "path": _portable_input_path(path, run_root=args.run_root),
                    "sha256": _file_sha256(path),
                }
                for (method, checkpoint), path in sorted(score_paths.items())
            },
            "style_summary": (
                _portable_input_path(args.style_summary, run_root=args.run_root)
                if args.style_summary
                else None
            ),
            "movement_root": (
                _portable_input_path(args.movement_root, run_root=args.run_root)
                if args.movement_root
                else None
            ),
        },
    }
    _write_text(
        args.output_dir / "RESULTS_COMPLETE.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--run-root", type=Path, required=True)
    root.add_argument("--pairs", type=Path, required=True)
    root.add_argument("--score-root", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--style-summary", type=Path)
    root.add_argument("--movement-root", type=Path)
    root.add_argument("--bootstrap-resamples", type=int, default=5000)
    root.add_argument("--rolling-window", type=int, default=50)
    root.add_argument("--min-domain-count", type=int, default=30)
    root.add_argument("--max-domains", type=int, default=10)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.bootstrap_resamples <= 0 or args.max_domains <= 0:
        raise PreferenceReportError("bootstrap_resamples/max_domains must be positive")
    summary = build(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "checkpoints": summary["checkpoints"],
                "pair_count": summary["pair_count"],
                "external_llm_judge_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
