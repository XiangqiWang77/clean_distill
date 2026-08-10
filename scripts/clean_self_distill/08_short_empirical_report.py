#!/usr/bin/env python3
"""Build the short TRSD empirical tables and publication-ready figures.

The script intentionally has no model dependency.  ``prepare-subset`` creates a
small, deterministically selected held-out split while keeping queries and sealed
labels physically separate.  ``report`` consumes only recorded scientific
artifacts; missing measurements remain ``null``/``N/A`` rather than being filled
with synthetic values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TRSD_METHOD_ID = "trsd:exponential_teacher_projection"
WRAPPER_ORDER = ("neutral", "terse", "verbose")
COLORS = {
    "navy": "#173F5F",
    "blue": "#2878B5",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "pink": "#B565A7",
    "gray": "#6B7280",
    "light": "#E5E7EB",
}


class ReportError(RuntimeError):
    """Raised when an input artifact cannot support the requested report."""


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise ReportError(f"JSONL input does not exist: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReportError(f"{source}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ReportError(f"{source}:{line_number}: row is not an object")
            rows.append(row)
    if not rows:
        raise ReportError(f"JSONL input is empty: {source}")
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return int(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ReportError(f"{context} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ReportError(f"{context} is not finite")
    return number


def _as_int(value: Any, context: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ReportError(f"{context} is not an integer: {value!r}") from error
    return number


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sem(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _fmt_pp(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:+.2f} pp"


def prepare_subset(args: argparse.Namespace) -> None:
    queries = _read_jsonl(args.queries)
    labels = _read_jsonl(args.labels)
    if args.per_source <= 0:
        raise ReportError("--per-source must be positive")

    query_by_id: dict[str, dict[str, Any]] = {}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    forbidden_query_keys = {
        "answer",
        "correct_answer",
        "reference_answer",
        "reference_solution",
        "solution",
        "feedback",
    }
    for index, row in enumerate(queries, 1):
        query_id = str(row.get("query_id", "")).strip()
        source = str(row.get("source", "")).strip()
        problem_hash = str(row.get("problem_sha256", "")).strip()
        if not query_id or not source or len(problem_hash) != 64:
            raise ReportError(f"Query row {index} lacks query_id/source/problem_sha256")
        if forbidden_query_keys & {str(key).casefold() for key in row}:
            raise ReportError(f"Query row {index} contains a sealed target field")
        if query_id in query_by_id:
            raise ReportError(f"Duplicate query_id in query manifest: {query_id}")
        query_by_id[query_id] = row
        by_source[source].append(row)

    label_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(labels, 1):
        query_id = str(row.get("query_id", "")).strip()
        if not query_id or not str(row.get("answer", "")).strip():
            raise ReportError(f"Sealed-label row {index} lacks query_id/answer")
        if query_id in label_by_id:
            raise ReportError(f"Duplicate query_id in labels: {query_id}")
        label_by_id[query_id] = row

    requested_sources = (
        [item.strip() for item in args.sources.split(",") if item.strip()]
        if args.sources
        else sorted(by_source)
    )
    if not requested_sources:
        raise ReportError("No held-out sources were found")
    selected_queries: list[dict[str, Any]] = []
    selected_labels: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for source in requested_sources:
        candidates = sorted(
            by_source.get(source, []),
            key=lambda row: (str(row["problem_sha256"]), str(row["query_id"])),
        )
        if len(candidates) < args.per_source:
            raise ReportError(
                f"Source {source!r} has {len(candidates)} queries; "
                f"exactly {args.per_source} are required"
            )
        chosen = candidates[: args.per_source]
        for query in chosen:
            query_id = str(query["query_id"])
            label = label_by_id.get(query_id)
            if label is None:
                raise ReportError(f"Selected query {query_id!r} has no sealed label")
            if str(label.get("problem_sha256", "")) != str(query["problem_sha256"]):
                raise ReportError(f"Problem hash mismatch for {query_id!r}")
            selected_queries.append(query)
            selected_labels.append(label)
        counts[source] = len(chosen)

    out_dir = Path(args.out_dir)
    query_path = out_dir / args.query_name
    label_path = out_dir / args.label_name
    _write_jsonl(query_path, selected_queries)
    _write_jsonl(label_path, selected_labels)
    manifest = {
        "schema_version": "trsd-short-heldout-subset-v1",
        "selection": "lexicographic(problem_sha256, query_id) within source",
        "per_source": args.per_source,
        "source_counts": counts,
        "total_queries": len(selected_queries),
        "query_path": str(query_path.resolve()),
        "label_path": str(label_path.resolve()),
        "query_sha256": _sha256(query_path),
        "label_sha256": _sha256(label_path),
        "labels_physically_separate": query_path.resolve() != label_path.resolve(),
    }
    _write_json(out_dir / "subset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def prepare_stream(args: argparse.Namespace) -> None:
    """Bind a short label-free prefix without changing the training protocol."""
    rows = _read_jsonl(args.queries)
    if args.episodes <= 0:
        raise ReportError("--episodes must be positive")
    if len(rows) < args.episodes:
        raise ReportError(
            f"Query stream has {len(rows)} rows but {args.episodes} are required"
        )
    forbidden = {
        "answer",
        "correct_answer",
        "reference_answer",
        "reference_solution",
        "solution",
        "feedback",
    }
    selected = rows[: args.episodes]
    for index, row in enumerate(selected, 1):
        if forbidden & {str(key).casefold() for key in row}:
            raise ReportError(f"Training query row {index} contains a sealed target field")
    output = Path(args.output)
    _write_jsonl(output, selected)
    payload = {
        "schema_version": "trsd-short-query-stream-v1",
        "selection": "first N rows of the frozen prepared distillation stream",
        "episodes": args.episodes,
        "output": str(output.resolve()),
        "sha256": _sha256(output),
        "label_free": True,
    }
    _write_json(output.with_suffix(output.suffix + ".manifest.json"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_scored_argument(value: str) -> tuple[int, Path]:
    episode_text, separator, path_text = value.partition(":")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("expected EPISODE:PATH")
    try:
        episode = int(episode_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint episode must be an integer") from error
    if episode < 0:
        raise argparse.ArgumentTypeError("checkpoint episode must be non-negative")
    return episode, Path(path_text)


def _select_acc1_rows(rows: Sequence[Mapping[str, Any]], context: str) -> list[dict[str, Any]]:
    profiles = {str(row.get("profile", "")) for row in rows if "profile" in row}
    if "acc1" in profiles:
        selected = [dict(row) for row in rows if row.get("profile") == "acc1"]
    elif profiles:
        raise ReportError(f"{context} has profiles {sorted(profiles)} but no acc1")
    else:
        selected = [dict(row) for row in rows if int(row.get("sample_index", 0)) == 0]
    if not selected:
        raise ReportError(f"{context} contains no Acc@1 rows")
    return selected


def _load_checkpoint_metrics(
    scored_inputs: Sequence[tuple[int, Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths_by_episode: dict[int, list[Path]] = defaultdict(list)
    for episode, path in scored_inputs:
        paths_by_episode[episode].append(path)
    curve: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    reference_coverage: dict[str, str] | None = None
    for episode in sorted(paths_by_episode):
        merged: list[dict[str, Any]] = []
        for path in paths_by_episode[episode]:
            merged.extend(_select_acc1_rows(_read_jsonl(path), str(path)))
        seen: set[str] = set()
        by_source: dict[str, list[float]] = defaultdict(list)
        all_correct: list[float] = []
        for row in merged:
            query_id = str(row.get("query_id", "")).strip()
            source = str(row.get("source", "")).strip()
            if not query_id or not source or "correct" not in row:
                raise ReportError(f"Episode {episode} scored row lacks query_id/source/correct")
            if query_id in seen:
                raise ReportError(f"Episode {episode} duplicates Acc@1 query {query_id!r}")
            seen.add(query_id)
            recorded_episode = row.get("checkpoint_episode")
            if recorded_episode is not None and int(recorded_episode) != episode:
                raise ReportError(
                    f"Episode binding mismatch: CLI={episode}, row={recorded_episode}"
                )
            correct = _as_float(row["correct"], f"episode {episode} correct")
            if correct < 0.0 or correct > 1.0:
                raise ReportError(f"Episode {episode} correctness is outside [0,1]")
            by_source[source].append(correct)
            all_correct.append(correct)
        current_coverage = {str(row["query_id"]): str(row["source"]) for row in merged}
        if reference_coverage is None:
            reference_coverage = current_coverage
        elif current_coverage != reference_coverage:
            missing = sorted(set(reference_coverage) - set(current_coverage))
            extra = sorted(set(current_coverage) - set(reference_coverage))
            source_changed = sorted(
                query_id
                for query_id in set(reference_coverage) & set(current_coverage)
                if reference_coverage[query_id] != current_coverage[query_id]
            )
            raise ReportError(
                f"Episode {episode} does not use the fixed checkpoint query set: "
                f"missing={missing[:5]} extra={extra[:5]} source_changed={source_changed[:5]}"
            )
        accuracy = statistics.fmean(all_correct)
        curve.append(
            {
                "episode": episode,
                "n_queries": len(all_correct),
                "acc1": accuracy,
                "acc1_percent": 100.0 * accuracy,
            }
        )
        for source in sorted(by_source):
            values = by_source[source]
            source_rows.append(
                {
                    "episode": episode,
                    "source": source,
                    "n_queries": len(values),
                    "correct_count": sum(values),
                    "acc1": statistics.fmean(values),
                    "acc1_percent": 100.0 * statistics.fmean(values),
                }
            )

    first = curve[0]
    final = curve[-1]
    enough = len(curve) >= 2 and final["episode"] > first["episode"]
    lhg = final["acc1"] - first["acc1"] if enough else None
    best_final_gap = (
        max(row["acc1"] for row in curve) - final["acc1"] if enough else None
    )
    aulc = None
    if enough:
        area = 0.0
        baseline = first["acc1"]
        for left, right in zip(curve, curve[1:]):
            width = right["episode"] - left["episode"]
            if width <= 0:
                raise ReportError("Checkpoint episodes are not strictly increasing")
            left_gain = left["acc1"] - baseline
            right_gain = right["acc1"] - baseline
            area += width * (left_gain + right_gain) / 2.0
        aulc = area / (final["episode"] - first["episode"])
    summary = {
        "baseline_episode": first["episode"],
        "final_episode": final["episode"],
        "checkpoint_count": len(curve),
        "baseline_acc1": first["acc1"],
        "final_acc1": final["acc1"],
        "best_acc1": max(row["acc1"] for row in curve),
        "best_episode": max(curve, key=lambda row: (row["acc1"], -row["episode"]))[
            "episode"
        ],
        "lhg": lhg,
        "normalized_trapezoidal_aulc": aulc,
        "best_final_gap": best_final_gap,
        "metric_note": (
            "LHG/AULC/best-final gap require at least two distinct checkpoints"
            if not enough
            else "AULC is the episode-axis-normalized trapezoidal area of Acc@1 gain over the first checkpoint"
        ),
    }
    for row in curve:
        row["gain_over_first"] = row["acc1"] - first["acc1"]
        row["gain_over_first_pp"] = 100.0 * row["gain_over_first"]
    return curve, source_rows, summary


def _find_epsilon_row(rows: Any, epsilon: float, context: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ReportError(f"{context} epsilon_sweep is not a list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and math.isclose(
            _as_float(row.get("epsilon"), f"{context}.epsilon"),
            epsilon,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(matches) != 1:
        available = [row.get("epsilon") for row in rows if isinstance(row, dict)]
        raise ReportError(f"{context} has no unique epsilon={epsilon}; available={available}")
    return dict(matches[0])


def _load_mechanism_metrics(
    paths: Sequence[Path], requested_epsilon: float | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    query_rows: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    epsilon_values: set[float] = set()
    for path in paths:
        for record_index, record in enumerate(_read_jsonl(path), 1):
            if record.get("record_type") not in (None, "query_mechanism"):
                continue
            query_id = str(record.get("query_id", "")).strip()
            if not query_id:
                raise ReportError(f"{path}:{record_index} mechanism row lacks query_id")
            if query_id in seen_queries:
                raise ReportError(f"Mechanism inputs duplicate query {query_id!r}")
            seen_queries.add(query_id)
            epsilon = (
                requested_epsilon
                if requested_epsilon is not None
                else _as_float(
                    record.get("selected_epsilon", record.get("training_epsilon")),
                    f"{path}:{record_index}.selected_epsilon",
                )
            )
            epsilon_values.add(epsilon)
            wrappers = record.get("wrappers")
            if not isinstance(wrappers, list):
                raise ReportError(f"{path}:{record_index} lacks wrappers")
            by_id = {
                str(wrapper.get("wrapper_id")): wrapper
                for wrapper in wrappers
                if isinstance(wrapper, dict)
            }
            missing = sorted(set(WRAPPER_ORDER) - set(by_id))
            if missing:
                raise ReportError(f"{path}:{record_index} lacks wrappers {missing}")
            for wrapper_id in WRAPPER_ORDER:
                wrapper = by_id[wrapper_id]
                raw = wrapper.get("raw")
                if not isinstance(raw, dict):
                    raise ReportError(f"{query_id}/{wrapper_id} lacks raw metrics")
                projected = _find_epsilon_row(
                    wrapper.get("epsilon_sweep"), epsilon, f"{query_id}/{wrapper_id}"
                )
                role = (
                    "neutral_answer_free_reference"
                    if wrapper_id == "neutral"
                    else "answer_free_style_only_control"
                )
                common = {
                    "query_id": query_id,
                    "source": record.get("source", "unknown"),
                    "wrapper": wrapper_id,
                    "control_role": role,
                    "answer_free": True,
                    "style_only_control": wrapper_id != "neutral",
                    "epsilon": epsilon,
                    "generated_tokens": record.get("generated_tokens"),
                    "truncated": record.get("truncated"),
                }
                query_rows.append(
                    {
                        **common,
                        "projection": "raw_privileged_surrogate",
                        "alpha": 1.0,
                        "achieved_mean_kl": raw.get("mean_kl"),
                        "constraint_active": False,
                        "task_logprob_gain": _as_float(
                            raw.get("task_logprob_gain"),
                            f"{query_id}/{wrapper_id}.raw.task_logprob_gain",
                        ),
                        "style_abs_logprob_shift": _as_float(
                            raw.get("style_abs_logprob_shift"),
                            f"{query_id}/{wrapper_id}.raw.style_abs_logprob_shift",
                        ),
                    }
                )
                query_rows.append(
                    {
                        **common,
                        "projection": "trsd_projected",
                        "alpha": _as_float(
                            projected.get("alpha"), f"{query_id}/{wrapper_id}.alpha"
                        ),
                        "achieved_mean_kl": _as_float(
                            projected.get("achieved_mean_kl"),
                            f"{query_id}/{wrapper_id}.achieved_mean_kl",
                        ),
                        "constraint_active": bool(projected.get("constraint_active")),
                        "task_logprob_gain": _as_float(
                            projected.get("task_logprob_gain"),
                            f"{query_id}/{wrapper_id}.task_logprob_gain",
                        ),
                        "style_abs_logprob_shift": _as_float(
                            projected.get("style_abs_logprob_shift"),
                            f"{query_id}/{wrapper_id}.style_abs_logprob_shift",
                        ),
                    }
                )
    if len(seen_queries) < 2:
        raise ReportError(
            "Multi-query mechanism figure requires at least two distinct query records"
        )
    if len(epsilon_values) != 1:
        raise ReportError(f"Mechanism inputs mix projected epsilons: {sorted(epsilon_values)}")

    aggregate_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        grouped[(row["wrapper"], row["projection"])].append(row)
    for wrapper_id in WRAPPER_ORDER:
        for projection in ("raw_privileged_surrogate", "trsd_projected"):
            rows = grouped[(wrapper_id, projection)]
            tasks = [float(row["task_logprob_gain"]) for row in rows]
            styles = [float(row["style_abs_logprob_shift"]) for row in rows]
            alphas = [float(row["alpha"]) for row in rows]
            aggregate_rows.append(
                {
                    "wrapper": wrapper_id,
                    "control_role": rows[0]["control_role"],
                    "answer_free": True,
                    "style_only_control": wrapper_id != "neutral",
                    "projection": projection,
                    "epsilon": next(iter(epsilon_values)),
                    "n_queries": len(rows),
                    "task_logprob_gain_mean": _mean(tasks),
                    "task_logprob_gain_sem": _sem(tasks),
                    "style_abs_logprob_shift_mean": _mean(styles),
                    "style_abs_logprob_shift_sem": _sem(styles),
                    "alpha_mean": _mean(alphas),
                    "constraint_activation_rate": (
                        _mean([float(bool(row["constraint_active"])) for row in rows])
                        if projection == "trsd_projected"
                        else None
                    ),
                }
            )
    epsilon = next(iter(epsilon_values))
    raw_style = [
        float(row["style_abs_logprob_shift"])
        for row in query_rows
        if row["projection"] == "raw_privileged_surrogate"
    ]
    projected_style = [
        float(row["style_abs_logprob_shift"])
        for row in query_rows
        if row["projection"] == "trsd_projected"
    ]
    raw_task = [
        float(row["task_logprob_gain"])
        for row in query_rows
        if row["projection"] == "raw_privileged_surrogate"
    ]
    projected_task = [
        float(row["task_logprob_gain"])
        for row in query_rows
        if row["projection"] == "trsd_projected"
    ]
    summary = {
        "query_count": len(seen_queries),
        "wrappers": list(WRAPPER_ORDER),
        "epsilon": epsilon,
        "all_wrappers_answer_free": True,
        "style_only_controls": ["terse", "verbose"],
        "raw_style_shift_mean": _mean(raw_style),
        "projected_style_shift_mean": _mean(projected_style),
        "raw_task_logprob_gain_mean": _mean(raw_task),
        "projected_task_logprob_gain_mean": _mean(projected_task),
        "style_shift_retention": (
            _mean(projected_style) / _mean(raw_style)
            if _mean(raw_style) not in (None, 0.0)
            else None
        ),
        "task_gain_retention": (
            _mean(projected_task) / _mean(raw_task)
            if _mean(raw_task) not in (None, 0.0)
            else None
        ),
    }
    return query_rows, aggregate_rows, summary


def _load_training_metrics(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = _read_jsonl(path)
    rows: list[dict[str, Any]] = []
    seen_episodes: set[int] = set()
    for index, raw in enumerate(raw_rows, 1):
        if raw.get("branch") != "clean" or raw.get("method_id") != TRSD_METHOD_ID:
            raise ReportError(
                f"{path}:{index} is not current exponential-projection TRSD; "
                "legacy probe/ridge or signed-support runs are not accepted"
            )
        episode = _as_int(raw.get("episode"), f"{path}:{index}.episode")
        if episode in seen_episodes:
            raise ReportError(f"Training journal duplicates episode {episode}")
        seen_episodes.add(episode)
        resource = raw.get("resource_usage")
        if not isinstance(resource, dict):
            resource = {}
        optimizer_step = bool(raw.get("optimizer_step", False))
        alpha_value = raw.get("trust_region_alpha")
        alpha = None if alpha_value is None else _as_float(alpha_value, "trust_region_alpha")
        rows.append(
            {
                "episode": episode,
                "query_id": raw.get("query_id"),
                "source": raw.get("source"),
                "response_tokens": raw.get("response_tokens"),
                "optimizer_step": optimizer_step,
                "no_op": not optimizer_step,
                "episode_seconds": (
                    None
                    if raw.get("episode_seconds") is None
                    else _as_float(raw.get("episode_seconds"), "episode_seconds")
                ),
                "gpu_peak_allocated_bytes": resource.get(
                    "cuda_peak_memory_allocated_bytes"
                ),
                "gpu_peak_delta_bytes": resource.get("cuda_peak_memory_delta_bytes"),
                "gpu_peak_reserved_bytes": resource.get(
                    "cuda_peak_memory_reserved_bytes"
                ),
                "gpu_peak_allocated_gib": _bytes_to_gib(
                    resource.get("cuda_peak_memory_allocated_bytes")
                ),
                "gpu_peak_delta_gib": _bytes_to_gib(
                    resource.get("cuda_peak_memory_delta_bytes")
                ),
                "gpu_peak_reserved_gib": _bytes_to_gib(
                    resource.get("cuda_peak_memory_reserved_bytes")
                ),
                "trust_region_alpha": alpha,
                "trust_region_kl_budget": raw.get("trust_region_kl_budget"),
                "trust_region_achieved_kl": raw.get("trust_region_achieved_kl"),
                "constraint_active": None if alpha is None else alpha < 1.0 - 1e-12,
                "guard_rejected": None,
                "guard_rejection_rate": None,
                "guard_note": "N/A: TRSD uses KL projection, not a proposal/update rejection guard",
            }
        )
    rows.sort(key=lambda row: row["episode"])
    cumulative_steps = 0
    cumulative_no_ops = 0
    cumulative_seconds = 0.0
    for row in rows:
        cumulative_steps += int(row["optimizer_step"])
        cumulative_no_ops += int(row["no_op"])
        if row["episode_seconds"] is not None:
            cumulative_seconds += float(row["episode_seconds"])
        row["cumulative_optimizer_steps"] = cumulative_steps
        row["cumulative_no_ops"] = cumulative_no_ops
        row["cumulative_training_seconds"] = cumulative_seconds

    times = [float(row["episode_seconds"]) for row in rows if row["episode_seconds"] is not None]
    peaks = [float(row["gpu_peak_allocated_gib"]) for row in rows if row["gpu_peak_allocated_gib"] is not None]
    deltas = [float(row["gpu_peak_delta_gib"]) for row in rows if row["gpu_peak_delta_gib"] is not None]
    activations = [
        float(bool(row["constraint_active"]))
        for row in rows
        if row["constraint_active"] is not None
    ]
    summary = {
        "episode_count": len(rows),
        "first_episode": rows[0]["episode"],
        "last_episode": rows[-1]["episode"],
        "actual_optimizer_steps": cumulative_steps,
        "no_op_count": cumulative_no_ops,
        "no_op_rate": cumulative_no_ops / len(rows),
        "total_training_seconds": sum(times) if times else None,
        "mean_seconds_per_episode": _mean(times),
        "seconds_per_optimizer_step": (
            sum(times) / cumulative_steps if times and cumulative_steps else None
        ),
        "max_gpu_peak_allocated_gib": max(peaks) if peaks else None,
        "max_gpu_peak_delta_gib": max(deltas) if deltas else None,
        "constraint_activation_rate": _mean(activations),
        "guard_rejection_rate": None,
        "guard_rejection_note": (
            "N/A: current TRSD has no rejection guard; every recorded optimizer_step "
            "is counted directly and the KL projection controls update magnitude"
        ),
    }
    return rows, summary


def _bytes_to_gib(value: Any) -> float | None:
    if value is None:
        return None
    number = _as_float(value, "GPU memory bytes")
    return number / (1024.0**3)


def _configure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ReportError("report requires matplotlib") from error
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save_figure(figure: Any, out_dir: Path, stem: str) -> None:
    figure.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def _plot_checkpoint_curve(
    plt: Any, curve: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], out_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.1, 4.25), constrained_layout=True)
    episodes = [int(row["episode"]) for row in curve]
    accuracy = [100.0 * float(row["acc1"]) for row in curve]
    axis.plot(
        episodes,
        accuracy,
        color=COLORS["blue"],
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        label="TRSD Acc@1",
    )
    axis.fill_between(episodes, accuracy, [accuracy[0]] * len(accuracy), color=COLORS["blue"], alpha=0.10)
    best_episode = int(summary["best_episode"])
    best_index = episodes.index(best_episode)
    axis.scatter(
        [best_episode], [accuracy[best_index]], marker="*", s=130, color=COLORS["orange"],
        edgecolor="white", linewidth=0.7, zorder=5, label="Best checkpoint"
    )
    metrics = (
        f"LHG: {_fmt_pp(summary['lhg'])}\n"
        f"Normalized AULC: {_fmt_pp(summary['normalized_trapezoidal_aulc'])}\n"
        f"Best−final gap: {_fmt_pp(summary['best_final_gap'])}"
    )
    axis.text(
        0.02, 0.97, metrics, transform=axis.transAxes, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": COLORS["light"], "alpha": 0.95},
    )
    axis.set_xlabel("Persistent self-distillation episodes")
    axis.set_ylabel("Held-out Acc@1 (%)")
    axis.set_title("Checkpoint learning dynamics (short TRSD pilot)")
    axis.legend(loc="best")
    if len(episodes) == 1:
        axis.text(
            0.5, 0.08, "One checkpoint only: learning-curve summaries are N/A",
            transform=axis.transAxes, ha="center", color=COLORS["gray"]
        )
    _save_figure(figure, out_dir, "checkpoint_long_horizon")
    plt.close(figure)


def _plot_mechanism_controls(
    plt: Any,
    query_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    out_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    projections = ("raw_privileged_surrogate", "trsd_projected")
    projection_labels = ("Raw privileged", f"TRSD, ε={summary['epsilon']:g}")
    colors = (COLORS["orange"], COLORS["teal"])
    wrapper_labels = ("Neutral", "Terse†", "Verbose†")
    width = 0.34
    centers = list(range(len(WRAPPER_ORDER)))
    metric_specs = (
        ("style_abs_logprob_shift_mean", "style_abs_logprob_shift_sem", "Style shift |log-ratio|", "Distributed response-shape drift"),
        ("task_logprob_gain_mean", "task_logprob_gain_sem", "Task-token signed log-prob gain", "Task-bearing token signal"),
    )
    lookup = {(row["wrapper"], row["projection"]): row for row in aggregate_rows}
    for axis, (metric, sem_metric, ylabel, title) in zip(axes, metric_specs):
        for projection_index, (projection, label, color) in enumerate(zip(projections, projection_labels, colors)):
            offset = (projection_index - 0.5) * width
            values = [float(lookup[(wrapper, projection)][metric]) for wrapper in WRAPPER_ORDER]
            errors = [float(lookup[(wrapper, projection)][sem_metric]) for wrapper in WRAPPER_ORDER]
            x_positions = [center + offset for center in centers]
            axis.bar(
                x_positions, values, width=width * 0.92, color=color, alpha=0.88,
                edgecolor="white", linewidth=0.6, label=label, yerr=errors,
                capsize=2.5, error_kw={"elinewidth": 0.8, "capthick": 0.8}
            )
            for wrapper_index, wrapper in enumerate(WRAPPER_ORDER):
                per_query = [
                    float(row[metric.replace("_mean", "")])
                    for row in query_rows
                    if row["wrapper"] == wrapper and row["projection"] == projection
                ]
                if per_query:
                    span = min(width * 0.46, 0.015 * max(len(per_query) - 1, 0))
                    for point_index, value in enumerate(per_query):
                        jitter = 0.0 if len(per_query) == 1 else -span + 2.0 * span * point_index / (len(per_query) - 1)
                        axis.scatter(
                            wrapper_index + offset + jitter,
                            value,
                            s=10,
                            color="#1F2937",
                            alpha=0.55,
                            linewidth=0,
                            zorder=4,
                        )
        axis.axhline(0.0, color="#9CA3AF", linewidth=0.8)
        axis.set_xticks(centers, wrapper_labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    axes[0].legend(loc="best")
    figure.suptitle(
        "Answer-free style controls: raw surrogate versus trajectory-KL projection",
        fontsize=12,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        -0.015,
        f"† Terse/verbose change response style only; no target answer, reference solution, or outcome feedback. "
        f"Bars: mean ± SEM over {summary['query_count']} queries; dots: queries.",
        ha="center",
        va="top",
        fontsize=8.2,
        color=COLORS["gray"],
    )
    _save_figure(figure, out_dir, "multiquery_style_controls")
    plt.close(figure)


def _plot_training_efficiency(
    plt: Any, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], out_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.3, 4.1), constrained_layout=True)
    episodes = [int(row["episode"]) for row in rows]

    axes[0].step(
        episodes,
        [int(row["cumulative_optimizer_steps"]) for row in rows],
        where="post",
        color=COLORS["blue"],
        linewidth=2.0,
        label="Optimizer steps",
    )
    axes[0].step(
        episodes,
        [int(row["cumulative_no_ops"]) for row in rows],
        where="post",
        color=COLORS["orange"],
        linewidth=2.0,
        label="No-ops",
    )
    axes[0].set_title("Executed updates")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative count")
    axes[0].legend(loc="best")

    time_pairs = [(row["episode"], row["episode_seconds"]) for row in rows if row["episode_seconds"] is not None]
    if time_pairs:
        axes[1].bar(
            [pair[0] for pair in time_pairs],
            [pair[1] for pair in time_pairs],
            color=COLORS["gold"],
            edgecolor="white",
            linewidth=0.5,
        )
        axes[1].axhline(
            float(summary["mean_seconds_per_episode"]), color=COLORS["navy"],
            linestyle="--", linewidth=1.3, label="Mean"
        )
        axes[1].legend(loc="best")
    else:
        axes[1].text(0.5, 0.5, "Timing not recorded", transform=axes[1].transAxes, ha="center")
    axes[1].set_title("Measured training time")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Seconds")

    peak_pairs = [(row["episode"], row["gpu_peak_allocated_gib"]) for row in rows if row["gpu_peak_allocated_gib"] is not None]
    delta_pairs = [(row["episode"], row["gpu_peak_delta_gib"]) for row in rows if row["gpu_peak_delta_gib"] is not None]
    if peak_pairs:
        axes[2].plot(
            [pair[0] for pair in peak_pairs], [pair[1] for pair in peak_pairs],
            color=COLORS["pink"], marker="o", markersize=3.5, linewidth=1.7,
            label="Peak allocated"
        )
    if delta_pairs:
        axes[2].plot(
            [pair[0] for pair in delta_pairs], [pair[1] for pair in delta_pairs],
            color=COLORS["teal"], marker="o", markersize=3.5, linewidth=1.7,
            label="Peak−baseline"
        )
    if not peak_pairs and not delta_pairs:
        axes[2].text(0.5, 0.5, "GPU memory not recorded", transform=axes[2].transAxes, ha="center")
    else:
        axes[2].legend(loc="best")
    axes[2].set_title("Recorded H100 memory")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("GiB")

    activation = summary["constraint_activation_rate"]
    footer = (
        f"Actual steps={summary['actual_optimizer_steps']}; no-ops={summary['no_op_count']}; "
        f"constraint active={_fmt(None if activation is None else 100.0 * activation, 1)}%; "
        "guard rejection=N/A (TRSD has no rejection guard)."
    )
    figure.suptitle("Actual optimization and resource accounting", fontsize=12, fontweight="semibold")
    figure.text(0.5, -0.02, footer, ha="center", va="top", fontsize=8.4, color=COLORS["gray"])
    _save_figure(figure, out_dir, "training_efficiency_resources")
    plt.close(figure)


def _markdown_report(
    checkpoint: Mapping[str, Any], mechanism: Mapping[str, Any], training: Mapping[str, Any]
) -> str:
    return f"""# Short TRSD empirical report

All values below are recomputed from recorded JSONL artifacts. Missing measurements
are reported as **N/A**; no result is imputed. The accepted training journal is
restricted to `{TRSD_METHOD_ID}`, so legacy probe/ridge results cannot enter this report.

## Checkpoint learning dynamics (short pilot)

| Checkpoints | Baseline Acc@1 | Final Acc@1 | LHG | Normalized AULC | Best−final gap |
|---:|---:|---:|---:|---:|---:|
| {checkpoint['checkpoint_count']} | {_fmt(100.0 * checkpoint['baseline_acc1'], 2)}% | {_fmt(100.0 * checkpoint['final_acc1'], 2)}% | {_fmt_pp(checkpoint['lhg'])} | {_fmt_pp(checkpoint['normalized_trapezoidal_aulc'])} | {_fmt_pp(checkpoint['best_final_gap'])} |

![Checkpoint curve](checkpoint_long_horizon.png)

`LHG = final − first`. Normalized AULC is trapezoidal area of the Acc@1 gain
over the observed episode axis, divided by the observed episode span. Best−final
gap is `max(checkpoint Acc@1) − final Acc@1`.

## Answer-free style controls

| Queries | Projection epsilon | Raw style shift | Projected style shift | Style retention | Raw task gain | Projected task gain | Task retention |
|---:|---:|---:|---:|---:|---:|---:|---:|
| {mechanism['query_count']} | {mechanism['epsilon']:.6g} | {_fmt(mechanism['raw_style_shift_mean'], 5)} | {_fmt(mechanism['projected_style_shift_mean'], 5)} | {_fmt(None if mechanism['style_shift_retention'] is None else 100.0 * mechanism['style_shift_retention'], 1)}% | {_fmt(mechanism['raw_task_logprob_gain_mean'], 5)} | {_fmt(mechanism['projected_task_logprob_gain_mean'], 5)} | {_fmt(None if mechanism['task_gain_retention'] is None else 100.0 * mechanism['task_gain_retention'], 1)}% |

![Style controls](multiquery_style_controls.png)

Neutral, terse, and verbose wrappers are answer-free. Terse and verbose are
explicit **style-only controls**: they contain no target answer, reference
solution, future trajectory, or post-outcome feedback. Bars are query means with
SEM; individual dots are the recorded query values.

## Actual optimization and resource accounting

| Episodes | Optimizer steps | No-ops | Train time | Mean sec/episode | Max peak allocated | Max peak delta | Constraint active | Guard rejection |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| {training['episode_count']} | {training['actual_optimizer_steps']} | {training['no_op_count']} | {_fmt(training['total_training_seconds'], 1)} s | {_fmt(training['mean_seconds_per_episode'], 2)} | {_fmt(training['max_gpu_peak_allocated_gib'], 2)} GiB | {_fmt(training['max_gpu_peak_delta_gib'], 2)} GiB | {_fmt(None if training['constraint_activation_rate'] is None else 100.0 * training['constraint_activation_rate'], 1)}% | N/A |

![Training accounting](training_efficiency_resources.png)

Guard rejection is N/A by design: current TRSD applies a trajectory-level KL
projection and has no proposal/update rejection guard. `optimizer_step` and no-op
counts are read directly from the episode journal.

## Machine-readable artifacts

- `checkpoint_curve.csv`
- `checkpoint_source_accuracy.csv`
- `mechanism_query_wrapper.csv`
- `mechanism_wrapper_summary.csv`
- `training_episode_resources.csv`
- `training_summary.csv`
- `summary.json`
"""


def report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve, source_rows, checkpoint_summary = _load_checkpoint_metrics(args.scored)
    mechanism_rows, mechanism_aggregate, mechanism_summary = _load_mechanism_metrics(
        [Path(path) for path in args.mechanism], args.epsilon
    )
    training_rows, training_summary = _load_training_metrics(Path(args.episodes))

    _write_csv(
        out_dir / "checkpoint_curve.csv",
        curve,
        (
            "episode",
            "n_queries",
            "acc1",
            "acc1_percent",
            "gain_over_first",
            "gain_over_first_pp",
        ),
    )
    _write_csv(
        out_dir / "checkpoint_source_accuracy.csv",
        source_rows,
        ("episode", "source", "n_queries", "correct_count", "acc1", "acc1_percent"),
    )
    _write_csv(
        out_dir / "mechanism_query_wrapper.csv",
        mechanism_rows,
        (
            "query_id",
            "source",
            "wrapper",
            "control_role",
            "answer_free",
            "style_only_control",
            "projection",
            "epsilon",
            "alpha",
            "achieved_mean_kl",
            "constraint_active",
            "task_logprob_gain",
            "style_abs_logprob_shift",
            "generated_tokens",
            "truncated",
        ),
    )
    _write_csv(
        out_dir / "mechanism_wrapper_summary.csv",
        mechanism_aggregate,
        (
            "wrapper",
            "control_role",
            "answer_free",
            "style_only_control",
            "projection",
            "epsilon",
            "n_queries",
            "task_logprob_gain_mean",
            "task_logprob_gain_sem",
            "style_abs_logprob_shift_mean",
            "style_abs_logprob_shift_sem",
            "alpha_mean",
            "constraint_activation_rate",
        ),
    )
    _write_csv(
        out_dir / "training_episode_resources.csv",
        training_rows,
        (
            "episode",
            "query_id",
            "source",
            "response_tokens",
            "optimizer_step",
            "no_op",
            "cumulative_optimizer_steps",
            "cumulative_no_ops",
            "episode_seconds",
            "cumulative_training_seconds",
            "gpu_peak_allocated_bytes",
            "gpu_peak_delta_bytes",
            "gpu_peak_reserved_bytes",
            "gpu_peak_allocated_gib",
            "gpu_peak_delta_gib",
            "gpu_peak_reserved_gib",
            "trust_region_alpha",
            "trust_region_kl_budget",
            "trust_region_achieved_kl",
            "constraint_active",
            "guard_rejection_rate",
            "guard_note",
        ),
    )
    _write_csv(
        out_dir / "training_summary.csv",
        [training_summary],
        tuple(training_summary),
    )

    summary = {
        "schema_version": "trsd-short-empirical-report-v1",
        "method_id": TRSD_METHOD_ID,
        "checkpoint": checkpoint_summary,
        "mechanism": mechanism_summary,
        "training": training_summary,
        "input_artifacts": {
            "episodes": str(Path(args.episodes).resolve()),
            "scored": [
                {"episode": episode, "path": str(path.resolve())}
                for episode, path in args.scored
            ],
            "mechanism": [str(Path(path).resolve()) for path in args.mechanism],
        },
    }
    _write_json(out_dir / "summary.json", summary)
    _atomic_write_text(
        out_dir / "report.md",
        _markdown_report(checkpoint_summary, mechanism_summary, training_summary),
    )

    plt = _configure_matplotlib()
    _plot_checkpoint_curve(plt, curve, checkpoint_summary, out_dir)
    _plot_mechanism_controls(
        plt, mechanism_rows, mechanism_aggregate, mechanism_summary, out_dir
    )
    _plot_training_efficiency(plt, training_rows, training_summary, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    subset = commands.add_parser(
        "prepare-subset", help="Select exactly N aligned query/label rows per source"
    )
    subset.add_argument("--queries", required=True)
    subset.add_argument("--labels", required=True)
    subset.add_argument("--per-source", type=int, required=True)
    subset.add_argument("--sources", help="Optional comma-separated source order")
    subset.add_argument("--out-dir", required=True)
    subset.add_argument("--query-name", default="heldout_queries.jsonl")
    subset.add_argument("--label-name", default="heldout_labels.sealed.jsonl")
    subset.set_defaults(func=prepare_subset)

    stream = commands.add_parser(
        "prepare-stream", help="Bind the first N label-free distillation queries"
    )
    stream.add_argument("--queries", required=True)
    stream.add_argument("--episodes", type=int, required=True)
    stream.add_argument("--output", required=True)
    stream.set_defaults(func=prepare_stream)

    report_parser = commands.add_parser(
        "report", help="Build tables and three figures from recorded artifacts"
    )
    report_parser.add_argument("--episodes", required=True, help="Current TRSD episodes.jsonl")
    report_parser.add_argument(
        "--scored",
        action="append",
        type=_parse_scored_argument,
        required=True,
        metavar="EPISODE:PATH",
        help="Scored checkpoint JSONL; repeat for checkpoints or shards",
    )
    report_parser.add_argument(
        "--mechanism",
        action="append",
        required=True,
        help="Mechanism JSONL; repeat for multiple queries",
    )
    report_parser.add_argument(
        "--epsilon",
        type=float,
        help="Exact projected epsilon to report (otherwise each row's selected epsilon)",
    )
    report_parser.add_argument("--out-dir", required=True)
    report_parser.set_defaults(func=report)
    return parser


def main() -> None:
    try:
        args = build_parser().parse_args()
        args.func(args)
    except ReportError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
