#!/usr/bin/env python3
"""Report runtime, memory, and mechanism statistics for ``timebox12h``.

This is intentionally a small, observational reporter.  It reads completed
episode journal rows, matches both Clean and Self-Proposed Privileged rows to
their online proposal costs, and optionally reads a pipe-delimited ``sacct``
export.  It does not launch or modify jobs.  Historical fixed-prompt
Privileged-SD artifacts are not modified or selected as the main comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "clean-self-distill-timebox-efficiency-v3"
DEFAULT_SLOWDOWN_THRESHOLD = 1.25
DEFAULT_SACCT_COLUMNS = (
    "JobIDRaw",
    "JobName",
    "State",
    "ElapsedRaw",
    "MaxRSS",
    "TRESUsageInMax",
    "TRESUsageInAve",
)
_TRES_VALUE_RE = {
    "gpumem": re.compile(r"(?:^|,)\s*(?:gres/)?gpumem=([^,]+)", re.IGNORECASE),
    "gpuutil": re.compile(r"(?:^|,)\s*(?:gres/)?gpuutil=([^,]+)", re.IGNORECASE),
}
_SIZE_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]?)B?$", re.IGNORECASE)


class TimeboxReportError(ValueError):
    """Raised when a time-box journal cannot support an honest report."""


def _load_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise TimeboxReportError(f"Missing required journal: {path}")
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise TimeboxReportError(f"{path}:{line_number} is not an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TimeboxReportError(f"Cannot read {path}: {exc}") from exc
    if required and not rows:
        raise TimeboxReportError(f"Journal is empty: {path}")
    return rows


def _finite_number(value: Any, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimeboxReportError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise TimeboxReportError(f"{context} must be finite and >= {minimum}")
    return result


def _finite_signed_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimeboxReportError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TimeboxReportError(f"{context} must be finite")
    return result


def _count(value: Any, context: str) -> int:
    number = _finite_number(value, context)
    if not number.is_integer():
        raise TimeboxReportError(f"{context} must be an integer count")
    return int(number)


def _summary(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": fmean(values),
        "median": median(values),
        "total": sum(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _mean_from_summary(value: Mapping[str, Any] | None) -> float | None:
    return None if value is None else float(value["mean"])


def _field_count(
    row: Mapping[str, Any], primary: str, alias: str, context: str
) -> int:
    if primary in row:
        return _count(row[primary], f"{context}.{primary}")
    return _count(row.get(alias, 0), f"{context}.{alias}")


def _branch_summary(
    rows: Sequence[Mapping[str, Any]], *, branch: str
) -> tuple[dict[str, Any], list[float]]:
    if not rows:
        raise TimeboxReportError(f"{branch} episode journal is empty")
    episode_seconds: list[float] = []
    teacher_positions = exposed_positions = compared_positions = exact_positions = 0
    style_error = task_error = 0.0
    style_tokens = task_tokens = 0
    crossings = crossing_eligible = regressions = regression_eligible = 0
    frontier_count = frontier_attainment_count = 0
    frontier_base_margin_sum = 0.0
    frontier_teacher_margin_sum = 0.0
    frontier_margin_gain_sum = 0.0
    ridge_seconds: list[float] = []
    seen_episodes: set[int] = set()
    for index, row in enumerate(rows, 1):
        context = f"{branch} row {index}"
        declared = str(row.get("branch", "")).strip().casefold()
        if declared != branch:
            raise TimeboxReportError(f"{context} declares branch={declared!r}")
        episode = _count(row.get("episode"), f"{context}.episode")
        if episode in seen_episodes:
            raise TimeboxReportError(f"{branch} duplicates episode {episode}")
        seen_episodes.add(episode)
        episode_seconds.append(
            _finite_number(row.get("episode_seconds"), f"{context}.episode_seconds")
        )

        audit = row.get("audit")
        if not isinstance(audit, Mapping):
            raise TimeboxReportError(f"{context}.audit must be an object")
        teacher_positions += _count(
            audit.get("teacher_positions"), f"{context}.audit.teacher_positions"
        )
        exposed_positions += _count(
            audit.get("hindsight_exposed_positions"),
            f"{context}.audit.hindsight_exposed_positions",
        )
        compared_positions += _count(
            audit.get("compared_positions"), f"{context}.audit.compared_positions"
        )
        exact_positions += _count(
            audit.get("exact_context_positions"),
            f"{context}.audit.exact_context_positions",
        )

        partition = row.get("style_task_error")
        if not isinstance(partition, Mapping):
            raise TimeboxReportError(f"{context}.style_task_error must be an object")
        style_error += _finite_number(
            partition.get("style_abs_error_sum"),
            f"{context}.style_task_error.style_abs_error_sum",
        )
        task_error += _finite_number(
            partition.get("task_abs_error_sum"),
            f"{context}.style_task_error.task_abs_error_sum",
        )
        style_tokens += _count(
            partition.get("style_token_count"),
            f"{context}.style_task_error.style_token_count",
        )
        task_tokens += _count(
            partition.get("task_token_count"),
            f"{context}.style_task_error.task_token_count",
        )

        ridge = row.get("ridge_metrics")
        if not isinstance(ridge, Mapping):
            raise TimeboxReportError(f"{context}.ridge_metrics must be an object")
        crossings += _field_count(
            ridge,
            "decision_boundary_crossing_count",
            "db_crossing_count",
            f"{context}.ridge_metrics",
        )
        crossing_eligible += _field_count(
            ridge,
            "decision_boundary_eligible_count",
            "db_eligible_count",
            f"{context}.ridge_metrics",
        )
        regressions += _field_count(
            ridge,
            "decision_boundary_regression_count",
            "regression_count",
            f"{context}.ridge_metrics",
        )
        regression_eligible += _field_count(
            ridge,
            "decision_boundary_regression_eligible_count",
            "regression_eligible_count",
            f"{context}.ridge_metrics",
        )
        comparable = _count(
            ridge.get("frontier_comparable_count", 0),
            f"{context}.ridge_metrics.frontier_comparable_count",
        )
        if comparable:
            base_margin = _finite_signed_number(
                ridge.get("frontier_margin_base_mean"),
                f"{context}.ridge_metrics.frontier_margin_base_mean",
            )
            teacher_margin = _finite_signed_number(
                ridge.get("frontier_margin_teacher_mean"),
                f"{context}.ridge_metrics.frontier_margin_teacher_mean",
            )
            margin_gain = _finite_signed_number(
                ridge.get("frontier_margin_gain_mean"),
                f"{context}.ridge_metrics.frontier_margin_gain_mean",
            )
            attainment = _count(
                ridge.get("frontier_target_margin_attainment_count"),
                f"{context}.ridge_metrics.frontier_target_margin_attainment_count",
            )
            if attainment > comparable:
                raise TimeboxReportError(f"{context} has impossible margin counters")
            frontier_count += comparable
            frontier_attainment_count += attainment
            frontier_base_margin_sum += comparable * base_margin
            frontier_teacher_margin_sum += comparable * teacher_margin
            frontier_margin_gain_sum += comparable * margin_gain
        if "specialization_seconds" in ridge:
            ridge_seconds.append(
                _finite_number(
                    ridge["specialization_seconds"],
                    f"{context}.ridge_metrics.specialization_seconds",
                )
            )

    if exposed_positions > teacher_positions or exact_positions > compared_positions:
        raise TimeboxReportError(f"{branch} has impossible HER/CP counters")
    if crossings > crossing_eligible or regressions > regression_eligible:
        raise TimeboxReportError(f"{branch} has impossible frontier counters")
    style_mean = style_error / style_tokens if style_tokens else None
    task_mean = task_error / task_tokens if task_tokens else None
    core = _summary(episode_seconds)
    assert core is not None
    return (
        {
            "episodes": len(rows),
            "latest_episode": max(seen_episodes),
            "core_episode_seconds": core,
            "core_throughput_episodes_per_hour": 3600.0 / float(core["mean"]),
            "cleanliness": {
                "HER": exposed_positions / teacher_positions if teacher_positions else 0.0,
                "CP": exact_positions / compared_positions if compared_positions else 0.0,
                "teacher_positions": teacher_positions,
                "hindsight_exposed_positions": exposed_positions,
                "compared_positions": compared_positions,
                "exact_context_positions": exact_positions,
            },
            "style_task_error": {
                "style_abs_error_sum": style_error,
                "style_token_count": style_tokens,
                "style_mean_abs_error_per_token": style_mean,
                "task_abs_error_sum": task_error,
                "task_token_count": task_tokens,
                "task_mean_abs_error_per_token": task_mean,
                "style_task_error_ratio": _ratio(style_mean, task_mean),
            },
            "decision_frontier": {
                "crossings": crossings,
                "crossing_eligible": crossing_eligible,
                "crossing_rate": crossings / crossing_eligible if crossing_eligible else 0.0,
                "regressions": regressions,
                "regression_eligible": regression_eligible,
                "regression_rate": regressions / regression_eligible if regression_eligible else 0.0,
            },
            "frontier_margin": {
                "comparable_count": frontier_count,
                "base_mean": (
                    frontier_base_margin_sum / frontier_count if frontier_count else None
                ),
                "teacher_mean": (
                    frontier_teacher_margin_sum / frontier_count if frontier_count else None
                ),
                "gain_mean": (
                    frontier_margin_gain_sum / frontier_count if frontier_count else None
                ),
                "target_attainment_count": frontier_attainment_count,
                "target_attainment_rate": (
                    frontier_attainment_count / frontier_count if frontier_count else None
                ),
            },
            "ridge_specialization_seconds": _summary(ridge_seconds),
        },
        episode_seconds,
    )


def _proposal_summary(
    rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    branch: str,
) -> tuple[dict[str, Any], list[float] | None]:
    by_query: dict[str, float] = {}
    for index, row in enumerate(rows, 1):
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise TimeboxReportError(
                f"{branch} proposal row {index} has no query_id"
            )
        if query_id in by_query:
            raise TimeboxReportError(
                f"{branch} proposal journal duplicates {query_id!r}"
            )
        cost = row.get("cost_audit")
        if not isinstance(cost, Mapping):
            raise TimeboxReportError(
                f"{branch} proposal row {index}.cost_audit must be an object"
            )
        by_query[query_id] = _finite_number(
            cost.get("end_to_end_seconds"),
            f"{branch} proposal row {index}.cost_audit.end_to_end_seconds",
        )

    matched: list[float] = []
    missing: list[str] = []
    episode_query_ids: set[str] = set()
    for index, row in enumerate(episode_rows, 1):
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise TimeboxReportError(f"{branch} row {index} has no query_id")
        episode_query_ids.add(query_id)
        if query_id not in by_query:
            missing.append(query_id)
        else:
            matched.append(by_query[query_id])
    extra = sorted(set(by_query) - episode_query_ids)
    complete = not missing
    return (
        {
            "recorded_proposals": len(rows),
            "matched_completed_episodes": len(matched),
            "missing_completed_episode_proposals": missing,
            "unmatched_recorded_proposals": extra,
            "complete_for_completed_episodes": complete,
            "seconds": _summary(matched),
        },
        matched if complete else None,
    )


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _parse_size_bytes(raw: str | None) -> int | None:
    if raw is None or not raw.strip():
        return None
    match = _SIZE_RE.fullmatch(raw.strip())
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2).upper()
    exponent = "KMGTPE".find(suffix) + 1 if suffix else 0
    return int(number * (1024**exponent))


def _parse_percent(raw: str | None) -> float | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw.strip().rstrip("%"))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _first_direct_field(
    row: Mapping[str, str], normalized: Mapping[str, str], needle: str
) -> tuple[str | None, str | None]:
    for key, value in row.items():
        normalized_key = normalized[key]
        if needle in normalized_key and "tres" not in normalized_key and value.strip():
            return value.strip(), key
    return None, None


def _tres_field(
    row: Mapping[str, str], normalized: Mapping[str, str], resource: str
) -> tuple[str | None, str | None]:
    # Peak memory is most meaningful from *Max; utilization from *Ave.
    preference = ("max", "ave") if resource == "gpumem" else ("ave", "max")
    keys = list(row)
    keys.sort(
        key=lambda key: next(
            (rank for rank, name in enumerate(preference) if name in normalized[key]),
            len(preference),
        )
    )
    for key in keys:
        value = row[key]
        if "tres" not in normalized[key] or not value.strip():
            continue
        match = _TRES_VALUE_RE[resource].search(value)
        if match:
            return match.group(1).strip(), key
    return None, None


def _job_scope(job_name: str) -> str | None:
    folded = job_name.casefold()
    if "proposed" in folded and "priv" in folded:
        return "proposed_privileged"
    if "clean" in folded and "priv" not in folded:
        return "clean"
    if "priv" in folded:
        return "privileged"
    if "heldout" in folded or "probe" in folded:
        return "heldout_probes"
    return None


def _read_sacct(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "supplied": False,
            "accounting_rows": 0,
            "records_with_resource_fields": [],
            "summary_by_scope": {},
        }
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            raw_rows = [row for row in csv.reader(handle, delimiter="|") if any(row)]
    except (OSError, UnicodeDecodeError) as exc:
        raise TimeboxReportError(f"Cannot read Slurm accounting {path}: {exc}") from exc
    if not raw_rows:
        raise TimeboxReportError(f"Slurm accounting file is empty: {path}")
    for row in raw_rows:
        if row and row[-1] == "":
            row.pop()
    first = [_normalize_header(value) for value in raw_rows[0]]
    has_header = bool(first) and (
        first[0] in {"jobid", "jobidraw"} or "maxrss" in first or "jobname" in first
    )
    if has_header:
        headers = [value.strip() for value in raw_rows.pop(0)]
    else:
        headers = list(DEFAULT_SACCT_COLUMNS)
    rows: list[dict[str, str]] = []
    for index, values in enumerate(raw_rows, 1):
        if len(values) > len(headers):
            raise TimeboxReportError(f"Slurm accounting row {index} has too many fields")
        padded = values + [""] * (len(headers) - len(values))
        rows.append(dict(zip(headers, (value.strip() for value in padded))))
    normalized_headers = {key: _normalize_header(key) for key in headers}
    parent_names: dict[str, str] = {}
    for row in rows:
        job_id = next(
            (row[key] for key in headers if normalized_headers[key] in {"jobidraw", "jobid"}),
            "",
        )
        job_name = next(
            (row[key] for key in headers if normalized_headers[key] == "jobname"),
            "",
        )
        if job_id and "." not in job_id and job_name:
            parent_names[job_id] = job_name

    resources: list[dict[str, Any]] = []
    for row in rows:
        job_id = next(
            (row[key] for key in headers if normalized_headers[key] in {"jobidraw", "jobid"}),
            "",
        )
        job_name = next(
            (row[key] for key in headers if normalized_headers[key] == "jobname"),
            "",
        )
        if job_name.casefold() in {"batch", "extern"}:
            job_name = parent_names.get(job_id.split(".", 1)[0], job_name)
        state = next(
            (row[key] for key in headers if normalized_headers[key] == "state"),
            "",
        )
        elapsed = next(
            (
                row[key]
                for key in headers
                if normalized_headers[key] in {"elapsed", "elapsedraw"}
            ),
            "",
        )
        max_rss, max_rss_source = _first_direct_field(
            row, normalized_headers, "maxrss"
        )
        gpumem, gpumem_source = _first_direct_field(
            row, normalized_headers, "gpumem"
        )
        gpuutil, gpuutil_source = _first_direct_field(
            row, normalized_headers, "gpuutil"
        )
        if gpumem is None:
            gpumem, gpumem_source = _tres_field(
                row, normalized_headers, "gpumem"
            )
        if gpuutil is None:
            gpuutil, gpuutil_source = _tres_field(
                row, normalized_headers, "gpuutil"
            )
        if max_rss is None and gpumem is None and gpuutil is None:
            continue
        resources.append(
            {
                "job_id": job_id or None,
                "job_name": job_name or None,
                "scope": _job_scope(job_name),
                "state": state or None,
                "elapsed": elapsed or None,
                "MaxRSS": max_rss,
                "MaxRSS_bytes": _parse_size_bytes(max_rss),
                "MaxRSS_source": max_rss_source,
                "gpumem": gpumem,
                "gpumem_bytes": _parse_size_bytes(gpumem),
                "gpumem_source": gpumem_source,
                "gpuutil": gpuutil,
                "gpuutil_percent": _parse_percent(gpuutil),
                "gpuutil_source": gpuutil_source,
            }
        )

    summary_by_scope: dict[str, Any] = {}
    scopes = sorted({str(row["scope"]) for row in resources if row["scope"] is not None})
    for scope in scopes:
        selected = [row for row in resources if row["scope"] == scope]
        rss = [int(row["MaxRSS_bytes"]) for row in selected if row["MaxRSS_bytes"] is not None]
        gpumem = [int(row["gpumem_bytes"]) for row in selected if row["gpumem_bytes"] is not None]
        gpuutil = [float(row["gpuutil_percent"]) for row in selected if row["gpuutil_percent"] is not None]
        summary_by_scope[scope] = {
            "peak_MaxRSS_bytes": max(rss) if rss else None,
            "peak_gpumem_bytes": max(gpumem) if gpumem else None,
            "mean_gpuutil_percent": fmean(gpuutil) if gpuutil else None,
            "resource_records": len(selected),
        }
    return {
        "supplied": True,
        "accounting_rows": len(rows),
        "records_with_resource_fields": resources,
        "summary_by_scope": summary_by_scope,
    }


def build_timebox_report(
    timebox_dir: str | Path,
    *,
    slurm_accounting: str | Path | None = None,
    slowdown_threshold: float = DEFAULT_SLOWDOWN_THRESHOLD,
) -> dict[str, Any]:
    root = Path(timebox_dir)
    if not math.isfinite(slowdown_threshold) or slowdown_threshold < 1.0:
        raise TimeboxReportError("slowdown_threshold must be finite and >= 1.0")
    clean_rows = _load_jsonl(root / "clean" / "episodes.jsonl")
    proposed_privileged_rows = _load_jsonl(
        root / "proposed_privileged" / "episodes.jsonl"
    )
    clean_proposal_rows = _load_jsonl(
        root / "clean" / "online_proposals.jsonl", required=False
    )
    proposed_privileged_proposal_rows = _load_jsonl(
        root / "proposed_privileged" / "online_proposals.jsonl", required=False
    )
    clean, clean_core_values = _branch_summary(clean_rows, branch="clean")
    proposed_privileged, proposed_privileged_core_values = _branch_summary(
        proposed_privileged_rows, branch="proposed_privileged"
    )
    clean_proposal, clean_proposal_values = _proposal_summary(
        clean_proposal_rows, clean_rows, branch="clean"
    )
    proposed_privileged_proposal, proposed_privileged_proposal_values = (
        _proposal_summary(
            proposed_privileged_proposal_rows,
            proposed_privileged_rows,
            branch="proposed_privileged",
        )
    )

    clean_end_to_end_values = (
        [
            core + proposal_seconds
            for core, proposal_seconds in zip(
                clean_core_values, clean_proposal_values
            )
        ]
        if clean_proposal_values is not None
        else None
    )
    proposed_privileged_end_to_end_values = (
        [
            core + proposal_seconds
            for core, proposal_seconds in zip(
                proposed_privileged_core_values,
                proposed_privileged_proposal_values,
            )
        ]
        if proposed_privileged_proposal_values is not None
        else None
    )
    clean_end_to_end = (
        _summary(clean_end_to_end_values) if clean_end_to_end_values is not None else None
    )
    proposed_privileged_end_to_end = (
        _summary(proposed_privileged_end_to_end_values)
        if proposed_privileged_end_to_end_values is not None
        else None
    )
    clean["end_to_end_episode_seconds"] = clean_end_to_end
    clean["end_to_end_throughput_episodes_per_hour"] = (
        None
        if clean_end_to_end is None
        else 3600.0 / float(clean_end_to_end["mean"])
    )
    proposed_privileged["end_to_end_episode_seconds"] = (
        proposed_privileged_end_to_end
    )
    proposed_privileged["end_to_end_throughput_episodes_per_hour"] = (
        3600.0 / float(proposed_privileged_end_to_end["mean"])
        if proposed_privileged_end_to_end is not None
        else None
    )

    core_ratio = _ratio(
        _mean_from_summary(clean["core_episode_seconds"]),
        _mean_from_summary(proposed_privileged["core_episode_seconds"]),
    )
    end_to_end_ratio = _ratio(
        _mean_from_summary(clean_end_to_end),
        _mean_from_summary(proposed_privileged_end_to_end),
    )
    core_within = core_ratio is not None and core_ratio <= slowdown_threshold
    end_to_end_within = (
        end_to_end_ratio is not None and end_to_end_ratio <= slowdown_threshold
    )
    overall_within = core_within and end_to_end_within
    if core_ratio is None or end_to_end_ratio is None:
        statement = "Runtime comparison is incomplete; no slowdown claim is supported."
    elif overall_within:
        statement = (
            f"Clean versus Self-Proposed Privileged qualifies as 'not much slower' "
            f"under the explicit <= "
            f"{slowdown_threshold:.2f}x rule (core {core_ratio:.3f}x; "
            f"end-to-end {end_to_end_ratio:.3f}x)."
        )
    else:
        statement = (
            f"Observed Clean/Self-Proposed-Privileged runtime exceeds the <= "
            f"{slowdown_threshold:.2f}x rule "
            f"on at least one measure (core {core_ratio:.3f}x; "
            f"end-to-end {end_to_end_ratio:.3f}x); no 'not much slower' claim is made."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "timebox12h",
        "branches": {
            "clean": clean,
            "proposed_privileged": proposed_privileged,
        },
        "proposal_end_to_end_seconds": {
            "clean": clean_proposal,
            "proposed_privileged": proposed_privileged_proposal,
        },
        "comparison": {
            "slowdown_threshold": slowdown_threshold,
            "baseline": "Self-Proposed Privileged-SD",
            "threshold_rule": (
                "Clean/Self-Proposed-Privileged mean seconds <= threshold"
            ),
            "core_slowdown_ratio_clean_over_proposed_privileged": core_ratio,
            "end_to_end_slowdown_ratio_clean_over_proposed_privileged": (
                end_to_end_ratio
            ),
            "core_within_threshold": core_within,
            "end_to_end_within_threshold": end_to_end_within,
            "overall_within_threshold": overall_within,
            "statement": statement,
        },
        "slurm_resources": _read_sacct(
            Path(slurm_accounting) if slurm_accounting is not None else None
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "—"
    number = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or suffix == "TiB":
            return f"{number:.2f} {suffix}"
        number /= 1024.0
    return f"{number:.2f} TiB"


def render_markdown(report: Mapping[str, Any]) -> str:
    branches = report["branches"]
    lines = [
        "# Time-box runtime and resource report",
        "",
        "All timing values are observed completed-journal values; active in-flight work is excluded.",
        "",
        "| Branch | Episodes | Core mean s | Core median s | Core total s | Core ep/h | E2E mean s | E2E total s | E2E ep/h | HER | CP | Style/task error | Crossings | Regressions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("clean", "Clean"),
        ("proposed_privileged", "Self-Proposed Privileged"),
    ):
        branch = branches[key]
        core = branch["core_episode_seconds"]
        e2e = branch["end_to_end_episode_seconds"]
        cleanliness = branch["cleanliness"]
        style = branch["style_task_error"]
        frontier = branch["decision_frontier"]
        lines.append(
            "| {label} | {episodes} | {core_mean} | {core_median} | {core_total} | "
            "{core_rate} | {e2e_mean} | {e2e_total} | {e2e_rate} | {her} | {cp} | "
            "{psr} | {crossings}/{crossing_eligible} | {regressions}/{regression_eligible} |".format(
                label=label,
                episodes=branch["episodes"],
                core_mean=_fmt(core["mean"]),
                core_median=_fmt(core["median"]),
                core_total=_fmt(core["total"]),
                core_rate=_fmt(branch["core_throughput_episodes_per_hour"]),
                e2e_mean=_fmt(None if e2e is None else e2e["mean"]),
                e2e_total=_fmt(None if e2e is None else e2e["total"]),
                e2e_rate=_fmt(branch["end_to_end_throughput_episodes_per_hour"]),
                her=_fmt(cleanliness["HER"]),
                cp=_fmt(cleanliness["CP"]),
                psr=_fmt(style["style_task_error_ratio"]),
                crossings=frontier["crossings"],
                crossing_eligible=frontier["crossing_eligible"],
                regressions=frontier["regressions"],
                regression_eligible=frontier["regression_eligible"],
            )
        )

    proposals = report["proposal_end_to_end_seconds"]
    clean_proposal = proposals["clean"]
    clean_proposal_seconds = clean_proposal["seconds"]
    proposed_privileged_proposal = proposals["proposed_privileged"]
    proposed_privileged_proposal_seconds = proposed_privileged_proposal["seconds"]
    ridge_seconds = branches["clean"]["ridge_specialization_seconds"]
    frontier_margin = branches["clean"]["frontier_margin"]
    lines.extend(
        [
            "",
            "## Proposal and adaptation costs",
            "",
            f"- Clean proposal end-to-end: "
            f"{_fmt(None if clean_proposal_seconds is None else clean_proposal_seconds['mean'])} s mean, "
            f"{_fmt(None if clean_proposal_seconds is None else clean_proposal_seconds['median'])} s median, "
            f"{_fmt(None if clean_proposal_seconds is None else clean_proposal_seconds['total'])} s total "
            f"({clean_proposal['matched_completed_episodes']}/{branches['clean']['episodes']} completed episodes matched).",
            f"- Self-Proposed Privileged proposal end-to-end: "
            f"{_fmt(None if proposed_privileged_proposal_seconds is None else proposed_privileged_proposal_seconds['mean'])} s mean, "
            f"{_fmt(None if proposed_privileged_proposal_seconds is None else proposed_privileged_proposal_seconds['median'])} s median, "
            f"{_fmt(None if proposed_privileged_proposal_seconds is None else proposed_privileged_proposal_seconds['total'])} s total "
            f"({proposed_privileged_proposal['matched_completed_episodes']}/"
            f"{branches['proposed_privileged']['episodes']} completed episodes matched).",
            f"- Ridge specialization: {_fmt(None if ridge_seconds is None else ridge_seconds['mean'])} s mean, "
            f"{_fmt(None if ridge_seconds is None else ridge_seconds['median'])} s median, "
            f"{_fmt(None if ridge_seconds is None else ridge_seconds['total'])} s total.",
            f"- Frontier margin gain: {_fmt(frontier_margin['gain_mean'])} mean across "
            f"{frontier_margin['comparable_count']} comparable frontiers; target margin "
            f"attained on {frontier_margin['target_attainment_count']}/"
            f"{frontier_margin['comparable_count']}.",
            "",
            "## Guarded runtime comparison",
            "",
            f"- Rule: Clean/Self-Proposed-Privileged mean seconds <= "
            f"{report['comparison']['slowdown_threshold']:.2f}x.",
            f"- Core raw ratio: "
            f"{_fmt(report['comparison']['core_slowdown_ratio_clean_over_proposed_privileged'])}x.",
            f"- End-to-end raw ratio: "
            f"{_fmt(report['comparison']['end_to_end_slowdown_ratio_clean_over_proposed_privileged'])}x.",
            f"- {report['comparison']['statement']}",
            "",
            "## Slurm resources",
            "",
        ]
    )
    resources = report["slurm_resources"]
    if not resources["supplied"]:
        lines.append("No Slurm accounting export was supplied.")
    elif not resources["records_with_resource_fields"]:
        lines.append(
            "The accounting export contained no populated MaxRSS, gpumem, or gpuutil fields."
        )
    else:
        lines.extend(
            [
                "| Job | Scope | State | MaxRSS | GPU memory | GPU util |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in resources["records_with_resource_fields"]:
            lines.append(
                f"| {row['job_id'] or '—'} | {row['scope'] or '—'} | {row['state'] or '—'} | "
                f"{row['MaxRSS'] or '—'} | {row['gpumem'] or '—'} | {row['gpuutil'] or '—'} |"
            )
        lines.extend(["", "Peak/mean resource summary:", ""])
        for scope, value in resources["summary_by_scope"].items():
            lines.append(
                f"- {scope}: MaxRSS {_human_bytes(value['peak_MaxRSS_bytes'])}; "
                f"GPU memory {_human_bytes(value['peak_gpumem_bytes'])}; "
                f"GPU util {_fmt(value['mean_gpuutil_percent'])}%."
            )
    return "\n".join(lines).rstrip() + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timebox-dir",
        required=True,
        help=(
            "Directory containing clean/ and proposed_privileged/ for the active "
            "timebox12h run."
        ),
    )
    parser.add_argument(
        "--slurm-accounting",
        help=(
            "Optional pipe-delimited sacct export. Headered input is preferred; "
            "headerless input uses JobIDRaw,JobName,State,ElapsedRaw,MaxRSS,"
            "TRESUsageInMax,TRESUsageInAve."
        ),
    )
    parser.add_argument(
        "--slowdown-threshold",
        type=float,
        default=DEFAULT_SLOWDOWN_THRESHOLD,
    )
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_timebox_report(
        args.timebox_dir,
        slurm_accounting=args.slurm_accounting,
        slowdown_threshold=args.slowdown_threshold,
    )
    json_path = Path(args.json_output)
    markdown_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["comparison"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
