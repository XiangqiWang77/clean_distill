#!/usr/bin/env python3
"""Build the four-method main table for the 64-episode time-box experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "clean-self-distill-timebox-main-table-v1"
SOURCES = ("amc23", "aime24", "aime25")
SCOPES = ("overall", *SOURCES)
CHECKPOINTS = (0, 16, 32, 48, 64)
EXPECTED_SOURCE_COUNTS = {"amc23": 83, "aime24": 30, "aime25": 30}
METHOD_ORDER = ("base", "csd_t", "clean64", "privileged64")
METHOD_LABELS = {
    "base": "Base",
    "csd_t": "CSD-T",
    "clean64": "Clean-SD",
    "privileged64": "Privileged-SD",
}


class MainTableError(ValueError):
    """Raised when scored inputs cannot support the requested table."""


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with source.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise MainTableError(f"{source}:{line_number} is not an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MainTableError(f"Cannot read {source}: {exc}") from exc
    if not rows:
        raise MainTableError(f"Input is empty: {source}")
    return rows


def _number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MainTableError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise MainTableError(f"{context} is outside its finite range")
    return result


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    number = _number(value, context, minimum=float(minimum))
    if not number.is_integer():
        raise MainTableError(f"{context} must be an integer")
    return int(number)


def _binary(value: Any, context: str) -> int:
    result = _number(value, context)
    if result not in (0.0, 1.0):
        raise MainTableError(f"{context} must be binary")
    return int(result)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise MainTableError("Cannot summarize an empty value list")
    return {
        "n": len(values),
        "mean": fmean(values),
        "median": median(values),
        "total": sum(values),
    }


def _load_scored(
    path: str | Path, *, method: str, checkpoint_episode: int
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(_load_jsonl(path), 1):
        context = f"{path} row {index}"
        if raw.get("method") != method:
            raise MainTableError(f"{context} has unexpected method {raw.get('method')!r}")
        if _integer(raw.get("checkpoint_episode"), f"{context}.checkpoint_episode") != checkpoint_episode:
            raise MainTableError(f"{context} has the wrong checkpoint episode")
        profile = str(raw.get("profile", "")).casefold()
        if profile != "acc1":
            # A scorer configured for Mean@4 may co-locate its non-headline rows.
            if profile.startswith("mean"):
                continue
            raise MainTableError(f"{context}.profile is not acc1/meanN")
        if _integer(raw.get("sample_index"), f"{context}.sample_index") != 0:
            raise MainTableError(f"{context} Acc@1 row is not sample 0")
        query_id = str(raw.get("query_id", "")).strip()
        source = str(raw.get("source", "")).strip().casefold()
        problem_sha = str(raw.get("problem_sha256", "")).strip().casefold()
        if not query_id or source not in SOURCES or len(problem_sha) != 64:
            raise MainTableError(f"{context} has invalid query metadata")
        if query_id in selected:
            raise MainTableError(f"{path} duplicates Acc@1 query {query_id!r}")
        row = dict(raw)
        row["source"] = source
        row["correct"] = _binary(raw.get("correct"), f"{context}.correct")
        row["seed"] = _integer(raw.get("seed"), f"{context}.seed")
        selected[query_id] = row
    if not selected:
        raise MainTableError(f"{path} has no Acc@1 rows")
    return selected


def _validate_universe(
    cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
    expected_source_counts: Mapping[str, int],
) -> None:
    base = cells["base"]
    counts = Counter(str(row["source"]) for row in base.values())
    if dict(counts) != dict(expected_source_counts):
        raise MainTableError(
            f"Base source counts {dict(counts)} != expected {dict(expected_source_counts)}"
        )
    base_ids = set(base)
    for name, rows in cells.items():
        if set(rows) != base_ids:
            raise MainTableError(f"{name} query universe differs from Base")
        for query_id, row in rows.items():
            reference = base[query_id]
            for key in ("source", "problem_sha256", "seed"):
                if row.get(key) != reference.get(key):
                    raise MainTableError(f"{name}/{query_id} is not paired on {key}")
            for key in ("temperature", "top_p", "top_k", "max_new_tokens"):
                if key in row and key in reference and row[key] != reference[key]:
                    raise MainTableError(f"{name}/{query_id} changes decoding field {key}")


def _scope_ids(
    rows: Mapping[str, Mapping[str, Any]], scope: str
) -> list[str]:
    if scope == "overall":
        return sorted(rows)
    return sorted(query_id for query_id, row in rows.items() if row["source"] == scope)


def _accuracy(rows: Mapping[str, Mapping[str, Any]], scope: str) -> float:
    ids = _scope_ids(rows, scope)
    if not ids:
        raise MainTableError(f"No queries in scope {scope}")
    return fmean(float(rows[query_id]["correct"]) for query_id in ids)


def _paired_changes(
    base: Mapping[str, Mapping[str, Any]],
    method: Mapping[str, Mapping[str, Any]],
    scope: str,
) -> dict[str, int]:
    ids = _scope_ids(base, scope)
    return {
        "wrong_to_correct": sum(
            base[query_id]["correct"] == 0 and method[query_id]["correct"] == 1
            for query_id in ids
        ),
        "correct_to_wrong": sum(
            base[query_id]["correct"] == 1 and method[query_id]["correct"] == 0
            for query_id in ids
        ),
    }


def _audit(rows: Mapping[str, Mapping[str, Any]], *, applicable: bool) -> dict[str, Any]:
    if not applicable:
        return {"HER": None, "CP": None, "raw_counts": None}
    totals = {
        "teacher_positions": 0,
        "hindsight_exposed_positions": 0,
        "compared_positions": 0,
        "exact_context_positions": 0,
    }
    for query_id, row in rows.items():
        raw = row.get("training_audit")
        if not isinstance(raw, Mapping):
            raise MainTableError(f"{query_id} has no training_audit")
        for key in totals:
            totals[key] += _integer(raw.get(key), f"{query_id}.training_audit.{key}")
    if (
        totals["teacher_positions"] <= 0
        or totals["compared_positions"] <= 0
        or totals["hindsight_exposed_positions"] > totals["teacher_positions"]
        or totals["exact_context_positions"] > totals["compared_positions"]
    ):
        raise MainTableError("Teacher audit counters are empty or impossible")
    return {
        "HER": totals["hindsight_exposed_positions"] / totals["teacher_positions"],
        "CP": totals["exact_context_positions"] / totals["compared_positions"],
        "raw_counts": totals,
    }


def _resource_summary(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    seconds: list[float] = []
    gpu: list[int] = []
    rss: list[int] = []
    for query_id, row in rows.items():
        resource = row.get("resource_usage")
        if not isinstance(resource, Mapping):
            raise MainTableError(f"{query_id} has no resource_usage")
        seconds.append(
            _number(
                resource.get("method_end_to_end_seconds"),
                f"{query_id}.resource_usage.method_end_to_end_seconds",
                minimum=0.0,
            )
        )
        if resource.get("cuda_peak_memory_allocated_bytes") is not None:
            gpu.append(
                _integer(
                    resource["cuda_peak_memory_allocated_bytes"],
                    f"{query_id}.resource_usage.cuda_peak_memory_allocated_bytes",
                )
            )
        if resource.get("process_peak_rss_bytes") is not None:
            rss.append(
                _integer(
                    resource["process_peak_rss_bytes"],
                    f"{query_id}.resource_usage.process_peak_rss_bytes",
                )
            )
    return {
        "seconds_per_query": fmean(seconds),
        "seconds_per_query_median": median(seconds),
        "seconds_total": sum(seconds),
        "peak_cuda_allocated_bytes": max(gpu) if gpu else None,
        "peak_process_rss_bytes": max(rss) if rss else None,
    }


def _journal_training_costs(
    clean_journal: str | Path,
    privileged_journal: str | Path,
    clean_proposals: str | Path,
) -> dict[str, Any]:
    journals: dict[str, list[dict[str, Any]]] = {
        "clean": _load_jsonl(clean_journal),
        "privileged": _load_jsonl(privileged_journal),
    }
    core: dict[str, list[float]] = {}
    ridge_seconds: list[float] = []
    for branch, rows in journals.items():
        episodes = sorted(
            _integer(row.get("episode"), f"{branch}.episode", minimum=1) for row in rows
        )
        if episodes != list(range(1, 65)):
            raise MainTableError(f"{branch} journal must contain exactly episodes 1..64")
        if any(str(row.get("branch", "")).casefold() != branch for row in rows):
            raise MainTableError(f"{branch} journal contains another branch")
        core[branch] = [
            _number(row.get("episode_seconds"), f"{branch}.episode_seconds", minimum=0.0)
            for row in rows
        ]
        if branch == "clean":
            for row in rows:
                ridge = row.get("ridge_metrics")
                if isinstance(ridge, Mapping) and ridge.get("specialization_seconds") is not None:
                    ridge_seconds.append(
                        _number(
                            ridge["specialization_seconds"],
                            "clean.ridge_metrics.specialization_seconds",
                            minimum=0.0,
                        )
                    )

    proposal_by_query: dict[str, float] = {}
    for index, row in enumerate(_load_jsonl(clean_proposals), 1):
        query_id = str(row.get("query_id", "")).strip()
        cost = row.get("cost_audit")
        if not query_id or query_id in proposal_by_query or not isinstance(cost, Mapping):
            raise MainTableError(f"Invalid/duplicate Clean proposal row {index}")
        proposal_by_query[query_id] = _number(
            cost.get("end_to_end_seconds"),
            f"Clean proposal {index}.cost_audit.end_to_end_seconds",
            minimum=0.0,
        )
    clean_rows = journals["clean"]
    clean_ids = [str(row.get("query_id", "")).strip() for row in clean_rows]
    if not all(clean_ids) or set(clean_ids) != set(proposal_by_query) or len(clean_ids) != 64:
        raise MainTableError("Clean proposals do not match the 64 completed episodes")
    proposal_values = [proposal_by_query[query_id] for query_id in clean_ids]
    clean_e2e = [
        episode + proposal
        for episode, proposal in zip(core["clean"], proposal_values)
    ]
    privileged_e2e = core["privileged"]
    clean_mean = fmean(clean_e2e)
    privileged_mean = fmean(privileged_e2e)
    return {
        "clean": {
            "core_episode_seconds": _stats(core["clean"]),
            "proposal_seconds": _stats(proposal_values),
            "ridge_specialization_seconds": _stats(ridge_seconds) if ridge_seconds else None,
            "end_to_end_episode_seconds": _stats(clean_e2e),
        },
        "privileged": {
            "core_episode_seconds": _stats(core["privileged"]),
            "end_to_end_episode_seconds": _stats(privileged_e2e),
        },
        "clean_over_privileged_core_ratio": (
            fmean(core["clean"]) / fmean(core["privileged"])
            if fmean(core["privileged"]) > 0
            else None
        ),
        "clean_over_privileged_end_to_end_ratio": (
            clean_mean / privileged_mean if privileged_mean > 0 else None
        ),
    }


def _load_optional_resource_report(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MainTableError(f"Cannot read resource report {source}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MainTableError("Resource report must be an object")
    resources = value.get("slurm_resources")
    if not isinstance(resources, Mapping):
        raise MainTableError("Resource report has no slurm_resources object")
    summary = resources.get("summary_by_scope")
    if not isinstance(summary, Mapping):
        raise MainTableError("Resource report has no summary_by_scope object")
    return {str(key): dict(item) for key, item in summary.items() if isinstance(item, Mapping)}


def build_main_table_report(
    *,
    base_scored: str | Path,
    csd_t_scored: str | Path,
    clean64_scored: str | Path,
    privileged64_scored: str | Path,
    clean16_scored: str | Path,
    clean32_scored: str | Path,
    clean48_scored: str | Path,
    privileged16_scored: str | Path,
    privileged32_scored: str | Path,
    privileged48_scored: str | Path,
    clean_journal: str | Path,
    privileged_journal: str | Path,
    clean_proposals: str | Path,
    resource_report: str | Path | None = None,
    expected_source_counts: Mapping[str, int] = EXPECTED_SOURCE_COUNTS,
) -> dict[str, Any]:
    cells = {
        "base": _load_scored(base_scored, method="base", checkpoint_episode=0),
        "csd_t": _load_scored(csd_t_scored, method="csd_t", checkpoint_episode=0),
        "clean64": _load_scored(clean64_scored, method="clean_sd", checkpoint_episode=64),
        "privileged64": _load_scored(
            privileged64_scored, method="privileged_sd", checkpoint_episode=64
        ),
        "clean16": _load_scored(clean16_scored, method="clean_sd", checkpoint_episode=16),
        "clean32": _load_scored(clean32_scored, method="clean_sd", checkpoint_episode=32),
        "clean48": _load_scored(clean48_scored, method="clean_sd", checkpoint_episode=48),
        "privileged16": _load_scored(
            privileged16_scored, method="privileged_sd", checkpoint_episode=16
        ),
        "privileged32": _load_scored(
            privileged32_scored, method="privileged_sd", checkpoint_episode=32
        ),
        "privileged48": _load_scored(
            privileged48_scored, method="privileged_sd", checkpoint_episode=48
        ),
    }
    _validate_universe(cells, expected_source_counts)
    base_accuracy = {scope: _accuracy(cells["base"], scope) for scope in SCOPES}
    methods: dict[str, Any] = {}
    for key in METHOD_ORDER:
        rows = cells[key]
        accuracy = {scope: _accuracy(rows, scope) for scope in SCOPES}
        gain = {
            scope: 100.0 * (accuracy[scope] - base_accuracy[scope])
            for scope in SCOPES
        }
        audit = _audit(rows, applicable=key != "base")
        paired = {
            scope: _paired_changes(cells["base"], rows, scope) for scope in SCOPES
        }
        method = {
            "label": METHOD_LABELS[key],
            "checkpoint_episode": int(next(iter(rows.values()))["checkpoint_episode"]),
            "accuracy": accuracy,
            "gain_vs_base_pp": gain,
            "STG_T_pp": gain if key == "csd_t" else None,
            "STG_S_pp": gain if key in {"clean64", "privileged64"} else None,
            "paired_changes_vs_base": paired,
            **audit,
            "HFG_pp": (
                None
                if key == "base"
                else {
                    scope: (1.0 - float(audit["HER"])) * float(audit["CP"]) * gain[scope]
                    for scope in SCOPES
                }
            ),
            "resources": _resource_summary(rows),
        }
        methods[key] = method

    teacher_gain = methods["csd_t"]["gain_vs_base_pp"]
    clean_gain = methods["clean64"]["gain_vs_base_pp"]
    retention: dict[str, float | None] = {}
    retention_reason: dict[str, str | None] = {}
    for scope in SCOPES:
        if teacher_gain[scope] <= 0.0:
            retention[scope] = None
            retention_reason[scope] = "CSD-T gain is not positive"
        else:
            retention[scope] = clean_gain[scope] / teacher_gain[scope]
            retention_reason[scope] = None
    methods["clean64"]["retention"] = retention
    methods["clean64"]["retention_undefined_reason"] = retention_reason
    for key in ("base", "csd_t", "privileged64"):
        methods[key]["retention"] = None
        methods[key]["retention_undefined_reason"] = "not applicable"

    curves: dict[str, dict[int, dict[str, float]]] = {"clean": {}, "privileged": {}}
    for branch in curves:
        curves[branch][0] = dict(base_accuracy)
        for episode in (16, 32, 48):
            curves[branch][episode] = {
                scope: _accuracy(cells[f"{branch}{episode}"], scope)
                for scope in SCOPES
            }
        curves[branch][64] = {
            scope: methods["clean64" if branch == "clean" else "privileged64"][
                "accuracy"
            ][scope]
            for scope in SCOPES
        }

    horizon: dict[str, Any] = {}
    for branch in ("clean", "privileged"):
        stability: dict[str, Any] = {}
        for scope in SCOPES:
            step_deltas = {
                f"{previous}_to_{current}": 100.0
                * (curves[branch][current][scope] - curves[branch][previous][scope])
                for previous, current in zip(CHECKPOINTS, CHECKPOINTS[1:])
            }
            values = list(step_deltas.values())
            negative = [value for value in values if value < 0.0]
            stability[scope] = {
                "step_deltas_pp": step_deltas,
                "negative_step_count": len(negative),
                "largest_drop_pp": abs(min(negative)) if negative else 0.0,
                "step_std_pp": pstdev(values),
            }
        horizon[branch] = {
            "LHG_pp": {
                scope: 100.0 * (curves[branch][64][scope] - curves[branch][0][scope])
                for scope in SCOPES
            },
            "discrete_AULC_pp": {
                scope: fmean(
                    100.0 * (curves[branch][episode][scope] - curves[branch][0][scope])
                    for episode in CHECKPOINTS[1:]
                )
                for scope in SCOPES
            },
            "stability": stability,
        }
    crossover = {
        scope: next(
            (
                episode
                for episode in CHECKPOINTS[1:]
                if curves["clean"][episode][scope]
                > curves["privileged"][episode][scope]
            ),
            None,
        )
        for scope in SCOPES
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "methods": methods,
        "checkpoint_curves": {
            branch: {str(episode): values for episode, values in curve.items()}
            for branch, curve in curves.items()
        },
        "long_horizon": horizon,
        "first_clean_over_privileged_crossover_episode": crossover,
        "definitions": {
            "STG_T_pp": "CSD-T Acc@1 minus Base Acc@1, in percentage points",
            "STG_S_pp": "persistent student Acc@1 minus Base Acc@1, in percentage points",
            "retention": "Clean-SD gain divided by CSD-T gain; undefined unless CSD-T gain > 0",
            "discrete_AULC_pp": "mean gain vs episode 0 at episodes 16, 32, 48, and 64",
            "stability": (
                "successive Acc@1 changes in percentage points; largest_drop_pp is "
                "the magnitude of the largest negative step (0 if none), and "
                "step_std_pp is the population standard deviation of four steps"
            ),
            "HFG_pp": "(1-HER) * CP * gain_vs_Base_pp",
            "seconds_per_query": "mean scored resource_usage.method_end_to_end_seconds",
        },
        "training_costs": _journal_training_costs(
            clean_journal, privileged_journal, clean_proposals
        ),
        "slurm_training_resources": _load_optional_resource_report(resource_report),
        "claim_note": (
            "Descriptive measurements only; no superiority, significance, or stability "
            "claim is inferred by this reporter."
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _gib(value: Any) -> str:
    return "—" if value is None else f"{float(value) / (1024**3):.2f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    methods = report["methods"]
    lines = [
        "# Clean Self-Distillation: 12-hour main table",
        "",
        "| Method | Overall Acc@1 | AMC23 | AIME24 | AIME25 | STG-T pp | STG-S pp | Retention | W→C | C→W | HER | CP | HFG pp | Sec/query | Peak GPU GiB | Peak RSS GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in METHOD_ORDER:
        method = methods[key]
        changes = method["paired_changes_vs_base"]["overall"]
        stg_t = None if method["STG_T_pp"] is None else method["STG_T_pp"]["overall"]
        stg_s = None if method["STG_S_pp"] is None else method["STG_S_pp"]["overall"]
        retention = None if method["retention"] is None else method["retention"]["overall"]
        hfg = None if method["HFG_pp"] is None else method["HFG_pp"]["overall"]
        resources = method["resources"]
        lines.append(
            f"| {method['label']} | {_pct(method['accuracy']['overall'])} | "
            f"{_pct(method['accuracy']['amc23'])} | {_pct(method['accuracy']['aime24'])} | "
            f"{_pct(method['accuracy']['aime25'])} | {_fmt(stg_t)} | {_fmt(stg_s)} | "
            f"{_fmt(retention)} | {changes['wrong_to_correct']} | "
            f"{changes['correct_to_wrong']} | {_fmt(method['HER'])} | {_fmt(method['CP'])} | "
            f"{_fmt(hfg)} | {_fmt(resources['seconds_per_query'])} | "
            f"{_gib(resources['peak_cuda_allocated_bytes'])} | "
            f"{_gib(resources['peak_process_rss_bytes'])} |"
        )

    curves = report["checkpoint_curves"]
    lines.extend(
        [
            "",
            "## Persistent checkpoint curve",
            "",
            "| Episode | Clean overall | Privileged overall | Clean AMC23 | Privileged AMC23 | Clean AIME24 | Privileged AIME24 | Clean AIME25 | Privileged AIME25 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for episode in CHECKPOINTS:
        clean = curves["clean"][str(episode)]
        privileged = curves["privileged"][str(episode)]
        lines.append(
            f"| {episode} | {_pct(clean['overall'])} | {_pct(privileged['overall'])} | "
            f"{_pct(clean['amc23'])} | {_pct(privileged['amc23'])} | "
            f"{_pct(clean['aime24'])} | {_pct(privileged['aime24'])} | "
            f"{_pct(clean['aime25'])} | {_pct(privileged['aime25'])} |"
        )

    horizon = report["long_horizon"]
    crossover = report["first_clean_over_privileged_crossover_episode"]
    training = report["training_costs"]
    slurm = report["slurm_training_resources"] or {}
    clean_slurm = slurm.get("clean", {})
    privileged_slurm = slurm.get("privileged", {})
    lines.extend(
        [
            "",
            "## Long-horizon and cost summary",
            "",
            "| Branch | LHG pp | Discrete AULC pp | Core train sec/episode | End-to-end train sec/episode | Slurm peak GPU GiB | Slurm peak RSS GiB |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Clean | {_fmt(horizon['clean']['LHG_pp']['overall'])} | "
            f"{_fmt(horizon['clean']['discrete_AULC_pp']['overall'])} | "
            f"{_fmt(training['clean']['core_episode_seconds']['mean'])} | "
            f"{_fmt(training['clean']['end_to_end_episode_seconds']['mean'])} | "
            f"{_gib(clean_slurm.get('peak_gpumem_bytes'))} | "
            f"{_gib(clean_slurm.get('peak_MaxRSS_bytes'))} |",
            f"| Privileged | {_fmt(horizon['privileged']['LHG_pp']['overall'])} | "
            f"{_fmt(horizon['privileged']['discrete_AULC_pp']['overall'])} | "
            f"{_fmt(training['privileged']['core_episode_seconds']['mean'])} | "
            f"{_fmt(training['privileged']['end_to_end_episode_seconds']['mean'])} | "
            f"{_gib(privileged_slurm.get('peak_gpumem_bytes'))} | "
            f"{_gib(privileged_slurm.get('peak_MaxRSS_bytes'))} |",
            "",
            f"First overall Clean > Privileged checkpoint: {crossover['overall'] if crossover['overall'] is not None else 'N/A'}.",
            "",
            "## Checkpoint-step stability",
            "",
            "| Branch | Δ0→16 pp | Δ16→32 pp | Δ32→48 pp | Δ48→64 pp | Negative steps | Largest drop pp | Step std pp |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for branch, label in (("clean", "Clean"), ("privileged", "Privileged")):
        stability = horizon[branch]["stability"]["overall"]
        deltas = stability["step_deltas_pp"]
        lines.append(
            f"| {label} | {_fmt(deltas['0_to_16'])} | {_fmt(deltas['16_to_32'])} | "
            f"{_fmt(deltas['32_to_48'])} | {_fmt(deltas['48_to_64'])} | "
            f"{stability['negative_step_count']} | {_fmt(stability['largest_drop_pp'])} | "
            f"{_fmt(stability['step_std_pp'])} |"
        )
    lines.extend(
        [
            "",
            str(report["claim_note"]),
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "base-scored",
        "csd-t-scored",
        "clean64-scored",
        "privileged64-scored",
        "clean16-scored",
        "clean32-scored",
        "clean48-scored",
        "privileged16-scored",
        "privileged32-scored",
        "privileged48-scored",
        "clean-journal",
        "privileged-journal",
        "clean-proposals",
        "json-output",
        "markdown-output",
    ):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--resource-report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_main_table_report(
        base_scored=args.base_scored,
        csd_t_scored=args.csd_t_scored,
        clean64_scored=args.clean64_scored,
        privileged64_scored=args.privileged64_scored,
        clean16_scored=args.clean16_scored,
        clean32_scored=args.clean32_scored,
        clean48_scored=args.clean48_scored,
        privileged16_scored=args.privileged16_scored,
        privileged32_scored=args.privileged32_scored,
        privileged48_scored=args.privileged48_scored,
        clean_journal=args.clean_journal,
        privileged_journal=args.privileged_journal,
        clean_proposals=args.clean_proposals,
        resource_report=args.resource_report,
    )
    json_output = Path(args.json_output)
    markdown_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: report["methods"][key]["accuracy"]["overall"] for key in METHOD_ORDER}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
