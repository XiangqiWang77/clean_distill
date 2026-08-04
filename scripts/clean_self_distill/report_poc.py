#!/usr/bin/env python3
"""Validate and aggregate the four-condition CSD proof-of-concept results.

This reporter is intentionally strict.  It accepts only complete, one-row-per-
query proposal, Task 1, and Task 2 JSONL shards, joins them to the authoritative
benchmark dataset, re-grades every response, re-audits accepted candidates, and
refuses duplicates, missing queries, source/problem drift, dirty source trees,
synthetic/dry-run markers, or incomplete audit/runtime diagnostics.

The generated core table contains exactly Base, Privileged Control, CSD-T, and
CSD-SD.  AMC23 is reported separately and AIME24+AIME25 are combined into the
held-out AIME score used for Gain and HFAG.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clean_self_distill.io import load_query_records
from src.clean_self_distill.propose import target_disjoint_audit
from src.opsd_format import extract_boxed_answer, grade_boxed_answer


DEFAULT_EXPECTED_COUNTS = {"amc23": 83, "aime24": 30, "aime25": 30}
METHODS = ("Base", "Privileged Control", "CSD-T", "CSD-SD")
SCHEMA_VERSION = "clean-self-distill-poc-report-v1"
MAX_CANDIDATE_FOURGRAM_OVERLAP_COUNT = 1
MAX_CANDIDATE_FOURGRAM_OVERLAP_RATE = 0.05


class ReportValidationError(ValueError):
    """Raised when inputs cannot support a trustworthy PoC report."""


def _normalize_source(value: Any) -> str:
    compact = "".join(
        character for character in str(value).strip().lower() if character.isalnum()
    )
    aliases = {
        "amc23": "amc23",
        "amc2023": "amc23",
        "aime24": "aime24",
        "aime2024": "aime24",
        "aime25": "aime25",
        "aime2025": "aime25",
    }
    return aliases.get(compact, compact)


def _problem_sha256(problem: str) -> str:
    return hashlib.sha256(problem.encode("utf-8")).hexdigest()


def _sha256_value(value: Any, context: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ReportValidationError(f"{context}: expected a 64-character SHA-256 digest")
    return digest


def _lookup(row: Mapping[str, Any], *names: str) -> Any:
    """Return the first present field, supporting dotted nested aliases."""
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
        if "." not in name:
            continue
        value: Any = row
        found = True
        for part in name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                found = False
                break
            value = value[part]
        if found and value is not None:
            return value
    return None


def _required(row: Mapping[str, Any], context: str, *names: str) -> Any:
    value = _lookup(row, *names)
    if value is None:
        raise ReportValidationError(
            f"{context}: missing required field; accepted names={list(names)}"
        )
    return value


def _number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ReportValidationError(
            f"{context}: expected a number, got boolean {value!r}"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportValidationError(
            f"{context}: expected a number, got {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ReportValidationError(f"{context}: value must be finite, got {result!r}")
    if minimum is not None and result < minimum:
        raise ReportValidationError(
            f"{context}: value must be >= {minimum}, got {result}"
        )
    return result


def _rate(value: Any, context: str) -> float:
    result = _number(value, context)
    if result < 0.0 or result > 1.0:
        raise ReportValidationError(f"{context}: rate must be in [0, 1], got {result}")
    return result


def _correct(value: Any, context: str) -> float:
    result = _rate(value, context)
    if not (
        math.isclose(result, 0.0, abs_tol=1e-12)
        or math.isclose(result, 1.0, abs_tol=1e-12)
    ):
        raise ReportValidationError(
            f"{context}: Acc@1 requires binary per-query correctness, got {result}"
        )
    return float(result >= 0.5)


def _boolean(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ReportValidationError(
        f"{context}: expected an explicit boolean, got {value!r}"
    )


def _assert_real_row(row: Mapping[str, Any], context: str) -> None:
    for key in (
        "mock",
        "is_mock",
        "synthetic",
        "dry_run",
        "is_dry_run",
        "runtime.mock",
        "runtime.synthetic",
        "runtime.dry_run",
        "provenance.mock",
        "provenance.synthetic",
        "provenance.dry_run",
    ):
        value = _lookup(row, key)
        if value is not None and _boolean(value, f"{context}.{key}"):
            raise ReportValidationError(f"{context}: refuses {key}=true result row")
    run_mode = _lookup(row, "run_mode", "runtime.run_mode", "provenance.run_mode")
    if run_mode is not None and str(run_mode).strip().lower() in {
        "mock",
        "synthetic",
        "dry-run",
        "dry_run",
        "smoke-fixture",
        "fixture",
    }:
        raise ReportValidationError(f"{context}: refuses run_mode={run_mode!r}")
    real_run = _lookup(row, "real_run", "runtime.real_run", "provenance.real_run")
    if real_run is not None and not _boolean(real_run, f"{context}.real_run"):
        raise ReportValidationError(f"{context}: real_run is explicitly false")


def _validate_runtime(row: Mapping[str, Any], context: str) -> dict[str, Any]:
    """Require enough runtime provenance to reject unmarked fixtures/dry runs."""
    runtime = _required(row, context, "runtime", "provenance.runtime")
    if not isinstance(runtime, Mapping):
        raise ReportValidationError(f"{context}.runtime: expected an object")
    timestamp = str(_required(runtime, f"{context}.runtime", "timestamp_utc")).strip()
    git_commit = str(_required(runtime, f"{context}.runtime", "git_commit")).strip()
    model = str(_required(runtime, f"{context}.runtime", "model")).strip()
    revision = (
        str(runtime.get("resolved_model_revision", "")).strip()
        or str(runtime.get("requested_model_revision", "")).strip()
    )
    conda_prefix = str(_required(runtime, f"{context}.runtime", "conda_prefix")).strip()
    torch_version = str(_required(runtime, f"{context}.runtime", "torch")).strip()
    cuda_runtime = str(_required(runtime, f"{context}.runtime", "cuda_runtime")).strip()
    git_dirty = _boolean(
        _required(runtime, f"{context}.runtime", "git_dirty"),
        f"{context}.runtime.git_dirty",
    )
    if not timestamp:
        raise ReportValidationError(f"{context}.runtime.timestamp_utc is empty")
    if len(git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in git_commit.lower()
    ):
        raise ReportValidationError(
            f"{context}.runtime.git_commit must be a full 40-character commit hash"
        )
    if git_dirty:
        raise ReportValidationError(
            f"{context}: formal report refuses runtime.git_dirty=true; "
            "commit or otherwise capture the executed source tree first"
        )
    if not model:
        raise ReportValidationError(f"{context}.runtime.model is empty")
    if not revision:
        raise ReportValidationError(f"{context}.runtime model revision is empty")
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision.lower()
    ):
        raise ReportValidationError(
            f"{context}.runtime model revision must be a pinned 40-character commit hash"
        )
    if not torch_version or not cuda_runtime:
        raise ReportValidationError(
            f"{context}.runtime must record non-empty torch and CUDA runtime versions"
        )
    if Path(conda_prefix).name != "TTT":
        raise ReportValidationError(
            f"{context}.runtime.conda_prefix={conda_prefix!r} is not the required TTT environment"
        )
    cuda_available = _boolean(
        _required(runtime, f"{context}.runtime", "cuda_available"),
        f"{context}.runtime.cuda_available",
    )
    gpu_count = _number(
        _required(runtime, f"{context}.runtime", "gpu_count"),
        f"{context}.runtime.gpu_count",
        minimum=0.0,
    )
    gpus = _required(runtime, f"{context}.runtime", "gpus")
    if not gpu_count.is_integer():
        raise ReportValidationError(f"{context}.runtime.gpu_count must be an integer")
    if not cuda_available or gpu_count < 1:
        raise ReportValidationError(
            f"{context}: result was not produced with an available GPU"
        )
    if not isinstance(gpus, list) or not gpus:
        raise ReportValidationError(f"{context}.runtime.gpus must be a non-empty list")
    if len(gpus) != int(gpu_count):
        raise ReportValidationError(
            f"{context}.runtime.gpu_count={int(gpu_count)} but gpus has {len(gpus)} entries"
        )
    if not all(isinstance(gpu, Mapping) for gpu in gpus):
        raise ReportValidationError(
            f"{context}.runtime.gpus entries must all be objects"
        )
    gpu_names = [str(gpu.get("name", "")).strip() for gpu in gpus]
    if len(gpu_names) != len(gpus) or not all(
        "b200" in name.lower() for name in gpu_names
    ):
        raise ReportValidationError(
            f"{context}: every visible GPU must be an NVIDIA B200, got {gpu_names}"
        )
    gpu_capabilities: list[tuple[int, int]] = []
    for gpu_index, gpu in enumerate(gpus):
        capability = gpu.get("capability")
        if not isinstance(capability, (list, tuple)) or len(capability) != 2:
            raise ReportValidationError(
                f"{context}.runtime.gpus[{gpu_index}].capability must be [major, minor]"
            )
        major = _number(
            capability[0],
            f"{context}.runtime.gpus[{gpu_index}].capability[0]",
            minimum=0.0,
        )
        minor = _number(
            capability[1],
            f"{context}.runtime.gpus[{gpu_index}].capability[1]",
            minimum=0.0,
        )
        if not major.is_integer() or not minor.is_integer():
            raise ReportValidationError(
                f"{context}.runtime.gpus[{gpu_index}].capability must contain integers"
            )
        parsed_capability = (int(major), int(minor))
        if parsed_capability != (10, 0):
            raise ReportValidationError(
                f"{context}: NVIDIA B200 must report CUDA capability (10, 0), "
                f"got {parsed_capability}"
            )
        gpu_capabilities.append(parsed_capability)
    row_model = str(
        _required(row, context, "model", "model_name", "model_path")
    ).strip()
    if row_model != model:
        raise ReportValidationError(
            f"{context}: row model={row_model!r} disagrees with runtime model={model!r}"
        )
    return {
        "git_commit": git_commit.lower(),
        "git_dirty": git_dirty,
        "model": model,
        "model_revision": revision,
        "torch": torch_version,
        "cuda_runtime": cuda_runtime,
        "gpu_capabilities": tuple(gpu_capabilities),
    }


def _read_jsonl(path: Path, task_name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ReportValidationError(
            f"{task_name}: shard does not exist or is not a file: {path}"
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReportValidationError(
                    f"{task_name}: invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ReportValidationError(
                    f"{task_name}: expected an object at {path}:{line_number}"
                )
            value = dict(value)
            value["__input_shard"] = str(path.resolve())
            value["__input_line"] = line_number
            rows.append(value)
    if not rows:
        raise ReportValidationError(f"{task_name}: empty shard: {path}")
    return rows


def _load_shards(
    paths: Sequence[str | Path], task_name: str
) -> dict[str, dict[str, Any]]:
    if not paths:
        raise ReportValidationError(
            f"{task_name}: at least one JSONL shard is required"
        )
    indexed: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        for row in _read_jsonl(path, task_name):
            context = f"{task_name} {row['__input_shard']}:{row['__input_line']}"
            _assert_real_row(row, context)
            query_id = str(_required(row, context, "query_id")).strip()
            if not query_id:
                raise ReportValidationError(f"{context}: query_id is empty")
            if query_id in indexed:
                raise ReportValidationError(
                    f"{task_name}: duplicate query_id={query_id!r} in {origins[query_id]} and {context}"
                )
            indexed[query_id] = row
            origins[query_id] = context
    return indexed


def _load_dataset(
    dataset_path: str | Path, expected_counts: Mapping[str, int]
) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.is_file():
        raise ReportValidationError(f"Dataset does not exist or is not a file: {path}")
    try:
        raw_records = load_query_records(path, include_targets=True)
    except Exception as exc:
        raise ReportValidationError(f"Failed to load dataset {path}: {exc}") from exc
    if not raw_records:
        raise ReportValidationError(f"Dataset has no usable problem rows: {path}")

    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    hashes: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for index, raw in enumerate(raw_records):
        query_id = str(raw.get("query_id", "")).strip()
        problem = str(raw.get("problem", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        source = _normalize_source(raw.get("source", ""))
        context = f"dataset row {index}"
        if not query_id:
            raise ReportValidationError(
                f"{context}: canonical loader produced an empty query_id"
            )
        if not problem:
            raise ReportValidationError(
                f"{context} query_id={query_id!r}: empty problem"
            )
        if not answer:
            raise ReportValidationError(
                f"{context} query_id={query_id!r}: authoritative answer is empty"
            )
        if not source:
            raise ReportValidationError(
                f"{context} query_id={query_id!r}: empty source"
            )
        if query_id in ids:
            raise ReportValidationError(f"Dataset duplicate query_id={query_id!r}")
        digest = _problem_sha256(problem)
        if digest in hashes:
            raise ReportValidationError(
                f"Dataset duplicate problem_sha256={digest}: query_id={hashes[digest]!r} and {query_id!r}"
            )
        ids.add(query_id)
        hashes[digest] = query_id
        counts[source] += 1
        records.append(
            {
                "query_id": query_id,
                "source": source,
                "problem": problem,
                "answer": answer,
                "problem_sha256": digest,
                "dataset_index": index,
            }
        )

    unexpected = sorted(set(counts) - set(expected_counts))
    if unexpected:
        raise ReportValidationError(
            f"Dataset contains unexpected sources {unexpected}; observed counts={dict(counts)}"
        )
    mismatches = {
        source: {"expected": int(expected), "observed": int(counts.get(source, 0))}
        for source, expected in expected_counts.items()
        if counts.get(source, 0) != expected
    }
    if mismatches:
        raise ReportValidationError(f"Dataset source-count mismatch: {mismatches}")
    return records


def _row_problem_hash(row: Mapping[str, Any], context: str) -> str:
    supplied = _lookup(
        row, "problem_sha256", "query_problem_sha256", "target_problem_sha256"
    )
    problem = _lookup(row, "problem", "query", "target_problem")
    if supplied is None and problem is None:
        raise ReportValidationError(
            f"{context}: problem_sha256 is required (or include problem text to derive it)"
        )
    derived = _problem_sha256(str(problem).strip()) if problem is not None else None
    if supplied is None:
        return str(derived)
    digest = _sha256_value(supplied, f"{context}.problem_sha256")
    if derived is not None and digest != derived:
        raise ReportValidationError(
            f"{context}: problem text hashes to {derived}, not supplied problem_sha256={digest}"
        )
    return digest


def _validate_proposal_rows(
    rows: Mapping[str, dict[str, Any]],
    dataset: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bind proposals to targets and independently re-audit every accepted candidate."""
    expected_ids = {record["query_id"] for record in dataset}
    actual_ids = set(rows)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ReportValidationError(
            "proposal: query coverage mismatch: "
            f"missing={missing[:10]}{'...' if len(missing) > 10 else ''}, "
            f"extra={extra[:10]}{'...' if len(extra) > 10 else ''}"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for record in dataset:
        query_id = record["query_id"]
        row = rows[query_id]
        context = f"proposal query_id={query_id!r}"
        source = _normalize_source(_required(row, context, "source", "data_source"))
        if source != record["source"]:
            raise ReportValidationError(
                f"{context}: source={source!r} does not match dataset "
                f"source={record['source']!r}"
            )
        digest = _row_problem_hash(row, context)
        if digest != record["problem_sha256"]:
            raise ReportValidationError(
                f"{context}: problem_sha256={digest} does not match dataset "
                f"problem_sha256={record['problem_sha256']}"
            )
        runtime_signature = _validate_runtime(row, context)
        candidates = _required(row, context, "specialization_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ReportValidationError(
                f"{context}.specialization_candidates must be a non-empty list"
            )
        declared_count = _lookup(row, "candidate_count")
        if declared_count is not None:
            count = _number(declared_count, f"{context}.candidate_count", minimum=1.0)
            if not count.is_integer() or int(count) != len(candidates):
                raise ReportValidationError(
                    f"{context}: candidate_count={declared_count!r} does not match "
                    f"accepted candidate list length={len(candidates)}"
                )

        candidate_audits: list[dict[str, Any]] = []
        seen_problems: set[str] = set()
        for index, candidate in enumerate(candidates):
            candidate_context = f"{context}.specialization_candidates[{index}]"
            if not isinstance(candidate, Mapping):
                raise ReportValidationError(f"{candidate_context}: expected an object")
            candidate_problem = str(
                _required(candidate, candidate_context, "problem")
            ).strip()
            if not candidate_problem:
                raise ReportValidationError(f"{candidate_context}.problem is empty")
            normalized_problem = " ".join(candidate_problem.casefold().split())
            if normalized_problem in seen_problems:
                raise ReportValidationError(
                    f"{candidate_context}: duplicate accepted candidate problem"
                )
            seen_problems.add(normalized_problem)
            if not _boolean(
                _required(candidate, candidate_context, "verifier_accepted"),
                f"{candidate_context}.verifier_accepted",
            ):
                raise ReportValidationError(
                    f"{candidate_context}: candidate was not accepted by the verifier"
                )

            audit = target_disjoint_audit(record["problem"], candidate_problem)
            if audit["literal_overlap_count"] != 0:
                raise ReportValidationError(
                    f"{candidate_context}: target-disjoint audit found shared "
                    f"numeric/entity literals (count={audit['literal_overlap_count']}, "
                    f"numbers={audit['shared_target_numbers']}, "
                    f"entities={audit['shared_target_entities']})"
                )
            if (
                audit["fourgram_overlap_count"]
                > MAX_CANDIDATE_FOURGRAM_OVERLAP_COUNT
                or audit["fourgram_overlap_rate"]
                > MAX_CANDIDATE_FOURGRAM_OVERLAP_RATE
            ):
                raise ReportValidationError(
                    f"{candidate_context}: target-disjoint audit found excessive "
                    f"4-gram overlap (count={audit['fourgram_overlap_count']}, "
                    f"rate={audit['fourgram_overlap_rate']})"
                )
            candidate_audits.append(audit)

        normalized[query_id] = {
            "source": source,
            "problem_sha256": digest,
            "runtime_signature": runtime_signature,
            "candidate_count": len(candidates),
            "candidate_audits": candidate_audits,
        }
    return normalized


def _field_number(
    row: Mapping[str, Any],
    context: str,
    names: Sequence[str],
    *,
    minimum: float | None = None,
) -> float:
    return _number(
        _required(row, context, *names), f"{context}.{names[0]}", minimum=minimum
    )


def _audit_values(row: Mapping[str, Any], context: str) -> tuple[float, float, float]:
    her = _rate(
        _required(
            row,
            context,
            "hindsight_exposure_rate",
            "HER",
            "her",
            "hindsight/hindsight_exposure_rate",
        ),
        f"{context}.hindsight_exposure_rate",
    )
    cpp = _rate(
        _required(
            row,
            context,
            "context_prefix_parity",
            "context_parity_rate",
            "CPP",
            "cpp",
            "hindsight/context_parity_rate",
        ),
        f"{context}.context_prefix_parity",
    )
    computed = (1.0 - her) * cpp
    supplied = _lookup(row, "hindsight_free_score", "HFS", "hfs")
    if supplied is not None:
        supplied_rate = _rate(supplied, f"{context}.hindsight_free_score")
        if not math.isclose(supplied_rate, computed, rel_tol=1e-9, abs_tol=1e-9):
            raise ReportValidationError(
                f"{context}: hindsight_free_score={supplied_rate} disagrees with "
                f"(1-HER)*CPP={computed}"
            )
    return her, cpp, computed


def _adaptation_values(
    row: Mapping[str, Any], context: str, task_name: str
) -> dict[str, float]:
    support = _field_number(
        row,
        context,
        (
            "proposal_end_to_end_seconds",
            "proposal_seconds",
            "support_generation_seconds",
            "candidate_generation_seconds",
        ),
        minimum=0.0,
    )
    specialization = _field_number(
        row,
        context,
        ("specialization_seconds", "ridge_specialization_seconds", "ridge_seconds"),
        minimum=0.0,
    )
    distillation = 0.0
    if task_name == "task2":
        distillation = _field_number(
            row,
            context,
            ("distillation_seconds", "student_distillation_seconds"),
            minimum=0.0,
        )
    component_total = support + specialization + distillation
    supplied_total = _lookup(
        row, "total_adaptation_seconds", "adaptation_total_seconds"
    )
    total = (
        _number(supplied_total, f"{context}.total_adaptation_seconds", minimum=0.0)
        if supplied_total is not None
        else component_total
    )
    if not math.isclose(total, component_total, rel_tol=1e-6, abs_tol=1e-6):
        raise ReportValidationError(
            f"{context}: total_adaptation_seconds={total} does not equal stage sum={component_total}"
        )
    return {
        "support_generation_seconds": support,
        "specialization_seconds": specialization,
        "distillation_seconds": distillation,
        "total_adaptation_seconds": total,
    }


def _diagnostics(
    row: Mapping[str, Any], context: str, prefix: str, *, required: bool = True
) -> tuple[float | None, bool | None]:
    token_value = _lookup(
        row,
        f"{prefix}_generated_tokens",
        f"{prefix}_output_tokens",
        f"{prefix}_tokens",
    )
    truncated_value = _lookup(row, f"{prefix}_truncated", f"{prefix}_was_truncated")
    if token_value is None and truncated_value is None and not required:
        return None, None
    if token_value is None or truncated_value is None:
        raise ReportValidationError(
            f"{context}: {prefix} diagnostics require both generated tokens and truncation flag"
        )
    tokens = _number(token_value, f"{context}.{prefix}_generated_tokens", minimum=0.0)
    truncated = _boolean(truncated_value, f"{context}.{prefix}_truncated")
    return tokens, truncated


def _validate_acc1_artifacts(
    row: Mapping[str, Any], context: str, prefixes: Sequence[str]
) -> None:
    eval_samples = _lookup(row, "eval_samples", "runtime.eval_samples")
    if eval_samples is not None:
        samples = _number(eval_samples, f"{context}.eval_samples", minimum=1.0)
        if not samples.is_integer() or int(samples) != 1:
            raise ReportValidationError(
                f"{context}: expected eval_samples=1 for Acc@1, got {eval_samples!r}"
            )
    for prefix in prefixes:
        responses = _required(row, context, f"{prefix}_responses", f"{prefix}_outputs")
        if not isinstance(responses, list) or len(responses) != 1:
            raise ReportValidationError(
                f"{context}.{prefix}_responses must contain exactly one Acc@1 response"
            )


def _authoritative_correctness(
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    context: str,
    prefixes: Sequence[str],
) -> dict[str, float]:
    """Re-grade every Acc@1 response exactly as ``train_eval._grade_response`` does."""
    declared_reference = str(
        _required(row, context, "reference_answer")
    ).strip()
    authoritative_answer = str(record["answer"]).strip()
    if declared_reference != authoritative_answer:
        raise ReportValidationError(
            f"{context}: reference_answer={declared_reference!r} disagrees with "
            f"authoritative answer={authoritative_answer!r}"
        )

    result: dict[str, float] = {}
    for prefix in prefixes:
        responses = _required(row, context, f"{prefix}_responses", f"{prefix}_outputs")
        # Shape was already checked by _validate_task_protocol; keep this helper
        # independently safe because it establishes the table's correctness.
        if not isinstance(responses, list) or len(responses) != 1:
            raise ReportValidationError(
                f"{context}.{prefix}_responses must contain exactly one Acc@1 response"
            )
        prediction = extract_boxed_answer(str(responses[0]))
        regraded = float(grade_boxed_answer(prediction, authoritative_answer))
        declared = _correct(
            _required(row, context, f"{prefix}_correct"),
            f"{context}.{prefix}_correct",
        )
        if not math.isclose(declared, regraded, abs_tol=1e-12):
            raise ReportValidationError(
                f"{context}: declared {prefix}_correct={declared} disagrees with "
                f"authoritative regrade={regraded} (parsed_answer={prediction!r})"
            )
        result[prefix] = regraded
    return result


def _validate_task_protocol(
    row: Mapping[str, Any], context: str, task_name: str
) -> dict[str, Any]:
    if task_name == "task1":
        marker = str(_required(row, context, "stage", "task")).strip().lower()
        if marker not in {"task1_fast_teacher", "task1", "csd-t", "csd_t"}:
            raise ReportValidationError(
                f"{context}: not a Task 1/CSD-T row: {marker!r}"
            )
        privileged_her = _rate(
            _required(row, context, "privileged_hindsight_exposure_rate"),
            f"{context}.privileged_hindsight_exposure_rate",
        )
        privileged_cpp = _rate(
            _required(row, context, "privileged_context_prefix_parity"),
            f"{context}.privileged_context_prefix_parity",
        )
        privileged_hfs = _rate(
            _required(row, context, "privileged_hindsight_free_score"),
            f"{context}.privileged_hindsight_free_score",
        )
        if not math.isclose(privileged_her, 1.0, abs_tol=1e-12):
            raise ReportValidationError(
                f"{context}: privileged control must have HER=1"
            )
        if not math.isclose(privileged_cpp, 0.0, abs_tol=1e-12):
            raise ReportValidationError(
                f"{context}: privileged control must have CPP=0"
            )
        if not math.isclose(privileged_hfs, 0.0, abs_tol=1e-12):
            raise ReportValidationError(
                f"{context}: privileged control must have HFS=0"
            )
        update_norm = _field_number(
            row,
            context,
            ("update_frobenius_norm", "ridge_update_frobenius_norm", "update_norm"),
            minimum=0.0,
        )
        adapter_rank = _field_number(
            row, context, ("adapter_rank", "ridge_rank"), minimum=1.0
        )
        if update_norm <= 0.0 or not adapter_rank.is_integer():
            raise ReportValidationError(
                f"{context}: CSD-T requires a nonzero ridge update and integral positive rank"
            )
        _validate_acc1_artifacts(row, context, ("base", "privileged", "teacher"))
        return {
            "protocol_no_op": False,
            "update_frobenius_norm": update_norm,
            "steps_completed": None,
        }

    marker = str(_required(row, context, "task", "stage")).strip().lower()
    if marker not in {"task2_clean_distillation", "task2", "csd-sd", "csd_sd"}:
        raise ReportValidationError(f"{context}: not a Task 2/CSD-SD row: {marker!r}")
    if not _boolean(
        _required(row, context, "teacher_destroyed_before_student_evaluation"),
        f"{context}.teacher_destroyed_before_student_evaluation",
    ):
        raise ReportValidationError(
            f"{context}: ridge teacher was not destroyed before student evaluation"
        )
    if not _boolean(
        _required(row, context, "student_reset_verified"),
        f"{context}.student_reset_verified",
    ):
        raise ReportValidationError(
            f"{context}: query-local student reset was not verified"
        )
    update_norm = _field_number(
        row,
        context,
        ("student_update_frobenius_norm", "student_update_norm"),
        minimum=0.0,
    )
    steps = _field_number(row, context, ("distillation_steps_completed",), minimum=0.0)
    if not steps.is_integer():
        raise ReportValidationError(
            f"{context}: distillation_steps_completed must be an integer"
        )
    trace = _required(row, context, "distillation_trace")
    if not isinstance(trace, list) or len(trace) != int(steps):
        raise ReportValidationError(
            f"{context}: distillation_trace length does not match completed steps"
        )
    if any(
        not isinstance(step, Mapping)
        or not _boolean(
            step.get("same_prefix"), f"{context}.distillation_trace.same_prefix"
        )
        for step in trace
    ):
        raise ReportValidationError(
            f"{context}: distillation trace contains prefix mismatch"
        )
    _validate_acc1_artifacts(row, context, ("base", "teacher", "distilled"))
    return {
        "protocol_no_op": bool(update_norm == 0.0 or int(steps) == 0),
        "update_frobenius_norm": update_norm,
        "steps_completed": int(steps),
    }


def _validate_task_rows(
    task_name: str,
    rows: Mapping[str, dict[str, Any]],
    dataset: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected_ids = {record["query_id"] for record in dataset}
    actual_ids = set(rows)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ReportValidationError(
            f"{task_name}: query coverage mismatch: missing={missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}, extra={extra[:10]}"
            f"{'...' if len(extra) > 10 else ''}"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for record in dataset:
        query_id = record["query_id"]
        row = rows[query_id]
        context = f"{task_name} query_id={query_id!r}"
        source = _normalize_source(_required(row, context, "source", "data_source"))
        if source != record["source"]:
            raise ReportValidationError(
                f"{context}: source={source!r} does not match dataset source={record['source']!r}"
            )
        digest = _row_problem_hash(row, context)
        if digest != record["problem_sha256"]:
            raise ReportValidationError(
                f"{context}: problem_sha256={digest} does not match dataset "
                f"problem_sha256={record['problem_sha256']}"
            )
        runtime_signature = _validate_runtime(row, context)
        protocol = _validate_task_protocol(row, context, task_name)
        prefixes = (
            ("base", "privileged", "teacher")
            if task_name == "task1"
            else ("base", "teacher", "distilled")
        )
        condition_correct = _authoritative_correctness(
            row, record, context, prefixes
        )
        if task_name == "task1":
            # Keep the canonical condition ordering used by downstream outputs.
            condition_correct = {
                "base": condition_correct["base"],
                "teacher": condition_correct["teacher"],
                "privileged": condition_correct["privileged"],
            }
        else:
            condition_correct = {
                "base": condition_correct["base"],
                "teacher": condition_correct["teacher"],
                "distilled": condition_correct["distilled"],
            }

        her, cpp, hfs = _audit_values(row, context)
        timing = _adaptation_values(row, context, task_name)
        base_tokens, base_truncated = _diagnostics(row, context, "base")
        teacher_tokens, teacher_truncated = _diagnostics(row, context, "teacher")
        distilled_tokens: float | None = None
        distilled_truncated: bool | None = None
        privileged_tokens: float | None = None
        privileged_truncated: bool | None = None
        if task_name == "task2":
            distilled_tokens, distilled_truncated = _diagnostics(
                row, context, "distilled"
            )
        else:
            privileged_tokens, privileged_truncated = _diagnostics(
                row, context, "privileged"
            )

        normalized[query_id] = {
            "source": source,
            "problem_sha256": digest,
            "runtime_signature": runtime_signature,
            "correct": condition_correct,
            "her": her,
            "cpp": cpp,
            "hfs": hfs,
            "protocol": protocol,
            "timing": timing,
            "diagnostics": {
                "base": {"generated_tokens": base_tokens, "truncated": base_truncated},
                "teacher": {
                    "generated_tokens": teacher_tokens,
                    "truncated": teacher_truncated,
                },
                "privileged": {
                    "generated_tokens": privileged_tokens,
                    "truncated": privileged_truncated,
                },
                "distilled": {
                    "generated_tokens": distilled_tokens,
                    "truncated": distilled_truncated,
                },
            },
        }
    return normalized


def _clean_raw_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("__input_")}


def _optional_number(row: Mapping[str, Any], *names: str) -> float | None:
    value = _lookup(row, *names)
    if value is None:
        return None
    return _number(value, names[0], minimum=0.0)


def _merge_rows(
    dataset: Sequence[dict[str, Any]],
    proposal_rows: Mapping[str, dict[str, Any]],
    task1_rows: Mapping[str, dict[str, Any]],
    task2_rows: Mapping[str, dict[str, Any]],
    proposal_normalized: Mapping[str, dict[str, Any]],
    task1_normalized: Mapping[str, dict[str, Any]],
    task2_normalized: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for record in dataset:
        query_id = record["query_id"]
        proposal = proposal_rows[query_id]
        raw1 = task1_rows[query_id]
        raw2 = task2_rows[query_id]
        norm1 = task1_normalized[query_id]
        norm2 = task2_normalized[query_id]
        for key in ("base", "teacher"):
            if not math.isclose(
                norm1["correct"][key], norm2["correct"][key], abs_tol=1e-12
            ):
                raise ReportValidationError(
                    f"query_id={query_id!r}: {key}_correct differs between task1 "
                    f"({norm1['correct'][key]}) and task2 ({norm2['correct'][key]})"
                )

        privileged_adaptation = _optional_number(
            raw1, "privileged_adaptation_seconds", "privileged_total_adaptation_seconds"
        )
        if privileged_adaptation is None:
            privileged_adaptation = 0.0
        conditions = {
            "Base": {
                "correct": norm1["correct"]["base"],
                "protocol_no_op": None,
                "HER": None,
                "CPP": None,
                "HFS": None,
                "adaptation_seconds": 0.0,
                **norm1["diagnostics"]["base"],
            },
            "Privileged Control": {
                "correct": norm1["correct"]["privileged"],
                "protocol_no_op": None,
                "HER": 1.0,
                "CPP": 0.0,
                "HFS": 0.0,
                "adaptation_seconds": privileged_adaptation,
                **norm1["diagnostics"]["privileged"],
            },
            "CSD-T": {
                "correct": norm1["correct"]["teacher"],
                "protocol_no_op": False,
                "HER": norm1["her"],
                "CPP": norm1["cpp"],
                "HFS": norm1["hfs"],
                "adaptation_seconds": norm1["timing"]["total_adaptation_seconds"],
                **norm1["diagnostics"]["teacher"],
            },
            "CSD-SD": {
                "correct": norm2["correct"]["distilled"],
                "protocol_no_op": norm2["protocol"]["protocol_no_op"],
                "HER": norm2["her"],
                "CPP": norm2["cpp"],
                "HFS": norm2["hfs"],
                "adaptation_seconds": norm2["timing"]["total_adaptation_seconds"],
                **norm2["diagnostics"]["distilled"],
            },
        }
        merged.append(
            {
                "schema_version": SCHEMA_VERSION,
                **record,
                "proposal_shard": proposal["__input_shard"],
                "proposal_line": proposal["__input_line"],
                "proposal_candidate_audits": proposal_normalized[query_id][
                    "candidate_audits"
                ],
                "task1_shard": raw1["__input_shard"],
                "task1_line": raw1["__input_line"],
                "task2_shard": raw2["__input_shard"],
                "task2_line": raw2["__input_line"],
                "conditions": conditions,
                "proposal_artifact": _clean_raw_row(proposal),
                "task1_artifact": _clean_raw_row(raw1),
                "task2_artifact": _clean_raw_row(raw2),
            }
        )
    return merged


def _condition_scope(rows: Sequence[dict[str, Any]], method: str) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "correct": 0,
            "accuracy": None,
            "HER": None,
            "CPP": None,
            "HFS": None,
            "mean_adaptation_seconds": None,
            "truncation_count": None,
            "mean_generated_tokens": None,
            "protocol_no_op_count": None,
            "protocol_no_op_rate": None,
        }
    values = [row["conditions"][method] for row in rows]
    accuracy = fmean(float(value["correct"]) for value in values)

    def optional_mean(field: str) -> float | None:
        present = [value[field] for value in values if value[field] is not None]
        if present and len(present) != len(values):
            raise ReportValidationError(
                f"{method}: {field} is present for only {len(present)}/{len(values)} rows"
            )
        return fmean(float(value) for value in present) if present else None

    token_values = [value["generated_tokens"] for value in values]
    if any(value is None for value in token_values) and not all(
        value is None for value in token_values
    ):
        raise ReportValidationError(
            f"{method}: generated-token diagnostics are only partially present"
        )
    truncation_values = [value["truncated"] for value in values]
    if any(value is None for value in truncation_values) and not all(
        value is None for value in truncation_values
    ):
        raise ReportValidationError(
            f"{method}: truncation diagnostics are only partially present"
        )
    no_op_values = [value["protocol_no_op"] for value in values]
    if any(value is None for value in no_op_values) and not all(
        value is None for value in no_op_values
    ):
        raise ReportValidationError(
            f"{method}: protocol no-op diagnostics are only partially present"
        )

    her = optional_mean("HER")
    cpp = optional_mean("CPP")
    if method == "Privileged Control":
        hfs = 0.0
    elif her is None or cpp is None:
        hfs = None
    else:
        hfs = (1.0 - her) * cpp
    return {
        "n": len(rows),
        "correct": int(sum(float(value["correct"]) for value in values)),
        "accuracy": accuracy,
        "HER": her,
        "CPP": cpp,
        "HFS": hfs,
        "mean_adaptation_seconds": optional_mean("adaptation_seconds"),
        "truncation_count": (
            int(sum(bool(value) for value in truncation_values))
            if truncation_values and truncation_values[0] is not None
            else None
        ),
        "mean_generated_tokens": (
            fmean(float(value) for value in token_values)
            if token_values and token_values[0] is not None
            else None
        ),
        "protocol_no_op_count": (
            int(sum(bool(value) for value in no_op_values))
            if no_op_values and no_op_values[0] is not None
            else None
        ),
        "protocol_no_op_rate": (
            fmean(float(bool(value)) for value in no_op_values)
            if no_op_values and no_op_values[0] is not None
            else None
        ),
    }


def _aggregate(merged: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scopes = {
        "amc23": [row for row in merged if row["source"] == "amc23"],
        "aime": [row for row in merged if row["source"] in {"aime24", "aime25"}],
        "overall": list(merged),
    }
    result: dict[str, Any] = {method: {} for method in METHODS}
    for method in METHODS:
        for scope, rows in scopes.items():
            result[method][scope] = _condition_scope(rows, method)

    for scope in scopes:
        base_accuracy = result["Base"][scope]["accuracy"]
        for method in METHODS:
            metrics = result[method][scope]
            gain_pp = (
                None
                if base_accuracy is None or metrics["accuracy"] is None
                else 100.0 * (metrics["accuracy"] - base_accuracy)
            )
            metrics["gain_vs_base_pp"] = None if method == "Base" else gain_pp
            metrics["HFAG_pp"] = (
                None
                if metrics["HFS"] is None or method == "Base" or gain_pp is None
                else metrics["HFS"] * gain_pp
            )

    base_aime = result["Base"]["aime"]["accuracy"]
    teacher_aime = result["CSD-T"]["aime"]["accuracy"]
    student_aime = result["CSD-SD"]["aime"]["accuracy"]
    teacher_gain = (
        teacher_aime - base_aime
        if teacher_aime is not None and base_aime is not None
        else None
    )
    student_gain = (
        student_aime - base_aime
        if student_aime is not None and base_aime is not None
        else None
    )
    retention = (
        student_gain / teacher_gain
        if teacher_gain is not None and student_gain is not None and teacher_gain > 0.0
        else None
    )
    return {
        "by_method": result,
        "csd_sd_teacher_gain_retention": retention,
        "retention_scope": "AIME24+AIME25",
    }


def _consistent_metadata(
    proposal_rows: Mapping[str, Mapping[str, Any]],
    task1_rows: Mapping[str, Mapping[str, Any]],
    task2_rows: Mapping[str, Mapping[str, Any]],
    cli_model: str | None,
    cli_max_tokens: int | None,
) -> tuple[str, int | None]:
    all_rows = [
        *proposal_rows.values(),
        *task1_rows.values(),
        *task2_rows.values(),
    ]
    runtime_signatures = {
        tuple(sorted(_validate_runtime(row, "result metadata").items()))
        for row in all_rows
    }
    if len(runtime_signatures) != 1:
        raise ReportValidationError(
            "Task shards disagree on git commit, model revision, or software runtime: "
            f"{sorted(runtime_signatures)}"
        )
    models = {
        str(value).strip()
        for row in all_rows
        if (value := _lookup(row, "model", "model_name", "model_path", "runtime.model"))
        is not None
        and str(value).strip()
    }
    if len(models) > 1:
        raise ReportValidationError(
            f"Result rows contain multiple models: {sorted(models)}"
        )
    if not models:
        raise ReportValidationError("Result rows do not identify the evaluated model")
    inferred_model = next(iter(models))
    if cli_model and inferred_model and cli_model != inferred_model:
        raise ReportValidationError(
            f"--model={cli_model!r} disagrees with result model={inferred_model!r}"
        )
    model = cli_model or inferred_model

    token_values: set[int] = set()
    for row in all_rows:
        value = _lookup(
            row,
            "eval_max_new_tokens",
            "max_new_tokens",
            "max_output_tokens",
            "runtime.eval_max_new_tokens",
        )
        if value is not None:
            number = _number(value, "eval_max_new_tokens", minimum=1.0)
            if not number.is_integer():
                raise ReportValidationError(
                    f"eval_max_new_tokens must be integral, got {number}"
                )
            token_values.add(int(number))
    if len(token_values) > 1:
        raise ReportValidationError(
            f"Result rows contain multiple evaluation token limits: {sorted(token_values)}"
        )
    if not token_values:
        raise ReportValidationError(
            "Result rows do not record the evaluation output-token limit"
        )
    inferred_tokens = next(iter(token_values))
    if cli_max_tokens is not None and cli_max_tokens < 1:
        raise ReportValidationError("--max-tokens must be positive")
    if (
        cli_max_tokens is not None
        and inferred_tokens is not None
        and cli_max_tokens != inferred_tokens
    ):
        raise ReportValidationError(
            f"--max-tokens={cli_max_tokens} disagrees with result limit={inferred_tokens}"
        )
    return model, cli_max_tokens if cli_max_tokens is not None else inferred_tokens


def _csv_rows(
    aggregate: Mapping[str, Any], model: str, max_tokens: int | None
) -> list[dict[str, Any]]:
    properties = {
        "Base": ("No", "No", "No"),
        "Privileged Control": ("Yes", "Optional", "No"),
        "CSD-T": ("No", "Yes", "No"),
        "CSD-SD": ("No", "Destroyed before evaluation", "Yes"),
    }
    rows = []
    retention = aggregate["csd_sd_teacher_gain_retention"]
    for method in METHODS:
        amc = aggregate["by_method"][method]["amc23"]
        aime = aggregate["by_method"][method]["aime"]
        overall = aggregate["by_method"][method]["overall"]
        audit_scope = aime if aime["n"] else overall
        uses_answer, teacher, student_update = properties[method]
        rows.append(
            {
                "Method": method,
                "Model": model,
                "Max Tokens": max_tokens,
                "Uses Target Answer": uses_answer,
                "Temporary Teacher": teacher,
                "Student Update": student_update,
                "AMC23 Acc@1 (%)": (
                    None if amc["accuracy"] is None else 100.0 * amc["accuracy"]
                ),
                "AIME Acc@1 (%)": (
                    None if aime["accuracy"] is None else 100.0 * aime["accuracy"]
                ),
                "Gain vs Base (pp)": aime["gain_vs_base_pp"],
                "HER": audit_scope["HER"],
                "CPP": audit_scope["CPP"],
                "HFS": audit_scope["HFS"],
                "HFAG (pp)": aime["HFAG_pp"],
                "Adaptation Sec/Query": overall["mean_adaptation_seconds"],
                "Truncated (All)": overall["truncation_count"],
                "Mean Output Tokens": overall["mean_generated_tokens"],
                "Protocol No-op Queries": overall["protocol_no_op_count"],
                "Protocol No-op Rate": overall["protocol_no_op_rate"],
                "CSD-SD Teacher-Gain Retention": retention
                if method == "CSD-SD"
                else None,
            }
        )
    return rows


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _render_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    fieldnames = list(rows[0])
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _format_csv_value(value) for key, value in row.items()})
    return buffer.getvalue()


def _markdown_value(column: str, value: Any) -> str:
    if value is None:
        return "N/A"
    if column in {
        "AMC23 Acc@1 (%)",
        "AIME Acc@1 (%)",
        "Gain vs Base (pp)",
        "HFAG (pp)",
    }:
        return f"{float(value):.2f}"
    if column in {"HER", "CPP", "HFS"}:
        return f"{float(value):.4f}"
    if column == "Adaptation Sec/Query":
        return f"{float(value):.3f}"
    if column == "Mean Output Tokens":
        return f"{float(value):.1f}"
    if column == "Protocol No-op Rate":
        return f"{100.0 * float(value):.1f}%"
    if column == "CSD-SD Teacher-Gain Retention":
        return f"{100.0 * float(value):.1f}%"
    return str(value)


def _render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = list(rows[0])
    lines = [
        "# Clean Self-Distillation PoC Core Results",
        "",
        "AIME combines AIME24 and AIME25. Gain and HFAG are held-out AIME percentage points. "
        "HER, CPP, and HFS use the same held-out AIME scope (or the available scope for an "
        "infrastructure smoke); latency, truncation, and token counts aggregate all validated queries.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_markdown_value(column, row[column]) for column in columns)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_experiment_summary(
    aggregate: Mapping[str, Any], observed_counts: Mapping[str, int], model: str
) -> str:
    base = aggregate["by_method"]["Base"]["aime"]
    teacher = aggregate["by_method"]["CSD-T"]["aime"]
    student = aggregate["by_method"]["CSD-SD"]["aime"]
    teacher_audit = (
        teacher if teacher["n"] else aggregate["by_method"]["CSD-T"]["overall"]
    )
    student_audit = (
        student if student["n"] else aggregate["by_method"]["CSD-SD"]["overall"]
    )
    teacher_gain = teacher["gain_vs_base_pp"]
    student_gain = student["gain_vs_base_pp"]
    if teacher_gain is None or student_gain is None:
        aime_text = (
            "The expected-count override contains no held-out AIME queries, so AIME accuracy, "
            "gain, HFAG, and teacher-gain retention are N/A."
        )
        conclusion = "This is a coverage/infrastructure smoke report, not PoC performance evidence."
    else:
        teacher_gain = float(teacher_gain)
        student_gain = float(student_gain)
        best_gain = max(teacher_gain, student_gain)
        if best_gain >= 3.0:
            conclusion = "The validated run meets the requested 3–4 percentage-point PoC signal threshold."
        elif best_gain > 0.0:
            conclusion = "The validated run is directionally positive but below the requested PoC threshold."
        else:
            conclusion = "The validated run does not show a positive held-out AIME accuracy gain."
        aime_text = (
            f"On held-out AIME24+AIME25, Base Acc@1 is {100.0 * base['accuracy']:.2f}%. "
            f"CSD-T reaches {100.0 * teacher['accuracy']:.2f}% ({teacher_gain:+.2f} pp), and "
            f"CSD-SD reaches {100.0 * student['accuracy']:.2f}% ({student_gain:+.2f} pp)."
        )
    retention = aggregate["csd_sd_teacher_gain_retention"]
    retention_text = (
        "N/A (CSD-T gain is non-positive)"
        if retention is None
        else f"{100.0 * retention:.1f}%"
    )
    teacher_hfag = teacher["HFAG_pp"]
    student_hfag = student["HFAG_pp"]
    student_overall = aggregate["by_method"]["CSD-SD"]["overall"]
    no_op_count = student_overall["protocol_no_op_count"] or 0
    no_op_rate = student_overall["protocol_no_op_rate"] or 0.0
    hfag_text = (
        "HFAG is N/A for the empty AIME smoke scope."
        if teacher_hfag is None or student_hfag is None
        else f"CSD-T HFAG={teacher_hfag:+.2f} pp and CSD-SD HFAG={student_hfag:+.2f} pp."
    )
    return (
        "# Experiment Summary\n\n"
        f"Validated {sum(observed_counts.values())} unique queries for {model}: "
        f"AMC23={observed_counts.get('amc23', 0)}, AIME24={observed_counts.get('aime24', 0)}, "
        f"AIME25={observed_counts.get('aime25', 0)}. Every query has one matched Task 1 row "
        "and one matched Task 2 row, with source and problem-hash parity.\n\n"
        f"{aime_text} "
        f"Accuracy-based CSD-SD teacher-gain retention is {retention_text}.\n\n"
        f"CSD-T has HER={teacher_audit['HER']:.4f}, CPP={teacher_audit['CPP']:.4f}, "
        f"HFS={teacher_audit['HFS']:.4f}. "
        f"CSD-SD has HER={student_audit['HER']:.4f}, CPP={student_audit['CPP']:.4f}, "
        f"HFS={student_audit['HFS']:.4f}. {hfag_text} "
        f"CSD-SD protocol no-ops: {no_op_count}/{student_overall['n']} "
        f"({100.0 * no_op_rate:.1f}%).\n\n"
        f"{conclusion}\n"
    )


def _parse_expected_counts(values: Iterable[str] | None) -> dict[str, int]:
    result = dict(DEFAULT_EXPECTED_COUNTS)
    for item in values or ():
        if "=" not in item:
            raise ReportValidationError(
                f"Invalid --expected-count {item!r}; expected SOURCE=COUNT"
            )
        source_value, count_value = item.split("=", 1)
        source = _normalize_source(source_value)
        if source not in DEFAULT_EXPECTED_COUNTS:
            raise ReportValidationError(
                f"Unsupported expected-count source={source_value!r}; use amc23, aime24, or aime25"
            )
        try:
            count = int(count_value)
        except ValueError as exc:
            raise ReportValidationError(
                f"Invalid expected count in {item!r}; count must be an integer"
            ) from exc
        if count < 0:
            raise ReportValidationError(
                f"Invalid expected count in {item!r}; count must be >= 0"
            )
        result[source] = count
    return result


def generate_report(
    *,
    dataset_path: str | Path,
    proposal_paths: Sequence[str | Path],
    task1_paths: Sequence[str | Path],
    task2_paths: Sequence[str | Path],
    output_dir: str | Path,
    expected_counts: Mapping[str, int] | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Validate inputs fully, then write the five required PoC artifacts."""
    counts = {
        _normalize_source(source): int(count)
        for source, count in (expected_counts or DEFAULT_EXPECTED_COUNTS).items()
    }
    if set(counts) != set(DEFAULT_EXPECTED_COUNTS):
        raise ReportValidationError(
            f"expected_counts must define exactly {sorted(DEFAULT_EXPECTED_COUNTS)}, got {sorted(counts)}"
        )
    if any(count < 0 for count in counts.values()):
        raise ReportValidationError(f"expected_counts must be non-negative: {counts}")
    dataset = _load_dataset(dataset_path, counts)
    proposal_rows = _load_shards(proposal_paths, "proposal")
    task1_rows = _load_shards(task1_paths, "task1")
    task2_rows = _load_shards(task2_paths, "task2")
    proposal_normalized = _validate_proposal_rows(proposal_rows, dataset)
    task1_normalized = _validate_task_rows("task1", task1_rows, dataset)
    task2_normalized = _validate_task_rows("task2", task2_rows, dataset)
    merged = _merge_rows(
        dataset,
        proposal_rows,
        task1_rows,
        task2_rows,
        proposal_normalized,
        task1_normalized,
        task2_normalized,
    )
    model_name, max_token_limit = _consistent_metadata(
        proposal_rows, task1_rows, task2_rows, model, max_tokens
    )
    aggregate = _aggregate(merged)
    table_rows = _csv_rows(aggregate, model_name, max_token_limit)

    observed_counts = dict(Counter(record["source"] for record in dataset))
    output_root = Path(output_dir)
    artifacts = {
        "merged_jsonl": output_root / "merged_per_query.jsonl",
        "core_csv": output_root / "core_results.csv",
        "core_markdown": output_root / "core_results.md",
        "summary_json": output_root / "summary.json",
        "experiment_summary": output_root / "experiment_summary.md",
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "validation": {
            "status": "passed",
            "real_rows_only": True,
            "clean_git_required": True,
            "authoritative_acc1_regrade": True,
            "candidate_target_disjoint_reaudit": True,
            "dataset_path": str(Path(dataset_path).resolve()),
            "expected_source_counts": counts,
            "observed_source_counts": observed_counts,
            "unique_query_count": len(dataset),
            "unique_problem_hash_count": len(
                {row["problem_sha256"] for row in dataset}
            ),
            "proposal_row_count": len(proposal_rows),
            "task1_row_count": len(task1_rows),
            "task2_row_count": len(task2_rows),
            "proposal_shards": [
                str(Path(path).resolve()) for path in proposal_paths
            ],
            "task1_shards": [str(Path(path).resolve()) for path in task1_paths],
            "task2_shards": [str(Path(path).resolve()) for path in task2_paths],
        },
        "model": model_name,
        "max_tokens": max_token_limit,
        "primary_gain_scope": "AIME24+AIME25",
        "metrics": aggregate,
        # Paths are relative to the report directory so the Slurm merger can
        # atomically rename report.attempt.<job> to report without invalidating
        # provenance recorded inside summary.json.
        "artifacts": {
            name: str(path.relative_to(output_root)) for name, path in artifacts.items()
        },
    }

    # No output directory is created until every validation and aggregation step
    # above has succeeded, so rejected inputs cannot leave plausible partial results.
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts["merged_jsonl"].write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in merged
        ),
        encoding="utf-8",
    )
    artifacts["core_csv"].write_text(_render_csv(table_rows), encoding="utf-8")
    artifacts["core_markdown"].write_text(
        _render_markdown(table_rows), encoding="utf-8"
    )
    artifacts["summary_json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifacts["experiment_summary"].write_text(
        _render_experiment_summary(aggregate, observed_counts, model_name),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, help="Authoritative AMC23+AIME24+AIME25 dataset"
    )
    parser.add_argument(
        "--proposal",
        required=True,
        action="append",
        help="Accepted proposal JSONL shard; repeat for every shard",
    )
    parser.add_argument(
        "--task1",
        required=True,
        action="append",
        help="Task 1 JSONL shard; repeat for every shard",
    )
    parser.add_argument(
        "--task2",
        required=True,
        action="append",
        help="Task 2 JSONL shard; repeat for every shard",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-count",
        action="append",
        metavar="SOURCE=COUNT",
        help="Override an expected count for smoke tests; repeat per source",
    )
    parser.add_argument(
        "--model", help="Optional model label; validated against row metadata"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Optional evaluation output limit; validated against row metadata",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        expected_counts = _parse_expected_counts(args.expected_count)
        summary = generate_report(
            dataset_path=args.dataset,
            proposal_paths=args.proposal,
            task1_paths=args.task1,
            task2_paths=args.task2,
            output_dir=args.output_dir,
            expected_counts=expected_counts,
            model=args.model,
            max_tokens=args.max_tokens,
        )
    except ReportValidationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
