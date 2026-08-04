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
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clean_self_distill.io import load_query_records
from src.clean_self_distill.propose import (
    skill_card_disjoint_audit,
    target_disjoint_audit,
)
from src.opsd_format import extract_boxed_answer, grade_boxed_answer


DEFAULT_EXPECTED_COUNTS = {"amc23": 83, "aime24": 30, "aime25": 30}
METHODS = ("Base", "Privileged Control", "CSD-T", "CSD-SD")
SCHEMA_VERSION = "clean-self-distill-poc-report-v1"
MAX_CANDIDATE_FOURGRAM_OVERLAP_COUNT = 1
MAX_CANDIDATE_FOURGRAM_OVERLAP_RATE = 0.05
EXPECTED_TTT_PREFIX = Path("/home/da839/.conda/envs/TTT")
EXPECTED_TTT_PYTHON = EXPECTED_TTT_PREFIX / "bin" / "python"
EXPECTED_CU128_OVERLAY = Path(
    "/home/da839/scratch_pi_mg269/da839/mfspd/pydeps-cu128"
)
CLEAN_TEACHER_SOURCES = {
    "original_query",
    "sanitized_skill_card",
    "proposed_candidates",
    "student_generated_prefix",
}
FIREWALL_SOURCE_ALLOWLISTS = {
    "candidate_proposer_sources": {"sanitized_skill_card"},
    "solver_sources": {"candidate_problem"},
    "verifier_sources": {"candidate_problem", "candidate_solution"},
}
SPECIALIZATION_READY = "ready"
SPECIALIZATION_INSUFFICIENT = "insufficient_verified_candidates"
SPECIALIZATION_STATUSES = {
    SPECIALIZATION_READY,
    SPECIALIZATION_INSUFFICIENT,
}
PLACEHOLDER_ARTIFACT_RE = re.compile(
    r"\b(?:redacted|placeholder|unspecified|omitted)"
    r"(?:\s+(?:detail|number|quantity|value|object|entity|term))?\b"
    r"|\b(?:generic|hidden|removed)\s+"
    r"(?:detail|number|quantity|value|object|entity|term)\b"
    r"|\btbd\b|\bto\s+be\s+filled\b"
    r"|\b(?:a\s+variable\s+quantity|a\s+symbolic\s+relation|"
    r"an\s+abstract\s+(?:object|element)|an\s+auxiliary\s+variable|"
    r"derived\s+conclusion)\b"
    r"|<\s*[A-Za-z_][A-Za-z0-9_ -]{0,80}\s*>",
    flags=re.IGNORECASE,
)


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


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    result = _number(value, context, minimum=float(minimum))
    if not result.is_integer():
        raise ReportValidationError(f"{context}: expected an integer, got {result}")
    return int(result)


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
    hostname = str(_required(runtime, f"{context}.runtime", "hostname")).strip()
    python_executable = str(
        _required(runtime, f"{context}.runtime", "python_executable")
    ).strip()
    torch_overlay = str(
        _required(runtime, f"{context}.runtime", "torch_overlay")
    ).strip()
    torch_module_path = str(
        _required(runtime, f"{context}.runtime", "torch_module_path")
    ).strip()
    raw_arch_flags = _required(runtime, f"{context}.runtime", "torch_arch_flags")
    if isinstance(raw_arch_flags, str):
        torch_arch_flags = tuple(raw_arch_flags.replace(",", " ").split())
    elif isinstance(raw_arch_flags, (list, tuple)):
        torch_arch_flags = tuple(str(value).strip() for value in raw_arch_flags)
    else:
        raise ReportValidationError(
            f"{context}.runtime.torch_arch_flags must be a string or list"
        )
    slurm_array_job_id = str(
        _required(runtime, f"{context}.runtime", "slurm_array_job_id")
    ).strip()
    slurm_array_task_id = str(
        _required(runtime, f"{context}.runtime", "slurm_array_task_id")
    ).strip()
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
    if not hostname or not python_executable or not torch_overlay or not torch_module_path:
        raise ReportValidationError(
            f"{context}.runtime must record hostname, python_executable, torch_overlay, "
            "and torch_module_path"
        )
    if Path(conda_prefix) != EXPECTED_TTT_PREFIX:
        raise ReportValidationError(
            f"{context}.runtime.conda_prefix={conda_prefix!r} is not the exact "
            f"required TTT prefix {str(EXPECTED_TTT_PREFIX)!r}"
        )
    if Path(python_executable) != EXPECTED_TTT_PYTHON:
        raise ReportValidationError(
            f"{context}.runtime.python_executable={python_executable!r} is not the "
            f"activated TTT interpreter {str(EXPECTED_TTT_PYTHON)!r}"
        )
    # The scratch mount is exposed through both /home/... and its canonical
    # /nfs/... path.  torch.__file__ follows the mount while PYTHONPATH keeps
    # the user-facing path, so compare and report their canonical identities.
    overlay_path = Path(torch_overlay).resolve(strict=False)
    module_path = Path(torch_module_path).resolve(strict=False)
    expected_overlay_path = EXPECTED_CU128_OVERLAY.resolve(strict=False)
    if overlay_path != expected_overlay_path:
        raise ReportValidationError(
            f"{context}.runtime.torch_overlay={torch_overlay!r} is not the exact "
            f"approved cu128 overlay {str(EXPECTED_CU128_OVERLAY)!r}"
        )
    if not module_path.is_relative_to(overlay_path):
        raise ReportValidationError(
            f"{context}.runtime.torch_module_path={torch_module_path!r} is not inside "
            f"torch_overlay={torch_overlay!r}"
        )
    if not torch_version.endswith("+cu128") or cuda_runtime != "12.8":
        raise ReportValidationError(
            f"{context}.runtime requires the B200 cu128 build, got "
            f"torch={torch_version!r}, cuda_runtime={cuda_runtime!r}"
        )
    if not torch_arch_flags or "sm_100" not in torch_arch_flags:
        raise ReportValidationError(
            f"{context}.runtime.torch_arch_flags must include sm_100, got {torch_arch_flags}"
        )
    if not slurm_array_job_id.isdigit():
        raise ReportValidationError(
            f"{context}.runtime.slurm_array_job_id must be numeric, got {slurm_array_job_id!r}"
        )
    if not slurm_array_task_id.isdigit():
        raise ReportValidationError(
            f"{context}.runtime.slurm_array_task_id must be numeric, got {slurm_array_task_id!r}"
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
        "python_executable": python_executable,
        "torch_overlay": str(overlay_path),
        "torch_module_path": str(module_path),
        "torch_arch_flags": tuple(sorted(torch_arch_flags)),
        "gpu_capabilities": tuple(gpu_capabilities),
        "allocation": (
            slurm_array_job_id,
            int(slurm_array_task_id),
            hostname,
        ),
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


def _validate_declared_audit(
    declared: Any, recomputed: Mapping[str, Any], context: str
) -> None:
    if not isinstance(declared, Mapping):
        raise ReportValidationError(f"{context}: expected an audit object")
    for key, expected in recomputed.items():
        if key not in declared:
            raise ReportValidationError(f"{context}: missing recomputed field {key!r}")
        if declared[key] != expected:
            raise ReportValidationError(
                f"{context}.{key}={declared[key]!r} disagrees with independent "
                f"re-audit={expected!r}"
            )


def _validate_firewall(row: Mapping[str, Any], context: str) -> dict[str, Any]:
    firewall = _required(row, context, "firewall_audit")
    if not isinstance(firewall, Mapping):
        raise ReportValidationError(f"{context}.firewall_audit: expected an object")
    for key in ("target_answer_loaded", "target_solution_loaded"):
        if _boolean(
            _required(firewall, f"{context}.firewall_audit", key),
            f"{context}.firewall_audit.{key}",
        ):
            raise ReportValidationError(
                f"{context}.firewall_audit.{key}=true violates the clean proposal boundary"
            )
    for key, expected in FIREWALL_SOURCE_ALLOWLISTS.items():
        raw_sources = _required(firewall, f"{context}.firewall_audit", key)
        if not isinstance(raw_sources, list):
            raise ReportValidationError(
                f"{context}.firewall_audit.{key}: expected a list"
            )
        actual = {str(source).strip().lower() for source in raw_sources}
        if actual != expected or len(raw_sources) != len(actual):
            raise ReportValidationError(
                f"{context}.firewall_audit.{key}={sorted(actual)}; "
                f"expected exactly {sorted(expected)}"
            )
    _sha256_value(
        _required(firewall, f"{context}.firewall_audit", "skill_prompt_sha256"),
        f"{context}.firewall_audit.skill_prompt_sha256",
    )
    _sha256_value(
        _required(firewall, f"{context}.firewall_audit", "candidate_prompt_sha256"),
        f"{context}.firewall_audit.candidate_prompt_sha256",
    )
    _integer(
        _required(firewall, f"{context}.firewall_audit", "skill_card_redaction_count"),
        f"{context}.firewall_audit.skill_card_redaction_count",
    )
    return dict(firewall)


def _validate_hashed_mapping(
    row: Mapping[str, Any], context: str, value_key: str, digest_key: str
) -> tuple[dict[str, Any], str]:
    value = _required(row, context, value_key)
    if not isinstance(value, Mapping) or not value:
        raise ReportValidationError(f"{context}.{value_key}: expected a non-empty object")
    digest = _sha256_value(
        _required(row, context, digest_key), f"{context}.{digest_key}"
    )
    recomputed = _canonical_json_sha256(value)
    if digest != recomputed:
        raise ReportValidationError(
            f"{context}.{digest_key}={digest} does not hash canonical {value_key}={recomputed}"
        )
    return dict(value), digest


def _validate_specialization_decision(
    row: Mapping[str, Any],
    context: str,
    *,
    candidate_count: int | None = None,
) -> dict[str, Any]:
    status = _required(row, context, "specialization_status")
    if not isinstance(status, str) or status not in SPECIALIZATION_STATUSES:
        raise ReportValidationError(
            f"{context}.specialization_status must be exactly one of "
            f"{sorted(SPECIALIZATION_STATUSES)}, got {status!r}"
        )
    failure_reason = _required(row, context, "specialization_failure_reason")
    if not isinstance(failure_reason, str):
        raise ReportValidationError(
            f"{context}.specialization_failure_reason must be a string"
        )
    no_op = _required(row, context, "specialization_no_op")
    if not isinstance(no_op, bool):
        raise ReportValidationError(
            f"{context}.specialization_no_op must be a JSON boolean"
        )

    if status == SPECIALIZATION_READY:
        if failure_reason != "" or no_op:
            raise ReportValidationError(
                f"{context}: ready specialization requires an empty failure reason "
                "and specialization_no_op=false"
            )
        if candidate_count is not None and candidate_count < 1:
            raise ReportValidationError(
                f"{context}: ready specialization requires at least one verified candidate"
            )
    else:
        if not failure_reason.strip() or not no_op:
            raise ReportValidationError(
                f"{context}: insufficient_verified_candidates requires a nonempty "
                "failure reason and specialization_no_op=true"
            )

    return {
        "specialization_status": status,
        "specialization_failure_reason": failure_reason,
        "specialization_no_op": no_op,
    }


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
        skill_card = _required(row, context, "skill_card")
        if not isinstance(skill_card, Mapping):
            raise ReportValidationError(f"{context}.skill_card: expected an object")
        skill_card_audit = skill_card_disjoint_audit(
            record["problem"], dict(skill_card)
        )
        if not _boolean(skill_card_audit.get("safe"), f"{context}.skill_card.safe"):
            raise ReportValidationError(
                f"{context}.skill_card fails the authoritative target-disjoint audit: "
                f"{skill_card_audit}"
            )
        _validate_declared_audit(
            _required(row, context, "skill_card_target_disjoint_audit"),
            skill_card_audit,
            f"{context}.skill_card_target_disjoint_audit",
        )
        _validate_firewall(row, context)
        proposal_training_sha256 = _sha256_value(
            _required(row, context, "proposal_training_sha256"),
            f"{context}.proposal_training_sha256",
        )
        candidates = _required(row, context, "specialization_candidates")
        if not isinstance(candidates, list):
            raise ReportValidationError(
                f"{context}.specialization_candidates must be a list"
            )
        specialization = _validate_specialization_decision(
            row, context, candidate_count=len(candidates)
        )
        declared_count = _integer(
            _required(row, context, "candidate_count"),
            f"{context}.candidate_count",
        )
        if declared_count != len(candidates):
            raise ReportValidationError(
                f"{context}: candidate_count={declared_count!r} does not match "
                f"accepted candidate list length={len(candidates)}"
            )
        requested_count = _integer(
            _required(row, context, "requested_candidate_count"),
            f"{context}.requested_candidate_count",
            minimum=1,
        )
        minimum_count = _integer(
            _required(row, context, "minimum_candidate_count"),
            f"{context}.minimum_candidate_count",
            minimum=1,
        )
        if minimum_count > requested_count or declared_count > requested_count:
            raise ReportValidationError(
                f"{context}: candidate counts violate minimum <= requested and "
                "accepted <= requested"
            )
        if specialization["specialization_status"] == SPECIALIZATION_READY:
            if declared_count < minimum_count:
                raise ReportValidationError(
                    f"{context}: ready specialization has {declared_count} accepted "
                    f"candidates below minimum_candidate_count={minimum_count}"
                )
        elif declared_count >= minimum_count:
            raise ReportValidationError(
                f"{context}: insufficient_verified_candidates has {declared_count} "
                f"accepted candidates meeting minimum_candidate_count={minimum_count}"
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
            candidate_solution = str(
                _required(candidate, candidate_context, "solution")
            ).strip()
            candidate_final_answer = str(
                _required(candidate, candidate_context, "final_answer")
            ).strip()
            for field, text in (
                ("problem", candidate_problem),
                ("solution", candidate_solution),
                ("final_answer", candidate_final_answer),
            ):
                if not text:
                    raise ReportValidationError(
                        f"{candidate_context}.{field} is empty"
                    )
                placeholder_artifacts = [
                    match.group(0) for match in PLACEHOLDER_ARTIFACT_RE.finditer(text)
                ]
                if placeholder_artifacts:
                    raise ReportValidationError(
                        f"{candidate_context}: accepted candidate {field} contains "
                        f"placeholder artifacts {placeholder_artifacts}"
                    )
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
            _validate_declared_audit(
                _required(candidate, candidate_context, "target_disjoint_audit"),
                audit,
                f"{candidate_context}.target_disjoint_audit",
            )
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

        proposal_training_payload = {
            "query_id": query_id,
            "problem_sha256": digest,
            "skill_card": dict(skill_card),
            "specialization_candidates": candidates,
            **specialization,
        }
        recomputed_training_sha256 = _canonical_json_sha256(
            proposal_training_payload
        )
        if proposal_training_sha256 != recomputed_training_sha256:
            raise ReportValidationError(
                f"{context}.proposal_training_sha256={proposal_training_sha256} does not "
                f"bind the canonical skill-card/candidate payload={recomputed_training_sha256}"
            )

        filter_summary = _required(row, context, "filter_summary")
        if not isinstance(filter_summary, Mapping):
            raise ReportValidationError(f"{context}.filter_summary: expected an object")
        accepted_count = _integer(
            _required(filter_summary, f"{context}.filter_summary", "accepted_count"),
            f"{context}.filter_summary.accepted_count",
        )
        if accepted_count != len(candidates):
            raise ReportValidationError(
                f"{context}.filter_summary.accepted_count={accepted_count} does not match "
                f"accepted candidate list length={len(candidates)}"
            )
        proposed_unique_count = _integer(
            _required(
                filter_summary, f"{context}.filter_summary", "proposed_unique_count"
            ),
            f"{context}.filter_summary.proposed_unique_count",
        )
        rejected_count = _integer(
            _required(filter_summary, f"{context}.filter_summary", "rejected_count"),
            f"{context}.filter_summary.rejected_count",
        )
        if proposed_unique_count < accepted_count:
            raise ReportValidationError(
                f"{context}.filter_summary proposed_unique_count is below accepted_count"
            )
        verification_yield = _rate(
            _required(filter_summary, f"{context}.filter_summary", "verification_yield"),
            f"{context}.filter_summary.verification_yield",
        )
        expected_yield = accepted_count / max(proposed_unique_count, 1)
        if not math.isclose(
            verification_yield, expected_yield, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ReportValidationError(
                f"{context}.filter_summary.verification_yield={verification_yield} "
                f"does not equal accepted/proposed={expected_yield}"
            )
        cost_audit = _required(row, context, "cost_audit")
        if not isinstance(cost_audit, Mapping):
            raise ReportValidationError(f"{context}.cost_audit: expected an object")
        proposal_generation_seconds = _number(
            _required(cost_audit, f"{context}.cost_audit", "total_generation_seconds"),
            f"{context}.cost_audit.total_generation_seconds",
            minimum=0.0,
        )
        proposal_end_to_end_seconds = _number(
            _required(cost_audit, f"{context}.cost_audit", "end_to_end_seconds"),
            f"{context}.cost_audit.end_to_end_seconds",
            minimum=0.0,
        )
        if proposal_generation_seconds > proposal_end_to_end_seconds + 1e-6:
            raise ReportValidationError(
                f"{context}.cost_audit generation time exceeds end-to-end time"
            )
        proposal_prompt_tokens = _number(
            _required(cost_audit, f"{context}.cost_audit", "total_prompt_tokens"),
            f"{context}.cost_audit.total_prompt_tokens",
            minimum=0.0,
        )
        proposal_completion_tokens = _number(
            _required(cost_audit, f"{context}.cost_audit", "total_completion_tokens"),
            f"{context}.cost_audit.total_completion_tokens",
            minimum=0.0,
        )

        normalized[query_id] = {
            "source": source,
            "problem_sha256": digest,
            "runtime_signature": runtime_signature,
            "proposal_training_sha256": proposal_training_sha256,
            **specialization,
            "candidate_count": len(candidates),
            "candidate_audits": candidate_audits,
            "verification_yield": verification_yield,
            "diagnostics": {
                "proposal_generation_seconds": proposal_generation_seconds,
                "proposal_end_to_end_seconds": proposal_end_to_end_seconds,
                "proposal_prompt_tokens": proposal_prompt_tokens,
                "proposal_completion_tokens": proposal_completion_tokens,
                "accepted_candidates": len(candidates),
                "verification_yield": verification_yield,
            },
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


def _audit_values(
    row: Mapping[str, Any],
    context: str,
    task_name: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _required(row, context, "hindsight_audit")
    if not isinstance(raw, Mapping):
        raise ReportValidationError(f"{context}.hindsight_audit: expected an object")

    counts = {
        key: _integer(
            _required(raw, f"{context}.hindsight_audit", key),
            f"{context}.hindsight_audit.{key}",
        )
        for key in (
            "teacher_context_events",
            "forbidden_context_events",
            "comparison_events",
            "context_equal_events",
            "compared_token_positions",
            "same_prefix_positions",
            "causal_events",
            "on_policy_events",
            "on_policy_equal_events",
        )
    }
    if counts["teacher_context_events"] < 1:
        raise ReportValidationError(
            f"{context}.hindsight_audit.teacher_context_events must be positive"
        )
    bounded_pairs = (
        ("forbidden_context_events", "teacher_context_events"),
        ("context_equal_events", "comparison_events"),
        ("same_prefix_positions", "compared_token_positions"),
        ("causal_events", "teacher_context_events"),
        ("on_policy_equal_events", "on_policy_events"),
    )
    for numerator, denominator in bounded_pairs:
        if counts[numerator] > counts[denominator]:
            raise ReportValidationError(
                f"{context}.hindsight_audit.{numerator} exceeds {denominator}"
            )

    source_counts_raw = _required(
        raw, f"{context}.hindsight_audit", "source_counts"
    )
    if not isinstance(source_counts_raw, Mapping) or not source_counts_raw:
        raise ReportValidationError(
            f"{context}.hindsight_audit.source_counts: expected a non-empty object"
        )
    source_counts: dict[str, int] = {}
    for source, value in source_counts_raw.items():
        normalized_source = str(source).strip().lower()
        if not normalized_source or normalized_source in source_counts:
            raise ReportValidationError(
                f"{context}.hindsight_audit.source_counts contains an empty/duplicate source"
            )
        source_counts[normalized_source] = _integer(
            value,
            f"{context}.hindsight_audit.source_counts.{normalized_source}",
        )
    unexpected_sources = sorted(set(source_counts) - CLEAN_TEACHER_SOURCES)
    if unexpected_sources:
        raise ReportValidationError(
            f"{context}.hindsight_audit contains non-clean teacher sources "
            f"{unexpected_sources}"
        )
    teacher_events = counts["teacher_context_events"]
    expected_source_counts = {"original_query": teacher_events}
    if not protocol["protocol_no_op"]:
        expected_source_counts.update(
            {
                "sanitized_skill_card": teacher_events,
                "proposed_candidates": teacher_events,
            }
        )
        if counts["on_policy_events"]:
            expected_source_counts["student_generated_prefix"] = counts[
                "on_policy_events"
            ]
    if source_counts != expected_source_counts:
        raise ReportValidationError(
            f"{context}.hindsight_audit.source_counts={source_counts} does not match "
            f"the exact specialization provenance={expected_source_counts}"
        )
    if counts["forbidden_context_events"] != 0:
        raise ReportValidationError(
            f"{context}: clean CSD row has forbidden teacher-context events"
        )
    if counts["causal_events"] != teacher_events:
        raise ReportValidationError(
            f"{context}: every teacher-context event must be causally scored"
        )

    expected_comparisons = int(protocol["comparison_events"])
    expected_equal_events = int(protocol["context_equal_events"])
    if counts["comparison_events"] != expected_comparisons:
        raise ReportValidationError(
            f"{context}.hindsight_audit.comparison_events={counts['comparison_events']} "
            f"does not match protocol evidence={expected_comparisons}"
        )
    if counts["context_equal_events"] != expected_equal_events:
        raise ReportValidationError(
            f"{context}.hindsight_audit.context_equal_events={counts['context_equal_events']} "
            f"does not match protocol evidence={expected_equal_events}"
        )
    if task_name == "task1":
        if counts["on_policy_events"] != 0 or counts["on_policy_equal_events"] != 0:
            raise ReportValidationError(f"{context}: Task 1 must not claim on-policy events")
        if counts["compared_token_positions"] < 1:
            raise ReportValidationError(
                f"{context}: Task 1 must record at least one compared token position"
            )
        if counts["same_prefix_positions"] != counts["compared_token_positions"]:
            raise ReportValidationError(
                f"{context}: Task 1 evaluation context hashes match but position counts do not"
            )
        if teacher_events != 1:
            raise ReportValidationError(
                f"{context}: Task 1 must record exactly one teacher-context event"
            )
    else:
        expected_positions = int(protocol["compared_token_positions"])
        expected_same_positions = int(protocol["same_prefix_positions"])
        if counts["compared_token_positions"] != expected_positions:
            raise ReportValidationError(
                f"{context}.hindsight_audit.compared_token_positions="
                f"{counts['compared_token_positions']} does not match trace={expected_positions}"
            )
        if counts["same_prefix_positions"] != expected_same_positions:
            raise ReportValidationError(
                f"{context}.hindsight_audit.same_prefix_positions="
                f"{counts['same_prefix_positions']} does not match trace={expected_same_positions}"
            )
        if counts["on_policy_events"] != expected_comparisons:
            raise ReportValidationError(
                f"{context}.hindsight_audit.on_policy_events does not match trace"
            )
        if counts["on_policy_equal_events"] != expected_equal_events:
            raise ReportValidationError(
                f"{context}.hindsight_audit.on_policy_equal_events does not match trace"
            )
        if teacher_events != expected_comparisons + 1:
            raise ReportValidationError(
                f"{context}: Task 2 teacher-context events must equal construction + trace"
            )

    her = counts["forbidden_context_events"] / teacher_events
    compared_positions = counts["compared_token_positions"]
    cpp = (
        counts["same_prefix_positions"] / compared_positions
        if compared_positions
        else 0.0
    )
    hfs = (1.0 - her) * cpp
    supplied_values = {
        "hindsight_exposure_rate": ("hindsight_exposure_rate", "HER", "her"),
        "context_prefix_parity": (
            "context_prefix_parity",
            "context_parity_rate",
            "CPP",
            "cpp",
        ),
        "hindsight_free_score": ("hindsight_free_score", "HFS", "hfs"),
        "same_prefix_fidelity": ("same_prefix_fidelity",),
    }
    expected_rates = {
        "hindsight_exposure_rate": her,
        "context_prefix_parity": cpp,
        "hindsight_free_score": hfs,
        "same_prefix_fidelity": cpp,
    }
    for label, aliases in supplied_values.items():
        supplied = _rate(
            _required(row, context, *aliases), f"{context}.{label}"
        )
        if not math.isclose(
            supplied, expected_rates[label], rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ReportValidationError(
                f"{context}.{label}={supplied} disagrees with raw-count recompute="
                f"{expected_rates[label]}"
            )
    return {
        **counts,
        "source_counts": source_counts,
        "HER": her,
        "CPP": cpp,
        "HFS": hfs,
    }


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
        "proposal_end_to_end_seconds": support,
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


def _validate_no_op_base_equivalence(
    row: Mapping[str, Any],
    context: str,
    compared_prefixes: Sequence[str],
) -> None:
    """Prove that an explicit no-op produced the deterministic Base result."""
    base_responses = _required(row, context, "base_responses", "base_outputs")
    if not isinstance(base_responses, list) or len(base_responses) != 1:
        raise ReportValidationError(
            f"{context}.base_responses must contain exactly one Acc@1 response"
        )
    base_parsed = _required(row, context, "base_parsed_answers")
    expected_base_parsed = [
        str(extract_boxed_answer(str(response)) or "").strip()
        for response in base_responses
    ]
    if base_parsed != expected_base_parsed:
        raise ReportValidationError(
            f"{context}: no-op parsed-answer drift: base_parsed_answers={base_parsed!r} "
            f"does not match independent parse={expected_base_parsed!r}"
        )

    base_correct = _correct(
        _required(row, context, "base_correct"), f"{context}.base_correct"
    )
    base_tokens, base_truncated = _diagnostics(row, context, "base")
    for prefix in compared_prefixes:
        responses = _required(
            row, context, f"{prefix}_responses", f"{prefix}_outputs"
        )
        if responses != base_responses:
            raise ReportValidationError(
                f"{context}: no-op response drift: {prefix}_responses must exactly "
                "equal base_responses under the deterministic shared seed"
            )
        parsed = _required(row, context, f"{prefix}_parsed_answers")
        if parsed != base_parsed:
            raise ReportValidationError(
                f"{context}: no-op parsed-answer drift: {prefix}_parsed_answers "
                "must exactly equal base_parsed_answers"
            )
        correct = _correct(
            _required(row, context, f"{prefix}_correct"),
            f"{context}.{prefix}_correct",
        )
        if correct != base_correct:
            raise ReportValidationError(
                f"{context}: no-op correctness drift: {prefix}_correct={correct} "
                f"does not equal base_correct={base_correct}"
            )
        tokens, truncated = _diagnostics(row, context, prefix)
        if tokens != base_tokens:
            raise ReportValidationError(
                f"{context}: no-op generated-token drift: {prefix}={tokens} "
                f"does not equal Base={base_tokens}"
            )
        if truncated != base_truncated:
            raise ReportValidationError(
                f"{context}: no-op truncation drift: {prefix}={truncated} "
                f"does not equal Base={base_truncated}"
            )

        base_nll = _lookup(row, "base_target_answer_nll")
        compared_nll = _lookup(row, f"{prefix}_target_answer_nll")
        if base_nll is not None and compared_nll is not None:
            base_nll_value = _number(
                base_nll, f"{context}.base_target_answer_nll", minimum=0.0
            )
            compared_nll_value = _number(
                compared_nll,
                f"{context}.{prefix}_target_answer_nll",
                minimum=0.0,
            )
            if not math.isclose(
                base_nll_value,
                compared_nll_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ReportValidationError(
                    f"{context}: no-op NLL drift: {prefix}={compared_nll_value} "
                    f"does not equal Base={base_nll_value}"
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
    row: Mapping[str, Any],
    context: str,
    task_name: str,
    specialization: Mapping[str, Any],
) -> dict[str, Any]:
    specialization_no_op = bool(specialization["specialization_no_op"])
    ridge_update_norm = _field_number(
        row,
        context,
        ("update_frobenius_norm", "ridge_update_frobenius_norm", "update_norm"),
        minimum=0.0,
    )
    adapter_rank = _field_number(
        row, context, ("adapter_rank", "ridge_rank"), minimum=0.0
    )
    uses_all_candidates = _required(row, context, "uses_all_candidates")
    if not isinstance(uses_all_candidates, bool):
        raise ReportValidationError(
            f"{context}.uses_all_candidates must be a JSON boolean"
        )
    if specialization_no_op:
        if (
            ridge_update_norm != 0.0
            or adapter_rank != 0.0
            or uses_all_candidates
        ):
            raise ReportValidationError(
                f"{context}: specialization no-op requires adapter_rank and ridge "
                "update_frobenius_norm to equal zero and uses_all_candidates=false"
            )
    elif (
        ridge_update_norm <= 0.0
        or adapter_rank <= 0.0
        or not adapter_rank.is_integer()
        or not uses_all_candidates
    ):
        raise ReportValidationError(
            f"{context}: ready specialization requires a nonzero ridge update and "
            "integral positive adapter rank with uses_all_candidates=true"
        )

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
        student_context_sha256 = _sha256_value(
            _required(row, context, "student_evaluation_context_sha256"),
            f"{context}.student_evaluation_context_sha256",
        )
        teacher_context_sha256 = _sha256_value(
            _required(row, context, "teacher_evaluation_context_sha256"),
            f"{context}.teacher_evaluation_context_sha256",
        )
        if student_context_sha256 != teacher_context_sha256:
            raise ReportValidationError(
                f"{context}: Task 1 student/teacher evaluation context hashes differ"
            )
        _validate_acc1_artifacts(row, context, ("base", "privileged", "teacher"))
        if specialization_no_op:
            _validate_no_op_base_equivalence(row, context, ("teacher",))
        return {
            "protocol_no_op": specialization_no_op,
            "update_frobenius_norm": ridge_update_norm,
            "adapter_rank": int(adapter_rank),
            "steps_completed": None,
            "comparison_events": 1,
            "context_equal_events": 1,
            "compared_token_positions": None,
            "same_prefix_positions": None,
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
    student_update_norm = _field_number(
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
    if specialization_no_op:
        if student_update_norm != 0.0 or steps != 0.0:
            raise ReportValidationError(
                f"{context}: specialization no-op requires student update and "
                "distillation steps to equal zero"
            )
    elif student_update_norm <= 0.0 or steps <= 0.0:
        raise ReportValidationError(
            f"{context}: ready specialization requires a positive student update "
            "and at least one distillation step"
        )
    trace = _required(row, context, "distillation_trace")
    if not isinstance(trace, list) or len(trace) != int(steps):
        raise ReportValidationError(
            f"{context}: distillation_trace length does not match completed steps"
        )
    trace_compared_positions = 0
    trace_same_prefix_positions = 0
    trace_equal_events = 0
    for trace_index, step in enumerate(trace):
        trace_context = f"{context}.distillation_trace[{trace_index}]"
        if not isinstance(step, Mapping):
            raise ReportValidationError(f"{trace_context}: expected an object")
        student_hash = _sha256_value(
            _required(step, trace_context, "student_context_sha256"),
            f"{trace_context}.student_context_sha256",
        )
        teacher_hash = _sha256_value(
            _required(step, trace_context, "teacher_context_sha256"),
            f"{trace_context}.teacher_context_sha256",
        )
        same_prefix = _boolean(
            _required(step, trace_context, "same_prefix"),
            f"{trace_context}.same_prefix",
        )
        hash_equal = student_hash == teacher_hash
        if same_prefix != hash_equal:
            raise ReportValidationError(
                f"{trace_context}: same_prefix={same_prefix} disagrees with context hashes"
            )
        prefix_tokens = _integer(
            _required(step, trace_context, "prefix_tokens"),
            f"{trace_context}.prefix_tokens",
            minimum=1,
        )
        compared_positions = _integer(
            _required(step, trace_context, "compared_positions"),
            f"{trace_context}.compared_positions",
            minimum=1,
        )
        if compared_positions != prefix_tokens:
            raise ReportValidationError(
                f"{trace_context}: compared_positions={compared_positions} does not equal "
                f"prefix_tokens={prefix_tokens}"
            )
        trace_compared_positions += compared_positions
        if same_prefix:
            trace_equal_events += 1
            trace_same_prefix_positions += compared_positions
        else:
            raise ReportValidationError(
                f"{trace_context}: Clean Self-Distillation requires identical prefixes"
            )
    _validate_acc1_artifacts(row, context, ("base", "teacher", "distilled"))
    if specialization_no_op:
        _validate_no_op_base_equivalence(
            row, context, ("teacher", "distilled")
        )
    return {
        "protocol_no_op": specialization_no_op,
        "update_frobenius_norm": student_update_norm,
        "ridge_update_frobenius_norm": ridge_update_norm,
        "adapter_rank": int(adapter_rank),
        "steps_completed": int(steps),
        "comparison_events": len(trace),
        "context_equal_events": trace_equal_events,
        "compared_token_positions": trace_compared_positions,
        "same_prefix_positions": trace_same_prefix_positions,
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
        proposal_training_sha256 = _sha256_value(
            _required(row, context, "proposal_training_sha256"),
            f"{context}.proposal_training_sha256",
        )
        specialization = _validate_specialization_decision(row, context)
        ridge_config, ridge_config_sha256 = _validate_hashed_mapping(
            row, context, "ridge_config", "ridge_config_sha256"
        )
        run_config, run_config_sha256 = _validate_hashed_mapping(
            row, context, "run_config", "run_config_sha256"
        )
        expected_mode = "task1" if task_name == "task1" else "task2"
        if str(_required(run_config, f"{context}.run_config", "mode")) != expected_mode:
            raise ReportValidationError(
                f"{context}.run_config.mode must be {expected_mode!r}"
            )
        configured_num_shards = _integer(
            _required(run_config, f"{context}.run_config", "num_shards"),
            f"{context}.run_config.num_shards",
            minimum=1,
        )
        if _integer(
            _required(run_config, f"{context}.run_config", "eval_samples"),
            f"{context}.run_config.eval_samples",
            minimum=1,
        ) != 1:
            raise ReportValidationError(f"{context}.run_config.eval_samples must equal 1")
        if str(_required(run_config, f"{context}.run_config", "model")).strip() != str(
            row["model"]
        ).strip():
            raise ReportValidationError(f"{context}.run_config.model disagrees with row model")
        run_revision = str(
            _required(run_config, f"{context}.run_config", "revision")
        ).strip()
        if run_revision != runtime_signature["model_revision"]:
            raise ReportValidationError(
                f"{context}.run_config.revision disagrees with resolved model revision"
            )
        for key, expected_value in ridge_config.items():
            if key not in run_config or run_config[key] != expected_value:
                raise ReportValidationError(
                    f"{context}.run_config.{key} does not match ridge_config"
                )
        protocol = _validate_task_protocol(
            row, context, task_name, specialization
        )
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

        audit_values = _audit_values(row, context, task_name, protocol)
        timing = _adaptation_values(row, context, task_name)
        peak_memory_bytes = _field_number(
            row, context, ("peak_memory_bytes",), minimum=0.0
        )
        max_input_tokens = _integer(
            _required(row, context, "max_input_tokens"),
            f"{context}.max_input_tokens",
        )
        max_output_tokens = _integer(
            _required(row, context, "max_output_tokens", "eval_max_new_tokens"),
            f"{context}.max_output_tokens",
            minimum=1,
        )
        support_generated_tokens = _field_number(
            row, context, ("support_generated_tokens",), minimum=0.0
        )
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
            "proposal_training_sha256": proposal_training_sha256,
            **specialization,
            "ridge_config": ridge_config,
            "ridge_config_sha256": ridge_config_sha256,
            "run_config": run_config,
            "run_config_sha256": run_config_sha256,
            "num_shards": configured_num_shards,
            "correct": condition_correct,
            "audit": audit_values,
            "protocol": protocol,
            "timing": timing,
            "stage_diagnostics": {
                **timing,
                "peak_memory_bytes": peak_memory_bytes,
                "max_input_tokens": max_input_tokens,
                "max_output_tokens": max_output_tokens,
                "support_generated_tokens": support_generated_tokens,
            },
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
        proposal_norm = proposal_normalized[query_id]
        norm1 = task1_normalized[query_id]
        norm2 = task2_normalized[query_id]
        proposal_training_sha256 = proposal_norm["proposal_training_sha256"]
        if (
            norm1["proposal_training_sha256"] != proposal_training_sha256
            or norm2["proposal_training_sha256"] != proposal_training_sha256
        ):
            raise ReportValidationError(
                f"query_id={query_id!r}: proposal_training_sha256 differs across "
                "proposal, Task 1, and Task 2 artifacts"
            )
        specialization = {
            key: proposal_norm[key]
            for key in (
                "specialization_status",
                "specialization_failure_reason",
                "specialization_no_op",
            )
        }
        for task_label, task_norm in (("Task 1", norm1), ("Task 2", norm2)):
            task_specialization = {
                key: task_norm[key] for key in specialization
            }
            if task_specialization != specialization:
                raise ReportValidationError(
                    f"query_id={query_id!r}: {task_label} specialization decision "
                    f"{task_specialization} disagrees with proposal {specialization}"
                )
        if norm1["ridge_config_sha256"] != norm2["ridge_config_sha256"]:
            raise ReportValidationError(
                f"query_id={query_id!r}: Task 1 and Task 2 ridge configurations differ"
            )
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
                "audit_counts": None,
                **norm1["diagnostics"]["base"],
            },
            "Privileged Control": {
                "correct": norm1["correct"]["privileged"],
                "protocol_no_op": None,
                "HER": 1.0,
                "CPP": 0.0,
                "HFS": 0.0,
                "adaptation_seconds": privileged_adaptation,
                "audit_counts": {
                    "teacher_context_events": 1,
                    "forbidden_context_events": 1,
                    "comparison_events": 0,
                    "compared_token_positions": 0,
                    "same_prefix_positions": 0,
                },
                **norm1["diagnostics"]["privileged"],
            },
            "CSD-T": {
                "correct": norm1["correct"]["teacher"],
                "protocol_no_op": norm1["protocol"]["protocol_no_op"],
                "HER": norm1["audit"]["HER"],
                "CPP": norm1["audit"]["CPP"],
                "HFS": norm1["audit"]["HFS"],
                "adaptation_seconds": norm1["timing"]["total_adaptation_seconds"],
                "audit_counts": norm1["audit"],
                **norm1["diagnostics"]["teacher"],
            },
            "CSD-SD": {
                "correct": norm2["correct"]["distilled"],
                "protocol_no_op": norm2["protocol"]["protocol_no_op"],
                "HER": norm2["audit"]["HER"],
                "CPP": norm2["audit"]["CPP"],
                "HFS": norm2["audit"]["HFS"],
                "adaptation_seconds": norm2["timing"]["total_adaptation_seconds"],
                "audit_counts": norm2["audit"],
                **norm2["diagnostics"]["distilled"],
            },
        }
        merged.append(
            {
                "schema_version": SCHEMA_VERSION,
                **record,
                "proposal_shard": proposal["__input_shard"],
                "proposal_line": proposal["__input_line"],
                "proposal_training_sha256": proposal_training_sha256,
                **specialization,
                "ridge_config_sha256": norm1["ridge_config_sha256"],
                "proposal_candidate_audits": proposal_normalized[query_id][
                    "candidate_audits"
                ],
                "task1_shard": raw1["__input_shard"],
                "task1_line": raw1["__input_line"],
                "task2_shard": raw2["__input_shard"],
                "task2_line": raw2["__input_line"],
                "conditions": conditions,
                "diagnostics": {
                    "proposal": proposal_norm["diagnostics"],
                    "task1": norm1["stage_diagnostics"],
                    "task2": norm2["stage_diagnostics"],
                },
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
            "audit_totals": None,
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

    audit_counts = [value["audit_counts"] for value in values]
    if all(value is None for value in audit_counts):
        her = cpp = hfs = None
        audit_totals = None
    elif any(value is None for value in audit_counts):
        raise ReportValidationError(f"{method}: raw audit counts are partially present")
    else:
        assert all(isinstance(value, Mapping) for value in audit_counts)
        audit_totals = {
            key: int(sum(int(value[key]) for value in audit_counts))
            for key in (
                "teacher_context_events",
                "forbidden_context_events",
                "comparison_events",
                "compared_token_positions",
                "same_prefix_positions",
            )
        }
        teacher_events = audit_totals["teacher_context_events"]
        compared_positions = audit_totals["compared_token_positions"]
        her = (
            audit_totals["forbidden_context_events"] / teacher_events
            if teacher_events
            else 0.0
        )
        cpp = (
            audit_totals["same_prefix_positions"] / compared_positions
            if compared_positions
            else 0.0
        )
        hfs = (1.0 - her) * cpp
    return {
        "n": len(rows),
        "correct": int(sum(float(value["correct"]) for value in values)),
        "accuracy": accuracy,
        "HER": her,
        "CPP": cpp,
        "HFS": hfs,
        "audit_totals": audit_totals,
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


def _scope_partitions(
    merged: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "amc23": [row for row in merged if row["source"] == "amc23"],
        "aime24": [row for row in merged if row["source"] == "aime24"],
        "aime25": [row for row in merged if row["source"] == "aime25"],
        "aime": [row for row in merged if row["source"] in {"aime24", "aime25"}],
        "overall": list(merged),
    }


def _diagnostic_scope(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "proposal": None,
            "CSD-T": None,
            "CSD-SD": None,
            "mean_generated_tokens_by_method": None,
        }

    proposal_rows = [row["diagnostics"]["proposal"] for row in rows]

    def mean(field: str, values: Sequence[Mapping[str, Any]]) -> float:
        return fmean(float(value[field]) for value in values)

    def task_summary(task_key: str) -> dict[str, Any]:
        values = [row["diagnostics"][task_key] for row in rows]
        return {
            "mean_proposal_seconds": mean("proposal_end_to_end_seconds", values),
            "mean_ridge_specialization_seconds": mean(
                "specialization_seconds", values
            ),
            "mean_distillation_seconds": mean("distillation_seconds", values),
            "mean_total_adaptation_seconds": mean(
                "total_adaptation_seconds", values
            ),
            "mean_peak_gpu_memory_bytes": mean("peak_memory_bytes", values),
            "max_peak_gpu_memory_bytes": max(
                float(value["peak_memory_bytes"]) for value in values
            ),
            "max_input_tokens": max(int(value["max_input_tokens"]) for value in values),
            "max_output_tokens": max(
                int(value["max_output_tokens"]) for value in values
            ),
            "mean_support_generated_tokens": mean(
                "support_generated_tokens", values
            ),
        }

    return {
        "n": len(rows),
        "proposal": {
            "mean_generation_seconds": mean(
                "proposal_generation_seconds", proposal_rows
            ),
            "mean_end_to_end_seconds": mean(
                "proposal_end_to_end_seconds", proposal_rows
            ),
            "mean_prompt_tokens": mean("proposal_prompt_tokens", proposal_rows),
            "mean_completion_tokens": mean(
                "proposal_completion_tokens", proposal_rows
            ),
            "mean_accepted_candidates_per_query": mean(
                "accepted_candidates", proposal_rows
            ),
            "mean_verification_yield": mean("verification_yield", proposal_rows),
        },
        "CSD-T": task_summary("task1"),
        "CSD-SD": task_summary("task2"),
        "mean_generated_tokens_by_method": {
            method: fmean(
                float(row["conditions"][method]["generated_tokens"]) for row in rows
            )
            for method in METHODS
        },
    }


def _aggregate(merged: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scopes = _scope_partitions(merged)
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
        "diagnostics_by_scope": {
            scope: _diagnostic_scope(rows) for scope, rows in scopes.items()
        },
        "csd_sd_teacher_gain_retention": retention,
        "retention_scope": "AIME24+AIME25",
    }


def _validate_allocation_provenance(
    proposal_rows: Mapping[str, Mapping[str, Any]],
    task1_rows: Mapping[str, Mapping[str, Any]],
    task2_rows: Mapping[str, Mapping[str, Any]],
    expected_counts: Mapping[str, int],
    expected_shard_count: int,
) -> dict[str, Any]:
    if expected_shard_count < 1:
        raise ReportValidationError("expected_shard_count must be positive")
    allocations: dict[tuple[str, int, str], dict[str, Any]] = {}
    for query_id in proposal_rows:
        query_task_ids: list[int] = []
        for label, row in (
            ("proposal", proposal_rows[query_id]),
            ("task1", task1_rows[query_id]),
            ("task2", task2_rows[query_id]),
        ):
            runtime_signature = _validate_runtime(
                row, f"{label} allocation query_id={query_id!r}"
            )
            array_job_id, task_id, hostname = runtime_signature["allocation"]
            query_task_ids.append(task_id)
            key = (array_job_id, task_id, hostname)
            visible_devices = str(
                _lookup(row, "runtime.cuda_visible_devices") or ""
            ).strip()
            existing = allocations.setdefault(
                key,
                {
                    "slurm_array_job_id": array_job_id,
                    "slurm_array_task_id": task_id,
                    "hostname": hostname,
                    "cuda_visible_devices": visible_devices,
                    "query_ids": set(),
                },
            )
            if (
                existing["cuda_visible_devices"]
                and visible_devices
                and existing["cuda_visible_devices"] != visible_devices
            ):
                raise ReportValidationError(
                    f"allocation {key} appears with multiple CUDA_VISIBLE_DEVICES values"
                )
            existing["query_ids"].add(query_id)
        # A timeout/requeue may split stages across array job IDs or hosts. The
        # content/config hashes bind the stages; the stable task id proves the
        # query remained on its deterministic shard.
        if len(set(query_task_ids)) != 1:
            raise ReportValidationError(
                f"query_id={query_id!r}: proposal/Task 1/Task 2 crossed shard task ids: "
                f"{query_task_ids}"
            )

    full_run = dict(expected_counts) == DEFAULT_EXPECTED_COUNTS
    observed_task_ids = {task_id for _, task_id, _ in allocations}
    expected_task_ids = set(range(expected_shard_count))
    if observed_task_ids != expected_task_ids:
        raise ReportValidationError(
            f"Report supplied {expected_shard_count} proposal/Task 1/Task 2 shard "
            f"triplets and therefore requires deterministic array task ids "
            f"{sorted(expected_task_ids)}, observed task ids {sorted(observed_task_ids)}"
        )
    return {
        "full_run_multi_b200_required": full_run,
        "expected_shard_count": expected_shard_count,
        "expected_array_task_ids": sorted(expected_task_ids),
        "distinct_array_task_allocations": len(allocations),
        "observed_array_task_ids": sorted(observed_task_ids),
        "allocations": [
            {
                **{key: value for key, value in allocation.items() if key != "query_ids"},
                "query_count": len(allocation["query_ids"]),
            }
            for _, allocation in sorted(allocations.items())
        ],
    }


def _shard_triplet_count(
    proposal_paths: Sequence[str | Path],
    task1_paths: Sequence[str | Path],
    task2_paths: Sequence[str | Path],
) -> int:
    path_counts = {
        "proposal": len(proposal_paths),
        "task1": len(task1_paths),
        "task2": len(task2_paths),
    }
    if 0 in path_counts.values() or len(set(path_counts.values())) != 1:
        raise ReportValidationError(
            "Reporter inputs must form one proposal/Task 1/Task 2 path triplet per "
            f"deterministic shard; got path counts {path_counts}"
        )
    return path_counts["proposal"]


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
    software_fields = (
        "git_commit",
        "git_dirty",
        "model",
        "model_revision",
        "torch",
        "cuda_runtime",
        "python_executable",
        "torch_overlay",
        "torch_module_path",
        "torch_arch_flags",
        "gpu_capabilities",
    )
    runtime_signatures = set()
    for row in all_rows:
        validated_runtime = _validate_runtime(row, "result metadata")
        runtime_signatures.add(
            tuple((key, validated_runtime[key]) for key in software_fields)
        )
    if len(runtime_signatures) != 1:
        raise ReportValidationError(
            "Task shards disagree on git commit, model revision, or software runtime: "
            f"{sorted(runtime_signatures)}"
        )
    task1_run_configs = {
        _sha256_value(
            _required(row, "task1 result metadata", "run_config_sha256"),
            "task1.run_config_sha256",
        )
        for row in task1_rows.values()
    }
    task2_run_configs = {
        _sha256_value(
            _required(row, "task2 result metadata", "run_config_sha256"),
            "task2.run_config_sha256",
        )
        for row in task2_rows.values()
    }
    if len(task1_run_configs) != 1 or len(task2_run_configs) != 1:
        raise ReportValidationError(
            "Task rows disagree on fixed run_config across queries/shards: "
            f"task1={sorted(task1_run_configs)}, task2={sorted(task2_run_configs)}"
        )
    ridge_configs = {
        _sha256_value(
            _required(row, "task result metadata", "ridge_config_sha256"),
            "task.ridge_config_sha256",
        )
        for row in [*task1_rows.values(), *task2_rows.values()]
    }
    if len(ridge_configs) != 1:
        raise ReportValidationError(
            "Task 1 and Task 2 rows disagree on the common ridge configuration: "
            f"{sorted(ridge_configs)}"
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
    amc_base = aggregate["by_method"]["Base"]["amc23"]
    amc_teacher = aggregate["by_method"]["CSD-T"]["amc23"]
    amc_student = aggregate["by_method"]["CSD-SD"]["amc23"]
    teacher_audit = (
        teacher if teacher["n"] else aggregate["by_method"]["CSD-T"]["overall"]
    )
    student_audit = (
        student if student["n"] else aggregate["by_method"]["CSD-SD"]["overall"]
    )
    teacher_gain = teacher["gain_vs_base_pp"]
    student_gain = student["gain_vs_base_pp"]
    amc_teacher_gain = amc_teacher["gain_vs_base_pp"]
    amc_student_gain = amc_student["gain_vs_base_pp"]
    amc_text = (
        "AMC23 is absent from this coverage override."
        if amc_teacher_gain is None or amc_student_gain is None
        else (
            f"On AMC23 dev, Base Acc@1 is {100.0 * amc_base['accuracy']:.2f}%. "
            f"CSD-T reaches {100.0 * amc_teacher['accuracy']:.2f}% "
            f"({float(amc_teacher_gain):+.2f} pp), and CSD-SD reaches "
            f"{100.0 * amc_student['accuracy']:.2f}% "
            f"({float(amc_student_gain):+.2f} pp)."
        )
    )
    if teacher_gain is None or student_gain is None:
        aime_text = (
            "The expected-count override contains no held-out AIME queries, so AIME accuracy, "
            "gain, HFAG, and teacher-gain retention are N/A."
        )
        conclusion = "This is a coverage/infrastructure smoke report, not PoC performance evidence."
    else:
        teacher_gain = float(teacher_gain)
        student_gain = float(student_gain)
        heldout_best_gain = max(teacher_gain, student_gain)
        dev_best_gain = max(
            float(amc_teacher_gain) if amc_teacher_gain is not None else float("-inf"),
            float(amc_student_gain) if amc_student_gain is not None else float("-inf"),
        )
        if heldout_best_gain >= 3.0:
            conclusion = (
                "The held-out AIME result meets the requested 3–4 percentage-point "
                "PoC signal threshold."
            )
        elif dev_best_gain >= 3.0:
            conclusion = (
                "The run shows a dev-only signal on AMC23; held-out AIME remains "
                "below the requested PoC threshold."
            )
        elif max(heldout_best_gain, dev_best_gain) > 0.0:
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
    teacher_overall = aggregate["by_method"]["CSD-T"]["overall"]
    student_overall = aggregate["by_method"]["CSD-SD"]["overall"]
    teacher_no_op_count = teacher_overall["protocol_no_op_count"] or 0
    teacher_no_op_rate = teacher_overall["protocol_no_op_rate"] or 0.0
    student_no_op_count = student_overall["protocol_no_op_count"] or 0
    student_no_op_rate = student_overall["protocol_no_op_rate"] or 0.0
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
        f"{amc_text} {aime_text} "
        f"Accuracy-based CSD-SD teacher-gain retention is {retention_text}.\n\n"
        f"CSD-T has HER={teacher_audit['HER']:.4f}, CPP={teacher_audit['CPP']:.4f}, "
        f"HFS={teacher_audit['HFS']:.4f}. "
        f"CSD-SD has HER={student_audit['HER']:.4f}, CPP={student_audit['CPP']:.4f}, "
        f"HFS={student_audit['HFS']:.4f}. {hfag_text} "
        f"CSD-T specialization no-ops: {teacher_no_op_count}/{teacher_overall['n']} "
        f"({100.0 * teacher_no_op_rate:.1f}%); "
        f"CSD-SD protocol no-ops: {student_no_op_count}/{student_overall['n']} "
        f"({100.0 * student_no_op_rate:.1f}%).\n\n"
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
    shard_triplet_count = _shard_triplet_count(
        proposal_paths, task1_paths, task2_paths
    )
    dataset = _load_dataset(dataset_path, counts)
    proposal_rows = _load_shards(proposal_paths, "proposal")
    task1_rows = _load_shards(task1_paths, "task1")
    task2_rows = _load_shards(task2_paths, "task2")
    proposal_normalized = _validate_proposal_rows(proposal_rows, dataset)
    task1_normalized = _validate_task_rows("task1", task1_rows, dataset)
    task2_normalized = _validate_task_rows("task2", task2_rows, dataset)
    configured_shard_counts = {
        row["num_shards"]
        for row in [*task1_normalized.values(), *task2_normalized.values()]
    }
    if configured_shard_counts != {shard_triplet_count}:
        raise ReportValidationError(
            "Task run_config.num_shards must equal the number of supplied shard "
            f"triplets={shard_triplet_count}, got {sorted(configured_shard_counts)}"
        )
    allocation_provenance = _validate_allocation_provenance(
        proposal_rows,
        task1_rows,
        task2_rows,
        counts,
        shard_triplet_count,
    )
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
            "specialization_ready_query_count": sum(
                not row["specialization_no_op"] for row in merged
            ),
            "specialization_no_op_query_count": sum(
                row["specialization_no_op"] for row in merged
            ),
            "proposal_row_count": len(proposal_rows),
            "task1_row_count": len(task1_rows),
            "task2_row_count": len(task2_rows),
            "allocation_provenance": allocation_provenance,
            "task1_run_config_sha256": next(
                iter(
                    {
                        row["run_config_sha256"]
                        for row in task1_normalized.values()
                    }
                )
            ),
            "task2_run_config_sha256": next(
                iter(
                    {
                        row["run_config_sha256"]
                        for row in task2_normalized.values()
                    }
                )
            ),
            "ridge_config_sha256": next(
                iter(
                    {
                        row["ridge_config_sha256"]
                        for row in task1_normalized.values()
                    }
                )
            ),
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
