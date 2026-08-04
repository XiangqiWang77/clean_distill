#!/usr/bin/env python3
"""Prepare the pinned, label-isolated empirical-study datasets.

The command is intentionally fail closed.  It streams parquet batches, selects
DeepMath examples by problem-content hash (not source row order), writes clean
query manifests separately from sealed targets, audits split overlap, and only
publishes a complete output directory after every check succeeds.

The formal CLI fixes the adopted proof-of-concept sizes at 1,000 distillation
episodes, 200 development examples, and AMC23/AIME24/AIME25 = 83/30/30.
Smaller counts are accepted by :func:`prepare_empirical_data` solely so the
same implementation can be exercised by lightweight unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from scripts.clean_self_distill.empirical_assets import (
    DEEPMATH_REVISION,
    DEEPMATH_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    sha256_file,
    tree_regular_bytes,
    verify_assets,
)
from src.clean_self_distill.heldout import validate_query_only_row
from src.clean_self_distill.io import (
    extract_answer,
    extract_problem,
    extract_solution,
    extract_source,
    stable_hash,
)


DISTILL_COUNT = 1_000
DEV_COUNT = 200
HELDOUT_COUNTS = {"amc23": 83, "aime24": 30, "aime25": 30}
MAX_NEW_DOWNLOAD_BYTES = 20_000_000_000
MAX_TASK_SCRATCH_BYTES = 100_000_000_000
SCHEMA_VERSION = "clean-self-distill-empirical-data-v1"
HELDOUT_BYTES = 105_032
HELDOUT_SHA256 = "42e7c50d0511fb52680ae6fc6cbfc46ff6c361771378dd5c6d228acb61be1cbf"


class DataFirewallError(ValueError):
    """Raised when an artifact could expose labels or violate the protocol."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def enforce_capacity_guards(
    *,
    task_root: Path,
    required_paths: Sequence[Path],
    new_download_bytes: int,
    max_new_download_bytes: int,
    max_task_bytes: int,
) -> int:
    """Enforce both user storage limits and return current task-root bytes."""
    root = task_root.resolve()
    if not root.is_dir():
        raise DataFirewallError(f"Task scratch root does not exist: {root}")
    for path in required_paths:
        if not _inside(path, root):
            raise DataFirewallError(f"Path escapes task scratch root: {path}")
    if new_download_bytes < 0 or new_download_bytes > max_new_download_bytes:
        raise DataFirewallError(
            "Pinned downloads exceed the user cap: "
            f"{new_download_bytes}>{max_new_download_bytes} bytes"
        )
    current_bytes = tree_regular_bytes(root)
    if current_bytes > max_task_bytes:
        raise DataFirewallError(
            f"Task scratch exceeds its cap: {current_bytes}>{max_task_bytes} bytes"
        )
    return current_bytes


def iter_parquet_rows(path: Path, *, batch_size: int = 256) -> Iterator[dict[str, Any]]:
    """Yield parquet records by bounded Arrow batches, never ``read_table``."""
    if batch_size <= 0:
        raise DataFirewallError("batch_size must be positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production dependency
        raise ImportError("Preparing parquet data requires pyarrow") from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            if not isinstance(row, dict):
                raise DataFirewallError(f"Non-object parquet row in {path}")
            yield row


def _explicit_difficulty(row: Mapping[str, Any]) -> int | None:
    values: list[Any] = []
    for key in ("difficulty", "level"):
        values.append(row.get(key))
    extra = row.get("extra_info")
    if isinstance(extra, Mapping):
        for key in ("difficulty", "level"):
            values.append(extra.get(key))
    for value in values:
        if value is None or str(value).strip() == "":
            continue
        match = re.search(r"(?<!\d)(10|[0-9])(?!\d)", str(value))
        if match:
            return int(match.group(1))
    return None


def _target_fingerprint(answer: str, solution: str) -> str:
    return hashlib.sha256((answer + "\0" + solution).encode("utf-8")).hexdigest()


def _clean_record(
    *, problem: str, source: str, query_id: str | None = None
) -> dict[str, str]:
    problem_hash = stable_hash(problem, 64)
    record = {
        "query_id": query_id or f"{source}:{problem_hash}",
        "problem": problem,
        "problem_sha256": problem_hash,
        "source": source,
    }
    return validate_query_only_row(record, context=f"query {record['query_id']}")


def _sealed_record(
    query: Mapping[str, str], *, answer: str, solution: str
) -> dict[str, str]:
    if not answer:
        raise DataFirewallError(f"Missing sealed answer for {query['query_id']}")
    return {
        "query_id": query["query_id"],
        "problem_sha256": query["problem_sha256"],
        "source": query["source"],
        "answer": answer,
        "reference_solution": solution,
    }


def select_deepmath_records(
    path: Path,
    *,
    count: int,
    batch_size: int = 256,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return the ``count`` smallest unique problem hashes from DeepMath."""
    if count <= 0:
        raise DataFirewallError("DeepMath selection count must be positive")
    # A max-heap represented by negative SHA-256 integers bounds retained raw
    # rows to ``count`` even when the source parquet is much larger.
    selected: list[tuple[int, str, dict[str, str]]] = []
    seen_targets: dict[str, str] = {}
    stats: dict[str, Any] = {
        "rows_streamed": 0,
        "eligible_unique_rows": 0,
        "duplicate_problem_rows": 0,
        "missing_problem_rows": 0,
        "missing_target_rows": 0,
        "explicit_out_of_range_rows": 0,
        "explicit_difficulty_rows": 0,
        "implicit_pinned_difficulty_rows": 0,
    }
    for raw in iter_parquet_rows(path, batch_size=batch_size):
        stats["rows_streamed"] += 1
        problem = extract_problem(raw).strip()
        if not problem:
            stats["missing_problem_rows"] += 1
            continue
        answer = extract_answer(raw).strip()
        solution = extract_solution(raw).strip()
        # Both fields are sealed for offline scoring / privileged controls.  A
        # row without either cannot support the preregistered comparison.
        if not answer or not solution:
            stats["missing_target_rows"] += 1
            continue
        difficulty = _explicit_difficulty(raw)
        if difficulty is None:
            stats["implicit_pinned_difficulty_rows"] += 1
        else:
            stats["explicit_difficulty_rows"] += 1
            if difficulty < 7 or difficulty > 10:
                stats["explicit_out_of_range_rows"] += 1
                continue

        problem_hash = stable_hash(problem, 64)
        target_hash = _target_fingerprint(answer, solution)
        previous = seen_targets.get(problem_hash)
        if previous is not None:
            stats["duplicate_problem_rows"] += 1
            if previous != target_hash:
                raise DataFirewallError(
                    "DeepMath contains duplicate problem text with conflicting targets: "
                    f"{problem_hash}"
                )
            continue
        seen_targets[problem_hash] = target_hash
        stats["eligible_unique_rows"] += 1
        retained = {
            "problem": problem,
            "problem_sha256": problem_hash,
            "answer": answer,
            "reference_solution": solution,
        }
        key = int(problem_hash, 16)
        item = (-key, problem_hash, retained)
        if len(selected) < count:
            heapq.heappush(selected, item)
        elif key < -selected[0][0]:
            heapq.heapreplace(selected, item)

    if len(selected) != count:
        raise DataFirewallError(
            f"DeepMath has only {len(selected)} eligible unique rows; need {count}"
        )
    records = [item[2] for item in sorted(selected, key=lambda item: item[1])]
    stats["selected_rows"] = len(records)
    stats["selection_rule"] = "ascending_sha256_of_exact_problem_utf8"
    return records, stats


def _heldout_source(row: Mapping[str, Any]) -> str:
    candidates: list[Any] = [
        row.get("source"),
        row.get("data_source"),
        row.get("dataset"),
        row.get("benchmark"),
    ]
    extra = row.get("extra_info")
    if isinstance(extra, Mapping):
        candidates.extend(
            extra.get(key)
            for key in ("source", "data_source", "dataset", "benchmark")
        )
    # Keep the shared extractor as a final schema-compatible fallback.
    candidates.append(extract_source(dict(row)))
    for value in candidates:
        compact = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
        if "amc23" in compact or "amc2023" in compact:
            return "amc23"
        if "aime24" in compact or "aime2024" in compact:
            return "aime24"
        if "aime25" in compact or "aime2025" in compact:
            return "aime25"
    raise DataFirewallError(
        "Cannot assign held-out row to exactly one of AMC23/AIME24/AIME25"
    )


def load_heldout_records(
    path: Path,
    *,
    expected_counts: Mapping[str, int],
    batch_size: int = 256,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    queries: list[dict[str, str]] = []
    labels: list[dict[str, str]] = []
    seen: set[str] = set()
    observed = {source: 0 for source in expected_counts}
    rows_streamed = 0
    for raw in iter_parquet_rows(path, batch_size=batch_size):
        rows_streamed += 1
        source = _heldout_source(raw)
        if source not in expected_counts:
            raise DataFirewallError(f"Unexpected held-out source {source!r}")
        problem = extract_problem(raw).strip()
        answer = extract_answer(raw).strip()
        solution = extract_solution(raw).strip()
        if not problem or not answer:
            raise DataFirewallError(
                f"Held-out {source} row {rows_streamed} lacks problem or answer"
            )
        problem_hash = stable_hash(problem, 64)
        if problem_hash in seen:
            raise DataFirewallError(f"Duplicate held-out problem {problem_hash}")
        seen.add(problem_hash)
        query = _clean_record(problem=problem, source=source)
        queries.append(query)
        labels.append(_sealed_record(query, answer=answer, solution=solution))
        observed[source] += 1

    if observed != dict(expected_counts):
        raise DataFirewallError(
            f"Held-out source counts mismatch expected={dict(expected_counts)} "
            f"observed={observed}"
        )
    order = {source: index for index, source in enumerate(expected_counts)}
    paired = sorted(
        zip(queries, labels),
        key=lambda pair: (order[pair[0]["source"]], pair[0]["problem_sha256"]),
    )
    queries = [query for query, _ in paired]
    labels = [label for _, label in paired]
    return queries, labels, {
        "rows_streamed": rows_streamed,
        "source_counts": observed,
        "selection_rule": "source_then_ascending_sha256_of_exact_problem_utf8",
    }


def _normalized_problem_hash(problem: str) -> str:
    normalized = " ".join(problem.split()).casefold()
    return stable_hash(normalized, 64)


def audit_overlap(splits: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, Any]:
    exact = {
        name: {row["problem_sha256"] for row in rows}
        for name, rows in splits.items()
    }
    normalized = {
        name: {_normalized_problem_hash(row["problem"]) for row in rows}
        for name, rows in splits.items()
    }
    for name, rows in splits.items():
        if len(exact[name]) != len(rows) or len(normalized[name]) != len(rows):
            raise DataFirewallError(f"Duplicate problems inside split {name!r}")

    pairwise: dict[str, dict[str, int]] = {}
    names = list(splits)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = f"{left}__{right}"
            exact_count = len(exact[left] & exact[right])
            normalized_count = len(normalized[left] & normalized[right])
            pairwise[key] = {
                "exact_problem_overlap": exact_count,
                "whitespace_casefold_overlap": normalized_count,
            }
            if exact_count or normalized_count:
                raise DataFirewallError(
                    f"Data split overlap detected for {left}/{right}: {pairwise[key]}"
                )
    return {
        "within_split_unique": {name: len(values) for name, values in exact.items()},
        "pairwise": pairwise,
        "passed": True,
    }


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _file_description(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def prepare_empirical_data(
    *,
    deepmath_path: Path,
    heldout_path: Path,
    output_dir: Path,
    task_root: Path,
    distill_count: int = DISTILL_COUNT,
    dev_count: int = DEV_COUNT,
    heldout_counts: Mapping[str, int] = HELDOUT_COUNTS,
    deepmath_revision: str = DEEPMATH_REVISION,
    heldout_revision: str = DEEPMATH_REVISION,
    deepmath_sha256: str | None = None,
    heldout_sha256: str | None = None,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    new_download_bytes: int = 0,
    max_new_download_bytes: int = MAX_NEW_DOWNLOAD_BYTES,
    max_task_bytes: int = MAX_TASK_SCRATCH_BYTES,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Build all clean/sealed splits and publish them as one atomic directory."""
    root = task_root.resolve()
    destination = output_dir.resolve()
    if destination == root or not destination.is_relative_to(root):
        raise DataFirewallError("output_dir must be a dedicated directory under task scratch")
    if destination.exists():
        raise DataFirewallError(f"Refusing to overwrite existing output: {destination}")
    if not deepmath_revision or not heldout_revision:
        raise DataFirewallError("Both dataset revisions must be pinned and nonempty")
    bytes_before = enforce_capacity_guards(
        task_root=root,
        required_paths=(deepmath_path, heldout_path, destination.parent),
        new_download_bytes=new_download_bytes,
        max_new_download_bytes=max_new_download_bytes,
        max_task_bytes=max_task_bytes,
    )

    selected, deepmath_stats = select_deepmath_records(
        deepmath_path, count=distill_count + dev_count, batch_size=batch_size
    )
    distill_raw = selected[:distill_count]
    dev_raw = selected[distill_count:]

    def split_records(
        rows: Sequence[Mapping[str, str]], split: str
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        queries: list[dict[str, str]] = []
        labels: list[dict[str, str]] = []
        for row in rows:
            query = _clean_record(
                problem=row["problem"],
                source="deepmath",
                query_id=f"deepmath:{row['problem_sha256']}",
            )
            queries.append(query)
            labels.append(
                _sealed_record(
                    query,
                    answer=row["answer"],
                    solution=row["reference_solution"],
                )
            )
        if len(queries) != (distill_count if split == "distill" else dev_count):
            raise AssertionError("Internal split-size error")
        return queries, labels

    distill_queries, distill_labels = split_records(distill_raw, "distill")
    dev_queries, dev_labels = split_records(dev_raw, "dev")
    heldout_queries, heldout_labels, heldout_stats = load_heldout_records(
        heldout_path, expected_counts=heldout_counts, batch_size=batch_size
    )
    overlap = audit_overlap(
        {
            "distill": distill_queries,
            "dev": dev_queries,
            "heldout": heldout_queries,
        }
    )
    proposal_queries = distill_queries + dev_queries + heldout_queries
    expected_total = distill_count + dev_count + sum(heldout_counts.values())
    if len(proposal_queries) != expected_total:
        raise AssertionError("Internal proposal-query count error")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        artifacts_and_rows = {
            "distill_queries.jsonl": distill_queries,
            "dev_queries.jsonl": dev_queries,
            "heldout_queries.jsonl": heldout_queries,
            "distill_labels.sealed.jsonl": distill_labels,
            "dev_labels.sealed.jsonl": dev_labels,
            "heldout_labels.sealed.jsonl": heldout_labels,
            "proposal_queries.jsonl": proposal_queries,
        }
        for name, rows in artifacts_and_rows.items():
            _atomic_jsonl(staging / name, rows)
        for name in (
            "distill_labels.sealed.jsonl",
            "dev_labels.sealed.jsonl",
            "heldout_labels.sealed.jsonl",
        ):
            os.chmod(staging / name, 0o600)

        artifacts = {
            name: _file_description(staging / name) for name in artifacts_and_rows
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "model": {"id": model_id, "revision": model_revision},
            "inputs": {
                "deepmath": {
                    "path": str(deepmath_path.resolve()),
                    "revision": deepmath_revision,
                    "sha256": deepmath_sha256 or sha256_file(deepmath_path),
                    "bytes": deepmath_path.stat().st_size,
                    "declared_difficulty": "7-10",
                },
                "heldout": {
                    "path": str(heldout_path.resolve()),
                    "revision": heldout_revision,
                    "sha256": heldout_sha256 or sha256_file(heldout_path),
                    "bytes": heldout_path.stat().st_size,
                },
            },
            "counts": {
                "distill": len(distill_queries),
                "dev": len(dev_queries),
                "heldout": len(heldout_queries),
                "heldout_by_source": heldout_stats["source_counts"],
                "proposal_queries": len(proposal_queries),
            },
            "canonical_order": {
                "deepmath": "ascending_sha256_of_exact_problem_utf8; first distill then dev",
                "heldout": "AMC23 then AIME24 then AIME25; hash order within source",
                "proposal_queries": "distill + dev + heldout",
                "query_id_sha256": _canonical_sha256(
                    [row["query_id"] for row in proposal_queries]
                ),
            },
            "data_firewall": {
                "query_files_physically_exclude_targets": True,
                "sealed_label_files": sorted(
                    name for name in artifacts if ".sealed." in name
                ),
                "clean_gpu_inputs": sorted(
                    name for name in artifacts if "queries.jsonl" in name
                ),
            },
            "deepmath_stream_audit": deepmath_stats,
            "heldout_stream_audit": heldout_stats,
            "overlap_audit": overlap,
            "capacity_guard": {
                "new_download_bytes": new_download_bytes,
                "max_new_download_bytes": max_new_download_bytes,
                "task_scratch_bytes_before_prepare": bytes_before,
                "max_task_scratch_bytes": max_task_bytes,
            },
            "artifacts": artifacts,
        }
        _atomic_json(staging / "manifest.json", manifest)

        # The staging directory is already inside task scratch, so this scan is
        # a true post-write upper bound.  Renaming it does not increase usage.
        bytes_after = tree_regular_bytes(root)
        if bytes_after > max_task_bytes:
            raise DataFirewallError(
                f"Prepared artifacts would exceed task scratch cap: "
                f"{bytes_after}>{max_task_bytes} bytes"
            )
        os.replace(staging, destination)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepmath", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deepmath-revision", default=DEEPMATH_REVISION)
    parser.add_argument("--heldout-revision", default=DEEPMATH_REVISION)
    parser.add_argument("--max-new-download-bytes", type=int, default=MAX_NEW_DOWNLOAD_BYTES)
    parser.add_argument("--max-task-bytes", type=int, default=MAX_TASK_SCRATCH_BYTES)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--full-asset-hash", action="store_true")
    args = parser.parse_args()

    deepmath = Path(args.deepmath)
    heldout = Path(args.heldout)
    task_root = Path(args.task_root)
    asset_manifest = verify_assets(
        model_dir=Path(args.model_dir),
        deepmath=deepmath,
        task_root=task_root,
        max_new_download_bytes=args.max_new_download_bytes,
        max_task_bytes=args.max_task_bytes,
        full_hash=args.full_asset_hash,
    )
    # Always verify the 588 MB dataset payload against the pinned revision.
    # ``--full-asset-hash`` additionally verifies every 8B model shard, but a
    # caller may omit that expensive 16 GB pass without weakening data-split
    # identity.
    actual_deepmath_sha256 = sha256_file(deepmath)
    if actual_deepmath_sha256 != DEEPMATH_SHA256:
        raise DataFirewallError(
            "DeepMath SHA-256 disagrees with the pinned RLCSD revision: "
            f"expected={DEEPMATH_SHA256} actual={actual_deepmath_sha256}"
        )
    if heldout.stat().st_size != HELDOUT_BYTES:
        raise DataFirewallError(
            "Held-out parquet size disagrees with the pinned RLCSD revision: "
            f"expected={HELDOUT_BYTES} actual={heldout.stat().st_size}"
        )
    actual_heldout_sha256 = sha256_file(heldout)
    if actual_heldout_sha256 != HELDOUT_SHA256:
        raise DataFirewallError(
            "Held-out parquet SHA-256 disagrees with the pinned RLCSD revision: "
            f"expected={HELDOUT_SHA256} actual={actual_heldout_sha256}"
        )
    # Count the held-out parquet as a downloaded asset as well.  It is small,
    # but excluding it would make the 20 GB audit semantically incomplete.
    new_download_bytes = int(asset_manifest["new_download_bytes"]) + heldout.stat().st_size
    manifest = prepare_empirical_data(
        deepmath_path=deepmath,
        heldout_path=heldout,
        output_dir=Path(args.output_dir),
        task_root=task_root,
        deepmath_revision=args.deepmath_revision,
        heldout_revision=args.heldout_revision,
        deepmath_sha256=actual_deepmath_sha256,
        heldout_sha256=actual_heldout_sha256,
        model_id=str(asset_manifest["model_id"]),
        model_revision=str(asset_manifest["model_revision"]),
        new_download_bytes=new_download_bytes,
        max_new_download_bytes=args.max_new_download_bytes,
        max_task_bytes=args.max_task_bytes,
        batch_size=args.batch_size,
    )
    print(_canonical_json(manifest))


if __name__ == "__main__":
    main()
