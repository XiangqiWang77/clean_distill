#!/usr/bin/env python3
"""Validation and cache helpers for the multi-B200 Slurm launcher.

This module deliberately lives beside the launchers rather than in ``src``.
It does not implement any experiment method; it only makes asset preparation,
restart repair, shard validation, and smoke-report materialization strict.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clean_self_distill.io import (
    iter_rows,
    load_query_records,
    validate_proposal_training_binding,
    validate_specialization_state,
)


MODEL_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.py",
    "*.jinja",
    "*.yaml",
    "*.yml",
)
EXPECTED_SOURCES = {"amc23": 83, "aime24": 30, "aime25": 30}


class LauncherValidationError(ValueError):
    """Raised when an asset or run artifact is unsafe to consume."""


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _size_bytes(paths: Iterable[Path]) -> int:
    """Count regular-file storage once, without following snapshot symlinks."""
    total = 0
    seen: set[tuple[int, int]] = set()
    for root in paths:
        if not root.exists() and not root.is_symlink():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for candidate in candidates:
            try:
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                stat = candidate.stat()
            except FileNotFoundError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            total += stat.st_size
    return total


def _source(value: Any) -> str:
    compact = "".join(character for character in str(value).lower() if character.isalnum())
    return {
        "amc23": "amc23",
        "amc2023": "amc23",
        "aime24": "aime24",
        "aime2024": "aime24",
        "aime25": "aime25",
        "aime2025": "aime25",
    }.get(compact, compact)


def _selected_model_files(model_info: Any) -> list[Any]:
    selected = [
        sibling
        for sibling in model_info.siblings
        if any(fnmatch.fnmatch(sibling.rfilename, pattern) for pattern in MODEL_ALLOW_PATTERNS)
    ]
    missing_sizes = [sibling.rfilename for sibling in selected if sibling.size is None]
    if missing_sizes:
        raise LauncherValidationError(
            "Hugging Face did not report sizes for selected files: "
            + ", ".join(missing_sizes)
        )
    if not any(sibling.rfilename.endswith(".safetensors") for sibling in selected):
        raise LauncherValidationError("Pinned model revision has no selected safetensors weights")
    required = {"config.json", "tokenizer_config.json"}
    selected_names = {sibling.rfilename for sibling in selected}
    missing = sorted(required - selected_names)
    if missing:
        raise LauncherValidationError(f"Pinned model revision is missing required files: {missing}")
    return selected


def _verify_snapshot_files(snapshot: Path) -> None:
    if not snapshot.is_dir():
        raise LauncherValidationError(f"Model snapshot is not a directory: {snapshot}")
    required = (snapshot / "config.json", snapshot / "tokenizer_config.json")
    missing = [str(path) for path in required if not path.is_file()]
    weights = list(snapshot.glob("*.safetensors"))
    if missing or not weights:
        raise LauncherValidationError(
            f"Incomplete model snapshot {snapshot}; missing={missing}, safetensors={len(weights)}"
        )
    empty = [str(path) for path in (*required, *weights) if path.stat().st_size == 0]
    if empty:
        raise LauncherValidationError(f"Model snapshot contains empty files: {empty}")


def cmd_prefetch_model(args: argparse.Namespace) -> None:
    if os.environ.get("HF_HUB_DISABLE_XET") != "1":
        raise LauncherValidationError("Set HF_HUB_DISABLE_XET=1 before model prefetch")
    from huggingface_hub import HfApi, snapshot_download

    cache_dir = Path(args.cache_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    other_paths = [Path(value).resolve() for value in args.other_path]
    cache_dir.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(args.model, revision=args.revision, files_metadata=True)
    resolved_revision = str(info.sha)
    if resolved_revision != args.revision:
        raise LauncherValidationError(
            f"Requested revision {args.revision}, Hugging Face resolved {resolved_revision}"
        )
    selected = _selected_model_files(info)
    selected_bytes = sum(int(sibling.size) for sibling in selected)
    other_bytes = _size_bytes(other_paths)
    if selected_bytes + other_bytes >= args.max_bytes:
        raise LauncherValidationError(
            "Pinned model plus existing dataset would violate the download ceiling: "
            f"model={selected_bytes}, other={other_bytes}, cap={args.max_bytes} bytes"
        )

    # This cache is intentionally dedicated to one model and one pinned revision.
    expected_repo_dir = f"models--{args.model.replace('/', '--')}"
    unexpected = [
        path.name
        for path in cache_dir.glob("models--*")
        if path.name != expected_repo_dir
    ]
    if unexpected:
        raise LauncherValidationError(
            f"Dedicated model cache contains other repositories: {sorted(unexpected)}"
        )
    snapshots_dir = cache_dir / expected_repo_dir / "snapshots"
    if snapshots_dir.is_dir():
        unexpected_revisions = [
            path.name for path in snapshots_dir.iterdir() if path.name != args.revision
        ]
        if unexpected_revisions:
            raise LauncherValidationError(
                "Dedicated model cache contains unpinned snapshot revisions: "
                f"{sorted(unexpected_revisions)}"
            )

    snapshot = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            cache_dir=cache_dir,
            allow_patterns=list(MODEL_ALLOW_PATTERNS),
            max_workers=args.max_workers,
        )
    )
    _verify_snapshot_files(snapshot)
    actual_bytes = _size_bytes([cache_dir, *other_paths])
    if actual_bytes >= args.max_bytes:
        raise LauncherValidationError(
            f"Downloaded assets use {actual_bytes} bytes, violating cap {args.max_bytes}"
        )
    manifest = {
        "schema_version": "clean-distill-prefetch-v1",
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot.resolve()),
        "allow_patterns": list(MODEL_ALLOW_PATTERNS),
        "selected_remote_bytes": selected_bytes,
        "download_roots_bytes": actual_bytes,
        "max_download_bytes_exclusive": args.max_bytes,
        "created_at_epoch": int(time.time()),
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))


def cmd_verify_model(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise LauncherValidationError(f"Missing model prefetch manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": args.revision,
    }
    mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise LauncherValidationError(f"Model prefetch manifest mismatch: {mismatch}")
    snapshot = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            allow_patterns=list(MODEL_ALLOW_PATTERNS),
            local_files_only=True,
        )
    )
    _verify_snapshot_files(snapshot)
    if snapshot.resolve() != Path(manifest["snapshot_path"]).resolve():
        raise LauncherValidationError(
            f"Cached snapshot moved: manifest={manifest['snapshot_path']}, actual={snapshot}"
        )
    print(snapshot)


def cmd_check_budget(args: argparse.Namespace) -> None:
    paths = [Path(value).resolve() for value in args.path]
    actual = _size_bytes(paths)
    if actual >= args.max_bytes:
        raise LauncherValidationError(
            f"Download roots use {actual} bytes, violating cap {args.max_bytes}"
        )
    print(json.dumps({"bytes": actual, "cap_exclusive": args.max_bytes, "paths": args.path}))


def cmd_validate_dataset(args: argparse.Namespace) -> None:
    path = Path(args.dataset)
    if not path.is_file() or path.stat().st_size == 0:
        raise LauncherValidationError(f"Missing or empty evaluation dataset: {path}")
    records = load_query_records(path, include_targets=True)
    counts = Counter(_source(record["source"]) for record in records)
    if dict(counts) != EXPECTED_SOURCES:
        raise LauncherValidationError(
            f"Evaluation dataset source counts are {dict(counts)}, expected {EXPECTED_SOURCES}"
        )
    missing_answers = [record["query_id"] for record in records if not record.get("answer")]
    if missing_answers:
        raise LauncherValidationError(
            f"Evaluation dataset has {len(missing_answers)} records without answers"
        )
    print(json.dumps({"records": len(records), "source_counts": dict(counts)}))


def cmd_repair_proposals(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if not path.exists():
        print(json.dumps({"path": str(path), "rows": 0, "repaired": False}))
        return
    raw = path.read_bytes()
    raw_lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    invalid_at: int | None = None
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            raise LauncherValidationError(
                f"Proposal JSONL contains a blank record at line {index + 1}: {path}"
            )
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index != len(raw_lines) - 1 or raw_line.endswith((b"\n", b"\r")):
                raise LauncherValidationError(
                    f"Proposal JSONL corruption is not an unterminated final write: {path}"
                ) from exc
            invalid_at = index
            break
        if not isinstance(value, dict):
            raise LauncherValidationError(
                f"Proposal JSONL line {index + 1} is not an object: {path}"
            )
        rows.append(value)
    ids = [str(row.get("query_id", "")).strip() for row in rows]
    if any(not query_id for query_id in ids) or len(ids) != len(set(ids)):
        raise LauncherValidationError(f"Proposal JSONL has empty or duplicate query IDs: {path}")
    repair_required = invalid_at is not None or bool(raw_lines and not raw.endswith(b"\n"))
    if repair_required:
        backup = path.with_name(f"{path.name}.corrupt.{int(time.time())}")
        shutil.copy2(path, backup)
        temporary = path.with_name(f".{path.name}.repair.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    print(
        json.dumps(
            {"path": str(path), "rows": len(rows), "repaired": repair_required}
        )
    )


def _expected_shard(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = load_query_records(
        args.dataset,
        include_targets=True,
        max_samples=args.max_samples,
    )
    return [
        record
        for index, record in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]


def cmd_validate_shard(args: argparse.Namespace) -> None:
    expected = _expected_shard(args)
    if not expected:
        raise LauncherValidationError(
            f"Shard {args.shard_index}/{args.num_shards} has no selected dataset records"
        )
    artifact = Path(args.artifact)
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise LauncherValidationError(f"Missing or empty {args.kind} artifact: {artifact}")
    rows = list(iter_rows(artifact))
    expected_ids = [record["query_id"] for record in expected]
    actual_ids = [str(row.get("query_id", "")).strip() for row in rows]
    if actual_ids != expected_ids:
        raise LauncherValidationError(
            f"{args.kind} shard IDs/order mismatch: expected={expected_ids}, actual={actual_ids}"
        )

    expected_by_id = {record["query_id"]: record for record in expected}
    expected_stage = {
        "task1": "task1_fast_teacher",
        "task2": "task2_clean_distillation",
    }.get(args.kind)
    for row in rows:
        query_id = str(row["query_id"])
        record = expected_by_id[query_id]
        if str(row.get("problem_sha256", "")) != record["problem_sha256"]:
            raise LauncherValidationError(f"{args.kind} problem hash mismatch for {query_id}")
        if _source(row.get("source", "")) != _source(record["source"]):
            raise LauncherValidationError(f"{args.kind} source mismatch for {query_id}")
        if args.kind == "proposal":
            try:
                status, _, _ = validate_specialization_state(
                    row, context=f"Proposal {query_id!r}"
                )
                validate_proposal_training_binding(
                    row, context=f"Proposal {query_id!r}"
                )
            except ValueError as exc:
                raise LauncherValidationError(str(exc)) from exc
            candidates = row["specialization_candidates"]
            candidate_count = row.get("candidate_count")
            requested_count = row.get("requested_candidate_count")
            minimum_count = row.get("minimum_candidate_count")
            if (
                isinstance(candidate_count, bool)
                or not isinstance(candidate_count, int)
                or candidate_count != len(candidates)
                or isinstance(requested_count, bool)
                or not isinstance(requested_count, int)
                or requested_count < 1
                or isinstance(minimum_count, bool)
                or not isinstance(minimum_count, int)
                or minimum_count < 1
                or minimum_count > requested_count
                or candidate_count > requested_count
            ):
                raise LauncherValidationError(
                    f"Proposal candidate-count contract is inconsistent for {query_id}"
                )
            if (status == "ready") != (candidate_count >= minimum_count):
                raise LauncherValidationError(
                    f"Proposal specialization status disagrees with the candidate "
                    f"quality gate for {query_id}"
                )
        if str(row.get("model", "")) != args.model:
            raise LauncherValidationError(f"{args.kind} model mismatch for {query_id}")
        if str(row.get("model_revision", "")) != args.revision:
            raise LauncherValidationError(f"{args.kind} revision mismatch for {query_id}")
        stage_marker = str(row.get("stage") or row.get("task") or "")
        if expected_stage is not None and stage_marker != expected_stage:
            raise LauncherValidationError(f"{args.kind} stage mismatch for {query_id}")
    print(
        json.dumps(
            {
                "kind": args.kind,
                "artifact": str(artifact),
                "records": len(rows),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
            sort_keys=True,
        )
    )


def cmd_prepare_report_dataset(args: argparse.Namespace) -> None:
    source_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve()
    metadata_path = Path(args.metadata).resolve()
    if args.max_samples is None:
        report_path = source_path
    else:
        if source_path.suffix.lower() != ".parquet":
            raise LauncherValidationError("Smoke report materialization currently requires parquet")
        import pyarrow.parquet as pq

        table = pq.read_table(source_path)
        if args.max_samples > table.num_rows:
            raise LauncherValidationError(
                f"max_samples={args.max_samples} exceeds dataset rows={table.num_rows}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        pq.write_table(table.slice(0, args.max_samples), temporary)
        temporary.replace(output_path)
        original_ids = [
            row["query_id"]
            for row in load_query_records(
                source_path, include_targets=False, max_samples=args.max_samples
            )
        ]
        materialized_ids = [
            row["query_id"]
            for row in load_query_records(output_path, include_targets=False)
        ]
        if materialized_ids != original_ids:
            output_path.unlink(missing_ok=True)
            raise LauncherValidationError(
                "Materialized smoke dataset changed canonical query IDs"
            )
        report_path = output_path

    records = load_query_records(report_path, include_targets=False)
    counts = Counter(_source(record["source"]) for record in records)
    unexpected = sorted(set(counts) - set(EXPECTED_SOURCES))
    if unexpected:
        raise LauncherValidationError(f"Unexpected report dataset sources: {unexpected}")
    normalized_counts = {source: int(counts.get(source, 0)) for source in EXPECTED_SOURCES}
    metadata = {
        "dataset": str(report_path),
        "records": len(records),
        "expected_counts": normalized_counts,
    }
    _atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, sort_keys=True))


def cmd_report_args(args: argparse.Namespace) -> None:
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    print(f"--dataset={metadata['dataset']}")
    for source in EXPECTED_SOURCES:
        print(f"--expected-count={source}={int(metadata['expected_counts'][source])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prefetch = subparsers.add_parser("prefetch-model")
    prefetch.add_argument("--model", required=True)
    prefetch.add_argument("--revision", required=True)
    prefetch.add_argument("--cache-dir", required=True)
    prefetch.add_argument("--manifest", required=True)
    prefetch.add_argument("--other-path", action="append", default=[])
    prefetch.add_argument("--max-bytes", type=int, required=True)
    prefetch.add_argument("--max-workers", type=int, default=4)
    prefetch.set_defaults(func=cmd_prefetch_model)

    verify = subparsers.add_parser("verify-model")
    verify.add_argument("--model", required=True)
    verify.add_argument("--revision", required=True)
    verify.add_argument("--cache-dir", required=True)
    verify.add_argument("--manifest", required=True)
    verify.set_defaults(func=cmd_verify_model)

    budget = subparsers.add_parser("check-budget")
    budget.add_argument("--path", action="append", required=True)
    budget.add_argument("--max-bytes", type=int, required=True)
    budget.set_defaults(func=cmd_check_budget)

    dataset = subparsers.add_parser("validate-dataset")
    dataset.add_argument("--dataset", required=True)
    dataset.set_defaults(func=cmd_validate_dataset)

    repair = subparsers.add_parser("repair-proposals")
    repair.add_argument("--path", required=True)
    repair.set_defaults(func=cmd_repair_proposals)

    shard = subparsers.add_parser("validate-shard")
    shard.add_argument("--dataset", required=True)
    shard.add_argument("--max-samples", type=int)
    shard.add_argument("--num-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument(
        "--kind", choices=("proposal", "adapter", "task1", "task2"), required=True
    )
    shard.add_argument("--artifact", required=True)
    shard.add_argument("--model", required=True)
    shard.add_argument("--revision", required=True)
    shard.set_defaults(func=cmd_validate_shard)

    report = subparsers.add_parser("prepare-report-dataset")
    report.add_argument("--dataset", required=True)
    report.add_argument("--max-samples", type=int)
    report.add_argument("--output", required=True)
    report.add_argument("--metadata", required=True)
    report.set_defaults(func=cmd_prepare_report_dataset)

    report_args = subparsers.add_parser("report-args")
    report_args.add_argument("--metadata", required=True)
    report_args.set_defaults(func=cmd_report_args)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except LauncherValidationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
