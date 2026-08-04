#!/usr/bin/env python3
"""Merge sharded corrective proposals in canonical query-manifest order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from src.clean_self_distill.heldout import load_query_only_manifest
from src.clean_self_distill.io import (
    iter_rows,
    validate_proposal_training_binding,
    validate_specialization_state,
)


class ProposalMergeError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def merge_proposals(
    query_manifest: str | Path, shard_paths: list[str | Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries = load_query_only_manifest(query_manifest)
    expected = {row["query_id"]: row for row in queries}
    proposals: dict[str, dict[str, Any]] = {}
    for shard_path in shard_paths:
        rows = list(iter_rows(shard_path))
        if not rows:
            raise ProposalMergeError(f"Proposal shard is empty: {shard_path}")
        for row_number, raw in enumerate(rows, 1):
            row = dict(raw)
            query_id = str(row.get("query_id", ""))
            if query_id not in expected:
                raise ProposalMergeError(f"Unexpected proposal query {query_id!r}")
            if query_id in proposals:
                raise ProposalMergeError(f"Duplicate proposal query {query_id!r}")
            query = expected[query_id]
            if (
                row.get("problem") != query["problem"]
                or row.get("problem_sha256") != query["problem_sha256"]
                or str(row.get("source", "")).casefold() != query["source"]
            ):
                raise ProposalMergeError(
                    f"{shard_path} row {row_number} disagrees with query manifest"
                )
            try:
                validate_specialization_state(row, context=f"Proposal {query_id}")
                validate_proposal_training_binding(row, context=f"Proposal {query_id}")
            except ValueError as exc:
                raise ProposalMergeError(str(exc)) from exc
            proposals[query_id] = row
    if set(proposals) != set(expected):
        missing = sorted(set(expected) - set(proposals))
        raise ProposalMergeError(f"Proposal coverage is incomplete: {missing[:10]}")
    ordered = [proposals[row["query_id"]] for row in queries]
    payload = "".join(_canonical(row) + "\n" for row in ordered).encode("utf-8")
    ready = sum(row["specialization_status"] == "ready" for row in ordered)
    manifest = {
        "schema_version": "clean-self-distill-proposal-merge-v1",
        "query_manifest": str(Path(query_manifest).resolve()),
        "query_count": len(queries),
        "shard_count": len(shard_paths),
        "ready_count": ready,
        "ready_rate": ready / len(queries),
        "merged_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return ordered, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    rows, manifest = merge_proposals(args.queries, args.shard)
    _atomic_jsonl(Path(args.output), rows)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
