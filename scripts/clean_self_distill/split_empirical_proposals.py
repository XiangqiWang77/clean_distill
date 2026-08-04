#!/usr/bin/env python3
"""Split the canonical 1,343-query proposal manifest without changing rows."""

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


class ProposalSplitError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    payload = "".join(_canonical(row) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_proposals(
    proposals_path: str | Path,
    splits: Mapping[str, tuple[str | Path, str | Path]],
) -> dict[str, Any]:
    proposals: dict[str, dict[str, Any]] = {}
    proposal_order: list[str] = []
    for index, raw in enumerate(iter_rows(proposals_path), 1):
        row = dict(raw)
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in proposals:
            raise ProposalSplitError(
                f"Proposal row {index} has a missing/duplicate query_id {query_id!r}"
            )
        validate_specialization_state(row, context=f"Proposal {query_id}")
        validate_proposal_training_binding(row, context=f"Proposal {query_id}")
        proposals[query_id] = row
        proposal_order.append(query_id)

    expected_global_order: list[str] = []
    manifest: dict[str, Any] = {
        "schema_version": "clean-self-distill-proposal-splits-v1",
        "source": str(Path(proposals_path).resolve()),
        "splits": {},
    }
    planned: list[tuple[str, str | Path, str | Path, list[dict[str, Any]]]] = []
    for name, (query_path, output_path) in splits.items():
        queries = load_query_only_manifest(query_path)
        ids = [row["query_id"] for row in queries]
        expected_global_order.extend(ids)
        missing = sorted(set(ids) - set(proposals))
        if missing:
            raise ProposalSplitError(f"Split {name} misses proposals {missing[:5]}")
        rows = [proposals[query_id] for query_id in ids]
        for query, row in zip(queries, rows):
            if (
                row.get("problem") != query["problem"]
                or row.get("problem_sha256") != query["problem_sha256"]
                or str(row.get("source", "")).casefold() != query["source"]
            ):
                raise ProposalSplitError(f"Split {name} binding mismatch for {query['query_id']}")
        planned.append((name, query_path, output_path, rows))
    if proposal_order != expected_global_order:
        raise ProposalSplitError(
            "Merged proposal order/coverage is not exactly distill+dev+heldout"
        )
    for name, query_path, output_path, rows in planned:
        digest = _atomic_jsonl(Path(output_path), rows)
        manifest["splits"][name] = {
            "count": len(rows),
            "query_manifest": str(Path(query_path).resolve()),
            "output": str(Path(output_path).resolve()),
            "sha256": digest,
        }
    manifest["count"] = len(proposal_order)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--distill-queries", required=True)
    parser.add_argument("--dev-queries", required=True)
    parser.add_argument("--heldout-queries", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    manifest = split_proposals(
        args.proposals,
        {
            "distill": (args.distill_queries, output / "distill_proposals.jsonl"),
            "dev": (args.dev_queries, output / "dev_proposals.jsonl"),
            "heldout": (args.heldout_queries, output / "heldout_proposals.jsonl"),
        },
    )
    target = Path(args.manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print(_canonical(manifest))


if __name__ == "__main__":
    main()
