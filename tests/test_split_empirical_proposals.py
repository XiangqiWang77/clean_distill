import hashlib
import json
from pathlib import Path

import pytest

from scripts.clean_self_distill.split_empirical_proposals import (
    ProposalSplitError,
    split_proposals,
)
from src.clean_self_distill.io import compute_proposal_training_sha256


def _query(query_id: str) -> dict:
    problem = f"Problem {query_id}"
    return {
        "query_id": query_id,
        "problem": problem,
        "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        "source": "deepmath",
    }


def _proposal(query: dict) -> dict:
    row = {
        **query,
        "skill_card": {
            "domain": "algebra",
            "skills": ["symbolic reasoning"],
            "reasoning_operators": ["verify"],
            "failure_modes": ["unchecked shortcut"],
            "difficulty": "hard",
            "constraints": [],
            "target_details_removed": True,
        },
        "specialization_candidates": [],
        "specialization_status": "insufficient_verified_candidates",
        "specialization_failure_reason": "none accepted",
        "specialization_no_op": True,
    }
    row["proposal_training_sha256"] = compute_proposal_training_sha256(row)
    return row


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_split_is_exact_and_ordered(tmp_path: Path):
    queries = [_query(f"q{i}") for i in range(4)]
    merged = tmp_path / "merged.jsonl"
    _write(merged, [_proposal(row) for row in queries])
    manifests = []
    outputs = []
    ranges = {"distill": (0, 2), "dev": (2, 3), "heldout": (3, 4)}
    for name in ("distill", "dev", "heldout"):
        start, stop = ranges[name]
        subset = queries[start:stop]
        query_path = tmp_path / f"{name}-q.jsonl"
        output_path = tmp_path / f"{name}-p.jsonl"
        _write(query_path, subset)
        manifests.append((name, (query_path, output_path)))
        outputs.append(output_path)
    manifest = split_proposals(merged, dict(manifests))
    assert manifest["count"] == 4
    assert [json.loads(line)["query_id"] for line in outputs[0].read_text().splitlines()] == ["q0", "q1"]


def test_split_rejects_noncanonical_merged_order(tmp_path: Path):
    queries = [_query("q0"), _query("q1")]
    merged = tmp_path / "merged.jsonl"
    query_path = tmp_path / "queries.jsonl"
    _write(merged, [_proposal(queries[1]), _proposal(queries[0])])
    _write(query_path, queries)
    with pytest.raises(ProposalSplitError, match="order/coverage"):
        split_proposals(merged, {"all": (query_path, tmp_path / "out.jsonl")})
