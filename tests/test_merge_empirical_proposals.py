import hashlib
import json
from pathlib import Path

import pytest

from scripts.clean_self_distill.merge_empirical_proposals import (
    ProposalMergeError,
    merge_proposals,
)
from src.clean_self_distill.io import compute_proposal_training_sha256


def query(query_id: str) -> dict:
    problem = f"Problem {query_id}"
    return {
        "query_id": query_id,
        "problem": problem,
        "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        "source": "deepmath",
    }


def proposal(item: dict) -> dict:
    row = {
        **item,
        "skill_card": {},
        "specialization_candidates": [],
        "specialization_status": "insufficient_verified_candidates",
        "specialization_failure_reason": "none accepted",
        "specialization_no_op": True,
    }
    row["proposal_training_sha256"] = compute_proposal_training_sha256(row)
    return row


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_merge_restores_query_order(tmp_path: Path):
    items = [query(f"q{i}") for i in range(3)]
    query_path = tmp_path / "queries.jsonl"
    write(query_path, items)
    shard0 = tmp_path / "s0.jsonl"
    shard1 = tmp_path / "s1.jsonl"
    write(shard0, [proposal(items[0]), proposal(items[2])])
    write(shard1, [proposal(items[1])])
    rows, manifest = merge_proposals(query_path, [shard0, shard1])
    assert [row["query_id"] for row in rows] == ["q0", "q1", "q2"]
    assert manifest["query_count"] == 3
    assert manifest["ready_rate"] == 0.0


def test_merge_rejects_duplicate_and_missing(tmp_path: Path):
    item = query("q")
    query_path = tmp_path / "queries.jsonl"
    shard0 = tmp_path / "s0.jsonl"
    shard1 = tmp_path / "s1.jsonl"
    write(query_path, [item])
    write(shard0, [proposal(item)])
    write(shard1, [proposal(item)])
    with pytest.raises(ProposalMergeError, match="Duplicate"):
        merge_proposals(query_path, [shard0, shard1])
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ProposalMergeError, match="empty"):
        merge_proposals(query_path, [empty])


def test_merge_can_rebind_recanonicalized_ids_by_problem_hash(tmp_path: Path):
    items = [query("heldout-q0"), query("heldout-q1")]
    query_path = tmp_path / "queries.jsonl"
    write(query_path, items)
    shard = tmp_path / "recanonicalized.jsonl"
    recanonicalized = []
    for index, item in enumerate(items):
        changed = proposal({**item, "query_id": f"generated-q{index}"})
        recanonicalized.append(changed)
    write(shard, recanonicalized)

    rows, manifest = merge_proposals(
        query_path, [shard], bind_by_problem_sha256=True
    )

    assert [row["query_id"] for row in rows] == ["heldout-q0", "heldout-q1"]
    assert manifest["identity_binding"] == "problem_sha256"
    for row, expected in zip(rows, items, strict=True):
        assert {key: row[key] for key in expected} == expected
        assert row["proposal_training_sha256"] == compute_proposal_training_sha256(
            row
        )
