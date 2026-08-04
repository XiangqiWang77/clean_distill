import json
from pathlib import Path

import pytest

from src.clean_self_distill.heldout import (
    HeldoutProtocolError,
    expected_prediction_keys,
    load_query_only_manifest,
    paired_sample_seed,
    score_prediction_rows,
    validate_query_only_row,
)


def query(query_id: str, problem: str, source: str = "amc23") -> dict:
    import hashlib

    return {
        "query_id": query_id,
        "problem": problem,
        "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        "source": source,
    }


def test_query_manifest_physically_rejects_labels(tmp_path: Path):
    row = {**query("q", "Compute 1+1."), "answer": "2"}
    with pytest.raises(HeldoutProtocolError, match="physically exposes"):
        validate_query_only_row(row, context="row")

    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(HeldoutProtocolError):
        load_query_only_manifest(path)


def test_expected_keys_preserve_global_shard_order():
    rows = [query(f"q{i}", f"Problem {i}") for i in range(3)]
    assert expected_prediction_keys(rows, num_shards=2, shard_index=1) == [
        ("q1", 0),
        ("q1", 1),
        ("q1", 2),
        ("q1", 3),
    ]
    assert paired_sample_seed(7, 2, 3) == 2028


def test_offline_scoring_emits_acc1_and_mean4_without_answer(tmp_path: Path):
    item = query("q", "Compute 1+1.")
    predictions = []
    for sample in range(4):
        predictions.append(
            {
                **item,
                "method": "clean_sd",
                "checkpoint_episode": 250,
                "checkpoint_sha256": "a" * 64,
                "sample_index": sample,
                "seed": sample,
                "response": "\\boxed{2}" if sample < 3 else "\\boxed{3}",
                "generated_tokens": 5,
                "truncated": False,
                "training_audit": {},
            }
        )
    rows = score_prediction_rows(
        predictions,
        {"q": {"answer": "2", "problem_sha256": item["problem_sha256"]}},
    )
    assert len(rows) == 5
    assert [row["correct"] for row in rows if row["profile"] == "mean4"] == [
        1.0,
        1.0,
        1.0,
        0.0,
    ]
    assert [row["correct"] for row in rows if row["profile"] == "acc1"] == [1.0]
    assert all("answer" not in row for row in rows)


def test_scoring_fails_closed_on_missing_sample():
    item = query("q", "Compute 1+1.")
    predictions = [
        {**item, "sample_index": sample, "response": "\\boxed{2}"}
        for sample in range(3)
    ]
    with pytest.raises(HeldoutProtocolError, match="exactly sample indices"):
        score_prediction_rows(
            predictions,
            {"q": {"answer": "2", "problem_sha256": item["problem_sha256"]}},
        )
