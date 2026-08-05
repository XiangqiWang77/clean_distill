import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from src.clean_self_distill.io import compute_proposal_training_sha256


dev_audit = importlib.import_module("scripts.clean_self_distill.09_build_dev_audit")


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _query(index: int) -> dict:
    problem = f"Independent DeepMath development problem {index}"
    return {
        "query_id": f"deepmath:{index:03d}",
        "problem": problem,
        "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        "source": "deepmath",
    }


def _proposal(query: dict, *, ready: bool) -> dict:
    candidates = []
    if ready:
        candidates = [
            {
                "candidate_id": "c00",
                "problem": "A target-disjoint exercise",
                "solution": "Use the verified step.",
                "final_answer": "1",
                "correct_trajectory": [{"step_index": 0, "text": "Verified step"}],
                "wrong_trajectory": [{"step_index": 0, "text": "Invalid shortcut"}],
                "wrong_final_answer": "0",
                "error_frontier": {
                    "wrong_step_index": 0,
                    "wrong_step_text": "Invalid shortcut",
                    "error_explanation": "The shortcut is invalid.",
                    "corrective_action": "Use the verified step.",
                    "verifier_valid": True,
                },
            }
        ]
    row = {
        **query,
        "skill_card": {
            "domain": "algebra",
            "skills": ["reasoning"],
            "reasoning_operators": ["check"],
            "difficulty": "hard",
            "constraints": [],
            "target_details_removed": True,
        },
        "specialization_candidates": candidates,
        "specialization_status": "ready" if ready else "insufficient_verified_candidates",
        "specialization_failure_reason": "" if ready else "not enough verified candidates",
        "specialization_no_op": not ready,
        "firewall_audit": {
            "target_answer_loaded": False,
            "target_solution_loaded": False,
        },
        "filter_summary": {
            "proposed_unique_count": 1,
            "accepted_count": int(ready),
            "rejected_count": int(not ready),
            "rejection_reason_counts": {} if ready else {"verification_failed": 1},
        },
    }
    row["proposal_training_sha256"] = compute_proposal_training_sha256(row)
    return row


def test_dev_audit_is_label_free_frozen_and_counts_corrective_frontier(tmp_path: Path):
    queries = [_query(index) for index in range(200)]
    proposals = [
        _proposal(query, ready=index == 0) for index, query in enumerate(queries)
    ]
    query_path = tmp_path / "dev_queries.jsonl"
    proposal_path = tmp_path / "dev_proposals.jsonl"
    _write(query_path, queries)
    _write(proposal_path, proposals)
    args = SimpleNamespace(
        queries=str(query_path),
        proposals=str(proposal_path),
        ridge_lambda=0.1,
        residual_step_size=0.8,
        reasoning_token_weight=0.25,
        answer_token_weight=1.0,
        frontier_positive_weight=8.0,
        frontier_negative_weight=8.0,
        frontier_target_margin=1.0,
        max_support_tokens=768,
        max_tokens_per_candidate=96,
        num_candidates=8,
        minimum_accepted_candidates=1,
        learning_rate=2e-5,
        training_max_sequence_tokens=16384,
        distill_token_chunk_size=128,
        evaluation_max_new_tokens=32768,
        evaluation_temperature=0.6,
        evaluation_top_p=0.95,
        evaluation_top_k=20,
    )

    result = dev_audit.build(args)

    assert result["query_count"] == 200
    assert result["labels_loaded"] is False
    assert result["heldout_labels_loaded"] is False
    assert result["proposal_audit"]["ready_queries"] == 1
    assert result["proposal_audit"]["correct_wrong_frontier_complete_candidates"] == 1
    assert result["proposal_audit"]["corrective_completeness_rate"] == 1.0
    selection = result["configuration_selection"]
    assert selection["status"] == "preregistered_frozen_configuration_no_dev_sweep"
    assert selection["dev_labels_used_for_selection"] is False
    assert selection["heldout_labels_used_for_selection"] is False
    assert selection["frozen"]["frontier_positive_weight"] == 8.0
    assert selection["frozen"]["frontier_target_margin"] == 1.0
    assert selection["frozen"]["distill_token_chunk_size"] == 128
    assert selection["frozen"]["gradient_checkpointing"] is True
    assert result["truncation_audit"]["status"].startswith("not_measurable")
    assert result["truncation_audit"]["claim"] == "none"
