"""Lightweight contracts for the offline mechanism/ablation join."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# The executable has a numeric filename, so import it without relying on an
# invalid Python identifier.
import importlib.util


_SPEC = importlib.util.spec_from_file_location(
    "build_empirical_aux",
    Path(__file__).parents[1] / "scripts/clean_self_distill/08_build_empirical_aux.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_AUX = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUX)


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def _trajectory() -> dict:
    return {
        "token_count": 2,
        "student_logprob_sum": -3.0,
        "teacher_logprob_sum": -2.0,
        "partition_version": "rlcsd-style-task-v1",
        "style_abs_error_sum": 1.0,
        "style_token_count": 1,
        "task_abs_error_sum": 1.0,
        "task_token_count": 1,
    }


def _audit(teacher_type: str) -> dict:
    return {
        "teacher_positions": 2,
        "hindsight_exposed_positions": (
            2 if teacher_type == "post_outcome_privilege" else 0
        ),
        "compared_positions": 2,
        "exact_context_positions": 2 if teacher_type == "clean_teacher" else 0,
    }


def test_mechanism_uses_all_queries_while_ablation_uses_matched_subset(tmp_path: Path):
    problems = {"q0": "Compute 1+1.", "q1": "Compute 2+2."}
    hashes = {
        query_id: hashlib.sha256(problem.encode()).hexdigest()
        for query_id, problem in problems.items()
    }
    labels = _write(
        tmp_path / "labels.jsonl",
        [
            {"query_id": query_id, "problem_sha256": hashes[query_id], "answer": answer}
            for query_id, answer in (("q0", "2"), ("q1", "4"))
        ],
    )
    scored: dict[str, list[dict]] = {
        "base": [],
        "csd_t": [],
        "csd_t_correct_only": [],
    }
    for query_index, query_id in enumerate(problems):
        ready = query_id == "q0"
        for sample in range(4):
            common = {
                "profile": "mean4",
                "query_id": query_id,
                "problem_sha256": hashes[query_id],
                "source": "amc23",
                "sample_index": sample,
                "seed": query_index * 1009 + sample,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_new_tokens": 32768,
                "correct": int(query_id == "q0" and sample == 0),
                "response": "\\boxed{2}" if query_id == "q0" else "\\boxed{0}",
            }
            scored["base"].append({**common, "method": "base"})
            metrics = {
                "specialization_no_op": not ready,
                "candidate_count": 4 if ready else 0,
                "support_tokens": 8 if ready else 0,
                "adapter_rank": 8 if ready else 0,
                "frontier_margins": (
                    [
                        {
                            "frontier_id": "f0",
                            "base_margin": -1.0,
                            "teacher_margin": 1.0,
                        }
                    ]
                    if ready
                    else []
                ),
                "proposal_base_target_nll": 1.0,
                "proposal_fit_signed_target_logit_gain": 0.8,
                "specialization_seconds": 0.5,
            }
            for method in ("csd_t", "csd_t_correct_only"):
                scored[method].append(
                    {
                        **common,
                        "method": method,
                        "specialization_metrics": metrics,
                        "trajectory_metrics": _trajectory(),
                        "training_audit": _audit("clean_teacher"),
                        "behavioral_diagnostics": {
                            "fabricated_reference_hallucination": False,
                            "hedging_token_count": 0,
                            "response_tokens": 2,
                            "mean_entropy": 0.5,
                            "truncated": False,
                        },
                        "proposal_end_to_end_seconds": 0.25,
                    }
                )

    mechanism_paths: dict[str, str] = {}
    for teacher_type in ("pre_decision_privilege", "post_outcome_privilege"):
        rows = []
        for query_id in problems:
            rows.append(
                {
                    "schema_version": "clean-self-distill-mechanism-trajectory-v1",
                    "record_type": "trajectory",
                    "teacher_type": teacher_type,
                    "query_id": query_id,
                    "problem_sha256": hashes[query_id],
                    "response": "\\boxed{2}" if query_id == "q0" else "\\boxed{0}",
                    "trajectory_metrics": _trajectory(),
                    "training_audit": _audit(teacher_type),
                    "behavioral_diagnostics": {
                        "fabricated_reference_hallucination": False,
                        "hedging_token_count": 0,
                        "response_tokens": 2,
                        "mean_entropy": 0.5,
                        "truncated": False,
                    },
                }
            )
        mechanism_paths[teacher_type] = _write(tmp_path / f"{teacher_type}.jsonl", rows)

    mechanism, ablation, audit = _AUX.build(
        labels_path=labels,
        base_paths=[_write(tmp_path / "base.jsonl", scored["base"])],
        signed_paths=[_write(tmp_path / "signed.jsonl", scored["csd_t"])],
        correct_only_paths=[
            _write(tmp_path / "correct.jsonl", scored["csd_t_correct_only"])
        ],
        pre_paths=[mechanism_paths["pre_decision_privilege"]],
        post_paths=[mechanism_paths["post_outcome_privilege"]],
    )
    trajectories = [row for row in mechanism if row["record_type"] == "trajectory"]
    assert len(trajectories) == 6
    assert {row["query_id"] for row in trajectories} == {"q0", "q1"}
    assert len(ablation) == 2
    assert {row["query_id"] for row in ablation} == {"q0"}
    assert {row["pre_update_support_target_nll"] for row in ablation} == {1.0}
    assert {row["support_objective_logit_gain"] for row in ablation} == {0.8}
    assert audit["mechanism_queries"] == 2
    assert audit["runtime_matched_ready_queries"] == 1
