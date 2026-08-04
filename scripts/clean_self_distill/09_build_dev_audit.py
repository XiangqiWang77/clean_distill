#!/usr/bin/env python3
"""Build a label-free Dev-200 protocol gate and frozen-configuration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from src.clean_self_distill.heldout import load_query_only_manifest
from src.clean_self_distill.io import (
    iter_rows,
    validate_proposal_training_binding,
    validate_specialization_state,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_corrective_candidate(candidate: Mapping[str, Any]) -> bool:
    correct = candidate.get("correct_trajectory")
    wrong = candidate.get("wrong_trajectory")
    frontier = candidate.get("error_frontier")
    return bool(
        isinstance(correct, list)
        and correct
        and isinstance(wrong, list)
        and wrong
        and isinstance(frontier, Mapping)
        and frontier.get("corrective_action")
        and frontier.get("wrong_step_text")
        and frontier.get("verifier_valid") is True
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    query_path = Path(args.queries)
    proposal_path = Path(args.proposals)
    queries = load_query_only_manifest(query_path)
    if len(queries) != 200:
        raise ValueError(f"Dev manifest must contain exactly 200 queries, got {len(queries)}")
    proposals = [dict(row) for row in iter_rows(proposal_path)]
    if len(proposals) != len(queries):
        raise ValueError("Dev proposal coverage is not exactly 200")

    status_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    accepted_counts: list[int] = []
    complete_candidates = total_candidates = 0
    total_unique = total_accepted = total_rejected = 0
    for index, (query, proposal) in enumerate(zip(queries, proposals)):
        if (
            proposal.get("query_id") != query["query_id"]
            or proposal.get("problem") != query["problem"]
            or proposal.get("problem_sha256") != query["problem_sha256"]
            or str(proposal.get("source", "")).casefold() != query["source"]
        ):
            raise ValueError(f"Dev proposal binding/order mismatch at row {index}")
        validate_proposal_training_binding(proposal, context=f"Dev proposal {index}")
        status, _, no_op = validate_specialization_state(
            proposal, context=f"Dev proposal {index}"
        )
        firewall = proposal.get("firewall_audit")
        if not isinstance(firewall, Mapping) or any(
            firewall.get(key) is not False
            for key in ("target_answer_loaded", "target_solution_loaded")
        ):
            raise ValueError(f"Dev proposal {index} violates the clean firewall")
        if no_op != (status != "ready"):
            raise ValueError(f"Dev proposal {index} has inconsistent no-op status")
        status_counts[status] += 1
        candidates = proposal.get("specialization_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"Dev proposal {index} candidates are not a list")
        if status == "ready" and len(candidates) < args.minimum_accepted_candidates:
            raise ValueError(
                f"Dev proposal {index} is ready with fewer than the frozen "
                f"minimum {args.minimum_accepted_candidates} candidates"
            )
        accepted_counts.append(len(candidates))
        total_candidates += len(candidates)
        complete_candidates += sum(
            _complete_corrective_candidate(candidate)
            for candidate in candidates
            if isinstance(candidate, Mapping)
        )
        summary = proposal.get("filter_summary", {})
        if isinstance(summary, Mapping):
            total_unique += int(summary.get("proposed_unique_count", 0))
            total_accepted += int(summary.get("accepted_count", 0))
            total_rejected += int(summary.get("rejected_count", 0))
            reasons = summary.get("rejection_reason_counts", {})
            if isinstance(reasons, Mapping):
                rejection_counts.update({str(key): int(value) for key, value in reasons.items()})

    ready = status_counts.get("ready", 0)
    return {
        "schema_version": "clean-self-distill-dev-audit-v1",
        "split": "deepmath_difficulty_7_10_dev",
        "query_count": len(queries),
        "labels_loaded": False,
        "heldout_labels_loaded": False,
        "query_manifest_sha256": _sha256(query_path),
        "proposal_manifest_sha256": _sha256(proposal_path),
        "proposal_audit": {
            "status_counts": dict(sorted(status_counts.items())),
            "ready_queries": ready,
            "ready_rate": ready / len(queries),
            "accepted_candidate_min": min(accepted_counts),
            "accepted_candidate_max": max(accepted_counts),
            "accepted_candidate_mean": sum(accepted_counts) / len(accepted_counts),
            "correct_wrong_frontier_complete_candidates": complete_candidates,
            "total_accepted_candidates": total_candidates,
            "corrective_completeness_rate": (
                complete_candidates / total_candidates if total_candidates else None
            ),
            "proposed_unique_count": total_unique,
            "filter_accepted_count": total_accepted,
            "filter_rejected_count": total_rejected,
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        },
        "truncation_audit": {
            "proposal_per_call_truncation_rate": None,
            "status": "not_measurable_from_aggregate_proposal_counters",
            "claim": "none",
            "heldout_measurement": (
                "Every raw evaluation row records generated_tokens and truncated; "
                "the offline report consumes the preregistered 32768-token opportunity."
            ),
        },
        "configuration_selection": {
            "status": "preregistered_frozen_configuration_no_dev_sweep",
            "dev_labels_used_for_selection": False,
            "heldout_labels_used_for_selection": False,
            "interpretation": (
                "Dev-200 is a label-free protocol/coverage gate in this PoC. "
                "No claim of dev-tuned hyperparameter optimality is made."
            ),
            "frozen": {
                "ridge_lambda": args.ridge_lambda,
                "residual_step_size": args.residual_step_size,
                "reasoning_token_weight": args.reasoning_token_weight,
                "answer_token_weight": args.answer_token_weight,
                "frontier_positive_weight": args.frontier_positive_weight,
                "frontier_negative_weight": args.frontier_negative_weight,
                "frontier_target_margin": args.frontier_target_margin,
                "max_support_tokens": args.max_support_tokens,
                "max_tokens_per_candidate": args.max_tokens_per_candidate,
                "num_candidates": args.num_candidates,
                "minimum_accepted_candidates": args.minimum_accepted_candidates,
                "learning_rate": args.learning_rate,
                "training_max_sequence_tokens": args.training_max_sequence_tokens,
                "evaluation_max_new_tokens": args.evaluation_max_new_tokens,
                "evaluation_temperature": args.evaluation_temperature,
                "evaluation_top_p": args.evaluation_top_p,
                "evaluation_top_k": args.evaluation_top_k,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ridge-lambda", type=float, required=True)
    parser.add_argument("--residual-step-size", type=float, required=True)
    parser.add_argument("--reasoning-token-weight", type=float, required=True)
    parser.add_argument("--answer-token-weight", type=float, required=True)
    parser.add_argument("--frontier-positive-weight", type=float, required=True)
    parser.add_argument("--frontier-negative-weight", type=float, required=True)
    parser.add_argument("--frontier-target-margin", type=float, required=True)
    parser.add_argument("--max-support-tokens", type=int, required=True)
    parser.add_argument("--max-tokens-per-candidate", type=int, required=True)
    parser.add_argument("--num-candidates", type=int, required=True)
    parser.add_argument("--minimum-accepted-candidates", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--training-max-sequence-tokens", type=int, required=True)
    parser.add_argument("--evaluation-max-new-tokens", type=int, required=True)
    parser.add_argument("--evaluation-temperature", type=float, required=True)
    parser.add_argument("--evaluation-top-p", type=float, required=True)
    parser.add_argument("--evaluation-top-k", type=int, required=True)
    args = parser.parse_args()
    result = build(args)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    print(json.dumps({"status": "passed", "ready_rate": result["proposal_audit"]["ready_rate"]}))


if __name__ == "__main__":
    main()
