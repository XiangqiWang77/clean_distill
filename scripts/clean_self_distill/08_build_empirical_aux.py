#!/usr/bin/env python3
"""Build scored mechanism and strictly matched ablation artifacts offline."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.clean_self_distill.heldout import load_sealed_labels
from src.clean_self_distill.io import iter_rows
from src.opsd_format import extract_boxed_answer, grade_boxed_answer


_REFERENCE_RE = re.compile(
    r"\b(?:answer\s*key|given\s+(?:answer|solution)|reference\s+(?:answer|solution)|"
    r"according\s+to\s+the\s+(?:answer|reference))\b",
    flags=re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(?:perhaps|maybe|possibly|likely|seems?|appears?|I\s+think|not\s+sure)\b",
    flags=re.IGNORECASE,
)


class AuxiliaryArtifactError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load(paths: Sequence[str]) -> list[dict[str, Any]]:
    return [dict(row) for path in paths for row in iter_rows(path)]


def _scored_index(paths: Sequence[str], method: str) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _load(paths):
        if row.get("profile") != "mean4" or row.get("method") != method:
            continue
        key = (str(row.get("query_id", "")), int(row.get("sample_index", -1)))
        if key in result:
            raise AuxiliaryArtifactError(f"Duplicate scored key {method}/{key}")
        result[key] = row
    if not result:
        raise AuxiliaryArtifactError(f"No scored rows for method {method}")
    return result


def _sample0(index: Mapping[tuple[str, int], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {query_id: row for (query_id, sample), row in index.items() if sample == 0}


def _trajectory_record(
    *,
    teacher_type: str,
    query_id: str,
    reward: int,
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    required = (
        "token_count",
        "student_logprob_sum",
        "teacher_logprob_sum",
        "partition_version",
        "style_abs_error_sum",
        "style_token_count",
        "task_abs_error_sum",
        "task_token_count",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise AuxiliaryArtifactError(
            f"Trajectory {teacher_type}/{query_id} misses {missing}"
        )
    audit_keys = (
        "teacher_positions",
        "hindsight_exposed_positions",
        "compared_positions",
        "exact_context_positions",
    )
    missing_audit = [key for key in audit_keys if key not in audit]
    if missing_audit:
        raise AuxiliaryArtifactError(
            f"Trajectory {teacher_type}/{query_id} misses audit {missing_audit}"
        )
    return {
        "record_type": "trajectory",
        "teacher_type": teacher_type,
        "query_id": query_id,
        "trajectory_id": f"{query_id}:sample0",
        "reward": int(reward),
        **{key: metrics[key] for key in required},
        "training_audit": {key: int(audit[key]) for key in audit_keys},
        "behavioral_diagnostics": dict(diagnostics),
    }


def _behavior(response: str, row: Mapping[str, Any]) -> dict[str, Any]:
    carried = row.get("behavioral_diagnostics")
    if isinstance(carried, Mapping):
        return dict(carried)
    return {
        "fabricated_reference_hallucination": bool(_REFERENCE_RE.search(response)),
        "hedging_token_count": len(_HEDGE_RE.findall(response)),
        "response_tokens": int(row.get("generated_tokens", 0)),
        "mean_entropy": None,
        "truncated": bool(row.get("truncated", False)),
    }


def _metric_identity(metrics: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(metrics.get("candidate_count", 0)),
        int(metrics.get("support_tokens", 0)),
        int(metrics.get("adapter_rank", 0)),
    )


def _decode_identity(row: Mapping[str, Any]) -> tuple[float, float, int, int]:
    return (
        float(row.get("temperature", -1.0)),
        float(row.get("top_p", -1.0)),
        int(row.get("top_k", -1)),
        int(row.get("max_new_tokens", -1)),
    )


def _frontier_identity(metrics: Mapping[str, Any]) -> tuple[tuple[str, int, float], ...]:
    values = metrics.get("frontier_margins", [])
    if not isinstance(values, list):
        return ()
    return tuple(
        (
            str(value.get("frontier_id", "")),
            int(value.get("candidate_index", -1)),
            float(value.get("base_margin", float("nan"))),
        )
        for value in values
        if isinstance(value, Mapping)
    )


def _mechanism_index(
    paths: Sequence[str],
    *,
    teacher_type: str,
    labels: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _load(paths):
        query_id = str(row.get("query_id", ""))
        if query_id not in labels or query_id in result:
            raise AuxiliaryArtifactError(
                f"Unexpected/duplicate {teacher_type} mechanism row {query_id!r}"
            )
        if (
            row.get("schema_version")
            != "clean-self-distill-mechanism-trajectory-v1"
            or row.get("record_type") != "trajectory"
            or row.get("teacher_type") != teacher_type
            or row.get("problem_sha256") != labels[query_id]["problem_sha256"]
        ):
            raise AuxiliaryArtifactError(
                f"Mechanism binding/schema mismatch for {teacher_type}/{query_id}"
            )
        if "correct" in row:
            raise AuxiliaryArtifactError(
                f"Mechanism generation row {teacher_type}/{query_id} was pre-scored"
            )
        result[query_id] = row
    if set(result) != set(labels):
        raise AuxiliaryArtifactError(
            f"{teacher_type} mechanism trajectory coverage is incomplete"
        )
    return result


def build(
    *,
    labels_path: str,
    base_paths: Sequence[str],
    signed_paths: Sequence[str],
    correct_only_paths: Sequence[str],
    pre_paths: Sequence[str],
    post_paths: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    labels = load_sealed_labels(labels_path)
    base = _scored_index(base_paths, "base")
    signed = _scored_index(signed_paths, "csd_t")
    correct_only = _scored_index(correct_only_paths, "csd_t_correct_only")
    expected = {(query_id, sample) for query_id in labels for sample in range(4)}
    for name, index in (("base", base), ("signed", signed), ("correct_only", correct_only)):
        if set(index) != expected:
            raise AuxiliaryArtifactError(f"{name} scored coverage is incomplete")
    for query_id in labels:
        for sample in range(4):
            base_row = base[(query_id, sample)]
            for name, index in (("signed", signed), ("correct_only", correct_only)):
                method_row = index[(query_id, sample)]
                if int(method_row.get("seed", -1)) != int(base_row.get("seed", -2)):
                    raise AuxiliaryArtifactError(
                        f"Unpaired seed for {name}/{query_id}/{sample}"
                    )
                if _decode_identity(method_row) != _decode_identity(base_row):
                    raise AuxiliaryArtifactError(
                        f"Unmatched decoding for {name}/{query_id}/{sample}"
                    )
        if _decode_identity(base[(query_id, 0)]) != (0.6, 0.95, 20, 32768):
            raise AuxiliaryArtifactError(
                f"{query_id} does not use the preregistered evaluation decoding"
            )

    signed0 = _sample0(signed)
    correct0 = _sample0(correct_only)
    matched: list[str] = []
    exclusions: dict[str, int] = {}
    for query_id in labels:
        left = correct0[query_id].get("specialization_metrics")
        right = signed0[query_id].get("specialization_metrics")
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            exclusions["missing_metrics"] = exclusions.get("missing_metrics", 0) + 1
            continue
        if bool(left.get("specialization_no_op")) or bool(right.get("specialization_no_op")):
            exclusions["specialization_no_op"] = exclusions.get("specialization_no_op", 0) + 1
            continue
        if _metric_identity(left) != _metric_identity(right):
            exclusions["unmatched_actual_compute"] = exclusions.get("unmatched_actual_compute", 0) + 1
            continue
        if min(_metric_identity(left)) <= 0:
            exclusions["empty_runtime_identity"] = exclusions.get("empty_runtime_identity", 0) + 1
            continue
        if not left.get("frontier_margins") or not right.get("frontier_margins"):
            exclusions["no_comparable_frontier"] = exclusions.get("no_comparable_frontier", 0) + 1
            continue
        if _frontier_identity(left) != _frontier_identity(right):
            exclusions["unmatched_frontier_identity"] = exclusions.get(
                "unmatched_frontier_identity", 0
            ) + 1
            continue
        matched.append(query_id)
    if not matched:
        raise AuxiliaryArtifactError("No runtime-matched ready ablation query")

    ablation: list[dict[str, Any]] = []
    for query_id in matched:
        seeds = [int(base[(query_id, sample)]["seed"]) for sample in range(4)]
        for variant, index in (
            ("correct_only", correct_only),
            ("correct_wrong_signed", signed),
        ):
            row0 = index[(query_id, 0)]
            metrics = row0["specialization_metrics"]
            for sample in range(1, 4):
                sample_metrics = index[(query_id, sample)].get("specialization_metrics")
                if not isinstance(sample_metrics, Mapping) or _metric_identity(
                    sample_metrics
                ) != _metric_identity(metrics):
                    raise AuxiliaryArtifactError(
                        f"Specialization identity changes across samples for {variant}/{query_id}"
                    )
            candidate_count, support_tokens, ridge_dimension = _metric_identity(metrics)
            trajectories = []
            target_samples = []
            for sample in range(4):
                base_row = base[(query_id, sample)]
                method_row = index[(query_id, sample)]
                target_samples.append(
                    {
                        "sample_index": sample,
                        "base_correct": int(base_row["correct"]),
                        "method_correct": int(method_row["correct"]),
                    }
                )
                trajectory = method_row.get("trajectory_metrics")
                if not isinstance(trajectory, Mapping):
                    raise AuxiliaryArtifactError(
                        f"Missing trajectory metrics for {variant}/{query_id}/{sample}"
                    )
                trajectories.append(
                    {
                        "reward": int(method_row["correct"]),
                        **dict(trajectory),
                    }
                )
            frontiers = [dict(value) for value in metrics["frontier_margins"]]
            proposal_seconds = float(row0.get("proposal_end_to_end_seconds", 0.0))
            adaptation_seconds = proposal_seconds + float(
                metrics.get("specialization_seconds", 0.0)
            )
            ablation.append(
                {
                    "query_id": query_id,
                    "variant": variant,
                    "heldout_query_denominator": len(labels),
                    "runtime_subset_exclusions": dict(exclusions),
                    "candidate_count": candidate_count,
                    "actual_support_tokens": support_tokens,
                    "ridge_dimension": ridge_dimension,
                    # The ridge fitter currently records the exact pre-update
                    # target NLL and an objective-aligned selected-vocabulary
                    # logit gain.  Do not mislabel the former as adapted NLL.
                    "pre_update_support_target_nll": float(
                        metrics["proposal_base_target_nll"]
                    ),
                    "support_objective_logit_gain": float(
                        metrics["proposal_fit_signed_target_logit_gain"]
                    ),
                    "adaptation_seconds": adaptation_seconds,
                    "decode_config": {
                        "temperature": float(row0["temperature"]),
                        "top_p": float(row0["top_p"]),
                        "top_k": int(row0["top_k"]),
                        "max_new_tokens": int(row0["max_new_tokens"]),
                        "num_samples": 4,
                    },
                    "seeds": seeds,
                    "target_samples": target_samples,
                    "trajectories": trajectories,
                    "frontiers": frontiers,
                }
            )

    pre = _mechanism_index(
        pre_paths, teacher_type="pre_decision_privilege", labels=labels
    )
    post = _mechanism_index(
        post_paths, teacher_type="post_outcome_privilege", labels=labels
    )
    mechanism: list[dict[str, Any]] = []
    for query_id in labels:
        for teacher_type, source in (
            ("pre_decision_privilege", pre[query_id]),
            ("post_outcome_privilege", post[query_id]),
        ):
            if source.get("teacher_type") != teacher_type:
                raise AuxiliaryArtifactError(f"Mechanism type mismatch for {query_id}")
            response = str(source.get("response", ""))
            reward = int(
                grade_boxed_answer(
                    extract_boxed_answer(response), labels[query_id]["answer"]
                )
            )
            metrics = source.get("trajectory_metrics")
            if not isinstance(metrics, Mapping):
                raise AuxiliaryArtifactError(f"Missing mechanism metrics for {query_id}")
            mechanism.append(
                _trajectory_record(
                    teacher_type=teacher_type,
                    query_id=query_id,
                    reward=reward,
                    metrics=metrics,
                    diagnostics=_behavior(response, source),
                    audit=source.get("training_audit", {}),
                )
            )
        clean = signed0[query_id]
        clean_metrics = clean.get("trajectory_metrics")
        if not isinstance(clean_metrics, Mapping):
            raise AuxiliaryArtifactError(f"Missing clean trajectory for {query_id}")
        mechanism.append(
            _trajectory_record(
                teacher_type="clean_teacher",
                query_id=query_id,
                reward=int(clean["correct"]),
                metrics=clean_metrics,
                diagnostics=_behavior(str(clean.get("response", "")), clean),
                audit=clean.get("training_audit", {}),
            )
        )
        for frontier in signed0[query_id]["specialization_metrics"]["frontier_margins"]:
            mechanism.append(
                {
                    "record_type": "frontier",
                    "teacher_type": "clean_teacher",
                    **dict(frontier),
                    "query_id": query_id,
                }
            )

    audit = {
        "schema_version": "clean-self-distill-empirical-aux-v1",
        "heldout_queries": len(labels),
        "mechanism_queries": len(labels),
        "runtime_matched_ready_queries": len(matched),
        "runtime_subset_rule": (
            "both variants ready, identical candidate/support/ridge counts, and comparable frontier"
        ),
        "exclusions": exclusions,
    }
    return mechanism, ablation, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--base-scored", action="append", required=True)
    parser.add_argument("--signed-scored", action="append", required=True)
    parser.add_argument("--correct-only-scored", action="append", required=True)
    parser.add_argument("--pre-decision", action="append", required=True)
    parser.add_argument("--post-outcome", action="append", required=True)
    parser.add_argument("--mechanism-output", required=True)
    parser.add_argument("--ablation-output", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()
    mechanism, ablation, audit = build(
        labels_path=args.labels,
        base_paths=args.base_scored,
        signed_paths=args.signed_scored,
        correct_only_paths=args.correct_only_scored,
        pre_paths=args.pre_decision,
        post_paths=args.post_outcome,
    )
    _atomic_jsonl(Path(args.mechanism_output), mechanism)
    _atomic_jsonl(Path(args.ablation_output), ablation)
    target = Path(args.audit_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


if __name__ == "__main__":
    main()
