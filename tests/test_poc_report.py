"""Contract tests for strict four-condition PoC aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.clean_self_distill.report_poc import (
    ReportValidationError,
    _validate_allocation_provenance,
    generate_report,
)
from src.clean_self_distill.io import (
    canonical_json_sha256,
    compute_proposal_training_sha256,
    load_query_records,
)
from src.clean_self_distill.privileged import build_privileged_control_artifact
from src.clean_self_distill.propose import (
    skill_card_disjoint_audit,
    target_disjoint_audit,
)


EXPECTED_SMOKE_COUNTS = {"amc23": 1, "aime24": 1, "aime25": 1}
RUNTIME = {
    "timestamp_utc": "2026-08-03T18:00:00+00:00",
    "git_commit": "a" * 40,
    "git_dirty": False,
    "model": "Qwen/Qwen3-8B",
    "resolved_model_revision": "b" * 40,
    "hostname": "b200-node",
    "python_executable": "/home/da839/.conda/envs/TTT/bin/python",
    "torch_overlay": "/home/da839/scratch_pi_mg269/da839/mfspd/pydeps-cu128",
    "torch_module_path": "/home/da839/scratch_pi_mg269/da839/mfspd/pydeps-cu128/torch/__init__.py",
    "torch_arch_flags": ["sm_80", "sm_90", "sm_100"],
    "slurm_array_job_id": "123456",
    "slurm_array_task_id": "0",
    "cuda_visible_devices": "0",
    "conda_prefix": "/home/da839/.conda/envs/TTT",
    "torch": "2.9.1+cu128",
    "cuda_runtime": "12.8",
    "cuda_available": True,
    "gpu_count": 1,
    "gpus": [{"index": 0, "name": "NVIDIA B200", "capability": [10, 0]}],
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(
    root: Path,
) -> tuple[
    Path,
    list[Path],
    list[Path],
    list[Path],
    list[dict],
    list[dict],
    list[dict],
]:
    dataset_path = root / "dataset.jsonl"
    dataset_rows = [
        {
            "id": "amc-1",
            "source": "amc23",
            "problem": "AMC smoke problem.",
            "answer": "101",
        },
        {
            "id": "aime24-1",
            "source": "aime24",
            "problem": "AIME 2024 smoke problem.",
            "answer": "202",
        },
        {
            "id": "aime25-1",
            "source": "aime25",
            "problem": "AIME 2025 smoke problem.",
            "answer": "303",
        },
    ]
    _write_jsonl(dataset_path, dataset_rows)
    canonical = load_query_records(dataset_path, include_targets=True)

    proposal_rows: list[dict] = []
    task1_rows: list[dict] = []
    task2_rows: list[dict] = []
    ridge_config = {
        "ridge_lambda": 0.1,
        "residual_step_size": 0.8,
        "max_tokens_per_candidate": 64,
        "max_support_tokens": 256,
        "hard_negatives": 8,
        "max_length": 8192,
        "num_specialization_candidates": None,
        "reasoning_token_weight": 0.25,
        "answer_token_weight": 1.0,
        "frontier_positive_weight": 8.0,
        "frontier_negative_weight": 8.0,
        "frontier_max_tokens": 24,
        "frontier_negative_probability_floor": 0.25,
        "max_update_norm": 2.0,
    }
    common_run_config = {
        "seed": 0,
        "eval_max_new_tokens": 8192,
        "eval_samples": 1,
        "num_shards": 1,
        "model": "Qwen/Qwen3-8B",
        "revision": "b" * 40,
        **ridge_config,
    }
    task1_run_config = {**common_run_config, "mode": "task1"}
    task2_run_config = {**common_run_config, "mode": "task2"}
    for index, record in enumerate(canonical):
        digest = hashlib.sha256(record["problem"].encode("utf-8")).hexdigest()
        is_amc = record["source"] == "amc23"
        answer = record["answer"]
        wrong_answer = "999"
        base_response = rf"Reasoning. \boxed{{{answer if is_amc else wrong_answer}}}"
        teacher_response = rf"Reasoning. \boxed{{{answer}}}"
        distilled_response = rf"Reasoning. \boxed{{{answer if record['source'] != 'aime25' else wrong_answer}}}"
        base_parsed_answer = answer if is_amc else wrong_answer
        distilled_parsed_answer = (
            answer if record["source"] != "aime25" else wrong_answer
        )
        skill_card = {
            "domain": "abstract algebra",
            "skills": ["symbolic manipulation"],
            "reasoning_operators": ["combine constraints"],
            "failure_modes": ["combining terms with the wrong operation"],
            "difficulty": "competition",
            "constraints": [],
            "target_details_removed": True,
        }
        candidate = {
            "candidate_id": "c00",
            "problem": "Compute a product of two symbolic quantities.",
            "solution": "Multiply the quantities.",
            "final_answer": "z",
            "correct_trajectory": [
                {"step_index": 0, "text": "Multiply the quantities."}
            ],
            "wrong_trajectory": [{"step_index": 0, "text": "Add the quantities."}],
            "wrong_final_answer": "w",
            "error_frontier": {
                "wrong_step_index": 0,
                "wrong_step_text": "Add the quantities.",
                "error_explanation": "Addition does not compute a product.",
                "corrective_action": "Multiply the quantities instead.",
                "verifier_valid": True,
            },
            "verifier_valid": True,
            "verifier_accepted": True,
            "frontier_verifier_valid": True,
        }
        candidate["target_disjoint_audit"] = target_disjoint_audit(
            record["problem"], candidate["problem"]
        )
        candidate["artifact_target_disjoint_audits"] = {}
        for artifact_name, artifact_text in {
            "correct_trajectory": "Multiply the quantities.",
            "wrong_trajectory": "Add the quantities.",
            "error_frontier": (
                "Add the quantities. Addition does not compute a product. "
                "Multiply the quantities instead."
            ),
        }.items():
            artifact_audit = target_disjoint_audit(record["problem"], artifact_text)
            artifact_audit.update(
                {
                    "safe": True,
                    "source_isolated_from_target": True,
                    "lexical_overlap_is_informational_only": True,
                }
            )
            candidate["artifact_target_disjoint_audits"][artifact_name] = artifact_audit
        candidate["generation_provenance"] = {
            "correct_trajectory": {
                "source": "independent_solver",
                "attempt_count": 1,
                "raw_response_sha256": "1" * 64,
                "message_sha256": "2" * 64,
            },
            "wrong_trajectory": {
                "source": "independent_failure_conditioned_model_generation",
                "sources": [
                    "candidate_problem",
                    "sanitized_skill_card_failure_modes",
                ],
                "correct_trajectory_exposed": False,
                "attempt_count": 1,
                "raw_response_sha256": "3" * 64,
                "message_sha256": "4" * 64,
            },
            "correct_trajectory_verifier": {
                "source": "independent_verifier",
                "sources": ["candidate_problem", "candidate_correct_trajectory"],
                "attempt_count": 1,
                "raw_response_sha256": "5" * 64,
                "message_sha256": "6" * 64,
            },
            "error_frontier": {
                "source": "independent_verifier",
                "sources": [
                    "candidate_problem",
                    "verified_correct_trajectory",
                    "model_wrong_trajectory",
                ],
                "attempt_count": 1,
                "raw_response_sha256": "7" * 64,
                "message_sha256": "8" * 64,
            },
        }
        proposal_row = {
            "schema_version": "clean-self-distill-proposals-v5",
            "query_id": record["query_id"],
            "source": record["source"],
            "problem": record["problem"],
            "problem_sha256": digest,
            "model": "Qwen/Qwen3-8B",
            "runtime": RUNTIME,
            "skill_card": skill_card,
            "skill_card_target_disjoint_audit": skill_card_disjoint_audit(
                record["problem"], skill_card
            ),
            "specialization_status": "ready",
            "specialization_failure_reason": "",
            "specialization_no_op": False,
            "candidate_count": 1,
            "requested_candidate_count": 1,
            "minimum_candidate_count": 1,
            "specialization_candidates": [candidate],
            "filter_summary": {
                "proposed_unique_count": 1,
                "accepted_count": 1,
                "rejected_count": 0,
                "verification_yield": 1.0,
            },
            "cost_audit": {
                "total_prompt_tokens": 20,
                "total_completion_tokens": 10,
                "total_generation_seconds": 0.8,
                "end_to_end_seconds": 1.0,
            },
            "firewall_audit": {
                "target_answer_loaded": False,
                "target_solution_loaded": False,
                "candidate_proposer_sources": ["sanitized_skill_card"],
                "correct_solver_sources": ["candidate_problem"],
                "wrong_trajectory_sources": [
                    "candidate_problem",
                    "sanitized_skill_card_failure_modes",
                ],
                "wrong_trajectory_correct_solution_exposed": False,
                "correct_verifier_sources": [
                    "candidate_problem",
                    "candidate_correct_trajectory",
                ],
                "frontier_verifier_sources": [
                    "candidate_problem",
                    "verified_correct_trajectory",
                    "model_wrong_trajectory",
                ],
                "solver_sources": ["candidate_problem"],
                "verifier_sources": [
                    "candidate_problem",
                    "candidate_correct_trajectory",
                    "model_wrong_trajectory",
                ],
                "all_accepted_candidate_artifacts_target_disjoint": True,
                "skill_card_redaction_count": 1,
                "skill_prompt_sha256": "c" * 64,
                "candidate_prompt_sha256": "d" * 64,
            },
        }
        proposal_row["proposal_training_sha256"] = compute_proposal_training_sha256(
            proposal_row
        )
        proposal_rows.append(proposal_row)
        proposal_sha = proposal_row["proposal_training_sha256"]
        privileged_artifact = build_privileged_control_artifact(
            record["problem"],
            answer,
            json.dumps(
                {
                    "reasoning_steps": [
                        "Identify the reusable invariant.",
                        "Reduce the cases symbolically and stop before evaluation.",
                    ]
                }
            ),
        )
        privileged_artifact["evaluated_model_prompt_sha256"] = "9" * 64
        task1_rows.append(
            {
                "stage": "task1_fast_teacher",
                "query_id": record["query_id"],
                "source": record["source"],
                "problem": record["problem"],
                "problem_sha256": digest,
                "reference_answer": answer,
                "real_run": True,
                "model": "Qwen/Qwen3-8B",
                "runtime": RUNTIME,
                "proposal_training_sha256": proposal_sha,
                "specialization_status": "ready",
                "specialization_failure_reason": "",
                "specialization_no_op": False,
                "ridge_config": ridge_config,
                "ridge_config_sha256": canonical_json_sha256(ridge_config),
                "run_config": task1_run_config,
                "run_config_sha256": canonical_json_sha256(task1_run_config),
                "max_output_tokens": 8192,
                "eval_samples": 1,
                "base_correct": 1.0 if is_amc else 0.0,
                "privileged_correct": 1.0,
                "teacher_correct": 1.0,
                "hindsight_exposure_rate": 0.0,
                "context_prefix_parity": 1.0,
                "hindsight_free_score": 1.0,
                "same_prefix_fidelity": 1.0,
                "hindsight_audit": {
                    "teacher_context_events": 1,
                    "forbidden_context_events": 0,
                    "comparison_events": 1,
                    "context_equal_events": 1,
                    "compared_token_positions": 4,
                    "same_prefix_positions": 4,
                    "causal_events": 1,
                    "on_policy_events": 0,
                    "on_policy_equal_events": 0,
                    "source_counts": {
                        "original_query": 1,
                        "sanitized_skill_card": 1,
                        "proposed_candidates": 1,
                    },
                },
                "student_evaluation_context_sha256": "e" * 64,
                "teacher_evaluation_context_sha256": "e" * 64,
                "privileged_hindsight_exposure_rate": 1.0,
                "privileged_context_prefix_parity": 0.0,
                "privileged_hindsight_free_score": 0.0,
                "privileged_control_artifact": privileged_artifact,
                "privileged_control_mode": privileged_artifact["control_mode"],
                "privileged_advantage_text_sha256": privileged_artifact[
                    "advantage_text_sha256"
                ],
                "privileged_answer_redaction_safe": True,
                "privileged_context_contains_literal_target_answer": False,
                "privileged_construction_used_target_answer": True,
                "privileged_cot_construction_seconds": 0.2,
                "privileged_cot_construction_attempts": 1,
                "privileged_cot_private_prompt_tokens": 20,
                "privileged_cot_construction_generated_tokens": 12,
                "privileged_cot_construction_truncated": False,
                "support_generation_seconds": 0.8,
                "proposal_end_to_end_seconds": 1.0,
                "specialization_seconds": 0.5,
                "total_adaptation_seconds": 1.5,
                "update_frobenius_norm": 2.0,
                "proposal_fit_signed_target_logit_gain": 0.8,
                "frontier_corrective_target_logit_gain": 1.0,
                "frontier_wrong_target_logit_change": -1.0,
                "frontier_corrective_tokens_selected": 4.0,
                "frontier_wrong_tokens_selected": 4.0,
                "update_norm_was_clipped": False,
                "adapter_rank": 16,
                "uses_all_candidates": True,
                "peak_memory_bytes": 1000,
                "max_input_tokens": 64,
                "support_generated_tokens": 10,
                "base_generated_tokens": 100 + index,
                "privileged_generated_tokens": 90 + index,
                "teacher_generated_tokens": 120 + index,
                "base_truncated": False,
                "privileged_truncated": False,
                "teacher_truncated": index == 2,
                "base_responses": [base_response],
                "privileged_responses": [teacher_response],
                "teacher_responses": [teacher_response],
                "base_parsed_answers": [base_parsed_answer],
                "privileged_parsed_answers": [answer],
                "teacher_parsed_answers": [answer],
                "base_target_answer_nll": 2.0,
                "teacher_target_answer_nll": 1.0,
            }
        )
        task2_rows.append(
            {
                "task": "task2_clean_distillation",
                "query_id": record["query_id"],
                "source": record["source"],
                "problem_sha256": digest,
                "reference_answer": answer,
                "real_run": True,
                "model": "Qwen/Qwen3-8B",
                "runtime": RUNTIME,
                "proposal_training_sha256": proposal_sha,
                "specialization_status": "ready",
                "specialization_failure_reason": "",
                "specialization_no_op": False,
                "ridge_config": ridge_config,
                "ridge_config_sha256": canonical_json_sha256(ridge_config),
                "run_config": task2_run_config,
                "run_config_sha256": canonical_json_sha256(task2_run_config),
                "max_output_tokens": 8192,
                "eval_samples": 1,
                "base_correct": 1.0 if is_amc else 0.0,
                "teacher_correct": 1.0,
                # Retain one of the two AIME teacher improvements.
                "distilled_correct": 0.0 if record["source"] == "aime25" else 1.0,
                "hindsight_exposure_rate": 0.0,
                "context_prefix_parity": 1.0,
                "hindsight_free_score": 1.0,
                "same_prefix_fidelity": 1.0,
                "hindsight_audit": {
                    "teacher_context_events": 2,
                    "forbidden_context_events": 0,
                    "comparison_events": 1,
                    "context_equal_events": 1,
                    "compared_token_positions": 4,
                    "same_prefix_positions": 4,
                    "causal_events": 2,
                    "on_policy_events": 1,
                    "on_policy_equal_events": 1,
                    "source_counts": {
                        "original_query": 2,
                        "sanitized_skill_card": 2,
                        "proposed_candidates": 2,
                        "student_generated_prefix": 1,
                    },
                },
                "support_generation_seconds": 0.8,
                "proposal_end_to_end_seconds": 1.0,
                "specialization_seconds": 0.5,
                "distillation_seconds": 0.5,
                "total_adaptation_seconds": 2.0,
                "teacher_destroyed_before_student_evaluation": True,
                "student_reset_verified": True,
                "update_frobenius_norm": 2.0,
                "proposal_fit_signed_target_logit_gain": 0.8,
                "frontier_corrective_target_logit_gain": 1.0,
                "frontier_wrong_target_logit_change": -1.0,
                "frontier_corrective_tokens_selected": 4.0,
                "frontier_wrong_tokens_selected": 4.0,
                "update_norm_was_clipped": False,
                "adapter_rank": 16,
                "student_update_frobenius_norm": 0.25,
                "uses_all_candidates": True,
                "distillation_steps_completed": 1,
                "distillation_trace": [
                    {
                        "step": 0,
                        "same_prefix": True,
                        "student_context_sha256": "f" * 64,
                        "teacher_context_sha256": "f" * 64,
                        "prefix_tokens": 4,
                        "compared_positions": 4,
                    }
                ],
                "peak_memory_bytes": 2000,
                "max_input_tokens": 64,
                "support_generated_tokens": 10,
                "base_generated_tokens": 100 + index,
                "teacher_generated_tokens": 120 + index,
                "distilled_generated_tokens": 110 + index,
                "base_truncated": False,
                "teacher_truncated": index == 2,
                "distilled_truncated": False,
                "base_responses": [base_response],
                "teacher_responses": [teacher_response],
                "distilled_responses": [distilled_response],
                "base_parsed_answers": [base_parsed_answer],
                "teacher_parsed_answers": [answer],
                "distilled_parsed_answers": [distilled_parsed_answer],
                "base_target_answer_nll": 2.0,
                "distilled_target_answer_nll": 1.5,
            }
        )

    proposal_a = root / "proposal.jsonl"
    task1_a = root / "task1.jsonl"
    task2_a = root / "task2.jsonl"
    _write_jsonl(proposal_a, proposal_rows)
    _write_jsonl(task1_a, task1_rows)
    _write_jsonl(task2_a, task2_rows)
    return (
        dataset_path,
        [proposal_a],
        [task1_a],
        [task2_a],
        proposal_rows,
        task1_rows,
        task2_rows,
    )


def _mark_specialization_no_op(
    proposal_row: dict,
    task1_row: dict,
    task2_row: dict,
    *,
    keep_partial_candidate: bool = False,
) -> None:
    reason = "verified candidates below minimum quality gate"
    if not keep_partial_candidate:
        proposal_row["specialization_candidates"] = []
    accepted_count = len(proposal_row["specialization_candidates"])
    proposal_row.update(
        {
            "specialization_status": "insufficient_verified_candidates",
            "specialization_failure_reason": reason,
            "specialization_no_op": True,
            "candidate_count": accepted_count,
            "requested_candidate_count": 3,
            "minimum_candidate_count": 2,
        }
    )
    proposal_row["filter_summary"] = {
        "proposed_unique_count": 1,
        "accepted_count": accepted_count,
        "rejected_count": 1 if accepted_count == 0 else 0,
        "verification_yield": float(accepted_count),
    }
    proposal_row["proposal_training_sha256"] = compute_proposal_training_sha256(
        proposal_row
    )

    for task_row in (task1_row, task2_row):
        task_row.update(
            {
                "proposal_training_sha256": proposal_row["proposal_training_sha256"],
                "specialization_status": "insufficient_verified_candidates",
                "specialization_failure_reason": reason,
                "specialization_no_op": True,
                "specialization_seconds": 0.0,
                "update_frobenius_norm": 0.0,
                "adapter_rank": 0,
                "uses_all_candidates": False,
                "proposal_fit_signed_target_logit_gain": 0.0,
                "frontier_corrective_target_logit_gain": 0.0,
                "frontier_wrong_target_logit_change": 0.0,
                "frontier_corrective_tokens_selected": 0.0,
                "frontier_wrong_tokens_selected": 0.0,
                "update_norm_was_clipped": False,
            }
        )

    task1_row["total_adaptation_seconds"] = task1_row["proposal_end_to_end_seconds"]
    task1_row["hindsight_audit"]["source_counts"] = {"original_query": 1}
    task1_row.update(
        {
            "teacher_responses": list(task1_row["base_responses"]),
            "teacher_parsed_answers": list(task1_row["base_parsed_answers"]),
            "teacher_correct": task1_row["base_correct"],
            "teacher_generated_tokens": task1_row["base_generated_tokens"],
            "teacher_truncated": task1_row["base_truncated"],
            "teacher_target_answer_nll": task1_row["base_target_answer_nll"],
        }
    )

    task2_row.update(
        {
            "distillation_seconds": 0.0,
            "total_adaptation_seconds": task2_row["proposal_end_to_end_seconds"],
            "student_update_frobenius_norm": 0.0,
            "distillation_steps_completed": 0,
            "distillation_trace": [],
            "context_prefix_parity": 0.0,
            "hindsight_free_score": 0.0,
            "same_prefix_fidelity": 0.0,
            "hindsight_audit": {
                "teacher_context_events": 1,
                "forbidden_context_events": 0,
                "comparison_events": 0,
                "context_equal_events": 0,
                "compared_token_positions": 0,
                "same_prefix_positions": 0,
                "causal_events": 1,
                "on_policy_events": 0,
                "on_policy_equal_events": 0,
                "source_counts": {"original_query": 1},
            },
        }
    )
    for prefix in ("teacher", "distilled"):
        task2_row[f"{prefix}_responses"] = list(task2_row["base_responses"])
        task2_row[f"{prefix}_parsed_answers"] = list(task2_row["base_parsed_answers"])
        task2_row[f"{prefix}_correct"] = task2_row["base_correct"]
        task2_row[f"{prefix}_generated_tokens"] = task2_row["base_generated_tokens"]
        task2_row[f"{prefix}_truncated"] = task2_row["base_truncated"]
    task2_row["distilled_target_answer_nll"] = task2_row["base_target_answer_nll"]


class PocReportTest(unittest.TestCase):
    def test_privileged_control_must_be_answer_redacted_cot_with_honest_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, _, task2, _, task1_rows, _ = _fixture(root)
            task1_rows[0]["privileged_control_artifact"][
                "advantage_text"
            ] = "The final answer is 101."
            tampered = root / "tampered-privileged.jsonl"
            _write_jsonl(tampered, task1_rows)
            with self.assertRaisesRegex(
                ReportValidationError, "contains a target-answer spelling"
            ):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=[tampered],
                    task2_paths=task2,
                    output_dir=root / "tampered-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

    def test_report_requires_full_32x_frontier_weight_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, _, task2, _, task1_rows, _ = _fixture(root)
            task1_rows[0]["ridge_config"] = {
                **task1_rows[0]["ridge_config"],
                "frontier_positive_weight": 7.5,
            }
            task1_rows[0]["ridge_config_sha256"] = canonical_json_sha256(
                task1_rows[0]["ridge_config"]
            )
            weakened = root / "weakened-frontier.jsonl"
            _write_jsonl(weakened, task1_rows)
            with self.assertRaisesRegex(ReportValidationError, "at least 32x"):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=[weakened],
                    task2_paths=task2,
                    output_dir=root / "weakened-frontier-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

    def test_generates_strict_four_condition_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, task2, _, _, _ = _fixture(root)
            output = root / "report"
            summary = generate_report(
                dataset_path=dataset,
                proposal_paths=proposals,
                task1_paths=task1,
                task2_paths=task2,
                output_dir=output,
                expected_counts=EXPECTED_SMOKE_COUNTS,
            )

            expected_files = {
                "merged_per_query.jsonl",
                "core_results.csv",
                "core_results.md",
                "summary.json",
                "experiment_summary.md",
            }
            self.assertEqual(expected_files, {path.name for path in output.iterdir()})
            self.assertEqual(summary["validation"]["unique_query_count"], 3)
            self.assertEqual(summary["validation"]["proposal_row_count"], 3)
            self.assertEqual(summary["model"], "Qwen/Qwen3-8B")
            self.assertEqual(summary["max_tokens"], 8192)
            self.assertIn("aime24", summary["metrics"]["by_method"]["Base"])
            self.assertIn("aime25", summary["metrics"]["by_method"]["Base"])
            diagnostics = summary["metrics"]["diagnostics_by_scope"]["overall"]
            self.assertEqual(
                diagnostics["proposal"]["mean_accepted_candidates_per_query"], 1.0
            )
            self.assertEqual(diagnostics["CSD-SD"]["max_peak_gpu_memory_bytes"], 2000.0)
            self.assertEqual(diagnostics["CSD-T"]["max_input_tokens"], 64)

            with (output / "core_results.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["Method"] for row in rows],
                ["Base", "Privileged Control", "CSD-T", "CSD-SD"],
            )
            by_method = {row["Method"]: row for row in rows}
            self.assertEqual(float(by_method["Base"]["AIME Acc@1 (%)"]), 0.0)
            self.assertEqual(float(by_method["CSD-T"]["AIME Acc@1 (%)"]), 100.0)
            self.assertEqual(float(by_method["CSD-T"]["HFAG (pp)"]), 100.0)
            self.assertEqual(float(by_method["CSD-SD"]["AIME Acc@1 (%)"]), 50.0)
            self.assertEqual(
                float(by_method["CSD-SD"]["CSD-SD Teacher-Gain Retention"]), 0.5
            )
            self.assertEqual(int(by_method["CSD-T"]["Truncated (All)"]), 1)
            self.assertEqual(
                float(by_method["Privileged Control"]["Mean Output Tokens"]), 91.0
            )
            self.assertEqual(float(by_method["Privileged Control"]["HER"]), 1.0)
            self.assertEqual(float(by_method["Privileged Control"]["CPP"]), 0.0)
            self.assertEqual(int(by_method["CSD-SD"]["Protocol No-op Queries"]), 0)

            merged = [
                json.loads(line)
                for line in (output / "merged_per_query.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(merged), 3)
            self.assertEqual(set(merged[0]["conditions"]), set(by_method))
            self.assertIn("proposal_artifact", merged[0])
            self.assertTrue(merged[0]["proposal_candidate_audits"])
            self.assertIn("task1_artifact", merged[0])
            self.assertIn("task2_artifact", merged[0])
            self.assertIn(
                "meets the requested", (output / "experiment_summary.md").read_text()
            )

    def test_default_counts_reject_smoke_dataset_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, task2, _, _, _ = _fixture(root)
            output = root / "report"
            with self.assertRaisesRegex(ReportValidationError, "source-count mismatch"):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=task2,
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_count_override_supports_single_source_infrastructure_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _, _, _, proposal_rows, task1_rows, task2_rows = _fixture(root)
            first_dataset_row = json.loads(
                dataset.read_text(encoding="utf-8").splitlines()[0]
            )
            smoke_dataset = root / "amc-only.jsonl"
            smoke_proposal = root / "amc-proposal.jsonl"
            smoke_task1 = root / "amc-task1.jsonl"
            smoke_task2 = root / "amc-task2.jsonl"
            _write_jsonl(smoke_dataset, [first_dataset_row])
            _write_jsonl(smoke_proposal, [proposal_rows[0]])
            _write_jsonl(smoke_task1, [task1_rows[0]])
            _write_jsonl(smoke_task2, [task2_rows[0]])
            output = root / "smoke-report"

            generate_report(
                dataset_path=smoke_dataset,
                proposal_paths=[smoke_proposal],
                task1_paths=[smoke_task1],
                task2_paths=[smoke_task2],
                output_dir=output,
                expected_counts={"amc23": 1, "aime24": 0, "aime25": 0},
            )
            with (output / "core_results.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["AIME Acc@1 (%)"] == "" for row in rows))
            self.assertIn(
                "coverage/infrastructure smoke report",
                (output / "experiment_summary.md").read_text(encoding="utf-8"),
            )

    def test_duplicate_shard_rows_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, task2, _, _, _ = _fixture(root)
            output = root / "report"
            with self.assertRaisesRegex(ReportValidationError, "duplicate query_id"):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=[*proposals, proposals[0]],
                    task1_paths=[*task1, task1[0]],
                    task2_paths=[*task2, task2[0]],
                    output_dir=output,
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )
            self.assertFalse(output.exists())

    def test_problem_hash_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            task2_rows[0]["problem_sha256"] = "0" * 64
            bad_task2 = root / "bad-task2.jsonl"
            _write_jsonl(bad_task2, task2_rows)
            output = root / "report"
            with self.assertRaisesRegex(
                ReportValidationError, "does not match dataset"
            ):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=[bad_task2],
                    output_dir=output,
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )
            self.assertFalse(output.exists())

    def test_missing_row_and_dry_run_row_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            missing_task2 = root / "missing-task2.jsonl"
            _write_jsonl(missing_task2, task2_rows[:-1])
            with self.assertRaisesRegex(
                ReportValidationError, "query coverage mismatch"
            ):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=[missing_task2],
                    output_dir=root / "missing-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

            task2_rows[0]["dry_run"] = True
            dry_task2 = root / "dry-task2.jsonl"
            _write_jsonl(dry_task2, task2_rows)
            with self.assertRaisesRegex(ReportValidationError, "refuses dry_run=true"):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=[dry_task2],
                    output_dir=root / "dry-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

    def test_runtime_provenance_and_cross_shard_signature_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, _, task2, _, task1_rows, _ = _fixture(root)
            task1_rows[0].pop("runtime")
            no_runtime = root / "no-runtime-task1.jsonl"
            _write_jsonl(no_runtime, task1_rows)
            with self.assertRaisesRegex(
                ReportValidationError, "missing required field"
            ):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=[no_runtime],
                    task2_paths=task2,
                    output_dir=root / "no-runtime-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            task2_rows[0]["runtime"] = {
                **RUNTIME,
                "resolved_model_revision": "c" * 40,
            }
            task2_rows[0]["run_config"] = {
                **task2_rows[0]["run_config"],
                "revision": "c" * 40,
            }
            task2_rows[0]["run_config_sha256"] = canonical_json_sha256(
                task2_rows[0]["run_config"]
            )
            mismatched_runtime = root / "mismatched-runtime-task2.jsonl"
            _write_jsonl(mismatched_runtime, task2_rows)
            with self.assertRaisesRegex(ReportValidationError, "software runtime"):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=[mismatched_runtime],
                    output_dir=root / "mismatched-runtime-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

    def test_restart_may_split_stages_across_array_jobs_on_same_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, _, _, _, task1_rows, task2_rows = _fixture(root)
            for row in task1_rows:
                row["runtime"] = {
                    **row["runtime"],
                    "slurm_array_job_id": "123457",
                    "hostname": "replacement-b200-node",
                }
            for row in task2_rows:
                row["runtime"] = {
                    **row["runtime"],
                    "slurm_array_job_id": "123458",
                    "hostname": "second-replacement-b200-node",
                }
            task1_path = root / "resumed-task1.jsonl"
            task2_path = root / "resumed-task2.jsonl"
            _write_jsonl(task1_path, task1_rows)
            _write_jsonl(task2_path, task2_rows)
            summary = generate_report(
                dataset_path=dataset,
                proposal_paths=proposals,
                task1_paths=[task1_path],
                task2_paths=[task2_path],
                output_dir=root / "resumed-report",
                expected_counts=EXPECTED_SMOKE_COUNTS,
            )
            self.assertEqual(
                summary["validation"]["allocation_provenance"][
                    "observed_array_task_ids"
                ],
                [0],
            )

    def test_sixteen_shard_range_allows_stage_restarts_across_jobs_and_hosts(self):
        def rows(job_id: str, hostname: str) -> dict[str, dict]:
            return {
                f"query-{task_id}": {
                    "model": RUNTIME["model"],
                    "runtime": {
                        **RUNTIME,
                        "slurm_array_job_id": job_id,
                        "slurm_array_task_id": str(task_id),
                        "hostname": hostname,
                    },
                }
                for task_id in range(16)
            }

        result = _validate_allocation_provenance(
            rows("700001", "proposal-b200"),
            rows("700002", "resumed-task1-b200"),
            rows("700003", "resumed-task2-b200"),
            EXPECTED_SMOKE_COUNTS,
            16,
        )
        self.assertEqual(result["expected_shard_count"], 16)
        self.assertEqual(result["expected_array_task_ids"], list(range(16)))
        self.assertEqual(result["observed_array_task_ids"], list(range(16)))
        self.assertEqual(result["distinct_array_task_allocations"], 48)

    def test_supplied_shard_triplets_drive_task_id_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _, _, _, proposal_rows, task1_rows, task2_rows = _fixture(root)
            for row_index, rows in enumerate(
                zip(proposal_rows, task1_rows, task2_rows, strict=True)
            ):
                task_id = 0 if row_index < 2 else 1
                for row in rows:
                    row["runtime"] = {
                        **row["runtime"],
                        "slurm_array_task_id": str(task_id),
                    }
            for row in [*task1_rows, *task2_rows]:
                row["run_config"] = {**row["run_config"], "num_shards": 2}
                row["run_config_sha256"] = canonical_json_sha256(row["run_config"])

            proposal_paths = [root / "proposal-0.jsonl", root / "proposal-1.jsonl"]
            task1_paths = [root / "task1-0.jsonl", root / "task1-1.jsonl"]
            task2_paths = [root / "task2-0.jsonl", root / "task2-1.jsonl"]
            for paths, rows in (
                (proposal_paths, proposal_rows),
                (task1_paths, task1_rows),
                (task2_paths, task2_rows),
            ):
                _write_jsonl(paths[0], rows[:2])
                _write_jsonl(paths[1], rows[2:])

            summary = generate_report(
                dataset_path=dataset,
                proposal_paths=proposal_paths,
                task1_paths=task1_paths,
                task2_paths=task2_paths,
                output_dir=root / "two-shard-report",
                expected_counts=EXPECTED_SMOKE_COUNTS,
            )
            allocation = summary["validation"]["allocation_provenance"]
            self.assertEqual(allocation["expected_shard_count"], 2)
            self.assertEqual(allocation["observed_array_task_ids"], [0, 1])

            for rows in (proposal_rows, task1_rows, task2_rows):
                rows[-1]["runtime"] = {
                    **rows[-1]["runtime"],
                    "slurm_array_task_id": "0",
                }
            for paths, rows in (
                (proposal_paths, proposal_rows),
                (task1_paths, task1_rows),
                (task2_paths, task2_rows),
            ):
                _write_jsonl(paths[0], rows[:2])
                _write_jsonl(paths[1], rows[2:])
            with self.assertRaisesRegex(
                ReportValidationError,
                r"requires deterministic array task ids \[0, 1\].*observed task ids \[0\]",
            ):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposal_paths,
                    task1_paths=task1_paths,
                    task2_paths=task2_paths,
                    output_dir=root / "missing-shard-id-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

    def test_crossed_stage_task_ids_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            task2_rows[0]["runtime"] = {
                **task2_rows[0]["runtime"],
                "slurm_array_task_id": "1",
            }
            task2_path = root / "crossed-task2.jsonl"
            _write_jsonl(task2_path, task2_rows)
            with self.assertRaisesRegex(
                ReportValidationError, "crossed shard task ids"
            ):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=[task2_path],
                    output_dir=root / "crossed-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

    def test_nfs_torch_module_path_matches_home_overlay_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            for row in task2_rows:
                row["runtime"] = {
                    **row["runtime"],
                    "torch_module_path": (
                        "/nfs/roberts/scratch/pi_mg269/da839/mfspd/"
                        "pydeps-cu128/torch/__init__.py"
                    ),
                }
            task2_path = root / "nfs-module-task2.jsonl"
            _write_jsonl(task2_path, task2_rows)
            summary = generate_report(
                dataset_path=dataset,
                proposal_paths=proposals,
                task1_paths=task1,
                task2_paths=[task2_path],
                output_dir=root / "nfs-module-report",
                expected_counts=EXPECTED_SMOKE_COUNTS,
            )
            self.assertEqual(summary["validation"]["status"], "passed")

    def test_runtime_must_use_actual_ttt_cu128_overlay(self):
        mutations = (
            ({"python_executable": "/scratch/not-TTT/bin/python"}, "activated TTT"),
            (
                {
                    "conda_prefix": "/tmp/TTT",
                    "python_executable": "/tmp/TTT/bin/python",
                },
                "exact required TTT prefix",
            ),
            (
                {
                    "torch_overlay": "/tmp/other-cu128",
                    "torch_module_path": "/tmp/other-cu128/torch/__init__.py",
                },
                "exact approved cu128 overlay",
            ),
            (
                {"torch_module_path": "/tmp/unbound/torch/__init__.py"},
                "not inside torch_overlay",
            ),
            ({"torch": "2.9.1+cu126", "cuda_runtime": "12.6"}, "cu128 build"),
        )
        for mutation, expected_error in mutations:
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
                task2_rows[0]["runtime"] = {**task2_rows[0]["runtime"], **mutation}
                bad_task2 = root / "bad-runtime-task2.jsonl"
                _write_jsonl(bad_task2, task2_rows)
                with self.assertRaisesRegex(ReportValidationError, expected_error):
                    generate_report(
                        dataset_path=dataset,
                        proposal_paths=proposals,
                        task1_paths=task1,
                        task2_paths=[bad_task2],
                        output_dir=root / "bad-runtime-report",
                        expected_counts=EXPECTED_SMOKE_COUNTS,
                    )

    def test_authoritative_regrade_rejects_declared_correctness_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, _, task2, _, task1_rows, _ = _fixture(root)
            # The stored response is correctly boxed, so this false declaration
            # must not be allowed to flow into Acc@1/HFAG.
            task1_rows[0]["teacher_correct"] = 0.0
            bad_task1 = root / "bad-correctness-task1.jsonl"
            _write_jsonl(bad_task1, task1_rows)
            with self.assertRaisesRegex(
                ReportValidationError,
                "declared teacher_correct=.*authoritative regrade",
            ):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=[bad_task1],
                    task2_paths=task2,
                    output_dir=root / "bad-correctness-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

    def test_dirty_runtime_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            task2_rows[0]["runtime"] = {**RUNTIME, "git_dirty": True}
            dirty_task2 = root / "dirty-task2.jsonl"
            _write_jsonl(dirty_task2, task2_rows)
            with self.assertRaisesRegex(ReportValidationError, "git_dirty=true"):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=[dirty_task2],
                    output_dir=root / "dirty-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

    def test_b200_compute_capability_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            task2_rows[0]["runtime"] = {
                **RUNTIME,
                "gpus": [{"index": 0, "name": "NVIDIA B200", "capability": [9, 0]}],
            }
            incompatible_task2 = root / "incompatible-task2.jsonl"
            _write_jsonl(incompatible_task2, task2_rows)
            with self.assertRaisesRegex(
                ReportValidationError, "capability \\(10, 0\\)"
            ):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=[incompatible_task2],
                    output_dir=root / "incompatible-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

    def test_explicit_specialization_no_op_is_retained_and_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                dataset,
                _,
                _,
                _,
                proposal_rows,
                task1_rows,
                task2_rows,
            ) = _fixture(root)
            _mark_specialization_no_op(proposal_rows[0], task1_rows[0], task2_rows[0])
            proposal_path = root / "no-op-proposal.jsonl"
            task1_path = root / "no-op-task1.jsonl"
            no_op_task2 = root / "no-op-task2.jsonl"
            _write_jsonl(proposal_path, proposal_rows)
            _write_jsonl(task1_path, task1_rows)
            _write_jsonl(no_op_task2, task2_rows)
            output = root / "no-op-report"
            summary = generate_report(
                dataset_path=dataset,
                proposal_paths=[proposal_path],
                task1_paths=[task1_path],
                task2_paths=[no_op_task2],
                output_dir=output,
                expected_counts=EXPECTED_SMOKE_COUNTS,
            )
            for method in ("CSD-T", "CSD-SD"):
                overall = summary["metrics"]["by_method"][method]["overall"]
                self.assertEqual(overall["protocol_no_op_count"], 1)
                self.assertAlmostEqual(overall["protocol_no_op_rate"], 1.0 / 3.0)
            merged_first = json.loads(
                (output / "merged_per_query.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                merged_first["specialization_status"],
                "insufficient_verified_candidates",
            )
            self.assertTrue(merged_first["specialization_no_op"])
            for method in ("CSD-T", "CSD-SD"):
                self.assertTrue(merged_first["conditions"][method]["protocol_no_op"])
            summary_text = (output / "experiment_summary.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("CSD-T specialization no-ops: 1/3", summary_text)
            self.assertIn("CSD-SD protocol no-ops: 1/3", summary_text)

    def test_partial_verified_candidates_may_be_retained_below_no_op_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _, _, _, proposal_rows, task1_rows, task2_rows = _fixture(root)
            _mark_specialization_no_op(
                proposal_rows[0],
                task1_rows[0],
                task2_rows[0],
                keep_partial_candidate=True,
            )
            proposal_path = root / "partial-proposal.jsonl"
            task1_path = root / "partial-task1.jsonl"
            task2_path = root / "partial-task2.jsonl"
            _write_jsonl(proposal_path, proposal_rows)
            _write_jsonl(task1_path, task1_rows)
            _write_jsonl(task2_path, task2_rows)
            summary = generate_report(
                dataset_path=dataset,
                proposal_paths=[proposal_path],
                task1_paths=[task1_path],
                task2_paths=[task2_path],
                output_dir=root / "partial-report",
                expected_counts=EXPECTED_SMOKE_COUNTS,
            )
            self.assertEqual(
                summary["validation"]["specialization_no_op_query_count"], 1
            )
            self.assertEqual(
                summary["metrics"]["diagnostics_by_scope"]["amc23"]["proposal"][
                    "mean_accepted_candidates_per_query"
                ],
                1.0,
            )

    def test_specialization_state_and_zero_update_contract_are_strict(self):
        cases = (
            ("empty_ready", "ready specialization requires at least one"),
            ("blank_reason", "requires a nonempty failure reason"),
            ("no_op_positive_rank", "specialization no-op requires adapter_rank"),
            (
                "ready_zero_update",
                "ready specialization requires a nonzero ridge update",
            ),
            ("task_reason_mismatch", "disagrees with proposal"),
            ("no_op_source_contamination", "exact specialization provenance"),
        )
        for case, expected_error in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (
                    dataset,
                    _,
                    _,
                    _,
                    proposal_rows,
                    task1_rows,
                    task2_rows,
                ) = _fixture(root)
                if case == "empty_ready":
                    proposal_rows[0]["specialization_candidates"] = []
                    proposal_rows[0]["candidate_count"] = 0
                elif case == "ready_zero_update":
                    task1_rows[0]["update_frobenius_norm"] = 0.0
                else:
                    _mark_specialization_no_op(
                        proposal_rows[0], task1_rows[0], task2_rows[0]
                    )
                    if case == "blank_reason":
                        proposal_rows[0]["specialization_failure_reason"] = "   "
                    elif case == "no_op_positive_rank":
                        task1_rows[0]["adapter_rank"] = 1
                    elif case == "task_reason_mismatch":
                        task2_rows[0][
                            "specialization_failure_reason"
                        ] = "different nonempty gate failure"
                    elif case == "no_op_source_contamination":
                        task2_rows[0]["hindsight_audit"]["source_counts"].update(
                            {
                                "sanitized_skill_card": 1,
                                "proposed_candidates": 1,
                            }
                        )

                proposal_path = root / f"{case}-proposal.jsonl"
                task1_path = root / f"{case}-task1.jsonl"
                task2_path = root / f"{case}-task2.jsonl"
                _write_jsonl(proposal_path, proposal_rows)
                _write_jsonl(task1_path, task1_rows)
                _write_jsonl(task2_path, task2_rows)
                with self.assertRaisesRegex(ReportValidationError, expected_error):
                    generate_report(
                        dataset_path=dataset,
                        proposal_paths=[proposal_path],
                        task1_paths=[task1_path],
                        task2_paths=[task2_path],
                        output_dir=root / f"{case}-report",
                        expected_counts=EXPECTED_SMOKE_COUNTS,
                    )

    def test_no_op_requires_empirical_base_equivalence(self):
        cases = (
            ("task1_response", "no-op response drift"),
            ("task1_parsed", "no-op parsed-answer drift"),
            ("task1_correct", "no-op correctness drift"),
            ("task1_tokens", "no-op generated-token drift"),
            ("task1_truncated", "no-op truncation drift"),
            ("task1_nll", "no-op NLL drift"),
            ("task2_teacher_response", "no-op response drift"),
            ("task2_distilled_response", "no-op response drift"),
            ("task2_distilled_nll", "no-op NLL drift"),
        )
        for case, expected_error in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (
                    dataset,
                    _,
                    _,
                    _,
                    proposal_rows,
                    task1_rows,
                    task2_rows,
                ) = _fixture(root)
                _mark_specialization_no_op(
                    proposal_rows[0], task1_rows[0], task2_rows[0]
                )
                if case == "task1_response":
                    task1_rows[0]["teacher_responses"] = [r"Drift. \boxed{999}"]
                elif case == "task1_parsed":
                    task1_rows[0]["teacher_parsed_answers"] = ["999"]
                elif case == "task1_correct":
                    task1_rows[0]["teacher_correct"] = 0.0
                elif case == "task1_tokens":
                    task1_rows[0]["teacher_generated_tokens"] += 1
                elif case == "task1_truncated":
                    task1_rows[0]["teacher_truncated"] = True
                elif case == "task1_nll":
                    task1_rows[0]["teacher_target_answer_nll"] += 0.25
                elif case == "task2_teacher_response":
                    task2_rows[0]["teacher_responses"] = [r"Drift. \boxed{999}"]
                elif case == "task2_distilled_response":
                    task2_rows[0]["distilled_responses"] = [r"Drift. \boxed{999}"]
                elif case == "task2_distilled_nll":
                    task2_rows[0]["distilled_target_answer_nll"] += 0.25

                proposal_path = root / f"{case}-proposal.jsonl"
                task1_path = root / f"{case}-task1.jsonl"
                task2_path = root / f"{case}-task2.jsonl"
                _write_jsonl(proposal_path, proposal_rows)
                _write_jsonl(task1_path, task1_rows)
                _write_jsonl(task2_path, task2_rows)
                with self.assertRaisesRegex(ReportValidationError, expected_error):
                    generate_report(
                        dataset_path=dataset,
                        proposal_paths=[proposal_path],
                        task1_paths=[task1_path],
                        task2_paths=[task2_path],
                        output_dir=root / f"{case}-report",
                        expected_counts=EXPECTED_SMOKE_COUNTS,
                    )

    def test_accepted_candidate_placeholder_artifacts_are_rejected(self):
        candidate_mutations = (
            ("problem", "Compute using a redacted number."),
            ("problem", "Replace the placeholder value before solving."),
            ("problem", "Find the unspecified quantity."),
            ("problem", "Use the omitted term."),
            ("problem", "Determine TBD."),
            ("problem", "Compute <replacement_value>."),
            ("problem", "Find a variable quantity."),
            ("problem", "Use a symbolic relation."),
            ("problem", "Determine an abstract object."),
            ("problem", "Compute with an abstract element."),
            ("problem", "Introduce an auxiliary variable."),
            ("problem", "State the derived conclusion."),
            ("solution", "Substitute the unspecified quantity."),
            ("final_answer", "<replacement_value>"),
        )
        for field, value in candidate_mutations:
            with self.subTest(
                field=field, value=value
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset, _, task1, task2, proposal_rows, _, _ = _fixture(root)
                proposal_rows[0]["specialization_candidates"][0][field] = value
                proposal_path = root / "placeholder-proposal.jsonl"
                _write_jsonl(proposal_path, proposal_rows)
                with self.assertRaisesRegex(
                    ReportValidationError, "placeholder artifacts"
                ):
                    generate_report(
                        dataset_path=dataset,
                        proposal_paths=[proposal_path],
                        task1_paths=task1,
                        task2_paths=task2,
                        output_dir=root / "placeholder-report",
                        expected_counts=EXPECTED_SMOKE_COUNTS,
                    )

    def test_accepted_candidate_training_fields_must_be_nonempty(self):
        for field in ("solution", "final_answer"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset, _, task1, task2, proposal_rows, _, _ = _fixture(root)
                proposal_rows[0]["specialization_candidates"][0][field] = ""
                proposal_path = root / "empty-training-field-proposal.jsonl"
                _write_jsonl(proposal_path, proposal_rows)
                with self.assertRaisesRegex(
                    ReportValidationError, rf"\.{field} is empty"
                ):
                    generate_report(
                        dataset_path=dataset,
                        proposal_paths=[proposal_path],
                        task1_paths=task1,
                        task2_paths=task2,
                        output_dir=root / "empty-training-field-report",
                        expected_counts=EXPECTED_SMOKE_COUNTS,
                    )

    def test_proposal_candidates_are_reaudited_against_authoritative_problem(self):
        mutations = (
            (0, "Practice amc combinatorics.", "re-audit"),
            (1, "Compute a product involving 2024.", "re-audit"),
        )
        for row_index, bad_problem, expected_error in mutations:
            with self.subTest(
                bad_problem=bad_problem
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset, _, task1, task2, proposal_rows, _, _ = _fixture(root)
                proposal_rows[row_index]["specialization_candidates"][0][
                    "problem"
                ] = bad_problem
                bad_proposals = root / "bad-proposals.jsonl"
                _write_jsonl(bad_proposals, proposal_rows)
                with self.assertRaisesRegex(ReportValidationError, expected_error):
                    generate_report(
                        dataset_path=dataset,
                        proposal_paths=[bad_proposals],
                        task1_paths=task1,
                        task2_paths=task2,
                        output_dir=root / "bad-proposal-report",
                        expected_counts=EXPECTED_SMOKE_COUNTS,
                    )

    def test_acc1_and_csd_sd_protocol_evidence_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            task2_rows[0]["eval_samples"] = 8
            task2_rows[0]["base_responses"] = ["same"] * 8
            task2_rows[0]["teacher_responses"] = ["same"] * 8
            task2_rows[0]["distilled_responses"] = ["same"] * 8
            many_samples = root / "many-sample-task2.jsonl"
            _write_jsonl(many_samples, task2_rows)
            with self.assertRaisesRegex(ReportValidationError, "eval_samples=1"):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=[many_samples],
                    output_dir=root / "many-sample-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

    def test_audit_digest_firewall_and_fixed_config_tampering_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, task2, proposal_rows, _, _ = _fixture(root)

            proposal_rows[0]["firewall_audit"]["target_answer_loaded"] = True
            bad_proposal = root / "bad-firewall.jsonl"
            _write_jsonl(bad_proposal, proposal_rows)
            with self.assertRaisesRegex(
                ReportValidationError, "clean proposal boundary"
            ):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=[bad_proposal],
                    task1_paths=task1,
                    task2_paths=task2,
                    output_dir=root / "bad-firewall-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            task2_rows[0]["distillation_trace"][0]["teacher_context_sha256"] = "0" * 64
            bad_trace = root / "bad-trace.jsonl"
            _write_jsonl(bad_trace, task2_rows)
            with self.assertRaisesRegex(
                ReportValidationError, "disagrees with context hashes"
            ):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=[bad_trace],
                    output_dir=root / "bad-trace-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            task2_rows[-1]["run_config"] = {
                **task2_rows[-1]["run_config"],
                "seed": 99,
            }
            task2_rows[-1]["run_config_sha256"] = canonical_json_sha256(
                task2_rows[-1]["run_config"]
            )
            bad_config = root / "bad-config.jsonl"
            _write_jsonl(bad_config, task2_rows)
            with self.assertRaisesRegex(ReportValidationError, "fixed run_config"):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=[bad_config],
                    output_dir=root / "bad-config-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )

    def test_summary_distinguishes_dev_only_from_heldout_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _, _, _, proposal_rows, task1_rows, task2_rows = _fixture(root)
            wrong = r"Reasoning. \boxed{999}"
            # AMC: Base wrong, both clean methods correct.
            task1_rows[0]["base_responses"] = [wrong]
            task1_rows[0]["base_correct"] = 0.0
            task2_rows[0]["base_responses"] = [wrong]
            task2_rows[0]["base_correct"] = 0.0
            # Held-out AIME: no method improves over the already-wrong Base.
            for index in (1, 2):
                task1_rows[index]["teacher_responses"] = [wrong]
                task1_rows[index]["teacher_correct"] = 0.0
                task2_rows[index]["teacher_responses"] = [wrong]
                task2_rows[index]["teacher_correct"] = 0.0
                task2_rows[index]["distilled_responses"] = [wrong]
                task2_rows[index]["distilled_correct"] = 0.0
            proposal_path = root / "proposal.jsonl"
            task1_path = root / "task1.jsonl"
            task2_path = root / "task2.jsonl"
            _write_jsonl(proposal_path, proposal_rows)
            _write_jsonl(task1_path, task1_rows)
            _write_jsonl(task2_path, task2_rows)
            output = root / "report"
            generate_report(
                dataset_path=dataset,
                proposal_paths=[proposal_path],
                task1_paths=[task1_path],
                task2_paths=[task2_path],
                output_dir=output,
                expected_counts=EXPECTED_SMOKE_COUNTS,
            )
            text = (output / "experiment_summary.md").read_text(encoding="utf-8")
            self.assertIn("dev-only signal", text)
            self.assertIn("held-out AIME remains below", text)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            task2_rows[0]["teacher_destroyed_before_student_evaluation"] = False
            unsafe_task2 = root / "unsafe-task2.jsonl"
            _write_jsonl(unsafe_task2, task2_rows)
            with self.assertRaisesRegex(ReportValidationError, "not destroyed"):
                generate_report(
                    dataset_path=dataset,
                    proposal_paths=proposals,
                    task1_paths=task1,
                    task2_paths=[unsafe_task2],
                    output_dir=root / "unsafe-report",
                    expected_counts=EXPECTED_SMOKE_COUNTS,
                )


if __name__ == "__main__":
    unittest.main()
