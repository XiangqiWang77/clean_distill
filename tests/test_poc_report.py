"""Contract tests for strict four-condition PoC aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.clean_self_distill.report_poc import ReportValidationError, generate_report
from src.clean_self_distill.io import load_query_records


EXPECTED_SMOKE_COUNTS = {"amc23": 1, "aime24": 1, "aime25": 1}
RUNTIME = {
    "timestamp_utc": "2026-08-03T18:00:00+00:00",
    "git_commit": "a" * 40,
    "git_dirty": False,
    "model": "Qwen/Qwen3-8B",
    "resolved_model_revision": "b" * 40,
    "conda_prefix": "/cluster/envs/TTT",
    "torch": "2.9.1+cu126",
    "cuda_runtime": "12.6",
    "cuda_available": True,
    "gpu_count": 1,
    "gpus": [{"index": 0, "name": "NVIDIA B200"}],
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
    for index, record in enumerate(canonical):
        digest = hashlib.sha256(record["problem"].encode("utf-8")).hexdigest()
        is_amc = record["source"] == "amc23"
        answer = record["answer"]
        wrong_answer = "999"
        base_response = rf"Reasoning. \boxed{{{answer if is_amc else wrong_answer}}}"
        teacher_response = rf"Reasoning. \boxed{{{answer}}}"
        distilled_response = rf"Reasoning. \boxed{{{answer if record['source'] != 'aime25' else wrong_answer}}}"
        proposal_rows.append(
            {
                "query_id": record["query_id"],
                "source": record["source"],
                "problem": record["problem"],
                "problem_sha256": digest,
                "model": "Qwen/Qwen3-8B",
                "runtime": RUNTIME,
                "candidate_count": 1,
                "specialization_candidates": [
                    {
                        "candidate_id": "c00",
                        "problem": "Compute a product of two symbolic quantities.",
                        "solution": "Multiply the quantities.",
                        "final_answer": "z",
                        "verifier_accepted": True,
                    }
                ],
            }
        )
        task1_rows.append(
            {
                "stage": "task1_fast_teacher",
                "query_id": record["query_id"],
                "source": record["source"],
                "problem_sha256": digest,
                "reference_answer": answer,
                "real_run": True,
                "model": "Qwen/Qwen3-8B",
                "runtime": RUNTIME,
                "max_output_tokens": 8192,
                "eval_samples": 1,
                "base_correct": 1.0 if is_amc else 0.0,
                "privileged_correct": 1.0,
                "teacher_correct": 1.0,
                "hindsight_exposure_rate": 0.0,
                "context_prefix_parity": 1.0,
                "hindsight_free_score": 1.0,
                "privileged_hindsight_exposure_rate": 1.0,
                "privileged_context_prefix_parity": 0.0,
                "privileged_hindsight_free_score": 0.0,
                "support_generation_seconds": 0.8,
                "proposal_end_to_end_seconds": 1.0,
                "specialization_seconds": 0.5,
                "total_adaptation_seconds": 1.5,
                "update_frobenius_norm": 2.0,
                "adapter_rank": 16,
                "base_generated_tokens": 100 + index,
                "privileged_generated_tokens": 90 + index,
                "teacher_generated_tokens": 120 + index,
                "base_truncated": False,
                "privileged_truncated": False,
                "teacher_truncated": index == 2,
                "base_responses": [base_response],
                "privileged_responses": [teacher_response],
                "teacher_responses": [teacher_response],
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
                "max_output_tokens": 8192,
                "eval_samples": 1,
                "base_correct": 1.0 if is_amc else 0.0,
                "teacher_correct": 1.0,
                # Retain one of the two AIME teacher improvements.
                "distilled_correct": 0.0 if record["source"] == "aime25" else 1.0,
                "hindsight_exposure_rate": 0.0,
                "context_prefix_parity": 1.0,
                "hindsight_free_score": 1.0,
                "support_generation_seconds": 0.8,
                "proposal_end_to_end_seconds": 1.0,
                "specialization_seconds": 0.5,
                "distillation_seconds": 0.5,
                "total_adaptation_seconds": 2.0,
                "teacher_destroyed_before_student_evaluation": True,
                "student_reset_verified": True,
                "student_update_frobenius_norm": 0.25,
                "distillation_steps_completed": 1,
                "distillation_trace": [{"step": 0, "same_prefix": True}],
                "base_generated_tokens": 100 + index,
                "teacher_generated_tokens": 120 + index,
                "distilled_generated_tokens": 110 + index,
                "base_truncated": False,
                "teacher_truncated": index == 2,
                "distilled_truncated": False,
                "base_responses": [base_response],
                "teacher_responses": [teacher_response],
                "distilled_responses": [distilled_response],
            }
        )

    # Exercise repeatable shards rather than relying on a single file.
    proposal_a = root / "proposal-a.jsonl"
    proposal_b = root / "proposal-b.jsonl"
    task1_a = root / "task1-a.jsonl"
    task1_b = root / "task1-b.jsonl"
    task2_a = root / "task2-a.jsonl"
    task2_b = root / "task2-b.jsonl"
    _write_jsonl(proposal_a, proposal_rows[:2])
    _write_jsonl(proposal_b, proposal_rows[2:])
    _write_jsonl(task1_a, task1_rows[:2])
    _write_jsonl(task1_b, task1_rows[2:])
    _write_jsonl(task2_a, task2_rows[:1])
    _write_jsonl(task2_b, task2_rows[1:])
    return (
        dataset_path,
        [proposal_a, proposal_b],
        [task1_a, task1_b],
        [task2_a, task2_b],
        proposal_rows,
        task1_rows,
        task2_rows,
    )


class PocReportTest(unittest.TestCase):
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
            self.assertEqual(
                int(by_method["CSD-SD"]["Protocol No-op Queries"]), 0
            )

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
                    proposal_paths=proposals,
                    task1_paths=[*task1, task1[0]],
                    task2_paths=task2,
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
                ReportValidationError, "declared teacher_correct=.*authoritative regrade"
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

    def test_csd_sd_no_op_is_retained_and_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, proposals, task1, _, _, _, task2_rows = _fixture(root)
            task2_rows[0]["student_update_frobenius_norm"] = 0.0
            task2_rows[0]["distillation_steps_completed"] = 0
            task2_rows[0]["distillation_trace"] = []
            no_op_task2 = root / "no-op-task2.jsonl"
            _write_jsonl(no_op_task2, task2_rows)
            output = root / "no-op-report"
            summary = generate_report(
                dataset_path=dataset,
                proposal_paths=proposals,
                task1_paths=task1,
                task2_paths=[no_op_task2],
                output_dir=output,
                expected_counts=EXPECTED_SMOKE_COUNTS,
            )
            overall = summary["metrics"]["by_method"]["CSD-SD"]["overall"]
            self.assertEqual(overall["protocol_no_op_count"], 1)
            self.assertAlmostEqual(overall["protocol_no_op_rate"], 1.0 / 3.0)
            merged_first = json.loads(
                (output / "merged_per_query.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertTrue(
                merged_first["conditions"]["CSD-SD"]["protocol_no_op"]
            )
            self.assertIn(
                "CSD-SD protocol no-ops: 1/3",
                (output / "experiment_summary.md").read_text(encoding="utf-8"),
            )

    def test_proposal_candidates_are_reaudited_against_authoritative_problem(self):
        mutations = (
            (0, "Practice amc combinatorics.", "entities"),
            (1, "Compute a product involving 2024.", "numeric/entity literals"),
        )
        for row_index, bad_problem, expected_error in mutations:
            with self.subTest(bad_problem=bad_problem), tempfile.TemporaryDirectory() as directory:
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
