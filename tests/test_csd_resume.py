"""Pure-helper regressions for preemption-safe Task 1/Task 2 JSONL resume."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.clean_self_distill.io import (
    compute_proposal_training_sha256,
    stable_hash,
)
from src.clean_self_distill.metrics import HindsightAudit
from src.clean_self_distill.train_eval import (
    _index_proposals_by_hash,
    _load_resumable_evaluation_rows,
    _row_audit_fields,
    _row_config_fields,
    _teacher_context_sources,
)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        model="Qwen/Qwen3-4B",
        revision="revision",
        runtime_metadata={
            "python_executable": "/test/python",
            "conda_prefix": "/test",
            "torch_overlay": "",
            "torch": "test",
            "torch_module_path": "/test/torch/__init__.py",
            "torch_arch_flags": [],
            "cuda_runtime": None,
            "model": "Qwen/Qwen3-4B",
            "requested_model_revision": "revision",
            "resolved_model_revision": "revision",
            "git_commit": "commit",
            "git_dirty": False,
            "slurm_array_task_id": "",
            "gpu_count": 0,
            "gpus": [],
        },
        resume=True,
        ridge_lambda=0.1,
        residual_step_size=0.8,
        max_tokens_per_candidate=64,
        max_support_tokens=256,
        num_specialization_candidates=None,
        hard_negatives=8,
        max_length=8192,
        seed=0,
    )


def _record(index: int, source: str) -> dict:
    problem = f"Independent problem {index}."
    return {
        "query_id": f"{source}:q{index}:fixture",
        "problem": problem,
        "problem_sha256": stable_hash(problem, 64),
        "source": source,
        "answer": str(index + 1),
        "solution": "",
    }


def _proposal(record: dict) -> dict:
    row = {
        "query_id": record["query_id"],
        "problem": record["problem"],
        "problem_sha256": record["problem_sha256"],
        "source": record["source"],
        "skill_card": {"skills": ["reason abstractly"]},
        "specialization_candidates": [],
        "specialization_status": "insufficient_verified_candidates",
        "specialization_failure_reason": "no candidates passed the quality gate",
        "specialization_no_op": True,
    }
    row["proposal_training_sha256"] = compute_proposal_training_sha256(row)
    return row


def _task1_row(record: dict, proposal: dict, args: SimpleNamespace, positions: int) -> dict:
    audit = HindsightAudit()
    audit.record_teacher_context(
        _teacher_context_sources(proposal, on_policy=False), causal=True
    )
    audit.record_same_prefix([1, 2], [1, 2], positions=positions)
    return {
        "stage": "task1_fast_teacher",
        "query_id": record["query_id"],
        "problem": record["problem"],
        "problem_sha256": record["problem_sha256"],
        "proposal_training_sha256": proposal["proposal_training_sha256"],
        "specialization_status": proposal["specialization_status"],
        "specialization_failure_reason": proposal[
            "specialization_failure_reason"
        ],
        "specialization_no_op": proposal["specialization_no_op"],
        "reference_answer": record["answer"],
        "source": record["source"],
        "model": args.model,
        "model_revision": args.runtime_metadata["resolved_model_revision"],
        "runtime": dict(args.runtime_metadata),
        **_row_config_fields(args),
        "student_evaluation_context_sha256": "same-context",
        "teacher_evaluation_context_sha256": "same-context",
        "base_responses": [r"Reasoning. \\boxed{1}"],
        "teacher_responses": [r"Reasoning. \\boxed{1}"],
        **_row_audit_fields(audit),
    }


def _encoded(row: dict, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (json.dumps(row, ensure_ascii=False) + suffix).encode("utf-8")


class CSDResumeHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = _args()
        self.records = [
            _record(0, "amc23"),
            _record(1, "aime24"),
            _record(2, "aime25"),
        ]
        self.proposals = {
            record["query_id"]: _proposal(record) for record in self.records
        }
        self.rows = [
            _task1_row(
                record,
                self.proposals[record["query_id"]],
                self.args,
                positions=index + 2,
            )
            for index, record in enumerate(self.records)
        ]
        self.proposals_by_hash = _index_proposals_by_hash(self.proposals)

    def _load(self, path: Path):
        return _load_resumable_evaluation_rows(
            path,
            self.records,
            self.proposals,
            self.proposals_by_hash,
            self.args,
            task="task1",
            stage="task1_fast_teacher",
        )

    @staticmethod
    def _backups(path: Path) -> list[Path]:
        return sorted(path.parent.glob(f"{path.name}.resume-backup.*"))

    def test_exact_prefix_restores_overall_and_per_source_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task1.jsonl"
            original = b"".join(_encoded(row) for row in self.rows[:2])
            path.write_bytes(original)

            rows, audit, audits_by_source = self._load(path)

            self.assertEqual(rows, self.rows[:2])
            self.assertEqual(audit.teacher_events, 2)
            self.assertEqual(audit.comparison_events, 2)
            self.assertEqual(audit.compared_token_positions, 5)
            self.assertEqual(audit.same_prefix_positions, 5)
            self.assertEqual(audit.source_counts, {"original_query": 2})
            self.assertEqual(audits_by_source["amc23"].teacher_events, 1)
            self.assertEqual(audits_by_source["aime24"].teacher_events, 1)
            self.assertNotIn("aime25", audits_by_source)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(self._backups(path), [])

    def test_valid_final_object_without_lf_is_retained_and_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task1.jsonl"
            original = _encoded(self.rows[0], newline=False)
            path.write_bytes(original)

            rows, audit, _ = self._load(path)

            self.assertEqual(rows, self.rows[:1])
            self.assertEqual(audit.teacher_events, 1)
            self.assertEqual(path.read_bytes(), _encoded(self.rows[0]))
            backups = self._backups(path)
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

    def test_invalid_unterminated_final_fragment_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task1.jsonl"
            original = _encoded(self.rows[0]) + b'{"query_id": "unterminated"'
            path.write_bytes(original)

            rows, audit, _ = self._load(path)

            self.assertEqual(rows, self.rows[:1])
            self.assertEqual(audit.teacher_events, 1)
            self.assertEqual(path.read_bytes(), _encoded(self.rows[0]))
            backups = self._backups(path)
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

            # Once normalized, recovery is idempotent and creates no new backup.
            self._load(path)
            self.assertEqual(self._backups(path), backups)

    def test_nonfinal_corruption_fails_closed_without_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task1.jsonl"
            original = (
                _encoded(self.rows[0])
                + b'{"query_id":\n'
                + _encoded(self.rows[1])
            )
            path.write_bytes(original)

            with self.assertRaisesRegex(ValueError, "only an unterminated final write"):
                self._load(path)

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(self._backups(path), [])

    def test_complete_nonobject_record_fails_closed_without_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task1.jsonl"
            original = b"[]\n"
            path.write_bytes(original)

            with self.assertRaisesRegex(ValueError, "not a JSON object"):
                self._load(path)

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(self._backups(path), [])

    def test_out_of_order_rows_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task1.jsonl"
            original = _encoded(self.rows[1]) + _encoded(self.rows[0])
            path.write_bytes(original)

            with self.assertRaisesRegex(ValueError, "not the exact dataset prefix"):
                self._load(path)

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(self._backups(path), [])

    def test_binding_drift_fails_closed(self):
        cases = (
            ("problem_sha256", "0" * 64, "problem hash disagrees"),
            (
                "proposal_training_sha256",
                "f" * 64,
                "proposal training binding disagrees",
            ),
        )
        for field, value, message in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "task1.jsonl"
                drifted = copy.deepcopy(self.rows[0])
                drifted[field] = value
                original = _encoded(drifted)
                path.write_bytes(original)

                with self.assertRaisesRegex(ValueError, message):
                    self._load(path)

                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(self._backups(path), [])


if __name__ == "__main__":
    unittest.main()
