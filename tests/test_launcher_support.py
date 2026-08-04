"""Focused regression tests for the Slurm shard validator."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.clean_self_distill.slurm.launcher_support import (
    cmd_repair_proposals,
    cmd_validate_shard,
)
from scripts.clean_self_distill.slurm.launcher_support import LauncherValidationError
from src.clean_self_distill.io import (
    compute_proposal_training_sha256,
    load_query_records,
)
from src.clean_self_distill.privileged import build_privileged_control_artifact


class LauncherSupportTest(unittest.TestCase):
    @staticmethod
    def _dataset(root: Path) -> tuple[Path, dict]:
        dataset = root / "dataset.jsonl"
        dataset.write_text(
            json.dumps(
                {
                    "id": "one",
                    "problem": "Find the value.",
                    "answer": "1",
                    "source": "amc23",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return dataset, load_query_records(dataset, include_targets=True)[0]

    def test_task2_accepts_the_task_marker_written_by_train_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            artifact = root / "task2.jsonl"
            artifact.write_text(
                json.dumps(
                    {
                        "query_id": record["query_id"],
                        "problem_sha256": record["problem_sha256"],
                        "source": record["source"],
                        "model": "model",
                        "model_revision": "revision",
                        "task": "task2_clean_distillation",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cmd_validate_shard(
                SimpleNamespace(
                    dataset=str(dataset),
                    max_samples=None,
                    num_shards=1,
                    shard_index=0,
                    kind="task2",
                    artifact=str(artifact),
                    model="model",
                    revision="revision",
                )
            )

    def test_task1_requires_answer_redacted_privileged_cot_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            control = build_privileged_control_artifact(
                record["problem"],
                record["answer"],
                json.dumps({"reasoning_steps": ["Use a reusable symbolic invariant."]}),
            )
            control["evaluated_model_prompt_sha256"] = "a" * 64
            row = {
                "query_id": record["query_id"],
                "problem_sha256": record["problem_sha256"],
                "source": record["source"],
                "model": "model",
                "model_revision": "revision",
                "stage": "task1_fast_teacher",
                "privileged_control_artifact": control,
                "privileged_hindsight_exposure_rate": 1.0,
                "privileged_context_prefix_parity": 0.0,
                "privileged_hindsight_free_score": 0.0,
            }
            artifact = root / "task1.jsonl"
            args = SimpleNamespace(
                dataset=str(dataset),
                max_samples=None,
                num_shards=1,
                shard_index=0,
                kind="task1",
                artifact=str(artifact),
                model="model",
                revision="revision",
            )
            artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
            cmd_validate_shard(args)

            row["privileged_control_artifact"][
                "advantage_text"
            ] = "The final answer is 1."
            artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                LauncherValidationError, "answer-redacted HER=1 CoT"
            ):
                cmd_validate_shard(args)

    def test_proposal_repair_normalizes_unterminated_valid_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposals.jsonl"
            path.write_bytes(b'{"query_id":"q1"}')
            cmd_repair_proposals(SimpleNamespace(path=str(path)))
            self.assertEqual(path.read_bytes(), b'{"query_id": "q1"}\n')
            self.assertEqual(len(list(path.parent.glob("*.corrupt.*"))), 1)

    def test_proposal_repair_drops_only_unterminated_syntax_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposals.jsonl"
            path.write_bytes(b'{"query_id":"q1"}\n{"query_id":')
            cmd_repair_proposals(SimpleNamespace(path=str(path)))
            self.assertEqual(path.read_bytes(), b'{"query_id": "q1"}\n')

    def test_proposal_repair_rejects_nonfinal_corruption_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposals.jsonl"
            original = b'{"query_id":\n{"query_id":"q2"}\n'
            path.write_bytes(original)
            with self.assertRaisesRegex(
                LauncherValidationError, "not an unterminated final write"
            ):
                cmd_repair_proposals(SimpleNamespace(path=str(path)))
            self.assertEqual(path.read_bytes(), original)

    def test_proposal_repair_rejects_complete_nonobject_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposals.jsonl"
            original = b"[]\n"
            path.write_bytes(original)
            with self.assertRaisesRegex(LauncherValidationError, "not an object"):
                cmd_repair_proposals(SimpleNamespace(path=str(path)))
            self.assertEqual(path.read_bytes(), original)

    def test_shard_done_marker_is_safe_under_nounset(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "clean_self_distill"
            / "slurm"
            / "run_shard.slurm"
        )
        match = re.search(
            r"(?ms)^csd_mark_done\(\) \{\n.*?^\}",
            launcher.read_text(encoding="utf-8"),
        )
        self.assertIsNotNone(match)
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "stage.done"
            environment = {
                **os.environ,
                "MARKER": str(marker),
                "SLURM_JOB_ID": "12345",
            }
            subprocess.run(
                [
                    "bash",
                    "-c",
                    "set -u\n" + match.group(0) + '\ncsd_mark_done "$MARKER"\n',
                ],
                check=True,
                env=environment,
            )
            self.assertIn("job_id=12345", marker.read_text(encoding="utf-8"))

    def test_requeue_watchdog_uses_valid_scontrol_syntax(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "clean_self_distill"
            / "slurm"
            / "run_shard.slurm"
        ).read_text(encoding="utf-8")
        self.assertNotIn('scontrol requeue Incomplete "$SLURM_JOB_ID"', launcher)
        self.assertIn('scontrol requeue "$SLURM_JOB_ID"', launcher)

    def test_proposal_accepts_explicit_empty_candidate_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            row = {
                **record,
                "schema_version": "clean-self-distill-proposals-v5",
                "model": "model",
                "model_revision": "revision",
                "skill_card": {"skills": ["reason abstractly"]},
                "specialization_candidates": [],
                "specialization_status": "insufficient_verified_candidates",
                "specialization_failure_reason": "0 candidates passed a minimum gate of 1",
                "specialization_no_op": True,
                "candidate_count": 0,
                "requested_candidate_count": 1,
                "minimum_candidate_count": 1,
            }
            row["proposal_training_sha256"] = compute_proposal_training_sha256(row)
            artifact = root / "proposal.jsonl"
            artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                dataset=str(dataset),
                max_samples=None,
                num_shards=1,
                shard_index=0,
                kind="proposal",
                artifact=str(artifact),
                model="model",
                revision="revision",
            )
            cmd_validate_shard(args)

            row.update(
                {
                    "specialization_status": "ready",
                    "specialization_failure_reason": "",
                    "specialization_no_op": False,
                }
            )
            # The shared producer refuses to hash this impossible state, so use
            # a syntactically valid digest to exercise the launcher's fail-closed gate.
            row["proposal_training_sha256"] = "0" * 64
            artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                LauncherValidationError, "ready specialization requires candidates"
            ):
                cmd_validate_shard(args)


if __name__ == "__main__":
    unittest.main()
