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
    _HORIZON_WINDOWS = ((0, 512), (512, 1024), (1024, 2048), (2048, 4096))

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

    @classmethod
    def _horizon_windows(cls, prefix_tokens: int) -> list[dict]:
        rows = []
        for start, end in cls._HORIZON_WINDOWS:
            token_count = max(min(end, prefix_tokens) - start, 0)
            rows.append(
                {
                    "start_token": start,
                    "end_token": end,
                    "token_count": token_count,
                    "measurement_point": "pre_update",
                    "pre_update_mean_teacher_student_kl": (
                        0.1 if token_count else None
                    ),
                    "pre_update_teacher_student_top1_agreement": (
                        0.9 if token_count else None
                    ),
                    "pre_update_mean_teacher_base_ridge_shift_l2": (
                        0.2 if token_count else None
                    ),
                }
            )
        return rows

    @classmethod
    def _task2_step(
        cls,
        *,
        prefix_tokens: int = 4096,
        prefix_cap: int = 4096,
        minimum_tokens: int = 4096,
        trajectory_complete: bool = False,
    ) -> dict:
        return {
            "prefix_tokens": prefix_tokens,
            "prefix_truncated": (
                prefix_tokens == prefix_cap and not trajectory_complete
            ),
            "trajectory_complete": trajectory_complete,
            "prefix_natural_completion": trajectory_complete,
            "long_horizon_threshold_reached": (
                prefix_tokens >= minimum_tokens
            ),
            "long_horizon_qualified": (
                trajectory_complete or prefix_tokens >= minimum_tokens
            ),
            "horizon_windows": cls._horizon_windows(prefix_tokens),
        }

    @classmethod
    def _task2_row(cls, record: dict, *, steps: list[dict] | None = None) -> dict:
        if steps is None:
            steps = [cls._task2_step()]
        return {
            "query_id": record["query_id"],
            "problem_sha256": record["problem_sha256"],
            "source": record["source"],
            "model": "model",
            "model_revision": "revision",
            "task": "task2_clean_distillation",
            "specialization_status": "ready",
            "specialization_no_op": False,
            "distillation_steps_completed": len(steps),
            "distillation_trace": steps,
            "distillation_config": {
                "steps": len(steps),
                "prefix_max_new_tokens": 4096,
                "long_horizon_min_tokens": 4096,
            },
        }

    @staticmethod
    def _validate_task2(dataset: Path, artifact: Path) -> None:
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

    def test_task2_accepts_a_long_horizon_trace_written_by_train_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            artifact = root / "task2.jsonl"
            artifact.write_text(
                json.dumps(self._task2_row(record)) + "\n", encoding="utf-8"
            )
            self._validate_task2(dataset, artifact)

    def test_task2_accepts_natural_completion_below_the_horizon_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            artifact = root / "task2.jsonl"
            step = self._task2_step(
                prefix_tokens=777,
                trajectory_complete=True,
            )
            artifact.write_text(
                json.dumps(self._task2_row(record, steps=[step])) + "\n",
                encoding="utf-8",
            )
            self._validate_task2(dataset, artifact)

    def test_task2_rejects_legacy_512_token_distillation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            artifact = root / "task2.jsonl"
            step = self._task2_step(
                prefix_tokens=512,
                prefix_cap=512,
                minimum_tokens=512,
            )
            row = self._task2_row(record, steps=[step])
            row["distillation_config"].update(
                {
                    "prefix_max_new_tokens": 512,
                    "long_horizon_min_tokens": 512,
                }
            )
            artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                LauncherValidationError, "long-horizon minimum must be at least 4096"
            ):
                self._validate_task2(dataset, artifact)

    def test_task2_requires_explicit_long_horizon_config_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            artifact = root / "task2.jsonl"
            for key in ("prefix_max_new_tokens", "long_horizon_min_tokens"):
                with self.subTest(key=key):
                    row = self._task2_row(record)
                    del row["distillation_config"][key]
                    artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        LauncherValidationError,
                        rf"distillation_config\.{key} must be a nonnegative integer",
                    ):
                        self._validate_task2(dataset, artifact)

    def test_task2_requires_exact_boolean_trajectory_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            artifact = root / "task2.jsonl"
            for key in (
                "prefix_truncated",
                "trajectory_complete",
                "long_horizon_qualified",
            ):
                with self.subTest(key=key):
                    row = self._task2_row(record)
                    row["distillation_trace"][0][key] = 1
                    artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        LauncherValidationError,
                        rf"{key} must be an explicit JSON boolean",
                    ):
                        self._validate_task2(dataset, artifact)

    def test_task2_rejects_incorrect_qualification_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            artifact = root / "task2.jsonl"
            row = self._task2_row(record)
            row["distillation_trace"][0]["long_horizon_qualified"] = False
            artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                LauncherValidationError,
                "must equal natural completion OR threshold attainment",
            ):
                self._validate_task2(dataset, artifact)

    def test_task2_rejects_inconsistent_optional_horizon_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            artifact = root / "task2.jsonl"
            mutations = (
                ("prefix_natural_completion", True, "must equal trajectory_complete"),
                (
                    "long_horizon_threshold_reached",
                    False,
                    "disagrees with prefix_tokens and the threshold",
                ),
            )
            for key, value, message in mutations:
                with self.subTest(key=key):
                    row = self._task2_row(record)
                    row["distillation_trace"][0][key] = value
                    artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(LauncherValidationError, message):
                        self._validate_task2(dataset, artifact)

    def test_task2_rejects_out_of_range_horizon_window_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            artifact = root / "task2.jsonl"
            mutations = (
                ("pre_update_mean_teacher_student_kl", -0.01, "nonnegative"),
                (
                    "pre_update_teacher_student_top1_agreement",
                    1.01,
                    r"must be in \[0, 1\]",
                ),
                (
                    "pre_update_mean_teacher_base_ridge_shift_l2",
                    -0.01,
                    "must be nonnegative",
                ),
            )
            for key, value, message in mutations:
                with self.subTest(key=key):
                    row = self._task2_row(record)
                    row["distillation_trace"][0]["horizon_windows"][0][key] = value
                    artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(LauncherValidationError, message):
                        self._validate_task2(dataset, artifact)

    def test_task2_requires_at_least_one_qualified_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            artifact = root / "task2.jsonl"
            step = self._task2_step(prefix_tokens=2048)
            row = self._task2_row(record, steps=[step])
            artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                LauncherValidationError, "has no long-horizon-qualified trajectory"
            ):
                self._validate_task2(dataset, artifact)

    def test_task2_no_op_requires_empty_trace_and_zero_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, record = self._dataset(root)
            artifact = root / "task2.jsonl"
            row = self._task2_row(record, steps=[])
            row.update(
                {
                    "specialization_status": "insufficient_verified_candidates",
                    "specialization_no_op": True,
                }
            )
            artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self._validate_task2(dataset, artifact)

            row["distillation_trace"] = [self._task2_step()]
            artifact.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                LauncherValidationError,
                "no-op requires an empty trace and zero completed steps",
            ):
                self._validate_task2(dataset, artifact)

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

    def test_formal_chain_pins_long_horizon_and_posthoc_report(self):
        root = Path(__file__).resolve().parents[1]
        environment = (
            root / "configs" / "clean_self_distill" / "b200_poc.env"
        ).read_text(encoding="utf-8")
        shard = (
            root
            / "scripts"
            / "clean_self_distill"
            / "slurm"
            / "run_shard.slurm"
        ).read_text(encoding="utf-8")
        merger = (
            root
            / "scripts"
            / "clean_self_distill"
            / "slurm"
            / "merge_report.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn("CSD_TRAIN_MAX_NEW_TOKENS=4096", environment)
        self.assertIn("CSD_LONG_HORIZON_MIN_PREFIX_TOKENS=4096", environment)
        self.assertIn("CSD_TARGET_HORIZON_SPLIT_TOKENS=2048", environment)
        self.assertIn(
            '--train-max-new-tokens "$CSD_TRAIN_MAX_NEW_TOKENS"', shard
        )
        self.assertIn(
            '--long-horizon-min-prefix-tokens '
            '"$CSD_LONG_HORIZON_MIN_PREFIX_TOKENS"',
            shard,
        )
        self.assertIn("report_horizon_metrics.py", merger)
        self.assertIn('--report-dir "$CSD_REPORT_ATTEMPT"', merger)

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
