"""Dependency-light tests for CSD boundaries and paper-facing metrics."""

import json
import tempfile
import unittest
from pathlib import Path

from src.clean_self_distill.io import load_query_records
from src.clean_self_distill.metrics import HindsightAudit, aggregate_teacher_metrics
from src.clean_self_distill.prompts import candidate_messages


class CleanSelfDistillTest(unittest.TestCase):
    def test_candidate_prompt_has_no_target_argument(self):
        target_secret = "TARGET_ENTITY_9173"
        card = {
            "domain": "algebra",
            "skills": ["linear equations"],
            "reasoning_operators": ["isolate a variable"],
            "difficulty": "medium",
            "constraints": [],
            "target_details_removed": True,
        }
        prompt = json.dumps(candidate_messages(card, 4))
        self.assertNotIn(target_secret, prompt)
        self.assertIn("linear equations", prompt)

    def test_proposer_loader_drops_targets(self):
        row = {
            "id": "aime-x",
            "problem": "Find x.",
            "answer": "9173",
            "solution": "secret solution",
            "source": "aime24",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            record = load_query_records(path, include_targets=False)[0]
        self.assertEqual(record["problem"], "Find x.")
        self.assertNotIn("answer", record)
        self.assertNotIn("solution", record)

    def test_hindsight_audit_detects_exposure_and_prefix_mismatch(self):
        audit = HindsightAudit()
        audit.record_teacher_context(["original_query", "target_answer"])
        audit.record_same_prefix([1, 2], [1, 3], positions=2, on_policy=True)
        metrics = audit.compute()
        self.assertEqual(metrics["hindsight/hindsight_exposure_rate"], 1.0)
        self.assertEqual(metrics["hindsight/context_parity_rate"], 0.0)
        self.assertEqual(metrics["hindsight/on_policy_same_prefix_rate"], 0.0)

    def test_hftg_and_fate(self):
        audit = HindsightAudit()
        audit.record_teacher_context(["original_query", "proposed_candidates"])
        audit.record_same_prefix([1, 2], [1, 2], positions=2)
        rows = [
            {
                "target_answer_nll_gain": 0.4,
                "specialization_seconds": 0.2,
                "base_correct": 0.0,
                "teacher_correct": 1.0,
            }
        ]
        metrics = aggregate_teacher_metrics(rows, audit)
        self.assertAlmostEqual(metrics["hindsight/hindsight_free_transfer_gain"], 0.4)
        self.assertAlmostEqual(metrics["speed/fast_adaptation_teacher_efficiency"], 2.0)


if __name__ == "__main__":
    unittest.main()
