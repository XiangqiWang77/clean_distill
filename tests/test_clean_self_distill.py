"""Dependency-light tests for CSD boundaries and paper-facing metrics."""

import json
import tempfile
import unittest
from pathlib import Path

from src.clean_self_distill.io import (
    compute_proposal_training_sha256,
    load_proposal_map,
    load_query_records,
    stable_hash,
)
from src.clean_self_distill.metrics import HindsightAudit, aggregate_teacher_metrics
from src.clean_self_distill.prompts import candidate_messages
from src.clean_self_distill.propose import (
    sanitize_skill_card,
    skill_card_disjoint_audit,
    target_disjoint_audit,
)


def _bound_proposal(query_id: str = "q", problem: str = "p") -> dict:
    row = {
        "query_id": query_id,
        "problem": problem,
        "problem_sha256": stable_hash(problem, 64),
        "source": "amc23",
        "skill_card": {
            "domain": "algebra",
            "skills": ["symbolic manipulation"],
            "reasoning_operators": ["substitute"],
            "difficulty": "medium",
            "constraints": [],
            "target_details_removed": True,
        },
        "specialization_candidates": [
            {
                "candidate_id": "c00",
                "problem": "An independent exercise.",
                "skill_tags": ["algebra"],
                "solution": "A verified derivation.",
                "final_answer": "ok",
                "verifier_valid": True,
                "verifier_accepted": True,
                "verifier_reason": "valid",
                "target_disjoint_audit": {"safe": True},
            }
        ],
    }
    row["proposal_training_sha256"] = compute_proposal_training_sha256(row)
    return row


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

    def test_skill_card_sanitizer_removes_target_symbols_and_expressions(self):
        problem = r"For triangle ABC, if $x+y=9173$, choose option C."
        malicious = {
            "domain": "geometry",
            "skills": [r"Use triangle ABC and the equation $x+y=9173$"],
            "reasoning_operators": ["select C"],
            "difficulty": "hard",
            "constraints": [],
        }
        clean, redactions = sanitize_skill_card(malicious, problem)
        serialized = json.dumps(clean)
        for secret in ("ABC", "9173", "x+y", "select C"):
            self.assertNotIn(secret, serialized)
        self.assertTrue(redactions)
        self.assertTrue(skill_card_disjoint_audit(problem, clean)["safe"])

    def test_skill_card_rejects_number_words_and_direct_answer_cues(self):
        problem = "Determine the requested quantity."
        malicious = {
            "domain": "arithmetic",
            "skills": ["combine terms"],
            "reasoning_operators": ["simplify"],
            "difficulty": "medium",
            "constraints": [
                "The final answer is forty-two.",
                r"Do not emit \boxed{forty-two}.",
                42,
                True,
            ],
        }
        unsafe = skill_card_disjoint_audit(problem, malicious)
        self.assertFalse(unsafe["safe"])
        self.assertEqual(unsafe["english_number_words"], ["forty", "two"])
        self.assertIn("final answer", unsafe["direct_answer_cues"])
        self.assertTrue(
            any(cue.startswith(r"\boxed") for cue in unsafe["direct_answer_cues"])
        )

        clean, redactions = sanitize_skill_card(malicious, problem)
        serialized = json.dumps(clean).lower()
        self.assertNotIn("forty", serialized)
        self.assertNotIn("final answer", serialized)
        self.assertNotIn(r"\boxed", serialized)
        self.assertFalse(any(ord(character) < 32 for character in serialized))
        self.assertEqual(clean["constraints"][-2], "redacted number")
        self.assertIs(clean["constraints"][-1], True)
        self.assertIn("<english-number-word>", redactions)
        self.assertIn("<direct-answer-cue>", redactions)
        self.assertTrue(skill_card_disjoint_audit(problem, clean)["safe"])

    def test_target_disjoint_audit_is_case_and_number_format_invariant(self):
        problem = "Alexandria labels a total of 1,000 objects in triangle ABC."
        candidate = "alexandria labels 1000 different objects in triangle abc."
        audit = target_disjoint_audit(problem, candidate)
        self.assertEqual(audit["shared_target_numbers"], ["1000"])
        self.assertEqual(audit["shared_target_entities"], ["abc", "alexandria"])
        self.assertEqual(audit["literal_overlap_count"], 3.0)

        generic = target_disjoint_audit(
            "Find the remainder when 7 is divided.",
            "Find the remainder when 11 is divided by 4.",
        )
        self.assertEqual(generic["shared_target_entities"], [])
        self.assertEqual(generic["literal_overlap_count"], 0.0)

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

    def test_query_ids_are_namespaced_across_combined_sources(self):
        rows = [
            {"problem": "AMC problem", "source": "amc23", "extra_info": {"index": 0}},
            {"problem": "AIME problem", "source": "aime24", "extra_info": {"index": 0}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            records = load_query_records(path, include_targets=False)
        self.assertEqual(len({record["query_id"] for record in records}), 2)
        self.assertTrue(records[0]["query_id"].startswith("amc23-"))
        self.assertTrue(records[1]["query_id"].startswith("aime24-"))

    def test_duplicate_proposal_ids_are_rejected(self):
        row = _bound_proposal(query_id="duplicate")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposals.jsonl"
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate proposal query_id"):
                load_proposal_map(path)

    def test_proposal_training_hash_rejects_candidate_tampering(self):
        row = _bound_proposal()
        reordered = {
            key: row[key]
            for key in reversed(list(row))
        }
        self.assertEqual(
            compute_proposal_training_sha256(reordered),
            row["proposal_training_sha256"],
        )
        tampered = json.loads(json.dumps(row))
        tampered["specialization_candidates"][0]["solution"] = "altered"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposals.jsonl"
            path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "proposal_training_sha256"):
                load_proposal_map(path)

    def test_proposal_loader_requires_complete_problem_source_and_training_binding(self):
        row = _bound_proposal()
        for missing in ("problem", "source", "proposal_training_sha256"):
            malformed = dict(row)
            malformed.pop(missing)
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "proposals.jsonl"
                path.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_proposal_map(path)

    def test_hindsight_audit_detects_exposure_and_prefix_mismatch(self):
        audit = HindsightAudit()
        audit.record_teacher_context(["original_query", "target_answer"])
        audit.record_same_prefix([1, 2], [1, 3], positions=2, on_policy=True)
        metrics = audit.compute()
        self.assertEqual(metrics["hindsight/hindsight_exposure_rate"], 1.0)
        self.assertEqual(metrics["hindsight/context_parity_rate"], 0.0)
        self.assertEqual(metrics["hindsight/on_policy_same_prefix_rate"], 0.0)

    def test_hindsight_cpp_uses_position_counts_and_preserves_raw_counts(self):
        audit = HindsightAudit()
        audit.record_teacher_context(["original_query", "proposed_candidates"])
        audit.record_same_prefix([1], [1], positions=2)
        audit.record_same_prefix([1], [2], positions=8, on_policy=True)
        metrics = audit.compute()
        self.assertEqual(metrics["hindsight/comparison_events"], 2)
        self.assertEqual(metrics["hindsight/context_equal_events"], 1)
        self.assertEqual(metrics["hindsight/compared_token_positions"], 10)
        self.assertEqual(metrics["hindsight/same_prefix_positions"], 2)
        self.assertEqual(metrics["hindsight/context_parity_rate"], 0.2)
        self.assertEqual(
            metrics["hindsight/source_counts"],
            {"original_query": 1, "proposed_candidates": 1},
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            audit.record_same_prefix([1], [1], positions=-1)

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
