"""Focused tests for the v5 contrastive proposal contract."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from src.clean_self_distill.io import compute_proposal_training_sha256
from src.clean_self_distill.propose import (
    MODEL_JSON_PARSER_VERSION,
    _contrastive_answer_audit,
    _parse_correct_trajectory_response,
    _parse_model_json_object,
    _parse_wrong_trajectory_response,
    _validate_error_frontier,
    propose_for_query,
)


class _ScriptedGenerator:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []
        self.tokenizer = SimpleNamespace(chat_template=None)
        self.enable_thinking = False

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("scripted generator received an unexpected call")
        return self.responses.pop(0)

    def counters(self) -> dict[str, float]:
        return {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "generation_seconds": 0.0,
        }


def _skill_card() -> str:
    return json.dumps(
        {
            "domain": "arithmetic",
            "skills": ["multiply independent quantities"],
            "reasoning_operators": ["combine factors"],
            "failure_modes": ["replace multiplication with addition"],
            "difficulty": "easy",
            "constraints": [],
        }
    )


def _candidate_batch() -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "candidate_id": "fresh",
                    "candidate_type": "failure_focused",
                    "problem": "Compute the product of 8 and 9.",
                    "skill_tags": ["arithmetic"],
                }
            ]
        }
    )


def _correct() -> str:
    return (
        "<FINAL_ANSWER>72</FINAL_ANSWER>"
        "<CORRECT_STEP><STEP_INDEX>0</STEP_INDEX>"
        "<STEP_TEXT>Multiply 8 by 9 to obtain 72.</STEP_TEXT></CORRECT_STEP>"
    )


def _wrong(answer: str = "17") -> str:
    return (
        f"<WRONG_FINAL_ANSWER>{answer}</WRONG_FINAL_ANSWER>"
        "<WRONG_STEP><STEP_INDEX>0</STEP_INDEX>"
        f"<STEP_TEXT>Add 8 and 9 to obtain {answer}.</STEP_TEXT></WRONG_STEP>"
    )


def _frontier(*, valid: bool = True) -> str:
    return json.dumps(
        {
            "wrong_trajectory_incorrect": valid,
            "prefix_before_error_valid": valid,
            "wrong_step_invalid": valid,
            "corrective_action_valid": valid,
            "wrong_step_index": 0,
            "error_explanation": "Addition is not the requested operation.",
            "corrective_action": "Replace addition with multiplication.",
        }
    )


class ProposalV5Test(unittest.TestCase):
    def test_rejects_observed_placeholder_and_noncontrastive_outputs(self):
        placeholder = _correct().replace("72", "checkable final answer")
        with self.assertRaisesRegex(ValueError, "placeholder"):
            _parse_correct_trajectory_response(placeholder)
        with self.assertRaisesRegex(ValueError, "identical"):
            _contrastive_answer_audit("72", r"\boxed{72}")

        wrong = [{"step_index": 0, "text": "Add the factors."}]
        frontier = json.loads(_frontier())
        frontier["error_explanation"] = "The selected step is already correct."
        frontier["corrective_action"] = "No correction is needed."
        with self.assertRaisesRegex(ValueError, "contradicts"):
            _validate_error_frontier(frontier, wrong)

    def test_tagged_parser_and_tolerant_json_preserve_math_text(self):
        tagged = _parse_correct_trajectory_response(_correct())
        self.assertEqual(tagged["final_answer"], "72")
        self.assertEqual(tagged["correct_trajectory"][0]["step_index"], 0)

        malformed_json = """```json
{"correct_trajectory":[{"step_index":0,"text":"Use \\frac{8}{9}.
Then simplify."}],"final_answer":"\\frac{8}{9}"}
```"""
        repaired = _parse_correct_trajectory_response(malformed_json)
        self.assertIn(r"\frac{8}{9}", repaired["correct_trajectory"][0]["text"])
        self.assertEqual(repaired["final_answer"], r"\frac{8}{9}")

    def test_tolerant_json_repairs_tex_punctuation_without_touching_json_quotes(self):
        malformed_batch = r'''{"candidates":[{"candidate_id":"c0","candidate_type":"atomic","problem":"Let $A \subseteq \mathbb{R}^2$, $A=\{0\}$, and use \! or \, spacing.","skill_tags":["say \"set\""]}]}'''
        parsed = _parse_model_json_object(malformed_batch)
        candidate = parsed["candidates"][0]
        self.assertEqual(
            candidate["problem"],
            r"Let $A \subseteq \mathbb{R}^2$, $A=\{0\}$, and use \! or \, spacing.",
        )
        self.assertEqual(candidate["skill_tags"], ['say "set"'])

    def test_tolerant_json_keeps_already_escaped_tex_backslashes(self):
        valid = json.dumps(
            {
                "problem": r"Compute \frac{1}{2} and inspect \{0\}.",
                "quoted": 'say "hello"',
            }
        )
        self.assertEqual(_parse_model_json_object(valid), json.loads(valid))

    def test_tagged_wrong_trajectory_allows_whitespace_in_closing_tags(self):
        # Real proposer output used this otherwise well-formed tag spelling.
        raw = (
            "<WRONG_FINAL_ANSWER>1</ WRONG_FINAL_ANSWER>"
            "<WRONG_STEP><STEP_INDEX>0</ STEP_INDEX>"
            "<STEP_TEXT>Assume the first repeated value proves a full period."
            "</ STEP_TEXT></ WRONG_STEP>"
        )
        parsed = _parse_wrong_trajectory_response(raw)
        self.assertEqual(parsed["wrong_final_answer"], "1")
        self.assertEqual(
            parsed["wrong_trajectory"],
            [
                {
                    "step_index": 0,
                    "text": "Assume the first repeated value proves a full period.",
                }
            ],
        )

    def test_bare_line_tag_trajectory_preserves_all_explicit_content(self):
        # Qwen3 sometimes drops only the XML brackets while preserving the
        # requested line-delimited schema. Parsing this form avoids discarding
        # a complete attempt without inferring any mathematical field.
        raw = """WRONG_FINAL_ANSWER
17

WRONG_STEP
STEP_INDEX: 0
STEP_TEXT: Add 8 and 9 instead of multiplying them.

WRONG_STEP
STEP_INDEX
1
STEP_TEXT
Conclude that the product is 17.
"""
        parsed = _parse_wrong_trajectory_response(raw)
        self.assertEqual(parsed["wrong_final_answer"], "17")
        self.assertEqual(
            parsed["wrong_trajectory"],
            [
                {
                    "step_index": 0,
                    "text": "Add 8 and 9 instead of multiplying them.",
                },
                {"step_index": 1, "text": "Conclude that the product is 17."},
            ],
        )

    def test_bare_line_tag_trajectory_rejects_missing_explicit_fields(self):
        missing_text = """WRONG_FINAL_ANSWER
17
WRONG_STEP
STEP_INDEX
0
"""
        with self.assertRaisesRegex(ValueError, "Malformed bare WRONG_STEP"):
            _parse_wrong_trajectory_response(missing_text)

    def test_tagged_wrong_trajectory_does_not_infer_missing_tags(self):
        missing_step_close = (
            "<WRONG_FINAL_ANSWER>1</WRONG_FINAL_ANSWER>"
            "<WRONG_STEP><STEP_INDEX>0</STEP_INDEX>"
            "<STEP_TEXT>Assume a repeated value proves a period.</WRONG_STEP>"
        )
        with self.assertRaisesRegex(ValueError, "Malformed WRONG_STEP block"):
            _parse_wrong_trajectory_response(missing_step_close)

        missing_answer = (
            "<WRONG_STEP><STEP_INDEX>0</STEP_INDEX>"
            "<STEP_TEXT>Assume a repeated value proves a period.</STEP_TEXT>"
            "</WRONG_STEP>"
        )
        with self.assertRaisesRegex(ValueError, "WRONG_FINAL_ANSWER"):
            _parse_wrong_trajectory_response(missing_answer)

    def test_v5_candidate_binds_two_trajectories_and_verified_frontier(self):
        proposer = _ScriptedGenerator([_skill_card(), _candidate_batch()])
        solver = _ScriptedGenerator([_correct(), _wrong()])
        verifier = _ScriptedGenerator(
            [
                json.dumps({"valid": True, "reason": "The product is correct."}),
                _frontier(),
            ]
        )
        row = propose_for_query(
            {
                "query_id": "q-v5",
                "problem": "Determine a quantity associated with 9173.",
                "source": "aime24",
            },
            proposer,
            solver,
            verifier,
            num_candidates=1,
            proposal_oversample=0,
            max_rounds=1,
            min_accepted_candidates=1,
            max_literal_overlap=0.0,
            max_fourgram_overlap=0.05,
            accept_verifier_corrections=False,
            stage_max_attempts=2,
        )

        self.assertEqual(row["schema_version"], "clean-self-distill-proposals-v5")
        self.assertEqual(row["model_json_parser_version"], MODEL_JSON_PARSER_VERSION)
        self.assertEqual(row["specialization_status"], "ready")
        self.assertEqual(row["candidate_count"], 1)
        candidate = row["specialization_candidates"][0]
        self.assertEqual(candidate["candidate_type"], "failure_focused")
        self.assertEqual(candidate["correct_trajectory"][0]["step_index"], 0)
        self.assertEqual(candidate["wrong_trajectory"][0]["step_index"], 0)
        self.assertEqual(candidate["wrong_final_answer"], "17")
        self.assertEqual(candidate["error_frontier"]["wrong_step_index"], 0)
        self.assertEqual(
            candidate["error_frontier"]["wrong_step_text"],
            candidate["wrong_trajectory"][0]["text"],
        )
        self.assertTrue(candidate["error_frontier"]["verifier_valid"])
        self.assertIn("Multiply 8 by 9", candidate["solution"])
        self.assertEqual(candidate["final_answer"], "72")
        self.assertTrue(
            all(
                audit["safe"]
                for audit in candidate["artifact_target_disjoint_audits"].values()
            )
        )
        self.assertFalse(
            candidate["generation_provenance"]["wrong_trajectory"][
                "correct_trajectory_exposed"
            ]
        )
        self.assertEqual(
            row["proposal_training_sha256"], compute_proposal_training_sha256(row)
        )

        wrong_prompt = json.dumps(solver.calls[1])
        self.assertNotIn("9173", wrong_prompt)
        self.assertNotIn("Multiply 8 by 9 to obtain 72", wrong_prompt)
        self.assertIn("replace multiplication with addition", wrong_prompt)
        self.assertNotIn("9173", json.dumps(verifier.calls))

    def test_bounded_retries_cover_parse_and_semantic_frontier_failures(self):
        proposer = _ScriptedGenerator([_skill_card(), _candidate_batch()])
        solver = _ScriptedGenerator(
            [
                "not parseable",
                _correct(),
                _wrong("18"),
                _wrong("17"),
            ]
        )
        verifier = _ScriptedGenerator(
            [
                json.dumps({"valid": True, "reason": "verified"}),
                _frontier(valid=False),
                _frontier(valid=True),
            ]
        )
        row = propose_for_query(
            {
                "query_id": "q-v5-retry",
                "problem": "Determine a quantity associated with 9173.",
                "source": "aime24",
            },
            proposer,
            solver,
            verifier,
            num_candidates=1,
            proposal_oversample=0,
            max_rounds=1,
            min_accepted_candidates=1,
            max_literal_overlap=0.0,
            max_fourgram_overlap=0.05,
            accept_verifier_corrections=False,
            stage_max_attempts=2,
        )

        self.assertEqual(row["candidate_count"], 1)
        calls = row["cost_audit"]["generation_calls_by_stage"]
        self.assertEqual(calls["correct_trajectory"], 2)
        self.assertEqual(calls["wrong_trajectory"], 2)
        self.assertEqual(calls["error_frontier_verifier"], 2)
        attempt = row["candidate_attempts"][0]
        self.assertFalse(attempt["correct_trajectory_attempts"][0]["parsed"])
        self.assertFalse(attempt["wrong_trajectory_attempts"][0]["accepted"])
        self.assertTrue(attempt["wrong_trajectory_attempts"][1]["accepted"])
        self.assertLessEqual(len(attempt["correct_trajectory_attempts"]), 2)
        self.assertLessEqual(len(attempt["wrong_trajectory_attempts"]), 2)


if __name__ == "__main__":
    unittest.main()
