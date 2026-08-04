"""Unit tests for the answer-redacted privileged reasoning control."""

from __future__ import annotations

import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from src.clean_self_distill.privileged import (
    PRIVILEGED_CONTEXT_SCHEMA_VERSION,
    PRIVILEGED_CONTROL_MODE,
    PrivilegedRedactionError,
    audit_answer_redaction,
    build_privileged_control_artifact,
    build_privileged_cot_generation_messages,
    build_privileged_evaluation_problem,
    build_privileged_evaluation_prompt,
    sanitize_privileged_advantage_text,
)
from src.clean_self_distill.train_eval import _generate_privileged_control_artifact


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PrivilegedControlTest(unittest.TestCase):
    def test_train_eval_uses_private_answer_only_for_cot_construction(self):
        tokenizer = SimpleNamespace(chat_template=None, eos_token_id=99)
        raw = json.dumps(
            {
                "reasoning_steps": [
                    "Apply the invariant to reduce the cases.",
                    "Stop before evaluating the remaining symbolic expression.",
                ]
            }
        )
        args = SimpleNamespace(
            privileged_cot_max_attempts=2,
            privileged_cot_max_new_tokens=32,
            privileged_cot_temperature=0.3,
            top_p=0.95,
            top_k=20,
        )
        with mock.patch(
            "src.clean_self_distill.train_eval.generate_response",
            return_value=(raw, torch.tensor([[1, 2]]), torch.tensor([[3, 99]])),
        ) as generated:
            artifact, metrics = _generate_privileged_control_artifact(
                object(),
                tokenizer,
                {"query_id": "q", "problem": "Find a residue.", "answer": "42"},
                args,
                seed=7,
            )

        private_prompt = generated.call_args.kwargs["prompt_override"]
        evaluated_prompt = metrics["privileged_evaluated_model_prompt"]
        self.assertIn("42", private_prompt)
        self.assertNotIn("42", artifact["advantage_text"])
        self.assertNotIn("42", artifact["evaluation_problem"])
        self.assertNotIn("42", evaluated_prompt)
        self.assertTrue(
            artifact["context_provenance"]["construction_used_target_answer"]
        )
        self.assertFalse(
            artifact["context_provenance"]["literal_target_answer_in_advantage_text"]
        )

    def test_generation_prompt_is_explicitly_private_and_answer_conditioned(self):
        messages = build_privileged_cot_generation_messages(
            "Find the requested residue.", "042"
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        rendered = json.dumps(messages)
        self.assertIn("Find the requested residue.", rendered)
        self.assertIn("042", rendered)
        self.assertIn("construction signal only", rendered)
        self.assertIn("never repeat", rendered)
        self.assertIn("reasoning_steps", rendered)

    def test_structured_output_keeps_only_reasoning_and_removes_answer_fields(self):
        raw = json.dumps(
            {
                "reasoning_steps": [
                    "Factor the expression into coprime components.",
                    "Apply the remainder theorem to each component.",
                ],
                "final_answer": "042",
                "confidence": 1.0,
            }
        )
        sanitized, audit = sanitize_privileged_advantage_text(raw, "42")

        self.assertIn("Factor the expression", sanitized)
        self.assertIn("Apply the remainder theorem", sanitized)
        self.assertNotIn("042", sanitized)
        self.assertEqual(
            audit["structured_fields_removed"], ["final_answer", "confidence"]
        )
        self.assertTrue(audit["safe"])
        self.assertTrue(audit["post_audit"]["safe"])

    def test_structured_output_without_reasoning_fails_closed(self):
        with self.assertRaisesRegex(PrivilegedRedactionError, "reasoning field"):
            sanitize_privileged_advantage_text(
                json.dumps({"final_answer": "42", "confidence": 1.0}), "42"
            )

    def test_removes_boxed_and_direct_answer_declarations(self):
        raw = (
            "First reduce the recurrence modulo the period.\n"
            r"The final answer is \boxed{042}."
            "\n"
            "Check the invariant before concluding."
        )
        sanitized, audit = sanitize_privileged_advantage_text(raw, "42")

        self.assertEqual(
            sanitized,
            "First reduce the recurrence modulo the period.\n"
            "Check the invariant before concluding.",
        )
        self.assertNotIn("boxed", sanitized.lower())
        self.assertGreaterEqual(audit["boxed_constructs_removed"], 1)
        self.assertGreaterEqual(
            audit["removed_reason_counts"].get("direct_answer_declaration", 0), 1
        )

    def test_removes_numeric_and_number_word_equivalents(self):
        spellings = ("042", "42.0", "4.2e1", "84/2", r"\frac{84}{2}", "forty-two")
        for spelling in spellings:
            with self.subTest(spelling=spelling):
                raw = (
                    "Use a parity split before evaluating the last expression.\n"
                    f"An intermediate line exposes {spelling}.\n"
                    "Retain the symbolic relation and stop before evaluation."
                )
                sanitized, audit = sanitize_privileged_advantage_text(raw, "42")
                self.assertNotIn("intermediate line", sanitized.lower())
                self.assertTrue(audit["safe"])
                self.assertEqual(
                    audit_answer_redaction(sanitized, "42")["answer_mention_count"], 0
                )

    def test_fraction_decimal_and_percent_equivalence_are_detected(self):
        cases = (
            (r"The quantity becomes \frac{1}{2}.", "0.5"),
            ("The quantity becomes 0.5.", r"\frac{1}{2}"),
            ("The quantity becomes 50%.", "0.5"),
        )
        for text, answer in cases:
            with self.subTest(text=text, answer=answer):
                audit = audit_answer_redaction(text, answer)
                self.assertFalse(audit["safe"])
                self.assertGreater(audit["answer_mention_count"], 0)

    def test_unicode_format_controls_cannot_hide_the_answer(self):
        raw = "Use modular periodicity.\nThe value 4\u200b2 appears here.\nStop symbolically."
        sanitized, audit = sanitize_privileged_advantage_text(raw, "42")

        self.assertNotIn("appears here", sanitized)
        self.assertTrue(audit["safe"])

    def test_tex_spacing_arithmetic_case_and_answer_wrappers_cannot_hide_answer(self):
        cases = (
            (r"Use the identity. Therefore, $6\cdot 7$.", "42"),
            (r"Use place value to obtain $4\!2$.", "42"),
            ("The surviving multiple-choice label is c.", "C"),
            ("The terminal value is 42.", r"\(42\)"),
        )
        for raw, answer in cases:
            with self.subTest(raw=raw, answer=answer):
                sanitized, audit = sanitize_privileged_advantage_text(
                    "Set up the reusable method.\n"
                    + raw
                    + "\nStop before the conclusion.",
                    answer,
                )
                self.assertNotIn("terminal", sanitized.casefold())
                self.assertNotIn("surviving", sanitized.casefold())
                self.assertNotIn("place value", sanitized.casefold())
                self.assertNotIn("therefore", sanitized.casefold())
                self.assertTrue(audit["safe"])
                self.assertTrue(audit_answer_redaction(sanitized, answer)["safe"])

    def test_direct_declaration_is_removed_even_for_encoded_expression(self):
        raw = (
            "Set up the recurrence.\n"
            r"The final answer is 6\cdot 7."
            "\n"
            "Stop after describing how to evaluate it."
        )
        sanitized, audit = sanitize_privileged_advantage_text(raw, "42")

        self.assertNotIn("6\\cdot 7", sanitized)
        self.assertGreaterEqual(
            audit["removed_reason_counts"].get("direct_answer_declaration", 0), 1
        )

    def test_non_numeric_multiple_choice_answer_is_removed(self):
        raw = "Compare the cases.\nThe correct answer is C.\nUse the surviving case."
        sanitized, audit = sanitize_privileged_advantage_text(raw, "C")

        self.assertEqual(sanitized, "Compare the cases.\nUse the surviving case.")
        self.assertTrue(audit["safe"])

    def test_redaction_fails_closed_when_no_reasoning_remains(self):
        with self.assertRaisesRegex(
            PrivilegedRedactionError, "removed the entire"
        ) as caught:
            sanitize_privileged_advantage_text(r"Final answer: \boxed{42}.", "42")

        self.assertFalse(caught.exception.audit["safe"])
        self.assertEqual(caught.exception.audit["post_redaction_characters"], 0)

    def test_evaluation_prompt_requires_safe_digest_bound_text(self):
        sanitized, audit = sanitize_privileged_advantage_text(
            "Use a telescoping sum.\nStop before evaluating the boundary term.",
            "42",
        )
        prompt = build_privileged_evaluation_prompt(
            "Find the sum.", sanitized, "42", redaction_audit=audit
        )
        self.assertEqual(
            prompt,
            build_privileged_evaluation_problem(
                "Find the sum.", sanitized, "42", redaction_audit=audit
            ),
        )
        self.assertIn("Find the sum.", prompt)
        self.assertIn(sanitized, prompt)
        self.assertNotIn("42", prompt)

        with self.assertRaisesRegex(PrivilegedRedactionError, "digest"):
            build_privileged_evaluation_prompt(
                "Find the sum.",
                sanitized + " A tampered safe-looking sentence.",
                "42",
                redaction_audit=audit,
            )
        with self.assertRaisesRegex(PrivilegedRedactionError, "unsafe"):
            build_privileged_evaluation_prompt(
                "Find the sum.", "The final answer is 42.", "42"
            )

    def test_artifact_hashes_and_provenance_make_hindsight_explicit(self):
        raw = json.dumps(
            {
                "reasoning_steps": [
                    "Introduce an auxiliary variable.",
                    "Use the invariant to reduce the remaining cases.",
                ],
                "final_answer": "042",
            }
        )
        artifact = build_privileged_control_artifact(
            "Find the requested value.", "42", raw
        )

        self.assertEqual(artifact["schema_version"], PRIVILEGED_CONTEXT_SCHEMA_VERSION)
        self.assertEqual(artifact["control_mode"], PRIVILEGED_CONTROL_MODE)
        self.assertEqual(artifact["advantage_text_pre_redaction_sha256"], _sha256(raw))
        self.assertEqual(
            artifact["advantage_text_sha256"], _sha256(artifact["advantage_text"])
        )
        self.assertEqual(
            artifact["evaluation_problem_sha256"],
            _sha256(artifact["evaluation_problem"]),
        )
        self.assertNotIn("raw_advantage_text", artifact)
        self.assertNotIn("042", artifact["advantage_text"])
        self.assertNotIn("042", artifact["evaluation_problem"])
        provenance = artifact["context_provenance"]
        self.assertTrue(provenance["construction_used_target_answer"])
        self.assertTrue(provenance["hindsight_exposed"])
        self.assertEqual(provenance["hindsight_exposure_rate"], 1.0)
        self.assertFalse(provenance["literal_target_answer_in_advantage_text"])
        self.assertIn("target_answer", provenance["forbidden_source_ancestry"])
        self.assertIn("future_target_tokens", provenance["forbidden_source_ancestry"])


if __name__ == "__main__":
    unittest.main()
