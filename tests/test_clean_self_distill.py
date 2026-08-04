"""Dependency-light tests for CSD boundaries and paper-facing metrics."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.clean_self_distill.io import (
    compute_proposal_training_sha256,
    load_proposal_map,
    load_query_records,
    stable_hash,
)
from src.clean_self_distill.metrics import HindsightAudit, aggregate_teacher_metrics
from src.clean_self_distill.prompts import candidate_messages
from src.clean_self_distill.propose import (
    _placeholder_artifact_audit,
    propose_for_query,
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
        "specialization_status": "ready",
        "specialization_failure_reason": "",
        "specialization_no_op": False,
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


class _ScriptedGenerator:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []
        self.tokenizer = SimpleNamespace(chat_template=None)
        self.enable_thinking = False

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("Scripted generator received an unexpected call")
        return self.responses.pop(0)

    def counters(self) -> dict[str, float]:
        return {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "generation_seconds": 0.0,
        }


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
        self.assertIn("never emit placeholders", prompt.lower())
        self.assertIn("redaction artifacts", prompt.lower())
        self.assertIn("fresh concrete details", prompt.lower())

    def test_skill_card_sanitizer_removes_target_symbols_and_expressions(self):
        problem = r"For triangle ABC, if $x+y=9173$, choose option C."
        malicious = {
            "domain": "geometry",
            "skills": [r"Use triangle ABC and the equation $x+y=9173$"],
            "reasoning_operators": ["select C"],
            "difficulty": "hard",
            "constraints": ["Apply x+y=9173 before choosing C"],
        }
        clean, redactions = sanitize_skill_card(malicious, problem)
        serialized = json.dumps(clean)
        for secret in ("ABC", "9173", "x+y", "select C"):
            self.assertNotIn(secret, serialized)
        self.assertNotIn("redacted", serialized.lower())
        self.assertIn("<inline-math-expression>", redactions)
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
                "Use a redacted detail and an unspecified quantity.",
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
        self.assertNotIn("redacted detail", serialized)
        self.assertNotIn("unspecified quantity", serialized)
        self.assertFalse(any(ord(character) < 32 for character in serialized))
        self.assertEqual(clean["constraints"][-2], "a variable quantity")
        self.assertIs(clean["constraints"][-1], True)
        self.assertNotIn("redacted", serialized)
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

        for opener in ("There", "Points", "Then", "Output"):
            opener_audit = target_disjoint_audit(
                f"{opener} begins an otherwise unrelated sentence.",
                f"A fresh exercise may use the word {opener.lower()} naturally.",
            )
            self.assertEqual(opener_audit["shared_target_entities"], [])
            self.assertEqual(opener_audit["literal_overlap_count"], 0.0)

        structural = target_disjoint_audit(
            "A recurrence uses -2, -1, 0, 1, 2, and 37 as coefficients.",
            "Build an unrelated polynomial from -2, -1, 0, 1, 2, and 41.",
        )
        self.assertEqual(structural["shared_target_numbers"], [])
        self.assertEqual(
            set(structural["ignored_target_structural_numbers"]),
            {"-2", "-1", "0", "1", "2"},
        )
        self.assertEqual(structural["literal_overlap_count"], 0.0)

        salient = target_disjoint_audit(
            "Use the values 3, 1/2, and 1,000 in a construction.",
            "An independent exercise also uses 3, 1/2, and 1000.",
        )
        self.assertEqual(
            set(salient["shared_target_numbers"]), {"3", "1/2", "1000"}
        )
        self.assertEqual(salient["literal_overlap_count"], 3.0)

        tex_fractions = target_disjoint_audit(
            r"Use \frac{1}{2}, \tfrac34, and \dfrac{-2}{5} in a construction.",
            "Use the ratios 1/2, 3/4, and -2/5 in a different construction.",
        )
        self.assertEqual(
            set(tex_fractions["shared_target_numbers"]),
            {"1/2", "3/4", "-2/5", "2/5"},
        )

        comma_sequences = target_disjoint_audit(
            "Plot (1,1) and list 1,2,3,4.",
            "Use the unrelated integers 11 and 1234.",
        )
        self.assertEqual(comma_sequences["shared_target_numbers"], [])
        self.assertEqual(comma_sequences["literal_overlap_count"], 0.0)

        coordinate_component = target_disjoint_audit(
            "Plot the point (1,234).",
            "Use 234 as a coefficient in a different problem.",
        )
        self.assertEqual(coordinate_component["shared_target_numbers"], ["234"])
        coordinate_ambiguous_grouping = target_disjoint_audit(
            "Plot the point (1,234).",
            "Use 1234 as a coefficient in a different problem.",
        )
        # The surface form is ambiguous between a coordinate and thousands
        # grouping, so the leakage audit deliberately keeps both readings.
        self.assertEqual(
            coordinate_ambiguous_grouping["shared_target_numbers"], ["1234"]
        )

        normalized_decimals = target_disjoint_audit(
            r"Use 03, 3.0, and \frac{02}{04}.",
            "Use 3 and 1/2 in another exercise.",
        )
        self.assertEqual(
            set(normalized_decimals["shared_target_numbers"]), {"3", "1/2"}
        )

        equivalent_fraction_decimal = target_disjoint_audit(
            "Use the fraction 2/4.",
            "Use the decimal .5 in another exercise.",
        )
        self.assertEqual(
            equivalent_fraction_decimal["shared_target_numbers"], ["1/2"]
        )

        tex_grouping = target_disjoint_audit(
            r"Evaluate \boxed{1,234} and x^{1,234}.",
            "Use the integer 1234 in another exercise.",
        )
        self.assertEqual(tex_grouping["shared_target_numbers"], ["1234"])
        tex_set = target_disjoint_audit(
            r"Choose an element of \{1,234\}.",
            "Use 234 in another exercise.",
        )
        self.assertEqual(tex_set["shared_target_numbers"], ["234"])
        tex_set_with_ellipsis = target_disjoint_audit(
            r"Choose an element of \{991,992,993,\ldots,1000\}.",
            "Use 991, 992, or 993 in another exercise.",
        )
        self.assertEqual(
            set(tex_set_with_ellipsis["shared_target_numbers"]),
            {"991", "992", "993"},
        )
        for delimiters in (("(", ")"), ("[", "]")):
            bracketed_ellipsis = target_disjoint_audit(
                rf"Choose from {delimiters[0]}991,992,993,\ldots,1000{delimiters[1]}.",
                "Use 991 in another exercise.",
            )
            self.assertEqual(bracketed_ellipsis["shared_target_numbers"], ["991"])

        grouped_set_endpoint = target_disjoint_audit(
            r"Choose an element of \{991,\ldots,1,000\}.",
            "Use 1000 in another exercise.",
        )
        self.assertEqual(grouped_set_endpoint["shared_target_numbers"], ["1000"])

        for math_delimiters in ((r"\(", r"\)"), (r"\[", r"\]")):
            display_math_grouping = target_disjoint_audit(
                rf"Evaluate {math_delimiters[0]}1,234+\cdots{math_delimiters[1]}.",
                "Use 1234 in another exercise.",
            )
            self.assertEqual(display_math_grouping["shared_target_numbers"], ["1234"])

        possessive_entity = target_disjoint_audit(
            "Kayla rolls several fair dice.",
            "Kayla's unrelated exercise concerns a polynomial.",
        )
        self.assertEqual(possessive_entity["shared_target_entities"], ["kayla"])

        fourgram = target_disjoint_audit(
            "Analyze this distinctive alpha beta gamma delta sequence.",
            "Construct a new alpha beta gamma delta example.",
        )
        self.assertGreaterEqual(fourgram["fourgram_overlap_count"], 1.0)

    def test_target_disjoint_numeric_adversarial_matrix(self):
        cases = (
            ("Use 500.", r"Use \frac{1,000}{2}.", "500"),
            (r"Use -\frac{1}{2}.", "Use -1/2.", "-1/2"),
            (r"Use \frac{1}{-2}.", "Use -1/2.", "-1/2"),
            ("Use 1/-2.", "Use -1/2.", "-1/2"),
            (r"Use \frac{1}2.", "Use 1/2.", "1/2"),
            (r"Use \frac1{2}.", "Use 1/2.", "1/2"),
            ("Use 1234.", r"Use 1{,}234.", "1234"),
            ("Use 1234.", r"Use 1\,234.", "1234"),
            ("Use 1234.", r"Use 1,\!234.", "1234"),
            (
                r"Choose from (1,000,1,001,\ldots,2,000).",
                "Use 1001.",
                "1001",
            ),
            (
                r"Choose x in \{x:3\le x\le1,000\}.",
                "Use 1000.",
                "1000",
            ),
            (
                r"Choose from 991,992,993,\ldots,1000.",
                "Use 991.",
                "991",
            ),
            (
                r"Choose from \bigl\{991,992,993,\dotsc,1000\bigr\}.",
                "Use 992.",
                "992",
            ),
            (
                r"Evaluate (1,234+\cdots+2,345).",
                "Use 1234.",
                "1234",
            ),
            (
                r"Choose from \(991,992,993,\mathellipsis,1000\).",
                "Use 993.",
                "993",
            ),
            (
                r"Choose from $991,992,993,\ldots,1000$.",
                "Use 993.",
                "993",
            ),
            (r"Use \frac{1}{1}.", "Use 1.", "1"),
            (r"Use (\frac{3}{2}).", "Use 3/2.", "3/2"),
            (r"Use \bigl(\frac{3}{2}\bigr).", "Use 1.5.", "3/2"),
            (r"Use {1 \over 2}.", "Use 0.5.", "1/2"),
            (r"Use 3 \frac{1}{2}.", "Use 3.5.", "7/2"),
            (r"Increase by 40\%.", "Use 0.4.", "2/5"),
            (r"Apply a 7.5\% tax.", "Use 0.075.", "3/40"),
            (r"Use |x|-\tfrac{1}{2}.", "Use 1/2.", "1/2"),
            ("Use x-15.", "Use 15.", "15"),
            (r"Evaluate $\boxed{1,234}$.", "Use 1234.", "1234"),
            (r"Evaluate $\sqrt{1,234}$.", "Use 1234.", "1234"),
            (r"Evaluate $1,234!$.", "Use 1234.", "1234"),
            (
                r"Choose from \{1,234, 2,345, \ldots, 10,000\}.",
                "Use 2345.",
                "2345",
            ),
            ("Choose from [991,992).", "Use 991.", "991"),
            ("Choose from (991,992].", "Use 992.", "992"),
            ("Choose (991,992,(3,4),993).", "Use 991.", "991"),
            (r"Choose 991,992,993,\ldots.", "Use 992.", "992"),
            (r"Choose \ldots,991,992,993.", "Use 993.", "993"),
            ("Choose {991,992,993}.", "Use 991.", "991"),
            (
                r"Choose \{991,\!992,\!993,\ldots,1000\}.",
                "Use 992.",
                "992",
            ),
            (r"Use 1\thinspace000.", "Use 1000.", "1000"),
            (r"Choose $991,992,993$.", "Use 992.", "992"),
            ("Choose (991,992+x).", "Use 991.", "991"),
            ("Choose [991,992/3].", "Use 992/3.", "992/3"),
            ("Choose (1,234+x, 2,345+y).", "Use 1234.", "1234"),
            (r"Use 3\frac{1}{2}.", "Use 3.5.", "7/2"),
            (r"Use 3\,\tfrac12.", "Use 3.5.", "7/2"),
            (r"Use 3{1\over2}.", "Use 3.5.", "7/2"),
            ("Use 3  \\frac{1}{2}.", "Use 3.5.", "7/2"),
            ("Use 3\n\\frac{1}{2}.", "Use 3.5.", "7/2"),
            (r"Use 3~\frac12.", "Use 3.5.", "7/2"),
            (r"Use x - \frac{1}{2}.", "Use 0.5.", "1/2"),
            ("Use x - 15.", "Use 15.", "15"),
            (r"Use 1\over2.", "Use 0.5.", "1/2"),
            (r"Use \cfrac{1}{2}.", "Use 0.5.", "1/2"),
            (r"Increase by 40\text{\%}.", "Use 0.4.", "2/5"),
            (r"Use \frac{- 1}{2}.", "Use -.5.", "-1/2"),
            ("Use - 1/2.", "Use -.5.", "-1/2"),
            ("Use 1\u202f000.", "Use 1000.", "1000"),
            ("Use 1\u00a0000.", "Use 1000.", "1000"),
            ("Use 1\u2009000.", "Use 1000.", "1000"),
            ("Use 1 000.", "Use 1000.", "1000"),
            (r"Use 1\text{,}000.", "Use 1000.", "1000"),
            ("Use 0.5.", "Use 5e-1.", "1/2"),
            ("Use 1000.", "Use 10e2.", "1000"),
            (r"Use 7\ \frac12.", "Use 7.5.", "15/2"),
            ("Use 1~000.", "Use 1000.", "1000"),
            (r"Apply 25~\%.", "Use 0.25.", "1/4"),
            (r"Apply 25{\%}.", "Use 0.25.", "1/4"),
            ("Use 1.e3.", "Use 1000.", "1000"),
            ("Use 37e0000000.", "Use 37.", "37"),
            ("Use 1e0000003.", "Use 1000.", "1000"),
            ("Use " + "0" * 1100 + "37.", "Use 37.", "37"),
            ("Use 37." + "0" * 1100, "Use 37.", "37"),
        )
        for target, candidate, expected in cases:
            with self.subTest(target=target, candidate=candidate):
                audit = target_disjoint_audit(target, candidate)
                self.assertIn(expected, audit["shared_target_numbers"])
                self.assertGreater(audit["literal_overlap_count"], 0.0)

        # Proposer output is untrusted and can approach the generation-token
        # budget. Oversized integer spellings must fail safely, not kill a
        # formal shard through Python's integer-to-string digit limit.
        oversized = target_disjoint_audit("Use 37.", "Use " + "9" * 5000 + ".")
        self.assertEqual(oversized["shared_target_numbers"], [])

    def test_insufficient_verified_candidates_persist_as_auditable_no_op(self):
        proposer = _ScriptedGenerator(
            [
                json.dumps(
                    {
                        "domain": "algebra",
                        "skills": ["reason about variable quantities"],
                        "reasoning_operators": ["compare cases"],
                        "difficulty": "medium",
                        "constraints": [],
                    }
                ),
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_id": "bad-0",
                                "problem": "Compute a total involving 9173 and 8.",
                                "skill_tags": ["algebra"],
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_id": "bad-1",
                                "problem": "Evaluate a product using 9 and 9173.",
                                "skill_tags": ["algebra"],
                            }
                        ]
                    }
                ),
            ]
        )
        solver = _ScriptedGenerator([])
        verifier = _ScriptedGenerator([])
        row = propose_for_query(
            {
                "query_id": "q-insufficient",
                "problem": "Determine a quantity associated with 9173.",
                "source": "aime24",
            },
            proposer,
            solver,
            verifier,
            num_candidates=2,
            proposal_oversample=0,
            max_rounds=2,
            min_accepted_candidates=2,
            max_literal_overlap=0.0,
            max_fourgram_overlap=0.05,
            accept_verifier_corrections=False,
        )

        self.assertEqual(row["specialization_candidates"], [])
        self.assertEqual(row["schema_version"], "clean-self-distill-proposals-v4")
        self.assertEqual(row["candidate_count"], 0)
        self.assertEqual(
            row["specialization_status"], "insufficient_verified_candidates"
        )
        self.assertTrue(row["specialization_no_op"])
        self.assertTrue(row["specialization_failure_reason"])
        self.assertEqual(len(row["skill_card_attempts"]), 1)
        self.assertEqual(len(row["proposal_rounds"]), 2)
        self.assertEqual(len(row["candidate_attempts"]), 2)
        self.assertEqual(row["filter_summary"]["accepted_count"], 0)
        self.assertEqual(row["filter_summary"]["rejected_count"], 2)
        self.assertEqual(
            row["proposal_training_sha256"], compute_proposal_training_sha256(row)
        )
        self.assertEqual(len(solver.calls), 0)
        self.assertEqual(len(verifier.calls), 0)

    def test_failed_skill_card_generation_persists_as_no_op(self):
        invalid_card = json.dumps(
            {
                "domain": "algebra",
                "skills": ["compare cases"],
                "reasoning_operators": None,
                "difficulty": "medium",
                "constraints": [],
            }
        )
        proposer = _ScriptedGenerator([invalid_card, invalid_card, invalid_card])
        solver = _ScriptedGenerator([])
        verifier = _ScriptedGenerator([])
        row = propose_for_query(
            {
                "query_id": "q-skill-failure",
                "problem": "There is a quantity associated with 9173.",
                "source": "aime24",
            },
            proposer,
            solver,
            verifier,
            num_candidates=4,
            proposal_oversample=2,
            max_rounds=3,
            min_accepted_candidates=4,
            max_literal_overlap=0.0,
            max_fourgram_overlap=0.05,
            accept_verifier_corrections=False,
        )

        self.assertTrue(row["skill_card_generation_failed"])
        self.assertTrue(row["specialization_no_op"])
        self.assertEqual(
            row["specialization_status"], "insufficient_verified_candidates"
        )
        self.assertIn("skill card", row["specialization_failure_reason"])
        self.assertEqual(row["specialization_candidates"], [])
        self.assertEqual(row["proposal_rounds"], [])
        self.assertEqual(row["candidate_attempts"], [])
        self.assertEqual(len(row["skill_card_attempts"]), 3)
        self.assertTrue(row["skill_card_target_disjoint_audit"]["safe"])
        self.assertEqual(len(proposer.calls), 3)
        self.assertEqual(len(solver.calls), 0)
        self.assertEqual(len(verifier.calls), 0)
        self.assertEqual(
            row["proposal_training_sha256"], compute_proposal_training_sha256(row)
        )

    def test_placeholder_artifacts_are_deterministically_rejected(self):
        proposer = _ScriptedGenerator(
            [
                json.dumps(
                    {
                        "domain": "algebra",
                        "skills": ["instantiate abstract quantities"],
                        "reasoning_operators": ["compare cases"],
                        "difficulty": "medium",
                        "constraints": [],
                    }
                ),
                json.dumps(
                    {
                        "candidates": [
                            {
                                "problem": "Compute using a redacted detail.",
                                "skill_tags": ["algebra"],
                            },
                            {
                                "problem": "Find the unspecified quantity.",
                                "skill_tags": ["algebra"],
                            },
                            {
                                "problem": "Replace the placeholder before solving.",
                                "skill_tags": ["algebra"],
                            },
                            {
                                "problem": "Evaluate <fresh_value> plus a term.",
                                "skill_tags": ["algebra"],
                            },
                        ]
                    }
                ),
            ]
        )
        solver = _ScriptedGenerator([])
        verifier = _ScriptedGenerator([])
        row = propose_for_query(
            {
                "query_id": "q-placeholder",
                "problem": "Determine a quantity associated with 9173.",
                "source": "aime24",
            },
            proposer,
            solver,
            verifier,
            num_candidates=4,
            proposal_oversample=0,
            max_rounds=1,
            min_accepted_candidates=4,
            max_literal_overlap=0.0,
            max_fourgram_overlap=0.05,
            accept_verifier_corrections=False,
        )

        self.assertEqual(row["specialization_candidates"], [])
        self.assertTrue(row["specialization_no_op"])
        self.assertEqual(len(row["candidate_attempts"]), 4)
        self.assertEqual(
            {attempt["reason"] for attempt in row["candidate_attempts"]},
            {"placeholder_artifact"},
        )
        self.assertTrue(
            all(
                not attempt["placeholder_artifact_audit"]["safe"]
                for attempt in row["candidate_attempts"]
            )
        )
        self.assertEqual(
            row["filter_summary"]["rejection_reason_counts"],
            {"placeholder_artifact": 4},
        )
        self.assertEqual(len(solver.calls), 0)
        self.assertEqual(len(verifier.calls), 0)
        self.assertEqual(
            row["proposal_training_sha256"], compute_proposal_training_sha256(row)
        )

    def test_sanitizer_standins_are_candidate_placeholder_artifacts(self):
        for text in (
            "Find a variable quantity.",
            "Use a symbolic relation.",
            "Determine an abstract object.",
            "Compute with an abstract element.",
            "Introduce an auxiliary variable.",
            "State the derived conclusion.",
        ):
            with self.subTest(text=text):
                audit = _placeholder_artifact_audit(text)
                self.assertFalse(audit["safe"])
                self.assertGreater(audit["artifact_count"], 0)

    def test_placeholder_artifact_in_solver_output_is_rejected(self):
        proposer = _ScriptedGenerator(
            [
                json.dumps(
                    {
                        "domain": "algebra",
                        "skills": ["multiply independent quantities"],
                        "reasoning_operators": ["combine factors"],
                        "difficulty": "medium",
                        "constraints": [],
                    }
                ),
                json.dumps(
                    {
                        "candidates": [
                            {
                                "problem": "Compute the product of 8 and 9.",
                                "skill_tags": ["arithmetic"],
                            }
                        ]
                    }
                ),
            ]
        )
        solver = _ScriptedGenerator(
            [
                json.dumps(
                    {
                        "solution": "Multiply by the redacted number.",
                        "final_answer": "72",
                    }
                )
            ]
        )
        verifier = _ScriptedGenerator([])
        row = propose_for_query(
            {
                "query_id": "q-solver-placeholder",
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
        )

        self.assertEqual(row["specialization_candidates"], [])
        self.assertTrue(row["specialization_no_op"])
        self.assertEqual(len(row["candidate_attempts"]), 1)
        attempt = row["candidate_attempts"][0]
        self.assertEqual(attempt["reason"], "placeholder_artifact")
        self.assertEqual(attempt["placeholder_artifact_source"], "solver_output")
        self.assertFalse(attempt["solver_placeholder_artifact_audit"]["safe"])
        self.assertFalse(attempt["placeholder_artifact_audit"]["safe"])
        self.assertEqual(
            row["filter_summary"]["rejection_reason_counts"],
            {"placeholder_artifact": 1},
        )
        self.assertEqual(len(verifier.calls), 0)
        self.assertEqual(
            row["proposal_training_sha256"], compute_proposal_training_sha256(row)
        )

    def test_verified_candidate_sets_ready_specialization_state(self):
        proposer = _ScriptedGenerator(
            [
                json.dumps(
                    {
                        "domain": "algebra",
                        "skills": ["multiply independent quantities"],
                        "reasoning_operators": ["combine factors"],
                        "difficulty": "medium",
                        "constraints": [],
                    }
                ),
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_id": "fresh-0",
                                "problem": "Compute the product of 8 and 9.",
                                "skill_tags": ["arithmetic"],
                            }
                        ]
                    }
                ),
            ]
        )
        solver = _ScriptedGenerator(
            [json.dumps({"solution": "Multiply the factors.", "final_answer": "72"})]
        )
        verifier = _ScriptedGenerator(
            [json.dumps({"valid": True, "reason": "The product is correct."})]
        )
        row = propose_for_query(
            {
                "query_id": "q-ready",
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
        )

        self.assertEqual(row["specialization_status"], "ready")
        self.assertEqual(row["schema_version"], "clean-self-distill-proposals-v4")
        self.assertEqual(row["specialization_failure_reason"], "")
        self.assertFalse(row["specialization_no_op"])
        self.assertEqual(row["candidate_count"], 1)
        self.assertEqual(
            row["proposal_training_sha256"], compute_proposal_training_sha256(row)
        )

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
