"""Prompts implementing the target-information firewall."""

from __future__ import annotations

import json
from typing import Any


SKILL_CARD_SYSTEM = """You are a mathematical curriculum analyst.
Extract only reusable capabilities. Never solve the target problem and never
repeat its entities, literal numbers, answer choices, or distinctive wording.
Return strict JSON and no prose outside the JSON object."""

SKILL_CARD_USER = """Analyze the target problem below only to identify the
skills needed to solve problems of this type.

TARGET PROBLEM:
{problem}

Return this schema:
{{
  "domain": "broad mathematical domain",
  "skills": ["reusable skill", "..."],
  "reasoning_operators": ["operation or inference", "..."],
  "difficulty": "easy|medium|hard|olympiad",
  "constraints": ["abstract structural constraint", "..."],
  "target_details_removed": true
}}

Do not include a solution, a final answer, target-specific names, or any
literal numeric value copied from the target."""

CANDIDATE_SYSTEM = """You design independent mathematical practice problems
from an abstract skill card. You have no access to the target problem. Return
strict JSON and no prose outside the JSON object. Every problem must be fully
specified: never emit placeholders, redaction artifacts, or references to
hidden, removed, omitted, generic, or unspecified details."""

CANDIDATE_USER = """Create {num_candidates} diverse, self-contained problems
that exercise the skill card below. They are specialization candidates, not a
mock test and not held-out checks. Do not solve them. Use varied structures,
entities, and numeric values. Instantiate abstract quantities and objects with
fresh concrete details. Do not copy placeholder-like wording from the skill
card or write terms such as "redacted detail", "redacted number",
"placeholder", "unspecified value", "unspecified quantity", "generic object",
or angle-bracket substitution tokens in a candidate.

SKILL CARD:
{skill_card}

Return:
{{"candidates": [
  {{"candidate_id": "c0", "problem": "...", "skill_tags": ["..."]}}
]}}
"""

SOLVER_SYSTEM = """Solve the supplied mathematical problem independently.
Return strict JSON. Put a checkable final answer in final_answer and a complete
derivation in solution. You do not know any target problem."""

SOLVER_USER = """Solve this specialization candidate:

{candidate_problem}

Return {{"solution": "...", "final_answer": "..."}}."""

VERIFIER_SYSTEM = """You are an independent mathematical verifier. Check the
candidate solution from first principles. You do not know any target problem.
Return strict JSON and no prose outside it."""

VERIFIER_USER = """CANDIDATE PROBLEM:
{candidate_problem}

PROPOSED SOLUTION:
{solution}

PROPOSED FINAL ANSWER:
{final_answer}

Return:
{{
  "valid": true,
  "reason": "brief verification",
  "corrected_solution": "same solution if valid, otherwise a corrected derivation",
  "corrected_final_answer": "final answer after verification"
}}
"""


def skill_card_messages(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SKILL_CARD_SYSTEM},
        {"role": "user", "content": SKILL_CARD_USER.format(problem=problem)},
    ]


def candidate_messages(skill_card: dict[str, Any], num_candidates: int) -> list[dict[str, str]]:
    # Crucially, this function has no problem/answer argument.
    return [
        {"role": "system", "content": CANDIDATE_SYSTEM},
        {
            "role": "user",
            "content": CANDIDATE_USER.format(
                num_candidates=num_candidates,
                skill_card=json.dumps(skill_card, ensure_ascii=False, indent=2),
            ),
        },
    ]


def solver_messages(candidate_problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SOLVER_SYSTEM},
        {"role": "user", "content": SOLVER_USER.format(candidate_problem=candidate_problem)},
    ]


def verifier_messages(candidate_problem: str, solution: str, final_answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {
            "role": "user",
            "content": VERIFIER_USER.format(
                candidate_problem=candidate_problem,
                solution=solution,
                final_answer=final_answer,
            ),
        },
    ]
