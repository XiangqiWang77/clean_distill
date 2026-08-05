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
  "failure_modes": ["reusable reasoning failure to guard against", "..."],
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

Every problem must have one checkable final answer, be reliably solvable in at
most four concise reasoning steps, and contain a tempting substantive failure
mode. Prefer concrete numeric, algebraic, or yes/no conclusions over an
open-ended proof when the skill permits. Keep the problem challenging enough
to expose a decision error, but do not rely on obscure facts or long arithmetic.
Before emitting a problem, silently verify that every expression is defined,
all variables and morphisms have compatible domains and codomains, and the
premises and requested conclusion are jointly well-posed.

SKILL CARD:
{skill_card}

Return:
{{"candidates": [
  {{"candidate_id": "c0", "candidate_type": "atomic", "problem": "...", "skill_tags": ["..."]}}
]}}

candidate_type is optional. When present, use only atomic, compositional, or
failure_focused, choosing exactly one value for each problem. Do not include
an answer, solution, correct trajectory, or
wrong trajectory; those are generated independently after the problem passes
the target-disjoint firewall.
"""

SOLVER_SYSTEM = """Solve the supplied mathematical problem independently.
You do not know any target problem. Return only the exact tagged format
requested by the user. Do not restate the problem. Keep each step concise and
put all mathematical text literally inside the tags; no JSON escaping is
needed."""

SOLVER_USER = """Solve this specialization candidate:

{candidate_problem}

Return exactly one FINAL_ANSWER element followed by one or more CORRECT_STEP
elements. The FINAL_ANSWER content must be the actual checkable answer, never
a description of what belongs there. Each CORRECT_STEP must contain exactly
one STEP_INDEX element and one STEP_TEXT element. STEP_INDEX values must start
at 0 and be consecutive; STEP_TEXT content must be the actual justified
mathematical step. Use only these tag names:
FINAL_ANSWER, CORRECT_STEP, STEP_INDEX, STEP_TEXT.

Use literal XML delimiters for every element. Start with
<FINAL_ANSWER> and close it with </FINAL_ANSWER>. Wrap each step in
<CORRECT_STEP> and </CORRECT_STEP>, with its index and text similarly wrapped
in their named opening and closing tags. Use at most four concise CORRECT_STEP
elements and keep the entire response under 500 tokens.

Do not copy schema descriptions into any element. Do not add markdown fences
or prose outside the required elements."""

WRONG_TRAJECTORY_SYSTEM = """Produce an independent fallible-model attempt for
the supplied mathematical practice problem. You do not know any target
problem, target answer, correct trajectory, or verifier feedback. Use a
plausible reasoning route and do not self-correct a substantive mistake.
Silently solve the problem first, then introduce one plausible failure that
changes the final answer; never reveal the silent correct conclusion.
Return only the exact tagged format requested by the user. Do not use JSON."""

WRONG_TRAJECTORY_USER = """Independently produce a plausible incorrect attempt
for this specialization candidate:

{candidate_problem}

Abstract reusable failure modes that may inspire the attempt (they are not a
solution):
{failure_modes}

Return exactly one WRONG_FINAL_ANSWER element followed by one or more
WRONG_STEP elements. The WRONG_FINAL_ANSWER content must be the attempt's
actual answer, never a description of what belongs there. Each WRONG_STEP
must contain exactly one STEP_INDEX element and one STEP_TEXT element.
STEP_INDEX values must start at 0 and be consecutive; at least one STEP_TEXT
must contain a substantive mathematical mistake. Use only these tag names:
WRONG_FINAL_ANSWER, WRONG_STEP, STEP_INDEX, STEP_TEXT.

Use literal XML delimiters for every element. Start with
<WRONG_FINAL_ANSWER> and close it with </WRONG_FINAL_ANSWER>. Wrap each step in
<WRONG_STEP> and </WRONG_STEP>, with its index and text similarly wrapped in
their named opening and closing tags. The emitted WRONG_FINAL_ANSWER must
differ mathematically from the answer you derived silently. Use one to three
concise WRONG_STEP elements and keep the entire response under 350 tokens.

Do not copy schema descriptions into any element. Do not add markdown fences
or prose outside the required elements. This call is independent: never claim
to have seen a correct solution or deliberately alter a stated correct
answer."""

VERIFIER_SYSTEM = """You are an independent mathematical verifier. Check the
candidate solution from first principles. You do not know any target problem.
First check that the candidate problem itself is well-posed and that every
expression, operation, and morphism is type-consistent. An undefined expression
or incompatible domain/codomain makes problem_well_posed=false and valid=false,
regardless of the proposed answer.
Judge mathematical correctness rather than presentation style; a concise
standard asymptotic or algebraic argument is valid when its conclusion follows.
Never manufacture a different answer: any corrected answer must agree with the
corrected derivation. Return strict JSON and no prose outside it."""

VERIFIER_USER = """CANDIDATE PROBLEM:
{candidate_problem}

PROPOSED SOLUTION:
{solution}

PROPOSED FINAL ANSWER:
{final_answer}

Return:
{{
  "problem_well_posed": true,
  "valid": true,
  "reason": "brief verification",
  "corrected_solution": "same solution if valid, otherwise a corrected derivation",
  "corrected_final_answer": "final answer after verification"
}}
"""

FRONTIER_VERIFIER_SYSTEM = """You are an independent mathematical trajectory
verifier. You know only a generated practice problem and two independently
generated attempts for that practice problem. You do not know any target
problem or target answer. Locate the earliest substantively invalid step in
the wrong attempt and verify a local correction. Return strict JSON and no
prose outside it."""

FRONTIER_VERIFIER_USER = """CANDIDATE PROBLEM:
{candidate_problem}

VERIFIED CORRECT TRAJECTORY:
{correct_trajectory}

VERIFIED CORRECT FINAL ANSWER:
{final_answer}

INDEPENDENT MODEL TRAJECTORY TO AUDIT:
{wrong_trajectory}

MODEL FINAL ANSWER:
{wrong_final_answer}

Return:
{{
  "wrong_trajectory_incorrect": true,
  "prefix_before_error_valid": true,
  "wrong_step_invalid": true,
  "corrective_action_valid": true,
  "wrong_step_index": 0,
  "error_explanation": "why this is the first invalid step",
  "corrective_action": "a precise local action that fixes the step"
}}

wrong_step_index must name an existing step in the model trajectory. Set
prefix_before_error_valid=true only when every earlier model step is valid;
set wrong_step_invalid=true only when the selected step is genuinely invalid;
and set corrective_action_valid=true only when the proposed action repairs
that error. Do not merely select the first textually different step."""


def skill_card_messages(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SKILL_CARD_SYSTEM},
        {"role": "user", "content": SKILL_CARD_USER.format(problem=problem)},
    ]


def candidate_messages(
    skill_card: dict[str, Any], num_candidates: int
) -> list[dict[str, str]]:
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
        {
            "role": "user",
            "content": SOLVER_USER.format(candidate_problem=candidate_problem),
        },
    ]


def wrong_trajectory_messages(
    candidate_problem: str, failure_modes: list[str]
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": WRONG_TRAJECTORY_SYSTEM},
        {
            "role": "user",
            "content": WRONG_TRAJECTORY_USER.format(
                candidate_problem=candidate_problem,
                failure_modes=json.dumps(failure_modes, ensure_ascii=False),
            ),
        },
    ]


def verifier_messages(
    candidate_problem: str, solution: str, final_answer: str
) -> list[dict[str, str]]:
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


def frontier_verifier_messages(
    candidate_problem: str,
    correct_trajectory: list[dict[str, Any]],
    final_answer: str,
    wrong_trajectory: list[dict[str, Any]],
    wrong_final_answer: str,
) -> list[dict[str, str]]:
    def render(steps: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"Step {int(step['step_index'])}: {str(step['text']).strip()}"
            for step in steps
        )

    return [
        {"role": "system", "content": FRONTIER_VERIFIER_SYSTEM},
        {
            "role": "user",
            "content": FRONTIER_VERIFIER_USER.format(
                candidate_problem=candidate_problem,
                correct_trajectory=render(correct_trajectory),
                final_answer=final_answer,
                wrong_trajectory=render(wrong_trajectory),
                wrong_final_answer=wrong_final_answer,
            ),
        },
    ]
