"""Answer-redacted privileged reasoning control.

The control in this module is intentionally *not* hindsight-free.  A private
generation call may use the target answer to produce a correct line of
reasoning.  Before that reasoning is shown to the evaluated model, this module
removes boxed conclusions, direct answer declarations, and every detectable
literal/equivalent spelling of the target answer.  The returned provenance
keeps the answer-conditioned ancestry explicit, so redaction cannot launder a
privileged control into a clean one.

Only the sanitized advantage text is returned.  The pre-redaction text is
represented by a SHA-256 digest in the artifact, which avoids persisting an
answer-bearing prompt payload by accident.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Mapping


PRIVILEGED_CONTEXT_SCHEMA_VERSION = "clean-self-distill-privileged-context-v1"
PRIVILEGED_REDACTION_SCHEMA_VERSION = "privileged-answer-redaction-v1"
PRIVILEGED_CONTROL_MODE = "answer-conditioned-cot-answer-redacted-v1"


_GENERATION_SYSTEM = """You construct a privileged, target-specific mathematical reasoning aid.
You may use the private ground-truth answer to keep the reasoning correct, but
the text returned to the caller must never state, quote, encode, or confirm that
answer.  Stop before the final numerical or multiple-choice conclusion.  Do not
use \\boxed, \\fbox, an `Answer:` field, or phrases such as `the final answer is`.
Return strict JSON with exactly this schema and no prose outside it:
{"reasoning_steps": ["answer-free step", "..."]}.
"""

_GENERATION_USER = """TARGET PROBLEM:
{problem}

PRIVATE GROUND-TRUTH ANSWER (construction signal only; never repeat it):
{target_answer}

Produce a correct reasoning path that gives a useful method/CoT advantage, but
omit the final-answer step and every literal or equivalent spelling of the
private answer.
"""

_EVALUATION_TEMPLATE = """{problem}

Privileged answer-redacted reasoning guidance:
--- BEGIN GUIDANCE ---
{advantage_text}
--- END GUIDANCE ---

Use the guidance as a reasoning aid, complete the solution independently, and
put your own final answer within \\boxed{{}}.
"""

_REASONING_KEYS = (
    "reasoning_steps",
    "steps",
    "advantage_text",
    "reasoning",
    "cot",
    "method",
    "solution",
)
_ANSWER_FIELD_RE = re.compile(
    r"(?:^|_)(?:answer|final_answer|result|final_result|boxed_answer)(?:$|_)",
    flags=re.IGNORECASE,
)
_BOX_MARKER_RE = re.compile(r"\\(?:boxed|fbox)\b", flags=re.IGNORECASE)
_BOX_OPEN_RE = re.compile(r"\\(?:boxed|fbox)\s*\{", flags=re.IGNORECASE)
_DIRECT_DECLARATION_RE = re.compile(
    r"(?:"
    r"\b(?:the\s+)?(?:final|correct|target|requested)\s+"
    r"(?:answer|result|value)\s*(?:is|equals?|must\s+be|should\s+be|=|:)"
    r"|\b(?:final\s+)?(?:answer|result)\s*(?:is|equals?|=|:)"
    r"|\b(?:answer|result)\s+to\s+the\s+(?:problem|question)\s+is\b"
    r")",
    flags=re.IGNORECASE,
)
_CONCLUSION_ONLY_RE = re.compile(
    r"^(?:therefore|thus|hence|consequently|so|finally)\s*[,.:;!?-]*$",
    flags=re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)

_NUMBER_ATOM = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TEX_FRACTION_RE = re.compile(
    rf"[-+]?\s*\\(?:frac|dfrac|tfrac|cfrac)\s*"
    rf"\{{\s*{_NUMBER_ATOM}\s*\}}\s*\{{\s*{_NUMBER_ATOM}\s*\}}"
)
_PLAIN_FRACTION_RE = re.compile(rf"(?<![\w.]){_NUMBER_ATOM}\s*/\s*{_NUMBER_ATOM}(?!\w)")
# A sentence-ending period may immediately follow a decimal/scientific value;
# forbidding only word continuation keeps ``42.0.`` detectable without letting
# the scanner start in the middle of another decimal.
_SCALAR_NUMBER_RE = re.compile(rf"(?<![\w.]){_NUMBER_ATOM}(?!\w)")
_BINARY_ARITHMETIC_RE = re.compile(
    rf"(?<![\w.])(?P<left>{_NUMBER_ATOM})\s*"
    rf"(?P<operator>\\(?:cdot|times|div)|[+*/×÷-])\s*"
    rf"(?P<right>{_NUMBER_ATOM})(?!\w)",
    flags=re.IGNORECASE,
)
_TEX_INVISIBLE_SPACING_RE = re.compile(
    r"\\(?:!|,|;|:|>|quad|qquad|enspace|thinspace|medspace|thickspace)\s*"
)
_TEX_FRACTION_VALUE_RE = re.compile(
    rf"(?P<sign>[-+]?)\s*\\(?:frac|dfrac|tfrac|cfrac)\s*"
    rf"\{{\s*(?P<numerator>{_NUMBER_ATOM})\s*\}}\s*"
    rf"\{{\s*(?P<denominator>{_NUMBER_ATOM})\s*\}}"
)
_PLAIN_FRACTION_VALUE_RE = re.compile(
    rf"(?P<numerator>{_NUMBER_ATOM})\s*/\s*(?P<denominator>{_NUMBER_ATOM})"
)


class PrivilegedRedactionError(ValueError):
    """Raised when an answer-free privileged context cannot be certified."""

    def __init__(self, message: str, *, audit: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.audit = dict(audit or {})


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256(serialized)


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise PrivilegedRedactionError(f"{name} must be non-empty")
    return text


def _visible_normalize(text: str) -> str:
    text = (
        unicodedata.normalize("NFKC", str(text))
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    # Format controls can hide an otherwise literal answer (for example 4<ZWSP>2).
    return "".join(
        character for character in text if unicodedata.category(character) != "Cf"
    )


def build_privileged_cot_generation_messages(
    problem: str, target_answer: str
) -> list[dict[str, str]]:
    """Build the private answer-conditioned CoT generation messages.

    These messages are privileged by construction and must never be reused by
    Base, CSD-T, CSD-SD, a candidate proposer, or a support verifier.
    """

    problem_text = _required_text(problem, "problem")
    answer_text = _required_text(target_answer, "target_answer")
    return [
        {"role": "system", "content": _GENERATION_SYSTEM},
        {
            "role": "user",
            "content": _GENERATION_USER.format(
                problem=problem_text,
                target_answer=answer_text,
            ),
        },
    ]


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.fullmatch(text)
    return match.group("body") if match is not None else text


def _payload_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_payload_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, Mapping):
        parts = [
            _payload_text(item)
            for key, item in value.items()
            if not _ANSWER_FIELD_RE.search(str(key))
        ]
        return "\n".join(part for part in parts if part)
    return "" if value is None else str(value).strip()


def _extract_reasoning_payload(text: str) -> tuple[str, list[str]]:
    candidate = _strip_code_fence(text.strip())
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        return candidate, []
    if not isinstance(parsed, Mapping):
        return _payload_text(parsed), []

    selected_key = next((key for key in _REASONING_KEYS if key in parsed), None)
    if selected_key is None:
        raise PrivilegedRedactionError(
            "Structured privileged output is missing a recognized reasoning field"
        )
    removed_keys = [str(key) for key in parsed if key != selected_key]
    return _payload_text(parsed[selected_key]), removed_keys


def _remove_boxed_constructs(text: str) -> tuple[str, int, int]:
    """Remove balanced boxed/fbox spans; truncate malformed spans fail-closed."""

    output: list[str] = []
    cursor = 0
    removed = 0
    malformed = 0
    while True:
        match = _BOX_OPEN_RE.search(text, cursor)
        if match is None:
            output.append(text[cursor:])
            break
        output.append(text[cursor : match.start()])
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        removed += 1
        if depth:
            malformed += 1
            cursor = len(text)
            break
        cursor = index
    cleaned = "".join(output)
    # A malformed unbraced command is itself unsafe context.  Remove its token;
    # any following answer literal is handled by the answer scan below.
    cleaned, unbraced = _BOX_MARKER_RE.subn(" ", cleaned)
    return cleaned, removed + unbraced, malformed + unbraced


def _decimal_fraction(text: str) -> Fraction:
    compact = re.sub(r"[\s,]", "", text).lstrip("+")
    return Fraction(Decimal(compact))


def _canonical_number(text: str) -> Fraction | None:
    compact = _visible_normalize(text).strip()
    compact = re.sub(r"\\(?:,|!|;|:|quad|qquad|thinspace)", "", compact)
    compact = compact.replace("~", "").replace("$", "")
    while len(compact) >= 2 and (
        (compact[0], compact[-1]) in {("{", "}"), ("(", ")"), ("[", "]")}
    ):
        compact = compact[1:-1].strip()

    tex_match = _TEX_FRACTION_VALUE_RE.fullmatch(compact)
    if tex_match is not None:
        try:
            numerator = _decimal_fraction(tex_match.group("numerator"))
            if tex_match.group("sign") == "-":
                numerator = -numerator
            denominator = _decimal_fraction(tex_match.group("denominator"))
            return numerator / denominator
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

    plain_match = _PLAIN_FRACTION_VALUE_RE.fullmatch(compact)
    if plain_match is not None:
        try:
            return _decimal_fraction(
                plain_match.group("numerator")
            ) / _decimal_fraction(plain_match.group("denominator"))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

    percent = compact.endswith("%")
    if percent:
        compact = compact[:-1].strip()
    try:
        value = _decimal_fraction(compact)
    except (InvalidOperation, ValueError):
        return None
    return value / 100 if percent else value


def _numeric_spans(text: str) -> list[tuple[int, int, Fraction]]:
    spans: list[tuple[int, int, Fraction]] = []
    occupied: list[tuple[int, int]] = []

    def add_matches(pattern: re.Pattern[str]) -> None:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(
                start < prior_end and end > prior_start
                for prior_start, prior_end in occupied
            ):
                continue
            suffix = text[end:]
            percent_match = re.match(r"\s*(?:\\%|%)", suffix)
            rendered = match.group(0)
            if percent_match is not None:
                rendered += "%"
                end += percent_match.end()
            value = _canonical_number(rendered)
            if value is not None:
                spans.append((start, end, value))
                occupied.append((start, end))

    add_matches(_TEX_FRACTION_RE)
    add_matches(_PLAIN_FRACTION_RE)
    add_matches(_SCALAR_NUMBER_RE)
    return sorted(spans)


_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)


def _integer_words(value: int) -> str | None:
    if not -999 <= value <= 999:
        return None
    if value < 0:
        nested = _integer_words(-value)
        return None if nested is None else f"negative {nested}"
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] if not ones else f"{_TENS[tens]} {_ONES[ones]}"
    hundreds, remainder = divmod(value, 100)
    prefix = f"{_ONES[hundreds]} hundred"
    if not remainder:
        return prefix
    nested = _integer_words(remainder)
    return f"{prefix} {nested}"


def _strip_answer_wrappers(answer: str) -> str:
    text = _visible_normalize(answer).strip()
    text = re.sub(r"^\s*\$+|\$+\s*$", "", text).strip()
    for opening, closing in ((r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(opening) and text.endswith(closing):
            text = text[len(opening) : -len(closing)].strip()
    for command in ("boxed", "fbox", "text", "mathrm"):
        match = re.fullmatch(
            rf"\\{command}\s*\{{(?P<body>.*)\}}", text, flags=re.DOTALL
        )
        if match is not None:
            text = match.group("body").strip()
    return text


def _binary_arithmetic_values(text: str) -> list[Fraction]:
    """Evaluate only explicit two-operand scalar arithmetic expressions."""

    values: list[Fraction] = []
    for match in _BINARY_ARITHMETIC_RE.finditer(text):
        try:
            left = _decimal_fraction(match.group("left"))
            right = _decimal_fraction(match.group("right"))
        except (InvalidOperation, ValueError):
            continue
        operator = match.group("operator").casefold()
        try:
            if operator == "+":
                value = left + right
            elif operator == "-":
                value = left - right
            elif operator in {"*", "×", r"\cdot", r"\times"}:
                value = left * right
            elif operator in {"/", "÷", r"\div"}:
                value = left / right
            else:  # pragma: no cover - regex and dispatch are intentionally paired.
                continue
        except ZeroDivisionError:
            continue
        values.append(value)
    return values


def _answer_mentions(text: str, target_answer: str) -> list[dict[str, Any]]:
    normalized_text = _visible_normalize(text)
    normalized_answer = _strip_answer_wrappers(target_answer)
    canonical_answer = _canonical_number(normalized_answer)
    found: list[dict[str, Any]] = []

    if canonical_answer is not None:
        numeric_variants = [normalized_text]
        spacing_compact = _TEX_INVISIBLE_SPACING_RE.sub("", normalized_text)
        if spacing_compact != normalized_text:
            numeric_variants.append(spacing_compact)
        for variant_index, numeric_text in enumerate(numeric_variants):
            for start, end, value in _numeric_spans(numeric_text):
                if value == canonical_answer:
                    found.append(
                        {
                            "kind": (
                                "numeric_equivalent"
                                if variant_index == 0
                                else "spacing_encoded_numeric_equivalent"
                            ),
                            "start": start,
                            "end": end,
                        }
                    )
            for value in _binary_arithmetic_values(numeric_text):
                if value == canonical_answer:
                    found.append(
                        {
                            "kind": "arithmetic_expression_equivalent",
                            "start": 0,
                            "end": 0,
                        }
                    )
        if canonical_answer.denominator == 1:
            words = _integer_words(canonical_answer.numerator)
            if words:
                word_pattern = (
                    r"(?<![A-Za-z])"
                    + r"(?:[\s-]+)".join(re.escape(part) for part in words.split())
                    + r"(?![A-Za-z])"
                )
                for match in re.finditer(
                    word_pattern, normalized_text, flags=re.IGNORECASE
                ):
                    found.append(
                        {
                            "kind": "english_number_equivalent",
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )
    else:
        literal = normalized_answer.strip()
        if literal:
            if len(literal) == 1 and literal.isalpha():
                pattern = rf"(?<![A-Za-z0-9]){re.escape(literal)}(?![A-Za-z0-9])"
                flags = re.IGNORECASE
            elif literal.isalnum():
                pattern = rf"(?<![A-Za-z0-9]){re.escape(literal)}(?![A-Za-z0-9])"
                flags = re.IGNORECASE
            else:
                pattern = re.escape(literal)
                flags = re.IGNORECASE
            for match in re.finditer(pattern, normalized_text, flags=flags):
                found.append(
                    {"kind": "literal", "start": match.start(), "end": match.end()}
                )

    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for item in found:
        unique[(str(item["kind"]), int(item["start"]), int(item["end"]))] = item
    return sorted(
        unique.values(), key=lambda item: (int(item["start"]), int(item["end"]))
    )


def audit_answer_redaction(text: str, target_answer: str) -> dict[str, Any]:
    """Audit whether ``text`` is free of detectable target-answer leakage."""

    candidate = _visible_normalize(_required_text(text, "advantage_text"))
    answer = _required_text(target_answer, "target_answer")
    mentions = _answer_mentions(candidate, answer)
    box_count = len(_BOX_MARKER_RE.findall(candidate))
    declaration_count = len(_DIRECT_DECLARATION_RE.findall(candidate))
    return {
        "schema_version": PRIVILEGED_REDACTION_SCHEMA_VERSION,
        "text_sha256": _sha256(candidate),
        "text_characters": len(candidate),
        "safe": not mentions and box_count == 0 and declaration_count == 0,
        "boxed_construct_count": box_count,
        "direct_answer_declaration_count": declaration_count,
        "answer_mention_count": len(mentions),
        "answer_mention_kinds": dict(
            sorted(Counter(str(item["kind"]) for item in mentions).items())
        ),
    }


def _split_reasoning_units(text: str) -> list[str]:
    units = re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z0-9\\])", text)
    return [unit.strip() for unit in units if unit.strip()]


def sanitize_privileged_advantage_text(
    raw_advantage_text: str, target_answer: str
) -> tuple[str, dict[str, Any]]:
    """Return answer-redacted advantage text plus a fail-closed audit.

    Any unit containing a direct answer declaration or a detectable answer
    spelling is removed in full.  Dropping the whole unit avoids leaving an
    equation-shaped blank that could itself act as a near-answer hint.
    """

    raw = _required_text(raw_advantage_text, "raw_advantage_text")
    answer = _required_text(target_answer, "target_answer")
    normalized = _visible_normalize(raw)
    extracted, removed_structured_fields = _extract_reasoning_payload(normalized)
    without_boxes, boxed_removed, malformed_boxes_removed = _remove_boxed_constructs(
        extracted
    )

    kept: list[str] = []
    reason_counts: Counter[str] = Counter()
    for unit in _split_reasoning_units(without_boxes):
        reasons: list[str] = []
        if _BOX_MARKER_RE.search(unit):
            reasons.append("boxed_construct")
        if _DIRECT_DECLARATION_RE.search(unit):
            reasons.append("direct_answer_declaration")
        if _answer_mentions(unit, answer):
            reasons.append("answer_literal_or_equivalent")
        if reasons:
            reason_counts.update(reasons)
            continue
        compact = re.sub(r"\s+", " ", unit).strip()
        if compact and not _CONCLUSION_ONLY_RE.fullmatch(compact):
            kept.append(compact)

    sanitized = "\n".join(kept).strip()
    pre_audit = audit_answer_redaction(normalized, answer)
    post_audit: dict[str, Any]
    if sanitized:
        post_audit = audit_answer_redaction(sanitized, answer)
    else:
        post_audit = {
            "schema_version": PRIVILEGED_REDACTION_SCHEMA_VERSION,
            "text_sha256": _sha256(""),
            "text_characters": 0,
            "safe": False,
            "boxed_construct_count": 0,
            "direct_answer_declaration_count": 0,
            "answer_mention_count": 0,
            "answer_mention_kinds": {},
        }

    audit = {
        "schema_version": PRIVILEGED_REDACTION_SCHEMA_VERSION,
        "safe": bool(sanitized) and bool(post_audit["safe"]),
        "pre_redaction_sha256": _sha256(raw),
        "normalized_pre_redaction_sha256": _sha256(normalized),
        "post_redaction_sha256": _sha256(sanitized),
        "pre_redaction_characters": len(raw),
        "post_redaction_characters": len(sanitized),
        "boxed_constructs_removed": boxed_removed,
        "malformed_boxed_constructs_removed": malformed_boxes_removed,
        "structured_fields_removed": removed_structured_fields,
        "removed_reason_counts": dict(sorted(reason_counts.items())),
        "redaction_count": (
            boxed_removed + len(removed_structured_fields) + sum(reason_counts.values())
        ),
        "pre_audit": pre_audit,
        "post_audit": post_audit,
    }
    if not sanitized:
        raise PrivilegedRedactionError(
            "Answer redaction removed the entire privileged advantage text",
            audit=audit,
        )
    if not post_audit["safe"]:
        raise PrivilegedRedactionError(
            "Could not certify that privileged advantage text is answer-free",
            audit=audit,
        )
    return sanitized, audit


def build_privileged_evaluation_prompt(
    problem: str,
    advantage_text: str,
    target_answer: str,
    *,
    redaction_audit: Mapping[str, Any] | None = None,
) -> str:
    """Build safe user-message content from a certified sanitized text.

    The caller must still render this content with the same chat template used
    by the Base and CSD conditions (for example, ``problem_prompt(tokenizer,
    content)``).  Keeping tokenizer rendering outside this dependency-light
    module prevents a subtly different chat wrapper from confounding the
    privileged control.
    """

    problem_text = _required_text(problem, "problem")
    advantage = _required_text(advantage_text, "advantage_text")
    answer = _required_text(target_answer, "target_answer")
    fresh_audit = audit_answer_redaction(advantage, answer)
    if not fresh_audit["safe"]:
        raise PrivilegedRedactionError(
            "Refusing to build a privileged evaluation prompt from unsafe text",
            audit=fresh_audit,
        )
    if redaction_audit is not None:
        if redaction_audit.get("safe") is not True:
            raise PrivilegedRedactionError("Supplied redaction audit is not safe")
        declared_hash = str(redaction_audit.get("post_redaction_sha256", ""))
        if declared_hash != _sha256(advantage):
            raise PrivilegedRedactionError(
                "Sanitized advantage text does not match its redaction audit digest"
            )
    return _EVALUATION_TEMPLATE.format(
        problem=problem_text,
        advantage_text=advantage,
    )


def build_privileged_evaluation_problem(
    problem: str,
    advantage_text: str,
    target_answer: str,
    *,
    redaction_audit: Mapping[str, Any] | None = None,
) -> str:
    """Explicitly named alias for the augmented problem/user-message content."""

    return build_privileged_evaluation_prompt(
        problem,
        advantage_text,
        target_answer,
        redaction_audit=redaction_audit,
    )


def build_privileged_control_artifact(
    problem: str,
    target_answer: str,
    raw_advantage_text: str,
) -> dict[str, Any]:
    """Build the JSON-serializable context artifact consumed by Task 1."""

    generation_messages = build_privileged_cot_generation_messages(
        problem, target_answer
    )
    advantage_text, redaction_audit = sanitize_privileged_advantage_text(
        raw_advantage_text, target_answer
    )
    evaluation_problem = build_privileged_evaluation_problem(
        problem,
        advantage_text,
        target_answer,
        redaction_audit=redaction_audit,
    )
    provenance = {
        "construction_sources": ["original_query", "target_answer"],
        "evaluation_context_sources": [
            "original_query",
            "answer_redacted_correct_cot",
        ],
        "forbidden_source_ancestry": ["target_answer", "future_target_tokens"],
        "construction_used_target_answer": True,
        "literal_target_answer_in_advantage_text": False,
        "hindsight_exposed": True,
        "hindsight_exposure_rate": 1.0,
        "context_prefix_parity": 0.0,
        "hindsight_free_score": 0.0,
    }
    return {
        "schema_version": PRIVILEGED_CONTEXT_SCHEMA_VERSION,
        "control_mode": PRIVILEGED_CONTROL_MODE,
        "advantage_text": advantage_text,
        "advantage_text_pre_redaction_sha256": _sha256(
            _required_text(raw_advantage_text, "raw_advantage_text")
        ),
        "advantage_text_sha256": _sha256(advantage_text),
        "generation_prompt_sha256": _canonical_json_sha256(generation_messages),
        "evaluation_problem": evaluation_problem,
        "evaluation_problem_sha256": _sha256(evaluation_problem),
        "redaction_audit": redaction_audit,
        "context_provenance": provenance,
        "context_provenance_sha256": _canonical_json_sha256(provenance),
    }


__all__ = [
    "PRIVILEGED_CONTEXT_SCHEMA_VERSION",
    "PRIVILEGED_CONTROL_MODE",
    "PRIVILEGED_REDACTION_SCHEMA_VERSION",
    "PrivilegedRedactionError",
    "audit_answer_redaction",
    "build_privileged_control_artifact",
    "build_privileged_cot_generation_messages",
    "build_privileged_evaluation_problem",
    "build_privileged_evaluation_prompt",
    "sanitize_privileged_advantage_text",
]
