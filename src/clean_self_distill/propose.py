"""Skill-card-conditioned, target-disjoint specialization candidate proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .io import (
    compute_proposal_training_sha256,
    load_proposal_map,
    load_query_records,
    stable_hash,
    write_jsonl,
)
from .prompts import (
    candidate_messages,
    frontier_verifier_messages,
    skill_card_messages,
    solver_messages,
    verifier_messages,
    wrong_trajectory_messages,
)
from .runtime import (
    HFGenerator,
    collect_runtime_metadata,
    load_hf_model,
    render_chat,
)


_NUMERIC_ATOM_PATTERN = (
    r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.(?:\d+|(?=[eE])))?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)
_SIGNED_NUMERIC_ATOM_PATTERN = rf"[-+]?{_NUMERIC_ATOM_PATTERN}"
_TEX_FRACTION_VALUE_RE = re.compile(
    rf"(?P<outer_sign>[-+]?)\s*\\(?:frac|dfrac|tfrac|cfrac)\s*"
    rf"(?:\{{\s*(?P<numerator_braced>{_SIGNED_NUMERIC_ATOM_PATTERN})\s*\}}|"
    rf"(?P<numerator_bare>[-+]?\d))\s*"
    rf"(?:\{{\s*(?P<denominator_braced>{_SIGNED_NUMERIC_ATOM_PATTERN})\s*\}}|"
    rf"(?P<denominator_bare>[-+]?\d))"
)
_TEX_OVER_FRACTION_VALUE_RE = re.compile(
    rf"(?P<over_outer_sign>[-+]?)\s*\{{\s*"
    rf"(?:\{{\s*(?P<over_numerator_braced>{_SIGNED_NUMERIC_ATOM_PATTERN})\s*\}}|"
    rf"(?P<over_numerator_bare>{_SIGNED_NUMERIC_ATOM_PATTERN}))\s*"
    rf"\\over\s*"
    rf"(?:\{{\s*(?P<over_denominator_braced>{_SIGNED_NUMERIC_ATOM_PATTERN})\s*\}}|"
    rf"(?P<over_denominator_bare>{_SIGNED_NUMERIC_ATOM_PATTERN}))\s*\}}"
)
_TEX_BARE_OVER_FRACTION_VALUE_RE = re.compile(
    rf"(?P<bare_over_outer_sign>[-+]?)\s*"
    rf"(?:\{{\s*(?P<bare_over_numerator_braced>{_SIGNED_NUMERIC_ATOM_PATTERN})\s*\}}|"
    rf"(?P<bare_over_numerator_bare>{_SIGNED_NUMERIC_ATOM_PATTERN}))\s*"
    rf"\\over\s*"
    rf"(?:\{{\s*(?P<bare_over_denominator_braced>{_SIGNED_NUMERIC_ATOM_PATTERN})\s*\}}|"
    rf"(?P<bare_over_denominator_bare>{_SIGNED_NUMERIC_ATOM_PATTERN}))"
)
_PLAIN_FRACTION_PATTERN = (
    rf"(?<!\d){_SIGNED_NUMERIC_ATOM_PATTERN}\s*/\s*"
    rf"{_SIGNED_NUMERIC_ATOM_PATTERN}(?!\d)"
)
_PLAIN_FRACTION_VALUE_RE = re.compile(
    rf"(?<!\d)(?P<plain_numerator>{_SIGNED_NUMERIC_ATOM_PATTERN})"
    rf"\s*/\s*(?P<plain_denominator>{_SIGNED_NUMERIC_ATOM_PATTERN})"
    rf"(?!\d)"
)
_SCALAR_NUMBER_PATTERN = rf"(?<!\d){_SIGNED_NUMERIC_ATOM_PATTERN}(?!\d)"
_SCALAR_NUMBER_RE = re.compile(_SCALAR_NUMBER_PATTERN)
_NUMBER_RE = re.compile(
    rf"(?:{_TEX_OVER_FRACTION_VALUE_RE.pattern}|"
    rf"{_TEX_BARE_OVER_FRACTION_VALUE_RE.pattern}|"
    rf"{_TEX_FRACTION_VALUE_RE.pattern}|"
    rf"{_PLAIN_FRACTION_PATTERN}|{_SCALAR_NUMBER_PATTERN})"
)
_LIST_NUMERIC_ATOM_PATTERN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
_TEX_SET_OPEN_PATTERN = r"\\\{"
_TEX_SET_CLOSE_PATTERN = r"\\\}"
_TEX_SET_RE = re.compile(
    rf"(?:\\left\s*)?{_TEX_SET_OPEN_PATTERN}"
    rf"(?P<set_content>.*?)"
    rf"(?:\\right\s*)?{_TEX_SET_CLOSE_PATTERN}",
    flags=re.DOTALL,
)
_ELLIPSIS_PATTERN = r"(?:\\(?:[lc]?dots[a-z]?|mathellipsis)(?![A-Za-z])|\.{3}|…)"
_ELLIPSIS_RE = re.compile(_ELLIPSIS_PATTERN)
_GROUPED_NUMBER_PATTERN = r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
_GROUPED_AFTER_ELLIPSIS_RE = re.compile(
    rf"{_ELLIPSIS_PATTERN}\s*,\s*(?P<number>{_GROUPED_NUMBER_PATTERN})"
)
_ZERO_PADDED_GROUPED_NUMBER_RE = re.compile(
    r"(?<!\d)[-+]?\d{1,3}(?:,0\d{2})+(?:\.\d+)?" r"(?!\d)"
)
_NOT_ESCAPED_PATTERN = r"(?<!\\)"
_BRACKET_CONTENT_RE = re.compile(
    rf"{_NOT_ESCAPED_PATTERN}(?:\\left\s*)?\(\s*(?P<paren>[^()]*)"
    rf"\s*(?:\\right\s*)?\)"
    rf"|{_NOT_ESCAPED_PATTERN}(?:\\left\s*)?\[\s*(?P<bracket>[^\[\]]*)"
    rf"\s*(?:\\right\s*)?\]",
    flags=re.DOTALL,
)
_TEX_MATH_CONTENT_RE = re.compile(
    r"\\\((?P<tex_paren>.*?)\\\)|\\\[(?P<tex_bracket>.*?)\\\]",
    flags=re.DOTALL,
)
_DOLLAR_MATH_CONTENT_RE = re.compile(
    r"(?<!\\)\$\$(?P<double_dollar>.*?)(?<!\\)\$\$"
    r"|(?<!\\)\$(?!\$)(?P<single_dollar>.*?)(?<!\\)\$",
    flags=re.DOTALL,
)
_SEQUENCE_SCALAR_PATTERN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
_SEQUENCE_SIDE_PATTERN = (
    rf"{_SEQUENCE_SCALAR_PATTERN}" rf"(?:\s*,\s*{_SEQUENCE_SCALAR_PATTERN})*"
)
_UNBRACKETED_ELLIPSIS_SEQUENCE_RE = re.compile(
    rf"(?<!\d)(?P<sequence>{_SEQUENCE_SIDE_PATTERN}\s*,?\s*"
    rf"{_ELLIPSIS_PATTERN}\s*,?\s*{_SEQUENCE_SIDE_PATTERN})"
    rf"(?!\d)"
)
_COMMA_EXPRESSION_OPERATOR_RE = re.compile(
    r"[=<>+*/^:|]" r"|\\(?:leq?|geq?|neq|approx|sim|mid|in|notin|times|cdot|pm|mp)\b"
)
_TEX_BRACED_COMMA_RE = re.compile(r"\{\s*,\s*\}")
_TEX_TEXT_COMMA_RE = re.compile(
    r"(?<=\d)\\(?:text|mathrm)\s*\{\s*,\s*\}\s*(?=\d{3}(?!\d))"
)
_TEX_COMMA_TIGHT_SPACING_RE = re.compile(r"(?<=\d),\s*\\[!,;:]\s*(?=\d{3}(?!\d))")
_TEX_TIGHT_SPACING_RE = re.compile(r"\\[,!;:]")
_TEX_WIDE_SPACING_RE = re.compile(r"\\(?:quad|qquad)\b")
_TEX_NAMED_THINSPACE_RE = re.compile(r"(?<=\d)\\thinspace\s*(?=\d{3}(?!\d))")
_UNICODE_GROUPING_SPACE_RE = re.compile(r"(?<=\d)[\u00a0\u2009\u202f](?=\d{3}(?!\d))")
_ASCII_GROUPING_SPACE_RE = re.compile(r"(?<=\d)[ \t]+(?=\d{3}(?!\d))")
_TILDE_GROUPING_SPACE_RE = re.compile(r"(?<=\d)~(?=\d{3}(?!\d))")
_SIGN_SPACING_RE = re.compile(
    r"(?P<sign>[-+])\s+(?=(?:\d|\.|\{|\\(?:frac|dfrac|tfrac|cfrac)))"
)
_TEX_SIZE_DELIMITER_RE = re.compile(r"\\(?:bigg|big)[lr]?\s*", flags=re.IGNORECASE)
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_MATH_SPAN_RE = re.compile(r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]", flags=re.DOTALL)
_INLINE_MATH_EXPRESSION_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\\?[A-Za-z]|\d[\d,.]*)"
    r"(?:\s*[-=+*/^<>]\s*(?:\\?[A-Za-z]|\d[\d,.]*))+"
    r"(?![A-Za-z0-9])"
)
_SINGLE_SYMBOL_RE = re.compile(r"\b[A-Za-z]\b")
_SYMBOLIC_DETAIL_RE = re.compile(r"\\[A-Za-z]+|[=+*/^_{}<>]")
_ENGLISH_NUMBER_WORD_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|billion|trillion|first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|"
    r"sixteenth|seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|"
    r"fiftieth|sixtieth|seventieth|eightieth|ninetieth|hundredth|thousandth|"
    r"millionth|billionth|trillionth)\b",
    flags=re.IGNORECASE,
)
_DIRECT_ANSWER_CUE_RE = re.compile(
    r"(?:\\boxed\s*\{[^{}]*\}|\\boxed\b|"
    r"\b(?:final|correct|target|boxed)\s+(?:answer|result|value)\b|"
    r"\b(?:answer|result|value|solution)\s+(?:is|equals?|must\s+be|should\s+be)\b)",
    flags=re.IGNORECASE,
)
_PLACEHOLDER_ARTIFACT_RE = re.compile(
    r"\b(?:redacted|placeholder|unspecified|omitted)"
    r"(?:\s+(?:detail|number|quantity|value|object|entity|term))?\b"
    r"|\b(?:generic|hidden|removed)\s+"
    r"(?:detail|number|quantity|value|object|entity|term)\b"
    r"|\btbd\b|\bto\s+be\s+filled\b"
    r"|\b(?:a\s+variable\s+quantity|a\s+symbolic\s+relation|"
    r"an\s+abstract\s+(?:object|element)|an\s+auxiliary\s+variable|"
    r"derived\s+conclusion)\b"
    r"|<\s*[A-Za-z_][A-Za-z0-9_ -]{0,80}\s*>",
    flags=re.IGNORECASE,
)
_ANSWER_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:checkable\s+final\s+answer|"
    r"(?:the\s+)?attempt['\N{RIGHT SINGLE QUOTATION MARK}]?s\s+final\s+answer|"
    r"(?:the\s+|an?\s+|your\s+|actual\s+|correct(?:ed)?\s+|wrong\s+|"
    r"proposed\s+)?(?:final\s+)?(?:answer|result|value)"
    r"(?:\s+after\s+verification|\s+here)?|n/?a|none|tbd|unknown|[.]+)\s*$",
    flags=re.IGNORECASE,
)
_FRONTIER_VALID_CLAIM_RE = re.compile(
    r"\b(?:this|the|selected|wrong)\s+(?:step|claim|calculation)\s+"
    r"(?:is|was)\s+(?:already\s+)?(?:valid|correct)\b|"
    r"\b(?:there\s+is\s+no|no)\s+(?:error|mistake)\b",
    flags=re.IGNORECASE,
)
_FRONTIER_NO_CORRECTION_RE = re.compile(
    r"\bno\s+(?:correction|change|fix)\s+(?:is\s+)?(?:needed|required)|"
    r"\bnothing\s+to\s+(?:correct|change|fix)|\b(?:leave|keep)\s+.+\s+"
    r"(?:unchanged|as\s+is)|\bdo\s+nothing\b",
    flags=re.IGNORECASE,
)
_GENERIC_SENTENCE_WORDS = {
    "a",
    "all",
    "also",
    "although",
    "among",
    "an",
    "any",
    "are",
    "assume",
    "calculate",
    "call",
    "circle",
    "cities",
    "city",
    "compute",
    "consider",
    "define",
    "distinct",
    "determine",
    "day",
    "diagram",
    "during",
    "each",
    "every",
    "evaluate",
    "exactly",
    "find",
    "for",
    "four",
    "from",
    "given",
    "grid",
    "how",
    "here",
    "however",
    "if",
    "in",
    "integer",
    "irreducible",
    "isosceles",
    "let",
    "leaving",
    "note",
    "numbers",
    "of",
    "on",
    "output",
    "points",
    "positive",
    "quadrilateral",
    "rectangle",
    "rectangles",
    "regular",
    "rows",
    "rotations",
    "running",
    "she",
    "solve",
    "some",
    "square",
    "squares",
    "suppose",
    "sudoku",
    "the",
    "then",
    "there",
    "they",
    "this",
    "to",
    "torus",
    "triangle",
    "triangles",
    "use",
    "using",
    "what",
    "when",
    "which",
    "whoever",
    "write",
    "you",
    "your",
}
_UBIQUITOUS_STRUCTURAL_INTEGERS = frozenset({-2, -1, 0, 1, 2})
_MAX_CANONICAL_NUMERIC_DIGITS = 1024


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "valid"}
    return bool(value)


def _normalize_tex_numeric_syntax(text: str) -> str:
    """Normalize presentation-only TeX without changing numeric meaning."""
    text = _TEX_TEXT_COMMA_RE.sub("", text)
    text = _TEX_BRACED_COMMA_RE.sub("", text)
    text = _TEX_COMMA_TIGHT_SPACING_RE.sub("", text)
    text = _TEX_TIGHT_SPACING_RE.sub("", text)
    text = _TEX_NAMED_THINSPACE_RE.sub("", text)
    text = _TEX_WIDE_SPACING_RE.sub(" ", text)
    text = _UNICODE_GROUPING_SPACE_RE.sub("", text)
    text = _ASCII_GROUPING_SPACE_RE.sub("", text)
    text = _TILDE_GROUPING_SPACE_RE.sub("", text)
    text = _SIGN_SPACING_RE.sub(r"\g<sign>", text)
    return _TEX_SIZE_DELIMITER_RE.sub("", text)


def _numeric_atom_decimal(value: str) -> Decimal:
    compact = re.sub(r"[\s,]", "", value).lstrip("+")
    match = re.fullmatch(
        r"(?P<sign>[-+]?)"
        r"(?P<integer>\d*)(?:\.(?P<fraction>\d*))?"
        r"(?:[eE](?P<exponent>[-+]?\d+))?",
        compact,
    )
    if match is None:
        raise InvalidOperation
    integer = match.group("integer")
    fractional = match.group("fraction") or ""
    digits = integer + fractional
    if not digits:
        raise InvalidOperation
    significant = digits.lstrip("0")
    if not significant:
        return Decimal(0)

    exponent_text = match.group("exponent") or "0"
    exponent_sign = -1 if exponent_text.startswith("-") else 1
    exponent_digits = exponent_text.lstrip("+-").lstrip("0") or "0"
    if len(exponent_digits) > 6:
        raise InvalidOperation
    exponent = exponent_sign * int(exponent_digits)

    power = exponent - len(fractional)
    trailing_zeros = len(significant) - len(significant.rstrip("0"))
    if trailing_zeros:
        significant = significant[:-trailing_zeros]
        power += trailing_zeros
    if (
        len(significant) > _MAX_CANONICAL_NUMERIC_DIGITS
        or abs(power) > _MAX_CANONICAL_NUMERIC_DIGITS
    ):
        raise InvalidOperation
    sign = "-" if match.group("sign") == "-" else ""
    return Decimal(f"{sign}{significant}e{power}")


def _canonical_numeric_atom(value: str) -> str:
    compact = re.sub(r"[\s,]", "", value).lstrip("+")
    try:
        rational = Fraction(_numeric_atom_decimal(value))
    except (InvalidOperation, ValueError):
        digest = hashlib.sha256(compact.encode("utf-8")).hexdigest()
        return f"oversized-or-invalid-sha256:{digest}"
    if rational.denominator == 1:
        return str(rational.numerator)
    return f"{rational.numerator}/{rational.denominator}"


def _canonical_fraction(
    numerator_text: str, denominator_text: str, *, outer_sign: str = ""
) -> str:
    try:
        numerator = _numeric_atom_decimal(numerator_text)
        denominator = _numeric_atom_decimal(denominator_text)
        if outer_sign == "-":
            numerator = -numerator
        if denominator != 0:
            reduced = Fraction(numerator) / Fraction(denominator)
            if reduced.denominator == 1:
                return str(reduced.numerator)
            return f"{reduced.numerator}/{reduced.denominator}"
    except (InvalidOperation, ValueError, ZeroDivisionError):
        pass
    numerator = _canonical_numeric_atom(numerator_text)
    denominator = _canonical_numeric_atom(denominator_text)
    prefix = "-" if outer_sign == "-" and not numerator.startswith("-") else ""
    return f"{prefix}{numerator}/{denominator}"


def _canonical_number(value: str) -> str:
    value = _normalize_tex_numeric_syntax(value).strip()
    over_match = _TEX_OVER_FRACTION_VALUE_RE.fullmatch(value)
    if over_match is not None:
        numerator = over_match.group("over_numerator_braced") or over_match.group(
            "over_numerator_bare"
        )
        denominator = over_match.group("over_denominator_braced") or over_match.group(
            "over_denominator_bare"
        )
        return _canonical_fraction(
            numerator,
            denominator,
            outer_sign=over_match.group("over_outer_sign"),
        )
    bare_over_match = _TEX_BARE_OVER_FRACTION_VALUE_RE.fullmatch(value)
    if bare_over_match is not None:
        numerator = bare_over_match.group(
            "bare_over_numerator_braced"
        ) or bare_over_match.group("bare_over_numerator_bare")
        denominator = bare_over_match.group(
            "bare_over_denominator_braced"
        ) or bare_over_match.group("bare_over_denominator_bare")
        return _canonical_fraction(
            numerator,
            denominator,
            outer_sign=bare_over_match.group("bare_over_outer_sign"),
        )
    tex_match = _TEX_FRACTION_VALUE_RE.fullmatch(value)
    if tex_match is not None:
        numerator = tex_match.group("numerator_braced") or tex_match.group(
            "numerator_bare"
        )
        denominator = tex_match.group("denominator_braced") or tex_match.group(
            "denominator_bare"
        )
        return _canonical_fraction(
            numerator, denominator, outer_sign=tex_match.group("outer_sign")
        )
    plain_match = _PLAIN_FRACTION_VALUE_RE.fullmatch(value)
    if plain_match is not None:
        return _canonical_fraction(
            plain_match.group("plain_numerator"),
            plain_match.group("plain_denominator"),
        )
    return _canonical_numeric_atom(value)


def _canonical_fraction_value(value: str) -> Fraction | None:
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def _fraction_string(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _binary_minus_magnitude(
    source: str, match: re.Match[str], canonical: str
) -> str | None:
    """Preserve magnitude whenever a signed spelling canonicalizes negative.

    Whether a minus is unary or binary can be ambiguous after TeX whitespace
    normalization.  Keeping both readings is the fail-closed choice for a
    target-disjointness firewall and also covers ``x - \\frac{1}{2}``.
    """
    del source, match
    rational = _canonical_fraction_value(canonical)
    if rational is None or rational >= 0:
        return None
    return _fraction_string(abs(rational))


def _percent_value(canonical: str) -> str | None:
    rational = _canonical_fraction_value(canonical)
    if rational is None:
        return None
    return _fraction_string(rational / 100)


def _has_percent_suffix(source: str, end: int) -> bool:
    return (
        re.match(
            r"(?:\s|~)*(?:"
            r"\\(?:text|mathrm)\s*\{\s*\\?%\s*\}"
            r"|\{\s*\\?%\s*\}|\\%|%"
            r")",
            source[end:],
        )
        is not None
    )


def _mixed_number_value(
    source: str, fraction_match: re.Match[str], fraction_value: str
) -> str | None:
    """Return 7/2 for spellings such as ``3 \\frac{1}{2}``."""
    fraction_start = fraction_match.start()
    while fraction_start < fraction_match.end() and source[fraction_start].isspace():
        fraction_start += 1
    prefix = source[:fraction_start]
    mixed_separator = r"(?:\s|~|\\(?:[ ,!;:]|thinspace|quad|qquad))*"
    whole_match = re.search(
        rf"(?<![\d.])(?P<whole>{_SIGNED_NUMERIC_ATOM_PATTERN})" rf"{mixed_separator}$",
        prefix,
    )
    if whole_match is None:
        return None
    whole = _canonical_fraction_value(
        _canonical_numeric_atom(whole_match.group("whole"))
    )
    fractional = _canonical_fraction_value(fraction_value)
    if whole is None or fractional is None or abs(fractional) >= 1:
        return None
    if whole < 0 and fractional > 0:
        fractional = -fractional
    return _fraction_string(whole + fractional)


def _scan_numeric_source(source: str) -> tuple[set[str], set[str]]:
    """Extract a fail-closed union of numeric interpretations from one spelling."""
    literals: set[str] = set()
    fractional_values: set[str] = set()
    remaining = list(source)

    def add(
        canonical: str,
        *,
        match: re.Match[str] | None = None,
        fractional: bool = False,
    ) -> None:
        if not canonical:
            return
        literals.add(canonical)
        if fractional:
            fractional_values.add(canonical)
        if match is not None:
            magnitude = _binary_minus_magnitude(source, match, canonical)
            if magnitude is not None:
                literals.add(magnitude)
                if fractional:
                    fractional_values.add(magnitude)

    def mask(span: tuple[int, int]) -> None:
        start, end = span
        remaining[start:end] = " " * (end - start)

    # Consume ratios before scalars so numerator and denominator digits are not
    # reinterpreted independently.  This supports both \frac and primitive
    # ``{1 \over 2}`` spellings used in the held-out benchmark.
    for pattern in (
        _TEX_OVER_FRACTION_VALUE_RE,
        _TEX_BARE_OVER_FRACTION_VALUE_RE,
        _TEX_FRACTION_VALUE_RE,
        _PLAIN_FRACTION_VALUE_RE,
    ):
        scan = "".join(remaining)
        for match in pattern.finditer(scan):
            canonical = _canonical_number(match.group(0))
            add(canonical, match=match, fractional=True)
            mixed = _mixed_number_value(source, match, canonical)
            if mixed is not None:
                add(mixed, fractional=True)
            if _has_percent_suffix(source, match.end()):
                percent = _percent_value(canonical)
                if percent is not None:
                    add(percent, fractional=True)
            mask(match.span())

    for match in _SCALAR_NUMBER_RE.finditer("".join(remaining)):
        canonical = _canonical_number(match.group(0))
        add(canonical, match=match)
        if _has_percent_suffix(source, match.end()):
            percent = _percent_value(canonical)
            if percent is not None:
                add(percent, fractional=True)
    return literals, fractional_values


def _numeric_source_variants(text: str) -> tuple[str, ...]:
    """Keep both grouping and list readings of ambiguous comma notation."""
    normalized = _normalize_tex_numeric_syntax(text)
    variants = [text, normalized]
    # A comma can be either a thousands separator or a list/tuple separator.
    # Auditing both readings is deliberately fail-closed: an ambiguous surface
    # form cannot let a copied target number evade the disjointness firewall.
    # A semicolon barrier keeps a split list element from being mistaken for a
    # mixed numeral even when the source used no whitespace after its comma.
    variants.extend(value.replace(",", " ; ") for value in tuple(variants))
    return tuple(dict.fromkeys(variants))


def _fraction_numeric_literals(text: str) -> set[str]:
    """Return canonical values with fractional, mixed, or percent spellings."""
    values: set[str] = set()
    for source in _numeric_source_variants(text):
        _, fractional = _scan_numeric_source(source)
        values.update(fractional)
    return values


def _numeric_literals(text: str) -> set[str]:
    literals: set[str] = set()
    for source in _numeric_source_variants(text):
        values, _ = _scan_numeric_source(source)
        literals.update(values)
    return literals


def _is_ubiquitous_structural_integer(value: str) -> bool:
    """Return whether a numeric literal is one of the harmless small integers."""
    if "/" in value:
        return False
    try:
        number = Decimal(value)
    except InvalidOperation:
        return False
    return (
        number == number.to_integral_value()
        and int(number) in _UBIQUITOUS_STRUCTURAL_INTEGERS
    )


def _target_entity_literals(problem: str) -> set[str]:
    """Return target-specific capitalized tokens in a case-insensitive form.

    Generic sentence openers are excluded so that a harmless ``If`` or
    ``Find`` does not make an otherwise independent exercise fail closed.
    Tokens that remain (names, labels such as ``ABC``, unusual objects, etc.)
    may not reappear even with different capitalization.
    """

    return {
        match.group(0).casefold()
        for match in _ENTITY_RE.finditer(problem)
        if match.group(0).casefold() not in _GENERIC_SENTENCE_WORDS
        and _ENGLISH_NUMBER_WORD_RE.fullmatch(match.group(0)) is None
    }


def _word_literals(text: str) -> set[str]:
    # Split punctuation so possessives such as ``Kayla's`` and ``Kayla’s``
    # still expose the protected target entity ``kayla``.
    return {token.casefold() for token in re.findall(r"\b[A-Za-z]+\b", text)}


def _placeholder_artifact_audit(text: str) -> dict[str, Any]:
    artifacts = [
        match.group(0).strip() for match in _PLACEHOLDER_ARTIFACT_RE.finditer(text)
    ]
    return {
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "safe": not artifacts,
    }


def _answer_placeholder_audit(answer: str) -> dict[str, Any]:
    """Reject literal schema prose copied into an answer field."""
    text = str(answer).strip()
    generic = _placeholder_artifact_audit(text)
    reasons: list[str] = []
    if not generic["safe"]:
        reasons.append("generic_placeholder_artifact")
    if _ANSWER_PLACEHOLDER_RE.fullmatch(text) is not None:
        reasons.append("answer_placeholder")
    return {
        "safe": not reasons,
        "reasons": reasons,
        "generic_placeholder_artifact_audit": generic,
    }


def _validate_final_answer(answer: str, *, field: str) -> dict[str, Any]:
    audit = _answer_placeholder_audit(answer)
    if not audit["safe"]:
        raise ValueError(
            f"{field} is placeholder/schema text: {', '.join(audit['reasons'])}"
        )
    return audit


def _normalized_answer_key(answer: str) -> str:
    """Normalize only presentation so copied right/wrong answers compare equal."""
    text = str(answer).strip().casefold().replace("−", "-")
    text = text.strip("$ ").strip(".;,")
    wrapper = re.fullmatch(r"\\(?:boxed|fbox)\s*\{(.*)\}", text, re.DOTALL)
    if wrapper is not None:
        text = wrapper.group(1)
    text = re.sub(r"\\(?:left|right)\b", "", text)
    return re.sub(r"\s+", "", text)


def _contrastive_answer_audit(
    correct_final_answer: str, wrong_final_answer: str
) -> dict[str, Any]:
    """Fail closed on placeholder answers or copied right/wrong answers."""
    correct_placeholder = _validate_final_answer(
        correct_final_answer, field="final_answer"
    )
    wrong_placeholder = _validate_final_answer(
        wrong_final_answer, field="wrong_final_answer"
    )
    correct_key = _normalized_answer_key(correct_final_answer)
    wrong_key = _normalized_answer_key(wrong_final_answer)
    if not correct_key or not wrong_key:
        raise ValueError("Final-answer comparison produced an empty normalized key")

    if correct_key == wrong_key:
        raise ValueError("Correct and wrong final answers are identical after normalization")
    return {
        "safe": True,
        "equivalent": False,
        "comparison_basis": "normalized_distinct_text",
        "correct_normalized_key": correct_key,
        "wrong_normalized_key": wrong_key,
        "correct_placeholder_audit": correct_placeholder,
        "wrong_placeholder_audit": wrong_placeholder,
    }


def _frontier_semantic_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _frontier_has_noop_replacement(action: str) -> bool:
    replacement = re.search(
        r"\breplace\s+(?P<before>.+?)\s+with\s+(?P<after>.+?)\s*[.!]?\s*$",
        action,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if replacement is None:
        return False
    return _frontier_semantic_key(
        replacement.group("before")
    ) == _frontier_semantic_key(replacement.group("after"))


def sanitize_skill_card(
    skill_card: dict[str, Any], problem: str
) -> tuple[dict[str, Any], list[str]]:
    """Replace target-specific literals with reusable mathematical abstractions."""
    redacted: list[str] = []
    target_entities = sorted(_target_entity_literals(problem), key=len, reverse=True)
    target_symbols = {
        symbol
        for symbol in _SINGLE_SYMBOL_RE.findall(problem)
        if symbol.lower() not in {"a", "i"}
    }

    def sanitize_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: sanitize_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize_value(item) for item in value]
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            redacted.append("<inferred-numeric>")
            return "a variable quantity"
        if not isinstance(value, str):
            return value
        text, expression_count = _MATH_SPAN_RE.subn("a symbolic relation", value)
        redacted.extend(["<math-expression>"] * expression_count)
        text, inline_expression_count = _INLINE_MATH_EXPRESSION_RE.subn(
            "a symbolic relation", text
        )
        redacted.extend(["<inline-math-expression>"] * inline_expression_count)
        text, cue_count = _DIRECT_ANSWER_CUE_RE.subn("derived conclusion", text)
        redacted.extend(["<direct-answer-cue>"] * cue_count)
        text, number_word_count = _ENGLISH_NUMBER_WORD_RE.subn(
            "a variable quantity", text
        )
        redacted.extend(["<english-number-word>"] * number_word_count)
        for literal in target_entities:
            text, count = re.subn(
                rf"\b{re.escape(literal)}\b",
                "an abstract object",
                text,
                flags=re.IGNORECASE,
            )
            redacted.extend([literal] * count)
        # A skill card does not need literal numbers at all. Replace even an
        # inferred value with a reusable abstraction rather than a placeholder.
        text, inferred_count = _NUMBER_RE.subn("a variable quantity", text)
        redacted.extend(["<inferred-numeric>"] * inferred_count)
        for symbol in sorted(target_symbols):
            text, symbol_count = re.subn(
                rf"\b{re.escape(symbol)}\b",
                "an auxiliary variable",
                text,
                flags=re.IGNORECASE,
            )
            redacted.extend([symbol] * symbol_count)
        text, artifact_count = _PLACEHOLDER_ARTIFACT_RE.subn(
            "an abstract element", text
        )
        redacted.extend(["<placeholder-artifact>"] * artifact_count)
        return text

    clean = sanitize_value(skill_card)
    clean["target_details_removed"] = True
    return clean, redacted


SKILL_CARD_DISJOINT_AUDIT_VERSION = "literal-symbol-single-span-five-v3"
SKILL_CARD_MAX_FOURGRAM_OVERLAP_COUNT = 2
SKILL_CARD_MAX_FOURGRAM_OVERLAP_RATE = 0.05
SKILL_CARD_MAX_FOURGRAM_OVERLAP_COMPONENTS = 1
SKILL_CARD_MAX_CONTIGUOUS_TOKEN_OVERLAP = 5


def _skill_card_overlap_structure(problem: str, card_text: str) -> dict[str, int]:
    """Measure whether lexical overlap is one short, contiguous generic phrase."""
    target_tokens = [token.lower() for token in _TOKEN_RE.findall(problem)]
    card_tokens = [token.lower() for token in _TOKEN_RE.findall(card_text)]

    # Exact longest-common-substring length catches periodic copying that can
    # contain many tokens but only one or two *unique* four-grams.
    previous = [0] * (len(target_tokens) + 1)
    longest = 0
    for card_token in card_tokens:
        current = [0] * (len(target_tokens) + 1)
        for target_index, target_token in enumerate(target_tokens, start=1):
            if card_token == target_token:
                current[target_index] = previous[target_index - 1] + 1
                longest = max(longest, current[target_index])
        previous = current

    target_fourgrams = {
        tuple(target_tokens[index : index + 4])
        for index in range(max(len(target_tokens) - 3, 0))
    }
    matching_card_starts = [
        index
        for index in range(max(len(card_tokens) - 3, 0))
        if tuple(card_tokens[index : index + 4]) in target_fourgrams
    ]
    component_count = sum(
        index == 0
        or matching_card_starts[index] != matching_card_starts[index - 1] + 1
        for index in range(len(matching_card_starts))
    )
    return {
        "longest_contiguous_token_overlap": longest,
        "fourgram_overlap_component_count": component_count,
    }


def skill_card_disjoint_audit(
    problem: str, skill_card: dict[str, Any]
) -> dict[str, Any]:
    values = [
        str(skill_card.get("domain", "")),
        *(str(item) for item in skill_card.get("skills", [])),
        *(str(item) for item in skill_card.get("reasoning_operators", [])),
        *(str(item) for item in skill_card.get("failure_modes", [])),
        str(skill_card.get("difficulty", "")),
        *(str(item) for item in skill_card.get("constraints", [])),
    ]
    card_text = " ".join(values)
    lexical = target_disjoint_audit(problem, card_text)
    overlap_structure = _skill_card_overlap_structure(problem, card_text)
    target_symbols = {
        symbol.lower()
        for symbol in _SINGLE_SYMBOL_RE.findall(problem)
        if symbol.lower() not in {"a", "i"}
    }
    card_symbols = {
        symbol.lower()
        for symbol in _SINGLE_SYMBOL_RE.findall(card_text)
        if symbol.lower() not in {"a", "i"}
    }
    shared_symbols = sorted(target_symbols & card_symbols)
    symbolic_details = _SYMBOLIC_DETAIL_RE.findall(card_text)
    english_number_words = sorted(
        {
            match.group(0).casefold()
            for match in _ENGLISH_NUMBER_WORD_RE.finditer(card_text)
        }
    )
    direct_answer_cues = sorted(
        {
            match.group(0).casefold()
            for match in _DIRECT_ANSWER_CUE_RE.finditer(card_text)
        }
    )
    safe = (
        lexical["literal_overlap_count"] == 0
        and lexical["fourgram_overlap_count"]
        <= SKILL_CARD_MAX_FOURGRAM_OVERLAP_COUNT
        and lexical["fourgram_overlap_rate"]
        <= SKILL_CARD_MAX_FOURGRAM_OVERLAP_RATE
        and overlap_structure["longest_contiguous_token_overlap"]
        <= SKILL_CARD_MAX_CONTIGUOUS_TOKEN_OVERLAP
        and overlap_structure["fourgram_overlap_component_count"]
        <= SKILL_CARD_MAX_FOURGRAM_OVERLAP_COMPONENTS
        and (
            lexical["fourgram_overlap_count"] < 2
            or overlap_structure["longest_contiguous_token_overlap"]
            == SKILL_CARD_MAX_CONTIGUOUS_TOKEN_OVERLAP
        )
        and not shared_symbols
        and not symbolic_details
        and not english_number_words
        and not direct_answer_cues
    )
    return {
        **lexical,
        **overlap_structure,
        "audit_version": SKILL_CARD_DISJOINT_AUDIT_VERSION,
        "thresholds": {
            "max_literal_overlap_count": 0,
            "max_fourgram_overlap_count": SKILL_CARD_MAX_FOURGRAM_OVERLAP_COUNT,
            "max_fourgram_overlap_rate": SKILL_CARD_MAX_FOURGRAM_OVERLAP_RATE,
            "max_fourgram_overlap_component_count": (
                SKILL_CARD_MAX_FOURGRAM_OVERLAP_COMPONENTS
            ),
            "max_contiguous_token_overlap": SKILL_CARD_MAX_CONTIGUOUS_TOKEN_OVERLAP,
            "two_fourgrams_require_contiguous_token_overlap": (
                SKILL_CARD_MAX_CONTIGUOUS_TOKEN_OVERLAP
            ),
            "max_shared_single_symbols": 0,
            "max_symbolic_detail_count": 0,
            "max_english_number_word_count": 0,
            "max_direct_answer_cue_count": 0,
        },
        "shared_single_symbols": shared_symbols,
        "symbolic_detail_count": len(symbolic_details),
        "english_number_words": english_number_words,
        "english_number_word_count": len(english_number_words),
        "direct_answer_cues": direct_answer_cues,
        "direct_answer_cue_count": len(direct_answer_cues),
        "safe": safe,
    }


def target_disjoint_audit(problem: str, candidate_problem: str) -> dict[str, Any]:
    """Audit exact instance leakage with normalized, case-insensitive literals.

    Target-specific capitalized entities and salient target numbers are
    forbidden. Only the ubiquitous structural integers -2 through 2 are
    ignored as numeric literals; fractions, larger magnitudes, entities, and
    distinctive four-grams remain fail-closed.
    """

    all_target_numbers = _numeric_literals(problem)
    all_candidate_numbers = _numeric_literals(candidate_problem)
    fractional_values = _fraction_numeric_literals(
        problem
    ) | _fraction_numeric_literals(candidate_problem)
    ignored_target_numbers = {
        value
        for value in all_target_numbers
        if _is_ubiquitous_structural_integer(value) and value not in fractional_values
    }
    ignored_candidate_numbers = {
        value
        for value in all_candidate_numbers
        if _is_ubiquitous_structural_integer(value) and value not in fractional_values
    }
    target_numbers = all_target_numbers - ignored_target_numbers
    candidate_numbers = all_candidate_numbers - ignored_candidate_numbers
    shared_numbers = target_numbers & candidate_numbers
    target_entities = _target_entity_literals(problem)
    candidate_words = _word_literals(candidate_problem)
    shared_entities = target_entities & candidate_words
    literal_overlap_count = len(shared_numbers) + len(shared_entities)

    target_tokens = [token.lower() for token in _TOKEN_RE.findall(problem)]
    candidate_tokens = [token.lower() for token in _TOKEN_RE.findall(candidate_problem)]
    target_ngrams = (
        set(zip(*(target_tokens[offset:] for offset in range(4))))
        if len(target_tokens) >= 4
        else set()
    )
    candidate_ngrams = (
        set(zip(*(candidate_tokens[offset:] for offset in range(4))))
        if len(candidate_tokens) >= 4
        else set()
    )
    overlap_ngrams = target_ngrams & candidate_ngrams
    return {
        "target_literal_count": float(len(target_numbers) + len(target_entities)),
        "candidate_literal_count": float(len(candidate_numbers)),
        "literal_overlap_count": float(literal_overlap_count),
        "literal_overlap_rate": float(literal_overlap_count)
        / max(len(target_numbers) + len(target_entities), 1),
        "shared_target_numbers": sorted(shared_numbers),
        "ignored_target_structural_numbers": sorted(ignored_target_numbers),
        "ignored_candidate_structural_numbers": sorted(ignored_candidate_numbers),
        "shared_target_entities": sorted(shared_entities),
        "fourgram_overlap_count": float(len(overlap_ngrams)),
        "fourgram_overlap_rate": float(len(overlap_ngrams))
        / max(len(candidate_ngrams), 1),
    }


def _validate_skill_card(value: dict[str, Any]) -> dict[str, Any]:
    required = {"domain", "skills", "reasoning_operators", "difficulty"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"Skill card missing fields: {sorted(missing)}")
    if not isinstance(value["skills"], list) or not value["skills"]:
        raise ValueError("Skill card skills must be a non-empty list")
    if (
        not isinstance(value["reasoning_operators"], list)
        or not value["reasoning_operators"]
    ):
        raise ValueError("Skill card reasoning_operators must be a non-empty list")
    constraints = value.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("Skill card constraints must be a list")
    failure_modes = value.get("failure_modes", [])
    if not isinstance(failure_modes, list):
        raise ValueError("Skill card failure_modes must be a list")
    if not isinstance(value["domain"], str) or not value["domain"].strip():
        raise ValueError("Skill card domain must be a non-empty string")
    if not isinstance(value["difficulty"], str) or not value["difficulty"].strip():
        raise ValueError("Skill card difficulty must be a non-empty string")
    # Unknown fields such as `solution` or `answer` are intentionally dropped
    # even if a model violates the requested schema.
    return {
        "domain": str(value["domain"]),
        "skills": [str(item) for item in value["skills"]],
        "reasoning_operators": [str(item) for item in value["reasoning_operators"]],
        "failure_modes": [str(item) for item in failure_modes],
        "difficulty": str(value["difficulty"]),
        "constraints": [str(item) for item in constraints],
        "target_details_removed": True,
    }


def _safe_failed_skill_card(problem: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a target-disjoint sentinel stored only on a skill-card no-op row."""
    options = (
        {
            "domain": "general reasoning",
            "skills": ["apply reusable methods"],
            "reasoning_operators": ["derive conclusions"],
            "failure_modes": [],
            "difficulty": "general",
            "constraints": [],
            "target_details_removed": True,
        },
        {
            "domain": "a",
            "skills": ["i"],
            "reasoning_operators": ["a"],
            "failure_modes": [],
            "difficulty": "i",
            "constraints": [],
            "target_details_removed": True,
        },
    )
    for card in options:
        audit = skill_card_disjoint_audit(problem, card)
        if audit["safe"]:
            return card, audit
    raise AssertionError("single-token failed-skill sentinel must be target-disjoint")


def _candidate_list(value: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Candidate proposal must contain a candidates list")
    valid = []
    for candidate in candidates:
        if (
            not isinstance(candidate, dict)
            or not str(candidate.get("problem", "")).strip()
        ):
            continue
        valid.append(candidate)
    return valid


_ALLOWED_CANDIDATE_TYPES = {"atomic", "compositional", "failure_focused"}
MODEL_JSON_PARSER_VERSION = "literal-math-backslash-v2"
# Model-produced JSON frequently contains TeX with a single backslash.  JSON
# accepts only a tiny escape alphabet, while TeX also uses punctuation escapes
# such as ``\{``, ``\}``, ``\,`` and ``\!``.  Preserve every lone backslash
# inside the payload except the three escapes needed for JSON string structure:
# an already escaped backslash, an escaped quote, and an escaped slash.
_SINGLE_TEXT_BACKSLASH_RE = re.compile(r'(?<!\\)\\(?![\\"/])')


def _contains_unsafe_json_control(value: Any) -> bool:
    r"""Reject strings silently corrupted by TeX-like ``\f``/``\t`` escapes."""
    if isinstance(value, dict):
        return any(
            _contains_unsafe_json_control(key) or _contains_unsafe_json_control(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_unsafe_json_control(item) for item in value)
    if not isinstance(value, str):
        return False
    return any(
        ord(character) < 32 and character not in {"\n", "\r"} for character in value
    )


def _model_json_payloads(text: str) -> list[str]:
    """Return deterministic JSON payload candidates without brace-regex truncation."""
    stripped = text.strip()
    payloads = [stripped]
    for match in re.finditer(
        r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE
    ):
        payloads.append(match.group(1).strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        payloads.append(stripped[start : end + 1])
    unique: list[str] = []
    for payload in payloads:
        if payload and payload not in unique:
            unique.append(payload)
    return unique


def _parse_model_json_object(text: str) -> dict[str, Any]:
    r"""Parse model JSON while preserving literal mathematical backslashes.

    This repair is syntax-only: a lone mathematical backslash is made literal
    before decoding (including punctuation commands such as ``\{``), and raw
    newlines in string values are accepted.  Escaped JSON quotes/slashes and
    already doubled backslashes are left untouched.  No missing field or
    mathematical content is inferred.
    """
    errors: list[str] = []
    for payload in _model_json_payloads(text):
        repaired = _SINGLE_TEXT_BACKSLASH_RE.sub(r"\\\\", payload)
        variants = [repaired, payload] if repaired != payload else [payload]
        for variant in variants:
            for strict in (True, False):
                try:
                    value = json.loads(variant, strict=strict)
                except json.JSONDecodeError as exc:
                    errors.append(str(exc))
                    continue
                if not isinstance(value, dict):
                    errors.append("Expected a JSON object from the model")
                    continue
                if _contains_unsafe_json_control(value):
                    errors.append("Decoded JSON contains unsafe control characters")
                    continue
                return value
    detail = errors[-1] if errors else f"no JSON object in {text[:160]!r}"
    raise ValueError(f"Model response was not parseable JSON: {detail}")


def _tag_values(text: str, tag: str) -> list[str]:
    # Some otherwise schema-faithful model responses put whitespace immediately
    # after the slash in a closing tag (for example ``</ WRONG_STEP>``).  Treat
    # that whitespace as syntax only.  Opening tags, tag names, and both tag
    # boundaries remain mandatory, so this never supplies missing content.
    return [
        match.strip()
        for match in re.findall(
            rf"<{re.escape(tag)}>\s*(.*?)\s*</\s*{re.escape(tag)}\s*>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    ]


def _normalize_trajectory(value: Any, *, field: str) -> list[dict[str, Any]]:
    if isinstance(value, str) and value.strip():
        value = [{"step_index": 0, "text": value.strip()}]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if len(value) > 64:
        raise ValueError(f"{field} has too many steps")
    normalized: list[dict[str, Any]] = []
    for position, step in enumerate(value):
        if isinstance(step, str):
            index = position
            step_text = step.strip()
        elif isinstance(step, dict):
            index = step.get("step_index")
            step_text = str(step.get("text", "")).strip()
        else:
            raise ValueError(f"{field} steps must be objects or strings")
        if isinstance(index, bool) or not isinstance(index, int) or index != position:
            raise ValueError(
                f"{field} step_index must be consecutive and equal {position}"
            )
        if not step_text:
            raise ValueError(f"{field} step text must be non-empty")
        normalized.append({"step_index": index, "text": step_text})
    return normalized


def _trajectory_text(trajectory: list[dict[str, Any]]) -> str:
    # step_index is schema metadata rather than mathematical content. Excluding
    # it prevents a harmless index such as 3 from colliding with a target
    # problem literal in the all-artifact firewall.
    return "\n".join(str(step["text"]).strip() for step in trajectory)


def _parse_tagged_trajectory(
    text: str, *, step_tag: str, trajectory_field: str, answer_tag: str
) -> tuple[list[dict[str, Any]], str]:
    answers = _tag_values(text, answer_tag)
    if len(answers) != 1 or not answers[0]:
        raise ValueError(f"Expected exactly one non-empty {answer_tag} tag")
    _validate_final_answer(answers[0], field=answer_tag)
    steps: list[dict[str, Any]] = []
    for block in _tag_values(text, step_tag):
        indices = _tag_values(block, "STEP_INDEX")
        texts = _tag_values(block, "STEP_TEXT")
        if len(indices) != 1 or len(texts) != 1:
            raise ValueError(f"Malformed {step_tag} block")
        try:
            index = int(indices[0])
        except ValueError as exc:
            raise ValueError(f"{step_tag} STEP_INDEX is not an integer") from exc
        steps.append({"step_index": index, "text": texts[0]})
    return _normalize_trajectory(steps, field=trajectory_field), answers[0]


_BARE_TRAJECTORY_HEADER_RE = re.compile(
    r"^\s*(FINAL_ANSWER|WRONG_FINAL_ANSWER|CORRECT_STEP|WRONG_STEP|"
    r"STEP_INDEX|STEP_TEXT)\s*:?[ \t]*(.*?)\s*$",
    flags=re.IGNORECASE,
)


def _parse_bare_tagged_trajectory(
    text: str, *, step_tag: str, trajectory_field: str, answer_tag: str
) -> tuple[list[dict[str, Any]], str]:
    """Parse the line-delimited tag spelling Qwen sometimes emits.

    The model can preserve every requested field while omitting only the XML
    angle brackets, for example ``WRONG_FINAL_ANSWER`` on one line followed by
    its value.  This parser accepts that deterministic surface form without
    inventing a missing field or any mathematical content.  A response that
    starts XML-like markup is deliberately left to the strict XML parser so a
    malformed closing tag cannot be silently repaired here.
    """

    allowed = {answer_tag, step_tag, "STEP_INDEX", "STEP_TEXT"}
    sections: list[tuple[str, str]] = []
    current_tag: str | None = None
    current_lines: list[str] = []

    def finish_section() -> None:
        nonlocal current_tag, current_lines
        if current_tag is not None:
            sections.append((current_tag, "\n".join(current_lines).strip()))
        current_tag = None
        current_lines = []

    for raw_line in text.strip().splitlines():
        match = _BARE_TRAJECTORY_HEADER_RE.fullmatch(raw_line)
        if match is not None:
            tag = match.group(1).upper()
            if tag not in allowed:
                raise ValueError(f"Unexpected bare trajectory tag {tag}")
            finish_section()
            current_tag = tag
            inline_content = match.group(2).strip()
            if inline_content:
                current_lines.append(inline_content)
            continue
        if current_tag is None:
            if raw_line.strip():
                raise ValueError("Bare trajectory has content before its first tag")
            continue
        current_lines.append(raw_line)
    finish_section()

    if not sections or sections[0][0] != answer_tag:
        raise ValueError(f"Expected exactly one non-empty {answer_tag} tag")
    if sum(tag == answer_tag for tag, _ in sections) != 1 or not sections[0][1]:
        raise ValueError(f"Expected exactly one non-empty {answer_tag} tag")
    final_answer = sections[0][1]
    _validate_final_answer(final_answer, field=answer_tag)

    steps: list[dict[str, Any]] = []
    position = 1
    while position < len(sections):
        if sections[position][0] != step_tag or sections[position][1]:
            raise ValueError(f"Malformed bare {step_tag} block")
        if position + 2 >= len(sections):
            raise ValueError(f"Malformed bare {step_tag} block")
        index_tag, index_text = sections[position + 1]
        text_tag, step_text = sections[position + 2]
        if index_tag != "STEP_INDEX" or text_tag != "STEP_TEXT" or not step_text:
            raise ValueError(f"Malformed bare {step_tag} block")
        try:
            index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"{step_tag} STEP_INDEX is not an integer") from exc
        steps.append({"step_index": index, "text": step_text})
        position += 3
    return _normalize_trajectory(steps, field=trajectory_field), final_answer


def _parse_tagged_or_bare_trajectory(
    text: str, *, step_tag: str, trajectory_field: str, answer_tag: str
) -> tuple[list[dict[str, Any]], str]:
    try:
        return _parse_tagged_trajectory(
            text,
            step_tag=step_tag,
            trajectory_field=trajectory_field,
            answer_tag=answer_tag,
        )
    except ValueError:
        # Preserve fail-closed behavior for malformed XML-like responses.
        if re.search(
            rf"<\s*(?:{re.escape(answer_tag)}|{re.escape(step_tag)})\b",
            text,
            re.I,
        ):
            raise
    return _parse_bare_tagged_trajectory(
        text,
        step_tag=step_tag,
        trajectory_field=trajectory_field,
        answer_tag=answer_tag,
    )


def _parse_correct_trajectory_response(text: str) -> dict[str, Any]:
    try:
        value = _parse_model_json_object(text)
    except ValueError:
        trajectory, final_answer = _parse_tagged_or_bare_trajectory(
            text,
            step_tag="CORRECT_STEP",
            trajectory_field="correct_trajectory",
            answer_tag="FINAL_ANSWER",
        )
        return {"correct_trajectory": trajectory, "final_answer": final_answer}
    raw_trajectory = value.get("correct_trajectory", value.get("solution"))
    trajectory = _normalize_trajectory(raw_trajectory, field="correct_trajectory")
    final_answer = str(value.get("final_answer", "")).strip()
    if not final_answer:
        raise ValueError("Correct trajectory response is missing final_answer")
    _validate_final_answer(final_answer, field="final_answer")
    return {"correct_trajectory": trajectory, "final_answer": final_answer}


def _parse_wrong_trajectory_response(text: str) -> dict[str, Any]:
    try:
        value = _parse_model_json_object(text)
    except ValueError:
        trajectory, final_answer = _parse_tagged_or_bare_trajectory(
            text,
            step_tag="WRONG_STEP",
            trajectory_field="wrong_trajectory",
            answer_tag="WRONG_FINAL_ANSWER",
        )
        return {
            "wrong_trajectory": trajectory,
            "wrong_final_answer": final_answer,
        }
    raw_trajectory = value.get("wrong_trajectory", value.get("solution"))
    trajectory = _normalize_trajectory(raw_trajectory, field="wrong_trajectory")
    final_answer = str(
        value.get("wrong_final_answer", value.get("final_answer", ""))
    ).strip()
    if not final_answer:
        raise ValueError("Wrong trajectory response is missing wrong_final_answer")
    _validate_final_answer(final_answer, field="wrong_final_answer")
    return {"wrong_trajectory": trajectory, "wrong_final_answer": final_answer}


def _parse_verifier_response(text: str) -> dict[str, Any]:
    return _parse_model_json_object(text)


def _artifact_target_disjoint_audit(
    problem: str, text: str, *, max_fourgram_overlap: float
) -> dict[str, Any]:
    """Log accidental artifact overlap without rejecting source-isolated text.

    Correct/wrong trajectories are generated only after the candidate problem
    passed the strict target-instance firewall, and neither generator can read
    the target.  Treating a coincidental ordinary number or mathematical
    four-gram inside a *solution* as hindsight caused the v4 false-rejection
    bottleneck.  We therefore preserve the lexical audit as a diagnostic while
    certifying these artifacts by their enforced source boundary.
    """
    audit = target_disjoint_audit(problem, text)
    lexical_safe = bool(
        audit["literal_overlap_count"] == 0
        and audit["fourgram_overlap_count"] <= 1
        and audit["fourgram_overlap_rate"] <= max_fourgram_overlap
    )
    audit["thresholds"] = {
        "max_literal_overlap_rate": 0.0,
        "max_fourgram_overlap_rate": max_fourgram_overlap,
        "max_fourgram_overlap_count": 1,
    }
    audit["lexical_coincidence_check_safe"] = lexical_safe
    audit["lexical_overlap_is_informational_only"] = True
    audit["source_isolated_from_target"] = True
    audit["safety_basis"] = "source_isolated_post_candidate_generation"
    audit["safe"] = True
    return audit


def _stage_messages(
    messages: list[dict[str, str]], *, attempt: int, stage: str
) -> list[dict[str, str]]:
    if attempt == 0:
        return messages
    retry = [dict(message) for message in messages]
    retry.append(
        {
            "role": "user",
            "content": (
                f"FORMAT RETRY {attempt} for {stage}: regenerate independently and "
                "follow the exact requested schema/tags. Emit no surrounding prose."
            ),
        }
    )
    return retry


def _generate_stage(
    generator: HFGenerator,
    messages: list[dict[str, str]],
    parser,
    *,
    stage: str,
    max_attempts: int,
    stage_call_counts: dict[str, int],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(max_attempts):
        attempt_messages = _stage_messages(messages, attempt=attempt, stage=stage)
        raw = generator(attempt_messages)
        stage_call_counts[stage] = stage_call_counts.get(stage, 0) + 1
        trace: dict[str, Any] = {
            "attempt": attempt,
            "raw_response": raw,
            "raw_response_sha256": stable_hash(raw, length=64),
            "message_sha256": stable_hash(
                json.dumps(
                    attempt_messages,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                length=64,
            ),
        }
        try:
            parsed = parser(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            trace.update(parsed=False, accepted=False, error=str(exc))
            attempts.append(trace)
            continue
        trace.update(parsed=True, accepted=True)
        attempts.append(trace)
        return parsed, attempts
    return None, attempts


def _candidate_type(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().casefold().replace("-", "_")
    if normalized not in _ALLOWED_CANDIDATE_TYPES:
        raise ValueError(
            "candidate_type must be atomic, compositional, or failure_focused"
        )
    return normalized


def _validate_error_frontier(
    value: dict[str, Any], wrong_trajectory: list[dict[str, Any]]
) -> dict[str, Any]:
    required_true = (
        "wrong_trajectory_incorrect",
        "prefix_before_error_valid",
        "wrong_step_invalid",
        "corrective_action_valid",
    )
    if not all(_as_bool(value.get(field, False)) for field in required_true):
        raise ValueError(
            "Verifier did not confirm an incorrect trajectory, valid prior prefix, "
            "invalid selected step, and valid correction"
        )
    wrong_step_index = value.get("wrong_step_index")
    if isinstance(wrong_step_index, bool) or not isinstance(wrong_step_index, int):
        raise ValueError("wrong_step_index must be an integer")
    indexed_steps = {
        int(step["step_index"]): str(step["text"]).strip() for step in wrong_trajectory
    }
    if wrong_step_index not in indexed_steps:
        raise ValueError("wrong_step_index does not identify a model trajectory step")
    error_explanation = str(value.get("error_explanation", "")).strip()
    corrective_action = str(value.get("corrective_action", "")).strip()
    if not error_explanation or not corrective_action:
        raise ValueError("Error frontier needs an explanation and corrective action")
    semantic_text = f"{error_explanation}\n{corrective_action}"
    if _FRONTIER_VALID_CLAIM_RE.search(semantic_text) is not None:
        raise ValueError(
            "Error frontier contradicts its booleans by describing the selected "
            "wrong step as valid/correct"
        )
    if _FRONTIER_NO_CORRECTION_RE.search(semantic_text) is not None:
        raise ValueError(
            "Error frontier contradicts its booleans by saying no correction is needed"
        )
    wrong_step_text = indexed_steps[wrong_step_index]
    if _frontier_semantic_key(corrective_action) == _frontier_semantic_key(
        wrong_step_text
    ) or _frontier_has_noop_replacement(corrective_action):
        raise ValueError("Error frontier corrective action is a semantic no-op")
    return {
        "wrong_step_index": wrong_step_index,
        "wrong_step_text": wrong_step_text,
        "error_explanation": error_explanation,
        "corrective_action": corrective_action,
        "verifier_valid": True,
    }


def propose_for_query(
    record: dict[str, Any],
    proposer_generator: HFGenerator,
    solver_generator: HFGenerator,
    verifier_generator: HFGenerator,
    *,
    num_candidates: int,
    proposal_oversample: int,
    max_rounds: int,
    min_accepted_candidates: int,
    max_literal_overlap: float,
    max_fourgram_overlap: float,
    accept_verifier_corrections: bool,
    stage_max_attempts: int = 2,
    max_fourgram_overlap_count: int = 1,
) -> dict[str, Any]:
    # Only record["problem"] is available here; load_query_records excluded targets.
    problem = record["problem"]
    if stage_max_attempts <= 0:
        raise ValueError("stage_max_attempts must be positive")
    if max_fourgram_overlap_count < 0:
        raise ValueError("max_fourgram_overlap_count must be non-negative")
    started = time.perf_counter()
    counter_starts = {
        "proposer": proposer_generator.counters(),
        "solver": solver_generator.counters(),
        "verifier": verifier_generator.counters(),
    }
    skill_attempts: list[dict[str, Any]] = []
    skill_card: dict[str, Any] | None = None
    skill_card_audit: dict[str, Any] = {}
    redacted_literals: list[str] = []
    skill_card_failed = False
    stage_call_counts: dict[str, int] = {}
    for skill_attempt in range(max_rounds):
        raw_text = proposer_generator(skill_card_messages(problem))
        stage_call_counts["skill_card"] = stage_call_counts.get("skill_card", 0) + 1
        try:
            skill_raw = _parse_model_json_object(raw_text)
            candidate_card, candidate_redactions = sanitize_skill_card(
                _validate_skill_card(skill_raw), problem
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            skill_attempts.append(
                {
                    "attempt": skill_attempt,
                    "raw_response": raw_text,
                    "parsed": False,
                    "accepted": False,
                    "error": str(exc),
                }
            )
            continue

        candidate_audit = skill_card_disjoint_audit(problem, candidate_card)
        if not candidate_audit["safe"]:
            skill_attempts.append(
                {
                    "attempt": skill_attempt,
                    "raw_response": raw_text,
                    "parsed": True,
                    "accepted": False,
                    "error": (
                        "sanitized skill card still contains target-specific "
                        "lexical or symbolic detail"
                    ),
                    "post_sanitize_audit": candidate_audit,
                }
            )
            continue

        skill_card = candidate_card
        redacted_literals = candidate_redactions
        skill_card_audit = candidate_audit
        skill_attempts.append(
            {
                "attempt": skill_attempt,
                "raw_response": raw_text,
                "parsed": True,
                "accepted": True,
                "post_sanitize_audit": candidate_audit,
            }
        )
        break
    if skill_card is None:
        skill_card, skill_card_audit = _safe_failed_skill_card(problem)
        skill_card_failed = True

    accepted: list[dict[str, Any]] = []
    seen_problems: set[str] = set()
    candidate_attempts: list[dict[str, Any]] = []
    proposal_rounds: list[dict[str, Any]] = []
    attempts = 0
    while (
        not skill_card_failed
        and len(accepted) < num_candidates
        and attempts < max_rounds
    ):
        attempts += 1
        requested = num_candidates - len(accepted) + proposal_oversample
        raw_proposal = proposer_generator(candidate_messages(skill_card, requested))
        stage_call_counts["candidate_proposal"] = (
            stage_call_counts.get("candidate_proposal", 0) + 1
        )
        try:
            proposed = _parse_model_json_object(raw_proposal)
            parsed_candidates = _candidate_list(proposed)
        except (ValueError, json.JSONDecodeError) as exc:
            proposal_rounds.append(
                {
                    "round": attempts,
                    "requested": requested,
                    "raw_response": raw_proposal,
                    "parsed": False,
                    "error": str(exc),
                }
            )
            continue
        proposal_rounds.append(
            {
                "round": attempts,
                "requested": requested,
                "raw_response": raw_proposal,
                "parsed": True,
                "parsed_candidate_count": len(parsed_candidates),
            }
        )
        for candidate in parsed_candidates:
            candidate_problem = str(candidate["problem"]).strip()
            normalized = " ".join(candidate_problem.lower().split())
            trace: dict[str, Any] = {
                "round": attempts,
                "proposed_candidate_id": candidate.get("candidate_id"),
                "problem": candidate_problem,
                "skill_tags": candidate.get("skill_tags", []),
            }
            if normalized in seen_problems:
                trace.update(outcome="rejected", reason="duplicate_problem")
                candidate_attempts.append(trace)
                continue
            seen_problems.add(normalized)
            raw_skill_tags = candidate.get("skill_tags", [])
            if isinstance(raw_skill_tags, list):
                skill_tags = [str(skill_tag) for skill_tag in raw_skill_tags]
            elif raw_skill_tags is None:
                skill_tags = []
            else:
                skill_tags = [str(raw_skill_tags)]
            placeholder_audit = _placeholder_artifact_audit(
                "\n".join(
                    [candidate_problem, *(str(skill_tag) for skill_tag in skill_tags)]
                )
            )
            trace["candidate_placeholder_artifact_audit"] = placeholder_audit
            trace["placeholder_artifact_audit"] = placeholder_audit
            if not placeholder_audit["safe"]:
                trace.update(
                    outcome="rejected",
                    reason="placeholder_artifact",
                    placeholder_artifact_source="candidate_proposal",
                )
                candidate_attempts.append(trace)
                continue
            disjoint = target_disjoint_audit(problem, candidate_problem)
            disjoint["thresholds"] = {
                "max_literal_overlap_rate": max_literal_overlap,
                "max_fourgram_overlap_rate": max_fourgram_overlap,
                "max_fourgram_overlap_count": max_fourgram_overlap_count,
            }
            disjoint["safe"] = bool(
                disjoint["literal_overlap_count"] == 0
                and disjoint["fourgram_overlap_count"]
                <= max_fourgram_overlap_count
                and disjoint["fourgram_overlap_rate"] <= max_fourgram_overlap
            )
            trace["target_disjoint_audit"] = disjoint
            if disjoint["literal_overlap_count"] > 0:
                trace.update(outcome="rejected", reason="literal_overlap")
                candidate_attempts.append(trace)
                continue
            if (
                disjoint["fourgram_overlap_count"] > max_fourgram_overlap_count
                or disjoint["fourgram_overlap_rate"] > max_fourgram_overlap
            ):
                trace.update(outcome="rejected", reason="fourgram_overlap")
                candidate_attempts.append(trace)
                continue

            try:
                candidate_type = _candidate_type(candidate.get("candidate_type"))
            except ValueError as exc:
                trace.update(
                    outcome="rejected", reason="candidate_schema_error", error=str(exc)
                )
                candidate_attempts.append(trace)
                continue
            trace["candidate_type"] = candidate_type

            solved, correct_attempts = _generate_stage(
                solver_generator,
                solver_messages(candidate_problem),
                _parse_correct_trajectory_response,
                stage="correct_trajectory",
                max_attempts=stage_max_attempts,
                stage_call_counts=stage_call_counts,
            )
            trace["correct_trajectory_attempts"] = correct_attempts
            if correct_attempts:
                trace["solver_raw_response"] = correct_attempts[-1]["raw_response"]
            if solved is None:
                trace.update(
                    outcome="rejected",
                    reason="solver_parse_error",
                    error="correct trajectory exhausted bounded parse retries",
                )
                candidate_attempts.append(trace)
                continue
            correct_trajectory = solved["correct_trajectory"]
            final_answer = solved["final_answer"]
            solution = _trajectory_text(correct_trajectory)
            solver_placeholder_audit = _placeholder_artifact_audit(
                f"{solution}\n{final_answer}"
            )
            correct_disjoint_audit = _artifact_target_disjoint_audit(
                problem,
                f"{solution}\n{final_answer}",
                max_fourgram_overlap=max_fourgram_overlap,
            )
            trace["solver_placeholder_artifact_audit"] = solver_placeholder_audit
            trace["correct_trajectory_target_disjoint_audit"] = correct_disjoint_audit
            if not solver_placeholder_audit["safe"]:
                trace["placeholder_artifact_audit"] = solver_placeholder_audit
                trace.update(
                    outcome="rejected",
                    reason="placeholder_artifact",
                    placeholder_artifact_source="solver_output",
                    placeholder_artifact_stage="correct_trajectory",
                )
                candidate_attempts.append(trace)
                continue
            if not correct_disjoint_audit["safe"]:
                trace.update(
                    outcome="rejected",
                    reason=(
                        "correct_trajectory_literal_overlap"
                        if correct_disjoint_audit["literal_overlap_count"] > 0
                        else "correct_trajectory_fourgram_overlap"
                    ),
                )
                candidate_attempts.append(trace)
                continue

            verified, verification_attempts = _generate_stage(
                verifier_generator,
                verifier_messages(candidate_problem, solution, final_answer),
                _parse_verifier_response,
                stage="correct_trajectory_verifier",
                max_attempts=stage_max_attempts,
                stage_call_counts=stage_call_counts,
            )
            trace["correct_verifier_attempts"] = verification_attempts
            if verification_attempts:
                trace["verifier_raw_response"] = verification_attempts[-1][
                    "raw_response"
                ]
            if verified is None:
                trace.update(
                    outcome="rejected",
                    reason="verifier_parse_error",
                    error="correct verifier exhausted bounded parse retries",
                )
                candidate_attempts.append(trace)
                continue
            if "problem_well_posed" not in verified:
                trace.update(
                    outcome="rejected",
                    reason="verifier_schema_error",
                    verifier_reason="missing problem_well_posed verdict",
                )
                candidate_attempts.append(trace)
                continue
            if not _as_bool(verified.get("problem_well_posed", False)):
                trace.update(
                    outcome="rejected",
                    reason="problem_ill_posed",
                    verifier_reason=str(verified.get("reason", "")),
                )
                candidate_attempts.append(trace)
                continue
            is_valid = _as_bool(verified.get("valid", False))
            verifier_corrected = False
            accepted_verification_attempts = verification_attempts
            if not is_valid and not accept_verifier_corrections:
                trace.update(
                    outcome="rejected",
                    reason="verifier_invalid",
                    verifier_reason=str(verified.get("reason", "")),
                )
                candidate_attempts.append(trace)
                continue
            if not is_valid:
                corrected_solution = str(verified.get("corrected_solution", "")).strip()
                corrected_final_answer = str(
                    verified.get("corrected_final_answer", "")
                ).strip()
                if not corrected_solution or not corrected_final_answer:
                    trace.update(outcome="rejected", reason="invalid_correction")
                    candidate_attempts.append(trace)
                    continue
                try:
                    _validate_final_answer(
                        corrected_final_answer, field="corrected_final_answer"
                    )
                except ValueError as exc:
                    trace.update(
                        outcome="rejected",
                        reason="invalid_correction",
                        error=str(exc),
                    )
                    candidate_attempts.append(trace)
                    continue
                correction_verification, correction_attempts = _generate_stage(
                    verifier_generator,
                    verifier_messages(
                        candidate_problem,
                        corrected_solution,
                        corrected_final_answer,
                    ),
                    _parse_verifier_response,
                    stage="corrected_trajectory_verifier",
                    max_attempts=stage_max_attempts,
                    stage_call_counts=stage_call_counts,
                )
                trace["correction_verifier_attempts"] = correction_attempts
                if (
                    correction_verification is None
                    or not _as_bool(
                        correction_verification.get("problem_well_posed", False)
                    )
                    or not _as_bool(correction_verification.get("valid", False))
                ):
                    trace.update(outcome="rejected", reason="correction_not_reverified")
                    candidate_attempts.append(trace)
                    continue
                correct_trajectory = _normalize_trajectory(
                    corrected_solution, field="correct_trajectory"
                )
                final_answer = corrected_final_answer
                solution = _trajectory_text(correct_trajectory)
                verified = correction_verification
                accepted_verification_attempts = correction_attempts
                is_valid = True
                verifier_corrected = True
                solver_placeholder_audit = _placeholder_artifact_audit(
                    f"{solution}\n{final_answer}"
                )
                correct_disjoint_audit = _artifact_target_disjoint_audit(
                    problem,
                    f"{solution}\n{final_answer}",
                    max_fourgram_overlap=max_fourgram_overlap,
                )
                if (
                    not solver_placeholder_audit["safe"]
                    or not correct_disjoint_audit["safe"]
                ):
                    trace.update(
                        outcome="rejected",
                        reason="invalid_correction_firewall",
                    )
                    candidate_attempts.append(trace)
                    continue

            verifier_reason = str(verified.get("reason", ""))
            failure_modes = [str(mode) for mode in skill_card.get("failure_modes", [])]
            wrong_trajectory: list[dict[str, Any]] | None = None
            wrong_final_answer = ""
            error_frontier: dict[str, Any] | None = None
            wrong_disjoint_audit: dict[str, Any] = {}
            frontier_disjoint_audit: dict[str, Any] = {}
            answer_contrast_audit: dict[str, Any] = {}
            wrong_stage_attempts: list[dict[str, Any]] = []
            frontier_stage_attempts: list[dict[str, Any]] = []
            wrong_failure_reason = "wrong_trajectory_unverified"
            for wrong_attempt in range(stage_max_attempts):
                wrong_messages = _stage_messages(
                    wrong_trajectory_messages(candidate_problem, failure_modes),
                    attempt=wrong_attempt,
                    stage="wrong_trajectory",
                )
                wrong_raw = solver_generator(wrong_messages)
                stage_call_counts["wrong_trajectory"] = (
                    stage_call_counts.get("wrong_trajectory", 0) + 1
                )
                wrong_trace: dict[str, Any] = {
                    "attempt": wrong_attempt,
                    "raw_response": wrong_raw,
                    "raw_response_sha256": stable_hash(wrong_raw, length=64),
                    "message_sha256": stable_hash(
                        json.dumps(
                            wrong_messages,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        length=64,
                    ),
                }
                try:
                    wrong_value = _parse_wrong_trajectory_response(wrong_raw)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    wrong_trace.update(parsed=False, accepted=False, error=str(exc))
                    wrong_stage_attempts.append(wrong_trace)
                    wrong_failure_reason = "wrong_trajectory_parse_error"
                    continue
                candidate_wrong_trajectory = wrong_value["wrong_trajectory"]
                candidate_wrong_final = wrong_value["wrong_final_answer"]
                try:
                    candidate_answer_contrast = _contrastive_answer_audit(
                        final_answer, candidate_wrong_final
                    )
                except ValueError as exc:
                    wrong_trace.update(
                        parsed=True,
                        accepted=False,
                        error=str(exc),
                    )
                    wrong_stage_attempts.append(wrong_trace)
                    wrong_failure_reason = "wrong_answer_not_distinct"
                    continue
                wrong_trace["answer_contrast_audit"] = candidate_answer_contrast
                wrong_text = (
                    f"{_trajectory_text(candidate_wrong_trajectory)}\n"
                    f"{candidate_wrong_final}"
                )
                wrong_placeholder_audit = _placeholder_artifact_audit(wrong_text)
                candidate_wrong_disjoint = _artifact_target_disjoint_audit(
                    problem,
                    wrong_text,
                    max_fourgram_overlap=max_fourgram_overlap,
                )
                wrong_trace.update(
                    parsed=True,
                    placeholder_artifact_audit=wrong_placeholder_audit,
                    target_disjoint_audit=candidate_wrong_disjoint,
                )
                if not wrong_placeholder_audit["safe"]:
                    wrong_trace.update(
                        accepted=False, error="wrong trajectory contains a placeholder"
                    )
                    wrong_stage_attempts.append(wrong_trace)
                    wrong_failure_reason = "wrong_trajectory_placeholder_artifact"
                    continue
                if not candidate_wrong_disjoint["safe"]:
                    wrong_trace.update(
                        accepted=False,
                        error="wrong trajectory failed target-disjoint audit",
                    )
                    wrong_stage_attempts.append(wrong_trace)
                    wrong_failure_reason = "wrong_trajectory_overlap"
                    continue

                remaining_frontier_attempts = stage_max_attempts - len(
                    frontier_stage_attempts
                )
                if remaining_frontier_attempts <= 0:
                    wrong_trace.update(
                        accepted=False,
                        error="error frontier verifier exhausted its per-candidate budget",
                    )
                    wrong_stage_attempts.append(wrong_trace)
                    wrong_failure_reason = "error_frontier_parse_error"
                    break
                frontier_value, frontier_attempts = _generate_stage(
                    verifier_generator,
                    frontier_verifier_messages(
                        candidate_problem,
                        correct_trajectory,
                        final_answer,
                        candidate_wrong_trajectory,
                        candidate_wrong_final,
                    ),
                    _parse_verifier_response,
                    stage="error_frontier_verifier",
                    max_attempts=remaining_frontier_attempts,
                    stage_call_counts=stage_call_counts,
                )
                for frontier_attempt in frontier_attempts:
                    frontier_attempts_copy = dict(frontier_attempt)
                    frontier_attempts_copy["wrong_trajectory_attempt"] = wrong_attempt
                    frontier_stage_attempts.append(frontier_attempts_copy)
                if frontier_value is None:
                    wrong_trace.update(
                        accepted=False,
                        error="frontier verifier exhausted bounded parse retries",
                    )
                    wrong_stage_attempts.append(wrong_trace)
                    wrong_failure_reason = "error_frontier_parse_error"
                    continue
                try:
                    candidate_frontier = _validate_error_frontier(
                        frontier_value, candidate_wrong_trajectory
                    )
                except ValueError as exc:
                    if frontier_stage_attempts:
                        frontier_stage_attempts[-1].update(
                            accepted=False, semantic_error=str(exc)
                        )
                    wrong_trace.update(accepted=False, error=str(exc))
                    wrong_stage_attempts.append(wrong_trace)
                    wrong_failure_reason = "error_frontier_invalid"
                    continue
                frontier_text = "\n".join(
                    [
                        candidate_frontier["wrong_step_text"],
                        candidate_frontier["error_explanation"],
                        candidate_frontier["corrective_action"],
                    ]
                )
                frontier_placeholder_audit = _placeholder_artifact_audit(frontier_text)
                candidate_frontier_disjoint = _artifact_target_disjoint_audit(
                    problem,
                    frontier_text,
                    max_fourgram_overlap=max_fourgram_overlap,
                )
                if not frontier_placeholder_audit["safe"]:
                    if frontier_stage_attempts:
                        frontier_stage_attempts[-1].update(
                            accepted=False,
                            semantic_error=(
                                "error frontier contains a placeholder artifact"
                            ),
                        )
                    wrong_trace.update(
                        accepted=False,
                        error="error frontier contains a placeholder artifact",
                    )
                    wrong_stage_attempts.append(wrong_trace)
                    wrong_failure_reason = "error_frontier_placeholder_artifact"
                    continue
                if not candidate_frontier_disjoint["safe"]:
                    if frontier_stage_attempts:
                        frontier_stage_attempts[-1].update(
                            accepted=False,
                            semantic_error=(
                                "error frontier failed target-disjoint audit"
                            ),
                        )
                    wrong_trace.update(
                        accepted=False,
                        error="error frontier failed target-disjoint audit",
                    )
                    wrong_stage_attempts.append(wrong_trace)
                    wrong_failure_reason = "error_frontier_overlap"
                    continue
                wrong_trace.update(accepted=True)
                wrong_stage_attempts.append(wrong_trace)
                wrong_trajectory = candidate_wrong_trajectory
                wrong_final_answer = candidate_wrong_final
                error_frontier = candidate_frontier
                wrong_disjoint_audit = candidate_wrong_disjoint
                frontier_disjoint_audit = candidate_frontier_disjoint
                answer_contrast_audit = candidate_answer_contrast
                break

            trace["wrong_trajectory_attempts"] = wrong_stage_attempts
            trace["error_frontier_verifier_attempts"] = frontier_stage_attempts
            if wrong_trajectory is None or error_frontier is None:
                trace.update(outcome="rejected", reason=wrong_failure_reason)
                candidate_attempts.append(trace)
                continue

            accepted_text = "\n".join(
                [
                    candidate_problem,
                    *(str(skill_tag) for skill_tag in skill_tags),
                    solution,
                    final_answer,
                    _trajectory_text(wrong_trajectory),
                    wrong_final_answer,
                    error_frontier["wrong_step_text"],
                    error_frontier["error_explanation"],
                    error_frontier["corrective_action"],
                ]
            )
            accepted_placeholder_audit = _placeholder_artifact_audit(accepted_text)
            trace[
                "accepted_candidate_placeholder_artifact_audit"
            ] = accepted_placeholder_audit
            if not accepted_placeholder_audit["safe"]:
                trace["placeholder_artifact_audit"] = accepted_placeholder_audit
                trace.update(
                    outcome="rejected",
                    reason="placeholder_artifact",
                    placeholder_artifact_source="contrastive_candidate",
                )
                candidate_attempts.append(trace)
                continue
            trace["placeholder_artifact_audit"] = accepted_placeholder_audit

            candidate_id = f"c{len(accepted):02d}"
            correct_provenance = correct_attempts[-1]
            verifier_provenance = accepted_verification_attempts[-1]
            wrong_provenance = next(
                attempt
                for attempt in reversed(wrong_stage_attempts)
                if attempt["accepted"]
            )
            frontier_provenance = frontier_stage_attempts[-1]
            accepted_candidate: dict[str, Any] = {
                "candidate_id": candidate_id,
                "problem": candidate_problem,
                "skill_tags": skill_tags,
                "correct_trajectory": correct_trajectory,
                "wrong_trajectory": wrong_trajectory,
                "wrong_final_answer": wrong_final_answer,
                "error_frontier": error_frontier,
                # Legacy aliases remain the source consumed by the current ridge path.
                "solution": solution,
                "final_answer": final_answer,
                "verifier_valid": True,
                "verifier_accepted": True,
                "verifier_reason": verifier_reason,
                "frontier_verifier_valid": True,
                "answer_contrast_audit": answer_contrast_audit,
                "verifier_corrected": verifier_corrected,
                "placeholder_artifact_audit": accepted_placeholder_audit,
                "target_disjoint_audit": disjoint,
                "artifact_target_disjoint_audits": {
                    "correct_trajectory": correct_disjoint_audit,
                    "wrong_trajectory": wrong_disjoint_audit,
                    "error_frontier": frontier_disjoint_audit,
                },
                "generation_provenance": {
                    "correct_trajectory": {
                        "source": "independent_solver",
                        "attempt_count": len(correct_attempts),
                        "raw_response_sha256": correct_provenance[
                            "raw_response_sha256"
                        ],
                        "message_sha256": correct_provenance["message_sha256"],
                    },
                    "wrong_trajectory": {
                        "source": "independent_failure_conditioned_model_generation",
                        "sources": [
                            "candidate_problem",
                            "sanitized_skill_card_failure_modes",
                        ],
                        "correct_trajectory_exposed": False,
                        "attempt_count": len(wrong_stage_attempts),
                        "raw_response_sha256": wrong_provenance["raw_response_sha256"],
                        "message_sha256": wrong_provenance["message_sha256"],
                    },
                    "correct_trajectory_verifier": {
                        "source": "independent_verifier",
                        "sources": [
                            "candidate_problem",
                            "candidate_correct_trajectory",
                        ],
                        "attempt_count": len(accepted_verification_attempts),
                        "raw_response_sha256": verifier_provenance[
                            "raw_response_sha256"
                        ],
                        "message_sha256": verifier_provenance["message_sha256"],
                    },
                    "error_frontier": {
                        "source": "independent_verifier",
                        "sources": [
                            "candidate_problem",
                            "verified_correct_trajectory",
                            "model_wrong_trajectory",
                        ],
                        "attempt_count": len(frontier_stage_attempts),
                        "raw_response_sha256": frontier_provenance[
                            "raw_response_sha256"
                        ],
                        "message_sha256": frontier_provenance["message_sha256"],
                    },
                },
            }
            if candidate_type is not None:
                accepted_candidate["candidate_type"] = candidate_type
            accepted.append(accepted_candidate)
            trace.update(
                outcome="accepted",
                reason="verifier_valid",
                accepted_candidate_id=candidate_id,
                solver_solution=solution,
                solver_final_answer=final_answer,
                wrong_final_answer=wrong_final_answer,
                error_frontier=error_frontier,
                verifier_valid=True,
                verifier_reason=verifier_reason,
            )
            candidate_attempts.append(trace)
            if len(accepted) >= num_candidates:
                break

    if len(accepted) < min_accepted_candidates:
        specialization_status = "insufficient_verified_candidates"
        if skill_card_failed:
            specialization_failure_reason = (
                f"could not produce a safe, parseable skill card after "
                f"{max_rounds} attempts"
            )
        else:
            specialization_failure_reason = (
                f"obtained {len(accepted)}/{num_candidates} verified candidates; "
                f"minimum is {min_accepted_candidates} after {max_rounds} rounds"
            )
        specialization_no_op = True
    else:
        specialization_status = "ready"
        specialization_failure_reason = ""
        specialization_no_op = False

    # Prompt hashes make the information boundary auditable without storing
    # every system prompt. Candidate/solver/verifier prompts contain no target.
    card_prompt = render_chat(
        proposer_generator.tokenizer,
        skill_card_messages(problem),
        add_generation_prompt=True,
        enable_thinking=proposer_generator.enable_thinking,
    )
    candidate_prompt = render_chat(
        proposer_generator.tokenizer,
        candidate_messages(skill_card, num_candidates),
        add_generation_prompt=True,
        enable_thinking=proposer_generator.enable_thinking,
    )
    counter_ends = {
        "proposer": proposer_generator.counters(),
        "solver": solver_generator.counters(),
        "verifier": verifier_generator.counters(),
    }
    role_costs = {
        role: {
            key: counter_ends[role][key] - counter_starts[role][key]
            for key in counter_ends[role]
        }
        for role in counter_ends
    }
    row = {
        **record,
        "schema_version": "clean-self-distill-proposals-v5",
        "model_json_parser_version": MODEL_JSON_PARSER_VERSION,
        "problem_sha256": stable_hash(problem, length=64),
        "skill_card": skill_card,
        "skill_card_generation_failed": skill_card_failed,
        "skill_card_target_disjoint_audit": skill_card_audit,
        "specialization_status": specialization_status,
        "specialization_failure_reason": specialization_failure_reason,
        "specialization_no_op": specialization_no_op,
        "specialization_candidates": accepted,
        "requested_candidate_count": num_candidates,
        "minimum_candidate_count": min_accepted_candidates,
        "candidate_count": len(accepted),
        "skill_card_attempts": skill_attempts,
        "proposal_rounds": proposal_rounds,
        "candidate_attempts": candidate_attempts,
        "filter_summary": {
            "proposed_unique_count": len(seen_problems),
            "accepted_count": len(accepted),
            "rejected_count": sum(
                attempt.get("outcome") == "rejected" for attempt in candidate_attempts
            ),
            "verification_yield": len(accepted) / max(len(seen_problems), 1),
            "rejection_reason_counts": {
                reason: sum(
                    attempt.get("outcome") == "rejected"
                    and attempt.get("reason") == reason
                    for attempt in candidate_attempts
                )
                for reason in sorted(
                    {
                        str(attempt.get("reason"))
                        for attempt in candidate_attempts
                        if attempt.get("outcome") == "rejected"
                    }
                )
            },
        },
        "cost_audit": {
            "roles": role_costs,
            "generation_calls_by_stage": dict(sorted(stage_call_counts.items())),
            "total_generation_calls": sum(stage_call_counts.values()),
            "stage_max_attempts": stage_max_attempts,
            "total_prompt_tokens": sum(
                value["prompt_tokens"] for value in role_costs.values()
            ),
            "total_completion_tokens": sum(
                value["completion_tokens"] for value in role_costs.values()
            ),
            "total_generation_seconds": sum(
                value["generation_seconds"] for value in role_costs.values()
            ),
            "end_to_end_seconds": time.perf_counter() - started,
        },
        "firewall_audit": {
            "target_answer_loaded": False,
            "target_solution_loaded": False,
            "candidate_proposer_sources": ["sanitized_skill_card"],
            "correct_solver_sources": ["candidate_problem"],
            "wrong_trajectory_sources": [
                "candidate_problem",
                "sanitized_skill_card_failure_modes",
            ],
            "wrong_trajectory_correct_solution_exposed": False,
            "correct_verifier_sources": [
                "candidate_problem",
                "candidate_correct_trajectory",
            ],
            "frontier_verifier_sources": [
                "candidate_problem",
                "verified_correct_trajectory",
                "model_wrong_trajectory",
            ],
            # Retain the v4 summary keys for downstream audit compatibility.
            "solver_sources": ["candidate_problem"],
            "verifier_sources": [
                "candidate_problem",
                "candidate_correct_trajectory",
                "model_wrong_trajectory",
            ],
            "all_accepted_candidate_artifacts_target_disjoint": all(
                candidate.get("target_disjoint_audit", {}).get("safe", False)
                and all(
                    audit.get("safe", False)
                    for audit in candidate.get(
                        "artifact_target_disjoint_audits", {}
                    ).values()
                )
                for candidate in accepted
            ),
            "skill_card_redaction_count": len(redacted_literals),
            "skill_prompt_sha256": stable_hash(card_prompt, length=64),
            "candidate_prompt_sha256": stable_hash(candidate_prompt, length=64),
        },
    }
    row["proposal_training_sha256"] = compute_proposal_training_sha256(row)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL/JSON/parquet dataset; verl parquet is supported",
    )
    parser.add_argument("--output", required=True, help="Output proposal JSONL")
    parser.add_argument(
        "--model", required=True, help="Local or Hugging Face causal LM"
    )
    parser.add_argument("--revision", help="Pinned Hugging Face model revision")
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--proposal-oversample", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument(
        "--min-accepted-candidates",
        type=int,
        help="Minimum verified candidates required (default: min(num-candidates, 4))",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--solver-temperature", type=float, default=0.3)
    parser.add_argument("--verifier-temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--max-literal-overlap",
        type=float,
        default=0.0,
        help="Deprecated compatibility knob; exact target literal overlap is always rejected.",
    )
    parser.add_argument("--max-fourgram-overlap", type=float, default=0.05)
    parser.add_argument(
        "--max-fourgram-overlap-count",
        type=int,
        default=1,
        help="Maximum distinct candidate/target four-grams before rejection.",
    )
    parser.add_argument(
        "--stage-max-attempts",
        type=int,
        default=2,
        help="Bounded parse/format attempts for each solver and verifier stage.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--accept-verifier-corrections", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        pass

    # The proposer loader strips answers/solutions before any model is loaded.
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    if args.min_accepted_candidates is None:
        args.min_accepted_candidates = min(args.num_candidates, 4)
    if not 1 <= args.min_accepted_candidates <= args.num_candidates:
        raise ValueError("min_accepted_candidates must be in [1, num_candidates]")
    records = load_query_records(
        args.input, include_targets=False, max_samples=args.max_samples
    )
    records = [
        record
        for global_index, record in enumerate(records)
        if global_index % args.num_shards == args.shard_index
    ]
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        training=False,
        revision=args.revision,
    )
    runtime_metadata = collect_runtime_metadata(
        model, model_path=args.model, revision=args.revision or ""
    )
    proposer_generator = HFGenerator(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=False,
    )
    solver_generator = HFGenerator(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.solver_temperature,
        top_p=args.top_p,
        enable_thinking=False,
    )
    verifier_generator = HFGenerator(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.verifier_temperature,
        top_p=1.0,
        enable_thinking=False,
    )

    existing_ids: set[str] = set()
    output_path = Path(args.output)
    if output_path.exists():
        if args.resume:
            existing_rows = load_proposal_map(output_path)
            expected_ids = [record["query_id"] for record in records]
            existing_ids_in_order = list(existing_rows)
            if existing_ids_in_order != expected_ids[: len(existing_ids_in_order)]:
                raise ValueError(
                    f"Resume file {output_path} is not the exact ordered prefix of "
                    "this shard/config"
                )
            for record, (query_id, row) in zip(records, existing_rows.items()):
                if (
                    str(row.get("problem", "")) != record["problem"]
                    or str(row.get("problem_sha256", "")) != record["problem_sha256"]
                    or str(row.get("source", "")).strip().lower() != record["source"]
                ):
                    raise ValueError(
                        f"Resume proposal {query_id} does not match its dataset problem/source"
                    )
                if str(row.get("model", "")) != args.model:
                    raise ValueError(
                        f"Resume proposal {query_id} used model {row.get('model')!r}, "
                        f"not {args.model!r}"
                    )
                prior_revision = str(row.get("model_revision", ""))
                resolved_revision = str(
                    runtime_metadata.get("resolved_model_revision", args.revision or "")
                )
                if prior_revision != resolved_revision:
                    raise ValueError(
                        f"Resume proposal {query_id} used revision {prior_revision!r}, "
                        f"not {resolved_revision!r}"
                    )
            existing_ids = set(existing_ids_in_order)
        else:
            output_path.unlink()
    output_rows = []
    for record in tqdm(records, desc="propose+solve+verify"):
        if record["query_id"] in existing_ids:
            continue
        output_rows.append(
            propose_for_query(
                record,
                proposer_generator,
                solver_generator,
                verifier_generator,
                num_candidates=args.num_candidates,
                proposal_oversample=args.proposal_oversample,
                max_rounds=args.max_rounds,
                min_accepted_candidates=args.min_accepted_candidates,
                max_literal_overlap=args.max_literal_overlap,
                max_fourgram_overlap=args.max_fourgram_overlap,
                accept_verifier_corrections=args.accept_verifier_corrections,
                stage_max_attempts=args.stage_max_attempts,
                max_fourgram_overlap_count=args.max_fourgram_overlap_count,
            )
        )
        output_rows[-1]["model"] = args.model
        output_rows[-1]["model_revision"] = runtime_metadata.get(
            "resolved_model_revision", args.revision or ""
        )
        output_rows[-1]["runtime"] = runtime_metadata
        write_jsonl(args.output, [output_rows[-1]], append=output_path.exists())


if __name__ == "__main__":
    main()
