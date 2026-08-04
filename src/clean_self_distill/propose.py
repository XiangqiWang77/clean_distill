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
from .prompts import candidate_messages, skill_card_messages, solver_messages, verifier_messages
from .runtime import (
    HFGenerator,
    collect_runtime_metadata,
    load_hf_model,
    parse_json_object,
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
_SCALAR_NUMBER_PATTERN = (
    rf"(?<!\d){_SIGNED_NUMERIC_ATOM_PATTERN}(?!\d)"
)
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
_ELLIPSIS_PATTERN = (
    r"(?:\\(?:[lc]?dots[a-z]?|mathellipsis)(?![A-Za-z])|\.{3}|…)"
)
_ELLIPSIS_RE = re.compile(_ELLIPSIS_PATTERN)
_GROUPED_NUMBER_PATTERN = (
    r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
)
_GROUPED_AFTER_ELLIPSIS_RE = re.compile(
    rf"{_ELLIPSIS_PATTERN}\s*,\s*(?P<number>{_GROUPED_NUMBER_PATTERN})"
)
_ZERO_PADDED_GROUPED_NUMBER_RE = re.compile(
    r"(?<!\d)[-+]?\d{1,3}(?:,0\d{2})+(?:\.\d+)?"
    r"(?!\d)"
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
    rf"{_SEQUENCE_SCALAR_PATTERN}"
    rf"(?:\s*,\s*{_SEQUENCE_SCALAR_PATTERN})*"
)
_UNBRACKETED_ELLIPSIS_SEQUENCE_RE = re.compile(
    rf"(?<!\d)(?P<sequence>{_SEQUENCE_SIDE_PATTERN}\s*,?\s*"
    rf"{_ELLIPSIS_PATTERN}\s*,?\s*{_SEQUENCE_SIDE_PATTERN})"
    rf"(?!\d)"
)
_COMMA_EXPRESSION_OPERATOR_RE = re.compile(
    r"[=<>+*/^:|]"
    r"|\\(?:leq?|geq?|neq|approx|sim|mid|in|notin|times|cdot|pm|mp)\b"
)
_TEX_BRACED_COMMA_RE = re.compile(r"\{\s*,\s*\}")
_TEX_TEXT_COMMA_RE = re.compile(
    r"(?<=\d)\\(?:text|mathrm)\s*\{\s*,\s*\}\s*(?=\d{3}(?!\d))"
)
_TEX_COMMA_TIGHT_SPACING_RE = re.compile(
    r"(?<=\d),\s*\\[!,;:]\s*(?=\d{3}(?!\d))"
)
_TEX_TIGHT_SPACING_RE = re.compile(r"\\[,!;:]")
_TEX_WIDE_SPACING_RE = re.compile(r"\\(?:quad|qquad)\b")
_TEX_NAMED_THINSPACE_RE = re.compile(
    r"(?<=\d)\\thinspace\s*(?=\d{3}(?!\d))"
)
_UNICODE_GROUPING_SPACE_RE = re.compile(
    r"(?<=\d)[\u00a0\u2009\u202f](?=\d{3}(?!\d))"
)
_ASCII_GROUPING_SPACE_RE = re.compile(
    r"(?<=\d)[ \t]+(?=\d{3}(?!\d))"
)
_TILDE_GROUPING_SPACE_RE = re.compile(
    r"(?<=\d)~(?=\d{3}(?!\d))"
)
_SIGN_SPACING_RE = re.compile(
    r"(?P<sign>[-+])\s+(?=(?:\d|\.|\{|\\(?:frac|dfrac|tfrac|cfrac)))"
)
_TEX_SIZE_DELIMITER_RE = re.compile(
    r"\\(?:bigg|big)[lr]?\s*", flags=re.IGNORECASE
)
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_MATH_SPAN_RE = re.compile(
    r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]", flags=re.DOTALL
)
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
        rf"(?<![\d.])(?P<whole>{_SIGNED_NUMERIC_ATOM_PATTERN})"
        rf"{mixed_separator}$",
        prefix,
    )
    if whole_match is None:
        return None
    whole = _canonical_fraction_value(_canonical_numeric_atom(whole_match.group("whole")))
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


def sanitize_skill_card(skill_card: dict[str, Any], problem: str) -> tuple[dict[str, Any], list[str]]:
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


def skill_card_disjoint_audit(problem: str, skill_card: dict[str, Any]) -> dict[str, Any]:
    values = [
        str(skill_card.get("domain", "")),
        *(str(item) for item in skill_card.get("skills", [])),
        *(str(item) for item in skill_card.get("reasoning_operators", [])),
        str(skill_card.get("difficulty", "")),
        *(str(item) for item in skill_card.get("constraints", [])),
    ]
    card_text = " ".join(values)
    lexical = target_disjoint_audit(problem, card_text)
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
        {match.group(0).casefold() for match in _ENGLISH_NUMBER_WORD_RE.finditer(card_text)}
    )
    direct_answer_cues = sorted(
        {match.group(0).casefold() for match in _DIRECT_ANSWER_CUE_RE.finditer(card_text)}
    )
    safe = (
        lexical["literal_overlap_count"] == 0
        and lexical["fourgram_overlap_count"] <= 1
        and lexical["fourgram_overlap_rate"] <= 0.05
        and not shared_symbols
        and not symbolic_details
        and not english_number_words
        and not direct_answer_cues
    )
    return {
        **lexical,
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
    fractional_values = _fraction_numeric_literals(problem) | _fraction_numeric_literals(
        candidate_problem
    )
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
    target_ngrams = set(zip(*(target_tokens[offset:] for offset in range(4)))) if len(target_tokens) >= 4 else set()
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
        "fourgram_overlap_rate": float(len(overlap_ngrams)) / max(len(candidate_ngrams), 1),
    }


def _validate_skill_card(value: dict[str, Any]) -> dict[str, Any]:
    required = {"domain", "skills", "reasoning_operators", "difficulty"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"Skill card missing fields: {sorted(missing)}")
    if not isinstance(value["skills"], list) or not value["skills"]:
        raise ValueError("Skill card skills must be a non-empty list")
    if not isinstance(value["reasoning_operators"], list) or not value[
        "reasoning_operators"
    ]:
        raise ValueError("Skill card reasoning_operators must be a non-empty list")
    constraints = value.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("Skill card constraints must be a list")
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
            "difficulty": "general",
            "constraints": [],
            "target_details_removed": True,
        },
        {
            "domain": "a",
            "skills": ["i"],
            "reasoning_operators": ["a"],
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
        if not isinstance(candidate, dict) or not str(candidate.get("problem", "")).strip():
            continue
        valid.append(candidate)
    return valid


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
) -> dict[str, Any]:
    # Only record["problem"] is available here; load_query_records excluded targets.
    problem = record["problem"]
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
    for skill_attempt in range(max_rounds):
        raw_text = proposer_generator(skill_card_messages(problem))
        try:
            skill_raw = parse_json_object(raw_text)
            candidate_card, candidate_redactions = sanitize_skill_card(
                _validate_skill_card(skill_raw), problem
            )
            candidate_audit = skill_card_disjoint_audit(problem, candidate_card)
            if not candidate_audit["safe"]:
                raise ValueError(
                    "sanitized skill card still contains target-specific lexical or symbolic detail"
                )
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
        try:
            proposed = parse_json_object(raw_proposal)
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
                "max_fourgram_overlap_count": 1,
            }
            disjoint["safe"] = bool(
                disjoint["literal_overlap_count"] == 0
                and disjoint["fourgram_overlap_count"] <= 1
                and disjoint["fourgram_overlap_rate"] <= max_fourgram_overlap
            )
            trace["target_disjoint_audit"] = disjoint
            if disjoint["literal_overlap_count"] > 0:
                trace.update(outcome="rejected", reason="literal_overlap")
                candidate_attempts.append(trace)
                continue
            if (
                disjoint["fourgram_overlap_count"] > 1
                or disjoint["fourgram_overlap_rate"] > max_fourgram_overlap
            ):
                trace.update(outcome="rejected", reason="fourgram_overlap")
                candidate_attempts.append(trace)
                continue

            raw_solution = solver_generator(solver_messages(candidate_problem))
            trace["solver_raw_response"] = raw_solution
            try:
                solved = parse_json_object(raw_solution)
            except (ValueError, json.JSONDecodeError) as exc:
                trace.update(outcome="rejected", reason="solver_parse_error", error=str(exc))
                candidate_attempts.append(trace)
                continue
            solution = str(solved.get("solution", "")).strip()
            final_answer = str(solved.get("final_answer", "")).strip()
            if not solution or not final_answer:
                trace.update(outcome="rejected", reason="solver_missing_fields")
                candidate_attempts.append(trace)
                continue
            solver_placeholder_audit = _placeholder_artifact_audit(
                f"{solution}\n{final_answer}"
            )
            trace["solver_placeholder_artifact_audit"] = solver_placeholder_audit
            if not solver_placeholder_audit["safe"]:
                trace["placeholder_artifact_audit"] = solver_placeholder_audit
                trace.update(
                    outcome="rejected",
                    reason="placeholder_artifact",
                    placeholder_artifact_source="solver_output",
                )
                candidate_attempts.append(trace)
                continue
            raw_verification = verifier_generator(
                verifier_messages(candidate_problem, solution, final_answer)
            )
            trace["verifier_raw_response"] = raw_verification
            try:
                verified = parse_json_object(raw_verification)
            except (ValueError, json.JSONDecodeError) as exc:
                trace.update(outcome="rejected", reason="verifier_parse_error", error=str(exc))
                candidate_attempts.append(trace)
                continue
            is_valid = _as_bool(verified.get("valid", False))
            if not is_valid and not accept_verifier_corrections:
                trace.update(
                    outcome="rejected",
                    reason="verifier_invalid",
                    verifier_reason=str(verified.get("reason", "")),
                )
                candidate_attempts.append(trace)
                continue
            if not is_valid:
                solution = str(verified.get("corrected_solution", "")).strip()
                final_answer = str(verified.get("corrected_final_answer", "")).strip()
                if not solution or not final_answer:
                    trace.update(outcome="rejected", reason="invalid_correction")
                    candidate_attempts.append(trace)
                    continue

            verifier_reason = str(verified.get("reason", ""))
            accepted_placeholder_audit = _placeholder_artifact_audit(
                "\n".join(
                    [
                        candidate_problem,
                        *(str(skill_tag) for skill_tag in skill_tags),
                        solution,
                        final_answer,
                    ]
                )
            )
            trace["accepted_candidate_placeholder_artifact_audit"] = (
                accepted_placeholder_audit
            )
            if not accepted_placeholder_audit["safe"]:
                trace["placeholder_artifact_audit"] = accepted_placeholder_audit
                trace.update(
                    outcome="rejected",
                    reason="placeholder_artifact",
                    placeholder_artifact_source="verifier_output",
                )
                candidate_attempts.append(trace)
                continue
            trace["placeholder_artifact_audit"] = accepted_placeholder_audit

            candidate_id = f"c{len(accepted):02d}"
            accepted.append(
                {
                    "candidate_id": candidate_id,
                    "problem": candidate_problem,
                    "skill_tags": skill_tags,
                    "solution": solution,
                    "final_answer": final_answer,
                    "verifier_valid": is_valid,
                    "verifier_accepted": True,
                    "verifier_reason": verifier_reason,
                    "placeholder_artifact_audit": accepted_placeholder_audit,
                    "target_disjoint_audit": disjoint,
                }
            )
            trace.update(
                outcome="accepted",
                reason="verifier_valid" if is_valid else "verifier_corrected",
                accepted_candidate_id=candidate_id,
                solver_solution=solution,
                solver_final_answer=final_answer,
                verifier_valid=is_valid,
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
        "schema_version": "clean-self-distill-proposals-v4",
        **record,
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
            "total_prompt_tokens": sum(value["prompt_tokens"] for value in role_costs.values()),
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
            "solver_sources": ["candidate_problem"],
            "verifier_sources": ["candidate_problem", "candidate_solution"],
            "skill_card_redaction_count": len(redacted_literals),
            "skill_prompt_sha256": stable_hash(card_prompt, length=64),
            "candidate_prompt_sha256": stable_hash(candidate_prompt, length=64),
        },
    }
    row["proposal_training_sha256"] = compute_proposal_training_sha256(row)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL/JSON/parquet dataset; verl parquet is supported")
    parser.add_argument("--output", required=True, help="Output proposal JSONL")
    parser.add_argument("--model", required=True, help="Local or Hugging Face causal LM")
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
    records = load_query_records(args.input, include_targets=False, max_samples=args.max_samples)
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
