"""Deterministic parsing, verification, and aggregation for logic benchmarks."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json|python|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:\*{1,2})?(?:final\s+)?answer(?:\*{1,2})?\s*:\s*(.+?)\s*$"
)

_FOL_OPERATORS = {"↔": 1, "→": 2, "∨": 3, "∧": 4}


def response_tail(response: str) -> str:
    """Return the answer-bearing portion after a model's reasoning channel."""
    text = str(response).strip()
    # vLLM strips GPT-OSS control tokens while retaining the literal channel
    # labels, yielding ``analysis...assistantfinal...`` in ``candidate.text``.
    # Only the final-channel payload is an answer; scoring the analysis channel
    # can select a provisional formula, option, or binary certificate.
    for marker in ("<|channel|>final<|message|>", "assistantfinal"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1].strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    for marker in ("<|return|>", "<|end|>"):
        if text.endswith(marker):
            text = text[: -len(marker)].strip()
    return text


def answer_fragments(response: str) -> list[str]:
    """Return answer candidates from most to least explicit."""
    tail = response_tail(response)
    candidates: list[str] = []
    candidates.extend(reversed([value.strip() for value in _BOXED_RE.findall(tail)]))
    candidates.extend(reversed([value.strip() for value in _ANSWER_RE.findall(tail)]))
    candidates.extend(reversed([value.strip() for value in _FENCE_RE.findall(tail)]))
    candidates.append(tail)
    if tail != response.strip():
        candidates.append(response.strip())
    return [value.strip().strip("*").strip() for value in candidates if value.strip(" *")]


def extract_binary_answer(response: str, length: int) -> str | None:
    """Extract the final binary certificate of the requested exact length."""
    pattern = re.compile(rf"(?<![01])([01]{{{int(length)}}})(?![01])")
    for fragment in answer_fragments(response):
        matches = pattern.findall(fragment)
        if matches:
            return matches[-1]
    return None


def extract_option_set(response: str) -> list[int] | None:
    """Extract a LogicSkills select-all-that-apply answer."""
    for fragment in answer_fragments(response):
        if re.search(r"(?i)\b(?:answer\s*:\s*)?(?:none|no\s+(?:option|statement)s?)\b", fragment):
            return []
        explicit = re.search(
            r"(?i)(?:final\s+)?answer\s*:\s*\**\s*(?:only\s+)?"
            r"(?:statement|option)?s?\s*([1-6](?:\s*(?:,|and)\s*[1-6])*)",
            fragment,
        )
        if explicit:
            return sorted({int(value) for value in re.findall(r"[1-6]", explicit.group(1))})
        explicit_statement = re.search(
            r"(?i)\bonly\s+(?:statement|option)\s+([1-6])\s+(?:must|is)", fragment
        )
        if explicit_statement:
            return [int(explicit_statement.group(1))]
        bracketed = re.findall(r"\[([^\[\]]*)\]", fragment)
        for content in reversed(bracketed):
            if re.fullmatch(r"\s*(?:[1-6]\s*(?:,|and)?\s*)*", content, re.I):
                return sorted({int(value) for value in re.findall(r"[1-6]", content)})
        match = re.search(
            r"(?i)(?:option(?:s)?|statement(?:s)?)\s*[:#]?\s*((?:[1-6]\s*(?:,|and)?\s*)+)",
            fragment,
        )
        if match:
            return sorted({int(value) for value in re.findall(r"[1-6]", match.group(1))})
        if re.fullmatch(r"\s*[1-6](?:\s*(?:,|and)\s*[1-6])*[.\s]*", fragment, re.I):
            return sorted({int(value) for value in re.findall(r"[1-6]", fragment)})
    return None


def _formula_tokens(value: str) -> list[str] | None:
    """Tokenize common standard FOL notation used in model final answers."""
    text = value.strip().strip("`$")
    replacements = {
        "\\forall": "∀",
        "\\exists": "∃",
        "\\neg": "¬",
        "\\lnot": "¬",
        "\\land": "∧",
        "\\wedge": "∧",
        "\\lor": "∨",
        "\\vee": "∨",
        "\\rightarrow": "→",
        "\\to": "→",
        "\\leftrightarrow": "↔",
        "<->": "↔",
        "->": "→",
        "&": "∧",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = re.sub(r"\bnot\b", "¬", text, flags=re.I)
    text = re.sub(r"\band\b", "∧", text, flags=re.I)
    text = re.sub(r"\bor\b", "∨", text, flags=re.I)
    # Convert standard predicate application P(x,a) to the benchmark's Pxa.
    predicate_call = re.compile(r"([A-Z])\s*\(\s*([a-z])\s*(?:,\s*([a-z])\s*)?\)")
    while True:
        converted = predicate_call.sub(
            lambda match: match.group(1) + match.group(2) + (match.group(3) or ""), text
        )
        if converted == text:
            break
        text = converted
    text = re.sub(r"\s+", "", text)
    tokens = re.findall(r"[∀∃¬∧∨→↔()]|[A-Z][a-z]{1,2}|[a-z]", text)
    return tokens if "".join(tokens) == text else None


def canonicalize_formula(value: str) -> str | None:
    """Convert standard unary/binary FOL syntax to LogicSkills' strict WFF grammar."""
    tokens = _formula_tokens(value)
    if not tokens:
        return None
    position = 0

    def parse_prefix() -> tuple[Any, ...]:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("unexpected end of formula")
        token = tokens[position]
        position += 1
        if token == "¬":
            return ("not", parse_prefix())
        if token in {"∀", "∃"}:
            if position >= len(tokens) or not re.fullmatch(r"[s-z]", tokens[position]):
                raise ValueError("quantifier is not followed by a variable")
            variable = tokens[position]
            position += 1
            # Parenthesized scope is overwhelmingly the model convention.  A
            # bare scope consumes the remaining expression, matching the
            # benchmark grammar's QUANTIFIER VARIABLE WFF production.
            if position < len(tokens) and tokens[position] == "(":
                child = parse_prefix()
            else:
                child = parse_expression(0)
            return ("quant", token, variable, child)
        if token == "(":
            child = parse_expression(0)
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("unbalanced formula parentheses")
            position += 1
            return child
        if re.fullmatch(r"[A-Z][a-z]{1,2}", token):
            return ("atom", token)
        raise ValueError(f"unexpected formula token: {token}")

    def parse_expression(minimum_precedence: int) -> tuple[Any, ...]:
        nonlocal position
        left = parse_prefix()
        while position < len(tokens):
            operator = tokens[position]
            precedence = _FOL_OPERATORS.get(operator)
            if precedence is None or precedence < minimum_precedence:
                break
            position += 1
            right_associative = operator in {"→", "↔"}
            right = parse_expression(precedence if right_associative else precedence + 1)
            left = ("binary", operator, left, right)
        return left

    def serialize(node: tuple[Any, ...]) -> str:
        if node[0] == "atom":
            return str(node[1])
        if node[0] == "not":
            return "¬" + serialize(node[1])
        if node[0] == "quant":
            return str(node[1]) + str(node[2]) + serialize(node[3])
        if node[0] == "binary":
            return "(" + serialize(node[2]) + str(node[1]) + serialize(node[3]) + ")"
        raise ValueError(f"unknown formula node: {node[0]}")

    try:
        parsed = parse_expression(0)
        if position != len(tokens):
            return None
        return serialize(parsed)
    except ValueError:
        return None


def extract_formula(response: str) -> str | None:
    """Extract a single first-order formula without invoking an LLM repair step."""
    for fragment in answer_fragments(response):
        value = fragment.strip().strip("`").strip()
        value = re.sub(r"(?i)^formalization\s*:\s*", "", value).strip()
        value = re.sub(r"(?i)^(?:final\s+)?answer\s*:\s*", "", value).strip()
        if "\n" in value:
            nonempty = [line.strip() for line in value.splitlines() if line.strip()]
            formula_lines = [
                line for line in nonempty if any(symbol in line for symbol in ("∀", "∃", "¬", "∧", "∨", "→", "↔"))
            ]
            if formula_lines:
                value = formula_lines[-1]
        if value and len(value) <= 4096:
            return value
    return None


def _balanced_object(text: str) -> str | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        quote: str | None = None
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in {'"', "'"}:
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def _literal(value: str) -> Any:
    value = value.strip().rstrip(",")
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(value)


def extract_countermodel(response: str) -> dict[str, Any] | None:
    """Parse either a JSON object or the benchmark's documented line format."""
    for fragment in answer_fragments(response):
        obj_text = _balanced_object(fragment)
        if obj_text:
            try:
                value = _literal(obj_text)
            except (ValueError, SyntaxError):
                value = None
            if isinstance(value, dict):
                return value

    model: dict[str, Any] = {}
    for line in response_tail(response).splitlines():
        line = line.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
        line = re.sub(r"^\s*[-*]\s*", "", line).strip()
        line = line.replace("`", "").replace("\\(", "").replace("\\)", "")
        line = re.sub(r"^\*{1,2}(.+?)\*{1,2}\s*:", r"\1:", line)
        match = re.match(r"^[\"']?([A-Za-z][A-Za-z0-9_]*)[\"']?\s*:\s*(.+?)\s*$", line)
        if not match:
            match = re.match(r"^\*{1,2}([A-Za-z][A-Za-z0-9_]*)\*{1,2}\s*:\s*(.+?)\s*$", line)
        if not match:
            match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", line)
        if not match:
            continue
        key, raw_value = match.groups()
        if key.lower() in {"constants", "predicates", "monadic", "binary"}:
            continue
        if key.lower() == "domain":
            key = "Domain"
        try:
            model[key] = _literal(raw_value)
        except (ValueError, SyntaxError):
            continue
    return model or None


def nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(nested_tuple(item) for item in value)
    if isinstance(value, dict):
        return {key: nested_tuple(item) for key, item in value.items()}
    return value


def serialize_countermodel(model: Mapping[str, Any]) -> str:
    """Serialize parsed model syntax, including Python set literals, deterministically."""

    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (set, frozenset)):
            converted = [json_safe(item) for item in value]
            return sorted(
                converted,
                key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False),
            )
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    return json.dumps(json_safe(model), sort_keys=True, ensure_ascii=False)


def merge_smt(model_smt: str, sentence_smt: str) -> str:
    declarations: list[str] = []
    assertions: list[str] = []
    seen: set[str] = set()
    for line in (model_smt + "\n" + sentence_smt).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("(declare-"):
            if line not in seen:
                declarations.append(line)
                seen.add(line)
        else:
            assertions.append(line)
    declarations.sort(key=lambda value: 0 if "(declare-sort" in value else 1)
    return "\n".join(declarations + assertions)


def validate_countermodel(
    model: Mapping[str, Any], monadic: Iterable[str], binary: Iterable[str], names: Iterable[str]
) -> bool:
    if set(model.get("Domain", [])) != {0, 1, 2}:
        return False
    domain = list(model["Domain"])
    for symbol in [*names, *monadic, *binary]:
        if symbol not in model:
            return False
    for symbol in names:
        if not isinstance(model[symbol], int) or model[symbol] not in domain:
            return False
    for symbol in monadic:
        value = model[symbol]
        if not isinstance(value, list) or any(not isinstance(item, int) or item not in domain for item in value):
            return False
    for symbol in binary:
        value = model[symbol]
        if not isinstance(value, list):
            return False
        if any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(item, int) or item not in domain for item in pair)
            for pair in value
        ):
            return False
    return True


def countermodel_to_smt(
    model: Mapping[str, Any], names: Iterable[str], monadic: Iterable[str], binary: Iterable[str]
) -> str:
    domain = list(model["Domain"])
    element = {item: f"d{item}" for item in domain}
    lines = [f"(declare-const {element[item]} Object)" for item in domain]
    for name in names:
        lines.extend((f"(declare-const {name} Object)", f"(assert (= {name} {element[model[name]]}))"))
    for predicate in monadic:
        lines.append(f"(declare-fun {predicate} (Object) Bool)")
        true_values = set(model[predicate])
        for item in domain:
            atom = f"({predicate} {element[item]})"
            lines.append(f"(assert {atom})" if item in true_values else f"(assert (not {atom}))")
    for predicate in binary:
        lines.append(f"(declare-fun {predicate} (Object Object) Bool)")
        true_pairs = {tuple(pair) for pair in model[predicate]}
        for left in domain:
            for right in domain:
                atom = f"({predicate} {element[left]} {element[right]})"
                lines.append(
                    f"(assert {atom})" if (left, right) in true_pairs else f"(assert (not {atom}))"
                )
    closure = " ".join(f"(= x {element[item]})" for item in domain)
    lines.append(f"(assert (forall ((x Object)) (or {closure})))")
    return "\n".join(lines)


def score_logicskills(
    response: str,
    *,
    task: str,
    target: str,
    payload: Mapping[str, Any],
    logic_repo: Path,
) -> tuple[bool, str | None]:
    """Apply the benchmark's exact or Z3-backed verifier."""
    if task == "validity":
        answer = extract_option_set(response)
        return answer == sorted(payload["correct_options"]), None if answer is None else json.dumps(answer)

    import sys

    repo_text = str(logic_repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    import z3
    from Syntax.convert_to_smt import ast_to_smt2

    if task == "symbolization":
        formula = extract_formula(response)
        if formula is None:
            return False, None
        from Syntax.parse import parser
        from Syntax.transform import transformer

        def parse_formula(value: str) -> Any | None:
            for candidate in (value, f"({value})"):
                try:
                    return transformer.transform(parser.parse(candidate))
                except Exception:
                    pass
            return None

        canonical_formula = canonicalize_formula(formula)
        model_ast = parse_formula(canonical_formula or formula)
        expected_ast = parse_formula(str(payload["expected_formula"]))
        if model_ast is None or expected_ast is None:
            return False, canonical_formula or formula
        try:
            # Match the official checker: the proposed formula must entail the
            # reference formula. No heuristic or LLM syntax repair is used.
            negated = ast_to_smt2(("not", ("imp", model_ast, expected_ast)))
            solver = z3.Solver()
            solver.add(z3.parse_smt2_string(negated["smt2"]))
            return solver.check() == z3.unsat, canonical_formula or formula
        except Exception:
            return False, canonical_formula or formula

    if task == "countermodel":
        model = extract_countermodel(response)
        if model is None:
            return False, None
        try:
            sentence = ast_to_smt2(("not", nested_tuple(payload["argument_ast"])))
            names = sentence.get("names", [])
            monadic = sentence.get("monadic_predicates", [])
            binary = sentence.get("binary_predicates", [])
            if not validate_countermodel(model, monadic, binary, names):
                return False, serialize_countermodel(model)
            merged = merge_smt(
                countermodel_to_smt(model, names, monadic, binary), sentence["smt2"]
            )
            solver = z3.Solver()
            solver.add(z3.parse_smt2_string(merged))
            return solver.check() == z3.sat, serialize_countermodel(model)
        except Exception:
            return False, serialize_countermodel(model)
    raise ValueError(f"Unsupported LogicSkills task: {task}")


def grouped_accuracy(rows: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> list[dict[str, Any]]:
    """Aggregate exact verifier accuracy for arbitrary metadata slices."""
    keys = tuple(keys)
    groups: dict[tuple[Any, ...], list[bool]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(bool(row["correct"]))
    result = []
    for values, correct in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        item = {key: value for key, value in zip(keys, values)}
        item.update({"correct": sum(correct), "total": len(correct), "accuracy": sum(correct) / len(correct)})
        result.append(item)
    return result
