"""Skill-card-conditioned, target-disjoint specialization candidate proposal."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
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


_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d[\d,]*(?:\.\d+)?)?"
)
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_MATH_SPAN_RE = re.compile(
    r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]", flags=re.DOTALL
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
_GENERIC_SENTENCE_WORDS = {
    "a",
    "an",
    "assume",
    "compute",
    "consider",
    "determine",
    "each",
    "evaluate",
    "find",
    "for",
    "from",
    "given",
    "how",
    "if",
    "in",
    "let",
    "of",
    "on",
    "solve",
    "suppose",
    "the",
    "to",
    "what",
    "when",
    "which",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "valid"}
    return bool(value)


def _canonical_number(value: str) -> str:
    return re.sub(r"[\s,]", "", value).lstrip("+")


def _numeric_literals(text: str) -> set[str]:
    return {
        normalized
        for raw in _NUMBER_RE.findall(text)
        if (normalized := _canonical_number(raw))
    }


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
    }


def _word_literals(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text)}


def sanitize_skill_card(skill_card: dict[str, Any], problem: str) -> tuple[dict[str, Any], list[str]]:
    """Redact target-specific literals before the proposer sees the card."""
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
            return "redacted number"
        if not isinstance(value, str):
            return value
        text, expression_count = _MATH_SPAN_RE.subn("redacted expression", value)
        redacted.extend(["<math-expression>"] * expression_count)
        text, cue_count = _DIRECT_ANSWER_CUE_RE.subn("redacted conclusion", text)
        redacted.extend(["<direct-answer-cue>"] * cue_count)
        text, number_word_count = _ENGLISH_NUMBER_WORD_RE.subn(
            "redacted quantity", text
        )
        redacted.extend(["<english-number-word>"] * number_word_count)
        for literal in target_entities:
            text, count = re.subn(
                rf"\b{re.escape(literal)}\b",
                "redacted detail",
                text,
                flags=re.IGNORECASE,
            )
            redacted.extend([literal] * count)
        # A skill card does not need literal numbers at all. This also removes
        # a number the analyst may have inferred by silently solving the target.
        text, inferred_count = _NUMBER_RE.subn("redacted number", text)
        redacted.extend(["<inferred-numeric>"] * inferred_count)
        for symbol in sorted(target_symbols):
            text, symbol_count = re.subn(
                rf"\b{re.escape(symbol)}\b",
                "redacted symbol",
                text,
                flags=re.IGNORECASE,
            )
            redacted.extend([symbol] * symbol_count)
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

    All target numbers and target-specific capitalized entities are forbidden.
    This is deliberately stricter than a fuzzy similarity score: the proposer
    never receives these literals, so sharing one is unnecessary and can be
    rejected without weakening the intended skill match.
    """

    target_numbers = _numeric_literals(problem)
    candidate_numbers = _numeric_literals(candidate_problem)
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
    # Unknown fields such as `solution` or `answer` are intentionally dropped
    # even if a model violates the requested schema.
    return {
        "domain": str(value["domain"]),
        "skills": [str(item) for item in value["skills"]],
        "reasoning_operators": [str(item) for item in value["reasoning_operators"]],
        "difficulty": str(value["difficulty"]),
        "constraints": [str(item) for item in value.get("constraints", [])],
        "target_details_removed": True,
    }


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
        except (ValueError, json.JSONDecodeError) as exc:
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
        raise RuntimeError(
            f"{record['query_id']}: could not produce a safe, parseable skill card"
        )

    accepted: list[dict[str, Any]] = []
    seen_problems: set[str] = set()
    candidate_attempts: list[dict[str, Any]] = []
    proposal_rounds: list[dict[str, Any]] = []
    attempts = 0
    while len(accepted) < num_candidates and attempts < max_rounds:
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

            candidate_id = f"c{len(accepted):02d}"
            accepted.append(
                {
                    "candidate_id": candidate_id,
                    "problem": candidate_problem,
                    "skill_tags": candidate.get("skill_tags", []),
                    "solution": solution,
                    "final_answer": final_answer,
                    "verifier_valid": is_valid,
                    "verifier_accepted": True,
                    "verifier_reason": str(verified.get("reason", "")),
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
                verifier_reason=str(verified.get("reason", "")),
            )
            candidate_attempts.append(trace)
            if len(accepted) >= num_candidates:
                break

    if len(accepted) < min_accepted_candidates:
        raise RuntimeError(
            f"{record['query_id']}: obtained {len(accepted)}/{num_candidates} verified candidates; "
            f"minimum is {min_accepted_candidates} after {max_rounds} rounds"
        )

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
        "schema_version": "clean-self-distill-proposals-v3",
        **record,
        "problem_sha256": stable_hash(problem, length=64),
        "skill_card": skill_card,
        "skill_card_target_disjoint_audit": skill_card_audit,
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
            expected_ids = {record["query_id"] for record in records}
            unexpected_ids = set(existing_rows) - expected_ids
            if unexpected_ids:
                raise ValueError(
                    f"Resume file {output_path} contains {len(unexpected_ids)} rows "
                    "outside this shard/config"
                )
            for query_id, row in existing_rows.items():
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
            existing_ids = set(existing_rows)
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
