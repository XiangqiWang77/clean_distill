"""Skill-card-conditioned, target-disjoint specialization candidate proposal."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from typing import Any

from tqdm import tqdm

from .io import load_query_records, stable_hash, write_jsonl
from .prompts import candidate_messages, skill_card_messages, solver_messages, verifier_messages
from .runtime import HFGenerator, load_hf_model, parse_json_object, render_chat


_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+)?")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "valid"}
    return bool(value)


def _target_literals(problem: str) -> set[str]:
    literals = set(_NUMBER_RE.findall(problem))
    literals.update(_ENTITY_RE.findall(problem))
    return {literal for literal in literals if literal}


def sanitize_skill_card(skill_card: dict[str, Any], problem: str) -> tuple[dict[str, Any], list[str]]:
    """Redact target-specific literals before the proposer sees the card."""
    serialized = json.dumps(skill_card, ensure_ascii=False)
    redacted = []
    for literal in sorted(_target_literals(problem), key=len, reverse=True):
        if literal in serialized:
            serialized = serialized.replace(literal, "<redacted>")
            redacted.append(literal)
    # A skill card does not need literal numbers at all. This also removes a
    # number that the analyst may have inferred by silently solving the target.
    serialized, inferred_count = _NUMBER_RE.subn("<numeric-redacted>", serialized)
    redacted.extend(["<inferred-numeric>"] * inferred_count)
    clean = json.loads(serialized)
    clean["target_details_removed"] = True
    return clean, redacted


def target_disjoint_audit(problem: str, candidate_problem: str) -> dict[str, float]:
    target_literals = _target_literals(problem)
    candidate_literals = _target_literals(candidate_problem)
    literal_overlap = target_literals & candidate_literals

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
        "target_literal_count": float(len(target_literals)),
        "candidate_literal_count": float(len(candidate_literals)),
        "literal_overlap_count": float(len(literal_overlap)),
        "literal_overlap_rate": float(len(literal_overlap)) / max(len(candidate_literals), 1),
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
    max_literal_overlap: float,
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
    skill_raw = parse_json_object(proposer_generator(skill_card_messages(problem)))
    skill_card, redacted_literals = sanitize_skill_card(_validate_skill_card(skill_raw), problem)

    accepted: list[dict[str, Any]] = []
    seen_problems: set[str] = set()
    attempts = 0
    while len(accepted) < num_candidates and attempts < max_rounds:
        attempts += 1
        requested = num_candidates - len(accepted) + proposal_oversample
        proposed = parse_json_object(
            proposer_generator(candidate_messages(skill_card, requested))
        )
        for candidate in _candidate_list(proposed):
            candidate_problem = str(candidate["problem"]).strip()
            normalized = " ".join(candidate_problem.lower().split())
            if normalized in seen_problems:
                continue
            seen_problems.add(normalized)
            disjoint = target_disjoint_audit(problem, candidate_problem)
            if disjoint["literal_overlap_rate"] > max_literal_overlap:
                continue

            solved = parse_json_object(solver_generator(solver_messages(candidate_problem)))
            solution = str(solved.get("solution", "")).strip()
            final_answer = str(solved.get("final_answer", "")).strip()
            if not solution or not final_answer:
                continue
            verified = parse_json_object(
                verifier_generator(verifier_messages(candidate_problem, solution, final_answer))
            )
            is_valid = _as_bool(verified.get("valid", False))
            if not is_valid and not accept_verifier_corrections:
                continue
            if not is_valid:
                solution = str(verified.get("corrected_solution", "")).strip()
                final_answer = str(verified.get("corrected_final_answer", "")).strip()
                if not solution or not final_answer:
                    continue

            accepted.append(
                {
                    "candidate_id": f"c{len(accepted):02d}",
                    "problem": candidate_problem,
                    "skill_tags": candidate.get("skill_tags", []),
                    "solution": solution,
                    "final_answer": final_answer,
                    "verifier_valid": is_valid,
                    "verifier_reason": str(verified.get("reason", "")),
                    "target_disjoint_audit": disjoint,
                }
            )
            if len(accepted) >= num_candidates:
                break

    if len(accepted) < num_candidates:
        raise RuntimeError(
            f"{record['query_id']}: obtained {len(accepted)}/{num_candidates} verified candidates "
            f"after {max_rounds} rounds"
        )

    # Prompt hashes make the information boundary auditable without storing
    # every system prompt. Candidate/solver/verifier prompts contain no target.
    card_prompt = render_chat(
        proposer_generator.tokenizer,
        skill_card_messages(problem),
        add_generation_prompt=True,
    )
    candidate_prompt = render_chat(
        proposer_generator.tokenizer,
        candidate_messages(skill_card, num_candidates),
        add_generation_prompt=True,
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
    return {
        "schema_version": "clean-self-distill-proposals-v1",
        **record,
        "problem_sha256": stable_hash(problem, length=64),
        "skill_card": skill_card,
        "specialization_candidates": accepted,
        "candidate_count": len(accepted),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL/JSON/parquet dataset; verl parquet is supported")
    parser.add_argument("--output", required=True, help="Output proposal JSONL")
    parser.add_argument("--model", required=True, help="Local or Hugging Face causal LM")
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--proposal-oversample", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=4)
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
        default=1.0,
        help="Optional lexical audit filter. Context isolation, not zero accidental overlap, is the firewall.",
    )
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
    records = load_query_records(args.input, include_targets=False, max_samples=args.max_samples)
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        training=False,
    )
    proposer_generator = HFGenerator(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    solver_generator = HFGenerator(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.solver_temperature,
        top_p=args.top_p,
    )
    verifier_generator = HFGenerator(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.verifier_temperature,
        top_p=1.0,
    )

    output_rows = []
    for record in tqdm(records, desc="propose+solve+verify"):
        output_rows.append(
            propose_for_query(
                record,
                proposer_generator,
                solver_generator,
                verifier_generator,
                num_candidates=args.num_candidates,
                proposal_oversample=args.proposal_oversample,
                max_rounds=args.max_rounds,
                max_literal_overlap=args.max_literal_overlap,
                accept_verifier_corrections=args.accept_verifier_corrections,
            )
        )
        write_jsonl(args.output, [output_rows[-1]], append=len(output_rows) > 1)


if __name__ == "__main__":
    main()
