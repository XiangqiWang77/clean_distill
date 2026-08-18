"""Pure validation and aggregation helpers for Arena preference likelihood.

The evaluation artifact contains existing human-preference pairs only.  A
model is never asked to generate an answer or judge another model's answer:
it teacher-forces the two recorded responses and reports their mean token
log-probability difference.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


PAIR_SCHEMA_VERSION = "arena-human-preference-pair-v1"
SCORE_SCHEMA_VERSION = "arena-human-preference-logprob-v1"
SCORE_SUMMARY_SCHEMA_VERSION = "arena-human-preference-summary-v1"

METHOD_ORDER = (
    "Base",
    "LGSD-Small",
    "LGSD-Medium",
    "LGSD-Large",
    "OPSD",
)


class ArenaPreferenceError(ValueError):
    """Raised when a preference-pair or score artifact is inconsistent."""


def normalize_prompt(value: str) -> str:
    """Canonical prompt text used for split-disjointness checks."""
    return " ".join(str(value).split()).casefold()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArenaPreferenceError(f"{context} must be a non-empty string")
    return value.strip()


def _sha256(value: object, *, context: str) -> str:
    text = _text(value, context=context).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ArenaPreferenceError(f"{context} must be a SHA-256 hex digest")
    return text


def _finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArenaPreferenceError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ArenaPreferenceError(f"{context} must be finite")
    return result


def _positive_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArenaPreferenceError(f"{context} must be a positive integer")
    return value


def validate_preference_pair(
    row: Mapping[str, Any], *, row_number: int | None = None
) -> dict[str, Any]:
    """Validate one evaluation-only human-preference pair."""
    prefix = f"pair row {row_number}" if row_number is not None else "pair"
    if row.get("schema_version") != PAIR_SCHEMA_VERSION:
        raise ArenaPreferenceError(f"{prefix} has the wrong schema_version")
    if row.get("evaluation_only") is not True:
        raise ArenaPreferenceError(f"{prefix} must be marked evaluation_only")
    if row.get("label_source") != "lmarena_human_vote":
        raise ArenaPreferenceError(f"{prefix} must use an existing LMArena vote")
    if row.get("external_judge_used") is not False:
        raise ArenaPreferenceError(f"{prefix} must not use an external judge")

    query_id = _text(row.get("query_id"), context=f"{prefix}.query_id")
    prompt = _text(row.get("prompt"), context=f"{prefix}.prompt")
    prompt_sha256 = _sha256(
        row.get("prompt_sha256"), context=f"{prefix}.prompt_sha256"
    )
    normalized_prompt_sha256 = _sha256(
        row.get("normalized_prompt_sha256"),
        context=f"{prefix}.normalized_prompt_sha256",
    )
    if prompt_sha256 != sha256_text(prompt):
        raise ArenaPreferenceError(f"{prefix}.prompt_sha256 disagrees with prompt")
    if normalized_prompt_sha256 != sha256_text(normalize_prompt(prompt)):
        raise ArenaPreferenceError(
            f"{prefix}.normalized_prompt_sha256 disagrees with prompt"
        )

    preferred = _text(
        row.get("preferred_response"), context=f"{prefix}.preferred_response"
    )
    rejected = _text(
        row.get("rejected_response"), context=f"{prefix}.rejected_response"
    )
    preferred_sha256 = _sha256(
        row.get("preferred_response_sha256"),
        context=f"{prefix}.preferred_response_sha256",
    )
    rejected_sha256 = _sha256(
        row.get("rejected_response_sha256"),
        context=f"{prefix}.rejected_response_sha256",
    )
    if preferred_sha256 != sha256_text(preferred):
        raise ArenaPreferenceError(
            f"{prefix}.preferred_response_sha256 disagrees with response"
        )
    if rejected_sha256 != sha256_text(rejected):
        raise ArenaPreferenceError(
            f"{prefix}.rejected_response_sha256 disagrees with response"
        )
    if preferred_sha256 == rejected_sha256:
        raise ArenaPreferenceError(f"{prefix} has identical preferred/rejected text")

    raw_domains = row.get("domains")
    if not isinstance(raw_domains, list) or not raw_domains:
        raise ArenaPreferenceError(f"{prefix}.domains must be a non-empty list")
    domains: list[str] = []
    for index, raw_domain in enumerate(raw_domains):
        domain = _text(raw_domain, context=f"{prefix}.domains[{index}]")
        if domain in domains:
            raise ArenaPreferenceError(f"{prefix}.domains contains duplicates")
        domains.append(domain)
    if domains != sorted(domains):
        raise ArenaPreferenceError(f"{prefix}.domains must be sorted")

    source = _text(row.get("source"), context=f"{prefix}.source")
    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "evaluation_only": True,
        "label_source": "lmarena_human_vote",
        "external_judge_used": False,
        "query_id": query_id,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "normalized_prompt_sha256": normalized_prompt_sha256,
        "preferred_response": preferred,
        "preferred_response_sha256": preferred_sha256,
        "rejected_response": rejected,
        "rejected_response_sha256": rejected_sha256,
        "domains": domains,
        "source": source,
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ArenaPreferenceError(
                        f"{source}:{line_number} must contain one JSON object"
                    )
                rows.append(dict(raw))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArenaPreferenceError(f"cannot read {source}: {exc}") from exc
    if not rows:
        raise ArenaPreferenceError(f"{source} contains no rows")
    return rows


def load_preference_pairs(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        validate_preference_pair(row, row_number=index)
        for index, row in enumerate(read_jsonl(path), 1)
    ]
    ids = [str(row["query_id"]) for row in rows]
    normalized = [str(row["normalized_prompt_sha256"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ArenaPreferenceError(f"{path} contains duplicate query_id values")
    if len(normalized) != len(set(normalized)):
        raise ArenaPreferenceError(f"{path} contains duplicate normalized prompts")
    return rows


def validate_score_row(
    row: Mapping[str, Any], *, row_number: int | None = None
) -> dict[str, Any]:
    """Validate one model/pair teacher-forced log-probability row."""
    prefix = f"score row {row_number}" if row_number is not None else "score row"
    if row.get("schema_version") != SCORE_SCHEMA_VERSION:
        raise ArenaPreferenceError(f"{prefix} has the wrong schema_version")
    if row.get("external_judge_used") is not False:
        raise ArenaPreferenceError(f"{prefix} unexpectedly uses an external judge")
    method = _text(row.get("method"), context=f"{prefix}.method")
    if method not in METHOD_ORDER:
        raise ArenaPreferenceError(f"{prefix}.method is not recognized: {method}")
    checkpoint = row.get("checkpoint")
    if isinstance(checkpoint, bool) or not isinstance(checkpoint, int) or checkpoint < 0:
        raise ArenaPreferenceError(f"{prefix}.checkpoint must be non-negative")
    query_id = _text(row.get("query_id"), context=f"{prefix}.query_id")
    global_index = row.get("global_query_index")
    if isinstance(global_index, bool) or not isinstance(global_index, int) or global_index < 0:
        raise ArenaPreferenceError(f"{prefix}.global_query_index must be non-negative")

    preferred_tokens = _positive_integer(
        row.get("preferred_token_count"),
        context=f"{prefix}.preferred_token_count",
    )
    rejected_tokens = _positive_integer(
        row.get("rejected_token_count"),
        context=f"{prefix}.rejected_token_count",
    )
    preferred_logprob = _finite(
        row.get("preferred_mean_logprob"),
        context=f"{prefix}.preferred_mean_logprob",
    )
    rejected_logprob = _finite(
        row.get("rejected_mean_logprob"),
        context=f"{prefix}.rejected_mean_logprob",
    )
    margin = _finite(
        row.get("preference_margin"), context=f"{prefix}.preference_margin"
    )
    if not math.isclose(
        margin,
        preferred_logprob - rejected_logprob,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ArenaPreferenceError(f"{prefix}.preference_margin is inconsistent")
    correct = row.get("preference_correct")
    expected_correct = preferred_logprob > rejected_logprob
    if not isinstance(correct, bool) or correct != expected_correct:
        raise ArenaPreferenceError(f"{prefix}.preference_correct is inconsistent")

    domains = row.get("domains")
    if (
        not isinstance(domains, list)
        or not domains
        or any(not isinstance(item, str) or not item for item in domains)
        or domains != sorted(set(domains))
    ):
        raise ArenaPreferenceError(f"{prefix}.domains must be sorted and distinct")
    prompt_tokens = _positive_integer(
        row.get("prompt_token_count"), context=f"{prefix}.prompt_token_count"
    )
    return {
        **dict(row),
        "method": method,
        "checkpoint": checkpoint,
        "query_id": query_id,
        "global_query_index": global_index,
        "preferred_token_count": preferred_tokens,
        "rejected_token_count": rejected_tokens,
        "preferred_mean_logprob": preferred_logprob,
        "rejected_mean_logprob": rejected_logprob,
        "preference_margin": margin,
        "preference_correct": correct,
        "domains": list(domains),
        "prompt_token_count": prompt_tokens,
    }


def summarize_score_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute pair-macro preference statistics for one model checkpoint."""
    checked = [
        validate_score_row(row, row_number=index)
        for index, row in enumerate(rows, 1)
    ]
    if not checked:
        raise ArenaPreferenceError("cannot summarize an empty score set")
    identities = {(row["method"], row["checkpoint"]) for row in checked}
    if len(identities) != 1:
        raise ArenaPreferenceError("score rows mix method/checkpoint identities")
    query_ids = [str(row["query_id"]) for row in checked]
    global_indices = [int(row["global_query_index"]) for row in checked]
    if len(query_ids) != len(set(query_ids)):
        raise ArenaPreferenceError("score rows contain duplicate query_id values")
    if len(global_indices) != len(set(global_indices)):
        raise ArenaPreferenceError("score rows contain duplicate global indices")

    count = len(checked)
    preferred = [float(row["preferred_mean_logprob"]) for row in checked]
    rejected = [float(row["rejected_mean_logprob"]) for row in checked]
    margins = [float(row["preference_margin"]) for row in checked]
    correct = [bool(row["preference_correct"]) for row in checked]
    preferred_truncated = [
        bool((row.get("preferred_truncation") or {}).get("applied"))
        for row in checked
    ]
    rejected_truncated = [
        bool((row.get("rejected_truncation") or {}).get("applied"))
        for row in checked
    ]
    method, checkpoint = next(iter(identities))
    return {
        "schema_version": SCORE_SUMMARY_SCHEMA_VERSION,
        "method": method,
        "checkpoint": checkpoint,
        "pair_count": count,
        "preferred_mean_logprob": math.fsum(preferred) / count,
        "rejected_mean_logprob": math.fsum(rejected) / count,
        "preference_margin": math.fsum(margins) / count,
        "preference_accuracy": math.fsum(float(value) for value in correct) / count,
        "preferred_token_count": sum(
            int(row["preferred_token_count"]) for row in checked
        ),
        "rejected_token_count": sum(
            int(row["rejected_token_count"]) for row in checked
        ),
        "preferred_truncation_fraction": math.fsum(
            float(value) for value in preferred_truncated
        )
        / count,
        "rejected_truncation_fraction": math.fsum(
            float(value) for value in rejected_truncated
        )
        / count,
        "aggregation": "pair_macro_after_per_response_token_mean",
        "preference_definition": "mean_logprob(y+|x)-mean_logprob(y-|x)",
        "external_judge_used": False,
    }


def align_score_rows(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return candidate rows aligned to a complete query-matched reference."""
    left = {
        str(row["query_id"]): validate_score_row(row, row_number=index)
        for index, row in enumerate(reference, 1)
    }
    right = {
        str(row["query_id"]): validate_score_row(row, row_number=index)
        for index, row in enumerate(candidate, 1)
    }
    if len(left) != len(reference) or len(right) != len(candidate):
        raise ArenaPreferenceError("cannot align score rows with duplicate query IDs")
    if set(left) != set(right):
        missing = sorted(set(left) - set(right))[:5]
        extra = sorted(set(right) - set(left))[:5]
        raise ArenaPreferenceError(
            f"score coverage differs: missing={missing}, extra={extra}"
        )
    ordered_ids = sorted(
        left,
        key=lambda query_id: (
            int(left[query_id]["global_query_index"]),
            query_id,
        ),
    )
    aligned: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for query_id in ordered_ids:
        reference_row = left[query_id]
        candidate_row = right[query_id]
        if (
            reference_row["global_query_index"]
            != candidate_row["global_query_index"]
            or reference_row.get("prompt_sha256")
            != candidate_row.get("prompt_sha256")
            or reference_row.get("preferred_response_sha256")
            != candidate_row.get("preferred_response_sha256")
            or reference_row.get("rejected_response_sha256")
            != candidate_row.get("rejected_response_sha256")
            or reference_row["domains"] != candidate_row["domains"]
        ):
            raise ArenaPreferenceError(
                f"score identity differs for query {query_id!r}"
            )
        aligned.append((reference_row, candidate_row))
    return aligned


__all__ = [
    "ArenaPreferenceError",
    "METHOD_ORDER",
    "PAIR_SCHEMA_VERSION",
    "SCORE_SCHEMA_VERSION",
    "SCORE_SUMMARY_SCHEMA_VERSION",
    "align_score_rows",
    "load_preference_pairs",
    "normalize_prompt",
    "read_jsonl",
    "sha256_text",
    "summarize_score_rows",
    "validate_preference_pair",
    "validate_score_row",
]
