"""Input/output helpers for JSONL and verl-compatible parquet datasets."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:length]


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value using the repository's canonical serialization."""
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_specialization_state(
    row: dict[str, Any], *, context: str = "proposal"
) -> tuple[str, str, bool]:
    """Validate the shared ready/no-op specialization state machine."""
    status = row.get("specialization_status")
    reason = row.get("specialization_failure_reason")
    no_op = row.get("specialization_no_op")
    if status not in {"ready", "insufficient_verified_candidates"}:
        raise ValueError(
            f"{context} specialization_status must be exactly 'ready' or "
            "'insufficient_verified_candidates'"
        )
    if not isinstance(reason, str):
        raise ValueError(f"{context} specialization_failure_reason must be a string")
    if not isinstance(no_op, bool):
        raise ValueError(f"{context} specialization_no_op must be a boolean")
    candidates = row.get("specialization_candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"{context} specialization_candidates must be a list")
    if status == "ready":
        if reason != "" or no_op or not candidates:
            raise ValueError(
                f"{context} ready specialization requires candidates, an empty "
                "failure reason, and specialization_no_op=false"
            )
    elif not reason.strip() or not no_op:
        raise ValueError(
            f"{context} insufficient specialization requires a nonempty failure "
            "reason and specialization_no_op=true"
        )
    return status, reason, no_op


def proposal_training_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Return the exact target-disjoint content used to fit a query teacher.

    Keeping this payload deliberately small and explicit lets proposal,
    adapter, and evaluation artifacts independently recompute the same binding
    while excluding runtime-only metadata.
    """
    query_id = str(row.get("query_id", "")).strip()
    problem_sha256 = str(row.get("problem_sha256", "")).strip().lower()
    skill_card = row.get("skill_card")
    candidates = row.get("specialization_candidates")
    if not query_id:
        raise ValueError("Proposal training payload is missing query_id")
    if not re.fullmatch(r"[0-9a-f]{64}", problem_sha256):
        raise ValueError(
            f"Proposal {query_id!r} has an invalid 64-character problem_sha256"
        )
    if not isinstance(skill_card, dict):
        raise ValueError(f"Proposal {query_id!r} is missing a skill_card object")
    if not isinstance(candidates, list):
        raise ValueError(f"Proposal {query_id!r} specialization_candidates must be a list")
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise ValueError(
            f"Proposal {query_id!r} specialization_candidates must be JSON objects"
        )
    status, reason, no_op = validate_specialization_state(
        row, context=f"Proposal {query_id!r}"
    )
    return {
        "query_id": query_id,
        "problem_sha256": problem_sha256,
        "skill_card": skill_card,
        "specialization_candidates": candidates,
        "specialization_status": status,
        "specialization_failure_reason": reason,
        "specialization_no_op": no_op,
    }


def compute_proposal_training_sha256(row: dict[str, Any]) -> str:
    return canonical_json_sha256(proposal_training_payload(row))


def validate_proposal_training_binding(
    row: dict[str, Any], *, context: str = "proposal"
) -> str:
    """Fail closed when proposal training content is missing or was modified."""
    query_id = str(row.get("query_id", "")).strip()
    declared = str(row.get("proposal_training_sha256", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", declared):
        raise ValueError(
            f"{context} {query_id!r} is missing a valid proposal_training_sha256"
        )
    expected = compute_proposal_training_sha256(row)
    if declared != expected:
        raise ValueError(
            f"{context} {query_id!r} proposal_training_sha256 does not match "
            "its skill card, accepted candidates, and specialization state"
        )
    return expected


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for value in content:
            if isinstance(value, dict):
                parts.append(str(value.get("text", value.get("content", ""))))
            elif value is not None:
                parts.append(str(value))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def extract_problem(row: dict[str, Any]) -> str:
    """Extract a raw problem from common JSON/HF/verl schemas."""
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("problem"):
        return _content_to_text(extra["problem"]).strip()

    for key in ("problem", "question", "query"):
        if row.get(key):
            return _content_to_text(row[key]).strip()

    prompt = row.get("prompt", "")
    if isinstance(prompt, list):
        user_parts = []
        fallback_parts = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            content = _content_to_text(message.get("content", "")).strip()
            if not content:
                continue
            fallback_parts.append(content)
            if str(message.get("role", "")).lower() == "user":
                user_parts.append(content)
        return "\n".join(user_parts or fallback_parts).strip()
    return _content_to_text(prompt).strip()


def extract_answer(row: dict[str, Any]) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"]).strip()
    for key in ("answer", "ground_truth", "target"):
        if row.get(key) is not None:
            return str(row[key]).strip()
    return ""


def extract_solution(row: dict[str, Any]) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("solution"):
        return str(extra["solution"]).strip()
    for key in ("solution", "reference_solution", "rationale"):
        if row.get(key):
            return str(row[key]).strip()
    return ""


def extract_source(row: dict[str, Any]) -> str:
    return str(row.get("source", row.get("data_source", "unknown"))).strip().lower()


def _id_component(value: Any, *, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")
    return (slug[:48] or fallback) + "-" + stable_hash(text, 8)


def extract_query_id(row: dict[str, Any], problem: str, index: int) -> str:
    """Return a stable, dataset-namespaced identifier for a benchmark item.

    Verl datasets commonly restart ``extra_info.index`` at zero for every
    source.  A bare index therefore is not a globally unique query id in a
    concatenated AMC/AIME parquet.  Include both source and problem content so
    proposal maps and adapter caches cannot silently cross-contaminate items.
    """
    extra = row.get("extra_info")
    candidates = [row.get("query_id"), row.get("id"), row.get("index")]
    if isinstance(extra, dict):
        candidates.extend([extra.get("query_id"), extra.get("index"), extra.get("id")])
    raw_id: Any = None
    for value in candidates:
        if value is not None and str(value).strip():
            raw_id = value
            break
    source = _id_component(extract_source(row), fallback="unknown")
    item = _id_component(raw_id, fallback=f"q{index:06d}")
    return f"{source}:{item}:{stable_hash(problem, 16)}"


def iter_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                yield value
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            value = value.get("data", value.get("records", [value]))
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON list in {path}")
        for row in value:
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON objects in {path}")
            yield row
        return

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Reading parquet requires pyarrow (pip install pyarrow).") from exc
        table = pq.read_table(path)
        for row in table.to_pylist():
            yield row
        return

    raise ValueError(f"Unsupported input format {suffix!r}; use .jsonl, .json, or .parquet")


def load_query_records(
    path: str | Path,
    *,
    include_targets: bool,
    max_samples: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Normalize dataset rows.

    Candidate proposal calls this with ``include_targets=False``.  This is a
    deliberate information boundary: answers and solutions never enter the
    object passed to the proposer.
    """
    records = []
    seen_ids: dict[str, str] = {}
    for index, row in enumerate(iter_rows(path)):
        problem = extract_problem(row)
        if not problem:
            continue
        query_id = extract_query_id(row, problem, index)
        problem_sha256 = stable_hash(problem, 64)
        if query_id in seen_ids:
            raise ValueError(
                f"Duplicate canonical query_id {query_id!r} in {path}; "
                f"problem hashes {seen_ids[query_id]} and {problem_sha256}"
            )
        seen_ids[query_id] = problem_sha256
        record = {
            "query_id": query_id,
            "problem": problem,
            "problem_sha256": problem_sha256,
            "source": extract_source(row),
        }
        if include_targets:
            record["answer"] = extract_answer(row)
            record["solution"] = extract_solution(row)
        records.append(record)
        if max_samples is not None and len(records) >= max_samples:
            break
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_proposal_map(path: str | Path) -> dict[str, dict[str, Any]]:
    proposals = {}
    for row in iter_rows(path):
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError(f"Proposal row is missing query_id: {row.keys()}")
        if query_id in proposals:
            raise ValueError(f"Duplicate proposal query_id {query_id!r} in {path}")
        problem = str(row.get("problem", "")).strip()
        declared_hash = str(row.get("problem_sha256", "")).strip()
        if not problem or not declared_hash or stable_hash(problem, 64) != declared_hash:
            raise ValueError(
                f"Proposal {query_id!r} has a missing problem or invalid problem_sha256"
            )
        if not str(row.get("source", "")).strip():
            raise ValueError(f"Proposal {query_id!r} is missing its dataset source")
        validate_proposal_training_binding(row, context=f"Proposal in {path}")
        proposals[query_id] = row
    return proposals
