"""Input/output helpers for JSONL and verl-compatible parquet datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:length]


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


def extract_query_id(row: dict[str, Any], problem: str, index: int) -> str:
    extra = row.get("extra_info")
    candidates = [row.get("query_id"), row.get("id"), row.get("index")]
    if isinstance(extra, dict):
        candidates.extend([extra.get("query_id"), extra.get("index"), extra.get("id")])
    for value in candidates:
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"q{index:06d}-{stable_hash(problem)}"


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
    for index, row in enumerate(iter_rows(path)):
        problem = extract_problem(row)
        if not problem:
            continue
        record = {
            "query_id": extract_query_id(row, problem, index),
            "problem": problem,
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
        proposals[query_id] = row
    return proposals
