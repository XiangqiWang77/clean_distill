"""Label-isolated held-out generation and scoring for persistent students.

The GPU-facing generator accepts only a physically target-free query manifest.
Answers are joined later by the CPU scorer, after every response has already
been committed.  This keeps AMC23/AIME24/AIME25 labels out of model memory
during checkpoint evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import string
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.opsd_format import extract_boxed_answer, grade_boxed_answer

from .io import iter_rows, stable_hash


FORBIDDEN_QUERY_KEYS = frozenset(
    {
        "answer",
        "final_answer",
        "ground_truth",
        "label",
        "reference",
        "reference_answer",
        "reference_solution",
        "reward_model",
        "solution",
        "target",
        "feedback",
    }
)
EVAL_SAMPLE_COUNT = 4
EVAL_TEMPERATURE = 0.6
EVAL_TOP_P = 0.95
EVAL_TOP_K = 20


class HeldoutProtocolError(ValueError):
    """Raised when a held-out artifact violates the preregistered protocol."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).strip().casefold()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def validate_query_only_row(row: Mapping[str, Any], *, context: str) -> dict[str, str]:
    exposed = sorted(set(_walk_keys(row)) & FORBIDDEN_QUERY_KEYS)
    if exposed:
        raise HeldoutProtocolError(
            f"{context} physically exposes held-out fields {exposed}"
        )
    required = ("query_id", "problem", "problem_sha256", "source")
    normalized = {key: str(row.get(key, "")).strip() for key in required}
    if any(not normalized[key] for key in required):
        raise HeldoutProtocolError(f"{context} is missing one of {required}")
    expected = stable_hash(normalized["problem"], 64)
    if normalized["problem_sha256"].casefold() != expected:
        raise HeldoutProtocolError(f"{context}.problem_sha256 does not match problem")
    normalized["problem_sha256"] = expected
    normalized["source"] = normalized["source"].casefold()
    return normalized


def load_query_only_manifest(path: str | Path) -> list[dict[str, str]]:
    rows = [
        validate_query_only_row(row, context=f"{path} row {index}")
        for index, row in enumerate(iter_rows(path), 1)
    ]
    if not rows:
        raise HeldoutProtocolError(f"{path} is empty")
    ids = [row["query_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise HeldoutProtocolError(f"{path} has duplicate query_id values")
    return rows


def paired_sample_seed(
    base_seed: int,
    global_query_index: int,
    sample_index: int,
    *,
    sample_count: int = EVAL_SAMPLE_COUNT,
) -> int:
    if (
        global_query_index < 0
        or sample_count <= 0
        or not 0 <= sample_index < sample_count
    ):
        raise HeldoutProtocolError("Invalid query/sample index for paired seed")
    return int(base_seed) + global_query_index * 1009 + sample_index


def expected_prediction_keys(
    queries: Sequence[Mapping[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
    sample_count: int = EVAL_SAMPLE_COUNT,
) -> list[tuple[str, int]]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise HeldoutProtocolError("Require num_shards > 0 and a valid shard_index")
    if sample_count <= 0:
        raise HeldoutProtocolError("sample_count must be positive")
    return [
        (str(query["query_id"]), sample_index)
        for global_index, query in enumerate(queries)
        if global_index % num_shards == shard_index
        for sample_index in range(sample_count)
    ]


def load_resumable_predictions(
    path: str | Path,
    expected_keys: Sequence[tuple[str, int]],
    *,
    method: str,
    checkpoint_episode: int,
    checkpoint_sha256: str,
    generation_config_sha256: str | None = None,
) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows = [dict(row) for row in iter_rows(target)]
    if len(rows) > len(expected_keys):
        raise HeldoutProtocolError(f"{path} contains more rows than expected")
    for index, row in enumerate(rows):
        actual_key = (str(row.get("query_id", "")), row.get("sample_index"))
        if actual_key != expected_keys[index]:
            raise HeldoutProtocolError(
                f"{path} row {index + 1} is not the exact expected prefix"
            )
        if (
            row.get("method") != method
            or row.get("checkpoint_episode") != checkpoint_episode
            or row.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise HeldoutProtocolError(
                f"{path} row {index + 1} disagrees with checkpoint identity"
            )
        if (
            generation_config_sha256 is not None
            and row.get("generation_config_sha256") != generation_config_sha256
        ):
            raise HeldoutProtocolError(
                f"{path} row {index + 1} belongs to another generation configuration"
            )
        if "correct" in row or set(_walk_keys(row)) & FORBIDDEN_QUERY_KEYS:
            raise HeldoutProtocolError(
                f"{path} row {index + 1} contains a label before offline scoring"
            )
    return rows


def tree_sha256(path: str | Path) -> str:
    root = Path(path)
    if not root.exists():
        return "base"
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise HeldoutProtocolError(f"Checkpoint directory {root} has no files")
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_sealed_labels(path: str | Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for index, row in enumerate(iter_rows(path), 1):
        query_id = str(row.get("query_id", "")).strip()
        answer = str(row.get("answer", "")).strip()
        problem_sha256 = str(row.get("problem_sha256", "")).strip().casefold()
        if (
            not query_id
            or not answer
            or len(problem_sha256) != 64
            or any(character not in string.hexdigits for character in problem_sha256)
        ):
            raise HeldoutProtocolError(f"{path} row {index} is not a sealed label row")
        if query_id in labels:
            raise HeldoutProtocolError(f"{path} duplicates label {query_id!r}")
        labels[query_id] = {
            "answer": answer,
            "problem_sha256": problem_sha256,
        }
    if not labels:
        raise HeldoutProtocolError(f"{path} has no labels")
    return labels


def score_prediction_rows(
    predictions: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, str]],
    *,
    sample_count: int = EVAL_SAMPLE_COUNT,
) -> list[dict[str, Any]]:
    if sample_count <= 0:
        raise HeldoutProtocolError("sample_count must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(predictions, 1):
        if "correct" in row or set(_walk_keys(row)) & FORBIDDEN_QUERY_KEYS:
            raise HeldoutProtocolError(
                f"Prediction row {index} contains a target label before offline scoring"
            )
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise HeldoutProtocolError(f"Prediction row {index} has no query_id")
        grouped.setdefault(query_id, []).append(row)
    if set(grouped) != set(labels):
        missing = sorted(set(labels) - set(grouped))
        extra = sorted(set(grouped) - set(labels))
        raise HeldoutProtocolError(
            f"Prediction/label coverage mismatch missing={missing[:5]} extra={extra[:5]}"
        )

    scored: list[dict[str, Any]] = []
    for query_id in sorted(grouped):
        rows = sorted(grouped[query_id], key=lambda row: int(row["sample_index"]))
        indices = [int(row["sample_index"]) for row in rows]
        if indices != list(range(sample_count)):
            raise HeldoutProtocolError(
                f"{query_id} needs exactly sample indices 0..{sample_count - 1}"
            )
        label = labels[query_id]
        if any(row.get("problem_sha256") != label["problem_sha256"] for row in rows):
            raise HeldoutProtocolError(f"{query_id} problem hash disagrees with labels")
        identities = {
            (
                row.get("method"),
                row.get("checkpoint_episode"),
                row.get("checkpoint_sha256"),
                row.get("query_manifest_sha256"),
                row.get("temperature"),
                row.get("top_p"),
                row.get("top_k"),
                row.get("max_new_tokens"),
            )
            for row in rows
        }
        if len(identities) != 1:
            raise HeldoutProtocolError(
                f"{query_id} mixes method/checkpoint/decoding identities across samples"
            )
        for row in rows:
            parsed = extract_boxed_answer(str(row.get("response", "")))
            correct = float(grade_boxed_answer(parsed, label["answer"]))
            common = {
                key: row[key]
                for key in (
                    "method",
                    "checkpoint_episode",
                    "checkpoint_sha256",
                    "query_id",
                    "problem_sha256",
                    "source",
                    "sample_index",
                    "seed",
                    "generated_tokens",
                    "truncated",
                    "response",
                    "training_audit",
                    "temperature",
                    "top_p",
                    "top_k",
                    "max_new_tokens",
                    "prompt_tokens",
                    "query_manifest_sha256",
                    "generation_config_sha256",
                    "specialization_metrics",
                    "proposal_end_to_end_seconds",
                    "adaptation_seconds",
                    "distillation_trace",
                    "trajectory_metrics",
                    "behavioral_diagnostics",
                    "resource_usage",
                    "runtime",
                )
                if key in row
            }
            common.update(parsed_answer=str(parsed or ""), correct=correct)
            if sample_count > 1:
                scored.append({**common, "profile": f"mean{sample_count}"})
            if int(row["sample_index"]) == 0:
                scored.append({**common, "profile": "acc1"})
    return scored


def write_scored_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_jsonl(Path(path), rows)


def query_manifest_sha256(queries: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(list(queries)).encode("utf-8")).hexdigest()


def generation_config_sha256(payload: Mapping[str, Any]) -> str:
    """Bind restartable prediction shards to their exact scientific settings."""
    return hashlib.sha256(_canonical_json(dict(payload)).encode("utf-8")).hexdigest()
