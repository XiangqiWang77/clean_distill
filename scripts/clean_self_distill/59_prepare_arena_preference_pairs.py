#!/usr/bin/env python3
"""Build a held-out LMArena human-preference pair set.

The emitted file is evaluation-only and contains ``(prompt, preferred,
rejected)`` triples derived from existing non-tie human votes.  Training and
audit prompts are loaded only to prove normalized-prompt disjointness.  Model
identities and the original A/B orientation are deliberately omitted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.clean_self_distill.arena_preference import (
    PAIR_SCHEMA_VERSION,
    ArenaPreferenceError,
    normalize_prompt,
    sha256_text,
    validate_preference_pair,
)


DATASET_ID = "lmarena-ai/arena-human-preference-100k"
PREPARATION_SCHEMA_VERSION = "arena-human-preference-preparation-v1"
DEFAULT_SEED = 20_260_818
DEFAULT_COUNT = 600


class PreferencePreparationError(ArenaPreferenceError):
    """Raised when the source cannot yield a valid held-out pair set."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(value: pa.Scalar | None) -> str | None:
    if value is None or not value.is_valid:
        return None
    item = value.as_py()
    return item if isinstance(item, str) else None


def _scalar_bool(value: pa.Scalar | None) -> bool:
    if value is None or not value.is_valid:
        return False
    item = value.as_py()
    return bool(item) if isinstance(item, bool) else False


def _struct_field(value: pa.StructScalar, name: str) -> pa.Scalar | None:
    fields = {field.name.casefold(): field.name for field in value.type}
    actual = fields.get(name.casefold())
    return value[actual] if actual is not None else None


def _conversation_turns(
    conversation: pa.Scalar, *, context: str
) -> tuple[str, str]:
    if conversation is None or not conversation.is_valid or not (
        pa.types.is_list(conversation.type)
        or pa.types.is_large_list(conversation.type)
        or pa.types.is_fixed_size_list(conversation.type)
    ):
        raise PreferencePreparationError(
            f"{context} must be a list of role/content structs"
        )
    users: list[str] = []
    assistants: list[str] = []
    for message in conversation.values:
        if message is None or not message.is_valid:
            continue
        if not pa.types.is_struct(message.type):
            raise PreferencePreparationError(
                f"{context} must be a list of role/content structs"
            )
        role = _scalar_text(_struct_field(message, "role"))
        content = _scalar_text(_struct_field(message, "content"))
        if role is None or content is None:
            continue
        normalized_role = role.strip().casefold()
        if normalized_role == "user":
            users.append(content.strip())
        elif normalized_role == "assistant":
            assistants.append(content.strip())
    if len(users) != 1 or len(assistants) != 1 or not users[0] or not assistants[0]:
        raise PreferencePreparationError(
            f"{context} must contain exactly one non-empty user and assistant turn"
        )
    return users[0], assistants[0]


def _is_english(value: pa.Scalar) -> bool:
    language = _scalar_text(value)
    if language is None:
        return False
    normalized = language.strip().casefold().replace(" ", "")
    return (
        normalized in {"en", "eng", "english"}
        or normalized.startswith("en-")
        or normalized.startswith("en_")
        or normalized.startswith("english(")
    )


def _is_single_turn(value: pa.Scalar) -> bool:
    if value is None or not value.is_valid:
        return False
    item = value.as_py()
    return (
        not isinstance(item, bool)
        and isinstance(item, (int, float))
        and float(item) == 1.0
    )


def _winner_side(value: pa.Scalar) -> str | None:
    winner = _scalar_text(value)
    if winner is None:
        return None
    normalized = winner.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"a", "model_a", "response_a", "assistant_a"}:
        return "a"
    if normalized in {"b", "model_b", "response_b", "assistant_b"}:
        return "b"
    return None


def _true_leaf_names(value: object, prefix: tuple[str, ...] = ()) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            clean = str(key).strip().casefold().replace(" ", "_")
            result.update(_true_leaf_names(child, (*prefix, clean)))
    elif value is True and prefix:
        result.add(prefix[-1])
    return result


def _domains(
    *, category: pa.Scalar | None, is_code: pa.Scalar | None
) -> list[str]:
    names: set[str] = set()
    if category is not None and category.is_valid:
        names.update(_true_leaf_names(category.as_py()))
    if _scalar_bool(is_code):
        names.add("code")
    aliases = {
        "if": "instruction_following",
        "real_world": "real_world",
        "domain_knowledge": "domain_knowledge",
        "problem_solving": "problem_solving",
        "technical_accuracy": "technical_accuracy",
    }
    normalized = {aliases.get(name, name) for name in names}
    # These are version/container names rather than semantic slice labels.
    normalized.difference_update({"criteria_v0.1", "if_v0.1", "math_v0.1"})
    if not normalized:
        normalized.add("other")
    return sorted(normalized)


def _casefolded_names(schema: pa.Schema, *, path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in schema.names:
        key = name.casefold()
        if key in result:
            raise PreferencePreparationError(
                f"{path} has ambiguous columns differing only by case"
            )
        result[key] = name
    return result


def _projected_columns(parquet: pq.ParquetFile, *, path: Path) -> dict[str, str]:
    names = _casefolded_names(parquet.schema_arrow, path=path)
    required: dict[str, str] = {}
    for key in ("conversation_a", "conversation_b", "winner", "language", "turn"):
        actual = names.get(key)
        if actual is None:
            raise PreferencePreparationError(f"{path} lacks required column {key}")
        required[key] = actual
    for key in ("question_id", "category_tag", "is_code", "is_refusal"):
        if key in names:
            required[key] = names[key]
    return required


def _column(batch: pa.RecordBatch, name: str) -> pa.Array:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise PreferencePreparationError(f"projected column {name!r} is missing")
    return batch.column(index)


def _load_excluded_hashes(paths: Sequence[Path]) -> tuple[set[str], dict[str, Any]]:
    hashes: set[str] = set()
    audit: dict[str, Any] = {}
    for path in paths:
        count = 0
        digest = hashlib.sha256()
        try:
            handle = path.open("rb")
        except OSError as exc:
            raise PreferencePreparationError(f"cannot read {path}: {exc}") from exc
        with handle:
            for line_number, raw_line in enumerate(handle, 1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise PreferencePreparationError(
                        f"{path}:{line_number} contains invalid JSON"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise PreferencePreparationError(
                        f"{path}:{line_number} must contain an object"
                    )
                problem = row.get("problem")
                if not isinstance(problem, str) or not problem.strip():
                    raise PreferencePreparationError(
                        f"{path}:{line_number} lacks a non-empty problem"
                    )
                expected = sha256_text(problem.strip())
                if row.get("problem_sha256") != expected:
                    raise PreferencePreparationError(
                        f"{path}:{line_number} problem_sha256 mismatch"
                    )
                hashes.add(sha256_text(normalize_prompt(problem)))
                count += 1
        audit[path.name] = {
            "path": str(path.resolve()),
            "rows": count,
            "sha256": digest.hexdigest(),
        }
    if not hashes:
        raise PreferencePreparationError("excluded query files contain no prompts")
    return hashes, audit


def _rank(seed: int, normalized_hash: str) -> str:
    value = f"arena-human-preference-eval-v1\0{seed}\0{normalized_hash}"
    return sha256_text(value)


def _canonical_pair_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["prompt_sha256"]),
        str(record["preferred_response_sha256"]),
        str(record["rejected_response_sha256"]),
    )


def _heap_item(record: Mapping[str, Any]) -> tuple[int, int, str]:
    normalized_hash = str(record["normalized_prompt_sha256"])
    return (
        -int(str(record["rank_sha256"]), 16),
        -int(normalized_hash, 16),
        normalized_hash,
    )


def _rank_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(record["rank_sha256"]),
        str(record["normalized_prompt_sha256"]),
    )


def _bounded_add(
    record: dict[str, Any],
    *,
    count: int,
    selected: dict[str, dict[str, Any]],
    heap: list[tuple[int, int, str]],
) -> None:
    normalized_hash = str(record["normalized_prompt_sha256"])
    if len(selected) < count:
        selected[normalized_hash] = record
        heapq.heappush(heap, _heap_item(record))
        return
    worst_hash = heap[0][2]
    if _rank_key(record) >= _rank_key(selected[worst_hash]):
        return
    heapq.heappop(heap)
    del selected[worst_hash]
    selected[normalized_hash] = record
    heapq.heappush(heap, _heap_item(record))


def select_pairs(
    parquet_path: Path,
    *,
    excluded_normalized_hashes: set[str],
    count: int,
    seed: int,
    min_prompt_chars: int,
    max_prompt_chars: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    if count <= 0 or batch_size <= 0:
        raise PreferencePreparationError("count and batch_size must be positive")
    if min_prompt_chars <= 0 or max_prompt_chars < min_prompt_chars:
        raise PreferencePreparationError("invalid prompt character bounds")
    if not parquet_path.is_file():
        raise PreferencePreparationError(f"Parquet file does not exist: {parquet_path}")

    parquet = pq.ParquetFile(parquet_path)
    projected = _projected_columns(parquet, path=parquet_path)
    columns = list(projected.values())
    arrays: dict[str, pa.Array]
    selected: dict[str, dict[str, Any]] = {}
    heap: list[tuple[int, int, str]] = []
    seen_normalized: set[str] = set()
    stats: dict[str, int] = {
        "raw_rows_streamed": 0,
        "non_english_removed": 0,
        "non_single_turn_removed": 0,
        "tie_or_unknown_winner_removed": 0,
        "invalid_conversation_removed": 0,
        "mismatched_pair_prompt_removed": 0,
        "empty_or_equal_response_removed": 0,
        "prompt_bounds_removed": 0,
        "train_audit_overlap_removed": 0,
        "normalized_duplicates_removed": 0,
        "eligible_unique_pairs": 0,
        "selected_pairs": 0,
    }

    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=columns, use_threads=False
    ):
        arrays = {
            key: _column(batch, actual) for key, actual in projected.items()
        }
        for index in range(batch.num_rows):
            stats["raw_rows_streamed"] += 1
            if not _is_english(arrays["language"][index]):
                stats["non_english_removed"] += 1
                continue
            if not _is_single_turn(arrays["turn"][index]):
                stats["non_single_turn_removed"] += 1
                continue
            side = _winner_side(arrays["winner"][index])
            if side is None:
                stats["tie_or_unknown_winner_removed"] += 1
                continue
            try:
                prompt_a, response_a = _conversation_turns(
                    arrays["conversation_a"][index], context="conversation_a"
                )
                prompt_b, response_b = _conversation_turns(
                    arrays["conversation_b"][index], context="conversation_b"
                )
            except PreferencePreparationError:
                stats["invalid_conversation_removed"] += 1
                continue
            normalized_prompt = normalize_prompt(prompt_a)
            if not normalized_prompt or normalized_prompt != normalize_prompt(prompt_b):
                stats["mismatched_pair_prompt_removed"] += 1
                continue
            prompt = prompt_a.strip()
            if not min_prompt_chars <= len(prompt) <= max_prompt_chars:
                stats["prompt_bounds_removed"] += 1
                continue
            preferred = response_a.strip() if side == "a" else response_b.strip()
            rejected = response_b.strip() if side == "a" else response_a.strip()
            if not preferred or not rejected or preferred == rejected:
                stats["empty_or_equal_response_removed"] += 1
                continue
            normalized_hash = sha256_text(normalized_prompt)
            if normalized_hash in excluded_normalized_hashes:
                stats["train_audit_overlap_removed"] += 1
                continue

            category_array = arrays.get("category_tag")
            code_array = arrays.get("is_code")
            record: dict[str, Any] = {
                "rank_sha256": _rank(seed, normalized_hash),
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "normalized_prompt_sha256": normalized_hash,
                "preferred_response": preferred,
                "preferred_response_sha256": sha256_text(preferred),
                "rejected_response": rejected,
                "rejected_response_sha256": sha256_text(rejected),
                "domains": _domains(
                    category=(category_array[index] if category_array is not None else None),
                    is_code=(code_array[index] if code_array is not None else None),
                ),
            }
            if (
                record["preferred_response_sha256"]
                == record["rejected_response_sha256"]
            ):
                stats["empty_or_equal_response_removed"] += 1
                continue
            if normalized_hash in seen_normalized:
                stats["normalized_duplicates_removed"] += 1
                previous = selected.get(normalized_hash)
                if previous is not None and _canonical_pair_key(
                    record
                ) < _canonical_pair_key(previous):
                    selected[normalized_hash] = record
                continue
            seen_normalized.add(normalized_hash)
            _bounded_add(record, count=count, selected=selected, heap=heap)
        del arrays, batch
        gc.collect()

    stats["eligible_unique_pairs"] = len(seen_normalized)
    if len(selected) < count:
        raise PreferencePreparationError(
            f"need {count} held-out preference pairs, found {len(selected)}"
        )
    ordered = sorted(selected.values(), key=_rank_key)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(ordered):
        row = {
            "schema_version": PAIR_SCHEMA_VERSION,
            "evaluation_only": True,
            "label_source": "lmarena_human_vote",
            "external_judge_used": False,
            "query_id": f"arena-pref:heldout:{index:06d}",
            "prompt": record["prompt"],
            "prompt_sha256": record["prompt_sha256"],
            "normalized_prompt_sha256": record["normalized_prompt_sha256"],
            "preferred_response": record["preferred_response"],
            "preferred_response_sha256": record["preferred_response_sha256"],
            "rejected_response": record["rejected_response"],
            "rejected_response_sha256": record["rejected_response_sha256"],
            "domains": record["domains"],
            "source": "lmarena_arena_human_preference_100k",
        }
        rows.append(validate_preference_pair(row, row_number=index + 1))
    stats["selected_pairs"] = len(rows)
    return rows, stats, columns


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    destination = args.output_dir.resolve()
    if os.path.lexists(destination):
        raise PreferencePreparationError(
            f"refusing to overwrite existing output directory: {destination}"
        )
    excluded, exclude_audit = _load_excluded_hashes(
        [args.train_queries.resolve(), args.audit_queries.resolve()]
    )
    rows, stats, projected_columns = select_pairs(
        args.parquet.resolve(),
        excluded_normalized_hashes=excluded,
        count=args.count,
        seed=args.seed,
        min_prompt_chars=args.min_prompt_chars,
        max_prompt_chars=args.max_prompt_chars,
        batch_size=args.batch_size,
    )
    if {row["normalized_prompt_sha256"] for row in rows} & excluded:
        raise PreferencePreparationError("held-out pairs overlap train/audit prompts")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        pair_path = staging / "heldout_preference_pairs.jsonl"
        _atomic_text(
            pair_path,
            "".join(_canonical_json(row) + "\n" for row in rows),
        )
        domains: dict[str, int] = {}
        for row in rows:
            for domain in row["domains"]:
                domains[domain] = domains.get(domain, 0) + 1
        manifest = {
            "schema_version": PREPARATION_SCHEMA_VERSION,
            "dataset": {
                "id": DATASET_ID,
                "revision": args.revision,
                "parquet_path": str(args.parquet.resolve()),
                "parquet_bytes": args.parquet.stat().st_size,
                "parquet_sha256": _file_sha256(args.parquet.resolve()),
                "projected_columns": projected_columns,
            },
            "selection": {
                "seed": args.seed,
                "count": args.count,
                "rank": (
                    "sha256('arena-human-preference-eval-v1\\0' + seed + "
                    "'\\0' + normalized_prompt_sha256)"
                ),
                "strict_human_winners_only": True,
                "ties_removed": True,
                "single_turn_only": True,
                "english_only": True,
                "prompt_character_bounds": [
                    args.min_prompt_chars,
                    args.max_prompt_chars,
                ],
                "normalized_prompt_deduplication": True,
            },
            "leakage_audit": {
                "excluded_query_files": exclude_audit,
                "excluded_normalized_prompt_count": len(excluded),
                "heldout_overlap_count": 0,
            },
            "label_handling": {
                "source": "existing LMArena human vote",
                "orientation": "winner response -> y+, loser response -> y-",
                "model_ids_emitted": False,
                "original_a_b_orientation_emitted": False,
                "external_llm_judge_used": False,
                "bradley_terry_used": False,
            },
            "stream_audit": stats,
            "domain_counts_multilabel": dict(sorted(domains.items())),
            "artifact": {
                "path": pair_path.name,
                "rows": len(rows),
                "bytes": pair_path.stat().st_size,
                "sha256": _file_sha256(pair_path),
            },
        }
        _atomic_text(
            staging / "MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if os.path.lexists(destination):
            raise PreferencePreparationError(
                f"output appeared during preparation: {destination}"
            )
        os.rename(staging, destination)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--parquet", type=Path, required=True)
    root.add_argument("--train-queries", type=Path, required=True)
    root.add_argument("--audit-queries", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--revision", required=True)
    root.add_argument("--count", type=int, default=DEFAULT_COUNT)
    root.add_argument("--seed", type=int, default=DEFAULT_SEED)
    root.add_argument("--min-prompt-chars", type=int, default=8)
    root.add_argument("--max-prompt-chars", type=int, default=16_384)
    root.add_argument("--batch-size", type=int, default=128)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest = prepare(args)
    print(
        _canonical_json(
            {
                "output_dir": str(args.output_dir),
                "pairs": manifest["artifact"]["rows"],
                "external_llm_judge_used": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
