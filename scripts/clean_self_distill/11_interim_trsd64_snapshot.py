#!/usr/bin/env python3
"""Grade a coherent, explicitly provisional snapshot of an active TRSD-64 eval.

This reporter is intentionally CPU-only.  It first snapshots each prediction
shard without opening the sealed label file.  Only after all snapshots are on
disk does it load the labels, grade the partial TRSD predictions, and restrict
the already-scored reference methods to exactly the same query IDs.

The resulting numbers are *not* a benchmark estimate: generation completion
order is correlated with response length, dataset order, and difficulty.  The
report therefore labels every result as provisional and completion-order
selected.  Its only legitimate use is an interim paired diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.clean_self_distill.heldout import (
    extract_boxed_answer,
    grade_boxed_answer,
    load_sealed_labels,
)


class InterimSnapshotError(RuntimeError):
    """Raised when an interim snapshot is not internally coherent."""


METHOD_ORDER = ("base", "privileged_ep16", "privileged_ep64", "trsd_ep64")
METHOD_LABELS = {
    "base": "Base",
    "privileged_ep16": "Privileged-SD 16",
    "privileged_ep64": "Privileged-SD 64",
    "trsd_ep64": "TRSD 64 (partial)",
}
SOURCE_ORDER = ("combined", "amc23", "aime24", "aime25")
PROVISIONAL_STATUS = "PROVISIONAL_COMPLETION_ORDER_SELECTED_DO_NOT_GENERALIZE"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise InterimSnapshotError(f"{path}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise InterimSnapshotError(f"{path} has no rows")
    return rows


def snapshot_jsonl(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one whole immutable file generation through an open descriptor.

    Evaluation rewrites each shard via temporary-file + rename.  Opening the
    shard pins either the complete old inode or the complete new inode, even if
    another rename occurs while it is read.  Requiring a trailing newline and
    parsing every line rules out a torn JSON record.
    """

    captured_at = datetime.now(timezone.utc).isoformat()
    with source.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise InterimSnapshotError(f"{source} changed through its open descriptor")
    if len(payload) != before.st_size:
        raise InterimSnapshotError(
            f"Short read for {source}: got {len(payload)} of {before.st_size} bytes"
        )
    if not payload.endswith(b"\n"):
        raise InterimSnapshotError(f"{source} snapshot lacks a final newline")
    try:
        text = payload.decode("utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InterimSnapshotError(f"{source} is not a complete UTF-8 JSONL file") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise InterimSnapshotError(f"{source} snapshot has no object rows")
    atomic_write_text(destination, text)
    return {
        "source": str(source),
        "snapshot": str(destination),
        "captured_at_utc": captured_at,
        "device": before.st_dev,
        "inode": before.st_ino,
        "bytes": len(payload),
        "rows": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "coherence": "single_open_inode_complete_jsonl",
    }


def keyed(rows: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("profile", "acc1")) != "acc1":
            continue
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise InterimSnapshotError(f"{name} contains a row without query_id")
        if query_id in output:
            raise InterimSnapshotError(f"{name} duplicates {query_id}")
        output[query_id] = dict(row)
    if not output:
        raise InterimSnapshotError(f"{name} has no Acc@1 rows")
    return output


def strict_correct(row: Mapping[str, Any]) -> int:
    return int(float(row.get("correct", 0.0)) == 1.0 and not bool(row["truncated"]))


def grade_partial(
    predictions: Sequence[Mapping[str, Any]], labels_path: Path
) -> list[dict[str, Any]]:
    # Deliberately called only after every prediction shard has been snapshotted.
    labels = load_sealed_labels(labels_path)
    seen: set[str] = set()
    scored: list[dict[str, Any]] = []
    identity: tuple[Any, ...] | None = None
    for prediction in predictions:
        query_id = str(prediction.get("query_id", "")).strip()
        if not query_id or query_id in seen:
            raise InterimSnapshotError(f"Missing or duplicate partial query_id {query_id!r}")
        seen.add(query_id)
        if query_id not in labels:
            raise InterimSnapshotError(f"Partial query {query_id} has no sealed label")
        if "correct" in prediction:
            raise InterimSnapshotError("Prediction snapshot unexpectedly contains correctness")
        if int(prediction.get("sample_index", -1)) != 0:
            raise InterimSnapshotError(f"{query_id} is not the Acc@1 sample")
        if str(prediction.get("method")) != "trsd" or int(
            prediction.get("checkpoint_episode", -1)
        ) != 64:
            raise InterimSnapshotError(f"{query_id} is not a TRSD-64 prediction")
        label = labels[query_id]
        if str(prediction.get("problem_sha256")) != label["problem_sha256"]:
            raise InterimSnapshotError(f"{query_id} problem hash differs from sealed label")
        current_identity = (
            prediction.get("checkpoint_sha256"),
            prediction.get("query_manifest_sha256"),
            prediction.get("max_new_tokens"),
            prediction.get("temperature"),
            prediction.get("top_p"),
            prediction.get("top_k"),
            prediction.get("evaluation_prompt_version"),
        )
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise InterimSnapshotError("TRSD-64 snapshot mixes evaluation identities")
        parsed = extract_boxed_answer(str(prediction.get("response", "")))
        row = dict(prediction)
        row.update(
            parsed_answer=str(parsed or ""),
            correct=float(grade_boxed_answer(parsed, label["answer"])),
            profile="acc1",
        )
        scored.append(row)
    return sorted(scored, key=lambda row: str(row["query_id"]))


def validate_reference_match(
    reference: Mapping[str, Mapping[str, Any]],
    trsd: Mapping[str, Mapping[str, Any]],
    *,
    name: str,
) -> list[dict[str, Any]]:
    missing = sorted(set(trsd) - set(reference))
    if missing:
        raise InterimSnapshotError(f"{name} lacks partial IDs: {missing[:3]}")
    rows: list[dict[str, Any]] = []
    for query_id in sorted(trsd):
        base_row = reference[query_id]
        trsd_row = trsd[query_id]
        for field in ("problem_sha256", "source", "sample_index", "max_new_tokens"):
            if str(base_row.get(field)) != str(trsd_row.get(field)):
                raise InterimSnapshotError(f"{name}/{query_id}: {field} is not matched")
        rows.append(dict(base_row))
    return rows


def aggregate(method: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    groups: dict[str, Sequence[Mapping[str, Any]]] = {"combined": rows, **by_source}
    output: list[dict[str, Any]] = []
    for source in SOURCE_ORDER:
        values = list(groups.get(source, []))
        completed = [row for row in values if not bool(row["truncated"])]
        strict_n = sum(strict_correct(row) for row in values)
        completed_correct = sum(int(float(row.get("correct", 0.0)) == 1.0) for row in completed)
        output.append(
            {
                "status": PROVISIONAL_STATUS,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "source": source,
                "n": len(values),
                "strict_correct": strict_n,
                "strict_percent": 100.0 * strict_n / len(values) if values else None,
                "completed_n": len(completed),
                "completed_correct": completed_correct,
                "completed_only_percent": (
                    100.0 * completed_correct / len(completed) if completed else None
                ),
                "truncated_n": len(values) - len(completed),
            }
        )
    return output


def paired_effects(
    base: Mapping[str, Mapping[str, Any]],
    method: Mapping[str, Mapping[str, Any]],
    method_name: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in SOURCE_ORDER:
        ids = [
            query_id
            for query_id, row in base.items()
            if source == "combined" or str(row["source"]) == source
        ]
        wrong_to_correct = sum(
            not strict_correct(base[query_id]) and strict_correct(method[query_id])
            for query_id in ids
        )
        correct_to_wrong = sum(
            strict_correct(base[query_id]) and not strict_correct(method[query_id])
            for query_id in ids
        )
        base_correct = sum(strict_correct(base[query_id]) for query_id in ids)
        method_correct = sum(strict_correct(method[query_id]) for query_id in ids)
        output.append(
            {
                "status": PROVISIONAL_STATUS,
                "method": method_name,
                "method_label": METHOD_LABELS[method_name],
                "source": source,
                "paired_n": len(ids),
                "base_strict_correct": base_correct,
                "method_strict_correct": method_correct,
                "paired_delta_percentage_points": (
                    100.0 * (method_correct - base_correct) / len(ids) if ids else None
                ),
                "wrong_to_correct": wrong_to_correct,
                "correct_to_wrong": correct_to_wrong,
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise InterimSnapshotError(f"Cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def format_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}%"


def build_markdown(
    metric_rows: Sequence[Mapping[str, Any]],
    effect_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> str:
    lines = [
        "# PROVISIONAL TRSD-64 completion-order snapshot",
        "",
        "> **DO NOT treat these values as final benchmark performance.** The subset consists",
        "> only of queries that finished generation before the snapshot. Completion order is",
        "> correlated with response length, source ordering, truncation, and likely difficulty.",
        "> Paired filtering removes query-set mismatch between methods, but it does not remove",
        "> this completion-order selection bias.",
        "",
        f"Captured at `{manifest['captured_at_utc']}` with **{manifest['partial_query_count']}/143** TRSD-64 queries.",
        "Strict accuracy counts a response correct only when its boxed answer is correct **and**",
        "the response ended before the 10,240-token cap. Completed-only accuracy excludes truncations",
        "and is therefore an explicitly secondary, selection-biased diagnostic.",
        "",
        "## Strict and completed-only snapshot",
        "",
        "| Method | Source | n | Strict correct | Strict Acc@1 | Completed | Completed correct | Completed-only | Truncated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['method_label']} | {row['source']} | {row['n']} | "
            f"{row['strict_correct']} | {format_percent(row['strict_percent'])} | "
            f"{row['completed_n']} | {row['completed_correct']} | "
            f"{format_percent(row['completed_only_percent'])} | {row['truncated_n']} |"
        )
    lines.extend(
        [
            "",
            "## Paired effects versus Base on exactly the same finished TRSD IDs",
            "",
            "| Method | Source | Paired n | Delta (pp) | Wrong→correct | Correct→wrong |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in effect_rows:
        delta = row["paired_delta_percentage_points"]
        lines.append(
            f"| {row['method_label']} | {row['source']} | {row['paired_n']} | "
            f"{'N/A' if delta is None else f'{float(delta):+.2f}'} | "
            f"{row['wrong_to_correct']} | {row['correct_to_wrong']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This snapshot can answer only whether TRSD-64 is ahead or behind Base on the queries",
            "that happened to finish first. It cannot establish the final 143-query delta, statistical",
            "significance, or long-horizon superiority. Those require the already-submitted complete",
            "evaluation and its final sealed-label scoring job.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_reference(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--reference must be METHOD=PATH")
    name, raw_path = value.split("=", 1)
    if name not in METHOD_ORDER[:-1]:
        raise argparse.ArgumentTypeError(f"Unsupported reference method {name!r}")
    return name, Path(raw_path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trsd-prediction", action="append", type=Path, required=True)
    result.add_argument("--labels", type=Path, required=True)
    result.add_argument("--reference", action="append", type=parse_reference, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if len(args.trsd_prediction) != 4:
        raise InterimSnapshotError("Exactly four TRSD prediction shards are required")
    references = dict(args.reference)
    if set(references) != set(METHOD_ORDER[:-1]):
        raise InterimSnapshotError("References must contain Base, Privileged-SD 16, and 64")

    root = args.output_dir
    snapshot_root = root / "prediction_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    shard_metadata: list[dict[str, Any]] = []
    snapshot_paths: list[Path] = []
    for index, source in enumerate(args.trsd_prediction):
        destination = snapshot_root / f"shard_{index:02d}.jsonl"
        shard_metadata.append(snapshot_jsonl(source, destination))
        snapshot_paths.append(destination)

    # The data firewall opens sealed labels only after all four snapshots exist.
    shard_rows = [read_jsonl(path) for path in snapshot_paths]
    for path, rows in zip(snapshot_paths, shard_rows):
        digests = {str(row.get("generation_config_sha256", "")) for row in rows}
        if len(digests) != 1 or "" in digests:
            raise InterimSnapshotError(f"{path} mixes generation configurations")
    # The generation digest intentionally differs by shard because it binds the
    # shard index. All scientific settings checked in grade_partial are global.
    predictions = [row for rows in shard_rows for row in rows]
    trsd_scored = grade_partial(predictions, args.labels)
    atomic_write_jsonl(root / "trsd_ep64_partial_scored.jsonl", trsd_scored)
    trsd_by_id = keyed(trsd_scored, name="trsd_ep64")

    matched: dict[str, list[dict[str, Any]]] = {"trsd_ep64": trsd_scored}
    for name, path in references.items():
        reference_rows = keyed(read_jsonl(path), name=name)
        matched[name] = validate_reference_match(reference_rows, trsd_by_id, name=name)
    matched_by_id = {name: keyed(rows, name=name) for name, rows in matched.items()}

    metric_rows = [
        row
        for name in METHOD_ORDER
        for row in aggregate(name, matched[name])
    ]
    effect_rows = [
        row
        for name in METHOD_ORDER[1:]
        for row in paired_effects(matched_by_id["base"], matched_by_id[name], name)
    ]
    captured_at = datetime.now(timezone.utc).isoformat()
    source_counts: dict[str, int] = defaultdict(int)
    for row in trsd_scored:
        source_counts[str(row["source"])] += 1
    manifest = {
        "schema_version": "trsd64-interim-completion-snapshot-v1",
        "status": PROVISIONAL_STATUS,
        "captured_at_utc": captured_at,
        "partial_query_count": len(trsd_scored),
        "expected_final_query_count": 143,
        "source_counts": dict(sorted(source_counts.items())),
        "selection_warning": (
            "Completion-order subset; correlated with length/source/difficulty. "
            "Not a final benchmark estimator."
        ),
        "strict_definition": "boxed_answer_correct AND truncated=false",
        "completed_only_definition": "strict correct / count(truncated=false)",
        "label_access_order": "all_prediction_shards_snapshotted_before_sealed_labels_opened",
        "shards": shard_metadata,
        "reference_files": {name: str(path) for name, path in references.items()},
    }
    write_csv(root / "interim_metrics.csv", metric_rows)
    write_csv(root / "paired_effects_vs_base.csv", effect_rows)
    atomic_write_json(root / "snapshot_manifest.json", manifest)
    atomic_write_json(
        root / "interim_results.json",
        {"manifest": manifest, "metrics": metric_rows, "paired_effects": effect_rows},
    )
    atomic_write_text(root / "README.md", build_markdown(metric_rows, effect_rows, manifest))
    atomic_write_text(root / "COMPLETE", f"completed_at={captured_at}\nstatus={PROVISIONAL_STATUS}\n")
    print(build_markdown(metric_rows, effect_rows, manifest))


if __name__ == "__main__":
    main()
