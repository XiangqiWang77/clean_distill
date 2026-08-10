#!/usr/bin/env python3
"""Atomically summarize Base and four matched distillation evaluations."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED = (
    ("base", "Base", 0),
    ("privileged_16", "Privilege-SD 16", 16),
    ("trsd_16", "TRSD 16", 16),
    ("privileged_64", "Privilege-SD 64", 64),
    ("trsd_64", "TRSD 64", 64),
)
SOURCE_COUNTS = {"amc23": 83, "aime24": 30, "aime25": 30}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def strict_correct(row: Mapping[str, Any]) -> int:
    return int(bool(row.get("correct")) and not bool(row.get("truncated")))


def validate(rows: list[dict[str, Any]], method: str, episode: int) -> None:
    if len(rows) != 143:
        raise ValueError(f"{method}: expected 143 rows, found {len(rows)}")
    counts = Counter(str(row.get("source")) for row in rows)
    if counts != Counter(SOURCE_COUNTS):
        raise ValueError(f"{method}: source counts are {dict(counts)}")
    query_ids = [str(row.get("query_id")) for row in rows]
    if len(set(query_ids)) != 143:
        raise ValueError(f"{method}: query IDs must be unique")
    expected_method = {
        "base": "base",
        "privileged_16": "privileged_sd",
        "privileged_64": "privileged_sd",
        "trsd_16": "trsd",
        "trsd_64": "trsd",
    }[method]
    for row in rows:
        if row.get("method") != expected_method:
            raise ValueError(f"{method}: method identity mismatch")
        if int(row.get("checkpoint_episode", -1)) != episode:
            raise ValueError(f"{method}: checkpoint episode mismatch")
        if int(row.get("max_new_tokens", -1)) != 10_240:
            raise ValueError(f"{method}: evaluation budget mismatch")


def identity(rows: Iterable[Mapping[str, Any]]) -> list[tuple[str, str, str, int]]:
    return sorted(
        (
            str(row["query_id"]),
            str(row["problem_sha256"]),
            str(row["source"]),
            int(row["sample_index"]),
        )
        for row in rows
    )


def aggregate(method: str, label: str, episode: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in ("combined", "amc23", "aime24", "aime25"):
        selected = rows if source == "combined" else [row for row in rows if row["source"] == source]
        correct = sum(strict_correct(row) for row in selected)
        cap_hits = sum(bool(row.get("truncated")) for row in selected)
        tokens = sum(int(row.get("generated_tokens", 0)) for row in selected)
        seconds = sum(float((row.get("resource_usage") or {}).get("generation_seconds", 0.0)) for row in selected)
        result.append(
            {
                "method": method,
                "method_label": label,
                "episodes": episode,
                "dataset": source,
                "strict_correct": correct,
                "n": len(selected),
                "strict_acc1": correct / len(selected),
                "strict_acc1_percent": 100.0 * correct / len(selected),
                "budget_cap_hits": cap_hits,
                "budget_cap_hit_rate": cap_hits / len(selected),
                "mean_generated_tokens": tokens / len(selected),
                "mean_generation_seconds": seconds / len(selected),
            }
        )
    return result


def paired(reference: list[dict[str, Any]], method: list[dict[str, Any]]) -> dict[str, int]:
    ref = {str(row["query_id"]): strict_correct(row) for row in reference}
    cur = {str(row["query_id"]): strict_correct(row) for row in method}
    return {
        "wrong_to_correct": sum(ref[key] == 0 and cur[key] == 1 for key in ref),
        "correct_to_wrong": sum(ref[key] == 1 and cur[key] == 0 for key in ref),
    }


def csv_payload(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_if_complete(
    run_root: Path,
    *,
    model_id: str = "Qwen/Qwen3-1.7B",
    model_label: str = "Qwen3-1.7B",
) -> bool:
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / ".report.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        paths = {name: run_root / "eval" / name / "scored.jsonl" for name, _, _ in EXPECTED}
        completed = [name for name, path in paths.items() if path.is_file()]
        progress = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed_evaluations": completed,
            "expected_evaluations": [name for name, _, _ in EXPECTED],
        }
        atomic_text(run_root / "progress.json", json.dumps(progress, indent=2, sort_keys=True) + "\n")
        if len(completed) != len(EXPECTED):
            print(f"results waiting: {len(completed)}/{len(EXPECTED)} evaluations complete")
            return False

        loaded: dict[str, list[dict[str, Any]]] = {}
        table: list[dict[str, Any]] = []
        base_identity: list[tuple[str, str, str, int]] | None = None
        for name, label, episode in EXPECTED:
            rows = read_jsonl(paths[name])
            validate(rows, name, episode)
            current_identity = identity(rows)
            if base_identity is None:
                base_identity = current_identity
            elif current_identity != base_identity:
                raise ValueError(f"{name}: held-out identity differs from Base")
            loaded[name] = rows
            table.extend(aggregate(name, label, episode, rows))

        combined = {
            row["method"]: row
            for row in table
            if row["dataset"] == "combined"
        }
        p64_t64 = paired(loaded["privileged_64"], loaded["trsd_64"])
        comparisons = {
            "short_term_trsd16_vs_base_pp": 100.0
            * (combined["trsd_16"]["strict_acc1"] - combined["base"]["strict_acc1"]),
            "long_term_trsd64_vs_base_pp": 100.0
            * (combined["trsd_64"]["strict_acc1"] - combined["base"]["strict_acc1"]),
            "long_term_trsd64_vs_privileged64_pp": 100.0
            * (combined["trsd_64"]["strict_acc1"] - combined["privileged_64"]["strict_acc1"]),
            "trsd64_vs_privileged64_transitions": p64_t64,
        }
        results = run_root / "results"
        atomic_text(results / "main_accuracy.csv", csv_payload(table))
        summary = {
            "schema_version": "fiveway-math-results-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "training_data": "DeepMath",
            "evaluation_data": ["AMC23", "AIME24", "AIME25"],
            "episode_horizons": [16, 64],
            "accuracy": table,
            "comparisons": comparisons,
        }
        atomic_text(results / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")

        lines = [
            f"# {model_label} TRSD and Privilege-SD results",
            "",
            "| Method | Episodes | AMC23 | AIME24 | AIME25 | Combined |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        labels = {name: label for name, label, _ in EXPECTED}
        episodes = {name: episode for name, _, episode in EXPECTED}
        for name, _, _ in EXPECTED:
            rows = {row["dataset"]: row for row in table if row["method"] == name}
            cells = [f"{rows[source]['strict_acc1_percent']:.2f}% ({rows[source]['strict_correct']}/{rows[source]['n']})" for source in ("amc23", "aime24", "aime25", "combined")]
            lines.append(f"| {labels[name]} | {episodes[name]} | " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "## Three claims",
                "",
                f"- Short-term: TRSD-16 − Base = {comparisons['short_term_trsd16_vs_base_pp']:+.2f} pp.",
                f"- Long-term: TRSD-64 − Base = {comparisons['long_term_trsd64_vs_base_pp']:+.2f} pp.",
                f"- Equal 64-episode horizon: TRSD-64 − Privilege-SD64 = {comparisons['long_term_trsd64_vs_privileged64_pp']:+.2f} pp, with W→C/C→W = {p64_t64['wrong_to_correct']}/{p64_t64['correct_to_wrong']}.",
                "",
            ]
        )
        atomic_text(run_root / "RESULTS.md", "\n".join(lines))
        atomic_text(run_root / "RUN_COMPLETE", "complete\n")
        print(f"complete results: {run_root / 'RESULTS.md'}")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-label", default="Qwen3-1.7B")
    args = parser.parse_args()
    build_if_complete(args.run_root, model_id=args.model_id, model_label=args.model_label)


if __name__ == "__main__":
    main()
