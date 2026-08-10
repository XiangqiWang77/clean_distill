#!/usr/bin/env python3
"""Build a strict Acc@1 table from matched scored JSONL artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE_ORDER = ("amc23", "aime24", "aime25")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _parse_entry(text: str) -> tuple[str, int, Path]:
    parts = text.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("entry must be DISPLAY|EPISODES|SCORED.jsonl")
    try:
        episodes = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("entry episodes must be an integer") from exc
    return parts[0], episodes, Path(parts[2])


def _strict_summary(path: Path) -> dict[str, Any]:
    rows = [row for row in _read_jsonl(path) if row.get("profile") == "acc1"]
    if len(rows) != 143:
        raise ValueError(f"{path} must contain 143 Acc@1 rows, found {len(rows)}")
    query_ids = [str(row.get("query_id", "")) for row in rows]
    if any(not query_id for query_id in query_ids) or len(set(query_ids)) != 143:
        raise ValueError(f"{path} has missing or duplicate query ids")
    result: dict[str, Any] = {"path": str(path), "total": len(rows)}
    combined_correct = 0
    for source in SOURCE_ORDER:
        selected = [row for row in rows if str(row.get("source", "")).casefold() == source]
        correct = sum(
            bool(float(row.get("correct", 0.0))) and not bool(row.get("truncated"))
            for row in selected
        )
        result[source] = {"correct": correct, "total": len(selected)}
        combined_correct += correct
    if sum(result[source]["total"] for source in SOURCE_ORDER) != 143:
        raise ValueError(f"{path} contains an unexpected source")
    result["combined"] = {"correct": combined_correct, "total": 143}
    result["query_ids"] = sorted(query_ids)
    return result


def _format_cell(value: dict[str, int]) -> str:
    correct, total = value["correct"], value["total"]
    return f"{100.0 * correct / total:.2f}% ({correct}/{total})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", action="append", type=_parse_entry, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-label", default="Qwen3-8B")
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for display, episodes, path in args.entry:
        summary = _strict_summary(path)
        summary.update(display=display, episodes=episodes)
        summaries.append(summary)
    reference_ids = summaries[0]["query_ids"]
    if any(summary["query_ids"] != reference_ids for summary in summaries[1:]):
        raise ValueError("methods do not cover the identical 143 query ids")
    base = next((item for item in summaries if item["display"].casefold() == "base"), None)
    if base is None:
        raise ValueError("one entry must have display name Base")

    lines = [
        f"# {args.model_label} DeepMath matched baseline table",
        "",
        "Strict Acc@1 counts a sample as correct only when its boxed answer is correct "
        "and it finishes before the common 10,240-token limit.",
        "",
        "| Method | Episodes | AMC23 | AIME24 | AIME25 | Combined | Δ vs Base |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    base_accuracy = base["combined"]["correct"] / base["combined"]["total"]
    serializable: list[dict[str, Any]] = []
    for summary in summaries:
        accuracy = summary["combined"]["correct"] / summary["combined"]["total"]
        delta = 100.0 * (accuracy - base_accuracy)
        lines.append(
            "| {display} | {episodes} | {amc} | {a24} | {a25} | {combined} | {delta:+.2f} pp |".format(
                display=summary["display"],
                episodes=summary["episodes"],
                amc=_format_cell(summary["amc23"]),
                a24=_format_cell(summary["aime24"]),
                a25=_format_cell(summary["aime25"]),
                combined=_format_cell(summary["combined"]),
                delta=delta,
            )
        )
        serializable.append({key: value for key, value in summary.items() if key != "query_ids"})
    lines.extend(
        [
            "",
            "All entries use identical query IDs, deterministic query seeds, temperature "
            "0.6, top-p 0.95, top-k 20, one sample per query, the same generation prompt, "
            "and the same batched inference engine.",
            "",
        ]
    )
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (destination / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "baseline-comparison-table-v1",
                "strict_metric": "boxed_answer_correct AND truncated=false",
                "matched_query_count": 143,
                "methods": serializable,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
