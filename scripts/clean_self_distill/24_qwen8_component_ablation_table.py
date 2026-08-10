#!/usr/bin/env python3
"""Build Table 2 from the completed Qwen3-8B TRSD component ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROWS = (
    ("TRSD (trajectory projection)", "CSD_A2_TRSD64_SCORED"),
    ("− trajectory projection (direct OPSD)", "CSD_A2_DIRECT64_SCORED"),
    ("independent token budgets αt", "independent_token_budgets"),
    ("fixed global α", "fixed_global_alpha"),
    ("arithmetic probability path", "arithmetic_probability_path"),
    ("forward-KL student loss", "forward_kl_student_loss"),
    ("− same-prefix scoring", "without_same_prefix_scoring"),
    ("+ realized-update guard", "realized_update_guard"),
)
SOURCES = (("AMC23", "amc23"), ("AIME24", "aime24"), ("AIME25", "aime25"))


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def strict(row: dict[str, Any]) -> bool:
    truncated = row.get("truncated")
    if truncated is None:
        truncated = row.get("behavioral_diagnostics", {}).get("truncated", False)
    return bool(row.get("correct")) and not bool(truncated)


def summarize(label: str, path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 143 or len({row["query_id"] for row in rows}) != 143:
        raise ValueError(f"{label}: expected 143 unique rows in {path}")
    output: dict[str, Any] = {"Variant": label}
    combined = 0
    for display, source in SOURCES:
        selected = [row for row in rows if str(row.get("source", "")).lower() == source]
        expected = 83 if source == "amc23" else 30
        if len(selected) != expected:
            raise ValueError(f"{label}: {source} has {len(selected)} rows")
        correct = sum(strict(row) for row in selected)
        combined += correct
        output[display] = f"{100 * correct / expected:.2f}"
    output["Combined"] = f"{100 * combined / 143:.2f}"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--include",
        action="append",
        choices=tuple(source for _, source in ROWS if not source.startswith("CSD_")),
        help="component variant to include; repeat as needed (default: all)",
    )
    args = parser.parse_args()
    config = read_env(args.config)
    run_root = Path(config["CSD_A2_RUN_ROOT"])
    output: list[dict[str, Any]] = []
    for label, source in ROWS:
        if not source.startswith("CSD_") and args.include and source not in args.include:
            continue
        if source.startswith("CSD_"):
            path = Path(config[source])
        else:
            path = run_root / "eval" / source / "scored.jsonl"
        output.append(summarize(label, path))

    results = run_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    with (results / "table_2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    lines = [
        "# Table 2: Qwen3-8B component ablations at 64 episodes",
        "",
        "Every value is strict Acc@1 under the same frozen scorer as Table 1.",
        "",
        "| Variant | AMC23 | AIME24 | AIME25 | Combined |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['Variant']} | {row['AMC23']} | {row['AIME24']} | {row['AIME25']} | {row['Combined']} |"
        for row in output
    )
    (results / "TABLE_2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_root / "RESULTS_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(results / "TABLE_2.md")


if __name__ == "__main__":
    main()
