#!/usr/bin/env python3
"""Build paper Tables 14--16 exclusively from completed Qwen3-8B runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_RUNS = Path("/home/da839/scratch_pi_mg269/da839/clean_distill/runs")
DEFAULT_SOURCES = {
    "math": REPO_ROOT
    / "docs/experiments/expanded_validation_20260809/tables/qwen3_8b_math.csv",
    "logic": REPO_ROOT
    / "docs/experiments/expanded_validation_20260809/tables/qwen3_8b_logic_dataset.csv",
    "style": SCRATCH_RUNS
    / "trsd-style64-final-20260808-v2/matched64_style_summary.csv",
    "same_prefix": SCRATCH_RUNS
    / "trsd-style64-final-20260808-v2/same_prefix_mechanism_summary.csv",
    "wrappers": SCRATCH_RUNS
    / "trsd-short-empirical-20260807/report/mechanism_query_wrapper.csv",
    "demopsd": SCRATCH_RUNS
    / "baseline-deepmath16-20260808-03/demopsd/train/episodes.jsonl",
    "grpo": SCRATCH_RUNS
    / "baseline-deepmath16-20260808-03/grpo/train/episodes.jsonl",
    "privileged": SCRATCH_RUNS
    / "csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/episodes.jsonl",
    "trsd": SCRATCH_RUNS
    / "reverse-kl-matched64-20260807/trsd/train/episodes.jsonl",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty collection")
    return statistics.fmean(materialized)


def one(rows: Iterable[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in fields.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {fields}, found {len(matches)}")
    return matches[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = list(rows[0])
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows
    )
    return "\n".join(lines)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_table14(sources: dict[str, Path]) -> list[dict[str, Any]]:
    math = read_csv(sources["math"])
    logic = read_csv(sources["logic"])
    style = read_csv(sources["style"])
    prefix = read_csv(sources["same_prefix"])
    variants = (
        (
            "raw privileged target",
            "Privilege-SD 64",
            "privileged_sd_64",
            "raw_privileged",
            "raw_privileged_surrogate",
        ),
        (
            "+ trajectory projection (TRSD)",
            "TRSD 64",
            "trsd_64",
            "trsd_projected",
            "trsd_projected",
        ),
    )
    output: list[dict[str, Any]] = []
    for variant, math_method, logic_method, style_target, prefix_projection in variants:
        math_row = one(math, method=math_method)
        sat_row = one(logic, method=logic_method, dataset="satquest")
        ood_row = one(logic, method=logic_method, dataset="logicskills")
        style_row = one(style, target=style_target)
        prefix_row = one(prefix, projection=prefix_projection)
        activation = style_row["constraint_activation_rate"]
        cap_hits = "—" if activation == "N/A" else str(round(float(activation) * 64))
        output.append(
            {
                "Base method": "Privilege-SD",
                "Variant": variant,
                "Math Acc@1": f"{float(math_row['strict_acc1_percent']):.2f}",
                "SATQuest": f"{100 * float(sat_row['accuracy']):.2f}",
                "LogicSkills OOD": f"{100 * float(ood_row['accuracy']):.2f}",
                "Target KL": f"{float(style_row['target_student_kl']):.5f}",
                "Cap hits": cap_hits,
                "Style shift↓": f"{float(prefix_row['style_abs_logprob_shift']):.5f}",
            }
        )
    return output


def wrapper_summary(
    rows: list[dict[str, str]], wrapper: str | None
) -> tuple[dict[str, float], dict[str, float]]:
    selected = rows if wrapper is None else [row for row in rows if row["wrapper"] == wrapper]
    raw = [row for row in selected if row["projection"] == "raw_privileged_surrogate"]
    projected = [row for row in selected if row["projection"] == "trsd_projected"]
    expected = 9 if wrapper is None else 3
    if len(raw) != expected or len(projected) != expected:
        raise ValueError(f"wrapper={wrapper!r}: expected {expected} paired rows")

    def summarize(group: list[dict[str, str]]) -> dict[str, float]:
        return {
            "kl": mean(float(row["achieved_mean_kl"]) for row in group),
            "task": mean(float(row["task_logprob_gain"]) for row in group),
            "style": mean(float(row["style_abs_logprob_shift"]) for row in group),
            "alpha": mean(float(row["alpha"]) for row in group),
        }

    return summarize(raw), summarize(projected)


def build_table15(sources: dict[str, Path], trsd_acc: float) -> list[dict[str, Any]]:
    rows = read_csv(sources["wrappers"])
    definitions = (
        ("Answer-free reasoning method", "neutral"),
        ("Style-only directive (terse)", "terse"),
        ("Style-only directive (verbose)", "verbose"),
        ("Equivalent prompt wrappers", None),
    )
    projected_wrapper_means = [wrapper_summary(rows, name)[1]["task"] for name in ("neutral", "terse", "verbose")]
    wrapper_variance = statistics.pvariance(projected_wrapper_means)
    output: list[dict[str, Any]] = []
    for label, wrapper in definitions:
        raw, projected = wrapper_summary(rows, wrapper)
        output.append(
            {
                "Privilege source / probe": label,
                "Raw KL": f"{raw['kl']:.5f}",
                "Projected KL": f"{projected['kl']:.5f}",
                "Task gain Δ": f"{projected['task'] - raw['task']:+.5f}",
                "Style shift↓": f"{projected['style']:.5f}",
                "Wrapper var.↓": f"{wrapper_variance:.8f}" if wrapper is None else "—",
                "Shared Acc@1": f"{trsd_acc:.2f}",
            }
        )
    return output


def peak_gib(rows: list[dict[str, Any]]) -> str:
    values = [
        row.get("resource_usage", {}).get("cuda_peak_memory_allocated_bytes")
        for row in rows
    ]
    materialized = [float(value) for value in values if value is not None]
    return f"{max(materialized) / 2**30:.2f}" if materialized else "—"


def build_table16(sources: dict[str, Path]) -> list[dict[str, Any]]:
    definitions = (
        ("GRPO", "grpo", 16, "generated_tokens", "teacher_positions", "rollout_count"),
        ("DemoPSD", "demopsd", 16, "generated_tokens", "teacher_positions", "rollout_count"),
        ("Privilege-SD", "privileged", 64, "response_tokens", None, None),
        ("TRSD", "trsd", 64, "response_tokens", None, None),
    )
    output: list[dict[str, Any]] = []
    for method, source, expected, token_key, teacher_key, rollout_key in definitions:
        rows = read_jsonl(sources[source])
        if len(rows) != expected:
            raise ValueError(f"{method}: expected {expected} episodes, found {len(rows)}")
        total_seconds = sum(float(row["episode_seconds"]) for row in rows)
        generated = sum(int(row[token_key]) for row in rows)
        if teacher_key is None:
            teacher_positions = sum(int(row["audit"]["teacher_positions"]) for row in rows)
        else:
            teacher_positions = sum(int(row[teacher_key]) for row in rows)
        rollouts = len(rows) if rollout_key is None else sum(int(row[rollout_key]) for row in rows)
        output.append(
            {
                "Method": method,
                "Episodes": len(rows),
                "Rollouts": rollouts,
                "Generated tokens": generated,
                "Teacher positions": teacher_positions,
                "Update steps": sum(bool(row.get("optimizer_step")) for row in rows),
                "Total s": f"{total_seconds:.1f}",
                "s / episode": f"{total_seconds / len(rows):.1f}",
                "Peak GiB": peak_gib(rows),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs/experiments/qwen3_8b_reused_tables_20260809",
    )
    for name, default in DEFAULT_SOURCES.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    args = parser.parse_args()
    sources = {name: getattr(args, name) for name in DEFAULT_SOURCES}

    table14 = build_table14(sources)
    trsd_acc = float(one(read_csv(sources["math"]), method="TRSD 64")["strict_acc1_percent"])
    tables = {
        14: table14,
        15: build_table15(sources, trsd_acc),
        16: build_table16(sources),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number, rows in tables.items():
        write_csv(args.output_dir / f"table_{number}.csv", rows)

    captions = {
        14: (
            "Trajectory projection at the matched 64-episode horizon on Qwen3-8B. "
            "Math is AMC23/AIME24/AIME25 combined; SATQuest is logical ID/shifted-format "
            "evaluation; LogicSkills is external OOD."
        ),
        15: (
            "Robustness across completed answer-free same-prefix probes. Task gain Δ is "
            "projected minus raw task-token log-probability gain. All rows probe the same "
            "TRSD-64 checkpoint, so Acc@1 is shared."
        ),
        16: (
            "Recorded end-to-end training cost from completed Qwen3-8B runs. GRPO and "
            "DemoPSD are the completed 16-episode baselines; Privilege-SD and TRSD are "
            "the matched 64-episode long-horizon runs."
        ),
    }
    report = [
        "# Qwen3-8B reused-run Tables 14--16",
        "",
        "Every value below is regenerated from completed runs; no new training or evaluation is used.",
    ]
    for number, rows in tables.items():
        report.extend(
            [
                "",
                f"## Table {number}",
                "",
                captions[number],
                "",
                markdown_table(rows),
            ]
        )
    (args.output_dir / "TABLES_14_16.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "qwen3-8b-reused-paper-tables-v1",
        "new_gpu_jobs": 0,
        "sources": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in sources.items()
        },
        "outputs": {str(number): f"table_{number}.csv" for number in tables},
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output_dir), "new_gpu_jobs": 0}))


if __name__ == "__main__":
    main()
