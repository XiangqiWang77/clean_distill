#!/usr/bin/env python3
"""Collect Qwen3-8B seed-stability accuracy and matched H100 cost results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


CONDITIONS = {
    "thinking_seed0": ("thinking", 0),
    "thinking_seed1": ("thinking", 1479179816),
    "thinking_seed2": ("thinking", 1266198024),
    "nothinking_seed0": ("non-thinking", 0),
}
METHODS = ("OPSD", "LGSD")
EPISODES = (0, 16, 32, 48, 64)
DECODING_SEEDS = (20260819, 20260820, 20260821)
FULL_METHOD_VARIANTS = {
    "grpo_prm_64": ("GRPO-PRM", 64),
    "srpo_16": ("SRPO", 16),
    "srpo_64": ("SRPO", 64),
    "opsd_16": ("OPSD", 16),
    "opsd_64": ("OPSD", 64),
    "lgsd_16": ("LGSD", 16),
    "lgsd_64": ("LGSD", 64),
}


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, fieldnames: list[str], values: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_accuracy(run_root: Path, allow_partial: bool) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    expected = 0
    for condition, (mode, training_seed) in CONDITIONS.items():
        seeds = DECODING_SEEDS if condition == "thinking_seed0" else (20260820,)
        for method in METHODS:
            slug = method.lower()
            for episode in EPISODES:
                for decoding_seed in seeds:
                    expected += 1
                    root = (
                        run_root
                        / "eval_accuracy_dynamics"
                        / condition
                        / slug
                        / f"episode_{episode:04d}"
                        / f"decode_{decoding_seed}"
                    )
                    path = root / "scored.jsonl"
                    marker = root / "SCORE_COMPLETE"
                    if not path.is_file() or not marker.is_file():
                        if allow_partial:
                            continue
                        raise FileNotFoundError(f"missing completed score: {path}")
                    scored = rows(path)
                    query_ids = {str(row["query_id"]) for row in scored}
                    if len(scored) != 143 or len(query_ids) != 143:
                        raise ValueError(f"expected 143 unique rows: {path}")
                    correct = sum(float(row.get("correct", 0.0)) for row in scored)
                    output.append(
                        {
                            "mode": mode,
                            "training_condition": condition,
                            "training_seed": training_seed,
                            "decoding_seed": decoding_seed,
                            "method": method,
                            "episode": episode,
                            "correct": int(round(correct)),
                            "total": len(scored),
                            "strict_accuracy_pct": 100.0 * correct / len(scored),
                            "scored_sha256": digest(path),
                        }
                    )
    if not allow_partial and len(output) != expected:
        raise ValueError(f"expected {expected} accuracy rows, found {len(output)}")
    return output


def collect_cost(run_root: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for condition, seed in (("thinking_seed1", 1479179816), ("thinking_seed2", 1266198024)):
        for method in METHODS:
            slug = method.lower()
            train_root = run_root / "train" / condition / slug
            journal = train_root / "episodes.jsonl"
            episode_rows = rows(journal)
            if len(episode_rows) != 64:
                raise ValueError(f"expected 64 episodes: {journal}")
            checkpoint = train_root / "checkpoints" / "episode_0064"
            checkpoint_bytes = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
            output.append(
                {
                    "method": method,
                    "training_seed": seed,
                    "accelerator": "H100",
                    "gpu_count": 1,
                    "active_training_hours": sum(float(row["episode_seconds"]) for row in episode_rows) / 3600.0,
                    "checkpoint_mib": checkpoint_bytes / (2**20),
                    "peak_allocated_gib": max(
                        float(row["resource_usage"]["cuda_peak_memory_allocated_bytes"]) for row in episode_rows
                    )
                    / (2**30),
                }
            )
    return output


def collect_full_method_modes(run_root: Path, require_complete: bool) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for condition, (mode, training_seed) in CONDITIONS.items():
        for variant, (method, episode) in FULL_METHOD_VARIANTS.items():
            root = run_root / "eval" / condition / variant
            path = root / "scored.jsonl"
            marker = root / "SCORE_COMPLETE"
            if not path.is_file() or not marker.is_file():
                if require_complete:
                    raise FileNotFoundError(f"missing completed full-method score: {path}")
                continue
            scored = rows(path)
            query_ids = {str(row["query_id"]) for row in scored}
            if len(scored) != 143 or len(query_ids) != 143:
                raise ValueError(f"expected 143 unique rows: {path}")
            correct = sum(float(row.get("correct", 0.0)) for row in scored)
            output.append(
                {
                    "mode": mode,
                    "training_condition": condition,
                    "training_seed": training_seed,
                    "variant": variant,
                    "method": method,
                    "episode": episode,
                    "correct": int(round(correct)),
                    "total": len(scored),
                    "strict_accuracy_pct": 100.0 * correct / len(scored),
                    "scored_sha256": digest(path),
                }
            )
    if require_complete and len(output) != 28:
        raise ValueError(f"expected 28 full-method mode rows, found {len(output)}")
    return output


def summarize_full_method_modes(values: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for variant, (method, episode) in FULL_METHOD_VARIANTS.items():
        matching = [row for row in values if row["variant"] == variant]
        thinking = [float(row["strict_accuracy_pct"]) for row in matching if row["mode"] == "thinking"]
        nonthinking = [float(row["strict_accuracy_pct"]) for row in matching if row["mode"] == "non-thinking"]
        if len(thinking) != 3 or len(nonthinking) != 1:
            continue
        thinking_mean = sum(thinking) / len(thinking)
        output.append(
            {
                "variant": variant,
                "method": method,
                "episode": episode,
                "thinking_mean_accuracy_pct": thinking_mean,
                "thinking_min_accuracy_pct": min(thinking),
                "thinking_max_accuracy_pct": max(thinking),
                "nonthinking_accuracy_pct": nonthinking[0],
                "thinking_minus_nonthinking_pp": thinking_mean - nonthinking[0],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/qwen3-8b-deepmath-multiseed-20260820"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-full-methods", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accuracy = collect_accuracy(args.run_root, args.allow_partial)
    cost = collect_cost(args.run_root)
    full_modes = collect_full_method_modes(args.run_root, args.require_full_methods)
    fields = [
        "mode",
        "training_condition",
        "training_seed",
        "decoding_seed",
        "method",
        "episode",
        "correct",
        "total",
        "strict_accuracy_pct",
        "scored_sha256",
    ]
    write_csv(args.output_dir / "all_accuracy_runs.csv", fields, accuracy)
    write_csv(
        args.output_dir / "decoding_seed_accuracy.csv",
        fields,
        [row for row in accuracy if row["training_condition"] == "thinking_seed0"],
    )
    write_csv(
        args.output_dir / "training_seed_accuracy.csv",
        fields,
        [row for row in accuracy if row["mode"] == "thinking" and row["decoding_seed"] == 20260820],
    )
    write_csv(
        args.output_dir / "thinking_mode_accuracy.csv",
        fields,
        [row for row in accuracy if row["decoding_seed"] == 20260820],
    )
    write_csv(
        args.output_dir / "resource_cost.csv",
        [
            "method",
            "training_seed",
            "accelerator",
            "gpu_count",
            "active_training_hours",
            "checkpoint_mib",
            "peak_allocated_gib",
        ],
        cost,
    )
    full_mode_fields = [
        "mode",
        "training_condition",
        "training_seed",
        "variant",
        "method",
        "episode",
        "correct",
        "total",
        "strict_accuracy_pct",
        "scored_sha256",
    ]
    write_csv(args.output_dir / "thinking_mode_full_methods.csv", full_mode_fields, full_modes)
    write_csv(
        args.output_dir / "thinking_mode_full_methods_summary.csv",
        [
            "variant",
            "method",
            "episode",
            "thinking_mean_accuracy_pct",
            "thinking_min_accuracy_pct",
            "thinking_max_accuracy_pct",
            "nonthinking_accuracy_pct",
            "thinking_minus_nonthinking_pp",
        ],
        summarize_full_method_modes(full_modes),
    )
    print(
        json.dumps(
            {"accuracy_rows": len(accuracy), "cost_rows": len(cost), "full_method_mode_rows": len(full_modes)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
