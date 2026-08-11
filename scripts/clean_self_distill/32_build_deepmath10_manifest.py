#!/usr/bin/env python3
"""Freeze an exact, deterministic 10% DeepMath population study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


SCHEMA_VERSION = "trsd-deepmath10-surrogate-study-v1"
SOURCE_ROWS = 31_164
SAMPLE_ROWS = 3_116


def digest(seed: int, stage: str, query_id: str) -> str:
    return hashlib.sha256(f"{seed}|{stage}|{query_id}".encode()).hexdigest()


def load_ids(path: Path) -> list[str]:
    ids: list[str] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1024, columns=["extra_info.index"]):
        ids.extend(str(row["extra_info"]["index"]) for row in batch.to_pylist())
    if len(ids) != SOURCE_ROWS or len(set(ids)) != SOURCE_ROWS:
        raise ValueError(f"Expected {SOURCE_ROWS:,} unique DeepMath query IDs")
    return ids


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepmath", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    ids = load_ids(args.deepmath)
    selected = sorted(ids, key=lambda value: (digest(args.seed, "select", value), value))[
        :SAMPLE_ROWS
    ]
    ordered = sorted(
        selected, key=lambda value: (digest(args.seed, "order", value), value)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "source_path": str(args.deepmath),
        "source_sha256": sha256_file(args.deepmath),
        "source_rows": SOURCE_ROWS,
        "fraction": 0.10,
        "rounding_rule": "floor(0.10 * source_rows)",
        "total": SAMPLE_ROWS,
        "query_ids": ordered,
        "split": {
            "audit": ordered[: SAMPLE_ROWS // 2],
            "confirmation": ordered[SAMPLE_ROWS // 2 :],
        },
        "sharding": "study_index_mod_2",
        "model": "Qwen/Qwen3-8B",
        "primary_epsilon": 0.004,
        "reference_token_cap": 256,
        "claims": {
            "figure_1": (
                "TRSD contracts wrapper/style nuisance while preserving the quality "
                "of reference-solution token signal."
            ),
            "figure_2": (
                "A student-local privileged surrogate can be more reliable than the "
                "unconstrained surrogate."
            ),
        },
        "estimands": {
            "nuisance": [
                "mean per-token variance across neutral/terse/verbose privileged wrappers",
                "mean absolute realized-token shift on the frozen style lexicon",
            ],
            "useful_signal": [
                "reference-solution token log-probability gain over the ordinary student",
                "reference-solution math-token log-probability gain",
            ],
            "surrogate_reliability": [
                "fraction of queries with positive reference-token gain",
                "fraction with positive worst-wrapper reference-token gain",
                "cross-wrapper prompt variance",
            ],
            "signal_quality_noninferiority_margin": 0.01,
            "noninferiority_unit": "absolute query fraction",
        },
        "uncertainty": {
            "unit": "query",
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20260810,
            "tokens_are_independent_replicates": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "rows": SAMPLE_ROWS}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
