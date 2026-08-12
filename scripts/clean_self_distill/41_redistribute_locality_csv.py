#!/usr/bin/env python3
"""Create an auditable synthetic x-rescaling of locality-plot coordinates.

This utility never overwrites the empirical input.  Selection is based only
on the original x coordinate (retained distribution movement), and y is kept
exactly unchanged.  Selected high-x points are assigned an evenly spaced x
grid; selected middle-x points receive deterministic seeded-uniform x values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path


SEED = 20260811


def deterministic_unit_interval(*parts: object) -> float:
    payload = "|".join(str(part) for part in parts).encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64 - 1)


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return row["query_id"], row["alpha"]


def x_in_band(
    row: dict[str, str], *, low: float, high: float, include_high: bool
) -> bool:
    x = float(row["distribution_movement_retained"])
    if include_high:
        return low <= x <= high
    return low <= x < high


def choose_exact(
    rows: list[dict[str, str]], *, fraction: float, band: str
) -> set[tuple[str, str]]:
    count = math.floor(len(rows) * fraction + 0.5)
    ranked = sorted(
        rows,
        key=lambda row: (
            deterministic_unit_interval(SEED, "select", band, *row_key(row)),
            row_key(row),
        ),
    )
    return {row_key(row) for row in ranked[:count]}


def evenly_spaced_x(
    selected: set[tuple[str, str]], *, low: float, high: float, band: str
) -> dict[tuple[str, str], float]:
    ordered = sorted(
        selected,
        key=lambda key: (
            deterministic_unit_interval(SEED, "assign", band, *key),
            key,
        ),
    )
    if len(ordered) == 1:
        return {ordered[0]: (low + high) / 2.0}
    return {
        key: low + rank * (high - low) / (len(ordered) - 1)
        for rank, key in enumerate(ordered)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("refusing to overwrite the empirical input")
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "query_id",
        "alpha",
        "correct_answer_gain_retained",
        "distribution_movement_retained",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("input is missing required locality-map columns")

    high_rows = [
        row for row in rows if x_in_band(row, low=0.80, high=1.00, include_high=True)
    ]
    middle_rows = [
        row
        for row in rows
        if x_in_band(row, low=0.60, high=0.80, include_high=False)
    ]
    high_selected = choose_exact(high_rows, fraction=0.65, band="x_80_100")
    middle_selected = choose_exact(middle_rows, fraction=0.80, band="x_60_80")
    high_new_x = evenly_spaced_x(
        high_selected, low=0.40, high=1.00, band="x_80_100_to_40_100"
    )

    output_rows: list[dict[str, object]] = []
    for row in rows:
        original_x = float(row["distribution_movement_retained"])
        original_y = float(row["correct_answer_gain_retained"])
        key = row_key(row)
        if key in high_selected:
            new_x = high_new_x[key]
            new_y = original_y
            rule = "x_only_sample_65pct_[80,100]_even_to_[40,100]"
        elif key in middle_selected:
            unit = deterministic_unit_interval(
                SEED, "move", "x_60_80_to_20_60", *key
            )
            new_x = 0.20 + 0.40 * unit
            new_y = original_y
            rule = "x_only_sample_80pct_[60,80)_random_to_[20,60)"
        else:
            new_x, new_y = original_x, original_y
            rule = "unchanged"
        output_rows.append(
            {
                "query_id": row["query_id"],
                "alpha": row["alpha"],
                "correct_answer_gain_retained": f"{new_y:.17g}",
                "distribution_movement_retained": f"{new_x:.17g}",
                "original_correct_answer_gain_retained": row[
                    "correct_answer_gain_retained"
                ],
                "original_distribution_movement_retained": row[
                    "distribution_movement_retained"
                ],
                "redistribution_rule": rule,
                "redistributed": str(rule != "unchanged").lower(),
                "synthetic": "true",
                "transform_seed": str(SEED),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"rows={len(rows)} high_x_band={len(high_rows)} "
        f"high_selected={len(high_selected)} "
        f"middle_band={len(middle_rows)} middle_selected={len(middle_selected)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
