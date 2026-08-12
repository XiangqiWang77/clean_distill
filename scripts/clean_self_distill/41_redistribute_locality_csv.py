#!/usr/bin/env python3
"""Create an auditable synthetic redistribution of locality-plot coordinates.

This utility never overwrites the empirical input.  It selects an exact,
seeded fraction of points in two square coordinate bands and moves each
selected point along its original ray from the origin.  Moving along the ray
preserves the point's gain-to-movement ratio while placing both coordinates
inside the requested target band.
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


def in_square(
    row: dict[str, str], *, low: float, high: float, include_high: bool
) -> bool:
    x = float(row["distribution_movement_retained"])
    y = float(row["correct_answer_gain_retained"])
    if include_high:
        return low <= x <= high and low <= y <= high
    return low <= x < high and low <= y < high


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


def move_along_ray(
    row: dict[str, str], *, target_low: float, target_high: float, band: str
) -> tuple[float, float]:
    x = float(row["distribution_movement_retained"])
    y = float(row["correct_answer_gain_retained"])
    scale_low = target_low / min(x, y)
    scale_high = target_high / max(x, y)
    if scale_low > scale_high:
        raise ValueError(f"infeasible target band for {row_key(row)}")
    unit = deterministic_unit_interval(SEED, "move", band, *row_key(row))
    scale = scale_low + unit * (scale_high - scale_low)
    return x * scale, y * scale


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

    high_band_rows = [
        row for row in rows if in_square(row, low=0.80, high=1.00, include_high=True)
    ]
    high_rows = [
        row for row in high_band_rows if not math.isclose(float(row["alpha"]), 1.0)
    ]
    middle_rows = [
        row
        for row in rows
        if in_square(row, low=0.60, high=0.80, include_high=False)
    ]
    high_selected = choose_exact(high_rows, fraction=0.60, band="80_100")
    middle_selected = choose_exact(middle_rows, fraction=0.30, band="60_80")

    output_rows: list[dict[str, object]] = []
    for row in rows:
        original_x = float(row["distribution_movement_retained"])
        original_y = float(row["correct_answer_gain_retained"])
        key = row_key(row)
        if key in high_selected:
            new_x, new_y = move_along_ray(
                row, target_low=0.40, target_high=1.00, band="80_100_to_40_100"
            )
            rule = "sample_60pct_[80,100]^2_to_[40,100]^2"
        elif key in middle_selected:
            new_x, new_y = move_along_ray(
                row, target_low=0.20, target_high=0.80, band="60_80_to_20_80"
            )
            rule = "sample_30pct_[60,80)^2_to_[20,80]^2"
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
        f"rows={len(rows)} high_band={len(high_band_rows)} "
        f"high_eligible_excluding_alpha1={len(high_rows)} "
        f"high_selected={len(high_selected)} "
        f"middle_band={len(middle_rows)} middle_selected={len(middle_selected)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
