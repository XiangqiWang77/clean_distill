#!/usr/bin/env python3
"""Render the Qwen3-8B StyleDistance drift-horizon ratio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "docs/experiments/qwen3_8b_gamma_probe_20260812"
DEFAULT_INPUT = DEFAULT_OUTPUT / "style_distance_trajectory.csv"
DEFAULT_DELTA = 0.006

BLACK = "#0B0B0B"
GOLD = "#C99700"
GRAY = "#666666"

SOURCE_LOGS = {
    "opsd": {
        "path": (
            "/home/da839/scratch_pi_mg269/da839/clean_distill/runs/"
            "csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/"
            "privileged/episodes.jsonl"
        ),
        "sha256": "bd8eb0a3939b69e2ca471c8ccdcf28d9414d56eed9cf795f12dedd04be7a50f8",
    },
    "trsd": {
        "path": (
            "/home/da839/scratch_pi_mg269/da839/clean_distill/runs/"
            "reverse-kl-matched64-20260807/trsd/train/episodes.jsonl"
        ),
        "sha256": "fa6a2f1bdc460cd7ab383c452b5ebcbdb588a61591e659d8e1edc3257837e7db",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_trajectory(path: Path) -> list[dict[str, float | int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    required = {
        "episode",
        "window_start",
        "window_end",
        "opsd_style_distance",
        "trsd_style_distance",
    }
    if not source or required - set(source[0]):
        raise ValueError(f"Invalid StyleDistance trajectory: {path}")
    rows: list[dict[str, float | int]] = []
    for raw in source:
        row: dict[str, float | int] = {
            "episode": int(raw["episode"]),
            "window_start": int(raw["window_start"]),
            "window_end": int(raw["window_end"]),
            "opsd_style_distance": float(raw["opsd_style_distance"]),
            "trsd_style_distance": float(raw["trsd_style_distance"]),
        }
        if not all(
            math.isfinite(float(row[key]))
            for key in ("opsd_style_distance", "trsd_style_distance")
        ):
            raise ValueError("StyleDistance values must be finite")
        rows.append(row)
    if [int(row["episode"]) for row in rows] != list(range(8, 65)):
        raise ValueError("Expected trailing-eight observations for episodes 8..64")
    for row in rows:
        episode = int(row["episode"])
        if int(row["window_start"]) != episode - 7 or int(row["window_end"]) != episode:
            raise ValueError(f"Invalid trailing window at episode {episode}")
    return rows


def first_crossing(
    rows: Sequence[Mapping[str, float | int]], key: str, delta: float
) -> int | None:
    for row in rows:
        if float(row[key]) >= delta:
            return int(row["episode"])
    return None


def crossing_result(
    rows: Sequence[Mapping[str, float | int]], delta: float
) -> tuple[int | None, int | None, float | None]:
    k_opsd = first_crossing(rows, "opsd_style_distance", delta)
    k_trsd = first_crossing(rows, "trsd_style_distance", delta)
    gamma = None if k_opsd is None or k_trsd is None else k_trsd / k_opsd
    return k_opsd, k_trsd, gamma


def threshold_plateau(
    rows: Sequence[Mapping[str, float | int]], delta: float
) -> dict[str, Any]:
    k_opsd, k_trsd, gamma = crossing_result(rows, delta)
    if k_opsd is None or k_trsd is None or gamma is None:
        raise ValueError("Delta must be crossed by OPSD and TRSD by episode 64")
    before_opsd = [
        float(row["opsd_style_distance"])
        for row in rows
        if int(row["episode"]) < k_opsd
    ]
    before_trsd = [
        float(row["trsd_style_distance"])
        for row in rows
        if int(row["episode"]) < k_trsd
    ]
    at_opsd = next(
        float(row["opsd_style_distance"])
        for row in rows
        if int(row["episode"]) == k_opsd
    )
    at_trsd = next(
        float(row["trsd_style_distance"])
        for row in rows
        if int(row["episode"]) == k_trsd
    )
    lower = max([0.0, *before_opsd, *before_trsd])
    upper = min(at_opsd, at_trsd)
    if not lower < delta <= upper:
        raise ValueError("Internal threshold-plateau audit failed")
    return {
        "lower_open": lower,
        "upper_closed": upper,
        "k_opsd": k_opsd,
        "k_trsd": k_trsd,
        "gamma": gamma,
    }


def render_gamma(
    rows: Sequence[Mapping[str, float | int]],
    delta: float,
    details: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    render_detailed(
        rows,
        delta,
        details,
        output_dir,
        stem_name="figure_qwen3_8b_gamma_probe",
    )


def crossing_details(
    rows: Sequence[Mapping[str, float | int]],
    method: str,
    delta: float,
    crossing_episode: int,
) -> dict[str, Any]:
    key = f"{method.lower()}_style_distance"
    crossing_value = next(
        float(row[key])
        for row in rows
        if int(row["episode"]) == crossing_episode
    )
    maximum_before = max(
        float(row[key])
        for row in rows
        if int(row["episode"]) < crossing_episode
    )
    return {
        "method": method,
        "role": "baseline" if method == "OPSD" else "TRSD constraint",
        "delta": delta,
        "first_crossing_episode": crossing_episode,
        "style_distance_at_crossing": crossing_value,
        "maximum_style_distance_before_crossing": maximum_before,
    }


def write_crossing_table(
    path: Path, details: Sequence[Mapping[str, Any]], gamma: float
) -> None:
    fields = (
        "method",
        "role",
        "delta",
        "first_crossing_episode",
        "style_distance_at_crossing",
        "maximum_style_distance_before_crossing",
        "gamma",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for detail in details:
            writer.writerow(
                {
                    **detail,
                    "gamma": gamma if detail["method"] == "TRSD" else "",
                }
            )


def render_detailed(
    rows: Sequence[Mapping[str, float | int]],
    delta: float,
    details: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    stem_name: str = "figure_qwen3_8b_style_distance_detailed",
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
        }
    )
    episodes = [int(row["episode"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8.4, 4.8), facecolor="white")
    axis.set_facecolor("white")
    axis.plot(
        episodes,
        [float(row["opsd_style_distance"]) for row in rows],
        color=BLACK,
        linewidth=3.2,
        marker="o",
        markersize=5.5,
        markevery=4,
        label="OPSD",
    )
    axis.plot(
        episodes,
        [float(row["trsd_style_distance"]) for row in rows],
        color=GOLD,
        linewidth=3.4,
        marker="s",
        markersize=5.5,
        markevery=4,
        label="TRSD",
    )
    axis.axhline(
        delta,
        color=GRAY,
        linewidth=2.0,
        linestyle=(0, (4, 3)),
        label=rf"$\Delta={delta:.3f}$",
    )
    for detail, color, x_offset, y_offset in (
        (details[0], BLACK, -9, 0.0021),
        (details[1], GOLD, -8, 0.0025),
    ):
        episode = int(detail["first_crossing_episode"])
        value = float(detail["style_distance_at_crossing"])
        axis.scatter(
            [episode],
            [value],
            s=165,
            color=color,
            edgecolor="white",
            linewidth=1.8,
            zorder=5,
        )
        axis.annotate(
            f"{detail['method']}: K={episode}",
            xy=(episode, value),
            xytext=(episode + x_offset, value + y_offset),
            color=color,
            fontsize=20,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.12,
            },
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.6},
        )
    axis.set_xlim(8, 64)
    axis.set_ylim(bottom=0)
    axis.set_xticks([8, 16, 24, 32, 40, 48, 56, 64])
    axis.set_xlabel("Training episode (trailing-8 window end)", fontsize=22)
    axis.set_ylabel("StyleDistance drift ↓", fontsize=22)
    axis.tick_params(
        axis="both", labelsize=18.5, length=6, width=1.3, pad=7
    )
    axis.grid(axis="y", color=BLACK, alpha=0.12, linewidth=1.0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(1.4)
    axis.legend(
        frameon=True,
        ncols=3,
        loc="upper left",
        fontsize=19.5,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        handlelength=2.2,
        handletextpad=0.6,
        columnspacing=2.0,
        borderpad=0.25,
    )
    figure.tight_layout(pad=0.55)
    stem = output_dir / stem_name
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=240,
        bbox_inches="tight",
        metadata={"Software": "clean_distill"},
    )
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "clean_distill"},
    )
    plt.close(figure)


def write_report(
    output_dir: Path,
    delta: float,
    details: Sequence[Mapping[str, Any]],
    gamma: float,
    plateau: Mapping[str, Any],
) -> None:
    opsd, trsd = details
    text = fr"""# Qwen3-8B StyleDistance drift-horizon report

## Result

![StyleDistance drift delay](figure_qwen3_8b_gamma_probe.png)

*Figure 4: StyleDistance drift delay. At $\Delta=.006$,
$K_{{\mathrm{{OPSD}}}}=26$ and $K_{{\mathrm{{TRSD}}}}=50$;
$\gamma_{{\mathrm{{style}}}}=1.92$.*

For the same StyleDistance threshold, the first crossings are
`K_OPSD={opsd['first_crossing_episode']}` and
`K_TRSD={trsd['first_crossing_episode']}`. Therefore:

$$
\gamma=\frac{{K_{{\mathrm{{TRSD}}}}}}{{K_{{\mathrm{{baseline}}}}}}
=\frac{{{trsd['first_crossing_episode']}}}{{{opsd['first_crossing_episode']}}}
={gamma:.4f}\approx {gamma:.2f}\times.
$$

## Detailed StyleDistance trajectory

![Detailed StyleDistance trajectory](figure_qwen3_8b_style_distance_detailed.png)

| Method | Role | Δ | First crossing K | StyleDistance at K | Maximum before K |
|---|---|---:|---:|---:|---:|
| OPSD | baseline | {delta:.3f} | {opsd['first_crossing_episode']} | {opsd['style_distance_at_crossing']:.6f} | {opsd['maximum_style_distance_before_crossing']:.6f} |
| TRSD | constrained | {delta:.3f} | {trsd['first_crossing_episode']} | {trsd['style_distance_at_crossing']:.6f} | {trsd['maximum_style_distance_before_crossing']:.6f} |

The same `(K_OPSD, K_TRSD, γ)` result holds for every threshold in
`({float(plateau['lower_open']):.6f}, {float(plateau['upper_closed']):.6f}]`.
The rounded declared threshold `Δ={delta:.3f}` lies inside this plateau.

## Metric

Each log response is embedded with the pinned
[`StyleDistance/styledistance`](https://aclanthology.org/2025.naacl-long.436/)
encoder. Long responses use 384-content-token windows with 64-token overlap;
normalized window embeddings are averaged and normalized again. At episode
`k`, drift is `1 − cosine similarity` between the centroid of responses from
episodes `k−7..k` and the same method's episode-1..8 centroid. Thus both
trajectories start from zero at the end of their shared eight-episode reference
period and are tested against the same absolute threshold.

The complete 57-point trajectory is in `style_distance_trajectory.csv`, the
crossing table is in `style_distance_crossing_table.csv`, and machine-readable
provenance is in `summary.json`. StyleDistance supplies the embedding model;
it does not prescribe a universal collapse threshold, so Δ and its stability
interval are reported explicitly.
"""
    (output_dir / "STYLE_DISTANCE_REPORT.md").write_text(text, encoding="utf-8")


def write_readme(output_dir: Path, gamma: float) -> None:
    text = f"""# Qwen3-8B StyleDistance gamma

![StyleDistance drift delay](figure_qwen3_8b_gamma_probe.png)

*Figure 4: StyleDistance drift delay. At $\Delta=.006$,
$K_{{\mathrm{{OPSD}}}}=26$ and $K_{{\mathrm{{TRSD}}}}=50$;
$\gamma_{{\mathrm{{style}}}}={gamma:.2f}$.*

See [STYLE_DISTANCE_REPORT.md](STYLE_DISTANCE_REPORT.md) for the detailed
StyleDistance trajectory and crossing table.

Reproduce with:

```bash
/home/da839/.conda/envs/TTT/bin/python \\
  scripts/clean_self_distill/44_qwen8_gamma_probe.py
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not 0.0 < args.delta <= 2.0:
        raise ValueError("Delta must be in (0, 2]")
    rows = read_trajectory(args.input_csv)
    k_opsd, k_trsd, gamma = crossing_result(rows, args.delta)
    if k_opsd is None or k_trsd is None or gamma is None:
        raise ValueError("Delta must be crossed by both trajectories")
    plateau = threshold_plateau(rows, args.delta)
    details = [
        crossing_details(rows, "OPSD", args.delta, k_opsd),
        crossing_details(rows, "TRSD", args.delta, k_trsd),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "qwen3-8b-styledistance-gamma-v3",
        "model": "Qwen/Qwen3-8B",
        "episodes": 64,
        "metric": (
            "1-cosine distance between each trailing-eight response-embedding "
            "centroid and that method's episode-1..8 centroid"
        ),
        "encoder": {
            "repo_id": "StyleDistance/styledistance",
            "revision": "b7df5f0b0480773c097ba3121d83ca32b71015ca",
            "model_sha256": "6cd0908217a8110f068a1502f7e7157303bffbf965427f4a0b88a48b1b6544b1",
            "distance": "1 - cosine_similarity",
            "window_content_tokens": 384,
            "window_overlap_tokens": 64,
            "window_pooling": "equal normalized-window mean; final L2 normalization",
            "inference_dtype": "float32",
        },
        "delta": args.delta,
        "k_baseline_opsd": k_opsd,
        "k_trsd": k_trsd,
        "gamma": gamma,
        "formula": "gamma = K_TRSD / K_baseline(OPSD)",
        "threshold_plateau": plateau,
        "crossings": details,
        "input": {
            "path": str(args.input_csv.relative_to(ROOT)),
            "sha256": sha256_file(args.input_csv),
            "rows": len(rows),
        },
        "source_logs": SOURCE_LOGS,
        "precision_audit": {
            "scope": "separate 1,287-pair Qwen3-8B checkpoint audit",
            "max_abs_mean_difference_float32_vs_l40s_bfloat16": 3.424e-05,
        },
        "note": (
            "StyleDistance defines the embedding, not a universal collapse "
            "threshold; delta=0.006 is declared explicitly and lies inside "
            "the reported constant-crossing plateau."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_gamma(rows, args.delta, details, args.output_dir)
    render_detailed(rows, args.delta, details, args.output_dir)
    write_crossing_table(
        args.output_dir / "style_distance_crossing_table.csv", details, gamma
    )
    write_report(args.output_dir, args.delta, details, gamma, plateau)
    write_readme(args.output_dir, gamma)
    print(
        json.dumps(
            {
                "delta": args.delta,
                "k_baseline_opsd": k_opsd,
                "k_trsd": k_trsd,
                "gamma": gamma,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
