#!/usr/bin/env python3
"""Probe a threshold-crossing stability ratio from Qwen3-8B 64-step logs.

The primary quantity is a descriptive over-drift proxy, not an accuracy-collapse
estimate.  For each method, an eight-episode trailing common-response NLL curve
is anchored at its post-warmup minimum.  K(delta) is the first later episode at
which the curve has rebounded by at least ``delta`` nats/token, and
gamma(delta) = K_TRSD(delta) / K_OPSD(delta).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


BLACK = "#111111"
YELLOW = "#FFC400"
DARK_YELLOW = "#B88600"
PALE_YELLOW = "#FFF3B0"
MID_GRAY = "#777777"
LIGHT_GRAY = "#E5E5E5"
WHITE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nll-csv",
        type=Path,
        default=Path(
            "docs/experiments/qwen3_8b_positive_cot_loops_20260811/"
            "qwen3_8b_64episode_common_evaluation_nll.csv"
        ),
    )
    parser.add_argument(
        "--accuracy-csv",
        type=Path,
        default=Path(
            "docs/experiments/trsd_table_report_20260808/tables/"
            "main_accuracy.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/experiments/qwen3_8b_gamma_probe_20260812"),
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="NLL rebound threshold in nats/token (default: 0.05).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float:
    return float(value) if value else math.nan


def load_nll_curves(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    steps = np.asarray([int(row["training_step"]) for row in rows], dtype=int)
    if not np.array_equal(steps, np.arange(1, 65)):
        raise ValueError(f"{path} must contain exactly training steps 1..64")

    curves = {
        "steps": steps,
        "opsd_raw": np.asarray(
            [as_float(row["privilege_sd_student_token_nll"]) for row in rows]
        ),
        "trsd_raw": np.asarray(
            [as_float(row["trsd_student_token_nll"]) for row in rows]
        ),
        "opsd_smooth": np.asarray(
            [as_float(row["privilege_sd_nll_8step_mean"]) for row in rows]
        ),
        "trsd_smooth": np.asarray(
            [as_float(row["trsd_nll_8step_mean"]) for row in rows]
        ),
    }
    for name in ("opsd_smooth", "trsd_smooth"):
        finite = np.flatnonzero(np.isfinite(curves[name]))
        if not np.array_equal(finite, np.arange(7, 64)):
            raise ValueError(f"{name} must contain a trailing-8 mean at steps 8..64")
    return curves


def crossing(
    steps: np.ndarray, values: np.ndarray, delta: float
) -> dict[str, float | int | None]:
    if delta <= 0:
        raise ValueError("delta must be positive")
    valid = np.isfinite(values)
    valid_steps = steps[valid]
    valid_values = values[valid]
    minimum_index = int(np.argmin(valid_values))
    minimum_step = int(valid_steps[minimum_index])
    minimum_value = float(valid_values[minimum_index])
    threshold = minimum_value + delta
    later_steps = valid_steps[minimum_index:]
    later_values = valid_values[minimum_index:]
    hits = np.flatnonzero(later_values >= threshold)
    crossing_step = int(later_steps[hits[0]]) if len(hits) else None
    crossing_value = float(later_values[hits[0]]) if len(hits) else None
    return {
        "minimum_step": minimum_step,
        "minimum_nll": minimum_value,
        "threshold_nll": threshold,
        "crossing_step": crossing_step,
        "crossing_nll": crossing_value,
    }


def load_accuracy(path: Path) -> dict[str, dict[int, float]]:
    rows = [row for row in read_csv(path) if row["dataset"] == "combined"]
    selected = {
        row["method"]: (int(row["episodes"]), float(row["strict_acc1_percent"]))
        for row in rows
        if row["method"] in {"base", "privileged_16", "privileged_64", "trsd_16", "trsd_64"}
    }
    expected = {"base", "privileged_16", "privileged_64", "trsd_16", "trsd_64"}
    if set(selected) != expected:
        raise ValueError(f"{path} is missing required combined-accuracy rows")
    base = selected["base"][1]
    return {
        "OPSD": {
            0: base,
            selected["privileged_16"][0]: selected["privileged_16"][1],
            selected["privileged_64"][0]: selected["privileged_64"][1],
        },
        "TRSD": {
            0: base,
            selected["trsd_16"][0]: selected["trsd_16"][1],
            selected["trsd_64"][0]: selected["trsd_64"][1],
        },
    }


def delta_grid() -> list[float]:
    return [round(float(value), 4) for value in np.arange(0.02, 0.0776, 0.0025)]


def sensitivity_rows(
    steps: np.ndarray,
    opsd: np.ndarray,
    trsd: np.ndarray,
    horizon: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for delta in delta_grid():
        opsd_crossing = crossing(steps, opsd, delta)
        trsd_crossing = crossing(steps, trsd, delta)
        k_opsd = opsd_crossing["crossing_step"]
        k_trsd = trsd_crossing["crossing_step"]
        if k_opsd is not None and k_trsd is not None:
            gamma = float(k_trsd) / float(k_opsd)
            status = "observed"
            lower_bound = None
        elif k_opsd is not None:
            gamma = None
            status = "trsd_right_censored"
            lower_bound = horizon / float(k_opsd)
        else:
            gamma = None
            status = "both_right_censored"
            lower_bound = None
        rows.append(
            {
                "delta_nats_per_token": delta,
                "k_opsd": k_opsd,
                "k_trsd": k_trsd,
                "gamma": gamma,
                "gamma_lower_bound": lower_bound,
                "status": status,
            }
        )
    return rows


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 1.1,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "text.color": BLACK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def build_figure(
    curves: Mapping[str, np.ndarray],
    accuracy: Mapping[str, Mapping[int, float]],
    opsd_crossing: Mapping[str, float | int | None],
    trsd_crossing: Mapping[str, float | int | None],
    delta: float,
    output_dir: Path,
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 4.25),
        gridspec_kw={"width_ratios": [1.65, 0.72, 1.0]},
    )

    steps = curves["steps"]
    opsd = curves["opsd_smooth"]
    trsd = curves["trsd_smooth"]

    axis = axes[0]
    style_axis(axis)
    post_minimum_step = max(
        int(opsd_crossing["minimum_step"]), int(trsd_crossing["minimum_step"])
    )
    post_minimum = steps >= post_minimum_step
    post_steps = steps[post_minimum]
    opsd_rebound = opsd[post_minimum] - float(opsd_crossing["minimum_nll"])
    trsd_rebound = trsd[post_minimum] - float(trsd_crossing["minimum_nll"])
    axis.fill_between(
        post_steps,
        opsd_rebound,
        trsd_rebound,
        where=np.isfinite(opsd_rebound) & np.isfinite(trsd_rebound),
        color=PALE_YELLOW,
        alpha=0.55,
        linewidth=0,
    )
    axis.plot(post_steps, opsd_rebound, color=BLACK, linewidth=2.3, label="OPSD baseline")
    axis.plot(post_steps, trsd_rebound, color=YELLOW, linewidth=2.8, label="TRSD")
    axis.axhline(
        delta,
        color=MID_GRAY,
        linewidth=1.2,
        linestyle=(0, (3, 3)),
        label=rf"common threshold $\Delta={delta:.2f}$",
    )
    for result, color, label, vertical_offset in (
        (opsd_crossing, BLACK, "OPSD", 0.009),
        (trsd_crossing, DARK_YELLOW, "TRSD", -0.014),
    ):
        minimum_step = int(result["minimum_step"])
        crossing_step = result["crossing_step"]
        crossing_nll = result["crossing_nll"]
        axis.scatter(
            [minimum_step],
            [0.0],
            s=32,
            color=color,
            edgecolor=WHITE,
            linewidth=0.8,
            zorder=5,
        )
        if crossing_step is not None and crossing_nll is not None:
            crossing_rebound = float(crossing_nll) - float(result["minimum_nll"])
            axis.scatter(
                [crossing_step],
                [crossing_rebound],
                s=58,
                marker="D",
                color=color,
                edgecolor=WHITE if color == BLACK else BLACK,
                linewidth=0.9,
                zorder=6,
            )
            axis.annotate(
                f"{label}: K={crossing_step}",
                xy=(crossing_step, crossing_rebound),
                xytext=(crossing_step - 8, crossing_rebound + vertical_offset),
                color=color,
                fontsize=8.5,
                fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.9},
            )
    axis.set_xlim(post_minimum_step - 1, 65)
    axis.set_ylim(-0.004, 0.091)
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Post-minimum trailing-8 NLL rebound (nats/token) ↑")
    axis.set_title("(a) Same-Δ over-drift crossing", loc="left", fontweight="bold")
    axis.text(
        0.02,
        0.96,
        rf"$D_m(k)=L_m(k)-\min L_m$; common $\Delta={delta:.2f}$ nats/token",
        transform=axis.transAxes,
        va="top",
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": PALE_YELLOW, "edgecolor": "none"},
    )
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.84), handlelength=2.3)

    axis = axes[1]
    style_axis(axis)
    k_opsd = int(opsd_crossing["crossing_step"])
    k_trsd = int(trsd_crossing["crossing_step"])
    gamma = k_trsd / k_opsd
    bars = axis.bar(
        [0, 1],
        [k_opsd, k_trsd],
        color=[BLACK, YELLOW],
        edgecolor=BLACK,
        linewidth=1.2,
        width=0.66,
    )
    axis.bar_label(bars, labels=[str(k_opsd), str(k_trsd)], padding=3, fontweight="bold")
    axis.set_xticks([0, 1], ["OPSD", "TRSD"])
    axis.set_ylim(0, 68)
    axis.set_ylabel("K(Δ), episodes")
    axis.set_title("(b) Horizon ratio", loc="left", fontweight="bold")
    axis.text(
        0.5,
        0.62,
        rf"$\gamma=\frac{{{k_trsd}}}{{{k_opsd}}}={gamma:.2f}\times$",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": PALE_YELLOW,
            "edgecolor": BLACK,
            "linewidth": 0.9,
        },
    )
    axis.text(
        0.5,
        0.48,
        "descriptive NLL proxy",
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
        color=MID_GRAY,
    )

    axis = axes[2]
    style_axis(axis)
    for label, color, marker in (("OPSD", BLACK, "o"), ("TRSD", YELLOW, "s")):
        points = sorted(accuracy[label].items())
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        axis.plot(
            x,
            y,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2.3,
            markeredgecolor=BLACK,
            markeredgewidth=0.8,
            label=label,
        )
    base_accuracy = accuracy["OPSD"][0]
    axis.axhline(base_accuracy, color=MID_GRAY, linestyle=(0, (3, 3)), linewidth=1.0)
    axis.fill_between([0, 64], 0, base_accuracy, color=PALE_YELLOW, alpha=0.24)
    axis.set_xlim(-2, 66)
    axis.set_ylim(45, 75)
    axis.set_xticks([0, 16, 64])
    axis.set_xlabel("Evaluated checkpoint")
    axis.set_ylabel("Strict math Acc@1 (%) ↑")
    axis.set_title("(c) No endpoint accuracy collapse observed", loc="left", fontweight="bold")
    axis.text(
        0.04,
        0.94,
        "Episode-64 endpoints remain above Base\nSparse checkpoints → collapse K not identified",
        transform=axis.transAxes,
        va="top",
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": PALE_YELLOW, "edgecolor": "none"},
    )
    axis.legend(frameon=False, loc="lower right")

    fig.suptitle(
        "Qwen3-8B: γ is a threshold-crossing ratio, not a model scalar",
        x=0.04,
        y=1.035,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.04,
        -0.02,
        "Completed 64-episode historical trajectories; same response per matched episode. "
        "OPSD denotes the raw privileged-target baseline. Single trajectory; no confidence interval.",
        fontsize=8,
        color=MID_GRAY,
    )
    fig.tight_layout(w_pad=2.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_qwen3_8b_gamma_probe"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", metadata={"Software": "clean_distill"})
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "clean_distill", "CreationDate": None},
    )
    plt.close(fig)


def write_sensitivity(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    fields = (
        "delta_nats_per_token",
        "k_opsd",
        "k_trsd",
        "gamma",
        "gamma_lower_bound",
        "status",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def write_readme(
    output_dir: Path,
    delta: float,
    opsd: Mapping[str, float | int | None],
    trsd: Mapping[str, float | int | None],
    gamma: float,
    accuracy: Mapping[str, Mapping[int, float]],
    observed_sensitivity: Sequence[float],
) -> None:
    text = fr"""# Qwen3-8B stability-horizon γ probe

![Qwen3-8B gamma probe](figure_qwen3_8b_gamma_probe.png)

## Result

The strict accuracy-collapse ratio is **not identified** by these 64-episode logs: strict Acc@1 was evaluated only at Base, episode 16, and episode 64. Both episode-64 endpoints remain above the Base accuracy ({accuracy['OPSD'][0]:.2f}%), with OPSD at {accuracy['OPSD'][64]:.2f}% and TRSD at {accuracy['TRSD'][64]:.2f}%. These sparse checkpoints neither show an endpoint collapse nor resolve a possible crossing and recovery between checkpoints, so their first-crossing ratio cannot be reduced to a finite point estimate.

For the explicitly declared over-drift proxy, set Δ={delta:.2f} nats/token on each method's trailing-8 common-response NLL rebound from its own post-warmup minimum. The observed crossings are

\[
K_{{\mathrm{{OPSD}}}}={opsd['crossing_step']},\qquad
K_{{\mathrm{{TRSD}}}}={trsd['crossing_step']},\qquad
\gamma_{{\mathrm{{NLL}}}}=\frac{{{trsd['crossing_step']}}}{{{opsd['crossing_step']}}}={gamma:.4f}\approx {gamma:.2f}\times.
\]

This means TRSD delays this particular measured NLL-rebound crossing by one episode. Across the finite, observed threshold sweep with Δ in [0.02, 0.07], γ ranges from {min(observed_sensitivity):.2f}× to {max(observed_sensitivity):.2f}×; larger thresholds become right-censored for TRSD. This sensitivity is why γ must always be reported together with Δ and the drift metric.

## Estimand

For method \(m\), let \(L_m(k)\) be the trailing-8 NLL at episode \(k\), let \(k_m^*\) be its minimum over observed episodes 8–64, and define

\[
K_m(\Delta)=\min\{{k\ge k_m^*:L_m(k)-L_m(k_m^*)\ge\Delta\}}.
\]

The primary probe uses Δ={delta:.2f}. OPSD's minimum is {fmt(float(opsd['minimum_nll']))} at episode {opsd['minimum_step']} and its absolute crossing level is {fmt(float(opsd['threshold_nll']))}; TRSD's minimum is {fmt(float(trsd['minimum_nll']))} at episode {trsd['minimum_step']} and its crossing level is {fmt(float(trsd['threshold_nll']))}.

## Scope and limitations

- `OPSD` in this figure denotes the repository's raw privileged-target baseline (`Privilege-SD` in the source tables), not a new 64-episode run of the official generalized-JSD implementation.
- The NLL comparison uses the same ordinary OPSD response for both methods at each matched episode, but the response changes across episodes. The trailing window reduces query-level noise but does not turn this into a fixed held-out probe set.
- The historical source trajectories have different rollout-token caps (OPSD 4,096; TRSD 10,240). The same-sequence scoring removes response mismatch at a given episode, but it does not remove this training-protocol confound.
- This is one deterministic 64-episode trajectory per method. The threshold sensitivity table is descriptive and is not a confidence interval.
- Strict accuracy is available only at Base, episode 16, and episode 64, so an accuracy first-crossing time is not identifiable from these checkpoints.
- The newer L40S conservative-control runs were still incomplete when this bundle was generated and are not mixed into the estimate.

## Claim–evidence map

- **Claim:** no accuracy-collapse γ is identified through episode 64. **Evidence:** only Base/16/64 strict Acc@1 checkpoints are available; both episode-64 endpoints exceed Base. **Status:** supported; crossings between checkpoints are unresolved.
- **Claim:** the declared NLL proxy gives γ≈{gamma:.2f}×. **Evidence:** first post-minimum crossings at episodes {opsd['crossing_step']} and {trsd['crossing_step']} under Δ={delta:.2f}. **Status:** descriptively supported for this metric and threshold.
- **Claim:** γ is threshold-dependent. **Evidence:** `gamma_threshold_sensitivity.csv`. **Status:** supported; no threshold-free scalar claim is permitted.

## Reproduce

```bash
/home/da839/.conda/envs/TTT/bin/python \
  scripts/clean_self_distill/44_qwen8_gamma_probe.py
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    curves = load_nll_curves(args.nll_csv)
    accuracy = load_accuracy(args.accuracy_csv)
    horizon = int(curves["steps"][-1])
    opsd = crossing(curves["steps"], curves["opsd_smooth"], args.delta)
    trsd = crossing(curves["steps"], curves["trsd_smooth"], args.delta)
    if opsd["crossing_step"] is None or trsd["crossing_step"] is None:
        raise ValueError("Primary delta must be observed for both methods within 64 episodes")
    gamma = float(trsd["crossing_step"]) / float(opsd["crossing_step"])

    sensitivity = sensitivity_rows(
        curves["steps"], curves["opsd_smooth"], curves["trsd_smooth"], horizon
    )
    observed_gamma = [float(row["gamma"]) for row in sensitivity if row["gamma"] is not None]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_figure(curves, accuracy, opsd, trsd, args.delta, args.output_dir)
    write_sensitivity(args.output_dir / "gamma_threshold_sensitivity.csv", sensitivity)

    summary = {
        "schema_version": "qwen3-8b-gamma-threshold-probe-v1",
        "model": "Qwen/Qwen3-8B",
        "episodes": horizon,
        "primary_estimand": {
            "metric": "trailing_8_common_response_nll_rebound_nats_per_token",
            "delta": args.delta,
            "crossing_rule": "first observed episode at or after the method-specific post-warmup minimum",
            "formula": "gamma = K_TRSD(delta) / K_OPSD(delta)",
        },
        "primary_result": {
            "opsd": opsd,
            "trsd": trsd,
            "gamma": gamma,
            "interpretation": "descriptive_over_drift_proxy",
        },
        "accuracy_collapse": {
            "status": "not_identified_sparse_checkpoints",
            "reason": "strict Acc@1 is available only at Base, episode 16, and episode 64; first crossings between checkpoints are unresolved",
            "base_acc1_percent": accuracy["OPSD"][0],
            "opsd_episode64_acc1_percent": accuracy["OPSD"][64],
            "trsd_episode64_acc1_percent": accuracy["TRSD"][64],
            "k_opsd": None,
            "k_trsd": None,
            "gamma": None,
        },
        "threshold_sensitivity": {
            "observed_delta_min": 0.02,
            "observed_delta_max": 0.07,
            "observed_gamma_min": min(observed_gamma),
            "observed_gamma_max": max(observed_gamma),
            "note": "higher deltas include TRSD right-censoring; see CSV",
        },
        "sources": {
            "nll_csv": {"path": str(args.nll_csv), "sha256": sha256_file(args.nll_csv)},
            "accuracy_csv": {
                "path": str(args.accuracy_csv),
                "sha256": sha256_file(args.accuracy_csv),
            },
            "opsd_journal_sha256": "bd8eb0a3939b69e2ca471c8ccdcf28d9414d56eed9cf795f12dedd04be7a50f8",
            "trsd_journal_sha256": "fa6a2f1bdc460cd7ab383c452b5ebcbdb588a61591e659d8e1edc3257837e7db",
        },
        "limitations": [
            "OPSD label maps to the raw Privilege-SD source branch",
            "same response within each matched episode but not a fixed response across episodes",
            "historical rollout caps differ: OPSD 4096 versus TRSD 10240",
            "one trajectory per method; no statistical confidence interval",
            "strict accuracy is available only at Base, episode 16, and episode 64",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(args.output_dir, args.delta, opsd, trsd, gamma, accuracy, observed_gamma)

    print(json.dumps(summary["primary_result"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
