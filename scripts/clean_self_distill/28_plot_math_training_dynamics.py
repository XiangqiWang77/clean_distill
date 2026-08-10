#!/usr/bin/env python3
"""Plot the four-panel Qwen3-8B Math training-dynamics figure.

Panels (a)--(c) use the matched 64-query DeepMath training stream.  Panel
(d) uses strict Acc@1 from the common frozen AMC23/AIME24/AIME25 scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    # Preserve a frozen snapshot supplied through PYTHONPATH when the plot is
    # built by a report job; use the live checkout only as a local fallback.
    sys.path.append(str(ROOT))

from src.opsd_format import extract_boxed_answer, grade_boxed_answer  # noqa: E402


RUNS = Path("/home/da839/scratch_pi_mg269/da839/clean_distill/runs")
DEFAULT_PRIVILEGED = (
    RUNS
    / "csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/episodes.jsonl"
)
DEFAULT_TRSD = RUNS / "reverse-kl-matched64-20260807/trsd/train/episodes.jsonl"
DEFAULT_LABELS = (
    RUNS
    / "csd-qwen3-8b-three-sellpoints-poc-07/prepared/distill_labels.sealed.jsonl"
)
DEFAULT_ACCURACY = (
    ROOT / "docs/experiments/trsd_table_report_20260808/tables/main_accuracy.csv"
)
DEFAULT_OUTPUT = ROOT / "docs/figures/qwen3_8b_math_training_dynamics"

METHODS = ("Privilege-SD", "TRSD")
COLORS = {"Privilege-SD": "#D55E00", "TRSD": "#0072B2"}
EXPECTED_STEPS = list(range(1, 65))
EXPECTED_CHECKPOINTS = (16, 32, 48, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--privileged-journal", type=Path, default=DEFAULT_PRIVILEGED)
    parser.add_argument("--trsd-journal", type=Path, default=DEFAULT_TRSD)
    parser.add_argument("--deepmath-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--accuracy", type=Path, default=DEFAULT_ACCURACY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument(
        "--allow-partial-accuracy",
        action="store_true",
        help="Permit a preview before checkpoints 32 and 48 finish.",
    )
    return parser.parse_args()


def finite_float(value: object, *, field: str, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: {field} is not finite")
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for row in load_jsonl(path):
        query_id = str(row.get("query_id", "")).strip()
        answer = str(row.get("answer", "")).strip()
        problem_sha256 = str(row.get("problem_sha256", "")).strip().lower()
        if not query_id or not answer or len(problem_sha256) != 64:
            raise ValueError(f"Malformed sealed label for {query_id!r} in {path}")
        if query_id in labels:
            raise ValueError(f"Duplicate sealed label {query_id!r} in {path}")
        labels[query_id] = {
            "answer": answer,
            "problem_sha256": problem_sha256,
        }
    return labels


def load_journal(
    path: Path, method: str, labels: dict[str, dict[str, str]]
) -> list[dict[str, object]]:
    raw_rows = load_jsonl(path)
    episodes = [int(row.get("episode", -1)) for row in raw_rows]
    if episodes != EXPECTED_STEPS:
        raise ValueError(f"{path} must contain the exact episode sequence 1..64")

    rows: list[dict[str, object]] = []
    for line_no, raw in enumerate(raw_rows, 1):
        context = f"{path}:{line_no}"
        if raw.get("optimizer_step") is not True:
            raise ValueError(f"{context}: optimizer_step is not true")
        if str(raw.get("source", "")).lower() != "deepmath":
            raise ValueError(f"{context}: expected DeepMath, got {raw.get('source')!r}")

        query_id = str(raw.get("query_id", ""))
        problem_sha256 = str(raw.get("problem_sha256", "")).lower()
        label = labels.get(query_id)
        if label is None:
            raise ValueError(f"{context}: no sealed label for {query_id!r}")
        if problem_sha256 != label["problem_sha256"]:
            raise ValueError(f"{context}: problem hash disagrees with sealed label")

        response = str(raw.get("student_prefix", ""))
        extracted = extract_boxed_answer(response)
        correct = int(grade_boxed_answer(extracted, label["answer"]))
        entropy_proxy = -finite_float(
            raw.get("student_normalized_logprob"),
            field="student_normalized_logprob",
            context=context,
        )
        response_tokens = int(raw.get("response_tokens", 0))
        if entropy_proxy < 0 or response_tokens <= 0:
            raise ValueError(f"{context}: invalid entropy proxy or response length")

        rows.append(
            {
                "method": method,
                "training_step": int(raw["episode"]),
                "query_id": query_id,
                "problem_sha256": problem_sha256,
                "entropy_proxy_nats_per_token": entropy_proxy,
                "response_tokens": response_tokens,
                "reward": correct,
                "boxed_answer": "" if extracted is None else extracted,
            }
        )
    return rows


def validate_matched_streams(
    privileged: list[dict[str, object]], trsd: list[dict[str, object]]
) -> None:
    privileged_keys = [
        (row["query_id"], row["problem_sha256"]) for row in privileged
    ]
    trsd_keys = [(row["query_id"], row["problem_sha256"]) for row in trsd]
    if privileged_keys != trsd_keys:
        mismatch = next(
            index
            for index, (left, right) in enumerate(
                zip(privileged_keys, trsd_keys, strict=True), 1
            )
            if left != right
        )
        raise ValueError(f"Training streams first disagree at episode {mismatch}")


def normalize_method(raw: dict[str, str]) -> str | None:
    method = raw.get("method", "").strip()
    label = raw.get("method_label", "").strip()
    if method in METHODS:
        return method
    if method in {"base", "Base"} or label == "Base":
        return "Base"
    if method.startswith("privileged_") or label.startswith("Privilege-SD"):
        return "Privilege-SD"
    if method.startswith("trsd_") or label.startswith("TRSD"):
        return "TRSD"
    return None


def load_accuracy(
    path: Path, *, allow_partial: bool
) -> dict[str, list[tuple[int, float]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"{path} has no accuracy rows")

    final_schema = "combined_strict_acc1_percent" in raw_rows[0]
    points: dict[str, dict[int, float]] = {
        "Base": {},
        "Privilege-SD": {},
        "TRSD": {},
    }
    for raw in raw_rows:
        if final_schema:
            if raw.get("model") not in (None, "", "Qwen3-8B"):
                continue
            value_raw = raw.get("combined_strict_acc1_percent", "")
        else:
            if raw.get("dataset") != "combined":
                continue
            value_raw = raw.get("strict_acc1_percent", "")
        method = normalize_method(raw)
        if method is None:
            continue
        episode = int(raw.get("episodes", 0))
        value = finite_float(value_raw, field="strict accuracy", context=str(path))
        if not 0 <= value <= 100:
            raise ValueError(f"{path}: strict accuracy is outside [0, 100]")
        if episode in points[method]:
            raise ValueError(f"{path}: duplicate {method} checkpoint {episode}")
        points[method][episode] = value

    if set(points["Base"]) != {0}:
        raise ValueError(f"{path}: expected exactly one Base checkpoint at episode 0")
    for method in METHODS:
        observed = set(points[method])
        expected = set(EXPECTED_CHECKPOINTS)
        if allow_partial:
            if not observed or not observed <= expected:
                raise ValueError(f"{path}: invalid partial checkpoints for {method}: {observed}")
        elif observed != expected:
            raise ValueError(
                f"{path}: {method} checkpoints are {sorted(observed)}, "
                f"expected {list(EXPECTED_CHECKPOINTS)}"
            )

    return {
        method: sorted(values.items()) for method, values in points.items()
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if not 1 <= window <= len(values):
        raise ValueError(f"window must be in [1, {len(values)}], got {window}")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    result[window - 1 :] = np.convolve(
        values, np.ones(window, dtype=np.float64) / window, mode="valid"
    )
    return result


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.18,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_raw_and_mean(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    method: str,
    field: str,
    *,
    window: int,
    scale: float = 1.0,
) -> None:
    steps = np.asarray([int(row["training_step"]) for row in rows])
    values = np.asarray([float(row[field]) for row in rows]) / scale
    color = COLORS[method]
    ax.plot(steps, values, color=color, alpha=0.16, linewidth=0.9)
    ax.plot(
        steps,
        rolling_mean(values, window),
        color=color,
        linewidth=2.4,
        label=method,
        solid_capstyle="round",
    )


def build_figure(
    by_method: dict[str, list[dict[str, object]]],
    accuracy: dict[str, list[tuple[int, float]]],
    *,
    window: int,
) -> plt.Figure:
    configure_style()
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 3.55))

    panel_specs = (
        (axes[0], "entropy_proxy_nats_per_token", 1.0),
        (axes[1], "response_tokens", 1000.0),
        (axes[2], "reward", 1.0),
    )
    for ax, field, scale in panel_specs:
        for method in METHODS:
            plot_raw_and_mean(
                ax, by_method[method], method, field, window=window, scale=scale
            )
        ax.set_xlim(0, 65)
        ax.set_xticks([0, 16, 32, 48, 64])
        ax.set_xlabel("Training step")

    axes[0].set_title("(a) Math: entropy")
    axes[0].set_ylabel("Entropy proxy (nats/token)")
    axes[1].set_title("(b) Math: length")
    axes[1].set_ylabel("Response length (k tokens)")
    axes[1].set_ylim(bottom=0)
    axes[2].set_title("(c) Math: reward")
    axes[2].set_ylabel("Verifier reward")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_yticks([0.0, 0.5, 1.0])

    accuracy_ax = axes[3]
    base = accuracy["Base"][0]
    for method in METHODS:
        method_points = [base, *accuracy[method]]
        x = [point[0] for point in method_points]
        y = [point[1] for point in method_points]
        accuracy_ax.plot(
            x,
            y,
            color=COLORS[method],
            linewidth=2.4,
            marker="o",
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=method,
        )
    accuracy_ax.set_title("(d) Math: accuracy")
    accuracy_ax.set_xlabel("Checkpoint")
    accuracy_ax.set_ylabel("Strict Acc@1 (%)")
    accuracy_ax.set_xlim(-2, 66)
    accuracy_ax.set_xticks([0, 16, 32, 48, 64])
    accuracy_values = [
        value for method in ("Base", *METHODS) for _, value in accuracy[method]
    ]
    lower = max(0.0, 5.0 * math.floor((min(accuracy_values) - 5.0) / 5.0))
    upper = min(100.0, 5.0 * math.ceil((max(accuracy_values) + 5.0) / 5.0))
    accuracy_ax.set_ylim(lower, upper)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=2,
        handlelength=2.6,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.18, top=0.82, wspace=0.32)
    return fig


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    labels = load_labels(args.deepmath_labels)
    privileged = load_journal(args.privileged_journal, "Privilege-SD", labels)
    trsd = load_journal(args.trsd_journal, "TRSD", labels)
    validate_matched_streams(privileged, trsd)
    by_method = {"Privilege-SD": privileged, "TRSD": trsd}
    accuracy = load_accuracy(args.accuracy, allow_partial=args.allow_partial_accuracy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure(by_method, accuracy, window=args.window)
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        output_path = args.output_dir / f"math_training_dynamics.{extension}"
        figure.savefig(output_path, bbox_inches="tight", **kwargs)
        if extension == "svg":
            svg_text = output_path.read_text(encoding="utf-8")
            output_path.write_text(
                "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(figure)

    dynamics_rows = [row for method in METHODS for row in by_method[method]]
    write_csv(args.output_dir / "math_training_dynamics.csv", dynamics_rows)
    accuracy_rows = [
        {"method": method, "checkpoint": checkpoint, "strict_acc1_percent": value}
        for method, points in accuracy.items()
        for checkpoint, value in points
    ]
    write_csv(args.output_dir / "math_accuracy.csv", accuracy_rows)

    complete_accuracy = all(
        [checkpoint for checkpoint, _ in accuracy[method]]
        == list(EXPECTED_CHECKPOINTS)
        for method in METHODS
    )
    summary = {
        "schema_version": "qwen3-8b-math-training-dynamics-v1",
        "panels": [
            "(a) Math: entropy",
            "(b) Math: length",
            "(c) Math: reward",
            "(d) Math: accuracy",
        ],
        "training_stream": {"source": "deepmath", "matched_queries": 64},
        "smoothing": f"trailing_{args.window}_step_mean",
        "entropy_definition": (
            "negative mean student log-probability of realized rollout tokens; "
            "an on-policy entropy proxy, not exact full-vocabulary categorical entropy"
        ),
        "reward_definition": "frozen boxed-answer verifier reward in {0,1}",
        "accuracy_definition": (
            "Strict Acc@1 on frozen AMC23(83)+AIME24(30)+AIME25(30); "
            "10,240-token cap hits are incorrect"
        ),
        "accuracy_complete": complete_accuracy,
        "accuracy_checkpoints": {
            method: [checkpoint for checkpoint, _ in accuracy[method]]
            for method in ("Base", *METHODS)
        },
        "reward_correct": {
            method: sum(int(row["reward"]) for row in by_method[method])
            for method in METHODS
        },
        "inputs": {
            "privileged_journal": str(args.privileged_journal),
            "trsd_journal": str(args.trsd_journal),
            "deepmath_labels": str(args.deepmath_labels),
            "accuracy": str(args.accuracy),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = "complete" if complete_accuracy else "preview (checkpoints 32/48 pending)"
    readme = f"""# Qwen3-8B Math training dynamics

Status: **{status}**.

The four panels are `(a) Math: entropy`, `(b) Math: length`,
`(c) Math: reward`, and `(d) Math: accuracy`. Pale curves in panels (a)--(c)
are per-rollout values; thick curves are trailing {args.window}-step means.
The 64 training observations use the exact matched DeepMath query order.

The entropy panel reports realized-token surprisal
`-mean(log p_student(y_t | prefix))` as an on-policy entropy proxy. The
journals do not store exact full-vocabulary categorical entropy, so the figure
does not claim that stronger quantity. Reward is the frozen boxed-answer
verifier's binary score on each training rollout. Accuracy is Strict Acc@1 on
the common 143-question AMC23/AIME24/AIME25 scorer; a 10,240-token cap hit is
incorrect.

See `summary.json` for exact definitions and the CSV files for every plotted
value.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
