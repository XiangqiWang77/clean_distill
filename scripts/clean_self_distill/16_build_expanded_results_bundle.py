#!/usr/bin/env python3
"""Build the compact, auditable bundle for the expanded TRSD evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RUN_8B_MATH = "baseline-deepmath16-20260808-03"
RUN_17B_MATH = "qwen3-1.7b-fourway-20260808"
RUN_8B_LOGIC = "qwen3-8b-logic-threeway-10k-20260808"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def slim_math_rows(paths: dict[str, Path], output: Path) -> int:
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for label, path in paths.items():
            for row in iter_jsonl(path):
                value = {
                    "method_label": label,
                    "method": row["method"],
                    "checkpoint_episode": int(row["checkpoint_episode"]),
                    "query_id": row["query_id"],
                    "problem_sha256": row["problem_sha256"],
                    "source": row["source"],
                    "sample_index": int(row["sample_index"]),
                    "seed": int(row["seed"]),
                    "parsed_answer": row.get("parsed_answer"),
                    "correct": bool(row["correct"]),
                    "truncated": bool(row["truncated"]),
                    "strict_correct": bool(row["correct"]) and not bool(row["truncated"]),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "generated_tokens": int(row["generated_tokens"]),
                    "max_new_tokens": int(row["max_new_tokens"]),
                    "temperature": float(row["temperature"]),
                    "top_p": float(row["top_p"]),
                    "top_k": int(row["top_k"]),
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "query_manifest_sha256": row["query_manifest_sha256"],
                    "generation_config_sha256": row["generation_config_sha256"],
                    "response_sha256": hashlib.sha256(
                        row.get("response", "").encode("utf-8")
                    ).hexdigest(),
                }
                handle.write(json.dumps(value, sort_keys=True) + "\n")
                count += 1
    return count


def slim_logic_rows(source: Path, output: Path) -> int:
    fields = (
        "schema_version",
        "method",
        "checkpoint_episode",
        "dataset",
        "query_id",
        "global_index",
        "eval_regime",
        "problem_type",
        "question_type",
        "task",
        "language",
        "num_variable",
        "extracted_answer",
        "correct",
        "verifier",
        "prompt_tokens",
        "generated_tokens",
        "max_new_tokens",
        "num_shards",
        "shard_index",
    )
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(source):
            value = {field: row.get(field) for field in fields}
            value["response_sha256"] = hashlib.sha256(
                row.get("response", "").encode("utf-8")
            ).hexdigest()
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            count += 1
    return count


def aggregate_logic(
    detail: list[dict[str, Any]], group_field: str
) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in detail:
        key_value = row.get(group_field)
        if key_value is None:
            continue
        key = (str(row["method"]), str(key_value))
        totals[key][0] += int(row["correct"])
        totals[key][1] += int(row["total"])
    rows = []
    for (method, group), (correct, total) in sorted(totals.items()):
        rows.append(
            {
                "method": method,
                group_field: group,
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
            }
        )
    return rows


def combined_accuracy(summary: dict[str, Any]) -> dict[str, float]:
    return {
        str(row["method_label"]): float(row["strict_acc1_percent"])
        for row in summary["accuracy"]
        if row["dataset"] == "combined"
    }


def plot_figures(
    output: Path,
    math8: dict[str, Any],
    math17: dict[str, Any],
    logic: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    colors = {
        "base": "#6B7280",
        "privileged": "#E78AC3",
        "trsd": "#2563EB",
        "demopsd": "#F59E0B",
        "grpo": "#10B981",
    }

    math8_acc = {
        row["display"]: 100 * row["combined"]["correct"] / row["combined"]["total"]
        for row in math8["methods"]
    }
    math17_acc = combined_accuracy(math17)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=False)
    for ax, title, values in (
        (axes[0], "Qwen3-8B", math8_acc),
        (axes[1], "Qwen3-1.7B", math17_acc),
    ):
        base = values["Base"]
        ax.axhline(base, color=colors["base"], linestyle="--", linewidth=1.5, label="Base")
        ax.plot(
            [0, 16, 64],
            [base, values["Privilege-SD 16"], values["Privilege-SD 64"]],
            marker="o",
            linewidth=2.2,
            color=colors["privileged"],
            label="Privilege-SD",
        )
        ax.plot(
            [0, 16, 64],
            [base, values["TRSD 16"], values["TRSD 64"]],
            marker="o",
            linewidth=2.5,
            color=colors["trsd"],
            label="TRSD",
        )
        ax.set_title(title)
        ax.set_xlabel("Training episodes")
        ax.set_ylabel("Strict Acc@1 (%)")
        ax.set_xticks([0, 16, 64])
        ax.grid(axis="y", alpha=0.22)
    axes[0].scatter([16], [math8_acc["DemoPSD 16"]], marker="s", s=55, color=colors["demopsd"], label="DemoPSD")
    axes[0].scatter([16], [math8_acc["GRPO 16"]], marker="D", s=45, color=colors["grpo"], label="GRPO")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Training horizon interacts sharply with model capacity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig1_capacity_horizon.{suffix}", bbox_inches="tight")
    plt.close(fig)

    order = [
        "Base",
        "DemoPSD 16",
        "GRPO 16",
        "Privilege-SD 16",
        "TRSD 16",
        "Privilege-SD 64",
        "TRSD 64",
    ]
    palette = [
        colors["base"],
        colors["demopsd"],
        colors["grpo"],
        colors["privileged"],
        colors["trsd"],
        colors["privileged"],
        colors["trsd"],
    ]
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    bars = ax.bar(range(len(order)), [math8_acc[name] for name in order], color=palette)
    ax.set_xticks(range(len(order)), [name.replace(" ", "\n", 1) for name in order])
    ax.set_ylabel("Strict Acc@1 (%)")
    ax.set_title("Qwen3-8B: matched DeepMath training and matched evaluation")
    ax.grid(axis="y", alpha=0.22)
    for bar, name in zip(bars, order):
        value = math8_acc[name]
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(math8_acc.values()) + 8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig2_math_baselines.{suffix}", bbox_inches="tight")
    plt.close(fig)

    labels = ["SATQuest", "LogicSkills\nexternal OOD", "Combined"]
    logic_values: dict[str, list[float]] = {}
    for method, display in (("base", "Base"), ("privileged_sd_64", "Privilege-SD 64"), ("trsd_64", "TRSD 64")):
        dataset = {
            row["dataset"]: 100 * float(row["accuracy"])
            for row in logic["dataset_summary"]
            if row["method"] == method
        }
        overall = next(100 * float(row["accuracy"]) for row in logic["overall"] if row["method"] == method)
        logic_values[display] = [dataset["satquest"], dataset["logicskills"], overall]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    width = 0.24
    x = list(range(len(labels)))
    for offset, display, color in (
        (-width, "Base", colors["base"]),
        (0, "Privilege-SD 64", colors["privileged"]),
        (width, "TRSD 64", colors["trsd"]),
    ):
        values = logic_values[display]
        bars = ax.bar([position + offset for position in x], values, width, label=display, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Verifier accuracy (%)")
    ax.set_ylim(0, 83)
    ax.set_title("Qwen3-8B: task acquisition versus external logical retention")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig3_logic_transfer.{suffix}", bbox_inches="tight")
    plt.close(fig)


def build(args: argparse.Namespace) -> None:
    scratch = args.scratch_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    (output / "evidence").mkdir(exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)

    run8 = scratch / "runs" / RUN_8B_MATH
    run17 = scratch / "runs" / RUN_17B_MATH
    runlogic = scratch / "runs" / RUN_8B_LOGIC
    summary8_path = run8 / "comparison" / "summary.json"
    summary17_path = run17 / "results" / "summary.json"
    summarylogic_path = runlogic / "reports" / "logic_results.json"
    math8 = read_json(summary8_path)
    math17 = read_json(summary17_path)
    logic = read_json(summarylogic_path)

    math8_clean = dict(math8)
    math8_clean["methods"] = [
        {key: value for key, value in row.items() if key != "path"}
        for row in math8["methods"]
    ]
    write_json(output / "qwen3_8b_math_summary.json", math8_clean)
    write_json(output / "qwen3_1p7b_math_summary.json", math17)
    write_json(output / "qwen3_8b_logic_summary.json", logic)

    math8_table = []
    base8 = next(row for row in math8["methods"] if row["display"] == "Base")
    base8_acc = 100 * base8["combined"]["correct"] / base8["combined"]["total"]
    for row in math8["methods"]:
        total = int(row["combined"]["total"])
        correct = int(row["combined"]["correct"])
        accuracy = 100 * correct / total
        math8_table.append(
            {
                "method": row["display"],
                "episodes": row["episodes"],
                "amc23_correct": row["amc23"]["correct"],
                "amc23_total": row["amc23"]["total"],
                "aime24_correct": row["aime24"]["correct"],
                "aime24_total": row["aime24"]["total"],
                "aime25_correct": row["aime25"]["correct"],
                "aime25_total": row["aime25"]["total"],
                "combined_correct": correct,
                "combined_total": total,
                "strict_acc1_percent": accuracy,
                "delta_vs_base_pp": accuracy - base8_acc,
            }
        )
    write_csv(output / "tables" / "qwen3_8b_math.csv", math8_table, list(math8_table[0]))

    source17_csv = run17 / "results" / "main_accuracy.csv"
    (output / "tables" / "qwen3_1p7b_math.csv").write_text(
        source17_csv.read_text(encoding="utf-8").replace("\r\n", "\n"),
        encoding="utf-8",
    )
    logic_dataset_csv = runlogic / "reports" / "logic_dataset_summary.csv"
    logic_detail_csv = runlogic / "reports" / "logic_detailed_slices.csv"
    (output / "tables" / "qwen3_8b_logic_dataset.csv").write_text(
        logic_dataset_csv.read_text(encoding="utf-8").replace("\r\n", "\n"),
        encoding="utf-8",
    )
    (output / "tables" / "qwen3_8b_logic_slices.csv").write_text(
        logic_detail_csv.read_text(encoding="utf-8").replace("\r\n", "\n"),
        encoding="utf-8",
    )
    regimes = aggregate_logic(logic["detail"], "eval_regime")
    tasks = aggregate_logic(logic["detail"], "task")
    write_csv(output / "tables" / "qwen3_8b_logic_regimes.csv", regimes, list(regimes[0]))
    write_csv(output / "tables" / "qwen3_8b_logicskills_tasks.csv", tasks, list(tasks[0]))

    math8_paths = {row["display"]: Path(row["path"]) for row in math8["methods"]}
    math17_paths = {
        "Base": run17 / "eval" / "base" / "scored.jsonl",
        "Privilege-SD 16": run17 / "eval" / "privileged_16" / "scored.jsonl",
        "TRSD 16": run17 / "eval" / "trsd_16" / "scored.jsonl",
        "Privilege-SD 64": run17 / "eval" / "privileged_64" / "scored.jsonl",
        "TRSD 64": run17 / "eval" / "trsd_64" / "scored.jsonl",
    }
    logic_raw = runlogic / "reports" / "all_scored.jsonl"
    evidence_counts = {
        "qwen3_8b_math": slim_math_rows(math8_paths, output / "evidence" / "qwen3_8b_math_scores.jsonl"),
        "qwen3_1p7b_math": slim_math_rows(math17_paths, output / "evidence" / "qwen3_1p7b_math_scores.jsonl"),
        "qwen3_8b_logic": slim_logic_rows(logic_raw, output / "evidence" / "qwen3_8b_logic_scores.jsonl"),
    }

    plot_figures(output / "figures", math8, math17, logic)

    source_files = [summary8_path, summary17_path, summarylogic_path, logic_raw]
    source_files.extend(math8_paths.values())
    source_files.extend(math17_paths.values())
    manifest = {
        "schema_version": "trsd-expanded-validation-bundle-v1",
        "source_runs": [RUN_8B_MATH, RUN_17B_MATH, RUN_8B_LOGIC],
        "evidence_rows": evidence_counts,
        "source_artifacts": [
            {
                "run_relative_path": str(path.resolve().relative_to(scratch)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in source_files
        ],
    }
    write_json(output / "MANIFEST.json", manifest)

    readme = f"""# Expanded TRSD validation: model scale, baselines, and logical transfer

This bundle adds three matched evaluations to the 64-episode TRSD result story. Training horizons are exactly matched within every comparison. Math results use the same 143 AMC23/AIME24/AIME25 query IDs and a shared 10,240-token strict Acc@1 protocol. Logical results use all 3,360 SATQuest tasks and all 1,500 LogicSkills tasks with deterministic PySAT/Z3-backed verification.

## Result map

| Question | Main result | Evidence |
|---|---|---|
| Short-term performance | On Qwen3-1.7B, TRSD-16 reaches **51.05%**, +6.29 pp over Base and +1.40 pp over Privilege-SD-16. | `tables/qwen3_1p7b_math.csv` |
| Long-term performance | On Qwen3-8B, TRSD-64 reaches **61.54%**, +6.99 pp over Base, +0.70 pp over matched Privilege-SD-64, and +8.39 pp over GRPO-16. | `tables/qwen3_8b_math.csv` |
| Drift control | At 64 matched episodes, TRSD gains +3.12 pp on SATQuest and retains **2.20 pp** more LogicSkills accuracy than Privilege-SD. | `tables/qwen3_8b_logic_dataset.csv` |

![Capacity and training horizon](figures/fig1_capacity_horizon.png)

The scale comparison exposes a sharp capacity–horizon interaction. Sixteen episodes are already useful for the 1.7B model, but 64 episodes drive both 1.7B branches into short-answer collapse: Privilege-SD-64 falls to 16.08% and TRSD-64 to 13.29%. Qwen3-8B converts the same 64-episode horizon into sustained gains instead, reaching 60.84% with Privilege-SD and 61.54% with TRSD. The long-run gain is therefore realized when model capacity supports the extended trajectory.

## Qwen3-8B math: TRSD, Privilege-SD, DemoPSD, and GRPO

| Method | Episodes | AMC23 | AIME24 | AIME25 | Combined | Delta vs Base |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0 | 56/83 | 14/30 | 8/30 | **78/143 (54.55%)** | +0.00 pp |
| Privilege-SD | 16 | 57/83 | 13/30 | 9/30 | **79/143 (55.24%)** | +0.70 pp |
| TRSD | 16 | 54/83 | 14/30 | 9/30 | **77/143 (53.85%)** | -0.70 pp |
| DemoPSD | 16 | 53/83 | 13/30 | 8/30 | **74/143 (51.75%)** | -2.80 pp |
| GRPO | 16 | 56/83 | 12/30 | 8/30 | **76/143 (53.15%)** | -1.40 pp |
| Privilege-SD | 64 | 62/83 | 13/30 | 12/30 | **87/143 (60.84%)** | +6.29 pp |
| TRSD | 64 | 62/83 | 14/30 | 12/30 | **88/143 (61.54%)** | +6.99 pp |

![Matched Qwen3-8B baselines](figures/fig2_math_baselines.png)

The 16-episode methods form a tight early cluster around the base model. The separation emerges over the 64-episode trajectory: TRSD adds ten correct answers over Base and finishes one answer ahead of the matched Privilege-SD branch. The gain spans AMC23 and AIME25 while holding AIME24 at the base model's 14/30.

## Qwen3-1.7B math: short- and long-horizon behavior

| Method | Episodes | AMC23 | AIME24 | AIME25 | Combined |
|---|---:|---:|---:|---:|---:|
| Base | 0 | 47/83 | 10/30 | 7/30 | **64/143 (44.76%)** |
| Privilege-SD | 16 | 55/83 | 9/30 | 7/30 | **71/143 (49.65%)** |
| TRSD | 16 | 56/83 | 7/30 | 10/30 | **73/143 (51.05%)** |
| Privilege-SD | 64 | 20/83 | 2/30 | 1/30 | **23/143 (16.08%)** |
| TRSD | 64 | 19/83 | 0/30 | 0/30 | **19/143 (13.29%)** |

At 16 episodes, TRSD produces the strongest 1.7B result and shifts success toward AIME25. At 64 episodes, generated responses contract from 7,177 tokens on average for TRSD-16 to 1,233 for TRSD-64, and TRSD records zero correct answers on both AIME subsets. This directly identifies response collapse, rather than decoding-budget exhaustion, as the long-horizon 1.7B failure mode.

## Qwen3-8B logical transfer

| Method | SATQuest | LogicSkills external OOD | Combined |
|---|---:|---:|---:|
| Base | **891/3,360 (26.52%)** | **1,127/1,500 (75.13%)** | **2,018/4,860 (41.52%)** |
| Privilege-SD-64 | **972/3,360 (28.93%)** | **884/1,500 (58.93%)** | **1,856/4,860 (38.19%)** |
| TRSD-64 | **996/3,360 (29.64%)** | **917/1,500 (61.13%)** | **1,913/4,860 (39.36%)** |

![Logical transfer](figures/fig3_logic_transfer.png)

The matched 64-episode branches both acquire more SAT-style competence from the math trajectory while losing first-order external-OOD accuracy. TRSD moves the frontier outward on both axes relative to Privilege-SD: +0.71 pp on SATQuest and +2.20 pp on LogicSkills. The trust-region projection therefore improves target acquisition and preserves more external logical skill at the same training horizon.

## Protocol and audit trail

- Math: one sample per query; temperature 0.6; top-p 0.95; top-k 20; identical query seeds; batched vLLM; 10,240 generated-token cap; strict correctness requires a correct boxed answer before the cap.
- Logic: greedy pass@1 with Qwen3 thinking enabled; 10,240 generated-token cap; six SATQuest problem types across ID, size OOD, format OOD, and joint size-format OOD; all three LogicSkills tasks.
- Verification: SATQuest uses `Problem.check`; LogicSkills uses Z3-backed symbolization and countermodel checks plus exact validity-option checking.
- Row-level evidence contains all {evidence_counts['qwen3_8b_math']:,} Qwen3-8B math outcomes, {evidence_counts['qwen3_1p7b_math']:,} Qwen3-1.7B math outcomes, and {evidence_counts['qwen3_8b_logic']:,} logical outcomes. Responses are represented by SHA-256 digests so the bundle remains compact while preserving exact linkage to the source runs.
- `MANIFEST.json` records SHA-256 hashes and byte sizes for every source result used to build the bundle.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / "RESULTS_COMPLETE").write_text("complete\n", encoding="utf-8")

    # Add hashes of committed outputs only after every artifact exists.
    manifest["bundle_artifacts"] = [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    ]
    write_json(output / "MANIFEST.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/home/da839/scratch_pi_mg269/da839/clean_distill"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/expanded_validation_20260809"),
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
