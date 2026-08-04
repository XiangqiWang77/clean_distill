#!/usr/bin/env python3
"""Aggregate real CSD suite logs into the paper tables and Figures 2--9."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def accuracy_teacher_gain_retention(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    base = mean(float(row["base_correct"]) for row in rows)
    teacher = mean(float(row["teacher_correct"]) for row in rows)
    student = mean(float(row["distilled_correct"]) for row in rows)
    gain = teacher - base
    return (student - base) / gain if gain > 0 else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_detail(root: Path, pattern: str, method: str) -> list[dict[str, Any]]:
    result = []
    for path in root.glob(pattern):
        relative = path.relative_to(root)
        parts = relative.parts
        try:
            model = next(
                part
                for part in parts
                if part.startswith("qwen") or part.startswith("olmo")
            )
        except StopIteration:
            model = "unknown"
        seed_part = next((part for part in parts if part.startswith("seed_")), "seed_0")
        for row in read_jsonl(path):
            result.append(
                {
                    **row,
                    "model": model,
                    "seed": int(seed_part.split("_", 1)[1]),
                    "method": method,
                }
            )
    return result


def grouped_rows(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    value_fn: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(float(value_fn(row)))
    result = []
    for group_key, values in sorted(
        groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        record = dict(zip(keys, group_key))
        record.update(value=mean(values), n=len(values))
        result.append(record)
    return result


def grouped_seed_statistics(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    value_fn: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    """Macro-average each seed, then report a seed-level 95% t interval."""
    grouped: dict[tuple[Any, ...], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)][int(row.get("seed", 0))].append(
            float(value_fn(row))
        )
    t_critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    result = []
    for group_key, seed_values in sorted(
        grouped.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        seed_means = [mean(values) for _, values in sorted(seed_values.items())]
        center = mean(seed_means)
        if len(seed_means) > 1:
            standard_error = statistics.stdev(seed_means) / math.sqrt(len(seed_means))
            critical = t_critical.get(len(seed_means), 1.96)
            radius = critical * standard_error
        else:
            standard_error = float("nan")
            radius = float("nan")
        record = dict(zip(keys, group_key))
        record.update(
            value=center,
            seed_standard_error=standard_error,
            ci95_low=center - radius,
            ci95_high=center + radius,
            seeds=len(seed_means),
            query_seed_rows=sum(len(values) for values in seed_values.values()),
        )
        result.append(record)
    return result


def load_suite(root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "task1": collect_detail(
            root, "main/*/seed_*/csd_t/eval_task1_fast_teacher.jsonl", "CSD-T"
        ),
        "task2": collect_detail(
            root,
            "main/*/seed_*/csd_sd/eval_task2_clean_distillation.jsonl",
            "CSD-SD",
        ),
        "icl": collect_detail(
            root, "main/*/seed_*/support_icl/eval_support_icl.jsonl", "Support ICL"
        ),
        "head_sgd": collect_detail(
            root, "main/*/seed_*/head_sgd/eval_head_sgd.jsonl", "Head SGD"
        ),
        "support_lora": collect_detail(
            root,
            "main/*/seed_*/support_lora/eval_support_lora.jsonl",
            "Support LoRA",
        ),
        "sc": collect_detail(
            root,
            "main/*/seed_*/self_consistency/eval_task1_fast_teacher.jsonl",
            "Maj@8",
        ),
        "budget": collect_detail(
            root, "budget/*/seed_*/*/eval_task1_fast_teacher.jsonl", "budget"
        ),
        "hindsight": collect_detail(
            root, "hindsight/*/seed_*/eval_task1_fast_teacher.jsonl", "hindsight"
        ),
        "transfer": collect_detail(
            root,
            "transfer/*/seed_*/steps_*/eval_task2_clean_distillation.jsonl",
            "transfer",
        ),
        "sensitivity": collect_detail(
            root,
            "sensitivity/*/seed_*/*/*/eval_task1_fast_teacher.jsonl",
            "sensitivity",
        ),
        "ood_t": collect_detail(
            root, "ood/*/*/seed_*/csd_t/eval_task1_fast_teacher.jsonl", "CSD-T"
        ),
        "ood_sd": collect_detail(
            root,
            "ood/*/*/seed_*/csd_sd/eval_task2_clean_distillation.jsonl",
            "CSD-SD",
        ),
        "ood_icl": collect_detail(
            root,
            "ood/*/*/seed_*/support_icl/eval_support_icl.jsonl",
            "Support ICL",
        ),
        "ood_lora": collect_detail(
            root,
            "ood/*/*/seed_*/support_lora/eval_support_lora.jsonl",
            "Support LoRA",
        ),
    }


def support_rows(root: Path) -> list[dict[str, Any]]:
    result = []
    for path in root.glob("supports/headline/*/seed_*/proposals.jsonl"):
        parts = path.relative_to(root).parts
        model = parts[2]
        seed = int(parts[3].split("_", 1)[1])
        for proposal in read_jsonl(path):
            for candidate in proposal.get("specialization_candidates", []):
                audit = candidate.get("target_disjoint_audit", {})
                result.append(
                    {
                        "model": model,
                        "seed": seed,
                        "query_id": proposal.get("query_id"),
                        "candidate_id": candidate.get("candidate_id"),
                        "verifier_valid": float(
                            bool(candidate.get("verifier_valid", False))
                        ),
                        "literal_overlap_rate": float(
                            audit.get("literal_overlap_rate", 0.0)
                        ),
                        "fourgram_overlap_rate": float(
                            audit.get("fourgram_overlap_rate", 0.0)
                        ),
                        "literal_overlap_count": float(
                            audit.get("literal_overlap_count", 0.0)
                        ),
                        "fourgram_overlap_count": float(
                            audit.get("fourgram_overlap_count", 0.0)
                        ),
                    }
                )
    return result


def build_tables(
    root: Path,
    analysis_dir: Path,
    suite: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> None:
    main_long = []
    for row in suite["task1"]:
        main_long.extend(
            [
                {**row, "method": "Base", "score": float(row["base_correct"])},
                {**row, "method": "CSD-T", "score": float(row["teacher_correct"])},
            ]
        )
        if "privileged_correct" in row:
            main_long.append(
                {
                    **row,
                    "method": "Privileged control",
                    "score": float(row["privileged_correct"]),
                }
            )
    main_long.extend(
        {**row, "score": float(row["distilled_correct"])} for row in suite["task2"]
    )
    main_long.extend({**row, "score": float(row["correct"])} for row in suite["icl"])
    main_long.extend(
        {**row, "score": float(row["correct"])} for row in suite["head_sgd"]
    )
    main_long.extend(
        {**row, "score": float(row["correct"])} for row in suite["support_lora"]
    )
    main_long.extend(
        {**row, "score": float(row["base_majority_at_n"])} for row in suite["sc"]
    )
    main_with_average = main_long + [{**row, "source": "average"} for row in main_long]
    table_main = grouped_seed_statistics(
        main_with_average,
        ("model", "method", "source"),
        lambda row: row["score"],
    )
    write_csv(analysis_dir / "tables/table1_main.csv", table_main)

    budget_records = []
    for path in root.glob("budget/*/seed_*/samples_*/eval_task1_fast_teacher.jsonl"):
        parts = path.relative_to(root).parts
        model, seed, samples = (
            parts[1],
            int(parts[2].split("_")[1]),
            int(parts[3].split("_")[1]),
        )
        for row in read_jsonl(path):
            support_tokens = float(row.get("support_generated_tokens", 0.0))
            budget_records.extend(
                [
                    {
                        "model": model,
                        "seed": seed,
                        "samples": samples,
                        "method": "Base mean",
                        "accuracy": float(row["base_correct"]),
                        "generated_tokens": float(
                            row.get("base_generated_tokens", 0.0)
                        ),
                    },
                    {
                        "model": model,
                        "seed": seed,
                        "samples": samples,
                        "method": "Base majority",
                        "accuracy": float(
                            row.get("base_majority_at_n", row["base_correct"])
                        ),
                        "generated_tokens": float(
                            row.get("base_generated_tokens", 0.0)
                        ),
                    },
                    {
                        "model": model,
                        "seed": seed,
                        "samples": samples,
                        "method": "CSD-T mean",
                        "accuracy": float(row["teacher_correct"]),
                        "generated_tokens": support_tokens
                        + float(row.get("teacher_generated_tokens", 0.0)),
                    },
                    {
                        "model": model,
                        "seed": seed,
                        "samples": samples,
                        "method": "CSD-T majority",
                        "accuracy": float(
                            row.get("teacher_majority_at_n", row["teacher_correct"])
                        ),
                        "generated_tokens": support_tokens
                        + float(row.get("teacher_generated_tokens", 0.0)),
                    },
                ]
            )
    for suite_key, method_label in (
        ("icl", "Support ICL"),
        ("head_sgd", "Head SGD"),
        ("support_lora", "Support LoRA"),
    ):
        for row in suite[suite_key]:
            budget_records.append(
                {
                    "model": row["model"],
                    "seed": row["seed"],
                    "samples": 1,
                    "method": method_label,
                    "accuracy": float(row["correct"]),
                    "generated_tokens": float(row.get("support_generated_tokens", 0.0))
                    + float(row.get("generated_tokens", 0.0)),
                }
            )
    for row in suite["task2"]:
        budget_records.append(
            {
                "model": row["model"],
                "seed": row["seed"],
                "samples": 1,
                "method": "CSD-SD",
                "accuracy": float(row["distilled_correct"]),
                "generated_tokens": float(row.get("support_generated_tokens", 0.0))
                + float(row.get("distillation_rollout_tokens", 0.0))
                + float(row.get("distilled_generated_tokens", 0.0)),
            }
        )
    table_budget = grouped_seed_statistics(
        budget_records,
        ("model", "samples", "method"),
        lambda row: row["accuracy"],
    )
    token_groups = grouped_rows(
        budget_records,
        ("model", "samples", "method"),
        lambda row: row["generated_tokens"],
    )
    token_map = {
        (row["model"], row["samples"], row["method"]): row["value"]
        for row in token_groups
    }
    for row in table_budget:
        row["mean_generated_tokens"] = token_map[
            (row["model"], row["samples"], row["method"])
        ]
    write_csv(analysis_dir / "tables/table2_budget.csv", table_budget)

    supports = support_rows(root)
    table_support = []
    for model in sorted({row["model"] for row in supports}):
        group = [row for row in supports if row["model"] == model]
        table_support.append(
            {
                "model": model,
                "supports": len(group),
                "verifier_valid_rate": mean(row["verifier_valid"] for row in group),
                "literal_overlap_rate": mean(
                    row["literal_overlap_rate"] for row in group
                ),
                "fourgram_overlap_rate": mean(
                    row["fourgram_overlap_rate"] for row in group
                ),
            }
        )
    write_csv(analysis_dir / "tables/table3_support_audit.csv", table_support)

    teacher_table = []
    for model in sorted({row["model"] for row in suite["task1"]}):
        t_rows = [row for row in suite["task1"] if row["model"] == model]
        sd_rows = [row for row in suite["task2"] if row["model"] == model]
        positive_gain = sum(
            max(float(row["target_answer_nll_gain"]), 0.0) for row in t_rows
        )
        total_time = sum(float(row["specialization_seconds"]) for row in t_rows)
        teacher_table.append(
            {
                "model": model,
                "method": "CSD-T",
                "target_nll_gain": mean(
                    float(row["target_answer_nll_gain"]) for row in t_rows
                ),
                "specialization_success_rate": mean(
                    float(row["target_answer_nll_gain"] > 0) for row in t_rows
                ),
                "clean_accuracy_gain": mean(
                    float(row["teacher_correct"]) - float(row["base_correct"])
                    for row in t_rows
                ),
                "HER": mean(
                    float(row.get("hindsight_exposure_rate", 0.0)) for row in t_rows
                ),
                "CAR": mean(
                    float(row.get("clean_advantage_retention", 0.0)) for row in t_rows
                ),
                "CHS": 0.0,
                "answer_flip_rate": 0.0,
                "CPP": 1.0,
                "accuracy_teacher_gain_retention": accuracy_teacher_gain_retention(
                    sd_rows
                ),
                "adaptation_seconds": mean(
                    float(row["specialization_seconds"]) for row in t_rows
                ),
                "peak_memory_bytes": mean(
                    float(row.get("peak_memory_bytes", 0.0)) for row in t_rows
                ),
                "FATE": positive_gain / max(total_time, 1e-12),
            }
        )
        for suite_key, label in (
            ("head_sgd", "Head SGD"),
            ("support_lora", "Support LoRA"),
        ):
            baseline_rows = [row for row in suite[suite_key] if row["model"] == model]
            if not baseline_rows:
                continue
            baseline_positive_gain = sum(
                max(float(row["target_answer_nll_gain"]), 0.0) for row in baseline_rows
            )
            baseline_time = sum(
                float(row["adaptation_seconds"]) for row in baseline_rows
            )
            teacher_table.append(
                {
                    "model": model,
                    "method": label,
                    "target_nll_gain": mean(
                        float(row["target_answer_nll_gain"]) for row in baseline_rows
                    ),
                    "specialization_success_rate": mean(
                        float(row["target_answer_nll_gain"] > 0)
                        for row in baseline_rows
                    ),
                    "clean_accuracy_gain": mean(
                        float(row["correct"]) - float(row["base_correct"])
                        for row in baseline_rows
                    ),
                    "HER": 0.0,
                    "CAR": float("nan"),
                    "CHS": 0.0,
                    "answer_flip_rate": 0.0,
                    "CPP": 1.0,
                    "accuracy_teacher_gain_retention": float("nan"),
                    "adaptation_seconds": mean(
                        float(row["adaptation_seconds"]) for row in baseline_rows
                    ),
                    "peak_memory_bytes": mean(
                        float(row.get("peak_memory_bytes", 0.0))
                        for row in baseline_rows
                    ),
                    "FATE": baseline_positive_gain / max(baseline_time, 1e-12),
                }
            )
        privileged_rows = [row for row in t_rows if "privileged_correct" in row]
        if privileged_rows:
            teacher_table.append(
                {
                    "model": model,
                    "method": "Privileged answer-redacted CoT control",
                    "target_nll_gain": float("nan"),
                    "specialization_success_rate": float("nan"),
                    "clean_accuracy_gain": mean(
                        float(row["privileged_correct"]) - float(row["base_correct"])
                        for row in privileged_rows
                    ),
                    "HER": 1.0,
                    "CAR": 0.0,
                    "CHS": mean(
                        float(row["privileged_counterfactual_jsd"])
                        for row in privileged_rows
                        if "privileged_counterfactual_jsd" in row
                    )
                    if any(
                        "privileged_counterfactual_jsd" in row
                        for row in privileged_rows
                    )
                    else float("nan"),
                    "answer_flip_rate": mean(
                        float(row["privileged_answer_flip_rate"])
                        for row in privileged_rows
                        if "privileged_answer_flip_rate" in row
                    )
                    if any(
                        "privileged_answer_flip_rate" in row for row in privileged_rows
                    )
                    else float("nan"),
                    "CPP": 0.0,
                    "accuracy_teacher_gain_retention": float("nan"),
                    "adaptation_seconds": mean(
                        float(row["privileged_cot_construction_seconds"])
                        for row in privileged_rows
                        if "privileged_cot_construction_seconds" in row
                    )
                    if any(
                        "privileged_cot_construction_seconds" in row
                        for row in privileged_rows
                    )
                    else float("nan"),
                    "peak_memory_bytes": float("nan"),
                    "FATE": float("nan"),
                }
            )
    write_csv(analysis_dir / "tables/table4_teacher_metrics.csv", teacher_table)
    write_csv(
        analysis_dir / "tables/table5_adaptation_scaling.csv",
        [
            row
            for row in teacher_table
            if row["method"] != "Privileged answer-redacted CoT control"
        ],
    )

    sensitivity_table = []
    for path in root.glob("sensitivity/*/seed_*/*/*/eval_task1_fast_teacher.jsonl"):
        parts = path.relative_to(root).parts
        model, seed, sweep, value = (
            parts[1],
            int(parts[2].split("_")[1]),
            parts[3],
            parts[4],
        )
        rows = read_jsonl(path)
        sensitivity_table.append(
            {
                "model": model,
                "seed": seed,
                "variant": sweep,
                "value": value,
                "accuracy": mean(float(row["teacher_correct"]) for row in rows),
                "target_nll_gain": mean(
                    float(row["target_answer_nll_gain"]) for row in rows
                ),
                "adaptation_seconds": mean(
                    float(row["specialization_seconds"]) for row in rows
                ),
            }
        )
    write_csv(
        analysis_dir / "tables/table6_sensitivity_ablation.csv", sensitivity_table
    )
    write_csv(analysis_dir / "tables/table6_sensitivity.csv", sensitivity_table)

    ood_records = [
        {**row, "method": "Base", "score": float(row["base_correct"])}
        for row in suite["ood_t"]
    ]
    for method_key, rows in (("CSD-T", suite["ood_t"]), ("CSD-SD", suite["ood_sd"])):
        for row in rows:
            score = (
                row["teacher_correct"]
                if method_key == "CSD-T"
                else row["distilled_correct"]
            )
            ood_records.append({**row, "method": method_key, "score": float(score)})
    ood_records.extend(
        {**row, "method": "Support ICL", "score": float(row["correct"])}
        for row in suite["ood_icl"]
    )
    ood_records.extend(
        {**row, "method": "Support LoRA", "score": float(row["correct"])}
        for row in suite["ood_lora"]
    )
    write_csv(
        analysis_dir / "tables/table7_ood_generalization.csv",
        grouped_rows(
            ood_records, ("model", "method", "source"), lambda row: row["score"]
        ),
    )

    hyper_rows = []
    for group, values in config.items():
        if isinstance(values, dict):
            for name, value in values.items():
                if not isinstance(value, dict):
                    hyper_rows.append(
                        {"group": group, "hyperparameter": name, "value": value}
                    )
    write_csv(analysis_dir / "tables/table8_hyperparameters.csv", hyper_rows)

    cost_rows = []
    for row in suite["task1"]:
        cost_rows.append(
            {
                "model": row["model"],
                "seed": row["seed"],
                "source": row["source"],
                "support_generation_seconds": row.get(
                    "support_generation_seconds", 0.0
                ),
                "feature_extraction_seconds": row.get(
                    "feature_extraction_seconds", 0.0
                ),
                "closed_form_solve_seconds": row.get("closed_form_solve_seconds", 0.0),
                "specialization_seconds": row.get("specialization_seconds", 0.0),
                "support_generated_tokens": row.get("support_generated_tokens", 0.0),
                "teacher_generated_tokens": row.get("teacher_generated_tokens", 0.0),
            }
        )
    write_csv(analysis_dir / "tables/table9_cost_breakdown.csv", cost_rows)


def plot_figures(
    root: Path, analysis_dir: Path, suite: dict[str, list[dict[str, Any]]]
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Figure generation requires matplotlib (pip install matplotlib)."
        ) from exc

    figure_dir = analysis_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )

    supports = support_rows(root)
    if supports:
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        ax.scatter(
            [100 * row["literal_overlap_rate"] for row in supports],
            [100 * row["fourgram_overlap_rate"] for row in supports],
            s=10,
            alpha=0.28,
        )
        ax.set(
            xlabel="target literal overlap (%)",
            ylabel="target 4-gram overlap (%)",
            title="Support hygiene",
        )
        path = figure_dir / "fig2_support_hygiene.pdf"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        generated.append(str(path))

    if suite["task1"]:
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        x = [
            float(row.get("proposal_fit_target_logit_gain", 0.0))
            for row in suite["task1"]
        ]
        y = [float(row["target_answer_nll_gain"]) for row in suite["task1"]]
        ax.scatter(x, y, s=13, alpha=0.35)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set(
            xlabel="proposal-fit target-logit gain",
            ylabel="post-hoc target NLL gain",
            title="Fit signal vs target transfer",
        )
        path = figure_dir / "fig3_teacher_gain.pdf"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        generated.append(str(path))

        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        method_specs = (
            ("CSD-T", suite["task1"], "specialization_seconds", "o"),
            ("Head SGD", suite["head_sgd"], "adaptation_seconds", "s"),
            ("Support LoRA", suite["support_lora"], "adaptation_seconds", "^"),
        )
        colors = {
            model: color
            for model, color in zip(
                sorted({row["model"] for row in suite["task1"]}),
                ("#2A6FBB", "#B94B4B", "#2D8B57", "#7A55A3"),
            )
        }
        for method, method_rows, time_key, marker in method_specs:
            for model in sorted({row["model"] for row in method_rows}):
                rows = [row for row in method_rows if row["model"] == model]
                ax.scatter(
                    mean(float(row[time_key]) for row in rows),
                    mean(float(row["target_answer_nll_gain"]) for row in rows),
                    s=65,
                    marker=marker,
                    color=colors.get(model),
                    label=f"{model} / {method}",
                )
        ax.set(
            xlabel="specialization seconds/query",
            ylabel="target NLL gain",
            title="Fast-teacher efficiency frontier",
        )
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=6.5, ncol=2)
        path = figure_dir / "fig4_efficiency_frontier.pdf"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        generated.append(str(path))

    budget_points = []
    for path in root.glob("budget/*/seed_*/samples_*/eval_task1_fast_teacher.jsonl"):
        model = path.relative_to(root).parts[1]
        samples = int(path.relative_to(root).parts[3].split("_")[1])
        rows = read_jsonl(path)
        if not rows:
            continue
        budget_points.extend(
            [
                {
                    "samples": samples,
                    "tokens": mean(
                        float(row.get("base_generated_tokens", 0.0)) for row in rows
                    ),
                    "label": f"{model} / Base",
                    "score": mean(
                        float(row.get("base_majority_at_n", row["base_correct"]))
                        for row in rows
                    ),
                },
                {
                    "samples": samples,
                    "tokens": mean(
                        float(row.get("support_generated_tokens", 0.0))
                        + float(row.get("teacher_generated_tokens", 0.0))
                        for row in rows
                    ),
                    "label": f"{model} / CSD-T",
                    "score": mean(
                        float(row.get("teacher_majority_at_n", row["teacher_correct"]))
                        for row in rows
                    ),
                },
            ]
        )
    if budget_points:
        for suite_key, method_label in (
            ("icl", "Support ICL"),
            ("head_sgd", "Head SGD"),
            ("support_lora", "Support LoRA"),
        ):
            for row in suite[suite_key]:
                budget_points.append(
                    {
                        "samples": 1,
                        "tokens": float(row.get("support_generated_tokens", 0.0))
                        + float(row.get("generated_tokens", 0.0)),
                        "label": f"{row['model']} / {method_label}",
                        "score": float(row["correct"]),
                    }
                )
        for row in suite["task2"]:
            budget_points.append(
                {
                    "samples": 1,
                    "tokens": float(row.get("support_generated_tokens", 0.0))
                    + float(row.get("distillation_rollout_tokens", 0.0))
                    + float(row.get("distilled_generated_tokens", 0.0)),
                    "label": f"{row['model']} / CSD-SD",
                    "score": float(row["distilled_correct"]),
                }
            )
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        for label in sorted({point["label"] for point in budget_points}):
            sample_counts = sorted(
                {point["samples"] for point in budget_points if point["label"] == label}
            )
            xs = [
                mean(
                    point["tokens"]
                    for point in budget_points
                    if point["label"] == label and point["samples"] == samples
                )
                for samples in sample_counts
            ]
            ys = [
                mean(
                    point["score"]
                    for point in budget_points
                    if point["label"] == label and point["samples"] == samples
                )
                for samples in sample_counts
            ]
            ax.plot(xs, ys, marker="o", label=label)
        ax.set(
            xlabel="total generated tokens/query",
            ylabel="majority accuracy",
            title="Accuracy--budget frontier",
        )
        ax.legend(frameon=False)
        path = figure_dir / "fig5_accuracy_budget.pdf"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        generated.append(str(path))

    hindsight = suite["hindsight"] or suite["task1"]
    hindsight = [row for row in hindsight if "privileged_correct" in row]
    if hindsight:
        models = sorted({row["model"] for row in hindsight})
        clean_gain = [
            mean(
                float(r["teacher_correct"]) - float(r["base_correct"])
                for r in hindsight
                if r["model"] == m
            )
            for m in models
        ]
        privileged_gain = [
            mean(
                float(r["privileged_correct"]) - float(r["base_correct"])
                for r in hindsight
                if r["model"] == m
            )
            for m in models
        ]
        car = [
            mean(
                float(r.get("clean_advantage_retention", 0.0))
                for r in hindsight
                if r["model"] == m
            )
            for m in models
        ]
        has_counterfactual = all(
            "privileged_counterfactual_jsd" in row
            and "privileged_answer_flip_rate" in row
            for row in hindsight
        )
        chs = (
            [
                mean(
                    float(r["privileged_counterfactual_jsd"])
                    for r in hindsight
                    if r["model"] == m
                )
                for m in models
            ]
            if has_counterfactual
            else [float("nan")] * len(models)
        )
        flips = (
            [
                mean(
                    float(r["privileged_answer_flip_rate"])
                    for r in hindsight
                    if r["model"] == m
                )
                for m in models
            ]
            if has_counterfactual
            else [float("nan")] * len(models)
        )
        x = np.arange(len(models))
        width = 0.36
        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
        axes[0].bar(x - width / 2, clean_gain, width, label="CSD-T clean")
        axes[0].bar(x + width / 2, privileged_gain, width, label="privileged CoT")
        axes[0].set_ylabel("accuracy gain")
        axes[0].set_title("Utility")
        axes[0].legend(frameon=False, fontsize=7)
        axes[1].bar(x, car, color="#2D8B57")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("CAR")
        axes[1].set_title("Clean advantage retention")
        axes[2].bar(x - width / 2, chs, width, label="CHS (JSD)")
        axes[2].bar(x + width / 2, flips, width, label="answer flip")
        axes[2].set_title(
            "Counterfactual sensitivity" if has_counterfactual else "Not measured"
        )
        axes[2].legend(frameon=False, fontsize=7)
        for ax in axes:
            ax.set_xticks(x, models, rotation=28, ha="right")
        fig.suptitle("Hindsight audit: CSD has HER=0 and exact same-prefix parity")
        path = figure_dir / "fig6_hindsight_audit.pdf"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        generated.append(str(path))

    if suite["task1"]:
        values = sorted(
            suite["task1"],
            key=lambda row: float(row.get("proposal_fit_target_logit_gain", 0.0)),
        )
        bins = np.array_split(values, min(10, len(values)))
        xs = [
            mean(float(row.get("proposal_fit_target_logit_gain", 0.0)) for row in group)
            for group in bins
            if len(group)
        ]
        ys = [
            mean(float(float(row["target_answer_nll_gain"]) > 0) for row in group)
            for group in bins
            if len(group)
        ]
        fig, ax = plt.subplots(figsize=(5.4, 3.3))
        ax.plot(xs, ys, marker="o")
        ax.set_ylim(-0.03, 1.03)
        ax.set(
            xlabel="proposal-fit signal (binned)",
            ylabel="Pr(target NLL improves)",
            title="Specialization reliability without a runtime gate",
        )
        path = figure_dir / "fig7_transfer_reliability.pdf"
        legacy = figure_dir / "fig7_gate_calibration.pdf"
        fig.tight_layout()
        fig.savefig(path)
        fig.savefig(legacy)
        plt.close(fig)
        generated.extend([str(path), str(legacy)])

    transfer_points = []
    for path in root.glob(
        "transfer/*/seed_*/steps_*/eval_task2_clean_distillation.jsonl"
    ):
        steps = int(path.relative_to(root).parts[3].split("_")[1])
        for row in read_jsonl(path):
            transfer_points.append(
                (
                    steps,
                    float(row["distilled_correct"]),
                    float(row["base_correct"]),
                    float(row["teacher_correct"]),
                )
            )
    if transfer_points:
        steps = sorted({point[0] for point in transfer_points})
        accuracy = [
            mean(point[1] for point in transfer_points if point[0] == step)
            for step in steps
        ]
        retention = []
        for step in steps:
            group = [point for point in transfer_points if point[0] == step]
            base = mean(point[2] for point in group)
            teacher = mean(point[3] for point in group)
            student = mean(point[1] for point in group)
            retention.append(
                (student - base) / (teacher - base) if teacher > base else float("nan")
            )
        fig, left = plt.subplots(figsize=(5.4, 3.3))
        right = left.twinx()
        left.plot(
            steps, accuracy, marker="o", color="#2A6FBB", label="student accuracy"
        )
        right.plot(
            steps,
            retention,
            marker="s",
            color="#B94B4B",
            label="accuracy retention",
        )
        left.set(
            xlabel="query-local distillation steps",
            ylabel="student accuracy",
            title="Teacher-to-student transfer",
        )
        right.set_ylabel("accuracy teacher-gain retention")
        path = figure_dir / "fig8_distillation_transfer.pdf"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        generated.append(str(path))

    sensitivity_data: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in root.glob("sensitivity/*/seed_*/*/*/eval_task1_fast_teacher.jsonl"):
        parts = path.relative_to(root).parts
        sweep, value_slug = parts[3], parts[4]
        value = float(value_slug.replace("m", "-").replace("p", "."))
        rows = read_jsonl(path)
        sensitivity_data[sweep].append(
            (value, mean(float(row["teacher_correct"]) for row in rows))
        )
    if transfer_points:
        sensitivity_data["distill_steps"].extend(
            (step, mean(point[1] for point in transfer_points if point[0] == step))
            for step in sorted({p[0] for p in transfer_points})
        )
    if sensitivity_data:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
        for ax, sweep in zip(
            axes.flat,
            ("ridge_lambda", "support_tokens", "support_count", "distill_steps"),
        ):
            points = sensitivity_data.get(sweep, [])
            grouped = defaultdict(list)
            for value, score in points:
                grouped[value].append(score)
            xs = sorted(grouped)
            ys = [mean(grouped[x]) for x in xs]
            ax.plot(xs, ys, marker="o")
            if sweep == "ridge_lambda" and xs:
                ax.set_xscale("log")
            ax.set_title(sweep.replace("_", " "))
            ax.set_ylabel("accuracy")
        path = figure_dir / "fig9_sensitivity.pdf"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        generated.append(str(path))
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/clean_self_distill/paper_suite.yaml"),
    )
    parser.add_argument("--output-root", help="Override suite output_root")
    parser.add_argument("--analysis-dir", help="Default: <output_root>/analysis")
    parser.add_argument("--tables-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(args.output_root or config["output_root"])
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    analysis_dir = (
        Path(args.analysis_dir).resolve() if args.analysis_dir else root / "analysis"
    )
    suite = load_suite(root)
    build_tables(root, analysis_dir, suite, config)
    figures = [] if args.tables_only else plot_figures(root, analysis_dir, suite)
    manifest = {
        "output_root": str(root),
        "analysis_dir": str(analysis_dir),
        "row_counts": {key: len(value) for key, value in suite.items()},
        "figures": figures,
    }
    with (analysis_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
