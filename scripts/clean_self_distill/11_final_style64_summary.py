#!/usr/bin/env python3
"""Build the standalone matched P64-vs-TRSD64 style report (CPU only)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


REPORTER_PATH = Path(__file__).with_name("10_final_trsd_report.py")
SPEC = importlib.util.spec_from_file_location("final_trsd_report", REPORTER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import reporter functions from {REPORTER_PATH}")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def interval(values: list[float]) -> tuple[float, float]:
    return report.ci95(values)


def fmt(value: Any, digits: int = 6) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--privileged64-journal", type=Path, required=True)
    parser.add_argument("--trsd64-journal", type=Path, required=True)
    parser.add_argument("--mechanism-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    args = parser.parse_args()

    raw = {
        "privileged": report.load_journal(
            args.privileged64_journal,
            expected_episodes=64,
            name="Privilege-SD 64",
        ),
        "trsd": report.load_journal(
            args.trsd64_journal,
            expected_episodes=64,
            name="TRSD 64",
        ),
    }
    report.match_journals(raw["privileged"], raw["trsd"])
    episodes = {
        method: [report.episode_style_row(row, method) for row in rows]
        for method, rows in raw.items()
    }
    summaries = {
        method: report.aggregate_style(rows) for method, rows in episodes.items()
    }
    bootstrap = report.paired_bootstrap(
        episodes["privileged"],
        episodes["trsd"],
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )

    summary_rows: list[dict[str, Any]] = []
    for method in ("privileged", "trsd"):
        aggregate = summaries[method]
        row: dict[str, Any] = {
            "target": "raw_privileged" if method == "privileged" else "trsd_projected",
            **aggregate,
            "projection_alpha": (
                1.0 if method == "privileged" else aggregate["mean_alpha"]
            ),
            "target_student_kl": (
                aggregate["mean_teacher_student_kl"]
                if method == "privileged"
                else aggregate["mean_achieved_kl"]
            ),
        }
        for metric in ("style_error_per_token", "task_error_per_token", "psr"):
            low, high = interval(bootstrap[f"{method}_{metric}"])
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summary_rows.append(row)

    effect_rows: list[dict[str, Any]] = []
    for metric in ("style_error_per_token", "task_error_per_token", "psr"):
        privileged = float(summaries["privileged"][metric])
        trsd = float(summaries["trsd"][metric])
        delta_low, delta_high = interval(bootstrap[f"delta_{metric}"])
        ratio_low, ratio_high = interval(bootstrap[f"ratio_{metric}"])
        ratio = trsd / privileged
        effect_rows.append(
            {
                "metric": metric,
                "privileged": privileged,
                "trsd": trsd,
                "delta_trsd_minus_privileged": trsd - privileged,
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
                "ratio_trsd_over_privileged": ratio,
                "ratio_ci_low": ratio_low,
                "ratio_ci_high": ratio_high,
                "relative_change": ratio - 1.0,
                "relative_reduction": 1.0 - ratio,
                "relative_reduction_ci_low": 1.0 - ratio_high,
                "relative_reduction_ci_high": 1.0 - ratio_low,
            }
        )

    mechanism = report.mechanism_summary(args.mechanism_csv)
    raw_mechanism, projected_mechanism = mechanism
    pilot = {
        "n_queries": raw_mechanism["n_queries"],
        "n_query_wrappers": raw_mechanism["n_query_wrappers"],
        "raw_style_abs_logprob_shift": raw_mechanism["style_abs_logprob_shift"],
        "projected_style_abs_logprob_shift": projected_mechanism["style_abs_logprob_shift"],
        "style_relative_reduction": 1.0
        - float(projected_mechanism["style_abs_logprob_shift"])
        / float(raw_mechanism["style_abs_logprob_shift"]),
        "raw_task_logprob_gain": raw_mechanism["task_logprob_gain"],
        "projected_task_logprob_gain": projected_mechanism["task_logprob_gain"],
        "projected_alpha": projected_mechanism["mean_alpha"],
        "raw_mean_kl": raw_mechanism["mean_achieved_kl"],
        "projected_mean_kl": projected_mechanism["mean_achieved_kl"],
        "status": "descriptive_same_prefix_three_query_pilot",
    }
    privileged_kl = float(summary_rows[0]["target_student_kl"])
    trsd_kl = float(summary_rows[1]["target_student_kl"])
    privileged_hours = float(summary_rows[0]["training_hours"])
    trsd_hours = float(summary_rows[1]["training_hours"])
    operational_effects = {
        "target_kl_ratio_trsd_over_privileged": trsd_kl / privileged_kl,
        "target_kl_relative_reduction": 1.0 - trsd_kl / privileged_kl,
        "training_hours_ratio_trsd_over_privileged": trsd_hours / privileged_hours,
        "training_hours_relative_change": trsd_hours / privileged_hours - 1.0,
        "trsd_constraint_activation_rate": summary_rows[1][
            "constraint_activation_rate"
        ],
    }

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    report.write_csv(
        root / "matched64_episode_style_metrics.csv",
        [*episodes["privileged"], *episodes["trsd"]],
        list(episodes["privileged"][0]),
    )
    report.write_csv(
        root / "matched64_style_summary.csv", summary_rows, list(summary_rows[0])
    )
    report.write_csv(
        root / "matched64_paired_effects.csv", effect_rows, list(effect_rows[0])
    )
    report.write_csv(
        root / "same_prefix_mechanism_summary.csv", mechanism, list(mechanism[0])
    )

    effect = {row["metric"]: row for row in effect_rows}
    markdown = [
        "# Matched 64-episode style report",
        "",
        "The two journals are paired exactly by episode, query ID, stream index, and problem hash.",
        "",
        "| Target | Style/token [95% CI] | Task/token [95% CI] | PSR [95% CI] | Alpha | Target KL | Steps/no-op | Train h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        markdown.append(
            f"| {row['target']} | {row['style_error_per_token']:.6f} "
            f"[{row['style_error_per_token_ci_low']:.6f}, {row['style_error_per_token_ci_high']:.6f}] | "
            f"{row['task_error_per_token']:.6f} "
            f"[{row['task_error_per_token_ci_low']:.6f}, {row['task_error_per_token_ci_high']:.6f}] | "
            f"{row['psr']:.4f} [{row['psr_ci_low']:.4f}, {row['psr_ci_high']:.4f}] | "
            f"{fmt(row['projection_alpha'], 6)} | {fmt(row['target_student_kl'], 6)} | "
            f"{row['optimizer_steps']}/{row['no_op_episodes']} | {row['training_hours']:.3f} |"
        )
    markdown.extend(
        [
            "",
            "## Paired effects (TRSD minus raw privilege)",
            "",
            "| Metric | Delta [95% CI] | Relative change | Relative reduction [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in ("style_error_per_token", "task_error_per_token", "psr"):
        row = effect[metric]
        markdown.append(
            f"| {metric} | {row['delta_trsd_minus_privileged']:+.6f} "
            f"[{row['delta_ci_low']:+.6f}, {row['delta_ci_high']:+.6f}] | "
            f"{100*row['relative_change']:+.2f}% | {100*row['relative_reduction']:+.2f}% "
            f"[{100*row['relative_reduction_ci_low']:+.2f}, "
            f"{100*row['relative_reduction_ci_high']:+.2f}] |"
        )
    markdown.extend(
        [
            "",
            "Target-to-student KL was reduced by "
            f"{100*operational_effects['target_kl_relative_reduction']:.2f}% "
            f"({privileged_kl:.6f} → {trsd_kl:.6f}); TRSD's trajectory-level "
            f"constraint activated on {100*operational_effects['trsd_constraint_activation_rate']:.2f}% "
            "of episodes. Recorded training time was "
            f"{operational_effects['training_hours_ratio_trsd_over_privileged']:.2f}× "
            "the historical privileged run, noting the runs generated different total token counts.",
            "",
            "## Same-prefix mechanism pilot",
            "",
            f"Across {pilot['n_queries']} queries × neutral/terse/verbose wrappers, "
            f"style shift changed from {pilot['raw_style_abs_logprob_shift']:.6f} to "
            f"{pilot['projected_style_abs_logprob_shift']:.6f} "
            f"({100*pilot['style_relative_reduction']:.2f}% reduction). Task-token "
            f"log-probability gain changed from {pilot['raw_task_logprob_gain']:.6f} to "
            f"{pilot['projected_task_logprob_gain']:.6f}; projected alpha was "
            f"{pilot['projected_alpha']:.6f} and projected KL was "
            f"{pilot['projected_mean_kl']:.6f}.",
            "",
            "The pilot is descriptive because it contains three distinct queries. "
            "The 64-episode confidence intervals use a paired episode/query bootstrap.",
        ]
    )
    payload = {
        "schema_version": "trsd-style64-standalone-report-v1",
        "protocol": {
            "episodes": 64,
            "bootstrap_unit": "paired_episode_query",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "summaries": summary_rows,
        "paired_effects": effect_rows,
        "operational_effects": operational_effects,
        "same_prefix_pilot": pilot,
    }
    report.atomic_text(
        root / "summary.json", json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    report.atomic_text(root / "STYLE_RESULTS.md", "\n".join(markdown) + "\n")
    report.plot_style(root, summaries, bootstrap, mechanism)
    report.atomic_text(root / "REPORT_COMPLETE", "complete\n")


if __name__ == "__main__":
    try:
        main()
    except report.ReportError as error:
        raise SystemExit(f"Refusing style report build: {error}") from error
