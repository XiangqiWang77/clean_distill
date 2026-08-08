from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "clean_self_distill"
    / "10_final_trsd_report.py"
)
SPEC = importlib.util.spec_from_file_location("final_trsd_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def scored_row(
    query_id: str,
    correct: int | str,
    truncated: bool,
    *,
    source: str = "amc23",
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "problem_sha256": query_id.rjust(64, "0")[-64:],
        "source": source,
        "sample_index": 0,
        "correct": correct,
        "truncated": truncated,
        "generated_tokens": 10 if truncated else 5,
        "behavioral_diagnostics": {},
        "resource_usage": {},
        "parsed_answer": "",
    }


def test_five_method_table_order_contract() -> None:
    assert report.METHOD_ORDER == (
        "base",
        "privileged_16",
        "trsd_16",
        "privileged_64",
        "trsd_64",
    )
    assert report.METHOD_LABELS["trsd_16"] == "TRSD 16"
    assert "trsd_16" in report.MATCHED_INFERENCE_METHODS
    assert report.HISTORICAL_TRSD16_LABEL == "TRSD 16† (historical)"


@pytest.mark.parametrize(
    ("wrong_to_correct", "correct_to_wrong", "expected"),
    [
        (0, 0, 1.0),
        (1, 0, 1.0),
        (5, 1, 0.21875),
        (9, 1, 0.021484375),
        (10, 0, 0.001953125),
        (9, 8, 1.0),
    ],
)
def test_exact_mcnemar_two_sided(
    wrong_to_correct: int, correct_to_wrong: int, expected: float
) -> None:
    assert report.exact_mcnemar_two_sided_p(
        wrong_to_correct, correct_to_wrong
    ) == pytest.approx(expected)


def test_paired_bootstrap_is_query_aligned_and_order_invariant() -> None:
    base = [
        scored_row("q1", "1", False),
        scored_row("q2", "0", False),
        scored_row("q3", "1", False),
        scored_row("q4", "0", False),
    ]
    reversed_copy = list(reversed([dict(row) for row in base]))
    methods = {
        "base": base,
        "privileged_16": reversed_copy,
        "trsd_16": reversed_copy,
        "privileged_64": reversed_copy,
        "trsd_64": reversed_copy,
    }
    bootstrap, outcomes = report.heldout_paired_bootstrap(
        methods, dataset="combined", replicates=200, seed=17
    )
    assert outcomes["base"] == [1, 0, 1, 0]
    assert outcomes["privileged_16"] == outcomes["base"]
    assert set(bootstrap["privileged_16_delta_vs_base"]) == {0.0}
    assert report.ci95(bootstrap["privileged_16_delta_vs_base"]) == (0.0, 0.0)


def test_strict_bootstrap_counts_unfinished_wrong_and_pairs_deltas() -> None:
    base = [
        scored_row("q1", 1, False),
        scored_row("q2", 1, True),
        scored_row("q3", 0, False),
        scored_row("q4", 0, True),
    ]
    method = [
        scored_row("q1", 1, True),
        scored_row("q2", 1, False),
        scored_row("q3", 1, False),
        scored_row("q4", 1, True),
    ]
    methods = {
        "base": base,
        "privileged_16": method,
        "trsd_16": method,
        "privileged_64": method,
        "trsd_64": method,
    }
    bootstrap, outcomes = report.heldout_paired_bootstrap(
        methods, dataset="combined", replicates=1_000, seed=17
    )
    assert outcomes["base"] == [1, 0, 0, 0]
    assert outcomes["privileged_16"] == [0, 1, 1, 0]
    assert sum(outcomes["base"]) / 4 == 0.25
    assert sum(outcomes["privileged_16"]) / 4 == 0.50
    assert report.ci95(bootstrap["base_accuracy"]) == pytest.approx((0.0, 0.75))
    assert report.ci95(bootstrap["privileged_16_accuracy"]) == pytest.approx((0.0, 1.0))
    assert report.ci95(bootstrap["privileged_16_delta_vs_base"]) == pytest.approx(
        (-0.5, 1.0)
    )
    transitions = report.paired_transitions(base, method, "privileged_16")
    assert transitions["wrong_to_correct"] == 2
    assert transitions["correct_to_wrong"] == 1
    assert transitions["discordant_pairs"] == 3
    assert transitions["mcnemar_exact_two_sided_p"] == 1.0


def test_completed_only_is_descriptive_and_can_reverse_the_primary_effect() -> None:
    sources = ("amc23", "amc23", "aime24", "aime25")
    base = [
        scored_row(f"q{index}", correct, False, source=source)
        for index, (correct, source) in enumerate(zip((1, 1, 0, 0), sources), 1)
    ]
    method = [
        scored_row("q1", 1, False, source=sources[0]),
        scored_row("q2", 1, True, source=sources[1]),
        scored_row("q3", 0, True, source=sources[2]),
        scored_row("q4", 0, True, source=sources[3]),
    ]
    base_combined = report.aggregate_scored("base", base)[0]
    method_combined = report.aggregate_scored("trsd_64", method)[0]
    assert base_combined["unfinished_as_wrong_acc1"] == 0.5
    assert base_combined["completed_only_acc1"] == 0.5
    assert method_combined["unfinished_as_wrong_acc1"] == 0.25
    assert method_combined["completed_only_acc1"] == 1.0

    all_truncated = [
        scored_row(f"z{index}", 1, True, source=source)
        for index, source in enumerate(sources, 1)
    ]
    truncated_combined = report.aggregate_scored("trsd_64", all_truncated)[0]
    assert truncated_combined["unfinished_as_wrong_acc1"] == 0.0
    assert truncated_combined["completed_n"] == 0
    assert truncated_combined["completed_only_acc1"] is None
    assert "N/A (0/0)" in report.accuracy_cell(truncated_combined)


def test_historical_trsd16_reference_is_point_only_and_uses_historical_base() -> None:
    sources = ("amc23", "amc23", "aime24", "aime25")
    historical_base = [
        scored_row(f"q{index}", correct, truncated, source=source)
        for index, (correct, truncated, source) in enumerate(
            zip((1, 1, 0, 0), (False, True, False, False), sources), 1
        )
    ]
    historical_trsd = [
        scored_row(f"q{index}", correct, truncated, source=source)
        for index, (correct, truncated, source) in enumerate(
            zip((1, 1, 1, 1), (False, False, False, True), sources), 1
        )
    ]
    rows = report.build_historical_trsd16_reference(
        historical_trsd, historical_base
    )
    combined = rows[0]
    assert combined["historical_base_strict_accuracy"] == 0.25
    assert combined["trsd16_strict_accuracy"] == 0.75
    assert combined["strict_delta_vs_historical_base"] == 0.50
    assert combined["trsd16_completed_only_accuracy"] == 1.0
    assert combined["trsd16_boxed_any_eos_accuracy"] == 1.0
    assert combined["inference_status"] == "point_estimate_only_not_compared_to_current_base"
    markdown = report.historical_reference_markdown(rows)
    assert "without an explicit evaluation-prompt-version" in markdown
    assert "neither receives CI or p-values" in markdown


def test_historical_trsd16_signature_rejects_current_prompt_artifact() -> None:
    historical = [
        {"checkpoint_episode": 16, "evaluation_prompt_version": None},
        {"checkpoint_episode": 16},
    ]
    report.validate_historical_trsd16(historical)
    current = [
        {
            "checkpoint_episode": 16,
            "evaluation_prompt_version": "explicit-generation-budget-v1",
        }
    ]
    with pytest.raises(report.ReportError, match="unexpectedly declares"):
        report.validate_historical_trsd16(current)


def test_current_robustness_includes_current_trsd16() -> None:
    sources = ("amc23", "amc23", "aime24", "aime25")
    base = [
        scored_row(f"q{index}", correct, False, source=source)
        for index, (correct, source) in enumerate(zip((1, 0, 1, 0), sources), 1)
    ]
    comparator = [dict(row) for row in base]
    current = {
        "base": base,
        "privileged_16": comparator,
        "trsd_16": comparator,
        "privileged_64": comparator,
        "trsd_64": comparator,
    }
    rows = report.build_heldout_robustness(current, replicates=100, seed=3)
    assert len(rows) == 4 * len(report.MATCHED_INFERENCE_METHODS)
    assert {row["method"] for row in rows} == set(report.MATCHED_INFERENCE_METHODS)
    assert any(row["method"] == "trsd_16" for row in rows)
