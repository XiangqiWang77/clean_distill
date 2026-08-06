from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.clean_self_distill.report_timebox_efficiency import (
    TimeboxReportError,
    build_timebox_report,
    render_markdown,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _episode(
    branch: str,
    episode: int,
    seconds: float,
    *,
    crossings: int = 0,
    eligible: int = 0,
    regressions: int = 0,
    regression_eligible: int = 0,
    ridge_seconds: float | None = None,
    comparable: int = 0,
    base_margin: float = 0.0,
    teacher_margin: float = 0.0,
    margin_gain: float = 0.0,
    margin_attainment: int = 0,
) -> dict:
    exact = 10 if branch == "clean" else 0
    ridge = {
        "decision_boundary_crossing_count": crossings,
        "decision_boundary_eligible_count": eligible,
        "decision_boundary_regression_count": regressions,
        "decision_boundary_regression_eligible_count": regression_eligible,
        "frontier_comparable_count": comparable,
        "frontier_margin_base_mean": base_margin,
        "frontier_margin_teacher_mean": teacher_margin,
        "frontier_margin_gain_mean": margin_gain,
        "frontier_target_margin_attainment_count": margin_attainment,
    }
    if ridge_seconds is not None:
        ridge["specialization_seconds"] = ridge_seconds
    return {
        "branch": branch,
        "episode": episode,
        "query_id": f"q{episode}",
        "episode_seconds": seconds,
        "audit": {
            "teacher_positions": 10,
            "hindsight_exposed_positions": 0,
            "compared_positions": 10,
            "exact_context_positions": exact,
        },
        "style_task_error": {
            "style_abs_error_sum": 2.0,
            "style_token_count": 2,
            "task_abs_error_sum": 8.0,
            "task_token_count": 4,
        },
        "ridge_metrics": ridge,
    }


def _fixture(root: Path, proposal_seconds: tuple[float, float] = (2.0, 4.0)) -> Path:
    clean = [
        _episode(
            "clean",
            1,
            10.0,
            crossings=1,
            eligible=2,
            ridge_seconds=1.0,
            comparable=2,
            base_margin=-2.0,
            teacher_margin=1.0,
            margin_gain=3.0,
            margin_attainment=1,
        ),
        _episode(
            "clean",
            2,
            14.0,
            crossings=2,
            eligible=2,
            regressions=1,
            regression_eligible=2,
            ridge_seconds=3.0,
            comparable=2,
            base_margin=-3.0,
            teacher_margin=2.0,
            margin_gain=5.0,
            margin_attainment=2,
        ),
    ]
    proposed_privileged = [
        _episode("proposed_privileged", 1, 10.0),
        _episode("proposed_privileged", 2, 10.0),
    ]
    clean_proposals = [
        {"query_id": f"q{index}", "cost_audit": {"end_to_end_seconds": seconds}}
        for index, seconds in enumerate(proposal_seconds, 1)
    ]
    proposed_privileged_proposals = [
        {"query_id": f"q{index}", "cost_audit": {"end_to_end_seconds": 1.0}}
        for index in (1, 2)
    ]
    _write_jsonl(root / "clean" / "episodes.jsonl", clean)
    _write_jsonl(
        root / "proposed_privileged" / "episodes.jsonl", proposed_privileged
    )
    _write_jsonl(root / "clean" / "online_proposals.jsonl", clean_proposals)
    _write_jsonl(
        root / "proposed_privileged" / "online_proposals.jsonl",
        proposed_privileged_proposals,
    )
    return root


def test_reports_observed_costs_and_blocks_unsupported_slowdown_claim(tmp_path: Path):
    report = build_timebox_report(_fixture(tmp_path / "timebox12h"))
    clean = report["branches"]["clean"]
    assert clean["core_episode_seconds"]["mean"] == pytest.approx(12.0)
    assert clean["end_to_end_episode_seconds"]["mean"] == pytest.approx(15.0)
    assert report["proposal_end_to_end_seconds"]["clean"]["seconds"]["median"] == 3.0
    assert report["proposal_end_to_end_seconds"]["proposed_privileged"]["seconds"][
        "mean"
    ] == 1.0
    assert clean["ridge_specialization_seconds"]["total"] == 4.0
    assert clean["cleanliness"]["HER"] == 0.0
    assert clean["cleanliness"]["CP"] == 1.0
    assert clean["style_task_error"]["style_task_error_ratio"] == 0.5
    assert clean["decision_frontier"]["crossings"] == 3
    assert clean["decision_frontier"]["regressions"] == 1
    assert clean["frontier_margin"]["gain_mean"] == pytest.approx(4.0)
    assert clean["frontier_margin"]["base_mean"] == pytest.approx(-2.5)
    assert clean["frontier_margin"]["teacher_mean"] == pytest.approx(1.5)
    assert clean["frontier_margin"]["target_attainment_rate"] == pytest.approx(0.75)
    comparison = report["comparison"]
    assert comparison[
        "core_slowdown_ratio_clean_over_proposed_privileged"
    ] == pytest.approx(1.2)
    assert comparison[
        "end_to_end_slowdown_ratio_clean_over_proposed_privileged"
    ] == pytest.approx(15 / 11)
    assert comparison["core_within_threshold"] is True
    assert comparison["end_to_end_within_threshold"] is False
    assert comparison["overall_within_threshold"] is False
    assert "no 'not much slower' claim is made" in comparison["statement"]
    assert "End-to-end raw ratio: 1.364x" in render_markdown(report)
    assert "Frontier margin gain: 4.000 mean across 4 comparable frontiers" in render_markdown(report)


def test_allows_slowdown_wording_only_when_core_and_end_to_end_pass(tmp_path: Path):
    report = build_timebox_report(
        _fixture(tmp_path / "timebox12h", proposal_seconds=(0.0, 0.0))
    )
    assert report["comparison"]["overall_within_threshold"] is True
    assert "qualifies as 'not much slower'" in report["comparison"]["statement"]


def test_reads_headerless_sacct_resources_and_summarizes_memory(tmp_path: Path):
    root = _fixture(tmp_path / "timebox12h")
    sacct = tmp_path / "sacct.psv"
    sacct.write_text(
        "123|three-points-clean|COMPLETED|100|2G|gres/gpumem=40G|gres/gpuutil=80|\n"
        "124|proposed-priv-main|COMPLETED|90|1536M|gres/gpumem=32G|gres/gpuutil=70|\n",
        encoding="utf-8",
    )
    resources = build_timebox_report(root, slurm_accounting=sacct)["slurm_resources"]
    assert resources["accounting_rows"] == 2
    assert resources["records_with_resource_fields"][0]["gpumem"] == "40G"
    assert resources["summary_by_scope"]["clean"]["peak_MaxRSS_bytes"] == 2 * 1024**3
    assert resources["summary_by_scope"]["clean"]["peak_gpumem_bytes"] == 40 * 1024**3
    assert resources["summary_by_scope"]["proposed_privileged"][
        "mean_gpuutil_percent"
    ] == 70.0


def test_missing_proposal_prevents_end_to_end_claim(tmp_path: Path):
    root = _fixture(tmp_path / "timebox12h")
    rows = [
        {"query_id": "q1", "cost_audit": {"end_to_end_seconds": 2.0}},
    ]
    _write_jsonl(root / "clean" / "online_proposals.jsonl", rows)
    report = build_timebox_report(root)
    assert report["branches"]["clean"]["end_to_end_episode_seconds"] is None
    assert report["comparison"]["overall_within_threshold"] is False
    assert report["comparison"]["end_to_end_slowdown_ratio_clean_over_privileged"] is None


def test_rejects_impossible_context_counts(tmp_path: Path):
    root = _fixture(tmp_path / "timebox12h")
    rows = [
        _episode("clean", 1, 10.0),
    ]
    rows[0]["audit"]["exact_context_positions"] = 11
    _write_jsonl(root / "clean" / "episodes.jsonl", rows)
    with pytest.raises(TimeboxReportError, match="impossible HER/CP"):
        build_timebox_report(root)
