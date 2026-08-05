from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.clean_self_distill.report_timebox_main_table import (
    MainTableError,
    build_main_table_report,
    render_markdown,
)


SOURCES = ("amc23", "aime24", "aime25")
EXPECTED = {source: 2 for source in SOURCES}


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _query_ids() -> list[tuple[str, str]]:
    return [(f"{source}-{index}", source) for source in SOURCES for index in (1, 2)]


def _scored(method: str, episode: int, correct_ids: set[str], *, exact: bool) -> list[dict]:
    rows = []
    for query_id, source in _query_ids():
        resource = {
            "method_end_to_end_seconds": {
                "base": 1.0,
                "clean_sd": 1.5,
                "privileged_sd": 1.4,
            }[method],
            "cuda_peak_memory_allocated_bytes": 8 * 1024**3,
            "process_peak_rss_bytes": 10 * 1024**3,
        }
        rows.append(
            {
                "method": method,
                "checkpoint_episode": episode,
                "profile": "acc1",
                "sample_index": 0,
                "query_id": query_id,
                "source": source,
                "problem_sha256": hashlib.sha256(query_id.encode()).hexdigest(),
                "seed": len(rows),
                "correct": int(query_id in correct_ids),
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": 4096,
                "training_audit": {
                    "teacher_positions": 10,
                    "hindsight_exposed_positions": 0,
                    "compared_positions": 10,
                    "exact_context_positions": 10 if exact else 0,
                },
                "resource_usage": resource,
            }
        )
    return rows


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    base = {"amc23-1", "aime24-1"}
    correct = {
        "base": base,
        "clean16": base | {"amc23-2"},
        "clean32": base | {"amc23-2", "aime25-1"},
        "clean48": base | {"amc23-2", "aime24-2", "aime25-1"},
        "clean64": base | {"amc23-2", "aime24-2", "aime25-1"},
        "privileged16": base | {"amc23-2", "aime25-1"},
        "privileged32": base | {"amc23-2", "aime25-1"},
        "privileged48": base | {"amc23-2", "aime25-1"},
        "privileged64": base | {"amc23-2", "aime25-1", "aime25-2"},
    }
    specs = {
        "base_scored": ("base", 0, True, "base"),
        "clean16_scored": ("clean_sd", 16, True, "clean16"),
        "clean32_scored": ("clean_sd", 32, True, "clean32"),
        "clean48_scored": ("clean_sd", 48, True, "clean48"),
        "clean64_scored": ("clean_sd", 64, True, "clean64"),
        "privileged16_scored": ("privileged_sd", 16, False, "privileged16"),
        "privileged32_scored": ("privileged_sd", 32, False, "privileged32"),
        "privileged48_scored": ("privileged_sd", 48, False, "privileged48"),
        "privileged64_scored": ("privileged_sd", 64, False, "privileged64"),
    }
    result: dict[str, Path] = {}
    for name, (method, episode, exact, pattern) in specs.items():
        result[name] = _write(
            tmp_path / f"{name}.jsonl",
            _scored(method, episode, correct[pattern], exact=exact),
        )

    clean_journal = []
    privileged_journal = []
    proposals = []
    for episode in range(1, 65):
        clean_journal.append(
            {
                "branch": "clean",
                "episode": episode,
                "query_id": f"train-{episode}",
                "episode_seconds": 10.0,
                "ridge_metrics": {"specialization_seconds": 1.0},
            }
        )
        privileged_journal.append(
            {
                "branch": "privileged",
                "episode": episode,
                "query_id": f"train-{episode}",
                "episode_seconds": 12.0,
                "ridge_metrics": {},
            }
        )
        proposals.append(
            {
                "query_id": f"train-{episode}",
                "cost_audit": {"end_to_end_seconds": 2.0},
            }
        )
    result["clean_journal"] = _write(tmp_path / "clean-journal.jsonl", clean_journal)
    result["privileged_journal"] = _write(
        tmp_path / "priv-journal.jsonl", privileged_journal
    )
    result["clean_proposals"] = _write(tmp_path / "proposals.jsonl", proposals)
    return result


def _build(paths: dict[str, Path], **kwargs):
    return build_main_table_report(
        **paths,
        expected_source_counts=EXPECTED,
        **kwargs,
    )


def test_builds_three_method_table_and_long_horizon_curve(tmp_path: Path):
    report = _build(_artifacts(tmp_path))
    assert list(report["methods"]) == ["base", "clean64", "privileged64"]
    assert report["methods"]["base"]["accuracy"]["overall"] == pytest.approx(2 / 6)
    assert report["methods"]["clean64"]["STG_S_pp"]["overall"] == pytest.approx(50.0)
    assert all("STG_T_pp" not in method for method in report["methods"].values())
    assert all("retention" not in method for method in report["methods"].values())
    changes = report["methods"]["clean64"]["paired_changes_vs_base"]["overall"]
    assert changes == {"wrong_to_correct": 3, "correct_to_wrong": 0}
    assert report["methods"]["clean64"]["HER"] == 0.0
    assert report["methods"]["clean64"]["CP"] == 1.0
    assert report["methods"]["privileged64"]["CP"] == 0.0
    assert report["methods"]["clean64"]["HFG_pp"]["overall"] == pytest.approx(50.0)
    assert report["methods"]["privileged64"]["HFG_pp"]["overall"] == 0.0
    assert report["first_clean_over_privileged_crossover_episode"]["overall"] == 48
    assert report["long_horizon"]["clean"]["LHG_pp"]["overall"] == pytest.approx(50.0)
    assert report["long_horizon"]["clean"]["discrete_AULC_pp"]["overall"] == pytest.approx(37.5)
    clean_stability = report["long_horizon"]["clean"]["stability"]["overall"]
    assert list(clean_stability["step_deltas_pp"].values()) == pytest.approx(
        [100 / 6, 100 / 6, 100 / 6, 0.0]
    )
    assert clean_stability["negative_step_count"] == 0
    assert clean_stability["largest_drop_pp"] == 0.0
    assert clean_stability["step_std_pp"] == pytest.approx(7.216878364870322)
    assert report["training_costs"]["clean"]["end_to_end_episode_seconds"]["mean"] == 12.0
    assert report["training_costs"]["clean_over_privileged_end_to_end_ratio"] == 1.0
    markdown = render_markdown(report)
    assert "| CSD-T |" not in markdown
    assert "episode-internal mechanism" in markdown
    assert "frontier margin gains" in markdown
    assert "First overall Clean > Privileged checkpoint: 48." in markdown
    assert "| Clean | 16.667 | 16.667 | 16.667 | 0.000 | 0 | 0.000 | 7.217 |" in markdown
    assert "no superiority" in markdown

def test_stability_records_checkpoint_regressions(tmp_path: Path):
    paths = _artifacts(tmp_path)
    base_correct = {"amc23-1", "aime24-1"}
    _write(
        paths["clean32_scored"],
        _scored("clean_sd", 32, base_correct, exact=True),
    )
    report = _build(paths)
    stability = report["long_horizon"]["clean"]["stability"]["overall"]
    assert stability["negative_step_count"] == 1
    assert stability["largest_drop_pp"] == pytest.approx(100 / 6)


def test_rejects_unpaired_query_universe(tmp_path: Path):
    paths = _artifacts(tmp_path)
    rows = [json.loads(line) for line in paths["clean32_scored"].read_text().splitlines()]
    _write(paths["clean32_scored"], rows[:-1])
    with pytest.raises(MainTableError, match="query universe differs"):
        _build(paths)


def test_loads_optional_slurm_resource_summary(tmp_path: Path):
    paths = _artifacts(tmp_path)
    resource = tmp_path / "resource.json"
    resource.write_text(
        json.dumps(
            {
                "slurm_resources": {
                    "summary_by_scope": {
                        "clean": {"peak_MaxRSS_bytes": 20 * 1024**3},
                        "privileged": {"peak_MaxRSS_bytes": 18 * 1024**3},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report = _build(paths, resource_report=resource)
    assert report["slurm_training_resources"]["clean"]["peak_MaxRSS_bytes"] == 20 * 1024**3
    assert "20.00" in render_markdown(report)
