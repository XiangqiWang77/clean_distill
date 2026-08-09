from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "clean_self_distill"
    / "12_build_table_evidence_bundle.py"
)
SPEC = importlib.util.spec_from_file_location("table_evidence_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def scored_rows(count: int = 143) -> list[dict[str, object]]:
    sources = ["amc23"] * 83 + ["aime24"] * 30 + ["aime25"] * 30
    return [
        {
            "profile": "acc1",
            "query_id": f"q{index:03d}",
            "problem_sha256": f"{index:064x}",
            "source": source,
            "sample_index": 0,
            "correct": int(index in {0, 1}),
            "truncated": index == 0,
            "generated_tokens": 10_240 if index == 0 else 32,
            "behavioral_diagnostics": {},
            "resource_usage": {
                "generation_seconds": 1.0,
                "cuda_peak_memory_allocated_bytes": 1024**3,
            },
        }
        for index, source in enumerate(sources[:count])
    ]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_current_trsd16_is_required_with_stable_bootstrap_offsets() -> None:
    assert bundle.AVAILABLE_METHODS == bundle.METHODS
    assert bundle.SCORED_FILENAME["trsd_16"] == "trsd_ep16_rkl_current.jsonl"
    assert bundle.PREDICTION_DIRECTORY["trsd_16"] == "trsd_ep16_rkl_current"
    assert bundle.CHECKPOINT_EPISODE["trsd_16"] == 16
    assert bundle.PAIRED_BASE_METHODS == (
        "privileged_16",
        "trsd_16",
        "privileged_64",
        "trsd_64",
    )
    assert bundle.PAIRED_BASE_SEED_OFFSET == {
        "privileged_16": 0,
        "privileged_64": 1,
        "trsd_64": 2,
        "trsd_16": 3,
    }


def test_trsd16_scored_input_requires_all_143_rows(tmp_path: Path) -> None:
    scored = tmp_path / "trsd_ep16_rkl_current.jsonl"
    write_jsonl(scored, scored_rows(142))
    with pytest.raises(bundle.EvidenceError, match="expected 143 rows, found 142"):
        bundle.load_scored(scored, "trsd_16")


def test_bundle_fails_before_writing_when_trsd16_is_incomplete(tmp_path: Path) -> None:
    scored_root = tmp_path / "scored"
    write_jsonl(scored_root / "base.jsonl", scored_rows())
    write_jsonl(scored_root / "privileged_ep16.jsonl", scored_rows())
    write_jsonl(
        scored_root / "trsd_ep16_rkl_current.jsonl",
        scored_rows(142),
    )
    output = tmp_path / "out"
    args = argparse.Namespace(
        scored_root=scored_root,
        prediction_root=tmp_path / "predictions",
        out=output,
    )
    with pytest.raises(bundle.EvidenceError, match="trsd_16: expected 143 rows"):
        bundle.build_bundle(args)
    assert not output.exists()


def test_strict_trsd16_accuracy_and_paired_inference_count_truncation_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bundle, "BOOTSTRAP_REPLICATES", 200)
    base = [
        {"query_id": "q1", "source": "amc23", "correct": 1, "truncated": False},
        {"query_id": "q2", "source": "amc23", "correct": 1, "truncated": True},
        {"query_id": "q3", "source": "amc23", "correct": 0, "truncated": False},
        {"query_id": "q4", "source": "amc23", "correct": 0, "truncated": False},
    ]
    trsd16 = [
        {"query_id": "q1", "source": "amc23", "correct": 1, "truncated": True},
        {"query_id": "q2", "source": "amc23", "correct": 1, "truncated": False},
        {"query_id": "q3", "source": "amc23", "correct": 1, "truncated": False},
        {"query_id": "q4", "source": "amc23", "correct": 1, "truncated": True},
    ]
    result = bundle.paired_inference(
        base,
        trsd16,
        reference="base",
        method="trsd_16",
        dataset="combined",
        seed=17,
    )
    assert result["reference_correct"] == 1
    assert result["method_correct"] == 2
    assert result["method_acc1"] == 0.5
    assert result["delta_percentage_points"] == 25.0
    assert result["wrong_to_correct"] == 2
    assert result["correct_to_wrong"] == 1
    assert result["method_label"] == "TRSD 16"
    assert result["estimand"] == "strict_acc1_full_denominator_unfinished_is_wrong"


def test_t16_training_audit_is_exact_prefix_of_t64_journal(tmp_path: Path) -> None:
    journal = tmp_path / "episodes.jsonl"
    rows = [
        {
            "episode": episode,
            "optimizer_step": True,
            "response_tokens": episode,
            "episode_seconds": 2.0,
            "audit": {
                "teacher_positions": 10,
                "on_policy_positions": 10,
                "exact_context_positions": 0,
                "hindsight_exposed_positions": 0,
            },
            "teacher_context_sources": [
                "student_on_policy_prefix",
                "student_centered_exponential_projection",
            ],
            "temporary_teacher_destroyed_after_update": True,
            "resource_usage": {
                "cuda_peak_memory_allocated_bytes": episode * 1024**3,
                "cuda_peak_memory_delta_bytes": episode * 1024**2,
            },
        }
        for episode in range(1, 65)
    ]
    write_jsonl(journal, rows)
    summary = bundle.summarize_journal(journal, 16)
    assert summary["episodes"] == 16
    assert summary["optimizer_steps"] == 16
    assert summary["response_tokens"] == sum(range(1, 17))
    assert summary["teacher_positions"] == 160
    assert summary["hindsight_exposure_rate"] == 0.0
    assert summary["on_policy_prefix_rate"] == 1.0
    assert summary["temporary_teacher_destroyed"] is True
    assert summary["max_gpu_peak_allocated_gib"] == 16.0


def test_source_has_no_trsd16_blank_or_omission_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    forbidden = (
        "trsd-16 is deliberately left blank",
        "trsd-16 is intentionally unreported",
        "trsd-16 omitted from every inference table",
        "not_reported_current_matched_evaluation_intentionally_omitted",
    )
    assert not any(phrase in source for phrase in forbidden)


def test_report_source_has_exactly_three_positive_claims() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count('"sellpoint_id": "S') == 3
    assert "Drift: TRSD projects" in source
    assert "Short-term performance: TRSD-16 preserves" in source
    assert "Long-term performance: TRSD-64 separates" in source
