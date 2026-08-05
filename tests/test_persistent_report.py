"""Contract tests for the persistent empirical-study reporter."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from scripts.clean_self_distill.report_persistent_metrics import (
    CHECKPOINT_SCHEMA_VERSION,
    EPISODE_SCHEMA_VERSION,
    PSR_PARTITION_VERSION,
    PersistentReportError,
    build_persistent_report,
    generate_persistent_report,
)


SOURCES = {"q-amc": "amc23", "q-aime": "aime24"}
EXPECTED_COUNTS = {"amc23": 1, "aime24": 1}
BASE = {"q-amc": [1, 1, 0, 0], "q-aime": [0, 0, 0, 0]}


def _scored_aliases(row: dict) -> list[dict]:
    mean = {**row, "profile": "mean4"}
    if row["sample_index"] == 0:
        return [mean, {**row, "profile": "acc1"}]
    return [mean]


def _heldout_rows() -> list[dict]:
    patterns = {
        ("base", 0): BASE,
        ("clean_sd", 250): {"q-amc": [1, 1, 1, 1], "q-aime": [0, 0, 0, 0]},
        ("clean_sd", 500): {"q-amc": [1, 1, 1, 1], "q-aime": [1, 0, 0, 0]},
        ("clean_sd", 750): {"q-amc": [1, 1, 1, 1], "q-aime": [1, 1, 0, 0]},
        ("clean_sd", 1000): {"q-amc": [1, 1, 1, 1], "q-aime": [1, 1, 1, 0]},
        ("privileged_sd", 250): BASE,
        ("privileged_sd", 500): {"q-amc": [1, 1, 1, 1], "q-aime": [0, 0, 0, 0]},
        ("privileged_sd", 750): {"q-amc": [1, 1, 1, 1], "q-aime": [1, 0, 0, 0]},
        ("privileged_sd", 1000): {"q-amc": [1, 1, 1, 1], "q-aime": [1, 1, 0, 0]},
    }
    rows: list[dict] = []
    for (method, episode), by_query in patterns.items():
        for query_index, (query_id, scores) in enumerate(by_query.items()):
            for sample_index, correct in enumerate(scores):
                row = {
                    "method": method,
                    "checkpoint_episode": episode,
                    "checkpoint_sha256": "base" if method == "base" else method[0] * 64,
                    "query_id": query_id,
                    "source": SOURCES[query_id],
                    "sample_index": sample_index,
                    "seed": query_index * 1009 + sample_index,
                    "correct": correct,
                    "response": f"response-{correct}",
                    "parsed_answer": str(correct),
                }
                rows.extend(_scored_aliases(row))
    return rows


def _audit(method: str) -> dict:
    if method == "base":
        values = (0, 0, 0, 0, 0)
    elif method == "privileged_sd":
        values = (0, 4, 0, 4, 4)
    else:
        values = (0, 4, 4, 4, 4)
    return dict(
        zip(
            (
                "hindsight_exposed_positions",
                "teacher_positions",
                "exact_context_positions",
                "compared_positions",
                "on_policy_positions",
            ),
            values,
        )
    )


def _short_rows() -> list[dict]:
    patterns = {
        "base": BASE,
        "privileged_sd": {"q-amc": [1, 1, 1, 0], "q-aime": [0, 0, 0, 0]},
        "csd_t": {"q-amc": [1, 1, 1, 1], "q-aime": [1, 0, 0, 0]},
        "csd_sd": {"q-amc": [1, 1, 1, 1], "q-aime": [1, 0, 0, 0]},
    }
    seconds = {"base": 0.0, "privileged_sd": 2.0, "csd_t": 3.0, "csd_sd": 3.0}
    rows: list[dict] = []
    for method, by_query in patterns.items():
        for query_index, (query_id, scores) in enumerate(by_query.items()):
            for sample_index, correct in enumerate(scores):
                timing: dict = {}
                if method == "privileged_sd":
                    timing = {
                        "specialization_metrics": {"applicable": False},
                        "distillation_trace": {
                            "episode_seconds": 2.0,
                            "audit": {
                                **_audit(method),
                                "on_policy_positions": 4,
                            },
                            "temporary_teacher_destroyed_after_update": True,
                        },
                    }
                elif method == "csd_t":
                    timing = {
                        "proposal_end_to_end_seconds": 1.25,
                        "specialization_metrics": {
                            "specialization_seconds": 1.75,
                            "feature_extraction_seconds": 1.5,
                            "closed_form_solve_seconds": 0.2,
                        },
                    }
                elif method == "csd_sd":
                    timing = {
                        "proposal_end_to_end_seconds": 1.0,
                        "specialization_metrics": {
                            "specialization_seconds": 0.5,
                            "feature_extraction_seconds": 0.4,
                            "closed_form_solve_seconds": 0.05,
                        },
                        "distillation_trace": {"episode_seconds": 2.0},
                    }
                    timing["distillation_trace"].update(
                        {
                            "audit": {
                                **_audit(method),
                                "on_policy_positions": 4,
                            },
                            "temporary_teacher_destroyed_after_update": True,
                        }
                    )
                rows.extend(
                    _scored_aliases(
                        {
                            "method": method,
                            "query_id": query_id,
                            "source": SOURCES[query_id],
                            "sample_index": sample_index,
                            "seed": query_index * 1009 + sample_index,
                            "correct": correct,
                            "generated_tokens": 100 + sample_index,
                            "truncated": sample_index == 3,
                            "adaptation_seconds": seconds[method],
                            "cleanliness_audit": _audit(method),
                            **timing,
                        }
                    )
                )
    return rows


def _trajectory(teacher_type: str, reward: int) -> dict:
    exposed = 2 if teacher_type == "post_outcome_privilege" else 0
    exact = 2 if teacher_type == "clean_teacher" else 0
    query_id = "q-amc" if reward == 0 else "q-aime"
    return {
        "record_type": "trajectory",
        "teacher_type": teacher_type,
        "query_id": query_id,
        "trajectory_id": f"{query_id}:sample0",
        "reward": reward,
        "teacher_logprob_sum": -1.0 if reward else -4.0,
        "student_logprob_sum": -3.0 if reward else -2.0,
        "token_count": 2,
        "partition_version": PSR_PARTITION_VERSION,
        "style_abs_error_sum": 2.0,
        "style_token_count": 2,
        "task_abs_error_sum": 4.0,
        "task_token_count": 4,
        "training_audit": {
            "teacher_positions": 2,
            "hindsight_exposed_positions": exposed,
            "compared_positions": 2,
            "exact_context_positions": exact,
        },
        "behavioral_diagnostics": {
            "fabricated_reference_hallucination": False,
            "hedging_token_count": 0,
            "response_tokens": 2,
            "mean_entropy": 0.5,
            "truncated": False,
        },
    }


def _mechanism_rows() -> list[dict]:
    rows: list[dict] = []
    for teacher_type in (
        "pre_decision_privilege",
        "post_outcome_privilege",
        "clean_teacher",
    ):
        rows.extend([_trajectory(teacher_type, 0), _trajectory(teacher_type, 1)])
        if teacher_type == "clean_teacher":
            rows.extend(
                [
                    {
                        "record_type": "frontier",
                        "teacher_type": teacher_type,
                        "query_id": "q-amc",
                        "frontier_id": "wrong",
                        "base_margin": -1.0,
                        "teacher_margin": 1.0,
                    },
                    {
                        "record_type": "frontier",
                        "teacher_type": teacher_type,
                        "query_id": "q-aime",
                        "frontier_id": "correct",
                        "base_margin": 1.0,
                        "teacher_margin": -1.0,
                    },
                ]
            )
    return rows


def _ablation_row(variant: str) -> dict:
    signed = variant == "correct_wrong_signed"
    return {
        "query_id": "ablation-q",
        "variant": variant,
        "heldout_query_denominator": 2,
        "runtime_subset_exclusions": {"specialization_no_op": 1},
        "candidate_count": 4,
        "actual_support_tokens": 384,
        "ridge_dimension": 4096,
        "decode_config": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 32768,
            "num_samples": 4,
        },
        "seeds": [0, 1, 2, 3],
        "pre_update_support_target_nll": 1.0 if signed else 1.2,
        "support_objective_logit_gain": 0.8 if signed else 0.3,
        "adaptation_seconds": 2.0,
        "target_samples": [
            {
                "sample_index": sample,
                "base_correct": int(sample == 0),
                "method_correct": int(sample < (3 if signed else 2)),
            }
            for sample in range(4)
        ],
        "trajectories": [
            {
                "reward": reward,
                "teacher_logprob_sum": -1.0 if reward else -4.0,
                "student_logprob_sum": -3.0 if reward else -2.0,
                "token_count": 2,
            }
            for reward in (0, 1)
        ],
        "frontiers": [
            {
                "frontier_id": "wrong",
                "base_margin": -1.0,
                "teacher_margin": 1.0 if signed else -0.5,
            },
            {
                "frontier_id": "correct",
                "base_margin": 1.0,
                "teacher_margin": 0.5,
            },
        ],
    }


def _ablation_rows() -> list[dict]:
    return [_ablation_row("correct_only"), _ablation_row("correct_wrong_signed")]


def _training_rows() -> list[dict]:
    rows: list[dict] = []
    for branch in ("clean", "privileged"):
        for episode in range(1, 1001):
            clean = branch == "clean"
            rows.append(
                {
                    "schema_version": EPISODE_SCHEMA_VERSION,
                    "branch": branch,
                    "variant": "correct_wrong_signed",
                    "method_id": (
                        "clean:correct_wrong_signed"
                        if clean
                        else "privileged:predecision_method"
                    ),
                    "episode": episode,
                    "stream_index": episode - 1,
                    "query_id": f"deepmath-{episode:04d}",
                    "problem_sha256": f"{episode:064x}",
                    "source": "deepmath",
                    "response_tokens": 2,
                    "optimizer_step": True,
                    "temporary_teacher_destroyed_after_update": True,
                    "distillation_loss": 0.1,
                    "student_logprob_sum": -2.0,
                    "student_normalized_logprob": -1.0,
                    "teacher_logprob_sum": -1.0,
                    "teacher_normalized_logprob": -0.5,
                    "style_task_error": {
                        "partition_version": PSR_PARTITION_VERSION,
                        "definition": "fixed fixture",
                        "style_abs_error_sum": 1.0,
                        "style_token_count": 1,
                        "task_abs_error_sum": 2.0,
                        "task_token_count": 2,
                    },
                    "audit": {
                        "teacher_positions": 10,
                        "hindsight_exposed_positions": 0,
                        "compared_positions": 10,
                        "exact_context_positions": 10 if clean else 0,
                        "on_policy_positions": 10 if clean else 0,
                    },
                    "ridge_metrics": {
                        "applicable": clean,
                        "specialization_no_op": False,
                        "support_tokens": 8 if clean else 0,
                        "db_eligible_count": 1 if clean else 0,
                        "db_crossing_count": 1 if clean else 0,
                        "regression_eligible_count": 1 if clean else 0,
                        "regression_count": 0,
                    },
                }
            )
    return rows


def _checkpoint_manifests() -> list[dict]:
    manifests: list[dict] = []
    for branch in ("clean", "privileged"):
        clean = branch == "clean"
        for episode in (0, 250, 500, 750, 1000):
            manifests.append(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "branch": branch,
                    "variant": "correct_wrong_signed",
                    "method_id": (
                        "clean:correct_wrong_signed"
                        if clean
                        else "privileged:predecision_method"
                    ),
                    "checkpoint_episode": episode,
                    "completed_episodes": episode,
                    "model_identity_sha256": "a" * 64,
                    "query_manifest_sha256": "b" * 64,
                    "proposal_manifest_sha256": "c" * 64,
                    "config_sha256": ("d" if clean else "e") * 64,
                    "cumulative_audit": {
                        "teacher_positions": episode * 10,
                        "hindsight_exposed_positions": 0,
                        "compared_positions": episode * 10,
                        "exact_context_positions": episode * (10 if clean else 0),
                        "on_policy_positions": episode * (10 if clean else 0),
                    },
                }
            )
    return manifests


def _build() -> dict:
    return build_persistent_report(
        _heldout_rows(),
        _short_rows(),
        _mechanism_rows(),
        _ablation_rows(),
        training_rows=_training_rows(),
        checkpoint_manifests=_checkpoint_manifests(),
        expected_source_counts=EXPECTED_COUNTS,
    )


def test_authoritative_metrics_are_reconstructed_from_raw_counts():
    report = _build()
    short = report["short_term"]["mean4"]["combined"]
    assert short["STG_T"] == pytest.approx(0.375)
    assert short["STG_S"] == pytest.approx(0.375)
    assert short["retention"] == pytest.approx(1.0)
    assert short["methods"]["csd_sd"]["HER"] == 0.0
    assert short["methods"]["csd_sd"]["CP"] == 1.0
    assert short["methods"]["csd_sd"]["HFG"] == pytest.approx(0.375)
    assert short["methods"]["privileged_sd"]["HFG"] == 0.0
    assert short["methods"]["csd_t"]["mean_generated_tokens"] == pytest.approx(101.5)
    assert short["methods"]["csd_t"]["truncation_count"] == 2
    assert short["methods"]["csd_t"]["truncation_rate"] == pytest.approx(0.25)
    assert short["methods"]["csd_t"]["latency_seconds_per_query"] == pytest.approx(
        {
            "end_to_end_seconds": 3.0,
            "proposal_seconds": 1.25,
            "ridge_specialization_seconds": 1.75,
            "feature_extraction_seconds": 1.5,
            "closed_form_solve_seconds": 0.2,
            "student_distillation_seconds": 0.0,
        }
    )
    assert short["methods"]["csd_sd"]["latency_seconds_per_query"][
        "student_distillation_seconds"
    ] == pytest.approx(1.5)

    long = report["long_horizon"]["mean4"]["combined"]
    assert long["A_0"] == pytest.approx(0.25)
    assert long["branches"]["clean_sd"]["LHG"] == pytest.approx(0.625)
    assert long["branches"]["clean_sd"]["AULC"] == pytest.approx(0.359375)
    assert long["branches"]["clean_sd"]["final_HFG"] == pytest.approx(0.625)
    assert long["branches"]["privileged_sd"]["final_HFG"] == 0.0
    assert long["clean_privilege_crossover_K_star"] == 250

    clean = report["mechanism"]["teachers"]["clean_teacher"]
    assert clean["RLRS"] == pytest.approx(2.0)
    assert clean["HER"] == 0.0
    assert clean["CP"] == 1.0
    assert clean["PSR"] == pytest.approx(1.0)
    assert clean["DBCR"] == 1.0
    assert clean["regression_rate"] == 1.0
    assert report["ablation"]["variants"]["correct_only"]["DBCR"] == 0.0
    assert report["ablation"]["variants"]["correct_wrong_signed"]["DBCR"] == 1.0
    assert report["ablation"]["variants"]["correct_wrong_signed"][
        "mean_pre_update_support_target_nll"
    ] == pytest.approx(1.0)
    assert report["ablation"]["variants"]["correct_wrong_signed"][
        "mean_support_objective_logit_gain"
    ] == pytest.approx(0.8)
    assert report["persistent_training_audit"]["paired_episode_order"] is True
    assert report["persistent_training_audit"]["branches"]["clean"]["completed_episodes"] == 1000
    assert report["persistent_training_audit"]["branches"]["clean"][
        "specialization_no_op_episodes"
    ] == 0


def test_retention_is_null_only_when_teacher_gain_denominator_is_zero():
    rows = _short_rows()
    for row in rows:
        if row["method"] == "csd_t":
            base_match = next(
                item
                for item in rows
                if item["method"] == "base"
                and item["profile"] == row["profile"]
                and item["query_id"] == row["query_id"]
                and item["sample_index"] == row["sample_index"]
            )
            row["correct"] = base_match["correct"]
    report = build_persistent_report(
        _heldout_rows(), rows, _mechanism_rows(), _ablation_rows(),
        training_rows=_training_rows(), checkpoint_manifests=_checkpoint_manifests(),
        expected_source_counts=EXPECTED_COUNTS,
    )
    assert report["short_term"]["acc1"]["combined"]["retention"] is None
    assert report["short_term"]["mean4"]["combined"]["retention"] is None


def test_incomplete_checkpoint_and_unpaired_seed_fail_closed():
    heldout = [
        row
        for row in _heldout_rows()
        if not (row["method"] == "clean_sd" and row["checkpoint_episode"] == 750)
    ]
    with pytest.raises(PersistentReportError, match="checkpoints"):
        build_persistent_report(
            heldout, _short_rows(), _mechanism_rows(), _ablation_rows(),
            training_rows=_training_rows(), checkpoint_manifests=_checkpoint_manifests(),
            expected_source_counts=EXPECTED_COUNTS,
        )

    short = _short_rows()
    for row in short:
        if row["method"] == "csd_sd" and row["query_id"] == "q-amc" and row["sample_index"] == 0:
            row["seed"] += 1
    with pytest.raises(PersistentReportError, match="not paired"):
        build_persistent_report(
            _heldout_rows(), short, _mechanism_rows(), _ablation_rows(),
            training_rows=_training_rows(), checkpoint_manifests=_checkpoint_manifests(),
            expected_source_counts=EXPECTED_COUNTS,
        )


def test_ablation_actual_support_tokens_must_match():
    ablation = _ablation_rows()
    ablation[1]["actual_support_tokens"] += 1
    with pytest.raises(PersistentReportError, match="actual_support_tokens"):
        build_persistent_report(
            _heldout_rows(), _short_rows(), _mechanism_rows(), ablation,
            training_rows=_training_rows(), checkpoint_manifests=_checkpoint_manifests(),
            expected_source_counts=EXPECTED_COUNTS,
        )


def test_partition_version_and_cleanliness_are_enforced():
    mechanism = _mechanism_rows()
    mechanism[0]["partition_version"] = "unregistered"
    with pytest.raises(PersistentReportError, match="partition"):
        build_persistent_report(
            _heldout_rows(), _short_rows(), mechanism, _ablation_rows(),
            training_rows=_training_rows(), checkpoint_manifests=_checkpoint_manifests(),
            expected_source_counts=EXPECTED_COUNTS,
        )

    short = _short_rows()
    next(row for row in short if row["method"] == "csd_t")["cleanliness_audit"][
        "hindsight_exposed_positions"
    ] = 1
    with pytest.raises(PersistentReportError, match="HER=0"):
        build_persistent_report(
            _heldout_rows(), short, _mechanism_rows(), _ablation_rows(),
            training_rows=_training_rows(), checkpoint_manifests=_checkpoint_manifests(),
            expected_source_counts=EXPECTED_COUNTS,
        )


def test_generate_writes_main_table_json_and_csv(tmp_path: Path):
    artifacts = {}
    for name, rows in {
        "training": _training_rows(),
        "heldout": _heldout_rows(),
        "short": _short_rows(),
        "mechanism": _mechanism_rows(),
        "ablation": _ablation_rows(),
    }.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        artifacts[name] = path
    output_json = tmp_path / "main_table.json"
    output_csv = tmp_path / "main_table.csv"
    checkpoint_paths = []
    for index, manifest in enumerate(_checkpoint_manifests()):
        path = tmp_path / f"checkpoint-{index}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        checkpoint_paths.append(path)
    generate_persistent_report(
        training_journal_paths=[artifacts["training"]],
        checkpoint_manifest_paths=checkpoint_paths,
        heldout_paths=[artifacts["heldout"]],
        short_term_paths=[artifacts["short"]],
        mechanism_paths=[artifacts["mechanism"]],
        ablation_paths=[artifacts["ablation"]],
        output_json=output_json,
        output_csv=output_csv,
        expected_source_counts=EXPECTED_COUNTS,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    table = list(csv.DictReader(output_csv.read_text(encoding="utf-8").splitlines()))
    assert payload["validation"]["status"] == "passed"
    assert len(payload["input_sha256"]["heldout"]) == 64
    assert len(table) == 8
    assert {row["profile"] for row in table} == {"acc1", "mean4"}


def test_short_latency_accepts_scorer_preserved_specialization_metrics():
    rows = _short_rows()
    for row in rows:
        if row["method"] == "csd_t":
            row.pop("adaptation_seconds")
            row["proposal_end_to_end_seconds"] = 1.25
            row["specialization_metrics"] = {
                "specialization_seconds": 1.75,
                "feature_extraction_seconds": 1.5,
                "closed_form_solve_seconds": 0.2,
            }
            row["training_audit"] = row.pop("cleanliness_audit")
    report = build_persistent_report(
        _heldout_rows(), rows, _mechanism_rows(), _ablation_rows(),
        training_rows=_training_rows(), checkpoint_manifests=_checkpoint_manifests(),
        expected_source_counts=EXPECTED_COUNTS,
    )
    assert (
        report["short_term"]["mean4"]["combined"]["methods"]["csd_t"]
        ["adaptation_seconds_per_query"]
        == 3.0
    )


def test_short_latency_components_fail_closed_on_inconsistent_total():
    rows = _short_rows()
    for row in rows:
        if row["method"] == "csd_sd":
            row["adaptation_seconds"] = 2.5
    with pytest.raises(PersistentReportError, match="proposal plus episode"):
        build_persistent_report(
            _heldout_rows(), rows, _mechanism_rows(), _ablation_rows(),
            training_rows=_training_rows(), checkpoint_manifests=_checkpoint_manifests(),
            expected_source_counts=EXPECTED_COUNTS,
        )


def test_short_clean_distillation_requires_on_policy_and_destroyed_teacher():
    rows = _short_rows()
    target = next(row for row in rows if row["method"] == "csd_sd")
    target["distillation_trace"]["audit"]["on_policy_positions"] = 3
    target["cleanliness_audit"]["on_policy_positions"] = 3
    with pytest.raises(PersistentReportError, match="on-policy prefixes"):
        build_persistent_report(
            _heldout_rows(), rows, _mechanism_rows(), _ablation_rows(),
            training_rows=_training_rows(), checkpoint_manifests=_checkpoint_manifests(),
            expected_source_counts=EXPECTED_COUNTS,
        )

    rows = _short_rows()
    target = next(row for row in rows if row["method"] == "csd_sd")
    target["distillation_trace"]["temporary_teacher_destroyed_after_update"] = False
    with pytest.raises(PersistentReportError, match="temporary teacher was destroyed"):
        build_persistent_report(
            _heldout_rows(), rows, _mechanism_rows(), _ablation_rows(),
            training_rows=_training_rows(), checkpoint_manifests=_checkpoint_manifests(),
            expected_source_counts=EXPECTED_COUNTS,
        )


def test_persistent_clean_requires_on_policy_and_destroyed_teacher():
    training = _training_rows()
    training[0]["audit"]["on_policy_positions"] = 9
    with pytest.raises(PersistentReportError, match="on-policy prefixes"):
        build_persistent_report(
            _heldout_rows(), _short_rows(), _mechanism_rows(), _ablation_rows(),
            training_rows=training, checkpoint_manifests=_checkpoint_manifests(),
            expected_source_counts=EXPECTED_COUNTS,
        )

    training = _training_rows()
    training[0]["temporary_teacher_destroyed_after_update"] = False
    with pytest.raises(PersistentReportError, match="temporary teacher was destroyed"):
        build_persistent_report(
            _heldout_rows(), _short_rows(), _mechanism_rows(), _ablation_rows(),
            training_rows=training, checkpoint_manifests=_checkpoint_manifests(),
            expected_source_counts=EXPECTED_COUNTS,
        )


def test_short_behavioral_diagnostics_are_required():
    rows = _short_rows()
    rows[0].pop("truncated")
    with pytest.raises(PersistentReportError, match="truncated"):
        build_persistent_report(
            _heldout_rows(), rows, _mechanism_rows(), _ablation_rows(),
            training_rows=_training_rows(), checkpoint_manifests=_checkpoint_manifests(),
            expected_source_counts=EXPECTED_COUNTS,
        )
