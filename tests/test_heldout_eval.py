import json
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.clean_self_distill.heldout import (
    HeldoutProtocolError,
    expected_prediction_keys,
    load_query_only_manifest,
    paired_sample_seed,
    score_prediction_rows,
    validate_query_only_row,
)


def _load_eval_script_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "clean_self_distill"
        / "05_heldout_eval.py"
    )
    spec = importlib.util.spec_from_file_location("lgsd_heldout_eval", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def query(query_id: str, problem: str, source: str = "amc23") -> dict:
    import hashlib

    return {
        "query_id": query_id,
        "problem": problem,
        "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        "source": source,
    }


def test_query_manifest_physically_rejects_labels(tmp_path: Path):
    row = {**query("q", "Compute 1+1."), "answer": "2"}
    with pytest.raises(HeldoutProtocolError, match="physically exposes"):
        validate_query_only_row(row, context="row")

    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(HeldoutProtocolError):
        load_query_only_manifest(path)


def test_lgsd_checkpoint_audit_fails_closed_on_method_and_kl_direction(
    tmp_path: Path,
):
    module = _load_eval_script_module()
    checkpoint = tmp_path / "episode_0064"
    checkpoint.mkdir()
    manifest = {
        "schema_version": "clean-self-distill-persistent-checkpoint-v1",
        "branch": "clean",
        "method_id": "trsd:exponential_teacher_projection",
        "checkpoint_episode": 64,
        "completed_episodes": 64,
        "model_id": "Qwen/Qwen3-8B",
        "model_revision": "revision",
        "cumulative_audit": {},
    }
    path = checkpoint / "checkpoint_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HeldoutProtocolError, match="not the requested lgsd"):
        module._load_training_audit(
            checkpoint,
            method="lgsd",
            checkpoint_episode=64,
            model_id="Qwen/Qwen3-8B",
            revision="revision",
        )

    manifest["method_id"] = "lgsd:geometric_kl_ball_projection:forward_kl_v1"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HeldoutProtocolError, match="not a forward-KL LGSD"):
        module._load_training_audit(
            checkpoint,
            method="lgsd",
            checkpoint_episode=64,
            model_id="Qwen/Qwen3-8B",
            revision="revision",
        )

    manifest[
        "distillation_kl_direction"
    ] = "projected_teacher_to_student_forward_kl_v1"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert module._load_training_audit(
        checkpoint,
        method="lgsd",
        checkpoint_episode=64,
        model_id="Qwen/Qwen3-8B",
        revision="revision",
    ) == {}


def test_veto_checkpoint_audit_requires_target_identity(tmp_path: Path):
    module = _load_eval_script_module()
    checkpoint = tmp_path / "episode_0064"
    checkpoint.mkdir()
    manifest = {
        "schema_version": "clean-self-distill-persistent-checkpoint-v1",
        "branch": "veto",
        "method_id": "veto:adaptive_target_reformulation:forward_kl_v1",
        "checkpoint_episode": 64,
        "completed_episodes": 64,
        "model_id": "Qwen/Qwen3-8B",
        "model_revision": "revision",
        "cumulative_audit": {},
        "distillation_kl_direction": "adaptive_target_to_student_forward_kl_v1",
    }
    path = checkpoint / "checkpoint_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HeldoutProtocolError, match="formula-faithful"):
        module._load_training_audit(
            checkpoint,
            method="veto",
            checkpoint_episode=64,
            model_id="Qwen/Qwen3-8B",
            revision="revision",
        )

    manifest["target_reformulation"] = "teacher_student_product_of_experts_v1"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert module._load_training_audit(
        checkpoint,
        method="veto",
        checkpoint_episode=64,
        model_id="Qwen/Qwen3-8B",
        revision="revision",
    ) == {}


def test_expected_keys_preserve_global_shard_order():
    rows = [query(f"q{i}", f"Problem {i}") for i in range(3)]
    assert expected_prediction_keys(rows, num_shards=2, shard_index=1) == [
        ("q1", 0),
        ("q1", 1),
        ("q1", 2),
        ("q1", 3),
    ]
    assert paired_sample_seed(7, 2, 3) == 2028
    assert expected_prediction_keys(
        rows, num_shards=2, shard_index=1, sample_count=1
    ) == [("q1", 0)]


def test_offline_scoring_emits_acc1_and_mean4_without_answer(tmp_path: Path):
    item = query("q", "Compute 1+1.")
    predictions = []
    for sample in range(4):
        predictions.append(
            {
                **item,
                "method": "trsd",
                "checkpoint_episode": 250,
                "checkpoint_sha256": "a" * 64,
                "sample_index": sample,
                "seed": sample,
                "response": "\\boxed{2}" if sample < 3 else "\\boxed{3}",
                "generated_tokens": 5,
                "truncated": False,
                "training_audit": {},
            }
        )
    rows = score_prediction_rows(
        predictions,
        {"q": {"answer": "2", "problem_sha256": item["problem_sha256"]}},
    )
    assert len(rows) == 5
    assert [row["correct"] for row in rows if row["profile"] == "mean4"] == [
        1.0,
        1.0,
        1.0,
        0.0,
    ]
    assert [row["correct"] for row in rows if row["profile"] == "acc1"] == [1.0]
    assert all("answer" not in row for row in rows)


def test_scoring_fails_closed_on_missing_sample():
    item = query("q", "Compute 1+1.")
    predictions = [
        {**item, "sample_index": sample, "response": "\\boxed{2}"}
        for sample in range(3)
    ]
    with pytest.raises(HeldoutProtocolError, match="exactly sample indices"):
        score_prediction_rows(
            predictions,
            {"q": {"answer": "2", "problem_sha256": item["problem_sha256"]}},
        )


def test_single_sample_scoring_emits_acc1_only():
    item = query("q", "Compute 1+1.")
    prediction = {
        **item,
        "method": "base",
        "checkpoint_episode": 0,
        "checkpoint_sha256": "base",
        "sample_index": 0,
        "seed": 0,
        "response": "\\boxed{2}",
        "generated_tokens": 5,
        "truncated": False,
        "training_audit": {},
        "resource_usage": {"evaluation_seconds": 1.25},
    }
    rows = score_prediction_rows(
        [prediction],
        {"q": {"answer": "2", "problem_sha256": item["problem_sha256"]}},
        sample_count=1,
    )
    assert len(rows) == 1
    assert rows[0]["profile"] == "acc1"
    assert rows[0]["correct"] == 1.0
    assert rows[0]["resource_usage"]["evaluation_seconds"] == 1.25


def test_scoring_allows_numeric_target_phase_timing_but_rejects_labels():
    item = query("q", "Compute 1+1.")
    prediction = {
        **item,
        "sample_index": 0,
        "response": "\\boxed{2}",
        "training_audit": {"phase_seconds": {"target": 0.25}},
    }
    rows = score_prediction_rows(
        [prediction],
        {"q": {"answer": "2", "problem_sha256": item["problem_sha256"]}},
        sample_count=1,
    )
    assert rows[0]["correct"] == 1.0

    contaminated = {**prediction, "metadata": {"target": "2"}}
    with pytest.raises(HeldoutProtocolError, match="metadata.target"):
        score_prediction_rows(
            [contaminated],
            {"q": {"answer": "2", "problem_sha256": item["problem_sha256"]}},
            sample_count=1,
        )


def test_score_cli_does_not_import_heavy_generation_dependencies(tmp_path: Path):
    item = query("q", "Compute 1+1.")
    prediction = {
        **item,
        "method": "base",
        "checkpoint_episode": 0,
        "checkpoint_sha256": "base",
        "sample_index": 0,
        "seed": 0,
        "response": "\\boxed{2}",
        "generated_tokens": 5,
        "truncated": False,
        "training_audit": {},
    }
    predictions = tmp_path / "predictions.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "scored.jsonl"
    predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    labels.write_text(
        json.dumps(
            {
                "query_id": item["query_id"],
                "problem_sha256": item["problem_sha256"],
                "answer": "2",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "clean_self_distill"
        / "05_heldout_eval.py"
    )
    launcher = textwrap.dedent(
        """
        import builtins
        import runpy
        import sys

        blocked = (
            "torch",
            "peft",
            "src.clean_self_distill.persistent",
            "src.clean_self_distill.runtime",
        )
        original_import = builtins.__import__

        def reject_heavy_generation_imports(name, globals=None, locals=None,
                                            fromlist=(), level=0):
            if any(name == module or name.startswith(module + ".")
                   for module in blocked):
                raise ImportError("generation dependency is unavailable: " + name)
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = reject_heavy_generation_imports
        script = sys.argv[1]
        sys.argv = [script, *sys.argv[2:]]
        runpy.run_path(script, run_name="__main__")
        """
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            launcher,
            str(script),
            "score",
            "--predictions",
            str(predictions),
            "--labels",
            str(labels),
            "--output",
            str(output),
            "--sample-count",
            "1",
        ],
        cwd=script.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    scored = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(scored) == 1
    assert scored[0]["profile"] == "acc1"
    assert scored[0]["correct"] == 1.0
