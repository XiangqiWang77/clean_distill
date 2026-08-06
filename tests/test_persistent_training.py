from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.clean_self_distill import persistent
from src.clean_self_distill.io import compute_proposal_training_sha256


class TinyTokenizer:
    chat_template = None
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens, return_tensors):
        del add_special_tokens, return_tensors
        values = [1 + (ord(character) % 7) for character in str(text)] or [1]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}

    def decode(self, ids, *, skip_special_tokens=False):
        del skip_special_tokens
        values = ids.tolist() if torch.is_tensor(ids) else list(ids)
        return "".join(str(value) for value in values)

    def convert_ids_to_tokens(self, token_id):
        return {2: "2", 3: "therefore"}.get(int(token_id), "text")

    def save_pretrained(self, path):
        Path(path, "tokenizer_config.json").write_text("{}", encoding="utf-8")


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 6)
        self.adapter_scale = torch.nn.Parameter(torch.tensor(0.05))
        self.embed.weight.requires_grad_(False)

    def forward(
        self,
        *,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
    ):
        del attention_mask, past_key_values, use_cache, return_dict
        embedded = self.embed(input_ids)
        positions = torch.arange(
            1, embedded.shape[1] + 1, device=embedded.device, dtype=embedded.dtype
        ).reshape(1, -1, 1)
        contextual = embedded.cumsum(dim=1) / positions
        hidden = contextual * (1.0 + self.adapter_scale)
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


class TinyPeftModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyDecoder()
        self.lm_head = torch.nn.Linear(6, 16, bias=False)
        self.lm_head.weight.requires_grad_(False)
        self.peft_config = {"default": {}}

    def get_input_embeddings(self):
        return self.model.embed

    def get_output_embeddings(self):
        return self.lm_head

    def save_pretrained(self, path):
        Path(path, "adapter_config.json").write_text("{}", encoding="utf-8")
        torch.save(
            {"adapter_scale": self.model.adapter_scale.detach().cpu()},
            Path(path, "adapter_model.safetensors"),
        )


class FakeRidgeAdapter:
    def __init__(self):
        self.metadata = {}

    def to(self, _device):
        return self

    def apply_to_logits(self, logits, _hidden):
        result = logits.clone()
        result[..., 2] += 0.8
        return result


def _query(index: int) -> dict[str, str]:
    problem = f"Compute {index}+{index}."
    problem_hash = hashlib.sha256(problem.encode("utf-8")).hexdigest()
    return {
        "query_id": f"deepmath:{problem_hash}",
        "problem": problem,
        "problem_sha256": problem_hash,
        "source": "deepmath",
    }


def _proposal(query: dict[str, str]) -> dict:
    row = {
        **query,
        "schema_version": "clean-self-distill-proposals-v5",
        "skill_card": {"skills": ["addition"]},
        "specialization_candidates": [
            {
                "candidate_id": "support",
                "problem": "Compute 8 times 9.",
                "correct_trajectory": [
                    {"step_index": 0, "text": "Multiply 8 by 9 to obtain 72."}
                ],
                "final_answer": "72",
                "wrong_trajectory": [
                    {"step_index": 0, "text": "Add 8 and 9 to obtain 17."}
                ],
                "wrong_final_answer": "17",
                "error_frontier": {
                    "wrong_step_index": 0,
                    "wrong_step_text": "Add 8 and 9 to obtain 17.",
                    "error_explanation": "Addition is not multiplication.",
                    "corrective_action": "Replace addition with multiplication.",
                },
                "solution": "Multiply 8 by 9 to obtain 72.",
            }
        ],
        "specialization_status": "ready",
        "specialization_failure_reason": "",
        "specialization_no_op": False,
        "firewall_audit": {
            "target_answer_loaded": False,
            "target_solution_loaded": False,
            "all_accepted_candidate_artifacts_target_disjoint": True,
        },
    }
    row["proposal_training_sha256"] = compute_proposal_training_sha256(row)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path, count: int):
    queries = [_query(index) for index in range(count)]
    proposals = [_proposal(query) for query in queries]
    query_path = tmp_path / "queries.jsonl"
    proposal_path = tmp_path / "proposals.jsonl"
    _write_jsonl(query_path, queries)
    _write_jsonl(proposal_path, proposals)
    loaded_queries, loaded_proposals, hashes = persistent.load_persistent_inputs(
        query_path, proposal_path, episodes=count
    )
    return loaded_queries, loaded_proposals, hashes


def _config(branch: str, count: int) -> persistent.PersistentConfig:
    return persistent.PersistentConfig(
        branch=branch,
        variant="correct_wrong_signed",
        model="/scratch/model",
        model_id="Qwen/Qwen3-8B",
        revision="pinned",
        episodes=count,
        scientific_checkpoints=tuple(range(count + 1)),
        rolling_checkpoint_interval=1,
        max_sequence_tokens=512,
        max_rollout_tokens=3,
        learning_rate=0.02,
        lora_rank=1,
        lora_alpha=1,
        ridge_max_length=128,
    )


@pytest.fixture
def fake_runtime(monkeypatch):
    def fake_generate(
        model,
        tokenizer,
        problem,
        *,
        adapter,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        seed,
        prompt_override=None,
    ):
        del model, problem, adapter, temperature, top_p, top_k, seed
        assert max_new_tokens >= 3
        prompt_ids = tokenizer(
            prompt_override, add_special_tokens=True, return_tensors="pt"
        )["input_ids"]
        response_ids = torch.tensor([[2, 3, 2]], dtype=torch.long)
        return "2 therefore 2", prompt_ids, response_ids

    def fake_fit(*args, **kwargs):
        del args, kwargs
        return FakeRidgeAdapter(), {
            "specialization_status": "ready",
            "specialization_failure_reason": "",
            "specialization_no_op": False,
            "support_tokens": 12.0,
            "decision_boundary_crossing_count": 2.0,
            "decision_boundary_eligible_count": 3.0,
            "decision_boundary_regression_count": 1.0,
            "decision_boundary_regression_eligible_count": 4.0,
        }

    monkeypatch.setattr(persistent, "generate_response", fake_generate)
    monkeypatch.setattr(persistent, "fit_ridge_adapter", fake_fit)


def test_clean_branch_is_persistent_and_publishes_scientific_checkpoints(
    tmp_path: Path, fake_runtime
):
    queries, proposals, hashes = _inputs(tmp_path, 2)
    model = TinyPeftModel()
    initial = float(model.model.adapter_scale.detach())
    output = tmp_path / "clean"
    result = persistent.run_persistent_training(
        model=model,
        tokenizer=TinyTokenizer(),
        queries=queries,
        proposals=proposals,
        config=_config("clean", 2),
        output_dir=output,
        input_hashes=hashes,
    )
    assert result["status"] == "complete"
    assert float(model.model.adapter_scale.detach()) != initial
    rows = persistent._read_jsonl(output / "episodes.jsonl")
    assert [row["episode"] for row in rows] == [1, 2]
    assert all(
        row["audit"]["exact_context_positions"] == row["response_tokens"]
        for row in rows
    )
    assert all(row["audit"]["hindsight_exposed_positions"] == 0 for row in rows)
    assert rows[0]["ridge_metrics"]["db_crossing_count"] == 2.0
    assert rows[0]["ridge_metrics"]["regression_eligible_count"] == 4.0
    assert rows[0]["style_task_error"]["partition_version"] == "rlcsd-style-task-v1"
    assert rows[0]["distill_token_chunk_size"] == _config("clean", 2).distill_token_chunk_size
    assert rows[0]["max_projected_chunk_tokens"] <= rows[0]["distill_token_chunk_size"]
    for episode in range(3):
        checkpoint = output / "checkpoints" / f"episode_{episode:04d}"
        assert (checkpoint / "adapter_config.json").exists()
        manifest = persistent._read_json(checkpoint / "checkpoint_manifest.json")
        assert manifest["checkpoint_episode"] == episode
        assert manifest["branch"] == "clean"
    assert result["cumulative_audit"]["context_parity"] == 1.0
    assert result["cumulative_audit"]["hindsight_exposure_rate"] == 0.0


def test_clean_incompatible_frontier_falls_back_to_explicit_no_op(
    tmp_path: Path, monkeypatch
):
    queries, proposals, _hashes = _inputs(tmp_path, 1)
    proposal = proposals[queries[0]["query_id"]]
    fit_candidate_counts: list[int] = []

    def incompatible_then_no_op(_model, _tokenizer, candidates, **kwargs):
        fit_candidate_counts.append(len(candidates))
        if candidates:
            raise RuntimeError(
                "Correct/wrong frontier tokens were not scored at the same state"
            )
        assert kwargs["specialization_status"] == "insufficient_verified_candidates"
        assert kwargs["specialization_no_op"] is True
        return FakeRidgeAdapter(), {
            "specialization_status": kwargs["specialization_status"],
            "specialization_failure_reason": kwargs[
                "specialization_failure_reason"
            ],
            "specialization_no_op": True,
            "support_tokens": 0.0,
        }

    monkeypatch.setattr(persistent, "fit_ridge_adapter", incompatible_then_no_op)
    model = TinyPeftModel()
    model.train()
    _adapter, metrics = persistent._fit_current_student_teacher(
        model, TinyTokenizer(), proposal, _config("clean", 1)
    )
    assert fit_candidate_counts == [1, 0]
    assert model.training is True
    assert metrics["specialization_no_op"] is True
    assert metrics["ridge_fit_fallback"] is True
    assert metrics["ridge_fit_used_candidate_count"] == 0
    assert "incompatible verified frontier" in metrics["ridge_fit_fallback_reason"]


def test_privileged_branch_is_predecision_and_never_fits_ridge(
    tmp_path: Path, fake_runtime, monkeypatch
):
    queries, proposals, hashes = _inputs(tmp_path, 1)

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("privileged branch must not construct a ridge teacher")

    monkeypatch.setattr(persistent, "fit_ridge_adapter", forbidden_fit)
    output = tmp_path / "privileged"
    result = persistent.run_persistent_training(
        model=TinyPeftModel(),
        tokenizer=TinyTokenizer(),
        queries=queries,
        proposals={},
        config=_config("privileged", 1),
        output_dir=output,
        input_hashes=hashes,
    )
    row = persistent._read_jsonl(output / "episodes.jsonl")[0]
    assert row["audit"]["exact_context_positions"] == 0
    assert row["audit"]["hindsight_exposed_positions"] == 0
    assert row["ridge_metrics"]["applicable"] is False
    assert row["privileged_prompt_version"] == "predecision-reasoning-method-v1"
    assert result["cumulative_audit"]["context_parity"] == 0.0


def test_proposed_privileged_uses_bound_probes_as_label_blind_teacher_context(
    tmp_path: Path, fake_runtime, monkeypatch
):
    queries, proposals, hashes = _inputs(tmp_path, 1)

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("proposed privileged must not construct a ridge teacher")

    monkeypatch.setattr(persistent, "fit_ridge_adapter", forbidden_fit)
    proposal = proposals[queries[0]["query_id"]]
    rendered = persistent.render_self_proposed_probe_context(proposal)
    assert "SANITIZED_SKILL_CARD" in rendered
    assert "Compute 8 times 9." in rendered
    assert "Verified correct trajectory" in rendered
    assert "Observed wrong trajectory" in rendered
    assert "First wrong step: 0" in rendered
    assert "Corrective action: Replace addition with multiplication." in rendered
    bounded, budget = persistent.bound_self_proposed_probe_context(
        TinyTokenizer(), proposal, max_probe_tokens=32
    )
    assert bounded
    assert budget["probe_context_budget_tokens"] == 32
    assert budget["probe_context_unbounded_tokens"] > 32
    assert budget["probe_context_rendered_tokens"] <= 32
    assert budget["probe_context_truncated"] is True

    output = tmp_path / "proposed-privileged"
    config = replace(
        _config("proposed_privileged", 1), max_sequence_tokens=4_096
    )
    result = persistent.run_persistent_training(
        model=TinyPeftModel(),
        tokenizer=TinyTokenizer(),
        queries=queries,
        proposals=proposals,
        config=config,
        output_dir=output,
        input_hashes=hashes,
    )
    row = persistent._read_jsonl(output / "episodes.jsonl")[0]
    assert row["method_id"] == "privileged:self_proposed_probes"
    assert row["privileged_prompt_version"] == "self-proposed-probes-v1"
    assert row["teacher_context_source"] == "self_proposed_probes"
    assert "self_proposed_probes" in row["teacher_context_sources"]
    assert row["student_context_sha256"] != row["teacher_context_sha256"]
    assert row["audit"] == {
        "teacher_positions": 3,
        "hindsight_exposed_positions": 0,
        "compared_positions": 3,
        "exact_context_positions": 0,
        "on_policy_positions": 3,
        "target_answer_loaded": False,
        "target_solution_loaded": False,
        "target_feedback_loaded": False,
        "teacher_context_source": "self_proposed_probes",
    }
    assert row["probe_metrics"]["proposal_training_sha256"] == proposal[
        "proposal_training_sha256"
    ]
    assert row["probe_metrics"]["candidate_count"] == 1
    assert row["probe_metrics"]["probe_context_budget_tokens"] == 768
    assert (
        row["probe_metrics"]["probe_context_rendered_tokens"]
        <= row["probe_metrics"]["probe_context_budget_tokens"]
    )
    assert row["ridge_metrics"]["applicable"] is False
    assert row["style_task_error"]["partition_version"] == "rlcsd-style-task-v1"
    assert result["cumulative_audit"]["hindsight_exposure_rate"] == 0.0
    assert result["cumulative_audit"]["context_parity"] == 0.0


def test_proposed_privileged_rejects_target_label_in_proposal(
    tmp_path: Path, fake_runtime
):
    queries, proposals, hashes = _inputs(tmp_path, 1)
    query_id = queries[0]["query_id"]
    proposals[query_id] = {**proposals[query_id], "target_answer": "0"}
    with pytest.raises(
        persistent.PersistentProtocolError, match="exposes target-level fields"
    ):
        persistent.run_persistent_training(
            model=TinyPeftModel(),
            tokenizer=TinyTokenizer(),
            queries=queries,
            proposals=proposals,
            config=replace(
                _config("proposed_privileged", 1), max_sequence_tokens=4_096
            ),
            output_dir=tmp_path / "forbidden-target-label",
            input_hashes=hashes,
        )


def test_signal_checkpoint_then_exact_resume(tmp_path: Path, fake_runtime):
    queries, proposals, hashes = _inputs(tmp_path, 1)
    config = _config("clean", 1)
    output = tmp_path / "resume"
    controller = persistent.SignalController()
    controller.requested = True
    interrupted = persistent.run_persistent_training(
        model=TinyPeftModel(),
        tokenizer=TinyTokenizer(),
        queries=queries,
        proposals=proposals,
        config=config,
        output_dir=output,
        input_hashes=hashes,
        signal_controller=controller,
    )
    assert interrupted["status"] == "interrupted"
    assert interrupted["completed_episodes"] == 0

    completed = persistent.run_persistent_training(
        model=TinyPeftModel(),
        tokenizer=TinyTokenizer(),
        queries=queries,
        proposals=proposals,
        config=config,
        output_dir=output,
        input_hashes=hashes,
        resume=True,
    )
    assert completed["status"] == "complete"
    assert completed["completed_episodes"] == 1
    assert len(persistent._read_jsonl(output / "episodes.jsonl")) == 1


def test_manifest_only_resume_reinitializes_but_nonempty_journal_fails_closed(
    tmp_path: Path, fake_runtime
):
    queries, proposals, hashes = _inputs(tmp_path, 1)
    config = _config("clean", 1)
    identity = persistent._checkpoint_identity(config, hashes)
    output = tmp_path / "manifest-only"
    output.mkdir()
    persistent._atomic_write_json(
        output / "run_manifest.json",
        {
            "schema_version": persistent.RUN_SCHEMA_VERSION,
            "branch": config.branch,
            "variant": config.variant,
            "method_id": config.method_id,
            "arguments": config.identity_payload(),
            "runtime": {},
            **identity,
        },
    )
    # SIGKILL during mkdtemp population can leave an uncommitted hidden
    # directory. It must not turn a manifest-only start into fake state.
    orphan = output / "checkpoints" / ".episode_0000.crash-window"
    orphan.mkdir(parents=True)
    (orphan / "partial").write_text("not committed", encoding="utf-8")

    completed = persistent.run_persistent_training(
        model=TinyPeftModel(),
        tokenizer=TinyTokenizer(),
        queries=queries,
        proposals=proposals,
        config=config,
        output_dir=output,
        input_hashes=hashes,
        resume=True,
    )
    assert completed["status"] == "complete"
    assert completed["completed_episodes"] == 1

    # The same manifest is not enough to legitimize a journal whose optimizer
    # state was lost. Rewinding that row would silently change the trajectory.
    journal_only = tmp_path / "journal-without-checkpoint"
    journal_only.mkdir()
    shutil.copy2(output / "run_manifest.json", journal_only / "run_manifest.json")
    shutil.copy2(output / "episodes.jsonl", journal_only / "episodes.jsonl")
    with pytest.raises(
        persistent.PersistentProtocolError,
        match="journal exists without a committed restart checkpoint",
    ):
        persistent.run_persistent_training(
            model=TinyPeftModel(),
            tokenizer=TinyTokenizer(),
            queries=queries,
            proposals=proposals,
            config=config,
            output_dir=journal_only,
            input_hashes=hashes,
            resume=True,
        )


def test_orphaned_published_rolling_checkpoint_is_validated_and_repairs_latest(
    tmp_path: Path, fake_runtime, monkeypatch
):
    queries, proposals, hashes = _inputs(tmp_path, 2)
    config = replace(
        _config("clean", 2),
        scientific_checkpoints=(0, 2),
        rolling_checkpoint_interval=1,
    )
    output = tmp_path / "rolling-before-pointer"
    controller = persistent.SignalController()
    original_train_one_episode = persistent.train_one_episode

    def interrupt_after_first_episode(**kwargs):
        row = original_train_one_episode(**kwargs)
        controller.requested = True
        return row

    monkeypatch.setattr(
        persistent, "train_one_episode", interrupt_after_first_episode
    )
    interrupted = persistent.run_persistent_training(
        model=TinyPeftModel(),
        tokenizer=TinyTokenizer(),
        queries=queries,
        proposals=proposals,
        config=config,
        output_dir=output,
        input_hashes=hashes,
        signal_controller=controller,
    )
    assert interrupted["status"] == "interrupted"
    assert interrupted["completed_episodes"] == 1
    rolling = output / "checkpoints" / "rolling_episode_0001"
    assert rolling.is_dir()

    # Model the exact crash window: directory rename succeeded, pointer rename
    # did not. A corrupt newest checkpoint must not be hidden by episode_0000.
    latest = output / "checkpoints" / "LATEST.json"
    latest.unlink()
    manifest_path = rolling / "checkpoint_manifest.json"
    manifest = persistent._read_json(manifest_path)
    persistent._atomic_write_json(
        manifest_path, {**manifest, "journal_prefix_sha256": "0" * 64}
    )
    rows = persistent._read_jsonl(output / "episodes.jsonl")
    identity = persistent._checkpoint_identity(config, hashes)
    with pytest.raises(
        persistent.PersistentProtocolError, match="journal prefix digest mismatch"
    ):
        persistent._find_resume_checkpoint(
            output,
            identity,
            config=config,
            journal_rows=rows,
        )

    persistent._atomic_write_json(manifest_path, manifest)
    selected = persistent._find_resume_checkpoint(
        output,
        identity,
        config=config,
        journal_rows=rows,
    )
    assert selected == rolling
    assert persistent._read_json(latest) == {
        "checkpoint_dir": "rolling_episode_0001",
        "completed_episodes": 1,
        "run_identity_sha256": identity["run_identity_sha256"],
    }

    resumed_stream_indexes = []

    def record_resumed_episode(**kwargs):
        resumed_stream_indexes.append(kwargs["stream_index"])
        return original_train_one_episode(**kwargs)

    monkeypatch.setattr(persistent, "train_one_episode", record_resumed_episode)
    completed = persistent.run_persistent_training(
        model=TinyPeftModel(),
        tokenizer=TinyTokenizer(),
        queries=queries,
        proposals=proposals,
        config=config,
        output_dir=output,
        input_hashes=hashes,
        resume=True,
    )
    assert completed["status"] == "complete"
    assert resumed_stream_indexes == [1]


def test_distill_chunk_size_is_part_of_restart_identity(tmp_path: Path):
    _queries, _proposals, hashes = _inputs(tmp_path, 1)
    config = _config("clean", 1)
    changed = replace(
        config, distill_token_chunk_size=config.distill_token_chunk_size + 1
    )
    assert config.identity_payload() != changed.identity_payload()
    assert (
        persistent._checkpoint_identity(config, hashes)["run_identity_sha256"]
        != persistent._checkpoint_identity(changed, hashes)["run_identity_sha256"]
    )


def test_query_firewall_rejects_physical_target(tmp_path: Path):
    query = {**_query(0), "answer": "0"}
    proposal = _proposal(_query(0))
    query_path = tmp_path / "queries.jsonl"
    proposal_path = tmp_path / "proposals.jsonl"
    _write_jsonl(query_path, [query])
    _write_jsonl(proposal_path, [proposal])
    with pytest.raises(persistent.PersistentProtocolError, match="physically exposes"):
        persistent.load_persistent_inputs(query_path, proposal_path, episodes=1)
