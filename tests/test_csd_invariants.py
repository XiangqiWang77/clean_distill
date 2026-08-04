"""Focused regressions for query binding and distillation invariants."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock
import tempfile

import torch

from src.clean_self_distill.io import (
    canonical_json_sha256,
    compute_proposal_training_sha256,
    stable_hash,
)
from src.clean_self_distill.ridge import (
    SparseRidgeAdapter,
    _answer_aware_token_allocations,
    fit_ridge_adapter,
    _positions_with_required,
    _validate_proposal_rows,
)
from src.clean_self_distill.train_eval import (
    _index_proposals_by_hash,
    _long_horizon_window_diagnostics,
    _proposal_for,
    _same_prefix_distillation_terms,
    _teacher_context_sources,
    _validate_clean_proposal_firewall,
    _validate_adapter_manifest_binding,
    _validate_long_horizon_config,
    evaluate,
    same_prefix_distillation_loss,
    per_query_distill_evaluate,
)


class _FakeAdapter:
    def __init__(self, metadata):
        self.metadata = metadata


class _TinyPeft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.0))
        self.embedding = torch.nn.Embedding(8, 1)
        self.embedding.weight.requires_grad_(False)
        self.adapter_enabled = True

    def get_input_embeddings(self):
        return self.embedding

    @contextmanager
    def disable_adapter(self):
        old = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = old


class _TinyOptimizer:
    """Minimal optimizer used to keep this CPU-only invariant test lightweight."""

    def __init__(self, parameters, *, lr, weight_decay):
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    def zero_grad(self, *, set_to_none=False):
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def step(self):
        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                if self.weight_decay:
                    update = update + self.weight_decay * parameter
                parameter.add_(update, alpha=-self.lr)


class _TrackingTeacher:
    live = 0

    def __init__(self):
        type(self).live += 1

    def __del__(self):
        type(self).live -= 1

    def apply_to_logits(self, logits, hidden):
        shift = logits.new_tensor([2.0, 0.0, 0.0])
        return logits + shift


def _runtime_metadata(model: str = "tiny", revision: str = "rev") -> dict:
    return {
        "python_executable": "/test/python",
        "conda_prefix": "/test",
        "torch_overlay": "",
        "torch": "test",
        "torch_module_path": "/test/torch/__init__.py",
        "torch_arch_flags": [],
        "cuda_runtime": None,
        "model": model,
        "requested_model_revision": "",
        "resolved_model_revision": revision,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "slurm_array_task_id": "",
        "gpu_count": 0,
        "gpus": [],
    }


def _bound_proposal(
    query_id: str,
    problem: str,
    source: str = "amc23",
    *,
    ready: bool = True,
) -> dict:
    row = {
        "schema_version": "clean-self-distill-proposals-v5",
        "query_id": query_id,
        "problem": problem,
        "problem_sha256": stable_hash(problem, 64),
        "source": source,
        "skill_card": {
            "domain": "algebra",
            "skills": ["substitution"],
            "reasoning_operators": ["simplify"],
            "difficulty": "medium",
            "constraints": [],
            "target_details_removed": True,
        },
        "specialization_candidates": [
            {
                "candidate_id": "c00",
                "problem": "An independent exercise.",
                "skill_tags": ["algebra"],
                "solution": "A verified derivation.",
                "final_answer": "ok",
                "correct_trajectory": [
                    {"step_index": 0, "text": "A verified derivation."}
                ],
                "wrong_trajectory": [
                    {"step_index": 0, "text": "Use an invalid shortcut."}
                ],
                "wrong_final_answer": "not-ok",
                "error_frontier": {
                    "wrong_step_index": 0,
                    "wrong_step_text": "Use an invalid shortcut.",
                    "error_explanation": "The shortcut is unjustified.",
                    "corrective_action": "Use the verified derivation.",
                    "verifier_valid": True,
                },
                "frontier_verifier_valid": True,
                "verifier_valid": True,
                "verifier_accepted": True,
                "verifier_reason": "valid",
                "target_disjoint_audit": {"safe": True},
            }
        ]
        if ready
        else [],
        "specialization_status": (
            "ready" if ready else "insufficient_verified_candidates"
        ),
        "specialization_failure_reason": "" if ready else "no candidates verified",
        "specialization_no_op": not ready,
    }
    row["proposal_training_sha256"] = compute_proposal_training_sha256(row)
    return row


class CSDInvariantTest(unittest.TestCase):
    def test_clean_ridge_firewall_fails_closed(self):
        proposal = _bound_proposal("q-firewall", "problem")
        with self.assertRaisesRegex(ValueError, "missing its firewall audit"):
            _validate_clean_proposal_firewall(proposal)

        proposal["firewall_audit"] = {
            "target_answer_loaded": False,
            "target_solution_loaded": False,
        }
        _validate_clean_proposal_firewall(proposal)

        contaminated = {**proposal, "answer": "target label"}
        with self.assertRaisesRegex(ValueError, "target-level fields"):
            _validate_clean_proposal_firewall(contaminated)

        ambiguous = {
            **proposal,
            "firewall_audit": {
                "target_answer_loaded": "false",
                "target_solution_loaded": False,
            },
        }
        with self.assertRaisesRegex(ValueError, "target_answer_loaded=false"):
            _validate_clean_proposal_firewall(ambiguous)

    def test_proposal_lookup_rejects_wrong_binding(self):
        record = {
            "query_id": "amc23:0:hash",
            "source": "amc23",
            "problem": "problem one",
            "problem_sha256": stable_hash("problem one", 64),
        }
        wrong = {
            "query_id": record["query_id"],
            "source": "aime24",
            "problem": "problem two",
            "problem_sha256": stable_hash("problem two", 64),
        }
        proposals = {record["query_id"]: wrong}
        with self.assertRaisesRegex(ValueError, "Proposal mismatch"):
            _proposal_for(record, proposals, _index_proposals_by_hash(proposals))

    def test_hash_fallback_requires_unique_source_match(self):
        digest = stable_hash("same problem", 64)
        record = {
            "query_id": "new-id",
            "source": "amc23",
            "problem": "same problem",
            "problem_sha256": digest,
        }
        rows = {
            "old-a": _bound_proposal("old-a", "same problem"),
            "old-b": _bound_proposal("old-b", "same problem"),
        }
        with self.assertRaisesRegex(KeyError, "matches=2"):
            _proposal_for(record, rows, _index_proposals_by_hash(rows))
        rows.pop("old-b")
        self.assertIs(
            _proposal_for(record, rows, _index_proposals_by_hash(rows)), rows["old-a"]
        )

    def test_adapter_tensor_metadata_must_match_manifest(self):
        ridge_config = {"frontier_positive_weight": 8.0}
        manifest = {
            "query_id": "q",
            "problem_sha256": "a" * 64,
            "proposal_training_sha256": "c" * 64,
            "specialization_status": "ready",
            "specialization_failure_reason": "",
            "specialization_no_op": False,
            "uses_all_candidates": True,
            "source": "amc23",
            "model": "Qwen/Qwen3-4B",
            "model_revision": "b" * 40,
            "ridge_config": ridge_config,
            "ridge_config_sha256": canonical_json_sha256(ridge_config),
        }
        metadata = dict(manifest)
        adapter = _FakeAdapter(metadata)
        _validate_adapter_manifest_binding(
            adapter,
            manifest,
            expected_model=manifest["model"],
            expected_revision=manifest["model_revision"],
        )
        for key in (
            "query_id",
            "problem_sha256",
            "proposal_training_sha256",
            "specialization_status",
            "specialization_no_op",
            "uses_all_candidates",
            "source",
            "model",
            "model_revision",
            "ridge_config_sha256",
        ):
            corrupted = dict(metadata)
            corrupted[key] = "wrong"
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "binding mismatch"
            ):
                _validate_adapter_manifest_binding(
                    _FakeAdapter(corrupted),
                    manifest,
                    expected_model=manifest["model"],
                    expected_revision=manifest["model_revision"],
                )

    def test_ridge_proposal_rows_reject_duplicates_and_bad_hash(self):
        row = _bound_proposal("q", "p")
        with self.assertRaisesRegex(ValueError, "Duplicate proposal"):
            _validate_proposal_rows([row, dict(row)], "fixture")
        bad = dict(row, problem_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "invalid problem binding"):
            _validate_proposal_rows([bad], "fixture")

    def test_topk_kl_preserves_other_probability_mass(self):
        student = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
        teacher = torch.tensor([[[10.0, 10.0, 0.0]]])
        for top_k in (1, 2):
            loss = same_prefix_distillation_loss(
                student,
                teacher,
                top_k=top_k,
                temperature=1.0,
                token_clip=0.0,
            )
            self.assertGreater(float(loss.item()), 0.0)
        with self.assertRaisesRegex(ValueError, "temperature"):
            same_prefix_distillation_loss(
                student, teacher, top_k=2, temperature=0.0, token_clip=0.0
            )

    def test_long_horizon_windows_are_exact_and_preserve_loss_gradient(self):
        student = torch.zeros((1, 2050, 3), requires_grad=True)
        teacher = torch.zeros_like(student)
        teacher[..., 0] = 1.0
        teacher_base = torch.zeros_like(student)
        loss, per_token_kl = _same_prefix_distillation_terms(
            student,
            teacher,
            top_k=3,
            temperature=1.0,
            token_clip=0.0,
        )
        windows = _long_horizon_window_diagnostics(
            student,
            teacher,
            teacher_base,
            per_token_kl,
        )
        self.assertEqual(
            [window["token_count"] for window in windows],
            [512, 512, 1024, 2],
        )
        for window in windows:
            self.assertEqual(window["measurement_point"], "pre_update")
            self.assertGreater(
                window["pre_update_mean_teacher_student_kl"], 0.0
            )
            self.assertEqual(
                window["pre_update_teacher_student_top1_agreement"], 1.0
            )
            self.assertAlmostEqual(
                window["pre_update_mean_teacher_base_ridge_shift_l2"], 1.0
            )
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.abs().sum().item()), 0.0)

    def test_long_horizon_config_is_bounded_by_rollout_budget(self):
        self.assertEqual(
            _validate_long_horizon_config(
                SimpleNamespace(
                    long_horizon_min_prefix_tokens=2048,
                    train_max_new_tokens=4096,
                )
            ),
            2048,
        )
        for minimum in (-1, 4097):
            with self.subTest(minimum=minimum), self.assertRaisesRegex(
                ValueError, "0 <= long_horizon"
            ):
                _validate_long_horizon_config(
                    SimpleNamespace(
                        long_horizon_min_prefix_tokens=minimum,
                        train_max_new_tokens=4096,
                    )
                )

    def test_answer_positions_are_always_selected(self):
        required = torch.tensor([17, 18, 19])
        selected = _positions_with_required(100, 8, required)
        self.assertTrue(set(required.tolist()).issubset(set(selected.tolist())))
        self.assertEqual(selected.numel(), 8)

    def test_answer_positions_expand_an_optional_sampling_budget(self):
        required = torch.arange(17, 68)
        selected = _positions_with_required(100, 32, required)
        self.assertTrue(set(required.tolist()).issubset(set(selected.tolist())))
        self.assertEqual(selected.numel(), 51)

    def test_answer_aware_allocation_rebalances_within_global_budget(self):
        # Exact required-position counts from the run05 query that exposed the
        # old uniform [32] * 8 allocation bug.
        required = [4, 2, 51, 13, 34, 11, 47, 12]
        allocations, metadata = _answer_aware_token_allocations(
            required,
            max_support_tokens=256,
            max_tokens_per_candidate=64,
        )
        self.assertEqual(sum(allocations), 256)
        self.assertTrue(
            all(
                allocation >= minimum
                for allocation, minimum in zip(allocations, required)
            )
        )
        self.assertTrue(all(allocation <= 64 for allocation in allocations))
        self.assertEqual(metadata["required_answer_tokens"], 174)
        self.assertFalse(metadata["support_budget_expanded"])
        self.assertEqual(metadata["support_budget_overflow_tokens"], 0)

    def test_answer_aware_allocation_records_required_budget_expansion(self):
        allocations, metadata = _answer_aware_token_allocations(
            [80, 80, 80, 80],
            max_support_tokens=256,
            max_tokens_per_candidate=64,
        )
        self.assertEqual(allocations, [80, 80, 80, 80])
        self.assertEqual(metadata["allocated_support_token_budget"], 320)
        self.assertTrue(metadata["support_budget_expanded"])
        self.assertEqual(metadata["support_budget_overflow_tokens"], 64)

    def test_empty_candidates_create_exact_rank_zero_noop(self):
        model = _TinyPeft()
        adapter, metrics = fit_ridge_adapter(
            model,
            tokenizer=None,
            candidates=[],
            query_id="q-noop",
        )
        logits = torch.randn(2, 3, 8)
        hidden = torch.randn(2, 3, 1)
        adapted = adapter.apply_to_logits(logits, hidden)
        self.assertIs(adapted, logits)
        self.assertEqual(adapter.rank, 0)
        self.assertEqual(adapter.adapted_vocab_size, 0)
        self.assertEqual(metrics["adapter_rank"], 0.0)
        self.assertEqual(metrics["update_frobenius_norm"], 0.0)
        self.assertEqual(
            metrics["specialization_status"], "insufficient_verified_candidates"
        )
        self.assertTrue(metrics["specialization_no_op"])
        self.assertFalse(metrics["uses_all_candidates"])

        proposal = _bound_proposal("q-noop", "problem", ready=False)
        self.assertEqual(
            _teacher_context_sources(proposal, on_policy=False), ["original_query"]
        )
        self.assertEqual(
            _teacher_context_sources(proposal, on_policy=True),
            ["original_query", "student_generated_prefix"],
        )
        contaminated = {
            **proposal,
            "firewall_audit": {
                "target_answer_loaded": True,
                "target_solution_loaded": False,
            },
        }
        self.assertEqual(
            _teacher_context_sources(contaminated, on_policy=False),
            ["original_query", "target_answer"],
        )

    def test_task1_insufficient_specialization_is_base_equivalent(self):
        model = _TinyPeft()
        problem = "Find the answer."
        digest = stable_hash(problem, 64)
        record = {
            "query_id": "amc23:q-task1-noop:hash",
            "problem": problem,
            "problem_sha256": digest,
            "source": "amc23",
            "answer": "1",
            "solution": "",
        }
        proposal = _bound_proposal(record["query_id"], problem, ready=False)
        proposal["cost_audit"] = {
            "total_completion_tokens": 0,
            "total_generation_seconds": 0.1,
            "end_to_end_seconds": 0.2,
        }
        proposal["firewall_audit"] = {
            "target_answer_loaded": False,
            "target_solution_loaded": False,
        }
        adapters = []

        def fake_generate(model, tokenizer, problem, *, adapter, **kwargs):
            adapters.append(adapter)
            if adapter is not None:
                self.assertEqual(adapter.rank, 0)
            return r"Reason. \boxed{1}", torch.tensor([[1, 2]]), torch.tensor([[3]])

        def fake_fit(*unused_args, **unused_kwargs):
            return SparseRidgeAdapter.no_op(1), {
                "specialization_status": "insufficient_verified_candidates",
                "specialization_failure_reason": "no candidates verified",
                "specialization_no_op": True,
                "uses_all_candidates": False,
                "specialization_seconds": 0.0,
                "feature_extraction_seconds": 0.0,
                "closed_form_solve_seconds": 0.0,
                "support_tokens": 0.0,
                "adapted_vocab_size": 0.0,
                "adapter_rank": 0.0,
                "ridge_lambda_effective": 0.0,
                "update_frobenius_norm": 0.0,
                "peak_memory_bytes": 0.0,
                "proposal_fit_target_logit_gain": 0.0,
                "proposal_base_target_nll": 0.0,
            }

        args = SimpleNamespace(
            output_dir="",
            allow_hindsight_exposure=False,
            num_specialization_candidates=None,
            target_score_mode="answer",
            eval_samples=1,
            eval_max_new_tokens=8,
            eval_temperature=0.0,
            top_p=1.0,
            top_k=0,
            seed=0,
            privileged_control=False,
            ridge_lambda=0.1,
            residual_step_size=0.8,
            max_tokens_per_candidate=64,
            max_support_tokens=256,
            hard_negatives=8,
            max_length=4096,
            model="tiny",
            runtime_metadata=_runtime_metadata(),
        )
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            with (
                mock.patch(
                    "src.clean_self_distill.train_eval._fit_current_adapter",
                    new=fake_fit,
                ),
                mock.patch(
                    "src.clean_self_distill.train_eval.generate_response",
                    new=fake_generate,
                ),
                mock.patch(
                    "src.clean_self_distill.train_eval.score_target_completion",
                    return_value=(1.0, 1.0, torch.tensor([[1, 2]])),
                ),
            ):
                rows, _, _ = evaluate(
                    model,
                    SimpleNamespace(eos_token_id=99),
                    [record],
                    {record["query_id"]: proposal},
                    args,
                    stage="task1_fast_teacher",
                    adapter_cache={},
                )
                args.resume = True
                with mock.patch(
                    "src.clean_self_distill.train_eval.generate_response",
                    side_effect=AssertionError(
                        "a complete prefix must perform no generation"
                    ),
                ):
                    resumed_rows, resumed_summary, resumed_audit = evaluate(
                        model,
                        SimpleNamespace(eos_token_id=99),
                        [record],
                        {record["query_id"]: proposal},
                        args,
                        stage="task1_fast_teacher",
                        adapter_cache={},
                    )
        row = rows[0]
        self.assertEqual(resumed_rows, rows)
        self.assertEqual(resumed_audit.comparison_events, 1)
        self.assertEqual(resumed_summary["overall"]["accuracy/base"], 1.0)
        self.assertEqual(len(adapters), 2)
        self.assertIsNone(adapters[0])
        self.assertEqual(adapters[1].rank, 0)
        self.assertEqual(row["base_responses"], row["teacher_responses"])
        self.assertEqual(
            row["base_target_answer_nll"], row["teacher_target_answer_nll"]
        )
        self.assertEqual(
            row["specialization_status"], "insufficient_verified_candidates"
        )
        self.assertTrue(row["specialization_no_op"])
        self.assertFalse(row["uses_all_candidates"])
        self.assertEqual(row["adapter_rank"], 0.0)
        self.assertEqual(row["update_frobenius_norm"], 0.0)
        self.assertEqual(row["hindsight_audit"]["source_counts"], {"original_query": 1})

    def test_task2_destroys_teacher_before_final_eval_and_resets_student(self):
        model = _TinyPeft()
        initial = model.theta.detach().clone()
        problem = "Find the answer."
        digest = stable_hash(problem, 64)
        record = {
            "query_id": "amc23:q0:hash",
            "problem": problem,
            "problem_sha256": digest,
            "source": "amc23",
            "answer": "1",
            "solution": "",
        }
        proposal = _bound_proposal(record["query_id"], problem)
        proposal["cost_audit"] = {
            "total_completion_tokens": 1,
            "total_generation_seconds": 0.1,
            "end_to_end_seconds": 0.2,
        }
        proposal["firewall_audit"] = {
            "target_answer_loaded": False,
            "target_solution_loaded": False,
        }
        call_count = 0

        def fake_generate(model, tokenizer, problem, *, adapter, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 4:
                self.assertEqual(_TrackingTeacher.live, 0)
            return r"Reason. \boxed{1}", torch.tensor([[1, 2]]), torch.tensor([[99]])

        def fake_backbone(model, *, input_ids, **kwargs):
            value = model.theta if model.adapter_enabled else model.theta.detach() * 0
            hidden = value.expand(input_ids.shape[0], input_ids.shape[1], 1)
            return hidden, None

        def fake_project(model, hidden):
            return torch.cat([hidden, -hidden, hidden * 0], dim=-1)

        def fake_fit(*unused_args, **unused_kwargs):
            return _TrackingTeacher(), {
                "specialization_status": "ready",
                "specialization_failure_reason": "",
                "specialization_no_op": False,
                "uses_all_candidates": True,
                "specialization_seconds": 0.3,
                "feature_extraction_seconds": 0.2,
                "closed_form_solve_seconds": 0.1,
                "support_tokens": 1.0,
                "adapted_vocab_size": 3.0,
                "adapter_rank": 1.0,
                "ridge_lambda_effective": 0.1,
                "update_frobenius_norm": 1.0,
                "peak_memory_bytes": 0.0,
                "proposal_fit_target_logit_gain": 1.0,
                "proposal_base_target_nll": 1.0,
            }

        args = SimpleNamespace(
            output_dir="",
            allow_hindsight_exposure=False,
            num_specialization_candidates=None,
            eval_samples=1,
            eval_max_new_tokens=8,
            eval_temperature=0.0,
            top_p=1.0,
            top_k=0,
            seed=0,
            learning_rate=0.1,
            weight_decay=0.0,
            distillation_steps=1,
            train_max_new_tokens=4,
            long_horizon_min_prefix_tokens=4,
            train_temperature=0.8,
            max_grad_norm=1.0,
            distill_top_k=2,
            distill_temperature=1.0,
            distill_token_clip=0.0,
            target_score_mode="answer",
            lora_rank=1,
            lora_alpha=1,
            ridge_lambda=0.1,
            residual_step_size=0.8,
            max_tokens_per_candidate=64,
            max_support_tokens=256,
            hard_negatives=8,
            max_length=4096,
            model="tiny",
            runtime_metadata=_runtime_metadata(),
        )
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            with (
                mock.patch(
                    "src.clean_self_distill.train_eval._fit_current_adapter",
                    new=fake_fit,
                ),
                mock.patch(
                    "src.clean_self_distill.train_eval.generate_response",
                    new=fake_generate,
                ),
                mock.patch(
                    "src.clean_self_distill.train_eval.backbone_forward",
                    new=fake_backbone,
                ),
                mock.patch(
                    "src.clean_self_distill.train_eval.project_logits",
                    new=fake_project,
                ),
                mock.patch(
                    "src.clean_self_distill.train_eval.score_plain_completion",
                    return_value=1.0,
                ),
                mock.patch(
                    "src.clean_self_distill.train_eval.torch.optim.AdamW",
                    new=_TinyOptimizer,
                ),
            ):
                rows, _, _ = per_query_distill_evaluate(
                    model,
                    SimpleNamespace(eos_token_id=99),
                    [record],
                    {record["query_id"]: proposal},
                    args,
                )
                args.resume = True
                with mock.patch(
                    "src.clean_self_distill.train_eval.generate_response",
                    side_effect=AssertionError(
                        "a complete prefix must perform no generation"
                    ),
                ):
                    (
                        resumed_rows,
                        resumed_summary,
                        resumed_audit,
                    ) = per_query_distill_evaluate(
                        model,
                        SimpleNamespace(eos_token_id=99),
                        [record],
                        {record["query_id"]: proposal},
                        args,
                    )
        self.assertEqual(resumed_rows, rows)
        self.assertEqual(resumed_audit.comparison_events, 1)
        self.assertEqual(resumed_summary["overall"]["accuracy/distilled_student"], 1.0)
        self.assertEqual(call_count, 4)
        self.assertEqual(_TrackingTeacher.live, 0)
        self.assertTrue(rows[0]["teacher_destroyed_before_student_evaluation"])
        self.assertTrue(rows[0]["student_reset_verified"])
        self.assertGreater(rows[0]["student_update_frobenius_norm"], 0.0)
        self.assertTrue(torch.equal(model.theta.detach(), initial))
        self.assertEqual(
            rows[0]["proposal_training_sha256"],
            proposal["proposal_training_sha256"],
        )
        self.assertEqual(
            rows[0]["ridge_config_sha256"],
            canonical_json_sha256(rows[0]["ridge_config"]),
        )
        self.assertEqual(
            rows[0]["run_config_sha256"],
            canonical_json_sha256(rows[0]["run_config"]),
        )
        self.assertEqual(rows[0]["distillation_trace"][0]["compared_positions"], 1)
        self.assertEqual(rows[0]["hindsight_audit"]["compared_token_positions"], 1)
        trace = rows[0]["distillation_trace"][0]
        self.assertFalse(trace["prefix_truncated"])
        self.assertTrue(trace["trajectory_complete"])
        self.assertTrue(trace["prefix_natural_completion"])
        self.assertFalse(trace["long_horizon_threshold_reached"])
        self.assertTrue(trace["long_horizon_qualified"])
        self.assertEqual(
            [window["token_count"] for window in trace["horizon_windows"]],
            [1, 0, 0, 0],
        )
        self.assertIsNone(
            trace["horizon_windows"][1][
                "pre_update_mean_teacher_student_kl"
            ]
        )
        self.assertEqual(
            rows[0]["distillation_config"]["long_horizon_min_tokens"], 4
        )
        self.assertEqual(rows[0]["long_horizon_actual_max_causal_depth_tokens"], 1)
        self.assertEqual(rows[0]["long_horizon_natural_completion_rollouts"], 1)
        self.assertTrue(rows[0]["long_horizon_qualified"])

    def test_task2_insufficient_specialization_skips_distillation(self):
        model = _TinyPeft()
        problem = "Find the answer."
        digest = stable_hash(problem, 64)
        record = {
            "query_id": "amc23:q-noop:hash",
            "problem": problem,
            "problem_sha256": digest,
            "source": "amc23",
            "answer": "1",
            "solution": "",
        }
        proposal = _bound_proposal(record["query_id"], problem, ready=False)
        proposal["cost_audit"] = {
            "total_completion_tokens": 0,
            "total_generation_seconds": 0.1,
            "end_to_end_seconds": 0.2,
        }
        proposal["firewall_audit"] = {
            "target_answer_loaded": False,
            "target_solution_loaded": False,
        }
        calls = []

        def fake_generate(model, tokenizer, problem, *, adapter, **kwargs):
            calls.append(adapter)
            if adapter is not None:
                self.assertEqual(adapter.rank, 0)
            return r"Reason. \boxed{1}", torch.tensor([[1, 2]]), torch.tensor([[3]])

        def fake_fit(*unused_args, **unused_kwargs):
            return SparseRidgeAdapter.no_op(1), {
                "specialization_status": "insufficient_verified_candidates",
                "specialization_failure_reason": "no candidates verified",
                "specialization_no_op": True,
                "uses_all_candidates": False,
                "specialization_seconds": 0.0,
                "feature_extraction_seconds": 0.0,
                "closed_form_solve_seconds": 0.0,
                "support_tokens": 0.0,
                "adapted_vocab_size": 0.0,
                "adapter_rank": 0.0,
                "ridge_lambda_effective": 0.0,
                "update_frobenius_norm": 0.0,
                "peak_memory_bytes": 0.0,
                "proposal_fit_target_logit_gain": 0.0,
                "proposal_base_target_nll": 0.0,
            }

        args = SimpleNamespace(
            output_dir="",
            allow_hindsight_exposure=False,
            num_specialization_candidates=None,
            eval_samples=1,
            eval_max_new_tokens=8,
            eval_temperature=0.0,
            top_p=1.0,
            top_k=0,
            seed=0,
            learning_rate=0.1,
            weight_decay=0.0,
            distillation_steps=3,
            train_max_new_tokens=4,
            long_horizon_min_prefix_tokens=4,
            train_temperature=0.8,
            max_grad_norm=1.0,
            distill_top_k=2,
            distill_temperature=1.0,
            distill_token_clip=0.0,
            target_score_mode="answer",
            lora_rank=1,
            lora_alpha=1,
            ridge_lambda=0.1,
            residual_step_size=0.8,
            max_tokens_per_candidate=64,
            max_support_tokens=256,
            hard_negatives=8,
            max_length=4096,
            model="tiny",
            runtime_metadata=_runtime_metadata(),
        )
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            with (
                mock.patch(
                    "src.clean_self_distill.train_eval._fit_current_adapter",
                    new=fake_fit,
                ),
                mock.patch(
                    "src.clean_self_distill.train_eval.generate_response",
                    new=fake_generate,
                ),
                mock.patch(
                    "src.clean_self_distill.train_eval.score_plain_completion",
                    return_value=1.0,
                ),
            ):
                rows, _, _ = per_query_distill_evaluate(
                    model,
                    SimpleNamespace(eos_token_id=99),
                    [record],
                    {record["query_id"]: proposal},
                    args,
                )
        row = rows[0]
        self.assertEqual(len(calls), 3)
        self.assertEqual(row["base_responses"], row["teacher_responses"])
        self.assertEqual(row["base_responses"], row["distilled_responses"])
        self.assertEqual(
            row["specialization_status"], "insufficient_verified_candidates"
        )
        self.assertTrue(row["specialization_no_op"])
        self.assertFalse(row["uses_all_candidates"])
        self.assertEqual(row["adapter_rank"], 0.0)
        self.assertEqual(row["update_frobenius_norm"], 0.0)
        self.assertEqual(row["student_update_frobenius_norm"], 0.0)
        self.assertEqual(row["distillation_steps_completed"], 0)
        self.assertEqual(row["distillation_trace"], [])
        self.assertEqual(row["distillation_rollout_tokens"], 0)
        self.assertEqual(row["distillation_seconds"], 0.0)
        self.assertEqual(row["long_horizon_actual_max_causal_depth_tokens"], 0)
        self.assertIsNone(row["long_horizon_qualified"])
        self.assertEqual(
            row["long_horizon_qualification_reason"],
            "not_applicable_specialization_no_op",
        )
        self.assertTrue(row["teacher_destroyed_before_student_evaluation"])
        self.assertEqual(row["hindsight_audit"]["source_counts"], {"original_query": 1})
        self.assertEqual(row["hindsight_audit"]["comparison_events"], 0)


if __name__ == "__main__":
    unittest.main()
