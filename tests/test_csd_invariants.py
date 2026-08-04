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
    _positions_with_required,
    _validate_proposal_rows,
)
from src.clean_self_distill.train_eval import (
    _index_proposals_by_hash,
    _proposal_for,
    _validate_adapter_manifest_binding,
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


class _TrackingTeacher:
    live = 0

    def __init__(self):
        type(self).live += 1

    def __del__(self):
        type(self).live -= 1

    def apply_to_logits(self, logits, hidden):
        shift = logits.new_tensor([2.0, 0.0, 0.0])
        return logits + shift


def _bound_proposal(query_id: str, problem: str, source: str = "amc23") -> dict:
    row = {
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
                "verifier_valid": True,
                "verifier_accepted": True,
                "verifier_reason": "valid",
                "target_disjoint_audit": {"safe": True},
            }
        ],
    }
    row["proposal_training_sha256"] = compute_proposal_training_sha256(row)
    return row


class CSDInvariantTest(unittest.TestCase):
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
        manifest = {
            "query_id": "q",
            "problem_sha256": "a" * 64,
            "proposal_training_sha256": "c" * 64,
            "source": "amc23",
            "model": "Qwen/Qwen3-4B",
            "model_revision": "b" * 40,
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
            "source",
            "model",
            "model_revision",
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

    def test_answer_positions_are_always_selected(self):
        required = torch.tensor([17, 18, 19])
        selected = _positions_with_required(100, 8, required)
        self.assertTrue(set(required.tolist()).issubset(set(selected.tolist())))
        self.assertEqual(selected.numel(), 8)

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
            return r"Reason. \boxed{1}", torch.tensor([[1, 2]]), torch.tensor([[3]])

        def fake_backbone(model, *, input_ids, **kwargs):
            value = model.theta if model.adapter_enabled else model.theta.detach() * 0
            hidden = value.expand(input_ids.shape[0], input_ids.shape[1], 1)
            return hidden, None

        def fake_project(model, hidden):
            return torch.cat([hidden, -hidden, hidden * 0], dim=-1)

        def fake_fit(*unused_args, **unused_kwargs):
            return _TrackingTeacher(), {
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
            runtime_metadata={"resolved_model_revision": "rev"},
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
            ):
                rows, _, _ = per_query_distill_evaluate(
                    model,
                    SimpleNamespace(eos_token_id=99),
                    [record],
                    {record["query_id"]: proposal},
                    args,
                )
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


if __name__ == "__main__":
    unittest.main()
