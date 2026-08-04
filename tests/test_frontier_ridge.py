"""Focused CPU tests for proposal-v5 frontier-weighted ridge specialization."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from src.clean_self_distill.ridge import (
    SparseRidgeAdapter,
    _build_sparse_residual,
    _candidate_frontier,
    _required_frontier_token_count,
)


def _v5_candidate() -> dict:
    return {
        "candidate_id": "c00",
        "problem": "Compute an independent quantity.",
        "correct_trajectory": [
            {"step_index": 0, "text": "Establish the relevant relation."},
            {"step_index": 1, "text": "Apply it to obtain the result."},
        ],
        "wrong_trajectory": [
            {"step_index": 0, "text": "Establish the relevant relation."},
            {"step_index": 1, "text": "Reverse the relation incorrectly."},
        ],
        "wrong_final_answer": "24",
        "error_frontier": {
            "wrong_step_index": 1,
            "wrong_step_text": "Reverse the relation incorrectly.",
            "error_explanation": "The relation cannot be reversed in that way.",
            "corrective_action": "Apply the relation in its original direction.",
            "verifier_valid": True,
        },
        "solution": (
            "Establish the relevant relation.\n\n" "Apply it to obtain the result."
        ),
        "final_answer": "42",
    }


class _CharacterTokenizer:
    """Tiny deterministic tokenizer sufficient for required-budget accounting."""

    eos_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del add_special_tokens
        if return_tensors != "pt":
            raise AssertionError("test tokenizer supports only PyTorch tensors")
        return {
            "input_ids": torch.tensor(
                [[1 + (ord(character) % 251) for character in text]],
                dtype=torch.long,
            )
        }


def _residual(
    *,
    weights: torch.Tensor,
    directions: torch.Tensor,
    target_probability: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = int(weights.numel())
    return _build_sparse_residual(
        labels=torch.full((rows,), 10, dtype=torch.long),
        top_ids=torch.tensor([[20, 30]] * rows, dtype=torch.long),
        top_probs=torch.tensor([[0.5, 0.3]] * rows, dtype=torch.float32),
        target_probs=torch.full((rows,), target_probability, dtype=torch.float32),
        step_size=1.0,
        row_weights=weights,
        row_directions=directions,
        negative_probability_floor=0.25,
    )


def _column(vocab_ids: torch.Tensor, token_id: int) -> int:
    matches = (vocab_ids == token_id).nonzero(as_tuple=False).reshape(-1)
    if matches.numel() != 1:
        raise AssertionError(f"expected exactly one column for token {token_id}")
    return int(matches.item())


class FrontierSchemaTest(unittest.TestCase):
    def test_valid_v5_trajectories_and_frontier_are_normalized(self):
        frontier = _candidate_frontier(_v5_candidate())

        self.assertEqual(
            frontier["correct_steps"],
            [
                "Establish the relevant relation.",
                "Apply it to obtain the result.",
            ],
        )
        self.assertEqual(
            frontier["wrong_steps"],
            [
                "Establish the relevant relation.",
                "Reverse the relation incorrectly.",
            ],
        )
        self.assertEqual(frontier["wrong_step_index"], 1)
        self.assertEqual(
            frontier["wrong_step_text"],
            frontier["wrong_steps"][frontier["wrong_step_index"]],
        )
        self.assertEqual(
            frontier["corrective_action"],
            "Apply the relation in its original direction.",
        )

    def test_v5_trajectory_and_frontier_contract_fails_closed(self):
        invalid_candidates: list[tuple[str, dict]] = []

        def changed(name: str, mutate) -> None:
            candidate = copy.deepcopy(_v5_candidate())
            mutate(candidate)
            invalid_candidates.append((name, candidate))

        changed("missing correct trajectory", lambda row: row.pop("correct_trajectory"))
        changed(
            "empty correct trajectory",
            lambda row: row.__setitem__("correct_trajectory", []),
        )
        changed(
            "non-object trajectory step",
            lambda row: row["correct_trajectory"].__setitem__(0, "not-an-object"),
        )
        changed(
            "nonsequential step index",
            lambda row: row["correct_trajectory"][1].__setitem__("step_index", 3),
        )
        changed(
            "boolean step index",
            lambda row: row["correct_trajectory"][0].__setitem__("step_index", True),
        )
        changed(
            "blank trajectory text",
            lambda row: row["wrong_trajectory"][0].__setitem__("text", "  "),
        )
        changed("missing wrong trajectory", lambda row: row.pop("wrong_trajectory"))
        changed("missing frontier", lambda row: row.pop("error_frontier"))
        changed(
            "frontier is not an object",
            lambda row: row.__setitem__("error_frontier", []),
        )
        changed(
            "boolean wrong-step index",
            lambda row: row["error_frontier"].__setitem__("wrong_step_index", True),
        )
        changed(
            "out-of-range wrong-step index",
            lambda row: row["error_frontier"].__setitem__("wrong_step_index", 2),
        )
        changed(
            "wrong-step text mismatch",
            lambda row: row["error_frontier"].__setitem__(
                "wrong_step_text", "A different step."
            ),
        )
        changed(
            "blank error explanation",
            lambda row: row["error_frontier"].__setitem__("error_explanation", ""),
        )
        changed(
            "blank corrective action",
            lambda row: row["error_frontier"].__setitem__("corrective_action", "\t"),
        )
        changed(
            "unverified frontier",
            lambda row: row["error_frontier"].__setitem__("verifier_valid", False),
        )

        for name, candidate in invalid_candidates:
            with self.subTest(name=name), self.assertRaises(ValueError):
                _candidate_frontier(candidate)

    def test_required_budget_includes_answer_eos_and_both_frontier_spans(self):
        tokenizer = _CharacterTokenizer()
        candidate = _v5_candidate()

        actual = _required_frontier_token_count(
            tokenizer,
            candidate,
            frontier_max_tokens=3,
        )

        # final answer "42" (2), EOS (1), corrective action (capped at 3),
        # and wrong action (capped at 3).
        self.assertEqual(actual, 2 + 1 + 3 + 3)
        with self.assertRaisesRegex(ValueError, "frontier_max_tokens"):
            _required_frontier_token_count(
                tokenizer,
                candidate,
                frontier_max_tokens=0,
            )


class SignedWeightedResidualTest(unittest.TestCase):
    def test_positive_row_boosts_label_and_suppresses_alternatives(self):
        residual, vocab_ids = _residual(
            weights=torch.tensor([1.0]),
            directions=torch.tensor([1.0]),
        )
        label = _column(vocab_ids, 10)
        alternatives = [_column(vocab_ids, token_id) for token_id in (20, 30)]

        self.assertGreater(float(residual[0, label]), 0.0)
        self.assertTrue(bool((residual[0, alternatives] < 0).all()))
        torch.testing.assert_close(
            residual.sum(dim=-1), torch.zeros(1), atol=1e-7, rtol=0
        )

    def test_negative_row_suppresses_wrong_label_and_boosts_alternatives(self):
        residual, vocab_ids = _residual(
            weights=torch.tensor([1.0]),
            directions=torch.tensor([-1.0]),
        )
        wrong_label = _column(vocab_ids, 10)
        alternatives = [_column(vocab_ids, token_id) for token_id in (20, 30)]

        self.assertLess(float(residual[0, wrong_label]), 0.0)
        self.assertTrue(bool((residual[0, alternatives] > 0).all()))
        # target_probability=0.2 is deliberately below the 0.25 hard-negative
        # floor, so the verified wrong action must still receive a -0.25 margin.
        self.assertAlmostEqual(float(residual[0, wrong_label]), -0.25, places=6)
        torch.testing.assert_close(
            residual.sum(dim=-1), torch.zeros(1), atol=1e-7, rtol=0
        )

    def test_frontier_weight_eight_is_32x_reasoning_weight_quarter(self):
        for direction in (1.0, -1.0):
            with self.subTest(direction=direction):
                residual, vocab_ids = _residual(
                    weights=torch.tensor([0.25, 8.0]),
                    directions=torch.tensor([direction, direction]),
                )
                label = _column(vocab_ids, 10)
                reasoning_magnitude = residual[0, label].abs()
                frontier_magnitude = residual[1, label].abs()
                torch.testing.assert_close(
                    frontier_magnitude,
                    reasoning_magnitude * 32.0,
                    atol=1e-6,
                    rtol=1e-6,
                )

    def test_invalid_weights_and_directions_fail_closed(self):
        invalid_weights = (
            torch.tensor([0.0]),
            torch.tensor([-1.0]),
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
            torch.tensor([1.0, 1.0]),
        )
        for weights in invalid_weights:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                _residual(weights=weights, directions=torch.tensor([1.0]))

        invalid_directions = (
            torch.tensor([0.0]),
            torch.tensor([2.0]),
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
            torch.tensor([1.0, -1.0]),
        )
        for directions in invalid_directions:
            with self.subTest(directions=directions), self.assertRaises(ValueError):
                _residual(weights=torch.tensor([1.0]), directions=directions)


class AdapterSchemaTest(unittest.TestCase):
    def test_v2_adapter_round_trip_and_v1_rejection(self):
        adapter = SparseRidgeAdapter(
            support_hidden=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            coefficients=torch.tensor([[0.5, -0.5], [0.25, -0.25]]),
            vocab_ids=torch.tensor([7, 11], dtype=torch.long),
            hidden_scale=2.0,
            ridge_lambda_effective=0.125,
            metadata={"frontier_positive_weight": 8.0},
        )
        self.assertEqual(
            adapter.state_dict()["schema_version"],
            "clean-self-distill-ridge-v2-frontier-weighted",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            v2_path = root / "adapter-v2.pt"
            adapter.save(v2_path)
            loaded = SparseRidgeAdapter.load(v2_path)

            torch.testing.assert_close(loaded.support_hidden, adapter.support_hidden)
            torch.testing.assert_close(loaded.coefficients, adapter.coefficients)
            torch.testing.assert_close(loaded.vocab_ids, adapter.vocab_ids)
            self.assertEqual(loaded.hidden_scale, adapter.hidden_scale)
            self.assertEqual(
                loaded.ridge_lambda_effective,
                adapter.ridge_lambda_effective,
            )
            self.assertEqual(loaded.metadata, adapter.metadata)

            legacy_state = adapter.state_dict()
            legacy_state["schema_version"] = "clean-self-distill-ridge-v1"
            v1_path = root / "adapter-v1.pt"
            torch.save(legacy_state, v1_path)
            with self.assertRaisesRegex(ValueError, "Unsupported adapter schema"):
                SparseRidgeAdapter.load(v1_path)


if __name__ == "__main__":
    unittest.main()
