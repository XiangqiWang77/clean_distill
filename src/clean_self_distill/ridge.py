"""Closed-form query-local LM-head specialization.

For support hidden states H and a sparse desired logit residual R, we solve

    C = (H H^T + lambda I)^-1 R

and apply the equivalent update without materializing a dense vocabulary head:

    delta_logits(h) = (h H^T) C.

Every proposed and verified candidate is used. There is no Fit/Check split.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm

from .io import (
    canonical_json_sha256,
    iter_rows,
    stable_hash,
    validate_proposal_training_binding,
    validate_specialization_state,
    write_jsonl,
)
from .runtime import (
    backbone_forward,
    collect_runtime_metadata,
    input_device,
    load_hf_model,
    project_logits,
    render_chat,
)


# A verified error frontier is a pairwise decision, not two unrelated token
# likelihood targets.  The signed variant therefore asks for a positive
# correct-minus-wrong logit margin at the exact first divergent token.  Keep
# this fixed (and record it in every adapter) for the preregistered PoC.
FRONTIER_TARGET_MARGIN = 1.0


def problem_prompt(tokenizer, problem: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                f"{problem.strip()}\n\nPlease reason step by step, and put your final answer "
                "within \\boxed{}."
            ),
        }
    ]
    return render_chat(tokenizer, messages, add_generation_prompt=True)


def candidate_completion(candidate: dict[str, Any]) -> str:
    solution = str(candidate.get("solution", "")).strip()
    final_answer = str(candidate.get("final_answer", "")).strip()
    if not solution:
        raise ValueError("Every specialization candidate needs a solution")
    marker = f"Final answer: \\boxed{{{final_answer}}}" if final_answer else ""
    if marker and marker not in solution:
        solution = f"{solution}\n\n{marker}"
    return solution


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_hidden_size(model) -> int:
    try:
        return int(model.get_input_embeddings().weight.shape[-1])
    except (AttributeError, IndexError, TypeError):
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", 0)
        if int(hidden_size or 0) <= 0:
            raise ValueError(
                "Could not determine model hidden size for a no-op adapter"
            )
        return int(hidden_size)


def _uniform_positions(length: int, max_positions: int) -> torch.Tensor:
    if length <= max_positions:
        return torch.arange(length, dtype=torch.long)
    # Uniform coverage keeps both reasoning transitions and the final answer.
    return torch.linspace(0, length - 1, steps=max_positions).round().long().unique()


def _positions_with_required(
    length: int,
    max_positions: int,
    required: torch.Tensor,
) -> torch.Tensor:
    required_values = {
        int(value)
        for value in required.reshape(-1).tolist()
        if 0 <= int(value) < length
    }
    # ``max_positions`` limits optional reasoning positions.  It must never
    # discard a required final-answer target; callers account for any such
    # expansion in their support-token allocation metadata.
    selection_budget = max(max_positions, len(required_values))
    if length <= selection_budget:
        return torch.arange(length, dtype=torch.long)
    selected = set(required_values)
    for value in (
        torch.linspace(0, length - 1, steps=selection_budget * 2)
        .round()
        .long()
        .tolist()
    ):
        if len(selected) >= selection_budget:
            break
        selected.add(int(value))
    if len(selected) < selection_budget:
        for value in range(length):
            selected.add(value)
            if len(selected) >= selection_budget:
                break
    return torch.tensor(sorted(selected), dtype=torch.long)


def _required_answer_token_count(tokenizer, candidate: dict[str, Any]) -> int:
    """Return the answer/EOS positions that cannot be sampled away."""
    final_answer = str(candidate.get("final_answer", "")).strip()
    if not final_answer:
        raise ValueError("Every specialization candidate needs a final answer")
    answer_ids = tokenizer(
        final_answer,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    answer_tokens = int(answer_ids.numel())
    if answer_tokens <= 0:
        raise ValueError("Candidate final answer tokenized to zero tokens")
    return answer_tokens + int(tokenizer.eos_token_id is not None)


def _trajectory_steps(candidate: dict[str, Any], key: str) -> list[str]:
    """Return a fail-closed, ordered trajectory from a proposal-v5 candidate."""
    value = candidate.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Every specialization candidate needs a non-empty {key}")
    steps: list[str] = []
    for expected_index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{key}[{expected_index}] must be an object")
        index = item.get("step_index")
        text = str(item.get("text", "")).strip()
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index != expected_index
        ):
            raise ValueError(
                f"{key}[{expected_index}].step_index must equal {expected_index}"
            )
        if not text:
            raise ValueError(f"{key}[{expected_index}].text must be non-empty")
        steps.append(text)
    return steps


def _candidate_frontier(candidate: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the paired correct/wrong error frontier."""
    correct_steps = _trajectory_steps(candidate, "correct_trajectory")
    wrong_steps = _trajectory_steps(candidate, "wrong_trajectory")
    frontier = candidate.get("error_frontier")
    if not isinstance(frontier, dict):
        raise ValueError(
            "Every specialization candidate needs an error_frontier object"
        )
    index = frontier.get("wrong_step_index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("error_frontier.wrong_step_index must be an integer")
    if index < 0 or index >= len(wrong_steps):
        raise ValueError("error_frontier.wrong_step_index is outside wrong_trajectory")
    wrong_step_text = str(frontier.get("wrong_step_text", "")).strip()
    error_explanation = str(frontier.get("error_explanation", "")).strip()
    corrective_action = str(frontier.get("corrective_action", "")).strip()
    if wrong_step_text != wrong_steps[index]:
        raise ValueError(
            "error_frontier.wrong_step_text must exactly match the indexed wrong step"
        )
    if not error_explanation or not corrective_action:
        raise ValueError(
            "error_frontier requires non-empty error_explanation and corrective_action"
        )
    if frontier.get("verifier_valid") is not True:
        raise ValueError("error_frontier.verifier_valid must be true")
    return {
        "correct_steps": correct_steps,
        "wrong_steps": wrong_steps,
        "wrong_step_index": index,
        "wrong_step_text": wrong_step_text,
        "error_explanation": error_explanation,
        "corrective_action": corrective_action,
    }


def _required_frontier_token_count(
    tokenizer,
    candidate: dict[str, Any],
    *,
    frontier_max_tokens: int,
) -> int:
    if frontier_max_tokens <= 0:
        raise ValueError("frontier_max_tokens must be positive")
    frontier = _candidate_frontier(candidate)
    # Match the exact boundary separator used by feature extraction; otherwise
    # a tokenizer merge could make the required-token budget one token short.
    corrective_ids = tokenizer(
        "\n\n" + frontier["corrective_action"],
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    wrong_ids = tokenizer(
        "\n\n" + frontier["wrong_step_text"],
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    if corrective_ids.numel() == 0 or wrong_ids.numel() == 0:
        raise ValueError(
            "Frontier correct/wrong actions must tokenize to non-empty spans"
        )
    return (
        _required_answer_token_count(tokenizer, candidate)
        + min(int(corrective_ids.numel()), frontier_max_tokens)
        + min(int(wrong_ids.numel()), frontier_max_tokens)
    )


def _frontier_action_tokens(
    tokenizer,
    frontier: dict[str, Any],
    *,
    frontier_max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return common action prefix and the two suffixes from first divergence.

    The leading separator is part of the action tokenization.  Moving its
    common tokens into the context makes suffix row zero the actual decision
    boundary rather than (typically) an identical newline token.  A strict
    token-prefix relation is made comparable with EOS when the tokenizer has
    one; identical actions fail closed because they provide no signed signal.
    """
    if frontier_max_tokens <= 0:
        raise ValueError("frontier_max_tokens must be positive")
    correct = tokenizer(
        "\n\n" + frontier["corrective_action"],
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    wrong = tokenizer(
        "\n\n" + frontier["wrong_step_text"],
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    if correct.numel() == 0 or wrong.numel() == 0:
        raise ValueError("Frontier correct/wrong actions tokenized to empty spans")

    common = 0
    shared_length = min(int(correct.numel()), int(wrong.numel()))
    while common < shared_length and int(correct[common]) == int(wrong[common]):
        common += 1
    if common == int(correct.numel()) == int(wrong.numel()):
        raise ValueError("Frontier correct/wrong actions have identical tokenizations")

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if common == int(correct.numel()):
        if eos_token_id is None:
            raise ValueError(
                "A strict-prefix corrective action needs an EOS token for comparison"
            )
        correct = torch.cat([correct, correct.new_tensor([int(eos_token_id)])])
    if common == int(wrong.numel()):
        if eos_token_id is None:
            raise ValueError(
                "A strict-prefix wrong action needs an EOS token for comparison"
            )
        wrong = torch.cat([wrong, wrong.new_tensor([int(eos_token_id)])])

    shared_prefix = correct[:common]
    correct_suffix = correct[common : common + frontier_max_tokens]
    wrong_suffix = wrong[common : common + frontier_max_tokens]
    if correct_suffix.numel() == 0 or wrong_suffix.numel() == 0:
        raise ValueError("Frontier divergence produced an empty action suffix")
    if int(correct_suffix[0]) == int(wrong_suffix[0]):
        raise AssertionError("Frontier suffixes do not begin at their first divergence")
    return shared_prefix, correct_suffix, wrong_suffix


def _answer_aware_token_allocations(
    required_token_counts: list[int],
    *,
    max_support_tokens: int,
    max_tokens_per_candidate: int,
) -> tuple[list[int], dict[str, Any]]:
    """Fairly allocate optional positions without dropping answer targets.

    ``max_support_tokens`` remains the normal global budget.  If the required
    answer/EOS positions alone exceed it, correctness takes precedence and the
    minimum expansion is reported explicitly instead of crashing mid-shard.
    """
    if not required_token_counts:
        raise ValueError("At least one required-token count is needed")
    if max_support_tokens <= 0:
        raise ValueError("max_support_tokens must be positive")
    if max_tokens_per_candidate <= 0:
        raise ValueError("max_tokens_per_candidate must be positive")
    if any(count <= 0 for count in required_token_counts):
        raise ValueError("Required-token counts must all be positive")

    required_total = sum(required_token_counts)
    capacities = [
        max(max_tokens_per_candidate, required) for required in required_token_counts
    ]
    allocated_budget = min(
        max(max_support_tokens, required_total),
        sum(capacities),
    )
    allocations = list(required_token_counts)
    remaining = allocated_budget - required_total

    # Water filling preserves the old approximately uniform allocation while
    # allowing a long structured answer (for example, a divisor list) to keep
    # every supervised answer token.
    while remaining:
        eligible = [
            index
            for index, (allocation, capacity) in enumerate(zip(allocations, capacities))
            if allocation < capacity
        ]
        if not eligible:
            break
        index = min(eligible, key=lambda value: (allocations[value], value))
        allocations[index] += 1
        remaining -= 1

    overflow = max(0, allocated_budget - max_support_tokens)
    return allocations, {
        "requested_max_support_tokens": max_support_tokens,
        "allocated_support_token_budget": allocated_budget,
        "required_supervision_tokens": required_total,
        # Retained as a compatibility alias for older reporters.  In v2 this
        # count includes answer/EOS plus both frontier action spans.
        "required_answer_tokens": required_total,
        "support_budget_expanded": overflow > 0,
        "support_budget_overflow_tokens": overflow,
    }


@dataclass
class SparseRidgeAdapter:
    support_hidden: torch.Tensor
    coefficients: torch.Tensor
    vocab_ids: torch.Tensor
    hidden_scale: float
    ridge_lambda_effective: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return int(self.support_hidden.shape[0])

    @property
    def adapted_vocab_size(self) -> int:
        return int(self.vocab_ids.numel())

    @classmethod
    def no_op(
        cls,
        hidden_size: int,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "SparseRidgeAdapter":
        """Create an explicit rank-0 adapter whose logit update is exactly zero."""
        if hidden_size <= 0:
            raise ValueError("A no-op adapter requires a positive hidden size")
        return cls(
            support_hidden=torch.empty((0, hidden_size), dtype=torch.float16),
            coefficients=torch.empty((0, 0), dtype=torch.float16),
            vocab_ids=torch.empty((0,), dtype=torch.long),
            hidden_scale=1.0,
            ridge_lambda_effective=0.0,
            metadata=dict(metadata or {}),
        )

    def to(self, device: torch.device | str) -> "SparseRidgeAdapter":
        return SparseRidgeAdapter(
            support_hidden=self.support_hidden.to(device=device, dtype=torch.float32),
            coefficients=self.coefficients.to(device=device, dtype=torch.float32),
            vocab_ids=self.vocab_ids.to(device),
            hidden_scale=self.hidden_scale,
            ridge_lambda_effective=self.ridge_lambda_effective,
            metadata=dict(self.metadata),
        )

    def selected_delta(
        self, hidden: torch.Tensor, chunk_size: int = 256
    ) -> torch.Tensor:
        original_shape = hidden.shape[:-1]
        flat_hidden = hidden.reshape(-1, hidden.shape[-1]).float() / self.hidden_scale
        support = self.support_hidden.to(flat_hidden.device, dtype=torch.float32)
        coefficients = self.coefficients.to(flat_hidden.device, dtype=torch.float32)
        chunks = []
        for start in range(0, flat_hidden.shape[0], chunk_size):
            query = flat_hidden[start : start + chunk_size]
            chunks.append((query @ support.T) @ coefficients)
        delta = (
            torch.cat(chunks, dim=0)
            if chunks
            else flat_hidden.new_zeros((0, coefficients.shape[1]))
        )
        return delta.reshape(*original_shape, coefficients.shape[1])

    def apply_to_logits(
        self, logits: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        if logits.shape[:-1] != hidden.shape[:-1]:
            raise ValueError(
                f"Logit/hidden prefix shapes differ: {logits.shape} vs {hidden.shape}"
            )
        if self.rank == 0:
            return logits
        adapted = logits.clone()
        delta = self.selected_delta(hidden).to(
            device=adapted.device, dtype=adapted.dtype
        )
        vocab_ids = self.vocab_ids.to(adapted.device)
        adapted[..., vocab_ids] = adapted[..., vocab_ids] + delta
        return adapted

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "clean-self-distill-ridge-v3-paired-hard-margin",
            "support_hidden": self.support_hidden.detach().cpu(),
            "coefficients": self.coefficients.detach().cpu(),
            "vocab_ids": self.vocab_ids.detach().cpu(),
            "hidden_scale": self.hidden_scale,
            "ridge_lambda_effective": self.ridge_lambda_effective,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "SparseRidgeAdapter":
        state = torch.load(path, map_location=map_location)
        if (
            state.get("schema_version")
            != "clean-self-distill-ridge-v3-paired-hard-margin"
        ):
            raise ValueError(f"Unsupported adapter schema in {path}")
        return cls(
            support_hidden=state["support_hidden"],
            coefficients=state["coefficients"],
            vocab_ids=state["vocab_ids"],
            hidden_scale=float(state["hidden_scale"]),
            ridge_lambda_effective=float(state["ridge_lambda_effective"]),
            metadata=dict(state.get("metadata", {})),
        )


@torch.inference_mode()
def _scored_completion_features(
    model,
    *,
    context_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    positions: torch.Tensor,
    hard_negatives: int,
) -> dict[str, torch.Tensor]:
    """Score selected teacher-forced completion positions after an exact context."""
    if context_ids.ndim != 1 or completion_ids.ndim != 1:
        raise ValueError("context_ids and completion_ids must be one-dimensional")
    if context_ids.numel() == 0 or completion_ids.numel() == 0:
        raise ValueError("Teacher-forced context and completion must be non-empty")
    if positions.numel() == 0:
        raise ValueError("At least one completion position must be selected")
    device = input_device(model)
    full_ids = torch.cat([context_ids, completion_ids]).unsqueeze(0).to(device)
    all_hidden, _ = backbone_forward(
        model,
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        use_cache=False,
    )
    start = int(context_ids.numel()) - 1
    hidden = all_hidden[0, start : start + int(completion_ids.numel())]
    positions = positions.to(hidden.device)
    hidden = hidden.index_select(0, positions)
    labels = completion_ids.to(hidden.device).index_select(0, positions)
    selected_logits = project_logits(model, hidden).float()
    k = min(max(1, hard_negatives), selected_logits.shape[-1])
    top_values, top_ids = torch.topk(selected_logits, k=k, dim=-1)
    log_normalizer = torch.logsumexp(selected_logits, dim=-1, keepdim=True)
    top_probs = torch.exp(top_values - log_normalizer)
    target_logits = selected_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    target_probs = torch.exp(target_logits - log_normalizer.squeeze(-1))
    target_log_probs = target_logits - log_normalizer.squeeze(-1)
    return {
        "hidden": hidden.float().cpu(),
        "labels": labels.cpu(),
        "top_ids": top_ids.cpu(),
        "top_probs": top_probs.cpu(),
        "target_probs": target_probs.cpu(),
        "target_log_probs": target_log_probs.cpu(),
        "target_logits": target_logits.cpu(),
    }


@torch.inference_mode()
def _candidate_features(
    model,
    tokenizer,
    candidate: dict[str, Any],
    *,
    max_tokens: int,
    hard_negatives: int,
    max_length: int,
    reasoning_token_weight: float,
    answer_token_weight: float,
    frontier_positive_weight: float,
    frontier_negative_weight: float,
    frontier_max_tokens: int,
    signed_frontier: bool,
) -> dict[str, torch.Tensor]:
    # Correct-only must consume only its verified correct trajectory during
    # adapter construction.  Wrong/frontier parsing and tokenization are
    # deferred to the post-fit diagnostic path below.
    correct_steps = _trajectory_steps(candidate, "correct_trajectory")
    frontier = _candidate_frontier(candidate) if signed_frontier else None
    prompt_ids = tokenizer(
        problem_prompt(tokenizer, str(candidate["problem"])),
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0]
    solution = "\n\n".join(correct_steps)
    final_answer = str(candidate.get("final_answer", "")).strip()
    if not solution or not final_answer:
        raise ValueError(
            "Every specialization candidate needs a solution and final answer"
        )
    reasoning_ids = tokenizer(
        solution,
        add_special_tokens=False,
        return_tensors="pt",
    )[
        "input_ids"
    ][0]
    answer_prefix_ids = tokenizer(
        "\n\nFinal answer: \\boxed{",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    answer_ids = tokenizer(
        final_answer,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    answer_suffix_ids = tokenizer("}", add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ][0]
    suffix_ids = torch.cat([answer_prefix_ids, answer_ids, answer_suffix_ids])
    if tokenizer.eos_token_id is not None:
        suffix_ids = torch.cat(
            [suffix_ids, suffix_ids.new_tensor([tokenizer.eos_token_id])]
        )

    shared_action_prefix_ids = torch.empty((0,), dtype=torch.long)
    corrective_ids = torch.empty((0,), dtype=torch.long)
    wrong_action_ids = torch.empty((0,), dtype=torch.long)
    if signed_frontier:
        assert frontier is not None
        (
            shared_action_prefix_ids,
            corrective_ids,
            wrong_action_ids,
        ) = _frontier_action_tokens(
            tokenizer,
            frontier,
            frontier_max_tokens=frontier_max_tokens,
        )

    original_completion_length = int(reasoning_ids.numel() + suffix_ids.numel())
    available = max_length - int(prompt_ids.numel())
    if available <= 0:
        raise ValueError("Candidate prompt exceeds max_length")
    if int(suffix_ids.numel()) > available:
        raise ValueError(
            "Candidate prompt leaves no room for its final-answer token span"
        )
    reasoning_available = available - int(suffix_ids.numel())
    completion_truncated = int(reasoning_ids.numel()) > reasoning_available
    if completion_truncated:
        prefix_length = reasoning_available // 2
        suffix_length = reasoning_available - prefix_length
        reasoning_ids = torch.cat(
            [
                reasoning_ids[:prefix_length],
                reasoning_ids[-suffix_length:] if suffix_length else reasoning_ids[:0],
            ]
        )
    answer_start = int(reasoning_ids.numel() + answer_prefix_ids.numel())
    answer_stop = answer_start + int(answer_ids.numel())
    completion_ids = torch.cat([reasoning_ids, suffix_ids])
    if completion_ids.numel() == 0:
        raise ValueError("Candidate completion tokenized to zero tokens")

    completion_len = int(completion_ids.numel())

    required_positions = torch.arange(answer_start, answer_stop, dtype=torch.long)
    if tokenizer.eos_token_id is not None:
        required_positions = torch.cat(
            [required_positions, torch.tensor([completion_len - 1], dtype=torch.long)]
        )
    # Both ablation variants use exactly min(allocation, available correct
    # completion positions) ridge rows.  Signed frontier rows replace ordinary
    # correct rows; they never increase actual support tokens or adapter rank.
    matched_total_budget = min(max_tokens, completion_len)
    if signed_frontier:
        frontier_capacity = matched_total_budget - int(required_positions.numel())
        if frontier_capacity < 2:
            raise ValueError(
                "Matched support budget leaves no room for both frontier directions"
            )
        positive_budget = min(
            int(corrective_ids.numel()), max(1, frontier_capacity // 2)
        )
        negative_budget = min(
            int(wrong_action_ids.numel()), max(1, frontier_capacity - positive_budget)
        )
        # If the negative span was shorter, give its unused share to the
        # positive span (and vice versa) without exceeding the fixed total.
        remaining = frontier_capacity - positive_budget - negative_budget
        if remaining > 0:
            positive_extra = min(
                remaining, int(corrective_ids.numel()) - positive_budget
            )
            positive_budget += positive_extra
            remaining -= positive_extra
        if remaining > 0:
            negative_budget += min(
                remaining, int(wrong_action_ids.numel()) - negative_budget
            )
        corrective_ids = corrective_ids[:positive_budget]
        wrong_action_ids = wrong_action_ids[:negative_budget]
        frontier_required = int(corrective_ids.numel() + wrong_action_ids.numel())
    else:
        frontier_required = 0
    main_budget = matched_total_budget - frontier_required
    if main_budget < int(required_positions.numel()):
        raise ValueError(
            "Allocated candidate support budget cannot preserve answer and frontier spans"
        )
    positions = _positions_with_required(
        completion_len, main_budget, required_positions
    )
    main = _scored_completion_features(
        model,
        context_ids=prompt_ids,
        completion_ids=completion_ids,
        positions=positions,
        hard_negatives=hard_negatives,
    )
    main_weights = torch.full(
        (positions.numel(),), float(reasoning_token_weight), dtype=torch.float32
    )
    main_kinds = torch.zeros(positions.numel(), dtype=torch.long)
    answer_mask = (positions >= answer_start) & (positions < answer_stop)
    if tokenizer.eos_token_id is not None:
        answer_mask |= positions == completion_len - 1
    main_weights[answer_mask] = float(answer_token_weight)
    main_kinds[answer_mask] = 1

    positive: Optional[dict[str, torch.Tensor]] = None
    negative: Optional[dict[str, torch.Tensor]] = None
    if signed_frontier:
        assert frontier is not None
        wrong_prefix = "\n\n".join(
            frontier["wrong_steps"][: frontier["wrong_step_index"]]
        )
        wrong_prefix_ids = tokenizer(
            wrong_prefix,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        max_action_tokens = max(
            int(corrective_ids.numel()), int(wrong_action_ids.numel())
        )
        prefix_available = (
            max_length
            - int(prompt_ids.numel())
            - int(shared_action_prefix_ids.numel())
            - max_action_tokens
        )
        if prefix_available < 0:
            raise ValueError("Candidate prompt leaves no room for frontier action tokens")
        if int(wrong_prefix_ids.numel()) > prefix_available:
            wrong_prefix_ids = (
                wrong_prefix_ids[-prefix_available:]
                if prefix_available
                else wrong_prefix_ids[:0]
            )
        frontier_context_ids = torch.cat(
            [prompt_ids, wrong_prefix_ids, shared_action_prefix_ids]
        )
        positive = _scored_completion_features(
            model,
            context_ids=frontier_context_ids,
            completion_ids=corrective_ids,
            positions=torch.arange(corrective_ids.numel(), dtype=torch.long),
            hard_negatives=hard_negatives,
        )
        negative = _scored_completion_features(
            model,
            context_ids=frontier_context_ids,
            completion_ids=wrong_action_ids,
            positions=torch.arange(wrong_action_ids.numel(), dtype=torch.long),
            hard_negatives=hard_negatives,
        )
        # Row zero is the exact same pre-decision state for both actions.  Fail
        # closed if an implementation change ever breaks that invariant.
        if not torch.equal(positive["hidden"][0], negative["hidden"][0]):
            raise RuntimeError(
                "Correct/wrong frontier tokens were not scored at the same state"
            )

    rows = (main, positive, negative) if signed_frontier else (main,)
    if signed_frontier:
        assert positive is not None and negative is not None
        fit_weights = torch.cat(
            [
                main_weights,
                torch.full(
                    (positive["labels"].numel(),),
                    float(frontier_positive_weight),
                    dtype=torch.float32,
                ),
                torch.full(
                    (negative["labels"].numel(),),
                    float(frontier_negative_weight),
                    dtype=torch.float32,
                ),
            ]
        )
        fit_directions = torch.cat(
            [
                torch.ones(main["labels"].numel() + positive["labels"].numel()),
                -torch.ones(negative["labels"].numel()),
            ]
        )
        fit_kinds = torch.cat(
            [
                main_kinds,
                torch.full((positive["labels"].numel(),), 2, dtype=torch.long),
                torch.full((negative["labels"].numel(),), 3, dtype=torch.long),
            ]
        )
        fit_pair_positive_ids = torch.full_like(fit_directions, -1, dtype=torch.long)
        fit_pair_negative_ids = torch.full_like(fit_directions, -1, dtype=torch.long)
        fit_pair_base_margins = torch.full_like(
            fit_directions, float("nan"), dtype=torch.float32
        )
        positive_row = int(main["labels"].numel())
        negative_row = positive_row + int(positive["labels"].numel())
        pair_positive_token = int(positive["labels"][0].item())
        pair_negative_token = int(negative["labels"][0].item())
        pair_base_margin = float(
            positive["target_logits"][0].item()
            - negative["target_logits"][0].item()
        )
        # Encode the exact pair on both duplicated first-token rows.  Their
        # respective positive/negative weights control the two signed halves;
        # the residual builder directly targets correct-minus-wrong margin.
        for row_index in (positive_row, negative_row):
            fit_pair_positive_ids[row_index] = pair_positive_token
            fit_pair_negative_ids[row_index] = pair_negative_token
            fit_pair_base_margins[row_index] = pair_base_margin
    else:
        fit_weights = main_weights
        fit_directions = torch.ones(main["labels"].numel())
        fit_kinds = main_kinds
        fit_pair_positive_ids = torch.full_like(
            fit_directions, -1, dtype=torch.long
        )
        fit_pair_negative_ids = torch.full_like(
            fit_directions, -1, dtype=torch.long
        )
        fit_pair_base_margins = torch.full_like(
            fit_directions, float("nan"), dtype=torch.float32
        )
    frontier_comparable = signed_frontier
    diagnostic_hidden = (
        positive["hidden"][0]
        if positive is not None
        else torch.empty((0,), dtype=torch.float32)
    )
    diagnostic_positive_token = (
        positive["labels"][0]
        if positive is not None
        else torch.tensor(-1, dtype=torch.long)
    )
    diagnostic_negative_token = (
        negative["labels"][0]
        if negative is not None
        else torch.tensor(-1, dtype=torch.long)
    )
    diagnostic_positive_logit = (
        positive["target_logits"][0]
        if positive is not None
        else torch.tensor(float("nan"))
    )
    diagnostic_negative_logit = (
        negative["target_logits"][0]
        if negative is not None
        else torch.tensor(float("nan"))
    )
    return {
        "hidden": torch.cat([row["hidden"] for row in rows], dim=0),
        "labels": torch.cat([row["labels"] for row in rows], dim=0),
        "top_ids": torch.cat([row["top_ids"] for row in rows], dim=0),
        "top_probs": torch.cat([row["top_probs"] for row in rows], dim=0),
        "target_probs": torch.cat([row["target_probs"] for row in rows], dim=0),
        "target_log_probs": torch.cat([row["target_log_probs"] for row in rows], dim=0),
        "row_weights": fit_weights,
        "row_directions": fit_directions,
        "row_kinds": fit_kinds,
        "pair_positive_ids": fit_pair_positive_ids,
        "pair_negative_ids": fit_pair_negative_ids,
        "pair_base_margins": fit_pair_base_margins,
        # The first *divergent* action tokens are scored at the exact same
        # pre-decision hidden state.  Leading separators/common words are
        # skipped; otherwise every margin would compare an identical newline.
        "frontier_comparable": torch.tensor(float(frontier_comparable)),
        "frontier_hidden": diagnostic_hidden,
        "frontier_positive_token": diagnostic_positive_token,
        "frontier_negative_token": diagnostic_negative_token,
        "frontier_base_positive_logit": diagnostic_positive_logit,
        "frontier_base_negative_logit": diagnostic_negative_logit,
        "completion_truncated": torch.tensor(float(completion_truncated)),
        "original_completion_tokens": torch.tensor(float(original_completion_length)),
        "answer_tokens_selected": torch.tensor(float(answer_ids.numel())),
        "reasoning_tokens_selected": torch.tensor(
            float((main_kinds == 0).sum().item())
        ),
        "frontier_corrective_tokens_selected": torch.tensor(
            float(corrective_ids.numel()) if signed_frontier else 0.0
        ),
        "frontier_wrong_tokens_selected": torch.tensor(
            float(wrong_action_ids.numel()) if signed_frontier else 0.0
        ),
    }


@torch.inference_mode()
def _frontier_diagnostic_feature(
    model,
    tokenizer,
    candidate: dict[str, Any],
    *,
    hard_negatives: int,
    max_length: int,
    frontier_max_tokens: int,
) -> dict[str, torch.Tensor]:
    """Score the exact frontier pair without adding either row to the fit.

    Correct-only uses this only after its adapter and reported adaptation time
    are finalized.  Thus wrong support cannot influence its coefficients,
    token budget, rank, or adaptation latency, while DBCR/regression remain
    measurable on the same candidate frontiers for both ablation variants.
    """
    frontier = _candidate_frontier(candidate)
    prompt_ids = tokenizer(
        problem_prompt(tokenizer, str(candidate["problem"])),
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0]
    shared, correct, wrong = _frontier_action_tokens(
        tokenizer,
        frontier,
        frontier_max_tokens=frontier_max_tokens,
    )
    wrong_prefix = "\n\n".join(
        frontier["wrong_steps"][: frontier["wrong_step_index"]]
    )
    wrong_prefix_ids = tokenizer(
        wrong_prefix,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    prefix_available = (
        max_length - int(prompt_ids.numel()) - int(shared.numel()) - 1
    )
    if prefix_available < 0:
        raise ValueError("Candidate prompt leaves no room for a frontier diagnostic")
    if int(wrong_prefix_ids.numel()) > prefix_available:
        wrong_prefix_ids = (
            wrong_prefix_ids[-prefix_available:]
            if prefix_available
            else wrong_prefix_ids[:0]
        )
    context_ids = torch.cat([prompt_ids, wrong_prefix_ids, shared])
    position = torch.tensor([0], dtype=torch.long)
    positive = _scored_completion_features(
        model,
        context_ids=context_ids,
        completion_ids=correct[:1],
        positions=position,
        hard_negatives=hard_negatives,
    )
    negative = _scored_completion_features(
        model,
        context_ids=context_ids,
        completion_ids=wrong[:1],
        positions=position,
        hard_negatives=hard_negatives,
    )
    if not torch.equal(positive["hidden"][0], negative["hidden"][0]):
        raise RuntimeError(
            "Correct/wrong diagnostic tokens were not scored at the same state"
        )
    return {
        "frontier_comparable": torch.tensor(1.0),
        "frontier_hidden": positive["hidden"][0],
        "frontier_positive_token": positive["labels"][0],
        "frontier_negative_token": negative["labels"][0],
        "frontier_base_positive_logit": positive["target_logits"][0],
        "frontier_base_negative_logit": negative["target_logits"][0],
    }


def _build_sparse_residual(
    labels: torch.Tensor,
    top_ids: torch.Tensor,
    top_probs: torch.Tensor,
    target_probs: torch.Tensor,
    step_size: float,
    *,
    row_weights: Optional[torch.Tensor] = None,
    row_directions: Optional[torch.Tensor] = None,
    pair_positive_ids: Optional[torch.Tensor] = None,
    pair_negative_ids: Optional[torch.Tensor] = None,
    pair_base_margins: Optional[torch.Tensor] = None,
    frontier_target_margin: float = FRONTIER_TARGET_MARGIN,
    negative_probability_floor: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build signed, per-position desired logit changes.

    Positive rows boost the verified next token and suppress its strongest
    alternatives.  Negative rows do the converse: they *directly suppress*
    the model-produced wrong action and transfer the same margin to competing
    tokens.  Exact first-divergence rows instead receive a direct pairwise
    residual on ``z(correct)-z(wrong)``.  Its requested correction is exactly
    the gap to ``frontier_target_margin``, so the paired action cannot disappear
    merely because it was absent from top-k alternatives.  ``row_weights`` are validated here
    but applied later as weighted least squares (sqrt-weighting both H and R);
    they do not silently change the desired target itself.
    """
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if not 0.0 <= negative_probability_floor <= 1.0:
        raise ValueError("negative_probability_floor must lie in [0, 1]")
    if not math.isfinite(float(frontier_target_margin)) or frontier_target_margin <= 0:
        raise ValueError("frontier_target_margin must be finite and positive")
    row_count = int(labels.numel())
    if top_ids.ndim != 2 or top_probs.shape != top_ids.shape:
        raise ValueError("top_ids and top_probs must be matching rank-2 tensors")
    if top_ids.shape[0] != row_count or target_probs.numel() != row_count:
        raise ValueError("Sparse residual inputs disagree on their row count")
    if row_weights is None:
        row_weights = torch.ones(row_count, dtype=torch.float32)
    if row_directions is None:
        row_directions = torch.ones(row_count, dtype=torch.float32)
    row_weights = row_weights.reshape(-1).float().cpu()
    row_directions = row_directions.reshape(-1).float().cpu()
    if row_weights.numel() != row_count or row_directions.numel() != row_count:
        raise ValueError("row_weights and row_directions must have one value per row")
    if not torch.isfinite(row_weights).all() or (row_weights <= 0).any():
        raise ValueError("row_weights must be finite and strictly positive")
    if not torch.isfinite(row_directions).all() or not torch.all(
        (row_directions == 1) | (row_directions == -1)
    ):
        raise ValueError("row_directions must contain only +1 or -1")

    if pair_positive_ids is None:
        pair_positive_ids = torch.full((row_count,), -1, dtype=torch.long)
    if pair_negative_ids is None:
        pair_negative_ids = torch.full((row_count,), -1, dtype=torch.long)
    if pair_base_margins is None:
        pair_base_margins = torch.full(
            (row_count,), float("nan"), dtype=torch.float32
        )
    pair_positive_ids = pair_positive_ids.reshape(-1).long().cpu()
    pair_negative_ids = pair_negative_ids.reshape(-1).long().cpu()
    pair_base_margins = pair_base_margins.reshape(-1).float().cpu()
    if not (
        pair_positive_ids.numel()
        == pair_negative_ids.numel()
        == pair_base_margins.numel()
        == row_count
    ):
        raise ValueError("Pairwise frontier tensors must have one value per row")
    pair_mask = pair_positive_ids >= 0
    if not torch.equal(pair_mask, pair_negative_ids >= 0):
        raise ValueError("Pairwise frontier token ids must be present together")
    if bool(pair_mask.any()):
        if not torch.isfinite(pair_base_margins[pair_mask]).all():
            raise ValueError("Pairwise frontier base margins must be finite")
        if bool((pair_positive_ids[pair_mask] == pair_negative_ids[pair_mask]).any()):
            raise ValueError("Pairwise frontier tokens must be different")
        expected_labels = torch.where(
            row_directions[pair_mask] > 0,
            pair_positive_ids[pair_mask],
            pair_negative_ids[pair_mask],
        )
        if not torch.equal(labels.reshape(-1).long().cpu()[pair_mask], expected_labels):
            raise ValueError(
                "Pairwise frontier row labels/directions do not match the action pair"
            )
    if bool(torch.isfinite(pair_base_margins[~pair_mask]).any()):
        raise ValueError("Non-pair rows cannot carry a pairwise base margin")

    vocab_parts = [labels.reshape(-1), top_ids.reshape(-1)]
    if bool(pair_mask.any()):
        vocab_parts.extend(
            [pair_positive_ids[pair_mask], pair_negative_ids[pair_mask]]
        )
    vocab_ids = torch.unique(torch.cat(vocab_parts)).sort().values
    id_to_column = {
        int(token_id): column for column, token_id in enumerate(vocab_ids.tolist())
    }
    residual = torch.zeros(row_count, vocab_ids.numel(), dtype=torch.float32)
    for row in range(row_count):
        if bool(pair_mask[row]):
            positive_id = int(pair_positive_ids[row])
            negative_id = int(pair_negative_ids[row])
            base_margin = float(pair_base_margins[row])
            margin_deficit = max(0.0, frontier_target_margin - base_margin)
            if margin_deficit > 0.0:
                # Half goes to each side, so the desired pairwise margin gain
                # is exactly ``magnitude`` before ridge regularization.
                magnitude = margin_deficit
                residual[row, id_to_column[positive_id]] += 0.5 * magnitude
                residual[row, id_to_column[negative_id]] -= 0.5 * magnitude
            continue
        label = int(labels[row])
        alternatives = [
            (int(token_id), max(float(probability), 0.0))
            for token_id, probability in zip(
                top_ids[row].tolist(), top_probs[row].tolist()
            )
            if int(token_id) != label
        ]
        alternative_mass = sum(probability for _, probability in alternatives)
        scale = step_size
        if float(row_directions[row]) > 0:
            magnitude = scale * max(0.0, 1.0 - float(target_probs[row]))
            residual[row, id_to_column[label]] += magnitude
            if alternative_mass > 0:
                for token_id, probability in alternatives:
                    residual[row, id_to_column[token_id]] -= (
                        magnitude * probability / alternative_mass
                    )
        else:
            magnitude = scale * max(
                float(target_probs[row]), negative_probability_floor
            )
            residual[row, id_to_column[label]] -= magnitude
            if alternative_mass > 0:
                for token_id, probability in alternatives:
                    residual[row, id_to_column[token_id]] += (
                        magnitude * probability / alternative_mass
                    )
    return residual, vocab_ids


def _cholesky_solve(
    kernel: torch.Tensor, residual: torch.Tensor, ridge: float
) -> torch.Tensor:
    regularized = kernel.clone()
    regularized.diagonal().add_(ridge)
    try:
        factor = torch.linalg.cholesky(regularized)
        return torch.cholesky_solve(residual, factor)
    except RuntimeError:
        return torch.linalg.solve(regularized, residual)


@torch.inference_mode()
def fit_ridge_adapter(
    model,
    tokenizer,
    candidates: list[dict[str, Any]],
    *,
    ridge_lambda: float = 0.1,
    residual_step_size: float = 0.8,
    max_tokens_per_candidate: int = 64,
    max_support_tokens: int = 256,
    hard_negatives: int = 8,
    max_length: int = 4096,
    reasoning_token_weight: float = 0.25,
    answer_token_weight: float = 1.0,
    frontier_positive_weight: float = 8.0,
    frontier_negative_weight: float = 8.0,
    frontier_max_tokens: int = 24,
    frontier_negative_probability_floor: float = 0.25,
    frontier_target_margin: float = FRONTIER_TARGET_MARGIN,
    signed_frontier: bool = True,
    max_update_norm: float = 2.0,
    query_id: str = "",
    specialization_status: Optional[str] = None,
    specialization_failure_reason: str = "",
    specialization_no_op: Optional[bool] = None,
) -> tuple[SparseRidgeAdapter, dict[str, Any]]:
    """Fit the temporary teacher using all verified proposed candidates."""
    positive_scalars = {
        "ridge_lambda": ridge_lambda,
        "residual_step_size": residual_step_size,
        "reasoning_token_weight": reasoning_token_weight,
        "answer_token_weight": answer_token_weight,
        "frontier_positive_weight": frontier_positive_weight,
        "frontier_negative_weight": frontier_negative_weight,
        "frontier_target_margin": frontier_target_margin,
        "max_update_norm": max_update_norm,
    }
    for name, value in positive_scalars.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if hard_negatives <= 0 or max_length <= 0 or frontier_max_tokens <= 0:
        raise ValueError(
            "hard_negatives, max_length, and frontier_max_tokens must be positive"
        )
    if not 0.0 <= frontier_negative_probability_floor <= 1.0:
        raise ValueError("frontier_negative_probability_floor must lie in [0, 1]")
    if specialization_status is None:
        specialization_status = (
            "ready" if candidates else "insufficient_verified_candidates"
        )
    if specialization_no_op is None:
        specialization_no_op = specialization_status != "ready"
    if specialization_status == "insufficient_verified_candidates" and not (
        specialization_failure_reason.strip()
    ):
        specialization_failure_reason = (
            "insufficient verified specialization candidates"
        )
    validate_specialization_state(
        {
            "specialization_candidates": candidates,
            "specialization_status": specialization_status,
            "specialization_failure_reason": specialization_failure_reason,
            "specialization_no_op": specialization_no_op,
        },
        context=f"Ridge specialization for {query_id!r}",
    )
    if specialization_no_op:
        adapter = SparseRidgeAdapter.no_op(
            _model_hidden_size(model),
            metadata={
                "query_id": query_id,
                "num_candidates": len(candidates),
                "support_tokens": 0,
                "max_support_tokens": max_support_tokens,
                "requested_max_support_tokens": max_support_tokens,
                "allocated_support_token_budget": 0,
                "required_supervision_tokens": 0,
                "required_answer_tokens": 0,
                "support_budget_expanded": False,
                "support_budget_overflow_tokens": 0,
                "hard_negatives": hard_negatives,
                "residual_step_size": residual_step_size,
                "reasoning_token_weight": reasoning_token_weight,
                "answer_token_weight": answer_token_weight,
                "frontier_positive_weight": frontier_positive_weight,
                "frontier_negative_weight": frontier_negative_weight,
                "frontier_max_tokens": frontier_max_tokens,
                "frontier_negative_probability_floor": (
                    frontier_negative_probability_floor
                ),
                "frontier_target_margin": frontier_target_margin,
                "frontier_margin_objective": (
                    "paired_first_divergence_hard_margin_v1"
                ),
                "row_weighting_mode": "weighted_least_squares_sqrt_v1",
                "signed_frontier": signed_frontier,
                "max_update_norm": max_update_norm,
                "uses_all_candidates": False,
                "fit_check_split": False,
                "teacher_context_sources": [],
                "specialization_status": specialization_status,
                "specialization_failure_reason": specialization_failure_reason,
                "specialization_no_op": True,
            },
        )
        return adapter, {
            "specialization_status": specialization_status,
            "specialization_failure_reason": specialization_failure_reason,
            "specialization_no_op": True,
            "uses_all_candidates": False,
            "specialization_seconds": 0.0,
            "feature_extraction_seconds": 0.0,
            "closed_form_solve_seconds": 0.0,
            "support_tokens": 0.0,
            "requested_max_support_tokens": float(max_support_tokens),
            "allocated_support_token_budget": 0.0,
            "required_supervision_tokens": 0.0,
            "required_answer_tokens": 0.0,
            "support_budget_expanded": False,
            "support_budget_overflow_tokens": 0.0,
            "adapted_vocab_size": 0.0,
            "adapter_rank": 0.0,
            "ridge_lambda_effective": 0.0,
            "update_frobenius_norm": 0.0,
            "unclipped_update_frobenius_norm": 0.0,
            "update_norm_was_clipped": False,
            "proposal_fit_target_logit_gain": 0.0,
            "proposal_fit_signed_target_logit_gain": 0.0,
            "frontier_corrective_target_logit_gain": 0.0,
            "frontier_wrong_target_logit_change": 0.0,
            "frontier_margin_base_mean": 0.0,
            "frontier_margin_teacher_mean": 0.0,
            "frontier_margin_gain_mean": 0.0,
            "frontier_target_margin": float(frontier_target_margin),
            "frontier_margin_objective": (
                "paired_first_divergence_hard_margin_v1"
            ),
            "row_weighting_mode": "weighted_least_squares_sqrt_v1",
            "frontier_comparable_count": 0.0,
            "frontier_target_margin_attainment_count": 0.0,
            "frontier_target_margin_attainment_rate": 0.0,
            "frontier_margins": [],
            "candidate_count": len(candidates),
            "decision_boundary_crossing_count": 0.0,
            "decision_boundary_eligible_count": 0.0,
            "decision_boundary_crossing_rate": 0.0,
            "decision_boundary_crossing_definition": (
                "base_margin<=0 and teacher_margin>0"
            ),
            "decision_boundary_regression_count": 0.0,
            "decision_boundary_regression_eligible_count": 0.0,
            "decision_boundary_regression_rate": 0.0,
            "decision_boundary_regression_definition": (
                "base_margin>0 and teacher_margin<=0"
            ),
            "proposal_base_target_nll": 0.0,
            "candidate_completion_truncated_count": 0.0,
            "candidate_original_completion_tokens": 0.0,
            "answer_tokens_selected": 0.0,
            "reasoning_tokens_selected": 0.0,
            "frontier_corrective_tokens_selected": 0.0,
            "frontier_wrong_tokens_selected": 0.0,
        }
    device = input_device(model)
    _synchronize(device)
    total_start = time.perf_counter()
    feature_start = total_start
    # Identical allocations are essential for the preregistered matched
    # Correct-only vs Correct+Wrong ablation.  Frontier rows replace optional
    # correct rows inside each allocation rather than expanding it.
    required_answer_counts = [
        _required_answer_token_count(tokenizer, candidate) for candidate in candidates
    ]
    # Reserve the two first-divergence directions for *both* variants.  The
    # correct-only fit spends those rows on additional correct-trajectory
    # tokens; the signed fit replaces them with correct/wrong frontier rows.
    # This makes matched row/rank identity guaranteed rather than contingent
    # on there happening to be spare optional budget.
    required_token_counts = [count + 2 for count in required_answer_counts]
    token_allocations, allocation_metadata = _answer_aware_token_allocations(
        required_token_counts,
        max_support_tokens=max_support_tokens,
        max_tokens_per_candidate=max_tokens_per_candidate,
    )
    allocation_metadata["required_answer_tokens"] = sum(required_answer_counts)
    allocation_metadata["required_supervision_tokens"] = sum(required_token_counts)
    features = []
    for candidate, token_budget in zip(candidates, token_allocations):
        features.append(
            _candidate_features(
                model,
                tokenizer,
                candidate,
                max_tokens=token_budget,
                hard_negatives=hard_negatives,
                max_length=max_length,
                reasoning_token_weight=reasoning_token_weight,
                answer_token_weight=answer_token_weight,
                frontier_positive_weight=frontier_positive_weight,
                frontier_negative_weight=frontier_negative_weight,
                frontier_max_tokens=frontier_max_tokens,
                signed_frontier=signed_frontier,
            )
        )
    _synchronize(device)
    feature_seconds = time.perf_counter() - feature_start

    hidden = torch.cat([item["hidden"] for item in features], dim=0).float()
    labels = torch.cat([item["labels"] for item in features], dim=0).long()
    top_ids = torch.cat([item["top_ids"] for item in features], dim=0).long()
    top_probs = torch.cat([item["top_probs"] for item in features], dim=0).float()
    target_probs = torch.cat([item["target_probs"] for item in features], dim=0).float()
    target_log_probs = torch.cat(
        [item["target_log_probs"] for item in features], dim=0
    ).float()
    row_weights = torch.cat([item["row_weights"] for item in features], dim=0).float()
    row_directions = torch.cat(
        [item["row_directions"] for item in features], dim=0
    ).float()
    row_kinds = torch.cat([item["row_kinds"] for item in features], dim=0).long()
    pair_positive_ids = torch.cat(
        [item["pair_positive_ids"] for item in features], dim=0
    ).long()
    pair_negative_ids = torch.cat(
        [item["pair_negative_ids"] for item in features], dim=0
    ).long()
    pair_base_margins = torch.cat(
        [item["pair_base_margins"] for item in features], dim=0
    ).float()
    residual, vocab_ids = _build_sparse_residual(
        labels,
        top_ids,
        top_probs,
        target_probs,
        residual_step_size,
        row_weights=row_weights,
        row_directions=row_directions,
        pair_positive_ids=pair_positive_ids,
        pair_negative_ids=pair_negative_ids,
        pair_base_margins=pair_base_margins,
        frontier_target_margin=frontier_target_margin,
        negative_probability_floor=frontier_negative_probability_floor,
    )

    # Scaling keeps the kernel O(1) across hidden widths.  Applying sqrt(w) to
    # both H and R exactly implements sum_i w_i ||h_i DeltaW-r_i||^2 rather
    # than changing the residual target by the weight.  This distinction is
    # essential for a meaningful fixed 1.0-logit hard-margin target.
    hidden_scale = math.sqrt(hidden.shape[-1])
    row_sqrt_weights = row_weights.clamp_min(0).sqrt().unsqueeze(-1)
    support_hidden = (hidden / hidden_scale) * row_sqrt_weights
    weighted_residual = residual * row_sqrt_weights
    solve_start = time.perf_counter()
    kernel = support_hidden @ support_hidden.T
    ridge_effective = float(
        ridge_lambda * kernel.diagonal().mean().clamp(min=1e-8).item()
    )
    coefficients = _cholesky_solve(kernel, weighted_residual, ridge_effective)
    # The deployed selected-column head update is
    #   Delta W = support_hidden.T @ coefficients / hidden_scale.
    # Compute its Frobenius norm without materializing hidden_size x vocab.
    update_norm_sq = torch.sum(coefficients * (kernel @ coefficients)) / (
        hidden_scale**2
    )
    unclipped_update_frobenius_norm = float(update_norm_sq.clamp(min=0).sqrt().item())
    update_norm_was_clipped = unclipped_update_frobenius_norm > max_update_norm
    if update_norm_was_clipped:
        coefficients.mul_(max_update_norm / unclipped_update_frobenius_norm)
        update_frobenius_norm = float(max_update_norm)
    else:
        update_frobenius_norm = unclipped_update_frobenius_norm
    solve_seconds = time.perf_counter() - solve_start
    # Deployment specialization ends here. Diagnostics below are deliberately
    # excluded from the reported adaptation wall time.
    total_seconds = time.perf_counter() - total_start

    adapter = SparseRidgeAdapter(
        support_hidden=support_hidden.to(torch.float16),
        coefficients=coefficients.to(torch.float16),
        vocab_ids=vocab_ids,
        hidden_scale=hidden_scale,
        ridge_lambda_effective=ridge_effective,
        metadata={
            "query_id": query_id,
            "num_candidates": len(candidates),
            "support_tokens": int(hidden.shape[0]),
            "max_support_tokens": max_support_tokens,
            **allocation_metadata,
            "token_allocations": token_allocations,
            "hard_negatives": hard_negatives,
            "residual_step_size": residual_step_size,
            "reasoning_token_weight": reasoning_token_weight,
            "answer_token_weight": answer_token_weight,
            "frontier_positive_weight": frontier_positive_weight,
            "frontier_negative_weight": frontier_negative_weight,
            "frontier_max_tokens": frontier_max_tokens,
            "frontier_negative_probability_floor": (
                frontier_negative_probability_floor
            ),
            "frontier_target_margin": frontier_target_margin,
            "frontier_margin_objective": "paired_first_divergence_hard_margin_v1",
            "row_weighting_mode": "weighted_least_squares_sqrt_v1",
            "signed_frontier": signed_frontier,
            "max_update_norm": max_update_norm,
            "uses_all_candidates": True,
            "fit_check_split": False,
            "teacher_context_sources": (
                [
                    "proposed_candidate_problem",
                    "verified_correct_trajectory",
                    "verified_wrong_trajectory",
                    "verified_error_frontier",
                ]
                if signed_frontier
                else [
                    "proposed_candidate_problem",
                    "verified_correct_trajectory",
                ]
            ),
            "specialization_status": specialization_status,
            "specialization_failure_reason": specialization_failure_reason,
            "specialization_no_op": False,
        },
    )

    # Measure how much the closed-form update fits its own proposed set.
    support_device = device
    h = hidden.to(support_device)
    base_target_lp = target_log_probs.to(support_device)
    delta = adapter.to(support_device).selected_delta(h)
    columns = {
        int(token_id): index for index, token_id in enumerate(vocab_ids.tolist())
    }
    label_columns = torch.tensor(
        [columns[int(label)] for label in labels], device=support_device
    )
    # The exact adapted normalizer requires full logits. This selected-vocab
    # proxy is reported only as a fit diagnostic, never as target evaluation.
    target_delta = delta.gather(-1, label_columns.unsqueeze(-1)).squeeze(-1)
    support_margin_gain = target_delta.mean().item()
    signed_target_delta = target_delta * row_directions.to(support_device)
    signed_support_margin_gain = signed_target_delta.mean().item()

    def _kind_mean(kind: int) -> float:
        mask = row_kinds.to(support_device) == kind
        if not bool(mask.any()):
            return 0.0
        return float(target_delta[mask].mean().item())

    frontier_corrective_gain = _kind_mean(2)
    frontier_wrong_change = _kind_mean(3)

    diagnostic_features = features
    if not signed_frontier:
        # This deliberately happens after ``total_seconds`` was frozen and
        # after the correct-only coefficients were solved.  It is an offline
        # causal diagnostic, not wrong-support supervision.
        diagnostic_features = [
            _frontier_diagnostic_feature(
                model,
                tokenizer,
                candidate,
                hard_negatives=hard_negatives,
                max_length=max_length,
                frontier_max_tokens=frontier_max_tokens,
            )
            for candidate in candidates
        ]

    frontier_base_margins: list[float] = []
    frontier_teacher_margins: list[float] = []
    adapter_columns = {
        int(token_id): index for index, token_id in enumerate(vocab_ids.tolist())
    }
    frontier_margin_rows: list[dict[str, Any]] = []
    for candidate_index, item in enumerate(diagnostic_features):
        if not bool(item["frontier_comparable"].item()):
            continue
        frontier_hidden = item["frontier_hidden"].reshape(1, -1).to(support_device)
        frontier_delta = adapter.selected_delta(frontier_hidden)[0]
        positive_token = int(item["frontier_positive_token"].item())
        negative_token = int(item["frontier_negative_token"].item())
        positive_delta = (
            float(frontier_delta[adapter_columns[positive_token]].item())
            if positive_token in adapter_columns
            else 0.0
        )
        negative_delta = (
            float(frontier_delta[adapter_columns[negative_token]].item())
            if negative_token in adapter_columns
            else 0.0
        )
        base_margin = float(
            item["frontier_base_positive_logit"].item()
            - item["frontier_base_negative_logit"].item()
        )
        frontier_base_margins.append(base_margin)
        frontier_teacher_margins.append(
            base_margin + positive_delta - negative_delta
        )
        frontier_margin_rows.append(
            {
                "frontier_id": str(
                    candidates[candidate_index].get(
                        "candidate_id", f"candidate_{candidate_index:03d}"
                    )
                ),
                "candidate_index": candidate_index,
                "positive_token_id": positive_token,
                "negative_token_id": negative_token,
                "base_positive_logit": float(
                    item["frontier_base_positive_logit"].item()
                ),
                "base_negative_logit": float(
                    item["frontier_base_negative_logit"].item()
                ),
                "teacher_positive_logit_delta": positive_delta,
                "teacher_negative_logit_delta": negative_delta,
                "base_margin": base_margin,
                "teacher_margin": frontier_teacher_margins[-1],
                "margin_gain": positive_delta - negative_delta,
                "crossing_eligible": base_margin <= 0.0,
                "crossed_decision_boundary": (
                    base_margin <= 0.0 and frontier_teacher_margins[-1] > 0.0
                ),
                "regression_eligible": base_margin > 0.0,
                "regressed_decision_boundary": (
                    base_margin > 0.0 and frontier_teacher_margins[-1] <= 0.0
                ),
                "target_margin_attained": (
                    frontier_teacher_margins[-1] >= frontier_target_margin
                ),
            }
        )
    crossing_eligible = sum(margin <= 0.0 for margin in frontier_base_margins)
    crossing_count = sum(
        base <= 0.0 and teacher > 0.0
        for base, teacher in zip(frontier_base_margins, frontier_teacher_margins)
    )
    regression_eligible = sum(margin > 0.0 for margin in frontier_base_margins)
    regression_count = sum(
        base > 0.0 and teacher <= 0.0
        for base, teacher in zip(frontier_base_margins, frontier_teacher_margins)
    )
    target_margin_attainment_count = sum(
        margin >= frontier_target_margin for margin in frontier_teacher_margins
    )
    base_margin_mean = sum(frontier_base_margins) / max(
        len(frontier_base_margins), 1
    )
    teacher_margin_mean = sum(frontier_teacher_margins) / max(
        len(frontier_teacher_margins), 1
    )
    _synchronize(device)
    metrics = {
        "specialization_status": specialization_status,
        "specialization_failure_reason": specialization_failure_reason,
        "specialization_no_op": False,
        "uses_all_candidates": True,
        "specialization_seconds": total_seconds,
        "feature_extraction_seconds": feature_seconds,
        "closed_form_solve_seconds": solve_seconds,
        "support_tokens": float(hidden.shape[0]),
        "requested_max_support_tokens": float(max_support_tokens),
        "allocated_support_token_budget": float(
            allocation_metadata["allocated_support_token_budget"]
        ),
        "required_answer_tokens": float(allocation_metadata["required_answer_tokens"]),
        "required_supervision_tokens": float(
            allocation_metadata["required_supervision_tokens"]
        ),
        "support_budget_expanded": allocation_metadata["support_budget_expanded"],
        "support_budget_overflow_tokens": float(
            allocation_metadata["support_budget_overflow_tokens"]
        ),
        "adapted_vocab_size": float(vocab_ids.numel()),
        "adapter_rank": float(adapter.rank),
        "ridge_lambda_effective": ridge_effective,
        "update_frobenius_norm": update_frobenius_norm,
        "unclipped_update_frobenius_norm": unclipped_update_frobenius_norm,
        "update_norm_was_clipped": update_norm_was_clipped,
        "proposal_fit_target_logit_gain": support_margin_gain,
        "proposal_fit_signed_target_logit_gain": signed_support_margin_gain,
        "frontier_corrective_target_logit_gain": frontier_corrective_gain,
        "frontier_wrong_target_logit_change": frontier_wrong_change,
        "frontier_margin_base_mean": base_margin_mean,
        "frontier_margin_teacher_mean": teacher_margin_mean,
        "frontier_margin_gain_mean": teacher_margin_mean - base_margin_mean,
        "frontier_target_margin": float(frontier_target_margin),
        "frontier_margin_objective": "paired_first_divergence_hard_margin_v1",
        "row_weighting_mode": "weighted_least_squares_sqrt_v1",
        "frontier_comparable_count": float(len(frontier_base_margins)),
        "frontier_target_margin_attainment_count": float(
            target_margin_attainment_count
        ),
        "frontier_target_margin_attainment_rate": float(
            target_margin_attainment_count
        )
        / max(len(frontier_teacher_margins), 1),
        "frontier_margins": frontier_margin_rows,
        "candidate_count": len(candidates),
        "decision_boundary_crossing_count": float(crossing_count),
        "decision_boundary_eligible_count": float(crossing_eligible),
        "decision_boundary_crossing_rate": float(crossing_count)
        / max(crossing_eligible, 1),
        "decision_boundary_crossing_definition": (
            "base_margin<=0 and teacher_margin>0"
        ),
        "decision_boundary_regression_count": float(regression_count),
        "decision_boundary_regression_eligible_count": float(regression_eligible),
        "decision_boundary_regression_rate": float(regression_count)
        / max(regression_eligible, 1),
        "decision_boundary_regression_definition": (
            "base_margin>0 and teacher_margin<=0"
        ),
        "proposal_base_target_nll": float((-base_target_lp).mean().item()),
        "candidate_completion_truncated_count": float(
            sum(float(item["completion_truncated"].item()) for item in features)
        ),
        "candidate_original_completion_tokens": float(
            sum(float(item["original_completion_tokens"].item()) for item in features)
        ),
        "answer_tokens_selected": float(
            sum(float(item["answer_tokens_selected"].item()) for item in features)
        ),
        "reasoning_tokens_selected": float(
            sum(float(item["reasoning_tokens_selected"].item()) for item in features)
        ),
        "frontier_corrective_tokens_selected": float(
            sum(
                float(item["frontier_corrective_tokens_selected"].item())
                for item in features
            )
        ),
        "frontier_wrong_tokens_selected": float(
            sum(
                float(item["frontier_wrong_tokens_selected"].item())
                for item in features
            )
        ),
    }
    return adapter, metrics


def _safe_filename(query_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id).strip("_")
    return (slug[:80] or "query") + "-" + stable_hash(query_id) + ".pt"


def _validate_proposal_rows(rows: list[dict[str, Any]], source_path: str = "") -> None:
    seen_query_ids: set[str] = set()
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError(f"Proposal row is missing query_id in {source_path}")
        if query_id in seen_query_ids:
            raise ValueError(
                f"Duplicate proposal query_id {query_id!r} in {source_path}"
            )
        seen_query_ids.add(query_id)
        problem = str(row.get("problem", "")).strip()
        declared_hash = str(row.get("problem_sha256", "")).strip()
        source = str(row.get("source", "")).strip().lower()
        if (
            not problem
            or not declared_hash
            or stable_hash(problem, 64) != declared_hash
        ):
            raise ValueError(
                f"Proposal {query_id!r} has a missing or invalid problem binding"
            )
        if not source:
            raise ValueError(f"Proposal {query_id!r} is missing its dataset source")
        if row.get("schema_version") != "clean-self-distill-proposals-v5":
            raise ValueError(
                f"Proposal {query_id!r} does not use the mandatory v5 corrective schema"
            )
        for candidate in row.get("specialization_candidates", []):
            _candidate_frontier(candidate)
        validate_proposal_training_binding(
            row, context=f"Proposal in {source_path or '<memory>'}"
        )


def _validate_clean_firewall(row: dict[str, Any]) -> None:
    forbidden = {
        "answer",
        "feedback",
        "ground_truth",
        "label",
        "reference",
        "reference_answer",
        "reference_solution",
        "reward_model",
        "solution",
        "target",
        "target_answer",
        "target_solution",
    }
    exposed = sorted({str(key).strip().casefold() for key in row} & forbidden)
    if exposed:
        raise ValueError(
            f"Proposal {row.get('query_id')} exposes target-level fields {exposed}"
        )
    firewall = row.get("firewall_audit")
    if not isinstance(firewall, dict):
        raise ValueError(f"Proposal {row.get('query_id')} is missing firewall_audit")
    for key in ("target_answer_loaded", "target_solution_loaded"):
        if firewall.get(key) is not False:
            raise ValueError(
                f"Proposal {row.get('query_id')} does not prove {key}=false"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", help="Pinned Hugging Face model revision")
    parser.add_argument("--ridge-lambda", type=float, default=0.1)
    parser.add_argument("--residual-step-size", type=float, default=0.8)
    parser.add_argument("--max-tokens-per-candidate", type=int, default=96)
    parser.add_argument("--max-support-tokens", type=int, default=768)
    parser.add_argument("--hard-negatives", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--reasoning-token-weight", type=float, default=0.25)
    parser.add_argument("--answer-token-weight", type=float, default=1.0)
    parser.add_argument("--frontier-positive-weight", type=float, default=8.0)
    parser.add_argument("--frontier-negative-weight", type=float, default=8.0)
    parser.add_argument("--frontier-max-tokens", type=int, default=24)
    parser.add_argument(
        "--frontier-negative-probability-floor", type=float, default=0.25
    )
    parser.add_argument(
        "--frontier-target-margin",
        type=float,
        default=FRONTIER_TARGET_MARGIN,
        help="Required correct-minus-wrong logit margin at first divergence",
    )
    parser.add_argument(
        "--support-variant",
        choices=("correct_only", "correct_wrong_signed"),
        default="correct_wrong_signed",
        help=(
            "Matched ablation switch: correct_only fits only verified correct "
            "trajectory tokens; correct_wrong_signed also fits the weighted "
            "corrective/wrong frontier rows."
        ),
    )
    parser.add_argument("--max-update-norm", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-specialization-candidates", type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--allow-hindsight-exposure",
        action="store_true",
        help="Permit a proposal manifest marked as using target answer/solution (ablation only)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    rows = list(iter_rows(args.proposals))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    _validate_proposal_rows(rows, args.proposals)
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        training=False,
        revision=args.revision,
    )
    runtime_metadata = collect_runtime_metadata(
        model, model_path=args.model, revision=args.revision or ""
    )
    ridge_config = {
        "ridge_lambda": args.ridge_lambda,
        "residual_step_size": args.residual_step_size,
        "max_tokens_per_candidate": args.max_tokens_per_candidate,
        "max_support_tokens": args.max_support_tokens,
        "num_specialization_candidates": args.num_specialization_candidates,
        "hard_negatives": args.hard_negatives,
        "max_length": args.max_length,
        "reasoning_token_weight": args.reasoning_token_weight,
        "answer_token_weight": args.answer_token_weight,
        "frontier_positive_weight": args.frontier_positive_weight,
        "frontier_negative_weight": args.frontier_negative_weight,
        "frontier_max_tokens": args.frontier_max_tokens,
        "frontier_negative_probability_floor": (
            args.frontier_negative_probability_floor
        ),
        "frontier_target_margin": args.frontier_target_margin,
        "support_variant": args.support_variant,
        "max_update_norm": args.max_update_norm,
    }
    ridge_config_sha256 = canonical_json_sha256(ridge_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    for index, row in enumerate(tqdm(rows, desc="closed-form specialization")):
        query_id = str(row["query_id"])
        (
            specialization_status,
            specialization_failure_reason,
            specialization_no_op,
        ) = validate_specialization_state(row, context=f"Proposal {query_id!r}")
        if not args.allow_hindsight_exposure:
            _validate_clean_firewall(row)
        candidates = list(row.get("specialization_candidates", []))
        if args.num_specialization_candidates is not None:
            candidates = candidates[: args.num_specialization_candidates]
        device = input_device(model)
        memory_baseline = 0.0
        if device.type == "cuda":
            _synchronize(device)
            memory_baseline = float(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
        adapter, metrics = fit_ridge_adapter(
            model,
            tokenizer,
            candidates,
            ridge_lambda=args.ridge_lambda,
            residual_step_size=args.residual_step_size,
            max_tokens_per_candidate=args.max_tokens_per_candidate,
            max_support_tokens=args.max_support_tokens,
            hard_negatives=args.hard_negatives,
            max_length=args.max_length,
            reasoning_token_weight=args.reasoning_token_weight,
            answer_token_weight=args.answer_token_weight,
            frontier_positive_weight=args.frontier_positive_weight,
            frontier_negative_weight=args.frontier_negative_weight,
            frontier_max_tokens=args.frontier_max_tokens,
            frontier_negative_probability_floor=(
                args.frontier_negative_probability_floor
            ),
            frontier_target_margin=args.frontier_target_margin,
            signed_frontier=args.support_variant == "correct_wrong_signed",
            max_update_norm=args.max_update_norm,
            query_id=query_id,
            specialization_status=specialization_status,
            specialization_failure_reason=specialization_failure_reason,
            specialization_no_op=specialization_no_op,
        )
        if device.type == "cuda":
            _synchronize(device)
            peak_memory_bytes = float(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory_bytes = 0.0
        metrics.update(
            {
                "peak_memory_bytes": peak_memory_bytes,
                "specialization_memory_baseline_bytes": memory_baseline,
                "specialization_peak_memory_delta_bytes": max(
                    peak_memory_bytes - memory_baseline, 0.0
                ),
            }
        )
        problem_sha256 = row.get(
            "problem_sha256", stable_hash(str(row.get("problem", "")), 64)
        )
        proposal_training_sha256 = validate_proposal_training_binding(
            row, context=f"Proposal in {args.proposals}"
        )
        source = str(row.get("source", "unknown")).strip().lower()
        adapter.metadata.update(
            {
                "problem_sha256": problem_sha256,
                "proposal_training_sha256": proposal_training_sha256,
                "source": source,
                "model": args.model,
                "model_revision": runtime_metadata.get(
                    "resolved_model_revision", args.revision or ""
                ),
                "ridge_config_sha256": ridge_config_sha256,
            }
        )
        filename = _safe_filename(query_id)
        adapter.save(output_dir / filename)
        manifest = {
            "query_id": query_id,
            "adapter_path": filename,
            "problem_sha256": problem_sha256,
            "proposal_training_sha256": proposal_training_sha256,
            "source": source,
            "model": args.model,
            "model_revision": runtime_metadata.get(
                "resolved_model_revision", args.revision or ""
            ),
            "runtime": runtime_metadata,
            "ridge_config": ridge_config,
            "ridge_config_sha256": ridge_config_sha256,
            **metrics,
        }
        write_jsonl(manifest_path, [manifest], append=index > 0)


if __name__ == "__main__":
    main()
