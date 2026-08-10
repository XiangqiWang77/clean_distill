#!/usr/bin/env python3
"""Train restartable routed-distillation and outcome-RL baselines.

The two methods deliberately share data order, rollout generation, answer
verification, checkpoint format, and resource accounting.  Only their update
objectives differ.  Training labels are required because both methods use
verifiable rewards; held-out labels remain isolated in the existing offline
evaluation path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch

from baselines.objectives import (
    demopsd_reverse_kl,
    grpo_group_advantages,
    grpo_token_loss,
    opsd_generalized_jsd,
    srpo_entropy_weights,
    srpo_route_masks,
    srpo_topk_jsd,
)
from src.clean_self_distill.generation import problem_prompt
from src.clean_self_distill.heldout import (
    load_query_only_manifest,
    load_sealed_labels,
)
from src.clean_self_distill.runtime import (
    backbone_forward,
    collect_runtime_metadata,
    input_device,
    load_hf_model,
    project_logits,
    render_chat,
    unwrap_causal_lm,
)
from src.clean_self_distill.persistent import _decoded_token, _token_partition
from src.opsd_format import extract_boxed_answer, grade_boxed_answer


METHOD_IDS = {
    "demopsd": "baseline:demopsd:exact_full_vocab_v1",
    "grpo": "baseline:outcome_grpo:deepseekmath_v1",
    "opsd": "baseline:opsd:official_generalized_jsd_fixed_teacher_v1",
    "trsd_source": "trsd:privilege_source_projection_v1",
    "srpo": "baseline:srpo:sample_routed_dw_sdpo_v1",
}
CHECKPOINT_SCHEMA_VERSION = "clean-self-distill-persistent-checkpoint-v1"
RUN_SCHEMA_VERSION = "clean-distill-baseline-run-v1"
EPISODE_SCHEMA_VERSION = "clean-distill-baseline-episode-v1"
PRIVILEGED_PROMPT_VERSION = "demopsd-reprompt-correct-rollout-v1"
OPSD_PRIVILEGED_PROMPT_VERSION = "opsd-reference-solution-transition-v1"
PRIVILEGE_SOURCE_PROMPT_VERSION = "trsd-privilege-source-v1"
SRPO_TEACHER_PROMPT_VERSION = "srpo-correct-sibling-v1"
PRIVILEGE_SOURCES = (
    "verified_reference_solution",
    "answer_free_reasoning_method",
    "verifier_critique",
    "execution_solver_feedback",
    "equivalent_prompt_wrappers",
    "style_only_directive",
    "permuted_noninferable_context",
)


def _method_id(method: str, trajectory_projection: bool) -> str:
    value = METHOD_IDS[method]
    if trajectory_projection:
        if method in {"grpo", "srpo"}:
            raise BaselineProtocolError(
                "trajectory projection is outside this routed objective"
            )
        value += ":trajectory_projection_v1"
    return value


class BaselineProtocolError(ValueError):
    """Raised when a baseline run would violate its recorded protocol."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BaselineProtocolError(f"{path} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BaselineProtocolError(
                    f"{path}:{line_number} is not a JSON object"
                )
            rows.append(value)
    return rows


def _load_training_labels(path: str | Path) -> dict[str, dict[str, str]]:
    """Load verified training labels while retaining optional teacher solutions.

    The shared held-out loader intentionally strips every field except the
    answer and problem digest.  Baseline training is the authorized consumer
    of the separate training-label file, so it validates that sealed core and
    then attaches only its explicit reference-solution field.
    """

    sealed = load_sealed_labels(path)
    raw_rows = _read_jsonl(Path(path))
    if len(raw_rows) != len(sealed):
        raise BaselineProtocolError("training label rows disagree with sealed labels")
    for row in raw_rows:
        query_id = str(row.get("query_id", ""))
        if query_id not in sealed:
            raise BaselineProtocolError(f"unknown training label {query_id!r}")
        reference = row.get("reference_solution")
        if reference is not None:
            if not isinstance(reference, str) or not reference.strip():
                raise BaselineProtocolError(
                    f"{query_id} has an invalid reference solution"
                )
            sealed[query_id]["reference_solution"] = reference.strip()
    return sealed


def _tensor_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad or "lora_" in name
    }


def _restore_tensor_state(model, state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise BaselineProtocolError(
            f"checkpoint contains unknown trainable tensors: {missing[:3]}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


def _copy_lora(source, destination) -> None:
    source_parameters = dict(source.named_parameters())
    destination_parameters = dict(destination.named_parameters())
    names = sorted(name for name in source_parameters if "lora_" in name)
    if not names or any(name not in destination_parameters for name in names):
        raise BaselineProtocolError("student and EMA LoRA tensors do not match")
    with torch.no_grad():
        for name in names:
            target = destination_parameters[name]
            target.copy_(
                source_parameters[name].detach().to(
                    device=target.device, dtype=target.dtype
                )
            )


def _ema_update(student, ema_model, rate: float) -> None:
    student_parameters = dict(student.named_parameters())
    ema_parameters = dict(ema_model.named_parameters())
    with torch.no_grad():
        for name, target in ema_parameters.items():
            if "lora_" not in name:
                continue
            if name not in student_parameters:
                raise BaselineProtocolError(
                    f"student is missing EMA LoRA tensor {name!r}"
                )
            source = student_parameters[name]
            if source.shape != target.shape:
                raise BaselineProtocolError(
                    f"student and EMA LoRA tensor shapes differ for {name!r}"
                )
            # A balanced device map can place corresponding student and EMA
            # layers on different GPUs.  EMA tensors are small LoRA weights,
            # so move each source tensor to the target layer's device before
            # applying the in-place update.
            source_on_target = source.detach().to(
                device=target.device, dtype=target.dtype
            )
            target.mul_(1.0 - rate).add_(source_on_target, alpha=rate)


def _model_device_ids(token_ids: torch.Tensor, model) -> torch.Tensor:
    return token_ids.to(input_device(model), dtype=torch.long)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _tokenize_prompt(tokenizer, prompt: str, model) -> torch.Tensor:
    return tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
        "input_ids"
    ].to(input_device(model))


def _response_hidden(
    model,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    *,
    grad: bool,
) -> torch.Tensor:
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    with torch.set_grad_enabled(grad):
        hidden, _ = backbone_forward(
            model,
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            use_cache=False,
        )
    start = int(prompt_ids.shape[1]) - 1
    result = hidden[:, start : start + int(response_ids.shape[1])]
    if int(result.shape[1]) != int(response_ids.shape[1]):
        raise BaselineProtocolError("response hidden states have the wrong length")
    return result


def _sample_group_tokens(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generators: Sequence[torch.Generator],
) -> torch.Tensor:
    """Sample one token per row without row-wise softmax or host syncs."""

    if logits.ndim != 2 or int(logits.shape[0]) != len(generators):
        raise BaselineProtocolError("group logits and generators do not align")
    if temperature <= 0:
        return logits.argmax(dim=-1)
    filtered = logits.float() / float(temperature)
    if 0 < top_k < int(filtered.shape[-1]):
        threshold = torch.topk(filtered, k=top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(
            filtered, descending=True, dim=-1
        )
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(-1, sorted_indices, sorted_logits)
    probabilities = torch.softmax(filtered, dim=-1)
    # Keep one independent generator per rollout, but scan the vocabulary only
    # once for the complete batch. Row-wise torch.multinomial launches one
    # full-vocabulary sampling kernel per rollout and leaves an H100 mostly
    # idle at long decode horizons.
    uniforms = torch.stack(
        [
            torch.rand((), device=probabilities.device, generator=generator)
            for generator in generators
        ]
    )
    cumulative = probabilities.cumsum(dim=-1)
    cumulative[..., -1] = 1.0
    return torch.searchsorted(
        cumulative.contiguous(), uniforms[:, None], right=False
    ).squeeze(-1)


@torch.inference_mode()
def _generate_group(
    model,
    tokenizer,
    *,
    prompt: str,
    seeds: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[tuple[str, torch.Tensor]]:
    """Decode a same-prompt rollout group in one cached batched pass.

    Each row owns an independent CUDA generator, so the rollout seed contract
    remains explicit even though decoder work is vectorized across the group.
    Finished rows feed EOS while the remaining rows continue and are excluded
    from their returned response.
    """

    if not seeds:
        raise BaselineProtocolError("group generation requires at least one seed")
    was_training = model.training
    model.eval()
    device = input_device(model)
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
        "input_ids"
    ].to(device)
    batch_size = len(seeds)
    next_input = prompt_ids.expand(batch_size, -1).contiguous()
    generated_steps: list[torch.Tensor] = []
    finished: torch.Tensor | None = None
    generators: list[torch.Generator] | None = None
    past_key_values = None
    eos_id = tokenizer.eos_token_id
    filler_id = eos_id if eos_id is not None else int(tokenizer.pad_token_id or 0)

    for step in range(max_new_tokens):
        hidden, past_key_values = backbone_forward(
            model,
            input_ids=next_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = project_logits(model, hidden[:, -1, :])
        if generators is None:
            generators = []
            for seed in seeds:
                generator = torch.Generator(device=logits.device)
                generator.manual_seed(int(seed))
                generators.append(generator)
            finished = torch.zeros(batch_size, device=logits.device, dtype=torch.bool)
        sampled = _sample_group_tokens(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generators=generators,
        )
        if finished is None:
            raise AssertionError("generation state was not initialized")
        active_tokens = torch.where(
            finished,
            torch.full_like(sampled, filler_id),
            sampled,
        )
        generated_steps.append(active_tokens)
        if eos_id is not None:
            finished = finished | active_tokens.eq(int(eos_id))
        next_input = active_tokens[:, None].to(device)
        # Avoid a device-to-host synchronization for every token.  At most 63
        # filler tokens are decoded after the last live row reaches EOS, then
        # removed below; active rollout tokens and seeds are unchanged.
        if eos_id is not None and (step + 1) % 64 == 0:
            if bool(finished.all().item()):
                break

    result: list[tuple[str, torch.Tensor]] = []
    generated = torch.stack(generated_steps, dim=1).cpu()
    for row in generated:
        token_ids = row.tolist()
        if eos_id is not None and int(eos_id) in token_ids:
            token_ids = token_ids[: token_ids.index(int(eos_id)) + 1]
        response_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        response = tokenizer.decode(
            response_ids[0], skip_special_tokens=True
        ).strip()
        result.append((response, response_ids))
    model.train(was_training)
    return result


def _realized_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    values = logits.float()
    label_ids = labels.to(values.device, dtype=torch.long)
    return values.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(
        values, dim=-1
    )


def _target_to_student_kl(
    student_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    student_log_probs = torch.log_softmax(student_logits.detach().float(), dim=-1)
    target_log_probs = torch.log_softmax(target_logits.detach().float(), dim=-1)
    target_probs = target_log_probs.exp()
    return (target_probs * (target_log_probs - student_log_probs)).sum(dim=-1)


def _trajectory_projection_metrics(
    *,
    student_model,
    student_hidden: torch.Tensor,
    raw_target_for_chunk,
    labels: torch.Tensor,
    chunk_size: int,
    enabled: bool,
    kl_budget: float,
    binary_search_steps: int,
    token_partitions: Sequence[str] | None = None,
) -> dict[str, float | bool]:
    """Find one projection coefficient shared by the complete trajectory."""

    token_count = int(student_hidden.shape[1])

    realized_partition_gains: dict[str, list[float]] = {}

    def scan(alpha: float, *, realized: bool = False) -> tuple[float, float]:
        total_kl = 0.0
        target_minus_student = 0.0
        with torch.no_grad():
            for start in range(0, token_count, chunk_size):
                stop = min(start + chunk_size, token_count)
                student_logits = project_logits(
                    student_model, student_hidden[:, start:stop].detach()
                )
                raw_target = raw_target_for_chunk(student_logits, start, stop)
                target_logits = (
                    (1.0 - float(alpha)) * student_logits.detach()
                    + float(alpha) * raw_target.detach()
                )
                total_kl += float(
                    _target_to_student_kl(student_logits, target_logits).sum().item()
                )
                if realized:
                    chunk_labels = labels[:, start:stop]
                    gains = (
                        _realized_log_probs(target_logits, chunk_labels)
                        - _realized_log_probs(student_logits, chunk_labels)
                    )
                    target_minus_student += float(gains.sum().item())
                    if token_partitions is not None:
                        for partition, gain in zip(
                            token_partitions[start:stop],
                            gains.detach().reshape(-1).cpu().tolist(),
                        ):
                            realized_partition_gains.setdefault(partition, []).append(
                                float(gain)
                            )
                del student_logits, raw_target, target_logits
        return total_kl / token_count, target_minus_student / token_count

    raw_kl, _ = scan(1.0)
    if not enabled or raw_kl <= kl_budget:
        alpha = 1.0
        achieved = raw_kl
    else:
        low, high = 0.0, 1.0
        for _ in range(binary_search_steps):
            mid = (low + high) / 2.0
            value, _ = scan(mid)
            if value <= kl_budget:
                low = mid
            else:
                high = mid
        alpha = low
        achieved, _ = scan(alpha)
    _, realized_advantage = scan(alpha, realized=True)
    result: dict[str, float | bool] = {
        "enabled": bool(enabled),
        "alpha": float(alpha),
        "raw_target_kl": float(raw_kl),
        "projected_target_kl": float(achieved),
        "cap_active": bool(enabled and alpha < 1.0),
        "realized_target_logprob_advantage": float(realized_advantage),
    }
    for partition in ("task", "style", "other"):
        values = realized_partition_gains.get(partition, [])
        result[f"{partition}_token_logprob_gain"] = (
            sum(values) / len(values) if values else 0.0
        )
        result[f"{partition}_token_count"] = float(len(values))
    return result


def _privileged_prompt(tokenizer, problem: str, privileged_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                f"Question:\n{problem.strip()}\n\n"
                f"Privileged Information:\n{privileged_text.strip()}\n\n"
                "Student Response:\n"
            ),
        }
    ]
    return render_chat(tokenizer, messages, add_generation_prompt=True)


def _srpo_teacher_prompt(tokenizer, problem: str, sibling_text: str) -> str:
    """Render the correct-sibling teacher-information template from SRPO."""

    messages = [
        {
            "role": "user",
            "content": (
                f"{problem.strip()}\nCorrect solution:\n{sibling_text.strip()}\n"
                "Correctly solve the original question."
            ),
        }
    ]
    return render_chat(tokenizer, messages, add_generation_prompt=True)


def _opsd_privileged_prompt(tokenizer, problem: str, privileged_text: str) -> str:
    """Reference-conditioned teacher prompt from the released OPSD template."""

    messages = [
        {
            "role": "user",
            "content": (
                f"Problem: {problem.strip()}\n\n"
                "Here is a reference solution to this problem:\n"
                "=== Reference Solution Begin ===\n"
                f"{privileged_text.strip()}\n"
                "=== Reference Solution End ===\n\n"
                "After reading the reference solution above, make sure you truly "
                "understand the reasoning behind each step; do not copy or paraphrase "
                "it. Using your own words and independent reasoning, derive the same "
                "final answer. Think step by step, and put your final answer within "
                "\\boxed{}."
            ),
        }
    ]
    return render_chat(tokenizer, messages, add_generation_prompt=True)


def _source_privileged_prompt(
    tokenizer,
    problem: str,
    privileged_text: str,
    *,
    source: str,
    wrapper_index: int,
) -> str:
    """Render one auditable privileged-context construction."""

    if source == "equivalent_prompt_wrappers":
        wrappers = (
            "Use the following verified worked solution as private guidance. Derive the answer independently.",
            "Consult this correct solution privately, then solve the problem in your own reasoning and wording.",
            "The private material below is a verified derivation. Internalize it and produce an independent solution.",
        )
        instruction = wrappers[int(wrapper_index) % len(wrappers)]
    elif source == "verified_reference_solution":
        instruction = (
            "Use the following verified reference solution as private guidance, "
            "then independently derive the answer."
        )
    elif source == "answer_free_reasoning_method":
        instruction = "Apply the following private answer-free reasoning method."
    elif source == "verifier_critique":
        instruction = "Use this private verifier critique to revise the reasoning."
    elif source == "execution_solver_feedback":
        instruction = "Use this private execution/verifier feedback to reconstruct a correct solution."
    elif source == "style_only_directive":
        instruction = "Follow this private response-style directive; it contains no task answer."
    elif source == "permuted_noninferable_context":
        instruction = (
            "The following private context was deterministically permuted from a "
            "different problem and contains no inferable answer to this task."
        )
    else:
        raise BaselineProtocolError(f"unknown privilege source {source!r}")
    messages = [
        {
            "role": "system",
            "content": f"{instruction}\n\n{privileged_text.strip()}",
        },
        {
            "role": "user",
            "content": (
                f"{problem.strip()}\n\nReason step by step and put the final answer "
                "within \\boxed{}."
            ),
        },
    ]
    return render_chat(tokenizer, messages, add_generation_prompt=True)


def _privilege_source_text(
    *,
    source: str,
    reference_solution: str,
    permuted_reference_solution: str,
    answer: str,
    response: str,
    response_correct: bool,
) -> str:
    if source in {"verified_reference_solution", "equivalent_prompt_wrappers"}:
        return reference_solution
    if source == "answer_free_reasoning_method":
        return (
            "Decompose the problem into explicit subgoals; track constraints and "
            "invariants; check boundary cases; verify the route independently. Use "
            "only the problem statement and do not assume an answer."
        )
    if source == "verifier_critique":
        verdict = "accepted" if response_correct else "rejected"
        return (
            f"The deterministic answer verifier {verdict} the candidate. Re-check "
            "the derivation, locate the earliest unsupported step, and recompute the "
            "final answer. No reference answer is supplied."
        )
    if source == "execution_solver_feedback":
        candidate = extract_boxed_answer(response) or "<unparsed>"
        return (
            f"Deterministic checker result: candidate={candidate!r}; verified "
            f"answer={answer!r}. Reconstruct a valid derivation that reaches the "
            "verified result."
        )
    if source == "style_only_directive":
        return (
            "Write in concise academic prose. Use explicit transitions, numbered "
            "steps, and a final verification sentence. Do not add task facts."
        )
    if source == "permuted_noninferable_context":
        return permuted_reference_solution
    raise BaselineProtocolError(f"unknown privilege source {source!r}")


def _ordinary_prompt(tokenizer, problem: str, *, disable_thinking: bool) -> str:
    if not disable_thinking:
        return problem_prompt(tokenizer, problem)
    messages = [
        {
            "role": "user",
            "content": (
                f"{problem.strip()}\n\nPlease reason step by step, and put your "
                "final answer within \\boxed{}."
            ),
        }
    ]
    return render_chat(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _fit_privileged_prompt(
    tokenizer,
    model,
    *,
    problem: str,
    correct_ids: torch.Tensor,
    response_length: int,
    max_sequence_tokens: int,
    privileged_max_tokens: int,
    prompt_builder=_privileged_prompt,
) -> tuple[str, int]:
    """Keep the tail of a correct rollout while respecting the sequence cap."""

    available = min(int(correct_ids.numel()), int(privileged_max_tokens))
    while available > 0:
        privileged_text = tokenizer.decode(
            correct_ids.reshape(-1)[-available:], skip_special_tokens=True
        ).strip()
        prompt = prompt_builder(tokenizer, problem, privileged_text)
        prompt_length = int(
            tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
                "input_ids"
            ].numel()
        )
        if prompt_length + response_length <= max_sequence_tokens:
            return prompt, available
        available //= 2
    raise BaselineProtocolError(
        "privileged context cannot fit beside the student response"
    )


def _stream_demopsd_backward(
    *,
    student_model,
    ema_model,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    ema_student_hidden: torch.Tensor,
    chunk_size: int,
    beta: float,
    alpha_max: float,
    temperature: float,
    loss_scale: float,
    projection_alpha: float = 1.0,
) -> dict[str, float]:
    token_count = int(student_hidden.shape[1])
    hidden_gradient = torch.empty_like(student_hidden)
    total_loss = total_jsd = total_alpha = total_kl = 0.0
    for start in range(0, token_count, chunk_size):
        stop = min(start + chunk_size, token_count)
        chunk_tokens = stop - start
        hidden_leaf = student_hidden[:, start:stop].detach().requires_grad_(True)
        student_logits = project_logits(student_model, hidden_leaf)
        with torch.no_grad():
            teacher_logits = project_logits(ema_model, teacher_hidden[:, start:stop])
            reference_logits = project_logits(
                ema_model, ema_student_hidden[:, start:stop]
            )
        raw_loss, metrics = demopsd_reverse_kl(
            student_logits,
            teacher_logits,
            reference_logits,
            beta=beta,
            alpha_max=alpha_max,
            temperature=temperature,
        )
        if projection_alpha < 1.0:
            target_logits = (
                (1.0 - float(projection_alpha)) * student_logits.detach().float()
                + float(projection_alpha) * metrics.target_log_probs
            )
            student_log_probs = torch.log_softmax(
                student_logits.float() / float(temperature), dim=-1
            )
            target_log_probs = torch.log_softmax(
                target_logits.detach() / float(temperature), dim=-1
            )
            per_token_kl = (
                student_log_probs.exp() * (student_log_probs - target_log_probs)
            ).sum(dim=-1)
            chunk_loss = per_token_kl.mean() * (float(temperature) ** 2)
        else:
            target_logits = metrics.target_log_probs
            chunk_loss = raw_loss
            per_token_kl = metrics.per_token_kl
        weight = (chunk_tokens / token_count) * loss_scale
        (chunk_gradient,) = torch.autograd.grad(chunk_loss * weight, hidden_leaf)
        hidden_gradient[:, start:stop].copy_(chunk_gradient)
        total_loss += float(chunk_loss.detach().item()) * chunk_tokens
        total_jsd += float(metrics.jsd.float().sum().item())
        total_alpha += float(metrics.alpha.float().sum().item())
        total_kl += float(per_token_kl.float().sum().item())
        del student_logits, teacher_logits, reference_logits, chunk_loss, raw_loss
        del target_logits, per_token_kl, metrics
        del hidden_leaf, chunk_gradient
    student_hidden.backward(hidden_gradient)
    del hidden_gradient
    return {
        "loss": total_loss / token_count,
        "mean_jsd": total_jsd / token_count,
        "mean_alpha": total_alpha / token_count,
        "mean_kl": total_kl / token_count,
    }


def _stream_opsd_backward(
    *,
    student_model,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    chunk_size: int,
    beta: float,
    element_clip: float,
    temperature: float,
    loss_scale: float,
    projection_alpha: float = 1.0,
) -> dict[str, float]:
    token_count = int(student_hidden.shape[1])
    hidden_gradient = torch.empty_like(student_hidden)
    total_loss = total_jsd = total_student_entropy = total_target_entropy = 0.0
    for start in range(0, token_count, chunk_size):
        stop = min(start + chunk_size, token_count)
        chunk_tokens = stop - start
        hidden_leaf = student_hidden[:, start:stop].detach().requires_grad_(True)
        student_logits = project_logits(student_model, hidden_leaf)
        with torch.no_grad():
            raw_target_logits = project_logits(
                student_model, teacher_hidden[:, start:stop]
            )
            target_logits = (
                (1.0 - float(projection_alpha)) * student_logits.detach()
                + float(projection_alpha) * raw_target_logits
            )
        chunk_loss, metrics = opsd_generalized_jsd(
            student_logits,
            target_logits,
            beta=beta,
            temperature=temperature,
            element_clip=element_clip,
        )
        weight = (chunk_tokens / token_count) * loss_scale
        (chunk_gradient,) = torch.autograd.grad(chunk_loss * weight, hidden_leaf)
        hidden_gradient[:, start:stop].copy_(chunk_gradient)
        total_loss += float(chunk_loss.detach().item()) * chunk_tokens
        total_jsd += float(metrics.per_token_jsd.float().sum().item())
        total_student_entropy += float(metrics.student_entropy.float().sum().item())
        total_target_entropy += float(metrics.target_entropy.float().sum().item())
        del student_logits, raw_target_logits, target_logits, chunk_loss, metrics
        del hidden_leaf, chunk_gradient
    student_hidden.backward(hidden_gradient)
    del hidden_gradient
    return {
        "loss": total_loss / token_count,
        "mean_jsd": total_jsd / token_count,
        "student_entropy": total_student_entropy / token_count,
        "target_entropy": total_target_entropy / token_count,
    }


def _srpo_raw_weight_statistics(
    *,
    ema_model,
    teacher_hidden: torch.Tensor,
    chunk_size: int,
    entropy_beta: float,
) -> tuple[float, int, float]:
    """Scan one teacher trajectory without retaining vocabulary logits."""

    token_count = int(teacher_hidden.shape[1])
    raw_sum = entropy_sum = 0.0
    for start in range(0, token_count, chunk_size):
        stop = min(start + chunk_size, token_count)
        with torch.no_grad():
            teacher_logits = project_logits(ema_model, teacher_hidden[:, start:stop])
            entropy, raw_weight = srpo_entropy_weights(
                teacher_logits,
                beta=entropy_beta,
                normalizer=1.0,
            )
        raw_sum += float(raw_weight.sum().item())
        entropy_sum += float(entropy.sum().item())
        del teacher_logits, entropy, raw_weight
    return raw_sum, token_count, entropy_sum


def _stream_srpo_sdpo_backward(
    *,
    student_model,
    ema_model,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    chunk_size: int,
    top_k: int,
    entropy_beta: float,
    jsd_alpha: float,
    weight_normalizer: float,
    loss_scale: float,
) -> dict[str, float]:
    """Backpropagate one routed DW-SDPO trajectory in vocabulary chunks."""

    token_count = int(student_hidden.shape[1])
    hidden_gradient = torch.empty_like(student_hidden)
    total_loss = total_jsd = total_entropy = total_weight = 0.0
    for start in range(0, token_count, chunk_size):
        stop = min(start + chunk_size, token_count)
        chunk_tokens = stop - start
        hidden_leaf = student_hidden[:, start:stop].detach().requires_grad_(True)
        student_logits = project_logits(student_model, hidden_leaf)
        with torch.no_grad():
            teacher_logits = project_logits(ema_model, teacher_hidden[:, start:stop])
        chunk_loss, objective = srpo_topk_jsd(
            student_logits,
            teacher_logits,
            top_k=top_k,
            entropy_beta=entropy_beta,
            jsd_alpha=jsd_alpha,
            weight_normalizer=weight_normalizer,
        )
        weight = (chunk_tokens / token_count) * loss_scale
        (chunk_gradient,) = torch.autograd.grad(chunk_loss * weight, hidden_leaf)
        hidden_gradient[:, start:stop].copy_(chunk_gradient)
        total_loss += float(chunk_loss.detach().item()) * chunk_tokens
        total_jsd += float(objective.per_token_jsd.sum().item())
        total_entropy += float(objective.teacher_entropy.sum().item())
        total_weight += float(objective.normalized_entropy_weight.sum().item())
        del student_logits, teacher_logits, chunk_loss, objective
        del hidden_leaf, chunk_gradient
    student_hidden.backward(hidden_gradient)
    del hidden_gradient
    return {
        "loss": total_loss / token_count,
        "mean_jsd": total_jsd / token_count,
        "teacher_entropy": total_entropy / token_count,
        "normalized_entropy_weight": total_weight / token_count,
    }


def _stream_reverse_kl_backward(
    *,
    model,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    chunk_size: int,
    temperature: float,
    loss_scale: float,
    projection_alpha: float,
) -> dict[str, float]:
    """Backpropagate exact reverse KL to one projected contextual target."""

    token_count = int(student_hidden.shape[1])
    hidden_gradient = torch.empty_like(student_hidden)
    total_loss = total_kl = 0.0
    for start in range(0, token_count, chunk_size):
        stop = min(start + chunk_size, token_count)
        chunk_tokens = stop - start
        hidden_leaf = student_hidden[:, start:stop].detach().requires_grad_(True)
        student_logits = project_logits(model, hidden_leaf)
        with torch.no_grad():
            raw_target_logits = project_logits(model, teacher_hidden[:, start:stop])
            target_logits = (
                (1.0 - float(projection_alpha)) * student_logits.detach()
                + float(projection_alpha) * raw_target_logits
            )
            target_log_probs = torch.log_softmax(
                target_logits.float() / float(temperature), dim=-1
            )
        student_log_probs = torch.log_softmax(
            student_logits.float() / float(temperature), dim=-1
        )
        per_token_kl = (
            student_log_probs.exp() * (student_log_probs - target_log_probs)
        ).sum(dim=-1)
        chunk_loss = per_token_kl.mean() * (float(temperature) ** 2)
        weight = (chunk_tokens / token_count) * loss_scale
        (chunk_gradient,) = torch.autograd.grad(chunk_loss * weight, hidden_leaf)
        hidden_gradient[:, start:stop].copy_(chunk_gradient)
        total_loss += float(chunk_loss.detach().item()) * chunk_tokens
        total_kl += float(per_token_kl.detach().sum().item())
        del student_logits, raw_target_logits, target_logits, target_log_probs
        del student_log_probs, per_token_kl, chunk_loss, hidden_leaf, chunk_gradient
    student_hidden.backward(hidden_gradient)
    del hidden_gradient
    return {
        "loss": total_loss / token_count,
        "mean_kl": total_kl / token_count,
    }


def _stream_grpo_backward(
    *,
    model,
    student_hidden: torch.Tensor,
    reference_hidden: torch.Tensor,
    labels: torch.Tensor,
    advantage: float,
    chunk_size: int,
    clip_epsilon: float,
    clip_epsilon_high: float | None,
    kl_coefficient: float,
    loss_scale: float,
) -> dict[str, float]:
    token_count = int(student_hidden.shape[1])
    hidden_gradient = torch.empty_like(student_hidden)
    total_loss = total_kl = total_ratio = 0.0
    for start in range(0, token_count, chunk_size):
        stop = min(start + chunk_size, token_count)
        chunk_tokens = stop - start
        hidden_leaf = student_hidden[:, start:stop].detach().requires_grad_(True)
        current_logits = project_logits(model, hidden_leaf)
        current_log_probs = _realized_log_probs(
            current_logits, labels[:, start:stop]
        )
        # One policy update follows each exploration group, so the old policy
        # is exactly the pre-update current policy at ratio value one.
        old_log_probs = current_log_probs.detach()
        with torch.no_grad():
            reference_logits = project_logits(
                model, reference_hidden[:, start:stop]
            )
            reference_log_probs = _realized_log_probs(
                reference_logits, labels[:, start:stop]
            )
        chunk_loss, metrics = grpo_token_loss(
            current_log_probs,
            old_log_probs,
            reference_log_probs,
            advantage,
            clip_epsilon=clip_epsilon,
            clip_epsilon_high=clip_epsilon_high,
            kl_coefficient=kl_coefficient,
        )
        weight = (chunk_tokens / token_count) * loss_scale
        (chunk_gradient,) = torch.autograd.grad(chunk_loss * weight, hidden_leaf)
        hidden_gradient[:, start:stop].copy_(chunk_gradient)
        total_loss += float(chunk_loss.detach().item()) * chunk_tokens
        total_kl += float(metrics["per_token_kl"].float().sum().item())
        total_ratio += float(metrics["ratio"].float().sum().item())
        del current_logits, current_log_probs, old_log_probs
        del reference_logits, reference_log_probs, chunk_loss, metrics
        del hidden_leaf, chunk_gradient
    student_hidden.backward(hidden_gradient)
    del hidden_gradient
    return {
        "loss": total_loss / token_count,
        "mean_kl": total_kl / token_count,
        "mean_ratio": total_ratio / token_count,
    }


def _save_checkpoint(
    *,
    model,
    ema_model,
    tokenizer,
    optimizer,
    output_dir: Path,
    completed_episodes: int,
    method: str,
    args: argparse.Namespace,
    identity: Mapping[str, str],
    cumulative: Mapping[str, Any],
) -> Path:
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    target = checkpoints / f"episode_{completed_episodes:04d}"
    if target.exists():
        existing = _read_json(target / "checkpoint_manifest.json")
        if existing.get("run_identity_sha256") == identity["run_identity_sha256"]:
            return target
        raise BaselineProtocolError(f"refusing to overwrite mismatched {target}")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint.", dir=checkpoints))
    try:
        model.save_pretrained(temporary)
        tokenizer.save_pretrained(temporary)
        state = {
            "completed_episodes": completed_episodes,
            "student_trainable_state": _tensor_state(model),
            "ema_trainable_state": (
                _tensor_state(ema_model) if ema_model is not None else None
            ),
            "optimizer_state": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            **dict(identity),
        }
        torch.save(state, temporary / "trainer_state.pt")
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "scientific",
            "checkpoint_episode": completed_episodes,
            "completed_episodes": completed_episodes,
            "branch": method,
            "variant": "paper_reimplementation",
            "method_id": _method_id(method, args.trajectory_projection),
            "model_id": args.model_id,
            "model_revision": args.revision,
            "cumulative_audit": dict(cumulative),
            **dict(identity),
        }
        (temporary / "checkpoint_manifest.json").write_text(
            _canonical_json(manifest) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _atomic_json(
        checkpoints / "LATEST.json",
        {
            "checkpoint_dir": target.name,
            "completed_episodes": completed_episodes,
            **dict(identity),
        },
    )
    return target


def _load_resume(
    *,
    model,
    ema_model,
    optimizer,
    output_dir: Path,
    identity: Mapping[str, str],
) -> int:
    latest = _read_json(output_dir / "checkpoints" / "LATEST.json")
    if latest.get("run_identity_sha256") != identity["run_identity_sha256"]:
        raise BaselineProtocolError("LATEST checkpoint belongs to another run")
    checkpoint = output_dir / "checkpoints" / str(latest["checkpoint_dir"])
    state = torch.load(
        checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False
    )
    if state.get("run_identity_sha256") != identity["run_identity_sha256"]:
        raise BaselineProtocolError("trainer state belongs to another run")
    _restore_tensor_state(model, state["student_trainable_state"])
    if ema_model is not None:
        if state.get("ema_trainable_state") is None:
            raise BaselineProtocolError("EMA baseline resume lacks EMA state")
        _restore_tensor_state(ema_model, state["ema_trainable_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    random.setstate(state["python_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and state.get("cuda_random_state"):
        # A restart may deliberately use a more efficient device placement
        # than the checkpointing job (for example 2 GPUs instead of a 4-way
        # balanced map). Restore the RNG streams that still have visible
        # devices; rollout sampling itself uses explicit per-rollout seeds.
        for device_index, rng_state in enumerate(
            state["cuda_random_state"][: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(rng_state, device=device_index)
    return int(state["completed_episodes"])


def _validate_inputs(
    queries: Sequence[Mapping[str, str]],
    labels: Mapping[str, Mapping[str, str]],
    episodes: int,
) -> list[dict[str, str]]:
    if episodes <= 0 or episodes > len(queries):
        raise BaselineProtocolError(
            f"episodes must lie in [1, {len(queries)}], got {episodes}"
        )
    selected = [dict(row) for row in queries[:episodes]]
    for row in selected:
        label = labels.get(row["query_id"])
        if label is None or label["problem_sha256"] != row["problem_sha256"]:
            raise BaselineProtocolError(
                f"training label does not match {row['query_id']}"
            )
    return selected


def _cumulative(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {
        "teacher_positions": sum(
            int(row.get("teacher_positions", 0)) for row in rows
        ),
        "hindsight_exposed_positions": sum(
            int(row.get("hindsight_exposed_positions", 0)) for row in rows
        ),
        "compared_positions": sum(
            int(row.get("compared_positions", 0)) for row in rows
        ),
        "exact_context_positions": 0,
        "prompt_count": len(rows),
        "rollout_count": sum(int(row["rollout_count"]) for row in rows),
        "correct_rollout_count": sum(
            int(row["correct_rollout_count"]) for row in rows
        ),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
        "optimized_rollout_count": sum(
            int(row["optimized_rollout_count"]) for row in rows
        ),
        "optimizer_step_count": sum(bool(row["optimizer_step"]) for row in rows),
        "projection_cap_hits": sum(
            int(row.get("projection_cap_hits", 0)) for row in rows
        ),
        "guard_rejections": sum(int(row.get("guard_rejections", 0)) for row in rows),
        "srpo_sdpo_rollout_count": sum(
            int(row.get("srpo_sdpo_rollout_count", 0)) for row in rows
        ),
        "srpo_grpo_rollout_count": sum(
            int(row.get("srpo_grpo_rollout_count", 0)) for row in rows
        ),
        "srpo_teacher_available_count": sum(
            int(row.get("srpo_teacher_available_count", 0)) for row in rows
        ),
        "srpo_sdpo_tokens": sum(
            int(row.get("srpo_sdpo_tokens", 0)) for row in rows
        ),
        "srpo_grpo_tokens": sum(
            int(row.get("srpo_grpo_tokens", 0)) for row in rows
        ),
        "peak_cuda_allocated_bytes": max(
            (
                int(row.get("resource_usage", {}).get("cuda_peak_memory_allocated_bytes", 0))
                for row in rows
            ),
            default=0,
        ),
    }
    result["phase_seconds"] = {
        phase: sum(float(row.get("phase_seconds", {}).get(phase, 0.0)) for row in rows)
        for phase in ("rollout", "teacher", "target", "update")
    }
    return result


def _episode(
    *,
    model,
    ema_model,
    tokenizer,
    optimizer,
    query: Mapping[str, str],
    answer: str,
    reference_solution: str,
    permuted_reference_solution: str,
    stream_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _sync_cuda()
    cuda_baseline = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    phase_seconds = {
        "rollout": 0.0,
        "teacher": 0.0,
        "target": 0.0,
        "update": 0.0,
    }
    device = input_device(model)
    normal_prompt = _ordinary_prompt(
        tokenizer, query["problem"], disable_thinking=args.disable_thinking
    )
    normal_prompt_ids = _tokenize_prompt(tokenizer, normal_prompt, model)
    rollout_budget = min(
        args.max_rollout_tokens,
        args.max_sequence_tokens - int(normal_prompt_ids.shape[1]),
    )
    if rollout_budget <= 0:
        raise BaselineProtocolError("normal prompt leaves no rollout budget")

    rollouts: list[dict[str, Any]] = []
    episode_seed = int(args.seed) + stream_index * 100_003
    all_seeds = [episode_seed + index for index in range(args.group_size)]
    _sync_cuda()
    phase_started = time.perf_counter()
    for batch_start in range(0, args.group_size, args.generation_batch_size):
        batch_seeds = all_seeds[
            batch_start : batch_start + args.generation_batch_size
        ]
        generated = _generate_group(
            model,
            tokenizer,
            prompt=normal_prompt,
            seeds=batch_seeds,
            max_new_tokens=rollout_budget,
            temperature=args.train_temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        for seed, (response, ids) in zip(batch_seeds, generated):
            if ids.numel() == 0:
                raise BaselineProtocolError("generation produced an empty response")
            parsed = extract_boxed_answer(response)
            correct = bool(grade_boxed_answer(parsed, answer))
            rollouts.append(
                {
                    "seed": seed,
                    "response": response,
                    "response_ids": ids,
                    "correct": correct,
                }
            )
    _sync_cuda()
    phase_seconds["rollout"] += time.perf_counter() - phase_started
    rewards = torch.tensor(
        [float(item["correct"]) for item in rollouts], dtype=torch.float32
    )
    effective_learning_rate = float(args.learning_rate)
    if args.method == "srpo" and args.srpo_warmup_steps > 0:
        effective_learning_rate *= min(
            float(stream_index + 1) / float(args.srpo_warmup_steps), 1.0
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = effective_learning_rate
    optimizer.zero_grad(set_to_none=True)
    metrics: list[dict[str, float]] = []
    optimized_rollouts = 0
    privileged_index: int | None = None
    privileged_tokens = 0
    teacher_positions = 0
    hindsight_positions = 0
    guard_rejections = 0
    privilege_payload_hashes: list[str] = []
    srpo_sdpo_rollouts = 0
    srpo_grpo_rollouts = 0
    srpo_teacher_available_count = 0
    srpo_sdpo_tokens = 0
    srpo_grpo_tokens = 0
    srpo_entropy_weight_normalizer: float | None = None

    if args.method == "srpo":
        correct_indices = [
            index for index, item in enumerate(rollouts) if item["correct"]
        ]
        has_teacher = bool(correct_indices)
        teacher_available = torch.tensor(
            [
                any(other != index for other in correct_indices)
                for index in range(len(rollouts))
            ],
            dtype=torch.bool,
        )
        srpo_teacher_available_count = int(teacher_available.sum().item())
        sdpo_mask, grpo_mask = srpo_route_masks(rewards.bool(), teacher_available)
        srpo_sdpo_rollouts = int(sdpo_mask.sum().item())
        srpo_grpo_rollouts = int(grpo_mask.sum().item())
        teacher_prompts: dict[int, str] = {}
        if has_teacher:
            privileged_index = random.Random(episode_seed + 91_337).choice(
                correct_indices
            )
            correct_ids = rollouts[privileged_index]["response_ids"]
        raw_weight_sum = 0.0
        raw_weight_count = 0

        # SRPO normalizes entropy weights over all valid SDPO tokens.  Scan the
        # EMA teacher in chunks first, then recompute hidden states for the
        # backward pass so full-vocabulary logits are never retained.
        for rollout_index in sdpo_mask.nonzero(as_tuple=False).reshape(-1).tolist():
            rollout = rollouts[rollout_index]
            response_ids = _model_device_ids(rollout["response_ids"], ema_model)
            privileged_prompt, used_privileged_tokens = _fit_privileged_prompt(
                tokenizer,
                ema_model,
                problem=query["problem"],
                correct_ids=correct_ids,
                response_length=int(response_ids.shape[1]),
                max_sequence_tokens=args.teacher_max_sequence_tokens,
                privileged_max_tokens=args.privileged_max_tokens,
                prompt_builder=_srpo_teacher_prompt,
            )
            teacher_prompts[rollout_index] = privileged_prompt
            privileged_tokens = max(privileged_tokens, used_privileged_tokens)
            teacher_prompt_ids = _tokenize_prompt(
                tokenizer, privileged_prompt, ema_model
            )
            _sync_cuda()
            phase_started = time.perf_counter()
            ema_model.eval()
            with torch.no_grad():
                teacher_hidden = _response_hidden(
                    ema_model,
                    teacher_prompt_ids,
                    response_ids,
                    grad=False,
                )
                raw_sum, count, _ = _srpo_raw_weight_statistics(
                    ema_model=ema_model,
                    teacher_hidden=teacher_hidden,
                    chunk_size=args.token_chunk_size,
                    entropy_beta=args.srpo_entropy_beta,
                )
            _sync_cuda()
            phase_seconds["teacher"] += time.perf_counter() - phase_started
            raw_weight_sum += raw_sum
            raw_weight_count += count
            del teacher_hidden, teacher_prompt_ids, response_ids
        if raw_weight_count:
            srpo_entropy_weight_normalizer = raw_weight_sum / raw_weight_count
            if not math.isfinite(srpo_entropy_weight_normalizer) or (
                srpo_entropy_weight_normalizer <= 0
            ):
                raise FloatingPointError("SRPO entropy normalizer is not positive")

        advantages = grpo_group_advantages(rewards)
        total_routed_tokens = sum(
            int(item["response_ids"].numel()) for item in rollouts
        )
        for rollout_index, rollout in enumerate(rollouts):
            response_ids = _model_device_ids(rollout["response_ids"], model)
            token_count = int(response_ids.numel())
            loss_scale = token_count / total_routed_tokens
            _sync_cuda()
            phase_started = time.perf_counter()
            model.train()
            student_hidden = _response_hidden(
                model, normal_prompt_ids, response_ids, grad=True
            )
            if bool(sdpo_mask[rollout_index].item()):
                teacher_prompt_ids = _tokenize_prompt(
                    tokenizer, teacher_prompts[rollout_index], ema_model
                )
                ema_response_ids = response_ids.to(input_device(ema_model))
                ema_model.eval()
                with torch.no_grad():
                    teacher_hidden = _response_hidden(
                        ema_model,
                        teacher_prompt_ids,
                        ema_response_ids,
                        grad=False,
                    )
                objective = _stream_srpo_sdpo_backward(
                    student_model=model,
                    ema_model=ema_model,
                    student_hidden=student_hidden,
                    teacher_hidden=teacher_hidden,
                    chunk_size=args.token_chunk_size,
                    top_k=args.srpo_top_k,
                    entropy_beta=args.srpo_entropy_beta,
                    jsd_alpha=args.srpo_jsd_alpha,
                    weight_normalizer=float(srpo_entropy_weight_normalizer),
                    loss_scale=loss_scale,
                )
                teacher_positions += token_count
                hindsight_positions += token_count
                srpo_sdpo_tokens += token_count
                del teacher_hidden, teacher_prompt_ids, ema_response_ids
            else:
                with model.disable_adapter():
                    model.eval()
                    with torch.no_grad():
                        reference_hidden = _response_hidden(
                            model, normal_prompt_ids, response_ids, grad=False
                        )
                model.train()
                objective = _stream_grpo_backward(
                    model=model,
                    student_hidden=student_hidden,
                    reference_hidden=reference_hidden,
                    labels=response_ids,
                    advantage=float(advantages[rollout_index].item()),
                    chunk_size=args.token_chunk_size,
                    clip_epsilon=args.grpo_clip_epsilon,
                    clip_epsilon_high=args.srpo_grpo_clip_epsilon_high,
                    kl_coefficient=args.grpo_kl_coefficient,
                    loss_scale=loss_scale,
                )
                srpo_grpo_tokens += token_count
                del reference_hidden
            metrics.append(objective)
            optimized_rollouts += 1
            _sync_cuda()
            phase_seconds["update"] += time.perf_counter() - phase_started
            del student_hidden, response_ids
        optimizer_step = True
    elif args.method == "demopsd":
        correct_indices = [
            index for index, item in enumerate(rollouts) if item["correct"]
        ]
        if correct_indices:
            privileged_index = random.Random(episode_seed + 91_337).choice(
                correct_indices
            )
            correct_ids = rollouts[privileged_index]["response_ids"]
            for rollout in rollouts:
                response_ids = _model_device_ids(rollout["response_ids"], model)
                privileged_prompt, used_privileged_tokens = _fit_privileged_prompt(
                    tokenizer,
                    ema_model,
                    problem=query["problem"],
                    correct_ids=correct_ids,
                    response_length=int(response_ids.shape[1]),
                    max_sequence_tokens=args.teacher_max_sequence_tokens,
                    privileged_max_tokens=args.privileged_max_tokens,
                )
                privileged_tokens = max(privileged_tokens, used_privileged_tokens)
                teacher_prompt_ids = _tokenize_prompt(
                    tokenizer, privileged_prompt, ema_model
                )
                ema_response_ids = response_ids.to(input_device(ema_model))
                _sync_cuda()
                phase_started = time.perf_counter()
                model.train()
                student_hidden = _response_hidden(
                    model, normal_prompt_ids, response_ids, grad=True
                )
                _sync_cuda()
                phase_seconds["update"] += time.perf_counter() - phase_started

                _sync_cuda()
                phase_started = time.perf_counter()
                ema_model.eval()
                with torch.no_grad():
                    ema_student_hidden = _response_hidden(
                        ema_model,
                        normal_prompt_ids.to(input_device(ema_model)),
                        ema_response_ids,
                        grad=False,
                    )
                    teacher_hidden = _response_hidden(
                        ema_model,
                        teacher_prompt_ids,
                        ema_response_ids,
                        grad=False,
                    )
                _sync_cuda()
                phase_seconds["teacher"] += time.perf_counter() - phase_started

                def raw_target_for_chunk(
                    student_logits: torch.Tensor, start: int, stop: int
                ) -> torch.Tensor:
                    with torch.no_grad():
                        teacher_logits = project_logits(
                            ema_model, teacher_hidden[:, start:stop]
                        )
                        reference_logits = project_logits(
                            ema_model, ema_student_hidden[:, start:stop]
                        )
                        _, target_metrics = demopsd_reverse_kl(
                            student_logits,
                            teacher_logits,
                            reference_logits,
                            beta=args.demopsd_beta,
                            alpha_max=args.demopsd_alpha_max,
                            temperature=args.distill_temperature,
                        )
                    return target_metrics.target_log_probs

                _sync_cuda()
                phase_started = time.perf_counter()
                projection = _trajectory_projection_metrics(
                    student_model=model,
                    student_hidden=student_hidden,
                    raw_target_for_chunk=raw_target_for_chunk,
                    labels=response_ids,
                    chunk_size=args.token_chunk_size,
                    enabled=args.trajectory_projection,
                    kl_budget=args.projection_kl_budget,
                    binary_search_steps=args.projection_binary_search_steps,
                )
                _sync_cuda()
                phase_seconds["target"] += time.perf_counter() - phase_started

                if args.update_guard and float(
                    projection["realized_target_logprob_advantage"]
                ) <= 0.0:
                    guard_rejections += 1
                    del student_hidden, ema_student_hidden, teacher_hidden
                    del response_ids, ema_response_ids, teacher_prompt_ids
                    continue

                _sync_cuda()
                phase_started = time.perf_counter()
                objective = _stream_demopsd_backward(
                    student_model=model,
                    ema_model=ema_model,
                    student_hidden=student_hidden,
                    teacher_hidden=teacher_hidden,
                    ema_student_hidden=ema_student_hidden,
                    chunk_size=args.token_chunk_size,
                    beta=args.demopsd_beta,
                    alpha_max=args.demopsd_alpha_max,
                    temperature=args.distill_temperature,
                    loss_scale=1.0 / args.group_size,
                    projection_alpha=float(projection["alpha"]),
                )
                _sync_cuda()
                phase_seconds["update"] += time.perf_counter() - phase_started
                objective.update(
                    {
                        "projection_alpha": float(projection["alpha"]),
                        "raw_target_kl": float(projection["raw_target_kl"]),
                        "projected_target_kl": float(
                            projection["projected_target_kl"]
                        ),
                        "projection_cap_active": float(
                            bool(projection["cap_active"])
                        ),
                        "realized_target_logprob_advantage": float(
                            projection["realized_target_logprob_advantage"]
                        ),
                    }
                )
                metrics.append(
                    objective
                )
                length = int(response_ids.numel())
                optimized_rollouts += 1
                teacher_positions += length
                hindsight_positions += length
                del student_hidden, ema_student_hidden, teacher_hidden
                del response_ids, ema_response_ids, teacher_prompt_ids
        optimizer_step = bool(optimized_rollouts)
    elif args.method in {"opsd", "trsd_source"}:
        privilege_source = (
            "verified_reference_solution"
            if args.method == "opsd"
            else args.privilege_source
        )
        if (
            privilege_source
            in {"verified_reference_solution", "equivalent_prompt_wrappers"}
            and not reference_solution.strip()
        ):
            raise BaselineProtocolError(
                f"{query['query_id']} lacks the required reference solution"
            )
        for rollout in rollouts:
            privilege_text = _privilege_source_text(
                source=privilege_source,
                reference_solution=reference_solution,
                permuted_reference_solution=permuted_reference_solution,
                answer=answer,
                response=rollout["response"],
                response_correct=bool(rollout["correct"]),
            )
            privilege_payload_hashes.append(
                hashlib.sha256(privilege_text.encode("utf-8")).hexdigest()
            )
            privilege_ids = tokenizer(
                privilege_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            response_ids = _model_device_ids(rollout["response_ids"], model)
            privileged_prompt, used_privileged_tokens = _fit_privileged_prompt(
                tokenizer,
                model,
                problem=query["problem"],
                correct_ids=privilege_ids,
                response_length=int(response_ids.shape[1]),
                max_sequence_tokens=args.teacher_max_sequence_tokens,
                privileged_max_tokens=args.privileged_max_tokens,
                prompt_builder=(
                    _opsd_privileged_prompt
                    if args.method == "opsd"
                    else lambda tok, problem, text: _source_privileged_prompt(
                        tok,
                        problem,
                        text,
                        source=privilege_source,
                        wrapper_index=stream_index,
                    )
                ),
            )
            privileged_tokens = max(privileged_tokens, used_privileged_tokens)
            teacher_prompt_ids = _tokenize_prompt(tokenizer, privileged_prompt, model)

            _sync_cuda()
            phase_started = time.perf_counter()
            model.train()
            student_hidden = _response_hidden(
                model, normal_prompt_ids, response_ids, grad=True
            )
            _sync_cuda()
            phase_seconds["update"] += time.perf_counter() - phase_started

            _sync_cuda()
            phase_started = time.perf_counter()
            model.eval()
            teacher_context = (
                model.disable_adapter()
                if args.method == "opsd" and args.opsd_teacher_strategy == "fixed"
                else __import__("contextlib").nullcontext()
            )
            with torch.no_grad(), teacher_context:
                teacher_hidden = _response_hidden(
                    model,
                    teacher_prompt_ids,
                    response_ids,
                    grad=False,
                )
            model.train()
            _sync_cuda()
            phase_seconds["teacher"] += time.perf_counter() - phase_started

            def raw_target_for_chunk(
                _student_logits: torch.Tensor, start: int, stop: int
            ) -> torch.Tensor:
                return project_logits(model, teacher_hidden[:, start:stop]).detach()

            token_partitions = (
                [
                    _token_partition(_decoded_token(tokenizer, int(token_id)))
                    for token_id in response_ids.detach().cpu().reshape(-1).tolist()
                ]
                if args.method == "trsd_source"
                else None
            )

            _sync_cuda()
            phase_started = time.perf_counter()
            projection = _trajectory_projection_metrics(
                student_model=model,
                student_hidden=student_hidden,
                raw_target_for_chunk=raw_target_for_chunk,
                labels=response_ids,
                chunk_size=args.token_chunk_size,
                enabled=args.trajectory_projection,
                kl_budget=args.projection_kl_budget,
                binary_search_steps=args.projection_binary_search_steps,
                token_partitions=token_partitions,
            )
            _sync_cuda()
            phase_seconds["target"] += time.perf_counter() - phase_started

            if args.update_guard and float(
                projection["realized_target_logprob_advantage"]
            ) <= 0.0:
                guard_rejections += 1
                del student_hidden, teacher_hidden, response_ids, teacher_prompt_ids
                continue

            _sync_cuda()
            phase_started = time.perf_counter()
            if args.method == "opsd":
                objective = _stream_opsd_backward(
                    student_model=model,
                    student_hidden=student_hidden,
                    teacher_hidden=teacher_hidden,
                    chunk_size=args.token_chunk_size,
                    beta=args.opsd_beta,
                    element_clip=args.opsd_jsd_element_clip,
                    temperature=args.distill_temperature,
                    loss_scale=1.0 / args.group_size,
                    projection_alpha=float(projection["alpha"]),
                )
            else:
                objective = _stream_reverse_kl_backward(
                    model=model,
                    student_hidden=student_hidden,
                    teacher_hidden=teacher_hidden,
                    chunk_size=args.token_chunk_size,
                    temperature=args.distill_temperature,
                    loss_scale=1.0 / args.group_size,
                    projection_alpha=float(projection["alpha"]),
                )
            _sync_cuda()
            phase_seconds["update"] += time.perf_counter() - phase_started
            objective.update(
                {
                    "projection_alpha": float(projection["alpha"]),
                    "raw_target_kl": float(projection["raw_target_kl"]),
                    "projected_target_kl": float(
                        projection["projected_target_kl"]
                    ),
                    "projection_cap_active": float(bool(projection["cap_active"])),
                    "realized_target_logprob_advantage": float(
                        projection["realized_target_logprob_advantage"]
                    ),
                    "task_token_logprob_gain": float(
                        projection["task_token_logprob_gain"]
                    ),
                    "style_token_logprob_gain": float(
                        projection["style_token_logprob_gain"]
                    ),
                    "task_token_count": float(projection["task_token_count"]),
                    "style_token_count": float(projection["style_token_count"]),
                }
            )
            metrics.append(objective)
            length = int(response_ids.numel())
            optimized_rollouts += 1
            teacher_positions += length
            if privilege_source in {
                "verified_reference_solution",
                "verifier_critique",
                "execution_solver_feedback",
                "equivalent_prompt_wrappers",
            }:
                hindsight_positions += length
            del student_hidden, teacher_hidden, response_ids, teacher_prompt_ids
        optimizer_step = bool(optimized_rollouts)
    else:
        advantages = grpo_group_advantages(rewards)
        for rollout_index, rollout in enumerate(rollouts):
            response_ids = _model_device_ids(rollout["response_ids"], model)
            _sync_cuda()
            phase_started = time.perf_counter()
            model.train()
            student_hidden = _response_hidden(
                model, normal_prompt_ids, response_ids, grad=True
            )
            with model.disable_adapter():
                model.eval()
                with torch.no_grad():
                    reference_hidden = _response_hidden(
                        model, normal_prompt_ids, response_ids, grad=False
                    )
            model.train()
            metrics.append(
                _stream_grpo_backward(
                    model=model,
                    student_hidden=student_hidden,
                    reference_hidden=reference_hidden,
                    labels=response_ids,
                    advantage=float(advantages[rollout_index].item()),
                    chunk_size=args.token_chunk_size,
                    clip_epsilon=args.grpo_clip_epsilon,
                    clip_epsilon_high=None,
                    kl_coefficient=args.grpo_kl_coefficient,
                    loss_scale=1.0 / args.group_size,
                )
            )
            _sync_cuda()
            phase_seconds["update"] += time.perf_counter() - phase_started
            optimized_rollouts += 1
            del student_hidden, reference_hidden, response_ids
        optimizer_step = True

    grad_norm: float | None = None
    if optimizer_step:
        _sync_cuda()
        phase_started = time.perf_counter()
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            parameters, args.max_grad_norm
        )
        grad_norm = float(grad_norm_tensor.detach().item())
        if not math.isfinite(grad_norm):
            raise FloatingPointError("baseline gradient norm is not finite")
        optimizer.step()
        if ema_model is not None:
            _ema_update(model, ema_model, args.ema_rate)
        _sync_cuda()
        phase_seconds["update"] += time.perf_counter() - phase_started
    model.train()

    generated_tokens = sum(int(item["response_ids"].numel()) for item in rollouts)
    mean_metrics = {
        key: sum(item.get(key, 0.0) for item in metrics) / len(metrics)
        for key in sorted({key for item in metrics for key in item})
    } if metrics else {}
    row = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "method": args.method,
        "method_id": _method_id(args.method, args.trajectory_projection),
        "episode": stream_index + 1,
        "stream_index": stream_index,
        "query_id": query["query_id"],
        "problem_sha256": query["problem_sha256"],
        "source": query["source"],
        "episode_seed": episode_seed,
        "rollout_count": len(rollouts),
        "correct_rollout_count": int(rewards.sum().item()),
        "rewards": rewards.tolist(),
        "response_tokens": [
            int(item["response_ids"].numel()) for item in rollouts
        ],
        "response_token_sha256": [
            hashlib.sha256(
                ",".join(
                    map(str, item["response_ids"].detach().cpu().reshape(-1).tolist())
                ).encode("utf-8")
            ).hexdigest()
            for item in rollouts
        ],
        "generated_tokens": generated_tokens,
        "optimized_rollout_count": optimized_rollouts,
        "optimizer_step": optimizer_step,
        "gradient_norm": grad_norm,
        "effective_learning_rate": effective_learning_rate,
        "objective_metrics": mean_metrics,
        "privileged_rollout_index": privileged_index,
        "privileged_information_tokens": privileged_tokens,
        "privileged_prompt_version": (
            PRIVILEGED_PROMPT_VERSION
            if args.method == "demopsd"
            else SRPO_TEACHER_PROMPT_VERSION
            if args.method == "srpo"
            else OPSD_PRIVILEGED_PROMPT_VERSION
            if args.method == "opsd"
            else PRIVILEGE_SOURCE_PROMPT_VERSION
            if args.method == "trsd_source"
            else None
        ),
        "privilege_source": (
            args.privilege_source if args.method == "trsd_source" else None
        ),
        "privilege_wrapper_index": (
            stream_index % 3
            if args.method == "trsd_source"
            and args.privilege_source == "equivalent_prompt_wrappers"
            else None
        ),
        "privilege_payload_sha256": (
            _sha256(privilege_payload_hashes) if privilege_payload_hashes else None
        ),
        "trajectory_projection": bool(args.trajectory_projection),
        "projection_kl_budget": args.projection_kl_budget,
        "projection_cap_hits": sum(
            int(item.get("projection_cap_active", 0.0)) for item in metrics
        ),
        "update_guard": bool(args.update_guard),
        "guard_rejections": guard_rejections,
        "teacher_positions": teacher_positions,
        "hindsight_exposed_positions": hindsight_positions,
        "srpo_sdpo_rollout_count": srpo_sdpo_rollouts,
        "srpo_grpo_rollout_count": srpo_grpo_rollouts,
        "srpo_teacher_available_count": (
            srpo_teacher_available_count
        ),
        "srpo_sdpo_tokens": srpo_sdpo_tokens,
        "srpo_grpo_tokens": srpo_grpo_tokens,
        "srpo_entropy_weight_normalizer": srpo_entropy_weight_normalizer,
        "compared_positions": sum(
            int(item["response_ids"].numel()) for item in rollouts
        ),
        "phase_seconds": phase_seconds,
    }
    del rollouts, rewards, metrics
    gc.collect()
    if torch.cuda.is_available():
        _sync_cuda()
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated())
        cuda_peak_reserved = int(torch.cuda.max_memory_reserved())
        torch.cuda.empty_cache()
    else:
        cuda_peak_allocated = cuda_peak_reserved = 0
    row["episode_seconds"] = time.perf_counter() - started
    row["resource_usage"] = {
        "cuda_memory_baseline_bytes": cuda_baseline,
        "cuda_peak_memory_allocated_bytes": cuda_peak_allocated,
        "cuda_peak_memory_delta_bytes": max(cuda_peak_allocated - cuda_baseline, 0),
        "cuda_peak_memory_reserved_bytes": cuda_peak_reserved,
        "process_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    minimum_group = 1 if args.method in {"opsd", "trsd_source"} else 2
    if args.group_size < minimum_group:
        raise BaselineProtocolError(
            f"{args.method} group-size must be at least {minimum_group}"
        )
    if args.method in {"grpo", "srpo"} and args.trajectory_projection:
        raise BaselineProtocolError(
            f"{args.method} does not use trajectory target projection"
        )
    if args.method == "trsd_source" and not args.trajectory_projection:
        raise BaselineProtocolError("trsd_source requires trajectory projection")
    if args.generation_batch_size == 0:
        args.generation_batch_size = args.group_size
    if not 1 <= args.generation_batch_size <= args.group_size:
        raise BaselineProtocolError(
            "generation-batch-size must lie in [1, group-size]"
        )
    if args.max_sequence_tokens <= 0 or args.max_rollout_tokens <= 0:
        raise BaselineProtocolError("sequence limits must be positive")
    if args.teacher_max_sequence_tokens < args.max_sequence_tokens:
        raise BaselineProtocolError(
            "teacher-max-sequence-tokens must cover max-sequence-tokens"
        )
    if args.token_chunk_size <= 0:
        raise BaselineProtocolError("token-chunk-size must be positive")
    if not 0.0 <= args.opsd_beta <= 1.0:
        raise BaselineProtocolError("opsd-beta must lie in [0,1]")
    if args.opsd_jsd_element_clip < 0:
        raise BaselineProtocolError("opsd-jsd-element-clip cannot be negative")
    if args.srpo_top_k <= 0:
        raise BaselineProtocolError("srpo-top-k must be positive")
    if args.srpo_entropy_beta < 0:
        raise BaselineProtocolError("srpo-entropy-beta cannot be negative")
    if not 0.0 < args.srpo_jsd_alpha < 1.0:
        raise BaselineProtocolError("srpo-jsd-alpha must lie in (0,1)")
    if args.srpo_grpo_clip_epsilon_high < 0 or args.srpo_warmup_steps < 0:
        raise BaselineProtocolError("SRPO clip and warmup values cannot be negative")
    if args.projection_kl_budget <= 0 or args.projection_binary_search_steps <= 0:
        raise BaselineProtocolError("projection budget and search steps must be positive")
    queries = load_query_only_manifest(args.queries)
    labels = _load_training_labels(args.labels)
    selected = _validate_inputs(queries, labels, args.episodes)
    config = {
        "method": args.method,
        "method_id": _method_id(args.method, args.trajectory_projection),
        "model_id": args.model_id,
        "revision": args.revision,
        "episodes": args.episodes,
        "group_size": args.group_size,
        "generation_batch_size": args.generation_batch_size,
        "group_generation_engine": "cached_batched_independent_seed_v1",
        "max_sequence_tokens": args.max_sequence_tokens,
        "teacher_max_sequence_tokens": args.teacher_max_sequence_tokens,
        "max_rollout_tokens": args.max_rollout_tokens,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
        "train_temperature": args.train_temperature,
        "disable_thinking": args.disable_thinking,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "token_chunk_size": args.token_chunk_size,
        "demopsd_beta": args.demopsd_beta,
        "demopsd_alpha_max": args.demopsd_alpha_max,
        "ema_rate": args.ema_rate,
        "distill_temperature": args.distill_temperature,
        "distill_support": "exact_full_vocabulary",
        "grpo_clip_epsilon": args.grpo_clip_epsilon,
        "grpo_kl_coefficient": args.grpo_kl_coefficient,
    }
    # Keep legacy DemoPSD/GRPO run identities byte-identical.  New protocol
    # fields enter the identity only when the corresponding mechanism is used,
    # so an active restartable baseline can safely resume under this code.
    if args.method == "opsd":
        config.update(
            {
                "opsd_beta": args.opsd_beta,
                "opsd_jsd_element_clip": args.opsd_jsd_element_clip,
                "opsd_teacher_strategy": args.opsd_teacher_strategy,
            }
        )
    if args.method == "trsd_source":
        config["privilege_source"] = args.privilege_source
    if args.method == "srpo":
        config.update(
            {
                "srpo_top_k": args.srpo_top_k,
                "srpo_add_tail": True,
                "srpo_entropy_beta": args.srpo_entropy_beta,
                "srpo_jsd_alpha": args.srpo_jsd_alpha,
                "srpo_grpo_clip_epsilon_high": args.srpo_grpo_clip_epsilon_high,
                "srpo_warmup_steps": args.srpo_warmup_steps,
                "srpo_rollout_importance_clip": args.srpo_rollout_importance_clip,
                "srpo_routing": "incorrect_and_correct_sibling_available_to_sdpo_else_grpo",
                "distill_support": "ema_teacher_topk_plus_tail",
                "loss_normalization": "all_routed_tokens",
            }
        )
    if args.trajectory_projection:
        config.update(
            {
                "trajectory_projection": True,
                "projection_kl_budget": args.projection_kl_budget,
                "projection_binary_search_steps": args.projection_binary_search_steps,
            }
        )
    if args.update_guard:
        config["update_guard"] = True
    identity = {
        "config_sha256": _sha256(config),
        "query_manifest_sha256": _sha256(selected),
        "label_manifest_sha256": _sha256(
            [
                {
                    "query_id": row["query_id"],
                    "answer": labels[row["query_id"]]["answer"],
                    **(
                        {
                            "reference_solution_sha256": hashlib.sha256(
                                str(
                                    labels[row["query_id"]].get(
                                        "reference_solution", ""
                                    )
                                ).encode("utf-8")
                            ).hexdigest()
                        }
                        if args.method in {"opsd", "trsd_source"}
                        else {}
                    ),
                    "problem_sha256": row["problem_sha256"],
                }
                for row in selected
            ]
        ),
    }
    identity["run_identity_sha256"] = _sha256(identity)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    local_model = Path(args.model)
    revision = None if local_model.exists() else args.revision
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        training=True,
        revision=revision,
    )
    output_head = unwrap_causal_lm(model).get_output_embeddings()
    if output_head is None or any(
        parameter.requires_grad for parameter in output_head.parameters()
    ):
        raise BaselineProtocolError("baseline streaming requires a frozen LM head")
    ema_model = None
    if args.method in {"demopsd", "srpo"}:
        ema_model, _ = load_hf_model(
            args.model,
            dtype=args.dtype,
            device_map=args.ema_device_map or args.device_map,
            attn_implementation=args.attn_implementation,
            use_lora=True,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            training=False,
            revision=revision,
        )
        _copy_lora(model, ema_model)
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        ema_model.eval()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise BaselineProtocolError("LoRA model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    journal_path = output_dir / "episodes.jsonl"
    runtime = collect_runtime_metadata(
        model, model_path=args.model_id, revision=args.revision
    )
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "method": args.method,
        "method_id": _method_id(args.method, args.trajectory_projection),
        "arguments": config,
        "runtime": runtime,
        "paper_sources": {
            "demopsd": "https://arxiv.org/abs/2607.02502",
            "grpo": "https://arxiv.org/abs/2402.03300",
            "opsd": "https://arxiv.org/abs/2601.18734",
            "srpo": "https://arxiv.org/abs/2604.02288",
            "sdpo_implementation": "https://github.com/lasgroup/SDPO",
        },
        **identity,
    }
    rows = _read_jsonl(journal_path)
    if args.resume:
        if not manifest_path.exists():
            raise BaselineProtocolError("resume output lacks run_manifest.json")
        existing = _read_json(manifest_path)
        if existing.get("run_identity_sha256") != identity["run_identity_sha256"]:
            raise BaselineProtocolError("resume output belongs to another run")
        completed = _load_resume(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            output_dir=output_dir,
            identity=identity,
        )
        if len(rows) < completed:
            raise BaselineProtocolError("journal is shorter than its durable checkpoint")
        if len(rows) > completed:
            # A journal row is not a durable optimizer update until its atomic
            # checkpoint is published.  Replay any interrupted suffix.
            rows = rows[:completed]
            _atomic_jsonl(journal_path, rows)
    else:
        if manifest_path.exists() or rows:
            raise BaselineProtocolError("nonempty output requires --resume")
        _atomic_json(manifest_path, run_manifest)
        completed = 0
        _save_checkpoint(
            model=model,
            ema_model=ema_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            output_dir=output_dir,
            completed_episodes=0,
            method=args.method,
            args=args,
            identity=identity,
            cumulative=_cumulative(rows),
        )

    for stream_index in range(completed, args.episodes):
        query = selected[stream_index]
        row = _episode(
            model=model,
            ema_model=ema_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            query=query,
            answer=labels[query["query_id"]]["answer"],
            reference_solution=str(
                labels[query["query_id"]].get("reference_solution", "")
            ),
            permuted_reference_solution=str(
                labels[selected[(stream_index + 1) % len(selected)]["query_id"]].get(
                    "reference_solution", ""
                )
            ),
            stream_index=stream_index,
            args=args,
        )
        rows.append(row)
        _atomic_jsonl(journal_path, rows)
        completed = stream_index + 1
        if completed % args.checkpoint_interval == 0 or completed == args.episodes:
            _save_checkpoint(
                model=model,
                ema_model=ema_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                output_dir=output_dir,
                completed_episodes=completed,
                method=args.method,
                args=args,
                identity=identity,
                cumulative=_cumulative(rows),
            )

    result = {
        "status": "complete",
        "method": args.method,
        "completed_episodes": completed,
        "final_checkpoint": str(
            output_dir / "checkpoints" / f"episode_{completed:04d}"
        ),
        "cumulative_audit": _cumulative(rows),
        **identity,
    }
    _atomic_json(output_dir / "COMPLETE.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--method", choices=tuple(METHOD_IDS), required=True)
    value.add_argument("--queries", required=True)
    value.add_argument("--labels", required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--model-id", default="Qwen/Qwen3-8B")
    value.add_argument("--revision", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--episodes", type=int, default=16)
    value.add_argument("--group-size", type=int, default=8)
    value.add_argument(
        "--generation-batch-size",
        type=int,
        default=0,
        help="same-prompt decode batch; 0 uses the full rollout group",
    )
    value.add_argument("--max-sequence-tokens", type=int, default=16_384)
    value.add_argument("--teacher-max-sequence-tokens", type=int, default=40_960)
    value.add_argument("--max-rollout-tokens", type=int, default=16_384)
    value.add_argument("--privileged-max-tokens", type=int, default=2_048)
    value.add_argument("--learning-rate", type=float, default=1e-6)
    value.add_argument("--weight-decay", type=float, default=0.0)
    value.add_argument("--lora-rank", type=int, default=8)
    value.add_argument("--lora-alpha", type=int, default=16)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--train-temperature", type=float, default=1.0)
    value.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Validation-only Qwen chat-template switch; formal runs leave thinking on",
    )
    value.add_argument("--top-p", type=float, default=0.95)
    value.add_argument("--top-k", type=int, default=20)
    value.add_argument("--token-chunk-size", type=int, default=64)
    value.add_argument("--max-grad-norm", type=float, default=1.0)
    value.add_argument("--demopsd-beta", type=float, default=50.0)
    value.add_argument("--demopsd-alpha-max", type=float, default=0.15)
    value.add_argument("--ema-rate", type=float, default=0.05)
    value.add_argument("--srpo-top-k", type=int, default=100)
    value.add_argument("--srpo-entropy-beta", type=float, default=1.0)
    value.add_argument("--srpo-jsd-alpha", type=float, default=0.5)
    value.add_argument("--srpo-grpo-clip-epsilon-high", type=float, default=0.28)
    value.add_argument("--srpo-warmup-steps", type=int, default=10)
    value.add_argument("--srpo-rollout-importance-clip", type=float, default=2.0)
    value.add_argument("--opsd-beta", type=float, default=0.5)
    value.add_argument("--opsd-jsd-element-clip", type=float, default=0.05)
    value.add_argument(
        "--opsd-teacher-strategy",
        choices=("fixed", "current"),
        default="fixed",
    )
    value.add_argument(
        "--privilege-source",
        choices=PRIVILEGE_SOURCES,
        default="verified_reference_solution",
        help="Privileged-context construction for the projected source study",
    )
    value.add_argument("--trajectory-projection", action="store_true")
    value.add_argument("--projection-kl-budget", type=float, default=0.004)
    value.add_argument("--projection-binary-search-steps", type=int, default=6)
    value.add_argument(
        "--update-guard",
        action="store_true",
        help="Skip a dense update when its target lowers realized trajectory log-probability",
    )
    value.add_argument("--distill-temperature", type=float, default=1.0)
    value.add_argument("--grpo-clip-epsilon", type=float, default=0.2)
    value.add_argument("--grpo-kl-coefficient", type=float, default=0.04)
    value.add_argument("--checkpoint-interval", type=int, default=1)
    value.add_argument("--dtype", default="bfloat16")
    value.add_argument("--device-map", default="auto")
    value.add_argument(
        "--ema-device-map",
        default=None,
        help="optional EMA-teacher placement; defaults to --device-map",
    )
    value.add_argument("--attn-implementation", default="sdpa")
    value.add_argument("--resume", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    print(_canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
