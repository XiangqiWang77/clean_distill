"""Train and evaluate Clean Self-Distillation.

The student first generates an on-policy response.  The base student and the
query-local ridge teacher are then scored on the exact same prompt and response
prefixes.  Target answers/solutions are used only by the offline evaluator.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.opsd_format import extract_boxed_answer, grade_boxed_answer

from .io import (
    canonical_json_sha256,
    load_proposal_map,
    load_query_records,
    stable_hash,
    validate_proposal_training_binding,
    validate_specialization_state,
    write_jsonl,
)
from .metrics import HindsightAudit, aggregate_teacher_metrics
from .ridge import SparseRidgeAdapter, candidate_completion, fit_ridge_adapter, problem_prompt
from .runtime import (
    backbone_forward,
    collect_runtime_metadata,
    input_device,
    load_hf_model,
    project_logits,
    unwrap_causal_lm,
)


def _sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        threshold = torch.topk(logits, k=top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, sorted_indices, sorted_logits)
        logits = filtered
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=generator).squeeze(-1)


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    problem: str,
    *,
    adapter: Optional[SparseRidgeAdapter],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    prompt_override: Optional[str] = None,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Autoregressive generation with an optional query-local logit update."""
    was_training = model.training
    model.eval()
    device = input_device(model)
    prompt = prompt_override if prompt_override is not None else problem_prompt(tokenizer, problem)
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
    generated: list[torch.Tensor] = []
    past_key_values = None
    next_input = prompt_ids
    generator: Optional[torch.Generator] = None

    for _ in range(max_new_tokens):
        hidden_sequence, past_key_values = backbone_forward(
            model,
            input_ids=next_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden = hidden_sequence[:, -1, :]
        logits = project_logits(model, hidden)
        if adapter is not None:
            logits = adapter.apply_to_logits(logits, hidden)
        if generator is None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(seed)
        token = _sample_token(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
        )
        generated.append(token)
        next_input = token[:, None].to(device)
        if tokenizer.eos_token_id is not None and int(token.item()) == tokenizer.eos_token_id:
            break

    response_ids = (
        torch.stack(generated, dim=1)
        if generated
        else torch.empty((1, 0), dtype=torch.long, device=device)
    )
    response_ids = response_ids.to(prompt_ids.device)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
    model.train(was_training)
    return response, prompt_ids, response_ids


def _completion_tensors(tokenizer, problem: str, completion: str, device: torch.device):
    prompt_ids = tokenizer(
        problem_prompt(tokenizer, problem),
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"].to(device)
    completion_ids = tokenizer(
        completion,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    if completion_ids.numel() == 0:
        raise ValueError("Completion tokenized to zero tokens")
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    start = prompt_ids.shape[1] - 1
    return prompt_ids, completion_ids, full_ids, start


@torch.inference_mode()
def score_target_completion(
    model,
    tokenizer,
    problem: str,
    completion: str,
    adapter: SparseRidgeAdapter,
) -> tuple[float, float, torch.Tensor]:
    """Offline target NLL. The target completion is never an adapter input."""
    device = input_device(model)
    prompt_ids, completion_ids, full_ids, start = _completion_tensors(
        tokenizer, problem, completion, device
    )
    all_hidden, _ = backbone_forward(
        model,
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        use_cache=False,
    )
    length = completion_ids.shape[1]
    hidden = all_hidden[:, start : start + length]
    base_logits = project_logits(model, hidden)
    teacher_logits = adapter.apply_to_logits(base_logits, hidden)
    labels = completion_ids.to(base_logits.device)
    base_nll = F.cross_entropy(
        base_logits.reshape(-1, base_logits.shape[-1]), labels.reshape(-1), reduction="mean"
    )
    teacher_nll = F.cross_entropy(
        teacher_logits.reshape(-1, teacher_logits.shape[-1]), labels.reshape(-1), reduction="mean"
    )
    return float(base_nll.item()), float(teacher_nll.item()), full_ids


@torch.inference_mode()
def score_plain_completion(model, tokenizer, problem: str, completion: str) -> float:
    device = input_device(model)
    _, completion_ids, full_ids, start = _completion_tensors(tokenizer, problem, completion, device)
    all_hidden, _ = backbone_forward(
        model,
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        use_cache=False,
    )
    length = completion_ids.shape[1]
    logits = project_logits(model, all_hidden[:, start : start + length])
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        completion_ids.to(logits.device).reshape(-1),
        reduction="mean",
    )
    return float(nll.item())


@contextlib.contextmanager
def base_policy_context(model):
    """Disable the persistent LoRA while scoring the frozen base/teacher."""
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        yield
        return
    with disable_adapter():
        yield


def _capture_trainable_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


@torch.no_grad()
def _restore_trainable_state(model, state: dict[str, torch.Tensor]) -> None:
    named_parameters = dict(model.named_parameters())
    missing = set(state) - set(named_parameters)
    if missing:
        raise RuntimeError(f"Cannot reset query-local student; missing parameters: {sorted(missing)[:5]}")
    for name, value in state.items():
        named_parameters[name].copy_(value.to(named_parameters[name].device, named_parameters[name].dtype))


@torch.no_grad()
def _trainable_state_delta_norm(model, state: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if name not in state:
            continue
        delta = parameter.detach().float() - state[name].to(parameter.device, torch.float32)
        total += float(torch.sum(delta * delta).item())
    return total**0.5


@torch.no_grad()
def _trainable_state_matches(model, state: dict[str, torch.Tensor]) -> bool:
    named = dict(model.named_parameters())
    return all(
        name in named
        and torch.equal(
            named[name].detach().cpu(), value.to(dtype=named[name].dtype)
        )
        for name, value in state.items()
    )


def same_prefix_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    token_clip: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    if top_k <= 0:
        raise ValueError("distill_top_k must be positive")
    scaled_teacher = teacher_logits.float() / temperature
    scaled_student = student_logits.float() / temperature
    teacher_full_log_probs = F.log_softmax(scaled_teacher, dim=-1)
    student_full_log_probs = F.log_softmax(scaled_student, dim=-1)
    k = min(top_k, teacher_logits.shape[-1])
    if k == teacher_logits.shape[-1]:
        per_token = F.kl_div(
            student_full_log_probs,
            teacher_full_log_probs,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)
    else:
        # Preserve the probability mass omitted by teacher top-k as an explicit
        # "other" bucket. Independent renormalization over top-k can otherwise
        # report zero KL even when student and teacher assign very different
        # total mass to the selected set (and makes k=1 identically zero).
        teacher_ids = torch.topk(scaled_teacher, k=k, dim=-1).indices
        teacher_selected_log = torch.gather(teacher_full_log_probs, -1, teacher_ids)
        student_selected_log = torch.gather(student_full_log_probs, -1, teacher_ids)
        teacher_other = (1.0 - teacher_selected_log.exp().sum(dim=-1, keepdim=True)).clamp_min(
            1e-12
        )
        student_other = (1.0 - student_selected_log.exp().sum(dim=-1, keepdim=True)).clamp_min(
            1e-12
        )
        teacher_coarse_log = torch.cat([teacher_selected_log, teacher_other.log()], dim=-1)
        student_coarse_log = torch.cat([student_selected_log, student_other.log()], dim=-1)
        per_token = F.kl_div(
            student_coarse_log,
            teacher_coarse_log,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)
    if token_clip > 0:
        per_token = per_token.clamp(max=token_clip)
    return per_token.mean() * (temperature**2)


def _proposal_for(
    record: dict[str, Any],
    proposals: dict[str, dict[str, Any]],
    proposals_by_hash: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    problem_hash = str(record.get("problem_sha256") or stable_hash(record["problem"], 64))
    proposal = proposals.get(record["query_id"])
    if proposal is None:
        candidates = [
            row
            for row in proposals_by_hash.get(problem_hash, [])
            if str(row.get("source", "")).strip().lower() == record["source"]
        ]
        if len(candidates) != 1:
            raise KeyError(
                f"No unambiguous proposal set for query_id={record['query_id']} "
                f"source={record['source']} hash={problem_hash}; matches={len(candidates)}"
            )
        proposal = candidates[0]

    actual_hash = str(
        proposal.get("problem_sha256")
        or stable_hash(str(proposal.get("problem", "")), 64)
    )
    actual_source = str(proposal.get("source", "")).strip().lower()
    if actual_hash != problem_hash or actual_source != record["source"]:
        raise ValueError(
            f"Proposal mismatch for {record['query_id']}: expected "
            f"source/hash={record['source']}/{problem_hash}, got "
            f"{actual_source}/{actual_hash}"
        )
    validate_proposal_training_binding(
        proposal, context=f"Proposal selected for {record['query_id']}"
    )
    return proposal


def _index_proposals_by_hash(
    proposals: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in proposals.values():
        problem_hash = str(row.get("problem_sha256", "")).strip()
        if problem_hash:
            index[problem_hash].append(row)
    return dict(index)


def _fit_current_adapter(model, tokenizer, proposal: dict[str, Any], args):
    proposal_training_sha256 = validate_proposal_training_binding(
        proposal, context="Proposal used for ridge fitting"
    )
    specialization_status, specialization_failure_reason, specialization_no_op = (
        validate_specialization_state(
            proposal, context=f"Proposal {proposal.get('query_id')!r}"
        )
    )
    exposed_sources = set(_proposal_exposed_sources(proposal))
    if exposed_sources and not args.allow_hindsight_exposure:
        raise ValueError(
            f"Proposal {proposal.get('query_id')} is marked as hindsight-contaminated by "
            f"{sorted(exposed_sources)}. Pass --allow-hindsight-exposure only for an ablation."
        )
    was_training = model.training
    model.eval()
    device = input_device(model)
    baseline_memory = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline_memory = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    candidates = list(proposal.get("specialization_candidates", []))
    if args.num_specialization_candidates is not None:
        candidates = candidates[: args.num_specialization_candidates]
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
        query_id=str(proposal["query_id"]),
        specialization_status=specialization_status,
        specialization_failure_reason=specialization_failure_reason,
        specialization_no_op=specialization_no_op,
    )
    adapter.metadata.update(
        {
            "problem_sha256": proposal.get("problem_sha256"),
            "proposal_training_sha256": proposal_training_sha256,
            "source": proposal.get("source"),
            "model": args.model,
            "model_revision": args.runtime_metadata.get(
                "resolved_model_revision", args.revision or ""
            ),
        }
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = float(torch.cuda.max_memory_allocated(device))
        metrics["peak_memory_bytes"] = peak_allocated
        metrics["specialization_memory_baseline_bytes"] = baseline_memory
        metrics["specialization_peak_memory_delta_bytes"] = max(
            peak_allocated - baseline_memory, 0.0
        )
    else:
        metrics["peak_memory_bytes"] = 0.0
        metrics["specialization_memory_baseline_bytes"] = 0.0
        metrics["specialization_peak_memory_delta_bytes"] = 0.0
    model.train(was_training)
    return adapter.to(input_device(model)), metrics


def _load_adapter_cache(
    adapter_dir: Optional[str], expected_model: str, expected_revision: str = ""
) -> dict[str, tuple[SparseRidgeAdapter, dict[str, Any]]]:
    if not adapter_dir:
        return {}
    root = Path(adapter_dir)
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing adapter manifest: {manifest}")
    cache = {}
    from .io import iter_rows

    for row in iter_rows(manifest):
        query_id = str(row["query_id"])
        if query_id in cache:
            raise ValueError(f"Duplicate cached adapter query_id {query_id!r} in {manifest}")
        manifest_model = str(row.get("model", ""))
        if manifest_model and manifest_model != expected_model:
            raise ValueError(
                f"Cached adapter {query_id} was fitted with {manifest_model!r}, "
                f"not requested model {expected_model!r}"
            )
        manifest_revision = str(row.get("model_revision", ""))
        if expected_revision and manifest_revision != expected_revision:
            raise ValueError(
                f"Cached adapter {query_id} was fitted with revision "
                f"{manifest_revision!r}, not requested revision {expected_revision!r}"
            )
        adapter = SparseRidgeAdapter.load(root / row["adapter_path"])
        _validate_adapter_manifest_binding(
            adapter,
            row,
            expected_model=expected_model,
            expected_revision=expected_revision,
        )
        cache[query_id] = (adapter, row)
    return cache


def _validate_adapter_manifest_binding(
    adapter: SparseRidgeAdapter,
    manifest: dict[str, Any],
    *,
    expected_model: str,
    expected_revision: str = "",
) -> None:
    """Reject a stale/wrong tensor file hidden behind a plausible manifest."""
    manifest_status = manifest.get("specialization_status")
    manifest_reason = manifest.get("specialization_failure_reason")
    manifest_no_op = manifest.get("specialization_no_op")
    manifest_uses_all = manifest.get("uses_all_candidates")
    if manifest_status not in {"ready", "insufficient_verified_candidates"}:
        raise ValueError("Cached adapter manifest has an invalid specialization_status")
    if (
        not isinstance(manifest_reason, str)
        or not isinstance(manifest_no_op, bool)
        or not isinstance(manifest_uses_all, bool)
    ):
        raise ValueError("Cached adapter manifest has invalid specialization state types")
    if (manifest_status == "ready") != (manifest_reason == "" and not manifest_no_op):
        raise ValueError("Cached adapter manifest has an inconsistent specialization state")
    if manifest_status != "ready" and (not manifest_reason.strip() or not manifest_no_op):
        raise ValueError("Cached adapter manifest has an inconsistent specialization state")
    if manifest_uses_all is not (not manifest_no_op):
        raise ValueError("Cached adapter manifest has inconsistent candidate-use provenance")
    expected = {
        "query_id": str(manifest.get("query_id", "")),
        "problem_sha256": str(manifest.get("problem_sha256", "")),
        "proposal_training_sha256": str(
            manifest.get("proposal_training_sha256", "")
        ),
        "source": str(manifest.get("source", "")).strip().lower(),
        "model": expected_model,
    }
    if expected_revision:
        expected["model_revision"] = expected_revision
    metadata = adapter.metadata
    if (
        metadata.get("specialization_status") != manifest_status
        or not isinstance(metadata.get("specialization_no_op"), bool)
        or metadata.get("specialization_no_op") is not manifest_no_op
        or not isinstance(metadata.get("uses_all_candidates"), bool)
        or metadata.get("uses_all_candidates") is not manifest_uses_all
    ):
        raise ValueError(
            f"Cached adapter binding mismatch for {expected['query_id']!r}: "
            "specialization status/no-op differs between manifest and tensor metadata"
        )
    for key, value in expected.items():
        actual = str(metadata.get(key, ""))
        if key == "source":
            actual = actual.strip().lower()
        if not value or actual != value:
            raise ValueError(
                f"Cached adapter binding mismatch for {expected['query_id']!r}: "
                f"{key} manifest={value!r}, tensor_metadata={actual!r}"
            )
    metadata_reason = metadata.get("specialization_failure_reason")
    if metadata_reason != manifest_reason:
        raise ValueError(
            f"Cached adapter binding mismatch for {expected['query_id']!r}: "
            "specialization_failure_reason differs between manifest and tensor metadata"
        )


def _target_completion(record: dict[str, Any], mode: str) -> str:
    answer = str(record.get("answer", "")).strip()
    solution = str(record.get("solution", "")).strip()
    if mode == "solution_answer" and solution:
        if answer and answer not in solution:
            return f"{solution}\n\nFinal answer: \\boxed{{{answer}}}"
        return solution
    if not answer:
        raise ValueError(f"{record['query_id']} has no answer for target NLL evaluation")
    return f"The final answer is \\boxed{{{answer}}}."


def _grade_response(response: str, answer: str) -> float:
    prediction = extract_boxed_answer(response)
    return float(grade_boxed_answer(prediction, answer))


def _majority_grade(responses: list[str], answer: str) -> float:
    extracted = [str(extract_boxed_answer(response) or "").strip() for response in responses]
    extracted = [value for value in extracted if value]
    if not extracted:
        return 0.0
    majority_answer, _ = Counter(extracted).most_common(1)[0]
    return float(grade_boxed_answer(majority_answer, answer))


def _wrong_answer_hint(answer: str) -> str:
    """Create a deterministic, format-matched incorrect answer for audit only."""
    normalized = str(answer).strip()
    try:
        value = int(normalized)
    except ValueError:
        return f"{normalized}_wrong"
    candidate = value + 1
    if normalized.startswith("0") and len(normalized) > 1:
        return str(candidate).zfill(len(normalized))
    return str(candidate)


def _hinted_problem(problem: str, hint: str) -> str:
    return (
        f"{problem}\n\nA verifier reveals that the final answer is {hint}. "
        "Derive a solution consistent with this revealed answer."
    )


@torch.inference_mode()
def counterfactual_hint_jsd(
    model,
    tokenizer,
    problem: str,
    response: str,
    correct_hint: str,
    wrong_hint: str,
    *,
    max_positions: int,
) -> float:
    """JSD under correct/wrong target hints on the same response tokens.

    This is an evaluation-only privileged intervention. The clean CSD teacher
    never receives either hint, so its counterfactual sensitivity is exactly
    zero by construction.
    """
    device = input_device(model)
    response_ids = tokenizer(
        response,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    if response_ids.numel() == 0:
        return 0.0
    if response_ids.shape[1] > max_positions:
        response_ids = response_ids[:, :max_positions]

    distributions = []
    for hint in (correct_hint, wrong_hint):
        prompt_ids = tokenizer(
            problem_prompt(tokenizer, _hinted_problem(problem, hint)),
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"].to(device)
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        hidden_all, _ = backbone_forward(
            model,
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            use_cache=False,
        )
        start = prompt_ids.shape[1] - 1
        hidden = hidden_all[:, start : start + response_ids.shape[1]]
        distributions.append(F.softmax(project_logits(model, hidden).float(), dim=-1))
    correct_probs, wrong_probs = distributions
    midpoint = 0.5 * (correct_probs + wrong_probs)
    eps = 1e-12
    jsd = 0.5 * (
        (correct_probs * ((correct_probs + eps).log() - (midpoint + eps).log())).sum(dim=-1)
        + (wrong_probs * ((wrong_probs + eps).log() - (midpoint + eps).log())).sum(dim=-1)
    )
    return float(jsd.mean().item())


def _proposal_exposed_sources(proposal: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    firewall = proposal.get("firewall_audit", {})
    if isinstance(firewall, dict):
        if str(firewall.get("target_answer_loaded", False)).strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            sources.append("target_answer")
        if str(firewall.get("target_solution_loaded", False)).strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            sources.append("target_solution")
    return sources


def _teacher_context_sources(proposal: dict[str, Any], *, on_policy: bool) -> list[str]:
    if proposal.get("specialization_no_op") is True:
        sources = ["original_query"]
        if on_policy:
            sources.append("student_generated_prefix")
        sources.extend(_proposal_exposed_sources(proposal))
        return sources
    sources = ["original_query", "sanitized_skill_card", "proposed_candidates"]
    if on_policy:
        sources.append("student_generated_prefix")
    sources.extend(_proposal_exposed_sources(proposal))
    return sources


def _was_truncated(response_ids: torch.Tensor, max_new_tokens: int, tokenizer) -> bool:
    if int(response_ids.numel()) < max_new_tokens:
        return False
    if response_ids.numel() == 0 or tokenizer.eos_token_id is None:
        return True
    return int(response_ids[0, -1].item()) != int(tokenizer.eos_token_id)


def _parsed_answers(responses: list[str]) -> list[str]:
    return [str(extract_boxed_answer(response) or "").strip() for response in responses]


def _token_ids_sha256(token_ids: torch.Tensor) -> str:
    values = token_ids.detach().cpu().reshape(-1).tolist()
    return stable_hash(",".join(map(str, values)), 64)


def _ridge_config(args) -> dict[str, Any]:
    """Canonical query-local ridge configuration embedded in every task row."""
    return {
        "ridge_lambda": args.ridge_lambda,
        "residual_step_size": args.residual_step_size,
        "max_tokens_per_candidate": args.max_tokens_per_candidate,
        "max_support_tokens": args.max_support_tokens,
        "num_specialization_candidates": args.num_specialization_candidates,
        "hard_negatives": args.hard_negatives,
        "max_length": args.max_length,
    }


def _run_config(args) -> dict[str, Any]:
    """Return the shard/path-independent experimental configuration."""
    excluded = {
        "runtime_metadata",
        "output_dir",
        "proposals",
        "adapter_dir",
        "shard_index",
        "train_data",
        "eval_data",
        "resume",
    }
    return {
        key: value
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }


def _row_config_fields(args) -> dict[str, Any]:
    ridge_config = _ridge_config(args)
    run_config = _run_config(args)
    return {
        "ridge_config": ridge_config,
        "ridge_config_sha256": canonical_json_sha256(ridge_config),
        "run_config": run_config,
        "run_config_sha256": canonical_json_sha256(run_config),
    }


def _row_audit_fields(audit: HindsightAudit) -> dict[str, Any]:
    metrics = audit.compute()
    her = metrics["hindsight/hindsight_exposure_rate"]
    cpp = metrics["hindsight/context_parity_rate"]
    raw = {
        key: metrics[f"hindsight/{key}"]
        for key in (
            "teacher_context_events",
            "forbidden_context_events",
            "comparison_events",
            "context_equal_events",
            "compared_token_positions",
            "same_prefix_positions",
            "causal_events",
            "on_policy_events",
            "on_policy_equal_events",
            "source_counts",
        )
    }
    return {
        "hindsight_audit": raw,
        "hindsight_exposure_rate": her,
        "context_prefix_parity": cpp,
        "hindsight_free_score": (1.0 - her) * cpp,
        "same_prefix_fidelity": metrics["hindsight/same_prefix_fidelity"],
    }


_AUDIT_ROW_TO_ATTRIBUTE = {
    "teacher_context_events": "teacher_events",
    "forbidden_context_events": "exposed_events",
    "comparison_events": "comparison_events",
    "context_equal_events": "context_equal_events",
    "compared_token_positions": "compared_token_positions",
    "same_prefix_positions": "same_prefix_positions",
    "causal_events": "causal_events",
    "on_policy_events": "on_policy_events",
    "on_policy_equal_events": "on_policy_equal_events",
}


def _resume_nonnegative_integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _resume_nonnegative_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context} must be a non-negative finite number")
    return result


def _validate_resume_runtime(
    prior: dict[str, Any],
    current: dict[str, Any],
    args,
    *,
    context: str,
) -> None:
    stable_keys = (
        "python_executable",
        "conda_prefix",
        "torch_overlay",
        "torch",
        "torch_module_path",
        "torch_arch_flags",
        "cuda_runtime",
        "model",
        "requested_model_revision",
        "resolved_model_revision",
        "git_commit",
        "git_dirty",
        "slurm_array_task_id",
        "gpu_count",
    )
    for key in stable_keys:
        if key not in prior or key not in current:
            raise ValueError(f"{context}.runtime.{key} is required for safe resume")
        if prior[key] != current[key]:
            raise ValueError(f"{context}.runtime.{key} disagrees with this allocation")
    if current["model"] != args.model or current["requested_model_revision"] != (
        getattr(args, "revision", "") or ""
    ):
        raise ValueError(f"{context}.runtime model request disagrees with the active config")
    if not isinstance(current["torch_arch_flags"], list):
        raise ValueError(f"{context}.runtime.torch_arch_flags must be a list")
    gpu_signatures: dict[str, list[tuple[str, tuple[Any, ...]]]] = {}
    for label, runtime in (("prior", prior), ("current", current)):
        gpu_count = _resume_nonnegative_integer(
            runtime["gpu_count"], context=f"{context}.runtime.{label}.gpu_count"
        )
        gpus = runtime.get("gpus")
        if not isinstance(gpus, list) or len(gpus) != gpu_count:
            raise ValueError(f"{context}.runtime.{label}.gpus is missing or inconsistent")
        signatures: list[tuple[str, tuple[Any, ...]]] = []
        for gpu_index, gpu in enumerate(gpus):
            if not isinstance(gpu, dict):
                raise ValueError(
                    f"{context}.runtime.{label}.gpus[{gpu_index}] must be an object"
                )
            name = str(gpu.get("name", "")).strip()
            capability = gpu.get("capability")
            if not name or not isinstance(capability, list) or len(capability) != 2:
                raise ValueError(
                    f"{context}.runtime.{label}.gpus[{gpu_index}] lacks name/capability"
                )
            for capability_index, capability_value in enumerate(capability):
                _resume_nonnegative_integer(
                    capability_value,
                    context=(
                        f"{context}.runtime.{label}.gpus[{gpu_index}]."
                        f"capability[{capability_index}]"
                    ),
                )
            signatures.append((name, tuple(capability)))
        gpu_signatures[label] = signatures
    if gpu_signatures["prior"] != gpu_signatures["current"]:
        raise ValueError(f"{context}.runtime GPU identities disagree")

    array_task_id = str(current["slurm_array_task_id"]).strip()
    if array_task_id:
        if array_task_id != str(getattr(args, "shard_index", "")):
            raise ValueError(f"{context}.runtime.slurm_array_task_id disagrees with the shard")
        expected_python = "/home/da839/.conda/envs/TTT/bin/python"
        expected_overlay = "/home/da839/scratch_pi_mg269/da839/mfspd/pydeps-cu128"
        if (
            current["python_executable"] != expected_python
            or current["conda_prefix"] != str(Path(expected_python).parent.parent)
            or current["torch_overlay"] != expected_overlay
            or not Path(str(current["torch_module_path"])).resolve().is_relative_to(
                Path(expected_overlay).resolve()
            )
            or not str(current["torch"]).endswith("+cu128")
            or current["cuda_runtime"] != "12.8"
            or "sm_100" not in current["torch_arch_flags"]
            or current["gpu_count"] != 1
            or current["requested_model_revision"] != getattr(args, "revision", "")
            or current["resolved_model_revision"] != getattr(args, "revision", "")
        ):
            raise ValueError(f"{context}.runtime is not the pinned TTT CUDA 12.8 B200 stack")
        gpu = current["gpus"][0]
        if "B200" not in str(gpu["name"]).upper() or gpu["capability"] != [10, 0]:
            raise ValueError(f"{context}.runtime does not identify one exact B200")
        if current["git_dirty"] is not False:
            raise ValueError(f"{context}.runtime must come from a clean Git worktree")
        commit = str(current["git_commit"])
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError(f"{context}.runtime.git_commit is not a full Git SHA")


def _audit_from_completed_row(row: dict[str, Any], *, context: str) -> HindsightAudit:
    """Reconstruct and validate a query audit before accepting a resumed row."""
    raw = row.get("hindsight_audit")
    if not isinstance(raw, dict):
        raise ValueError(f"{context} is missing a hindsight_audit object")
    values = {
        attribute: _resume_nonnegative_integer(
            raw.get(row_key), context=f"{context}.hindsight_audit.{row_key}"
        )
        for row_key, attribute in _AUDIT_ROW_TO_ATTRIBUTE.items()
    }
    source_counts_raw = raw.get("source_counts")
    if not isinstance(source_counts_raw, dict) or not source_counts_raw:
        raise ValueError(f"{context}.hindsight_audit.source_counts must be non-empty")
    source_counts: dict[str, int] = {}
    for source, count in source_counts_raw.items():
        normalized = str(source).strip().lower()
        if not normalized or normalized in source_counts:
            raise ValueError(
                f"{context}.hindsight_audit.source_counts has an empty or duplicate source"
            )
        source_counts[normalized] = _resume_nonnegative_integer(
            count,
            context=f"{context}.hindsight_audit.source_counts.{normalized}",
        )
    audit = HindsightAudit(**values, source_counts=source_counts)
    if audit.teacher_events < 1:
        raise ValueError(f"{context} must contain at least one teacher-context event")
    for numerator, denominator in (
        (audit.exposed_events, audit.teacher_events),
        (audit.context_equal_events, audit.comparison_events),
        (audit.same_prefix_positions, audit.compared_token_positions),
        (audit.causal_events, audit.teacher_events),
        (audit.on_policy_equal_events, audit.on_policy_events),
    ):
        if numerator > denominator:
            raise ValueError(f"{context} contains impossible hindsight-audit counts")

    expected_fields = _row_audit_fields(audit)
    if raw != expected_fields["hindsight_audit"]:
        raise ValueError(f"{context} hindsight_audit is not canonically encoded")
    for key in (
        "hindsight_exposure_rate",
        "context_prefix_parity",
        "hindsight_free_score",
        "same_prefix_fidelity",
    ):
        supplied = row.get(key)
        if isinstance(supplied, bool) or not isinstance(supplied, (int, float)):
            raise ValueError(f"{context}.{key} must be numeric")
        supplied_float = float(supplied)
        if not math.isfinite(supplied_float) or not math.isclose(
            supplied_float,
            float(expected_fields[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{context}.{key} disagrees with its raw audit counts")
    return audit


def _read_repairable_jsonl_prefix(
    path: Path,
) -> tuple[list[dict[str, Any]], Optional[bytes]]:
    """Read JSONL and repair only an interrupted final write.

    A syntactically valid final object without its newline is retained and
    normalized. Any corruption before the final record, or an invalid record
    that was newline-terminated, fails closed.
    """
    if not path.exists():
        return [], None
    raw = path.read_bytes()
    if not raw:
        return [], None
    raw_lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    repair_required = False
    for line_index, raw_line in enumerate(raw_lines):
        context = f"{path}:{line_index + 1}"
        if not raw_line.strip():
            raise ValueError(f"{context} is an unexpected blank JSONL record")
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_final = line_index == len(raw_lines) - 1
            is_unterminated = not raw_line.endswith((b"\n", b"\r"))
            if not is_final or not is_unterminated:
                raise ValueError(
                    f"{context} is corrupt; only an unterminated final write is repairable"
                ) from exc
            repair_required = True
            break
        if not isinstance(value, dict):
            raise ValueError(f"{context} is not a JSON object")
        rows.append(value)
    if raw_lines and not raw.endswith(b"\n"):
        repair_required = True
    return rows, raw if repair_required else None


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Commit a complete per-shard prefix with an atomic same-directory rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    temporary = path.with_name(f".{path.name}.query-commit.{stamp}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_completed_row_protocol(
    row: dict[str, Any],
    audit: HindsightAudit,
    proposal: dict[str, Any],
    args,
    *,
    task: str,
    context: str,
) -> None:
    expected_context_audit = HindsightAudit()
    expected_context_audit.record_teacher_context(
        _teacher_context_sources(proposal, on_policy=False), causal=True
    )
    if task == "task1":
        if audit.teacher_events != 1 or audit.comparison_events != 1:
            raise ValueError(f"{context} is not a complete Task 1 audit")
        if audit.on_policy_events != 0 or audit.on_policy_equal_events != 0:
            raise ValueError(f"{context} incorrectly claims on-policy Task 1 events")
        if (
            audit.context_equal_events != 1
            or audit.compared_token_positions < 1
            or audit.same_prefix_positions != audit.compared_token_positions
            or row.get("student_evaluation_context_sha256")
            != row.get("teacher_evaluation_context_sha256")
        ):
            raise ValueError(f"{context} does not prove same-context Task 1 scoring")
        for key in ("base_responses", "teacher_responses"):
            if not isinstance(row.get(key), list) or not row[key]:
                raise ValueError(f"{context}.{key} is missing or empty")
    else:
        if row.get("task") != "task2_clean_distillation":
            raise ValueError(f"{context}.task is not task2_clean_distillation")
        if row.get("teacher_destroyed_before_student_evaluation") is not True:
            raise ValueError(f"{context} did not destroy its teacher before student evaluation")
        if row.get("student_reset_verified") is not True:
            raise ValueError(f"{context} did not verify query-local student reset")
        trace = row.get("distillation_trace")
        if not isinstance(trace, list):
            raise ValueError(f"{context}.distillation_trace must be a list")
        expected_steps = 0 if proposal["specialization_no_op"] else args.distillation_steps
        if len(trace) != expected_steps:
            raise ValueError(
                f"{context} trace length does not match the active distillation config"
            )
        compared_positions = 0
        for step_index, step in enumerate(trace):
            step_context = f"{context}.distillation_trace[{step_index}]"
            if not isinstance(step, dict) or step.get("same_prefix") is not True:
                raise ValueError(f"{step_context} does not prove same-prefix distillation")
            if step.get("step") != step_index:
                raise ValueError(f"{step_context}.step is not the canonical step index")
            if step.get("student_context_sha256") != step.get("teacher_context_sha256"):
                raise ValueError(f"{step_context} context hashes disagree")
            positions = _resume_nonnegative_integer(
                step.get("compared_positions"), context=f"{step_context}.compared_positions"
            )
            prefix_tokens = _resume_nonnegative_integer(
                step.get("prefix_tokens"), context=f"{step_context}.prefix_tokens"
            )
            if positions < 1 or positions != prefix_tokens:
                raise ValueError(f"{step_context} has invalid compared positions")
            prefix_ids = step.get("student_prefix_token_ids")
            if not isinstance(prefix_ids, list) or len(prefix_ids) != prefix_tokens:
                raise ValueError(f"{step_context} token IDs do not match prefix_tokens")
            for token_index, token_id in enumerate(prefix_ids):
                _resume_nonnegative_integer(
                    token_id,
                    context=f"{step_context}.student_prefix_token_ids[{token_index}]",
                )
            compared_positions += positions
            expected_context_audit.record_teacher_context(
                _teacher_context_sources(proposal, on_policy=True), causal=True
            )
        if (
            audit.teacher_events != len(trace) + 1
            or audit.comparison_events != len(trace)
            or audit.context_equal_events != len(trace)
            or audit.on_policy_events != len(trace)
            or audit.on_policy_equal_events != len(trace)
            or audit.compared_token_positions != compared_positions
            or audit.same_prefix_positions != compared_positions
        ):
            raise ValueError(f"{context} audit does not match its Task 2 trace")
        completed_steps = _resume_nonnegative_integer(
            row.get("distillation_steps_completed"),
            context=f"{context}.distillation_steps_completed",
        )
        if completed_steps != len(trace):
            raise ValueError(f"{context} completed-step count does not match its Task 2 trace")
        losses = row.get("distillation_losses")
        if not isinstance(losses, list) or len(losses) != len(trace):
            raise ValueError(f"{context} loss count does not match its Task 2 trace")
        finite_losses: list[float] = []
        for loss_index, (loss, step) in enumerate(zip(losses, trace)):
            if isinstance(loss, bool) or not isinstance(loss, (int, float)):
                raise ValueError(f"{context}.distillation_losses[{loss_index}] is not numeric")
            loss_value = float(loss)
            trace_loss = step.get("loss")
            if (
                not math.isfinite(loss_value)
                or isinstance(trace_loss, bool)
                or not isinstance(trace_loss, (int, float))
                or not math.isfinite(float(trace_loss))
                or loss_value != float(trace_loss)
            ):
                raise ValueError(f"{context} loss values do not match its Task 2 trace")
            finite_losses.append(loss_value)
        mean_loss = row.get("mean_distillation_loss")
        if isinstance(mean_loss, bool) or not isinstance(mean_loss, (int, float)):
            raise ValueError(f"{context}.mean_distillation_loss is not numeric")
        expected_mean_loss = sum(finite_losses) / max(len(finite_losses), 1)
        if not math.isfinite(float(mean_loss)) or not math.isclose(
            float(mean_loss), expected_mean_loss, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{context}.mean_distillation_loss disagrees with its losses")
        student_update_norm = _resume_nonnegative_number(
            row.get("student_update_frobenius_norm"),
            context=f"{context}.student_update_frobenius_norm",
        )
        if proposal["specialization_no_op"]:
            if student_update_norm != 0.0:
                raise ValueError(f"{context} no-op student update must be exactly zero")
        elif student_update_norm <= 0.0:
            raise ValueError(f"{context} ready student update must be positive")
        for key in ("base_responses", "teacher_responses", "distilled_responses"):
            if not isinstance(row.get(key), list) or not row[key]:
                raise ValueError(f"{context}.{key} is missing or empty")

    if (
        audit.teacher_events != expected_context_audit.teacher_events
        or audit.exposed_events != expected_context_audit.exposed_events
        or audit.causal_events != expected_context_audit.causal_events
        or audit.source_counts != expected_context_audit.source_counts
    ):
        raise ValueError(f"{context} audit provenance disagrees with its proposal and protocol")


_CACHED_SPECIALIZATION_METRIC_KEYS = (
    "specialization_status",
    "specialization_failure_reason",
    "specialization_no_op",
    "uses_all_candidates",
    "specialization_seconds",
    "feature_extraction_seconds",
    "closed_form_solve_seconds",
    "support_tokens",
    "requested_max_support_tokens",
    "allocated_support_token_budget",
    "required_answer_tokens",
    "support_budget_expanded",
    "support_budget_overflow_tokens",
    "adapted_vocab_size",
    "adapter_rank",
    "ridge_lambda_effective",
    "update_frobenius_norm",
    "peak_memory_bytes",
    "proposal_fit_target_logit_gain",
    "proposal_base_target_nll",
)


def _validate_cached_manifest_for_record(
    manifest: dict[str, Any],
    record: dict[str, Any],
    proposal: dict[str, Any],
    args,
    *,
    context: str,
) -> None:
    expected_revision = str(
        args.runtime_metadata.get(
            "resolved_model_revision", getattr(args, "revision", "") or ""
        )
    )
    mismatched = (
        manifest.get("query_id") != record["query_id"]
        or str(manifest.get("problem_sha256", "")) != record["problem_sha256"]
        or str(manifest.get("source", "")).strip().lower() != record["source"]
        or manifest.get("proposal_training_sha256")
        != proposal["proposal_training_sha256"]
        or manifest.get("model") != args.model
        or manifest.get("model_revision") != expected_revision
        or manifest.get("specialization_status") != proposal["specialization_status"]
        or manifest.get("specialization_failure_reason")
        != proposal["specialization_failure_reason"]
        or manifest.get("specialization_no_op") is not proposal["specialization_no_op"]
    )
    if mismatched:
        raise ValueError(
            f"{context}: cached adapter identity/state does not match the "
            "dataset, proposal, model, and revision"
        )
    missing_metrics = [
        key for key in _CACHED_SPECIALIZATION_METRIC_KEYS if key not in manifest
    ]
    if missing_metrics:
        raise ValueError(
            f"{context}: cached adapter manifest is missing metrics {missing_metrics}"
        )


def _load_resumable_evaluation_rows(
    detail_path: Path,
    records: list[dict[str, Any]],
    proposals: dict[str, dict[str, Any]],
    proposals_by_hash: dict[str, list[dict[str, Any]]],
    args,
    *,
    task: str,
    stage: Optional[str] = None,
    adapter_cache: Optional[
        dict[str, tuple[SparseRidgeAdapter, dict[str, Any]]]
    ] = None,
) -> tuple[list[dict[str, Any]], HindsightAudit, dict[str, HindsightAudit]]:
    """Accept only a fully bound, exact record prefix from a prior allocation."""
    rows, repair_bytes = _read_repairable_jsonl_prefix(detail_path)
    if len(rows) > len(records):
        raise ValueError(
            f"Resume artifact {detail_path} has {len(rows)} rows for only {len(records)} records"
        )
    expected_config = _row_config_fields(args)
    expected_revision = str(
        args.runtime_metadata.get(
            "resolved_model_revision", getattr(args, "revision", "") or ""
        )
    )
    audit = HindsightAudit()
    audits_by_source: dict[str, HindsightAudit] = defaultdict(HindsightAudit)
    for index, row in enumerate(rows):
        context = f"Resume artifact {detail_path} row {index + 1}"
        record = records[index]
        proposal = _proposal_for(record, proposals, proposals_by_hash)
        if row.get("query_id") != record["query_id"]:
            raise ValueError(
                f"{context} query_id={row.get('query_id')!r} is not the exact dataset prefix "
                f"query_id={record['query_id']!r}"
            )
        if row.get("problem") != record["problem"]:
            raise ValueError(f"{context} problem text disagrees with the dataset")
        if row.get("problem_sha256") != record["problem_sha256"]:
            raise ValueError(f"{context} problem hash disagrees with the dataset")
        if str(row.get("source", "")).strip().lower() != record["source"]:
            raise ValueError(f"{context} source disagrees with the dataset")
        if str(row.get("reference_answer", "")) != str(record.get("answer", "")):
            raise ValueError(f"{context} reference answer disagrees with the dataset")
        if row.get("proposal_training_sha256") != proposal["proposal_training_sha256"]:
            raise ValueError(f"{context} proposal training binding disagrees")
        for key in (
            "specialization_status",
            "specialization_failure_reason",
            "specialization_no_op",
        ):
            if row.get(key) != proposal.get(key):
                raise ValueError(f"{context}.{key} disagrees with its proposal")
        if row.get("model") != args.model or row.get("model_revision") != expected_revision:
            raise ValueError(f"{context} model or resolved revision disagrees")
        if task == "task1" and row.get("stage") != stage:
            raise ValueError(f"{context}.stage disagrees with the requested evaluation stage")
        if task == "task1" and getattr(args, "privileged_control", False):
            if not isinstance(row.get("privileged_responses"), list) or not row[
                "privileged_responses"
            ]:
                raise ValueError(f"{context}.privileged_responses is missing or empty")
        for key, expected_value in expected_config.items():
            if row.get(key) != expected_value:
                raise ValueError(f"{context}.{key} disagrees with the active configuration")
        prior_runtime = row.get("runtime")
        if not isinstance(prior_runtime, dict):
            raise ValueError(f"{context}.runtime must be an object")
        _validate_resume_runtime(
            prior_runtime,
            args.runtime_metadata,
            args,
            context=context,
        )
        if getattr(args, "adapter_dir", None):
            if adapter_cache is None or record["query_id"] not in adapter_cache:
                raise ValueError(f"{context} is missing its required cached adapter")
            manifest = adapter_cache[record["query_id"]][1]
            _validate_cached_manifest_for_record(
                manifest,
                record,
                proposal,
                args,
                context=context,
            )
            for key in _CACHED_SPECIALIZATION_METRIC_KEYS:
                if key not in row:
                    raise ValueError(f"{context}.{key} is required by its cached adapter")
                if row[key] != manifest[key]:
                    raise ValueError(f"{context}.{key} disagrees with its cached adapter")
        query_audit = _audit_from_completed_row(row, context=context)
        _validate_completed_row_protocol(
            row,
            query_audit,
            proposal,
            args,
            task=task,
            context=context,
        )
        audit.merge(query_audit)
        audits_by_source[record["source"]].merge(query_audit)
    if repair_bytes is not None:
        stamp = time.time_ns()
        backup = detail_path.with_name(f"{detail_path.name}.resume-backup.{stamp}")
        backup.write_bytes(repair_bytes)
        _atomic_write_jsonl(detail_path, rows)
    return rows, audit, audits_by_source


def _accuracy_teacher_gain_retention(rows: list[dict[str, Any]]) -> Optional[float]:
    if not rows:
        return None
    base = sum(float(row["base_correct"]) for row in rows) / len(rows)
    teacher = sum(float(row["teacher_correct"]) for row in rows) / len(rows)
    student = sum(float(row["distilled_correct"]) for row in rows) / len(rows)
    teacher_gain = teacher - base
    if teacher_gain <= 0:
        return None
    return (student - base) / teacher_gain


def evaluate(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    proposals: dict[str, dict[str, Any]],
    args,
    *,
    stage: str,
    adapter_cache: dict[str, tuple[SparseRidgeAdapter, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any], HindsightAudit]:
    was_training = model.training
    model.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"eval_{stage}.jsonl"
    if detail_path.exists() and not getattr(args, "resume", False):
        detail_path.unlink()
    proposals_by_hash = _index_proposals_by_hash(proposals)
    if getattr(args, "adapter_dir", None):
        expected_cache_ids = {record["query_id"] for record in records}
        actual_cache_ids = set(adapter_cache)
        if actual_cache_ids != expected_cache_ids:
            missing = sorted(expected_cache_ids - actual_cache_ids)
            extra = sorted(actual_cache_ids - expected_cache_ids)
            raise ValueError(
                "Cached adapter coverage does not match this evaluation shard: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
    rows, audit, audits_by_source = _load_resumable_evaluation_rows(
        detail_path,
        records,
        proposals,
        proposals_by_hash,
        args,
        task="task1",
        stage=stage,
        adapter_cache=adapter_cache,
    )
    completed = len(rows)

    for index, record in enumerate(
        tqdm(
            records[completed:],
            desc=f"evaluate:{stage}",
            initial=completed,
            total=len(records),
        ),
        start=completed,
    ):
        proposal = _proposal_for(record, proposals, proposals_by_hash)
        exposed_sources = set(_proposal_exposed_sources(proposal))
        if exposed_sources and not args.allow_hindsight_exposure:
            raise ValueError(
                f"Proposal {proposal.get('query_id')} is marked as hindsight-contaminated by "
                f"{sorted(exposed_sources)}. Pass --allow-hindsight-exposure only for an ablation."
            )
        cached = adapter_cache.get(record["query_id"])
        if cached is None:
            adapter, specialization_metrics = _fit_current_adapter(model, tokenizer, proposal, args)
        else:
            adapter, manifest = cached
            _validate_cached_manifest_for_record(
                manifest,
                record,
                proposal,
                args,
                context=f"Cached adapter {record['query_id']}",
            )
            adapter = adapter.to(input_device(model))
            specialization_metrics = {
                key: manifest[key]
                for key in _CACHED_SPECIALIZATION_METRIC_KEYS
            }

        target_completion = _target_completion(record, args.target_score_mode)
        base_nll, teacher_nll, scored_ids = score_target_completion(
            model, tokenizer, record["problem"], target_completion, adapter
        )
        student_scored_ids = scored_ids.detach().clone()
        teacher_scored_ids = scored_ids.detach().clone()
        # Ground truth is a causal evaluation continuation, not a teacher
        # construction source. Both models receive exactly the same prefixes.
        context_sources = _teacher_context_sources(proposal, on_policy=False)
        query_audit = HindsightAudit()
        query_audit.record_teacher_context(context_sources, causal=True)
        query_audit.record_same_prefix(
            student_scored_ids,
            teacher_scored_ids,
            positions=scored_ids.shape[1],
            on_policy=False,
        )
        audit.record_teacher_context(context_sources, causal=True)
        audit.record_same_prefix(
            student_scored_ids,
            teacher_scored_ids,
            positions=scored_ids.shape[1],
            on_policy=False,
        )
        source_audit = audits_by_source[record["source"]]
        source_audit.record_teacher_context(context_sources, causal=True)
        source_audit.record_same_prefix(
            student_scored_ids,
            teacher_scored_ids,
            positions=scored_ids.shape[1],
            on_policy=False,
        )

        base_scores = []
        teacher_scores = []
        base_responses = []
        teacher_responses = []
        base_response_tokens = []
        teacher_response_tokens = []
        privileged_scores = []
        privileged_wrong_scores = []
        privileged_jsds = []
        privileged_answer_flips = []
        privileged_responses = []
        privileged_wrong_responses = []
        base_truncated = []
        teacher_truncated = []
        privileged_truncated = []
        base_generation_seconds = 0.0
        teacher_generation_seconds = 0.0
        privileged_generation_seconds = 0.0
        max_input_tokens = 0
        for sample_index in range(args.eval_samples):
            seed = args.seed + index * 1009 + sample_index
            generation_started = time.perf_counter()
            base_response, base_prompt_ids, base_ids = generate_response(
                model,
                tokenizer,
                record["problem"],
                adapter=None,
                max_new_tokens=args.eval_max_new_tokens,
                temperature=args.eval_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=seed,
            )
            base_generation_seconds += time.perf_counter() - generation_started
            generation_started = time.perf_counter()
            teacher_response, teacher_prompt_ids, teacher_ids = generate_response(
                model,
                tokenizer,
                record["problem"],
                adapter=adapter,
                max_new_tokens=args.eval_max_new_tokens,
                temperature=args.eval_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=seed,
            )
            teacher_generation_seconds += time.perf_counter() - generation_started
            base_scores.append(_grade_response(base_response, record["answer"]))
            teacher_scores.append(_grade_response(teacher_response, record["answer"]))
            base_responses.append(base_response)
            teacher_responses.append(teacher_response)
            base_response_tokens.append(int(base_ids.numel()))
            teacher_response_tokens.append(int(teacher_ids.numel()))
            base_truncated.append(_was_truncated(base_ids, args.eval_max_new_tokens, tokenizer))
            teacher_truncated.append(
                _was_truncated(teacher_ids, args.eval_max_new_tokens, tokenizer)
            )
            max_input_tokens = max(
                max_input_tokens,
                int(base_prompt_ids.numel()),
                int(teacher_prompt_ids.numel()),
            )

            if args.privileged_control:
                wrong_hint = _wrong_answer_hint(record["answer"])
                privileged_prompt = problem_prompt(
                    tokenizer,
                    _hinted_problem(record["problem"], record["answer"]),
                )
                generation_started = time.perf_counter()
                privileged_response, privileged_prompt_ids, privileged_ids = generate_response(
                    model,
                    tokenizer,
                    record["problem"],
                    adapter=None,
                    max_new_tokens=args.eval_max_new_tokens,
                    temperature=args.eval_temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=seed,
                    prompt_override=privileged_prompt,
                )
                privileged_generation_seconds += time.perf_counter() - generation_started
                privileged_scores.append(_grade_response(privileged_response, record["answer"]))
                privileged_responses.append(privileged_response)
                privileged_truncated.append(
                    _was_truncated(privileged_ids, args.eval_max_new_tokens, tokenizer)
                )
                max_input_tokens = max(max_input_tokens, int(privileged_prompt_ids.numel()))
                wrong_prompt = problem_prompt(
                    tokenizer,
                    _hinted_problem(record["problem"], wrong_hint),
                )
                wrong_response, _, _ = generate_response(
                    model,
                    tokenizer,
                    record["problem"],
                    adapter=None,
                    max_new_tokens=args.eval_max_new_tokens,
                    temperature=args.eval_temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=seed,
                    prompt_override=wrong_prompt,
                )
                privileged_wrong_scores.append(
                    _grade_response(wrong_response, record["answer"])
                )
                privileged_wrong_responses.append(wrong_response)
                privileged_answer_flips.append(
                    float(
                        str(extract_boxed_answer(privileged_response) or "").strip()
                        != str(extract_boxed_answer(wrong_response) or "").strip()
                    )
                )
                privileged_jsds.append(
                    counterfactual_hint_jsd(
                        model,
                        tokenizer,
                        record["problem"],
                        base_response,
                        record["answer"],
                        wrong_hint,
                        max_positions=args.hindsight_prefix_tokens,
                    )
                )

        row = {
            "stage": stage,
            "query_id": record["query_id"],
            "problem": record["problem"],
            "problem_sha256": record["problem_sha256"],
            "proposal_training_sha256": proposal["proposal_training_sha256"],
            "specialization_status": proposal["specialization_status"],
            "specialization_failure_reason": proposal[
                "specialization_failure_reason"
            ],
            "specialization_no_op": proposal["specialization_no_op"],
            "reference_answer": record["answer"],
            "source": record["source"],
            "model": args.model,
            "model_revision": args.runtime_metadata.get("resolved_model_revision", ""),
            "runtime": args.runtime_metadata,
            **_row_config_fields(args),
            "base_correct": sum(base_scores) / len(base_scores),
            "teacher_correct": sum(teacher_scores) / len(teacher_scores),
            "base_pass_at_n": float(any(base_scores)),
            "teacher_pass_at_n": float(any(teacher_scores)),
            "base_majority_at_n": _majority_grade(base_responses, record["answer"]),
            "teacher_majority_at_n": _majority_grade(teacher_responses, record["answer"]),
            "base_target_answer_nll": base_nll,
            "teacher_target_answer_nll": teacher_nll,
            "target_answer_nll_gain": base_nll - teacher_nll,
            "student_evaluation_context_sha256": _token_ids_sha256(student_scored_ids),
            "teacher_evaluation_context_sha256": _token_ids_sha256(teacher_scored_ids),
            "base_responses": base_responses,
            "teacher_responses": teacher_responses,
            "base_parsed_answers": _parsed_answers(base_responses),
            "teacher_parsed_answers": _parsed_answers(teacher_responses),
            "base_generated_tokens": sum(base_response_tokens),
            "teacher_generated_tokens": sum(teacher_response_tokens),
            "base_generation_seconds": base_generation_seconds,
            "teacher_generation_seconds": teacher_generation_seconds,
            "base_truncated": bool(any(base_truncated)),
            "teacher_truncated": bool(any(teacher_truncated)),
            "base_truncated_count": sum(base_truncated),
            "teacher_truncated_count": sum(teacher_truncated),
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": args.eval_max_new_tokens,
            "support_generated_tokens": float(
                proposal.get("cost_audit", {}).get("total_completion_tokens", 0.0)
            ),
            "support_generation_seconds": float(
                proposal.get("cost_audit", {}).get("total_generation_seconds", 0.0)
            ),
            "proposal_end_to_end_seconds": float(
                proposal.get("cost_audit", {}).get("end_to_end_seconds", 0.0)
            ),
            **specialization_metrics,
            **_row_audit_fields(query_audit),
        }
        row["total_adaptation_seconds"] = (
            row["proposal_end_to_end_seconds"] + row["specialization_seconds"]
        )
        if privileged_scores:
            privileged_correct = sum(privileged_scores) / len(privileged_scores)
            clean_gain = row["teacher_correct"] - row["base_correct"]
            privileged_gain = privileged_correct - row["base_correct"]
            row.update(
                {
                    "privileged_correct": privileged_correct,
                    "privileged_responses": privileged_responses,
                    "privileged_wrong_hint_responses": privileged_wrong_responses,
                    "privileged_parsed_answers": _parsed_answers(privileged_responses),
                    "privileged_generated_tokens": sum(
                        len(tokenizer(response, add_special_tokens=False)["input_ids"])
                        for response in privileged_responses
                    ),
                    "privileged_generation_seconds": privileged_generation_seconds,
                    "privileged_truncated": bool(any(privileged_truncated)),
                    "privileged_truncated_count": sum(privileged_truncated),
                    "privileged_hindsight_exposure_rate": 1.0,
                    "privileged_context_prefix_parity": 0.0,
                    "privileged_hindsight_free_score": 0.0,
                    "privileged_wrong_hint_correct": sum(privileged_wrong_scores)
                    / len(privileged_wrong_scores),
                    "privileged_counterfactual_jsd": sum(privileged_jsds)
                    / len(privileged_jsds),
                    "privileged_answer_flip_rate": sum(privileged_answer_flips)
                    / len(privileged_answer_flips),
                    "clean_counterfactual_jsd": 0.0,
                    "clean_answer_flip_rate": 0.0,
                    "clean_advantage_retention": max(0.0, min(1.0, clean_gain / privileged_gain))
                    if privileged_gain > 0
                    else 0.0,
                }
            )
        _validate_completed_row_protocol(
            row,
            query_audit,
            proposal,
            args,
            task="task1",
            context=f"New Task 1 row {record['query_id']}",
        )
        rows.append(row)
        _atomic_write_jsonl(detail_path, rows)

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    summary: dict[str, Any] = {"stage": stage, "overall": aggregate_teacher_metrics(rows, audit)}
    summary["by_source"] = {
        source: aggregate_teacher_metrics(source_rows, audits_by_source[source])
        for source, source_rows in by_source.items()
    }
    with (output_dir / f"metrics_{stage}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    model.train(was_training)
    return rows, summary, audit


def support_icl_evaluate(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    proposals: dict[str, dict[str, Any]],
    args,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute-matched support ICL using the exact verified candidates."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "eval_support_icl.jsonl"
    if detail_path.exists():
        detail_path.unlink()
    proposals_by_hash = _index_proposals_by_hash(proposals)
    rows = []
    for index, record in enumerate(tqdm(records, desc="support ICL")):
        proposal = _proposal_for(record, proposals, proposals_by_hash)
        candidates = list(proposal.get("specialization_candidates", []))
        if args.num_specialization_candidates is not None:
            candidates = candidates[: args.num_specialization_candidates]
        demonstrations = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            demonstrations.append(
                f"Example {candidate_index}\nProblem: {candidate['problem']}\n"
                f"Solution: {candidate.get('solution', '')}\n"
                f"Final answer: \\boxed{{{candidate.get('final_answer', '')}}}"
            )
        icl_problem = (
            "Use the solved examples below only as demonstrations of reusable mathematical skills.\n\n"
            + "\n\n".join(demonstrations)
            + f"\n\nTARGET PROBLEM\n{record['problem']}"
        )
        prompt_override = problem_prompt(tokenizer, icl_problem)
        responses = []
        scores = []
        response_tokens = []
        for sample_index in range(args.eval_samples):
            response, _, response_ids = generate_response(
                model,
                tokenizer,
                record["problem"],
                adapter=None,
                max_new_tokens=args.eval_max_new_tokens,
                temperature=args.eval_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed + index * 1009 + sample_index,
                prompt_override=prompt_override,
            )
            responses.append(response)
            scores.append(_grade_response(response, record["answer"]))
            response_tokens.append(int(response_ids.numel()))
        row = {
            "method": "support_icl",
            "query_id": record["query_id"],
            "source": record["source"],
            "correct": sum(scores) / len(scores),
            "pass_at_n": float(any(scores)),
            "majority_at_n": _majority_grade(responses, record["answer"]),
            "responses": responses,
            "generated_tokens": sum(response_tokens),
            "support_generated_tokens": float(
                proposal.get("cost_audit", {}).get("total_completion_tokens", 0.0)
            ),
        }
        rows.append(row)
        write_jsonl(detail_path, [row], append=index > 0)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "accuracy/mean_at_n": sum(float(row["correct"]) for row in group) / max(len(group), 1),
            "accuracy/pass_at_n": sum(float(row["pass_at_n"]) for row in group)
            / max(len(group), 1),
            "accuracy/majority_at_n": sum(float(row["majority_at_n"]) for row in group)
            / max(len(group), 1),
        }

    summary = {
        "method": "support_icl",
        "overall": summarize(rows),
        "by_source": {source: summarize(group) for source, group in by_source.items()},
    }
    with (output_dir / "metrics_support_icl.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return rows, summary


def _support_supervised_loss(
    model,
    tokenizer,
    candidates: list[dict[str, Any]],
    *,
    max_length: int,
    max_support_tokens: int,
    max_tokens_per_candidate: int,
):
    """Average candidate-solution CE used by iterative adaptation baselines."""
    device = input_device(model)
    losses = []
    if not candidates:
        raise ValueError("At least one specialization candidate is required")
    if max_support_tokens < len(candidates):
        raise ValueError("max_support_tokens must allocate at least one token per candidate")
    base_allocation, remainder = divmod(max_support_tokens, len(candidates))
    allocations = [
        min(max_tokens_per_candidate, base_allocation + int(index < remainder))
        for index in range(len(candidates))
    ]
    for candidate, token_budget in zip(candidates, allocations):
        prompt_ids = tokenizer(
            problem_prompt(tokenizer, str(candidate["problem"])),
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"].to(device)
        completion_ids = tokenizer(
            candidate_completion(candidate),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        available = max_length - prompt_ids.shape[1]
        completion_ids = completion_ids[:, : max(available, 0)]
        if completion_ids.numel() == 0:
            continue
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        hidden_all, _ = backbone_forward(
            model,
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            use_cache=False,
        )
        start = prompt_ids.shape[1] - 1
        hidden = hidden_all[:, start : start + completion_ids.shape[1]]
        if completion_ids.shape[1] > token_budget:
            positions = (
                torch.linspace(
                    0,
                    completion_ids.shape[1] - 1,
                    steps=token_budget,
                    device=device,
                )
                .round()
                .long()
                .unique()
            )
            hidden = hidden.index_select(1, positions)
            completion_ids = completion_ids.index_select(1, positions)
        logits = project_logits(model, hidden)
        losses.append(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                completion_ids.reshape(-1),
                reduction="mean",
            )
        )
    if not losses:
        raise ValueError("No candidate solution tokens were available for supervised adaptation")
    return torch.stack(losses).mean()


def per_query_support_sft_evaluate(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    proposals: dict[str, dict[str, Any]],
    args,
    *,
    method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Query-local Support-LoRA or LM-head SGD with exact per-item reset."""
    if method not in {"support_lora", "head_sgd"}:
        raise ValueError(f"Unknown supervised support baseline: {method}")
    if method == "support_lora" and not hasattr(model, "disable_adapter"):
        raise RuntimeError("support_lora requires a PEFT model")
    if method == "head_sgd":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        output_head = unwrap_causal_lm(model).get_output_embeddings()
        if output_head is None:
            raise ValueError("The model has no LM head for head_sgd")
        for parameter in output_head.parameters():
            parameter.requires_grad_(True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"eval_{method}.jsonl"
    if detail_path.exists():
        detail_path.unlink()
    proposals_by_hash = _index_proposals_by_hash(proposals)
    initial_state = _capture_trainable_state(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"{method} has no trainable parameters")
    device = input_device(model)
    rows: list[dict[str, Any]] = []

    for index, record in enumerate(tqdm(records, desc=f"baseline:{method}")):
        _restore_trainable_state(model, initial_state)
        proposal = _proposal_for(record, proposals, proposals_by_hash)
        candidates = list(proposal.get("specialization_candidates", []))
        if args.num_specialization_candidates is not None:
            candidates = candidates[: args.num_specialization_candidates]
        target_completion = _target_completion(record, args.target_score_mode)
        with base_policy_context(model) if method == "support_lora" else contextlib.nullcontext():
            base_nll = score_plain_completion(model, tokenizer, record["problem"], target_completion)
            base_response, _, base_ids = generate_response(
                model,
                tokenizer,
                record["problem"],
                adapter=None,
                max_new_tokens=args.eval_max_new_tokens,
                temperature=args.eval_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed + index * 1009,
            )

        baseline_memory = 0.0
        if device.type == "cuda":
            baseline_memory = float(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer = (
            torch.optim.AdamW(parameters, lr=args.baseline_learning_rate, weight_decay=args.weight_decay)
            if method == "support_lora"
            else torch.optim.SGD(parameters, lr=args.baseline_learning_rate)
        )
        losses = []
        model.train()
        for _ in range(args.baseline_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = _support_supervised_loss(
                model,
                tokenizer,
                candidates,
                max_length=args.max_length,
                max_support_tokens=args.max_support_tokens,
                max_tokens_per_candidate=args.max_tokens_per_candidate,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        adaptation_seconds = time.perf_counter() - started
        peak_bytes = (
            max(float(torch.cuda.max_memory_allocated(device)) - baseline_memory, 0.0)
            if device.type == "cuda"
            else 0.0
        )

        model.eval()
        response, _, response_ids = generate_response(
            model,
            tokenizer,
            record["problem"],
            adapter=None,
            max_new_tokens=args.eval_max_new_tokens,
            temperature=args.eval_temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed + index * 1009,
        )
        adapted_nll = score_plain_completion(
            model, tokenizer, record["problem"], target_completion
        )
        row = {
            "method": method,
            "query_id": record["query_id"],
            "source": record["source"],
            "base_correct": _grade_response(base_response, record["answer"]),
            "correct": _grade_response(response, record["answer"]),
            "base_target_answer_nll": base_nll,
            "adapted_target_answer_nll": adapted_nll,
            "target_answer_nll_gain": base_nll - adapted_nll,
            "adaptation_seconds": adaptation_seconds,
            "peak_memory_bytes": peak_bytes,
            "steps_completed": len(losses),
            "mean_support_loss": sum(losses) / max(len(losses), 1),
            "base_response": base_response,
            "response": response,
            "base_generated_tokens": int(base_ids.numel()),
            "generated_tokens": int(response_ids.numel()),
            "support_generated_tokens": float(
                proposal.get("cost_audit", {}).get("total_completion_tokens", 0.0)
            ),
        }
        rows.append(row)
        write_jsonl(detail_path, [row], append=index > 0)
        del optimizer
        _restore_trainable_state(model, initial_state)

    summary = {
        "method": method,
        "protocol": "per_query_reset",
        "overall": {
            "accuracy/base": sum(float(row["base_correct"]) for row in rows) / max(len(rows), 1),
            "accuracy/adapted": sum(float(row["correct"]) for row in rows) / max(len(rows), 1),
            "teacher/target_answer_nll_gain": sum(
                float(row["target_answer_nll_gain"]) for row in rows
            )
            / max(len(rows), 1),
            "speed/mean_adaptation_seconds": sum(float(row["adaptation_seconds"]) for row in rows)
            / max(len(rows), 1),
            "speed/mean_peak_memory_bytes": sum(float(row["peak_memory_bytes"]) for row in rows)
            / max(len(rows), 1),
        },
    }
    with (output_dir / f"metrics_{method}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _restore_trainable_state(model, initial_state)
    return rows, summary


def per_query_distill_evaluate(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    proposals: dict[str, dict[str, Any]],
    args,
) -> tuple[list[dict[str, Any]], dict[str, Any], HindsightAudit]:
    """Paper Task 2: query-local CSD-SD with an exact reset per benchmark item."""
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("Task 2 requires a PEFT/LoRA student; do not pass --full-finetune")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "eval_task2_clean_distillation.jsonl"
    if detail_path.exists() and not getattr(args, "resume", False):
        detail_path.unlink()
    proposals_by_hash = _index_proposals_by_hash(proposals)
    initial_student_state = _capture_trainable_state(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    rows, audit, audits_by_source = _load_resumable_evaluation_rows(
        detail_path,
        records,
        proposals,
        proposals_by_hash,
        args,
        task="task2",
    )
    completed = len(rows)

    for index, record in enumerate(
        tqdm(
            records[completed:],
            desc="task2:query-local distillation",
            initial=completed,
            total=len(records),
        ),
        start=completed,
    ):
        _restore_trainable_state(model, initial_student_state)
        proposal = _proposal_for(record, proposals, proposals_by_hash)
        specialization_no_op = proposal["specialization_no_op"]
        exposed_sources = set(_proposal_exposed_sources(proposal))
        if exposed_sources and not args.allow_hindsight_exposure:
            raise ValueError(
                f"Proposal {proposal.get('query_id')} is hindsight-contaminated by "
                f"{sorted(exposed_sources)}"
            )

        # CSD-T is anchored to the untouched base checkpoint. The LoRA student
        # is disabled during support feature extraction and every teacher pass.
        with base_policy_context(model):
            teacher_adapter, specialization_metrics = _fit_current_adapter(
                model, tokenizer, proposal, args
            )
        base_responses: list[str] = []
        teacher_responses: list[str] = []
        base_response_tokens: list[int] = []
        teacher_response_tokens: list[int] = []
        base_truncated: list[bool] = []
        teacher_truncated: list[bool] = []
        base_generation_seconds = 0.0
        teacher_generation_seconds = 0.0
        max_input_tokens = 0
        for sample_index in range(args.eval_samples):
            seed = args.seed + index * 1009 + sample_index
            with base_policy_context(model):
                generation_started = time.perf_counter()
                base_response, base_prompt_ids, base_ids = generate_response(
                    model,
                    tokenizer,
                    record["problem"],
                    adapter=None,
                    max_new_tokens=args.eval_max_new_tokens,
                    temperature=args.eval_temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=seed,
                )
                base_generation_seconds += time.perf_counter() - generation_started
                generation_started = time.perf_counter()
                teacher_response, teacher_prompt_ids, teacher_ids = generate_response(
                    model,
                    tokenizer,
                    record["problem"],
                    adapter=teacher_adapter,
                    max_new_tokens=args.eval_max_new_tokens,
                    temperature=args.eval_temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=seed,
                )
                teacher_generation_seconds += time.perf_counter() - generation_started
            base_responses.append(base_response)
            teacher_responses.append(teacher_response)
            base_response_tokens.append(int(base_ids.numel()))
            teacher_response_tokens.append(int(teacher_ids.numel()))
            base_truncated.append(_was_truncated(base_ids, args.eval_max_new_tokens, tokenizer))
            teacher_truncated.append(
                _was_truncated(teacher_ids, args.eval_max_new_tokens, tokenizer)
            )
            max_input_tokens = max(
                max_input_tokens,
                int(base_prompt_ids.numel()),
                int(teacher_prompt_ids.numel()),
            )

        distill_device = input_device(model)
        distillation_memory_baseline = 0.0
        if distill_device.type == "cuda":
            torch.cuda.synchronize(distill_device)
            distillation_memory_baseline = float(torch.cuda.memory_allocated(distill_device))
            torch.cuda.reset_peak_memory_stats(distill_device)
        optimizer = (
            None
            if specialization_no_op
            else torch.optim.AdamW(
                parameters,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        )
        distillation_losses: list[float] = []
        distillation_rollout_tokens = 0
        distillation_trace: list[dict[str, Any]] = []
        query_audit = HindsightAudit()
        construction_sources = _teacher_context_sources(proposal, on_policy=False)
        query_audit.record_teacher_context(construction_sources, causal=True)
        audit.record_teacher_context(construction_sources, causal=True)
        audits_by_source[record["source"]].record_teacher_context(
            construction_sources, causal=True
        )
        if distill_device.type == "cuda":
            torch.cuda.synchronize(distill_device)
        distillation_started = time.perf_counter()
        distillation_steps = 0 if specialization_no_op else args.distillation_steps
        for distill_step in range(distillation_steps):
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            prefix_response, prompt_ids, response_ids = generate_response(
                model,
                tokenizer,
                record["problem"],
                adapter=None,
                max_new_tokens=args.train_max_new_tokens,
                temperature=args.train_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed + index * 100_003 + distill_step,
            )
            if response_ids.numel() == 0:
                continue
            distillation_rollout_tokens += int(response_ids.numel())
            student_full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            # Build the teacher causal input independently so the parity audit
            # compares two serialized inputs instead of asserting x == x.
            teacher_full_ids = torch.cat(
                [prompt_ids.detach().clone(), response_ids.detach().clone()], dim=1
            )
            model.train()
            student_hidden_all, _ = backbone_forward(
                model,
                input_ids=student_full_ids,
                attention_mask=torch.ones_like(student_full_ids),
                use_cache=False,
            )
            start = prompt_ids.shape[1] - 1
            length = response_ids.shape[1]
            student_hidden = student_hidden_all[:, start : start + length]
            student_logits = project_logits(model, student_hidden)
            with torch.no_grad(), base_policy_context(model):
                teacher_hidden_all, _ = backbone_forward(
                    model,
                    input_ids=teacher_full_ids,
                    attention_mask=torch.ones_like(teacher_full_ids),
                    use_cache=False,
                )
                teacher_hidden = teacher_hidden_all[:, start : start + length]
                teacher_base_logits = project_logits(model, teacher_hidden)
                teacher_logits = teacher_adapter.apply_to_logits(
                    teacher_base_logits, teacher_hidden
                )
                teacher_confidence = float(
                    F.softmax(teacher_logits.float(), dim=-1).amax(dim=-1).mean().item()
                )
                mean_ridge_logit_shift = float(
                    (teacher_logits.float() - teacher_base_logits.float())
                    .norm(dim=-1)
                    .mean()
                    .item()
                )
            loss = same_prefix_distillation_loss(
                student_logits,
                teacher_logits,
                top_k=args.distill_top_k,
                temperature=args.distill_temperature,
                token_clip=args.distill_token_clip,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            distillation_losses.append(float(loss.detach().item()))

            context_sources = _teacher_context_sources(proposal, on_policy=True)
            audit.record_teacher_context(context_sources, causal=True)
            audit.record_same_prefix(
                student_full_ids, teacher_full_ids, positions=length, on_policy=True
            )
            query_audit.record_teacher_context(context_sources, causal=True)
            query_audit.record_same_prefix(
                student_full_ids, teacher_full_ids, positions=length, on_policy=True
            )
            source_audit = audits_by_source[record["source"]]
            source_audit.record_teacher_context(context_sources, causal=True)
            source_audit.record_same_prefix(
                student_full_ids, teacher_full_ids, positions=length, on_policy=True
            )
            student_context_hash = _token_ids_sha256(student_full_ids)
            teacher_context_hash = _token_ids_sha256(teacher_full_ids)
            distillation_trace.append(
                {
                    "step": distill_step,
                    "student_prefix": prefix_response,
                    "student_prefix_token_ids": response_ids.detach().cpu()[0].tolist(),
                    "student_context_sha256": student_context_hash,
                    "teacher_context_sha256": teacher_context_hash,
                    "same_prefix": student_context_hash == teacher_context_hash,
                    "prefix_tokens": length,
                    "compared_positions": length,
                    "loss": float(loss.detach().item()),
                    "teacher_mean_max_probability": teacher_confidence,
                    "mean_ridge_logit_shift_l2": mean_ridge_logit_shift,
                }
            )

        if distill_device.type == "cuda":
            torch.cuda.synchronize(distill_device)
        distillation_seconds = (
            0.0
            if specialization_no_op
            else time.perf_counter() - distillation_started
        )
        distillation_peak_memory_bytes = (
            float(torch.cuda.max_memory_allocated(distill_device))
            if distill_device.type == "cuda"
            else 0.0
        )
        distillation_peak_memory_delta_bytes = max(
            distillation_peak_memory_bytes - distillation_memory_baseline, 0.0
        )

        # The query-local ridge teacher is destroyed before the distilled
        # student is allowed to produce its final evaluation response.
        del teacher_adapter
        teacher_logits = teacher_base_logits = teacher_hidden_all = teacher_hidden = None
        gc.collect()
        if distill_device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(distill_device)
        teacher_destroyed_before_student_evaluation = True

        distilled_responses: list[str] = []
        distilled_response_tokens: list[int] = []
        distilled_truncated: list[bool] = []
        distilled_generation_seconds = 0.0
        for sample_index in range(args.eval_samples):
            generation_started = time.perf_counter()
            response, distilled_prompt_ids, response_ids = generate_response(
                model,
                tokenizer,
                record["problem"],
                adapter=None,
                max_new_tokens=args.eval_max_new_tokens,
                temperature=args.eval_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed + index * 1009 + sample_index,
            )
            distilled_generation_seconds += time.perf_counter() - generation_started
            distilled_responses.append(response)
            distilled_response_tokens.append(int(response_ids.numel()))
            distilled_truncated.append(
                _was_truncated(response_ids, args.eval_max_new_tokens, tokenizer)
            )
            max_input_tokens = max(max_input_tokens, int(distilled_prompt_ids.numel()))

        # Labels enter only now, after all adaptation and final response
        # generation have completed. They cannot influence teacher/student
        # construction, optimizer control flow, or decoding.
        base_scores = [_grade_response(response, record["answer"]) for response in base_responses]
        teacher_scores = [
            _grade_response(response, record["answer"]) for response in teacher_responses
        ]
        distilled_scores = [
            _grade_response(response, record["answer"]) for response in distilled_responses
        ]
        target_completion = _target_completion(record, args.target_score_mode)
        with base_policy_context(model):
            base_nll = score_plain_completion(
                model, tokenizer, record["problem"], target_completion
            )
        distilled_nll = score_plain_completion(
            model, tokenizer, record["problem"], target_completion
        )
        student_update_frobenius_norm = _trainable_state_delta_norm(
            model, initial_student_state
        )
        del optimizer
        _restore_trainable_state(model, initial_student_state)
        student_reset_verified = _trainable_state_matches(model, initial_student_state)
        specialization_peak_memory_bytes = float(
            specialization_metrics.pop("peak_memory_bytes", 0.0)
        )
        specialization_metrics["specialization_peak_memory_bytes"] = (
            specialization_peak_memory_bytes
        )
        specialization_metrics["distillation_peak_memory_bytes"] = (
            distillation_peak_memory_bytes
        )
        specialization_metrics["distillation_memory_baseline_bytes"] = (
            distillation_memory_baseline
        )
        specialization_metrics["distillation_peak_memory_delta_bytes"] = (
            distillation_peak_memory_delta_bytes
        )
        specialization_metrics["peak_memory_bytes"] = max(
            specialization_peak_memory_bytes, distillation_peak_memory_bytes
        )

        row = {
            "task": "task2_clean_distillation",
            "query_id": record["query_id"],
            "problem": record["problem"],
            "problem_sha256": record["problem_sha256"],
            "proposal_training_sha256": proposal["proposal_training_sha256"],
            "specialization_status": proposal["specialization_status"],
            "specialization_failure_reason": proposal[
                "specialization_failure_reason"
            ],
            "specialization_no_op": proposal["specialization_no_op"],
            "reference_answer": record["answer"],
            "source": record["source"],
            "model": args.model,
            "model_revision": args.runtime_metadata.get("resolved_model_revision", ""),
            "runtime": args.runtime_metadata,
            **_row_config_fields(args),
            "base_correct": sum(base_scores) / len(base_scores),
            "teacher_correct": sum(teacher_scores) / len(teacher_scores),
            "distilled_correct": sum(distilled_scores) / len(distilled_scores),
            "base_pass_at_n": float(any(base_scores)),
            "teacher_pass_at_n": float(any(teacher_scores)),
            "distilled_pass_at_n": float(any(distilled_scores)),
            "base_majority_at_n": _majority_grade(base_responses, record["answer"]),
            "teacher_majority_at_n": _majority_grade(teacher_responses, record["answer"]),
            "distilled_majority_at_n": _majority_grade(
                distilled_responses, record["answer"]
            ),
            "base_target_answer_nll": base_nll,
            "distilled_target_answer_nll": distilled_nll,
            "distilled_target_answer_nll_gain": base_nll - distilled_nll,
            "distillation_steps_completed": len(distillation_losses),
            "distillation_losses": distillation_losses,
            "mean_distillation_loss": sum(distillation_losses)
            / max(len(distillation_losses), 1),
            "distillation_trace": distillation_trace,
            "distillation_config": {
                "steps": args.distillation_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "top_k": args.distill_top_k,
                "temperature": args.distill_temperature,
                "token_clip": args.distill_token_clip,
                "prefix_max_new_tokens": args.train_max_new_tokens,
                "prefix_temperature": args.train_temperature,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
            },
            "base_responses": base_responses,
            "teacher_responses": teacher_responses,
            "distilled_responses": distilled_responses,
            "base_parsed_answers": _parsed_answers(base_responses),
            "teacher_parsed_answers": _parsed_answers(teacher_responses),
            "distilled_parsed_answers": _parsed_answers(distilled_responses),
            "base_generated_tokens": sum(base_response_tokens),
            "teacher_generated_tokens": sum(teacher_response_tokens),
            "distilled_generated_tokens": sum(distilled_response_tokens),
            "base_generation_seconds": base_generation_seconds,
            "teacher_generation_seconds": teacher_generation_seconds,
            "distilled_generation_seconds": distilled_generation_seconds,
            "base_truncated": bool(any(base_truncated)),
            "teacher_truncated": bool(any(teacher_truncated)),
            "distilled_truncated": bool(any(distilled_truncated)),
            "base_truncated_count": sum(base_truncated),
            "teacher_truncated_count": sum(teacher_truncated),
            "distilled_truncated_count": sum(distilled_truncated),
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": args.eval_max_new_tokens,
            "distillation_rollout_tokens": distillation_rollout_tokens,
            "distillation_seconds": distillation_seconds,
            "teacher_destroyed_before_student_evaluation": (
                teacher_destroyed_before_student_evaluation
            ),
            "student_update_frobenius_norm": student_update_frobenius_norm,
            "student_reset_verified": student_reset_verified,
            "support_generated_tokens": float(
                proposal.get("cost_audit", {}).get("total_completion_tokens", 0.0)
            ),
            "support_generation_seconds": float(
                proposal.get("cost_audit", {}).get("total_generation_seconds", 0.0)
            ),
            "proposal_end_to_end_seconds": float(
                proposal.get("cost_audit", {}).get("end_to_end_seconds", 0.0)
            ),
            **specialization_metrics,
            **_row_audit_fields(query_audit),
        }
        row["total_adaptation_seconds"] = (
            row["proposal_end_to_end_seconds"]
            + row["specialization_seconds"]
            + row["distillation_seconds"]
        )
        if not student_reset_verified:
            raise RuntimeError(f"Query-local student reset failed for {record['query_id']}")
        _validate_completed_row_protocol(
            row,
            query_audit,
            proposal,
            args,
            task="task2",
            context=f"New Task 2 row {record['query_id']}",
        )
        rows.append(row)
        _atomic_write_jsonl(detail_path, rows)

    summary = aggregate_teacher_metrics(rows, audit)
    summary.update(
        {
            "accuracy/distilled_student": sum(float(row["distilled_correct"]) for row in rows)
            / max(len(rows), 1),
            "accuracy/distilled_student_pass_at_n": sum(
                float(row["distilled_pass_at_n"]) for row in rows
            )
            / max(len(rows), 1),
            "distillation/persistent_student_accuracy_gain": sum(
                float(row["distilled_correct"]) - float(row["base_correct"]) for row in rows
            )
            / max(len(rows), 1),
            "distillation/accuracy_teacher_gain_retention": (
                _accuracy_teacher_gain_retention(rows)
            ),
            "speed/mean_distillation_seconds": sum(
                float(row["distillation_seconds"]) for row in rows
            )
            / max(len(rows), 1),
            "speed/mean_total_adaptation_seconds": sum(
                float(row["total_adaptation_seconds"]) for row in rows
            )
            / max(len(rows), 1),
        }
    )
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    source_summaries = {}
    for source, source_rows in by_source.items():
        source_summary = aggregate_teacher_metrics(source_rows, audits_by_source[source])
        source_summary["accuracy/distilled_student"] = sum(
            float(row["distilled_correct"]) for row in source_rows
        ) / len(source_rows)
        source_summary["distillation/accuracy_teacher_gain_retention"] = (
            _accuracy_teacher_gain_retention(source_rows)
        )
        source_summaries[source] = source_summary
    output_summary = {
        "task": "task2_clean_distillation",
        "protocol": "per_query_reset",
        "overall": summary,
        "by_source": source_summaries,
    }
    with (output_dir / "metrics_task2_clean_distillation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(output_summary, handle, ensure_ascii=False, indent=2)
    _restore_trainable_state(model, initial_student_state)
    return rows, output_summary, audit


def train(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    proposals: dict[str, dict[str, Any]],
    args,
) -> tuple[dict[str, float], HindsightAudit]:
    proposals_by_hash = _index_proposals_by_hash(proposals)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters; use LoRA or --full-finetune with a trainable model")
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    audit = HindsightAudit()
    losses: list[float] = []
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        order = list(range(len(records)))
        random.Random(args.seed + epoch).shuffle(order)
        for local_index, record_index in enumerate(tqdm(order, desc=f"train epoch {epoch + 1}")):
            record = records[record_index]
            proposal = _proposal_for(record, proposals, proposals_by_hash)
            # The fit API receives candidates only. target answer and solution
            # remain in record and are never passed into teacher construction.
            adapter, specialization_metrics = _fit_current_adapter(model, tokenizer, proposal, args)
            response, prompt_ids, response_ids = generate_response(
                model,
                tokenizer,
                record["problem"],
                adapter=None,
                max_new_tokens=args.train_max_new_tokens,
                temperature=args.train_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed + epoch * 100_003 + record_index,
            )
            if response_ids.numel() == 0:
                continue

            model.train()
            full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            all_hidden, _ = backbone_forward(
                model,
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                use_cache=False,
            )
            start = prompt_ids.shape[1] - 1
            length = response_ids.shape[1]
            hidden = all_hidden[:, start : start + length]
            student_logits = project_logits(model, hidden)
            with torch.no_grad():
                teacher_logits = adapter.apply_to_logits(student_logits.detach(), hidden.detach())
            loss = same_prefix_distillation_loss(
                student_logits,
                teacher_logits,
                top_k=args.distill_top_k,
                temperature=args.distill_temperature,
                token_clip=args.distill_token_clip,
            )
            (loss / args.gradient_accumulation).backward()
            losses.append(float(loss.detach().item()))

            audit.record_teacher_context(_teacher_context_sources(proposal, on_policy=True), causal=True)
            audit.record_same_prefix(full_ids, full_ids, positions=length, on_policy=True)

            if (local_index + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.logging_steps == 0:
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch + 1,
                                "loss": losses[-1],
                                "specialization_seconds": specialization_metrics["specialization_seconds"],
                                **audit.compute(),
                            },
                            ensure_ascii=False,
                        )
                    )

        if len(order) % args.gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

    output_dir = Path(args.output_dir) / "student_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "train/steps": float(step),
        "train/mean_distillation_loss": sum(losses) / max(len(losses), 1),
        **audit.compute(),
    }
    with (Path(args.output_dir) / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics, audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "task1",
            "task2",
            "support_icl",
            "support_lora",
            "head_sgd",
            "train",
            "eval",
            "train-eval",
        ),
        default="task2",
        help=(
            "task1=CSD-T; task2=per-query CSD-SD; support_lora/head_sgd are "
            "query-local iterative baselines; train modes are streaming/global ablations"
        ),
    )
    parser.add_argument("--train-data", help="Training JSONL/JSON/parquet; must match proposal query ids")
    parser.add_argument("--eval-data", help="AIME/AMC JSONL/JSON/parquet or verl validation parquet")
    parser.add_argument(
        "--proposals",
        required=True,
        action="append",
        help="Output of 01_propose.py; repeat for train and eval proposal files",
    )
    parser.add_argument("--adapter-dir", help="Optional output of 02_specialize.py; eval mode only")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", help="Pinned Hugging Face model revision")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--full-finetune", action="store_true", help="Default training uses LoRA")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--distillation-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--baseline-steps", type=int, default=3)
    parser.add_argument("--baseline-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--distill-top-k", type=int, default=64)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument(
        "--distill-token-clip",
        type=float,
        default=0.0,
        help="Hard cap for per-token KL; 0 disables it (gradient-norm clipping remains active)",
    )
    parser.add_argument("--train-max-new-tokens", type=int, default=512)
    parser.add_argument("--train-temperature", type=float, default=0.8)
    parser.add_argument("--eval-max-new-tokens", type=int, default=8192)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-samples", type=int, default=1, help="mean@N; pass@N is also logged")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--ridge-lambda", type=float, default=0.1)
    parser.add_argument("--residual-step-size", type=float, default=0.8)
    parser.add_argument("--max-tokens-per-candidate", type=int, default=64)
    parser.add_argument("--max-support-tokens", type=int, default=256)
    parser.add_argument(
        "--num-specialization-candidates",
        type=int,
        help="Use the first m verified candidates; intended for support-count sensitivity only",
    )
    parser.add_argument("--hard-negatives", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--target-score-mode", choices=("answer", "solution_answer"), default="answer")
    parser.add_argument("--privileged-control", action="store_true")
    parser.add_argument(
        "--hindsight-prefix-tokens",
        type=int,
        default=32,
        help="Fixed-prefix positions used by the correct-vs-wrong hint JSD audit",
    )
    parser.add_argument(
        "--allow-hindsight-exposure",
        action="store_true",
        help="Permit proposals whose firewall audit reports target answer/solution access (ablation only)",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an exact, validated per-query Task 1/Task 2 JSONL prefix",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mode in {"train", "train-eval"} and not args.train_data:
        raise ValueError("--train-data is required for train modes")
    if args.mode in {
        "task1",
        "task2",
        "support_icl",
        "support_lora",
        "head_sgd",
        "eval",
        "train-eval",
    } and not args.eval_data:
        raise ValueError("--eval-data is required for eval modes")
    if args.adapter_dir and args.mode not in {"task1", "eval"}:
        raise ValueError(
            "Cached adapters are checkpoint-specific; --adapter-dir is supported only in task1/eval"
        )
    if args.mode == "task2" and args.full_finetune:
        raise ValueError("Paper Task 2 uses a query-local LoRA adapter; remove --full-finetune")
    if args.resume and args.mode not in {"task1", "task2"}:
        raise ValueError("--resume is supported only for per-query Task 1 and Task 2")
    if args.eval_samples < 1 or args.eval_max_new_tokens < 1:
        raise ValueError("eval_samples and eval_max_new_tokens must be positive")
    if args.distill_temperature <= 0 or args.distill_top_k <= 0:
        raise ValueError("distill_temperature and distill_top_k must be positive")
    if args.distillation_steps < 0:
        raise ValueError("distillation_steps must be non-negative")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")

    proposals: dict[str, dict[str, Any]] = {}
    for proposal_path in args.proposals:
        proposals.update(load_proposal_map(proposal_path))
    train_records = (
        load_query_records(args.train_data, include_targets=True, max_samples=args.max_train_samples)
        if args.train_data
        else []
    )
    eval_records = (
        load_query_records(args.eval_data, include_targets=True, max_samples=args.max_eval_samples)
        if args.eval_data
        else []
    )
    eval_records = [
        record
        for global_index, record in enumerate(eval_records)
        if global_index % args.num_shards == args.shard_index
    ]
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        use_lora=not args.full_finetune
        and args.mode in {"task2", "support_lora", "train", "train-eval"},
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        training=args.mode in {"task2", "support_lora", "head_sgd", "train", "train-eval"},
        revision=args.revision,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.runtime_metadata = collect_runtime_metadata(
        model, model_path=args.model, revision=args.revision or ""
    )
    run_configuration = {
        key: value
        for key, value in vars(args).items()
        if key != "runtime_metadata"
    }
    with (Path(args.output_dir) / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"arguments": run_configuration, "runtime": args.runtime_metadata},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    adapter_cache = _load_adapter_cache(
        args.adapter_dir,
        args.model,
        str(args.runtime_metadata.get("resolved_model_revision", args.revision or "")),
    )

    combined_summary: dict[str, Any] = {}
    if args.mode == "task1":
        _, task1_summary, _ = evaluate(
            model,
            tokenizer,
            eval_records,
            proposals,
            args,
            stage="task1_fast_teacher",
            adapter_cache=adapter_cache,
        )
        combined_summary["task1_fast_teacher"] = task1_summary

    if args.mode == "task2":
        _, task2_summary, _ = per_query_distill_evaluate(
            model,
            tokenizer,
            eval_records,
            proposals,
            args,
        )
        combined_summary["task2_clean_distillation"] = task2_summary

    if args.mode == "support_icl":
        _, icl_summary = support_icl_evaluate(
            model,
            tokenizer,
            eval_records,
            proposals,
            args,
        )
        combined_summary["support_icl"] = icl_summary

    if args.mode in {"support_lora", "head_sgd"}:
        _, baseline_summary = per_query_support_sft_evaluate(
            model,
            tokenizer,
            eval_records,
            proposals,
            args,
            method=args.mode,
        )
        combined_summary[args.mode] = baseline_summary

    if args.mode == "train-eval":
        _, pre_summary, _ = evaluate(
            model,
            tokenizer,
            eval_records,
            proposals,
            args,
            stage="before_distillation",
            adapter_cache={},
        )
        combined_summary["before_distillation"] = pre_summary

    if args.mode in {"train", "train-eval"}:
        train_metrics, _ = train(model, tokenizer, train_records, proposals, args)
        combined_summary["train"] = train_metrics

    if args.mode in {"eval", "train-eval"}:
        stage = "after_distillation" if args.mode == "train-eval" else "evaluation"
        _, eval_summary, _ = evaluate(
            model,
            tokenizer,
            eval_records,
            proposals,
            args,
            stage=stage,
            adapter_cache=adapter_cache,
        )
        combined_summary[stage] = eval_summary
        if args.mode == "train-eval":
            pre = combined_summary["before_distillation"]
            post = combined_summary["after_distillation"]
            gains: dict[str, Any] = {
                "overall_mean_at_n_gain": post["overall"].get("accuracy/base", 0.0)
                - pre["overall"].get("accuracy/base", 0.0),
                "by_source": {},
            }
            for source in sorted(set(pre.get("by_source", {})) & set(post.get("by_source", {}))):
                gains["by_source"][source] = (
                    post["by_source"][source].get("accuracy/base", 0.0)
                    - pre["by_source"][source].get("accuracy/base", 0.0)
                )
            combined_summary["persistent_student_gain"] = gains

    with (Path(args.output_dir) / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(combined_summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
