#!/usr/bin/env python3
"""Generate label-blind held-out responses or score them offline."""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
import time
from pathlib import Path
from typing import Any

from src.clean_self_distill.heldout import (
    EVAL_SAMPLE_COUNT,
    EVAL_TEMPERATURE,
    EVAL_TOP_K,
    EVAL_TOP_P,
    HeldoutProtocolError,
    _atomic_write_jsonl,
    expected_prediction_keys,
    generation_config_sha256,
    load_query_only_manifest,
    load_resumable_predictions,
    load_sealed_labels,
    paired_sample_seed,
    query_manifest_sha256,
    score_prediction_rows,
    tree_sha256,
    write_scored_rows,
)
from src.clean_self_distill.io import iter_rows
from src.clean_self_distill.io import (
    validate_proposal_training_binding,
    validate_specialization_state,
)


_REFERENCE_RE = re.compile(
    r"\b(?:answer\s*key|given\s+(?:answer|solution)|reference\s+(?:answer|solution)|"
    r"according\s+to\s+the\s+(?:answer|reference))\b",
    flags=re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(?:perhaps|maybe|possibly|likely|seems?|appears?|I\s+think|not\s+sure)\b",
    flags=re.IGNORECASE,
)


def _load_training_audit(
    checkpoint: Path | None,
    *,
    method: str,
    checkpoint_episode: int,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    if checkpoint is None:
        return {
            "teacher_positions": 0,
            "hindsight_exposed_positions": 0,
            "compared_positions": 0,
            "exact_context_positions": 0,
        }
    manifest = checkpoint.parent / "checkpoint_manifest.json"
    if not manifest.exists():
        manifest = checkpoint / "checkpoint_manifest.json"
    if not manifest.exists():
        raise HeldoutProtocolError(f"Missing checkpoint manifest beside {checkpoint}")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    expected_branch = {"clean_sd": "clean", "privileged_sd": "privileged"}.get(
        method
    )
    if (
        value.get("schema_version")
        != "clean-self-distill-persistent-checkpoint-v1"
        or expected_branch is None
        or value.get("branch") != expected_branch
        or value.get("checkpoint_episode") != checkpoint_episode
        or value.get("completed_episodes") != checkpoint_episode
        or value.get("model_id") != model_id
        or value.get("model_revision") != revision
        or (
            expected_branch == "clean"
            and value.get("variant") != "correct_wrong_signed"
        )
    ):
        raise HeldoutProtocolError(
            f"{manifest} is not the requested {method} episode-{checkpoint_episode} checkpoint"
        )
    audit = value.get("cumulative_audit")
    if not isinstance(audit, dict):
        raise HeldoutProtocolError(f"{manifest} lacks cumulative_audit")
    return audit


def _load_model(args: argparse.Namespace):
    from src.clean_self_distill.runtime import load_hf_model

    local_model = Path(args.model)
    revision = None if local_model.exists() else args.revision
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        training=False,
        revision=revision,
    )
    checkpoint = Path(args.adapter) if args.adapter else None
    if checkpoint is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
        model.eval()
    return model, tokenizer, checkpoint


def _load_proposals(path: str | None, queries: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    wanted = {query["query_id"]: query for query in queries}
    proposals: dict[str, dict[str, Any]] = {}
    for raw in iter_rows(path):
        query_id = str(raw.get("query_id", ""))
        if query_id not in wanted:
            continue
        if query_id in proposals:
            raise HeldoutProtocolError(f"Duplicate proposal {query_id!r}")
        row = dict(raw)
        query = wanted[query_id]
        if (
            row.get("problem") != query["problem"]
            or row.get("problem_sha256") != query["problem_sha256"]
            or str(row.get("source", "")).casefold() != query["source"]
        ):
            raise HeldoutProtocolError(f"Proposal {query_id!r} is bound to another query")
        validate_proposal_training_binding(row, context=f"Proposal {query_id}")
        validate_specialization_state(row, context=f"Proposal {query_id}")
        firewall = row.get("firewall_audit", {})
        if not isinstance(firewall, dict) or any(
            firewall.get(key) is not False
            for key in ("target_answer_loaded", "target_solution_loaded")
        ):
            raise HeldoutProtocolError(f"Proposal {query_id!r} is not clean")
        proposals[query_id] = row
    if set(proposals) != set(wanted):
        missing = sorted(set(wanted) - set(proposals))
        raise HeldoutProtocolError(f"Missing held-out proposals: {missing[:5]}")
    return proposals


def _fit_temporary_teacher(model, tokenizer, proposal: dict[str, Any], args: argparse.Namespace):
    from src.clean_self_distill.ridge import fit_ridge_adapter
    from src.clean_self_distill.runtime import input_device

    status, reason, no_op = validate_specialization_state(
        proposal, context=f"Proposal {proposal['query_id']}"
    )
    adapter, metrics = fit_ridge_adapter(
        model,
        tokenizer,
        list(proposal.get("specialization_candidates", [])),
        ridge_lambda=args.ridge_lambda,
        residual_step_size=args.residual_step_size,
        max_tokens_per_candidate=args.max_tokens_per_candidate,
        max_support_tokens=args.max_support_tokens,
        hard_negatives=args.hard_negatives,
        max_length=args.max_sequence_tokens,
        reasoning_token_weight=args.reasoning_token_weight,
        answer_token_weight=args.answer_token_weight,
        frontier_positive_weight=args.frontier_positive_weight,
        frontier_negative_weight=args.frontier_negative_weight,
        frontier_max_tokens=args.frontier_max_tokens,
        frontier_negative_probability_floor=args.frontier_negative_probability_floor,
        frontier_target_margin=args.frontier_target_margin,
        signed_frontier=args.support_variant == "correct_wrong_signed",
        max_update_norm=args.max_update_norm,
        query_id=str(proposal["query_id"]),
        specialization_status=status,
        specialization_failure_reason=reason,
        specialization_no_op=no_op,
    )
    return adapter.to(input_device(model)), metrics


def generate(args: argparse.Namespace) -> None:
    # Keep all GPU/model dependencies behind the generate boundary so the
    # label-only offline scorer remains usable on a small CPU node without a
    # PyTorch/PEFT environment.
    import torch

    from src.clean_self_distill.persistent import (
        file_sha256,
        style_task_error_from_trace,
    )
    from src.clean_self_distill.ridge import problem_prompt
    from src.clean_self_distill.runtime import collect_runtime_metadata
    from src.clean_self_distill.train_eval import generate_response

    queries = load_query_only_manifest(args.queries)
    query_digest = query_manifest_sha256(queries)
    expected = expected_prediction_keys(
        queries,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        sample_count=args.sample_count,
    )
    checkpoint = Path(args.adapter) if args.adapter else None
    persistent_method = args.method in {"clean_sd", "privileged_sd"}
    if persistent_method != (checkpoint is not None):
        raise HeldoutProtocolError(
            "Persistent clean_sd/privileged_sd methods require exactly one --adapter"
        )
    if checkpoint is not None and args.proposals:
        raise HeldoutProtocolError(
            "A query-local CSD-T evaluation cannot also load a persistent adapter"
        )
    checkpoint_hash = tree_sha256(checkpoint) if checkpoint else "base"
    generation_digest = generation_config_sha256(
        {
            "schema_version": "clean-self-distill-heldout-generation-config-v1",
            "method": args.method,
            "checkpoint_episode": args.checkpoint_episode,
            "checkpoint_sha256": checkpoint_hash,
            "query_manifest_sha256": query_digest,
            "proposal_manifest_sha256": file_sha256(args.proposals) if args.proposals else None,
            "model_id": args.model_id,
            "revision": args.revision,
            "sampling": {
                "sample_count": args.sample_count,
                "temperature": EVAL_TEMPERATURE,
                "top_p": EVAL_TOP_P,
                "top_k": EVAL_TOP_K,
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "max_prompt_tokens": args.max_prompt_tokens,
                "context_window": args.context_window,
            },
            "ridge": {
                key: getattr(args, key)
                for key in (
                    "support_variant",
                    "ridge_lambda",
                    "residual_step_size",
                    "max_tokens_per_candidate",
                    "max_support_tokens",
                    "hard_negatives",
                    "max_sequence_tokens",
                    "reasoning_token_weight",
                    "answer_token_weight",
                    "frontier_positive_weight",
                    "frontier_negative_weight",
                    "frontier_max_tokens",
                    "frontier_negative_probability_floor",
                    "frontier_target_margin",
                    "max_update_norm",
                )
            }
            if args.proposals
            else None,
            "shard": {"count": args.num_shards, "index": args.shard_index},
        }
    )
    rows = load_resumable_predictions(
        args.output,
        expected,
        method=args.method,
        checkpoint_episode=args.checkpoint_episode,
        checkpoint_sha256=checkpoint_hash,
        generation_config_sha256=generation_digest,
    )
    model, tokenizer, checkpoint = _load_model(args)
    checkpoint_audit = _load_training_audit(
        checkpoint,
        method=args.method,
        checkpoint_episode=args.checkpoint_episode,
        model_id=args.model_id,
        revision=args.revision,
    )
    proposals = _load_proposals(args.proposals, queries)
    if args.method.startswith("csd_t") != bool(proposals):
        raise HeldoutProtocolError(
            "CSD-T methods require --proposals and non-CSD-T methods must not receive it"
        )
    if args.method == "base" and args.checkpoint_episode != 0:
        raise HeldoutProtocolError("Base must be evaluated at checkpoint episode 0")
    if args.method == "csd_t" and args.support_variant != "correct_wrong_signed":
        raise HeldoutProtocolError("csd_t must use correct_wrong_signed support")
    if (
        args.method == "csd_t_correct_only"
        and args.support_variant != "correct_only"
    ):
        raise HeldoutProtocolError("csd_t_correct_only must use correct_only support")
    if args.method.startswith("csd_t") and args.checkpoint_episode != 0:
        raise HeldoutProtocolError("Query-local CSD-T must use checkpoint episode 0")
    runtime = collect_runtime_metadata(
        model, model_path=args.model_id or args.model, revision=args.revision or ""
    )
    key_to_global = {
        (query["query_id"], sample_index): (global_index, query)
        for global_index, query in enumerate(queries)
        if global_index % args.num_shards == args.shard_index
        for sample_index in range(args.sample_count)
    }
    active_query_id: str | None = None
    active_adapter = None
    active_specialization: dict[str, Any] = {}
    active_proposal_seconds = 0.0
    active_setup_seconds = 0.0
    active_memory_baseline = 0
    for query_id, sample_index in expected[len(rows) :]:
        global_index, query = key_to_global[(query_id, sample_index)]
        if query_id != active_query_id:
            active_query_id = query_id
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                active_memory_baseline = int(torch.cuda.memory_allocated())
                torch.cuda.reset_peak_memory_stats()
            else:
                active_memory_baseline = 0
            setup_started = time.perf_counter()
            if proposals:
                active_adapter, measured_specialization = _fit_temporary_teacher(
                    model, tokenizer, proposals[query_id], args
                )
                prior = next(
                    (row for row in rows if row.get("query_id") == query_id), None
                )
                if prior is not None:
                    saved = prior.get("specialization_metrics")
                    if not isinstance(saved, dict):
                        raise HeldoutProtocolError(
                            f"Partial query {query_id} lacks specialization metrics"
                        )
                    active_specialization = saved
                    active_proposal_seconds = float(
                        prior.get("proposal_end_to_end_seconds", 0.0)
                    )
                else:
                    active_specialization = measured_specialization
                    active_proposal_seconds = float(
                        proposals[query_id].get("cost_audit", {}).get(
                            "end_to_end_seconds", 0.0
                        )
                    )
            else:
                active_adapter, active_specialization = None, {}
                active_proposal_seconds = 0.0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            active_setup_seconds = time.perf_counter() - setup_started
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
            active_memory_baseline = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            active_setup_seconds = 0.0
        prompt = problem_prompt(tokenizer, query["problem"])
        prompt_tokens = int(
            tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
                "input_ids"
            ].numel()
        )
        if prompt_tokens > args.max_prompt_tokens:
            raise HeldoutProtocolError(
                f"{query_id} prompt has {prompt_tokens}>{args.max_prompt_tokens} tokens"
            )
        if prompt_tokens + args.max_new_tokens > args.context_window:
            raise HeldoutProtocolError(
                f"{query_id} cannot receive the preregistered {args.max_new_tokens}-token opportunity"
            )
        seed = paired_sample_seed(
            args.seed,
            global_index,
            sample_index,
            sample_count=args.sample_count,
        )
        online_trace: list[dict[str, Any]] = []
        generation_kwargs: dict[str, Any] = {}
        if active_adapter is not None:
            generation_kwargs["trace_sink"] = online_trace
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_started = time.perf_counter()
        response, _, response_ids = generate_response(
            model,
            tokenizer,
            query["problem"],
            adapter=active_adapter,
            max_new_tokens=args.max_new_tokens,
            temperature=EVAL_TEMPERATURE,
            top_p=EVAL_TOP_P,
            top_k=EVAL_TOP_K,
            seed=seed,
            **generation_kwargs,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        peak_allocated = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        )
        evaluation_seconds = active_setup_seconds + generation_seconds
        resource_usage = {
            "schema_version": "clean-self-distill-query-resource-v1",
            "generation_seconds": generation_seconds,
            "query_setup_seconds": active_setup_seconds,
            "evaluation_seconds": evaluation_seconds,
            "proposal_seconds": active_proposal_seconds,
            "method_end_to_end_seconds": evaluation_seconds
            + active_proposal_seconds,
            "cuda_memory_baseline_bytes": active_memory_baseline,
            "cuda_peak_memory_allocated_bytes": peak_allocated,
            "cuda_peak_memory_delta_bytes": max(
                peak_allocated - active_memory_baseline, 0
            ),
            "cuda_peak_memory_reserved_bytes": peak_reserved,
            "process_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        }
        generated = int(response_ids.numel())
        ended_by_eos = bool(
            generated
            and tokenizer.eos_token_id is not None
            and int(response_ids[0, -1].item()) == int(tokenizer.eos_token_id)
        )
        truncated = generated >= args.max_new_tokens and not ended_by_eos
        trajectory_metrics: dict[str, Any] | None = None
        if online_trace:
            trajectory_metrics = {
                "token_count": len(online_trace),
                "student_logprob_sum": sum(
                    float(item["student_logprob"]) for item in online_trace
                ),
                "teacher_logprob_sum": sum(
                    float(item["teacher_logprob"]) for item in online_trace
                ),
                **style_task_error_from_trace(tokenizer, online_trace),
            }
        trace_entropies = [
            float(item["teacher_entropy"])
            for item in online_trace
            if item.get("teacher_entropy") is not None
            and math.isfinite(float(item["teacher_entropy"]))
        ]
        behavioral_diagnostics = {
            "fabricated_reference_hallucination": bool(_REFERENCE_RE.search(response)),
            "hedging_token_count": len(_HEDGE_RE.findall(response)),
            "response_tokens": generated,
            "mean_entropy": (
                sum(trace_entropies) / len(trace_entropies) if trace_entropies else None
            ),
            "truncated": truncated,
        }
        training_audit = (
            {
                "teacher_positions": generated,
                "hindsight_exposed_positions": 0,
                "compared_positions": generated,
                "exact_context_positions": generated,
            }
            if proposals
            else checkpoint_audit
        )
        rows.append(
            {
                "schema_version": "clean-self-distill-heldout-prediction-v1",
                "method": args.method,
                "checkpoint_episode": args.checkpoint_episode,
                "checkpoint_sha256": checkpoint_hash,
                "query_manifest_sha256": query_digest,
                "generation_config_sha256": generation_digest,
                "query_id": query_id,
                "problem_sha256": query["problem_sha256"],
                "source": query["source"],
                "sample_index": sample_index,
                "seed": seed,
                "temperature": EVAL_TEMPERATURE,
                "top_p": EVAL_TOP_P,
                "top_k": EVAL_TOP_K,
                "max_new_tokens": args.max_new_tokens,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated,
                "truncated": truncated,
                "response": response,
                "training_audit": training_audit,
                "specialization_metrics": active_specialization,
                "trajectory_metrics": trajectory_metrics,
                "behavioral_diagnostics": behavioral_diagnostics,
                "proposal_end_to_end_seconds": active_proposal_seconds,
                "resource_usage": resource_usage,
                "runtime": runtime,
            }
        )
        _atomic_write_jsonl(Path(args.output), rows)


def score(args: argparse.Namespace) -> None:
    predictions = [dict(row) for path in args.predictions for row in iter_rows(path)]
    labels = load_sealed_labels(args.labels)
    scored = score_prediction_rows(
        predictions, labels, sample_count=args.sample_count
    )
    write_scored_rows(args.output, scored)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--queries", required=True)
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--model", required=True)
    generate_parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    generate_parser.add_argument("--revision", required=True)
    generate_parser.add_argument("--adapter")
    generate_parser.add_argument("--proposals")
    generate_parser.add_argument(
        "--method",
        choices=("base", "clean_sd", "privileged_sd", "csd_t", "csd_t_correct_only"),
        required=True,
    )
    generate_parser.add_argument("--checkpoint-episode", type=int, required=True)
    generate_parser.add_argument("--dtype", default="bfloat16")
    generate_parser.add_argument("--device-map", default="auto")
    generate_parser.add_argument("--max-new-tokens", type=int, default=32768)
    generate_parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    generate_parser.add_argument("--context-window", type=int, default=40960)
    generate_parser.add_argument("--num-shards", type=int, default=1)
    generate_parser.add_argument("--shard-index", type=int, default=0)
    generate_parser.add_argument("--sample-count", type=int, default=EVAL_SAMPLE_COUNT)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument(
        "--support-variant",
        choices=("correct_only", "correct_wrong_signed"),
        default="correct_wrong_signed",
    )
    generate_parser.add_argument("--ridge-lambda", type=float, default=0.1)
    generate_parser.add_argument("--residual-step-size", type=float, default=0.8)
    generate_parser.add_argument("--max-tokens-per-candidate", type=int, default=96)
    generate_parser.add_argument("--max-support-tokens", type=int, default=768)
    generate_parser.add_argument("--hard-negatives", type=int, default=8)
    generate_parser.add_argument("--max-sequence-tokens", type=int, default=16384)
    generate_parser.add_argument("--reasoning-token-weight", type=float, default=0.25)
    generate_parser.add_argument("--answer-token-weight", type=float, default=1.0)
    generate_parser.add_argument("--frontier-positive-weight", type=float, default=8.0)
    generate_parser.add_argument("--frontier-negative-weight", type=float, default=8.0)
    generate_parser.add_argument("--frontier-max-tokens", type=int, default=24)
    generate_parser.add_argument(
        "--frontier-negative-probability-floor", type=float, default=0.25
    )
    generate_parser.add_argument(
        "--frontier-target-margin", type=float, default=1.0
    )
    generate_parser.add_argument("--max-update-norm", type=float, default=2.0)
    generate_parser.set_defaults(func=generate)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--predictions", action="append", required=True)
    score_parser.add_argument("--labels", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--sample-count", type=int, default=EVAL_SAMPLE_COUNT)
    score_parser.set_defaults(func=score)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
