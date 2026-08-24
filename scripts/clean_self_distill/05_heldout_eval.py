#!/usr/bin/env python3
"""Generate label-blind held-out responses or score them offline."""

from __future__ import annotations

import argparse
import json
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
    expected_branch: str | None = None,
    expected_method_id: str | None = None,
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
    expected = {
        "lgsd": (
            "clean",
            "lgsd:geometric_kl_ball_projection:forward_kl_v1",
        ),
        "trsd": ("clean", "trsd:exponential_teacher_projection"),
        "opsd": (
            "privileged",
            "opsd:raw_privileged_teacher:forward_kl_v1",
        ),
        "privileged_sd": ("privileged", "privileged:predecision_method"),
        "demopsd": ("demopsd", "baseline:demopsd:exact_full_vocab_v1"),
        "grpo": ("grpo", "baseline:outcome_grpo:deepseekmath_v1"),
        "srpo": ("srpo", "baseline:srpo:sample_routed_dw_sdpo_v1"),
    }.get(method)
    if expected is not None:
        mapped_branch, mapped_method_id = expected
        expected_branch = expected_branch or mapped_branch
        expected_method_id = expected_method_id or mapped_method_id
    if expected_branch is None:
        raise HeldoutProtocolError(
            "Custom adapter methods require --expected-branch"
        )
    if (
        value.get("schema_version")
        != "clean-self-distill-persistent-checkpoint-v1"
        or value.get("branch") != expected_branch
        or (
            expected_method_id is not None
            and value.get("method_id") != expected_method_id
        )
        or value.get("checkpoint_episode") != checkpoint_episode
        or value.get("completed_episodes") != checkpoint_episode
        or value.get("model_id") != model_id
        or value.get("model_revision") != revision
    ):
        raise HeldoutProtocolError(
            f"{manifest} is not the requested {method} episode-{checkpoint_episode} checkpoint"
        )
    audit = value.get("cumulative_audit")
    if not isinstance(audit, dict):
        raise HeldoutProtocolError(f"{manifest} lacks cumulative_audit")
    result = dict(audit)
    # TRSD uses the same on-policy response prefix, but the raw surrogate is
    # evaluated with a teacher-only pre-decision prompt. Strict context parity
    # is therefore zero; old checkpoints recorded this field incorrectly.
    if (
        expected_branch == "clean"
        and value.get("method_id") == "trsd:exponential_teacher_projection"
    ):
        result["exact_context_positions"] = 0
        result["context_parity"] = 0.0
    if method in {"lgsd", "opsd"} and value.get(
        "distillation_kl_direction"
    ) != "projected_teacher_to_student_forward_kl_v1":
        raise HeldoutProtocolError(
            f"{manifest} is not a forward-KL {method.upper()} checkpoint"
        )
    return result


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
        attn_implementation=args.attn_implementation,
    )
    checkpoint = Path(args.adapter) if args.adapter else None
    if checkpoint is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
        model.eval()
    return model, tokenizer, checkpoint


def generate(args: argparse.Namespace) -> None:
    if args.engine == "vllm":
        generate_vllm(args)
        return

    # Keep all GPU/model dependencies behind the generate boundary so the
    # label-only offline scorer remains usable on a small CPU node without a
    # PyTorch/PEFT environment.
    import torch

    from src.clean_self_distill.generation import (
        EVALUATION_PROMPT_VERSION,
        evaluation_problem_prompt,
        generate_response,
    )
    from src.clean_self_distill.runtime import collect_runtime_metadata

    queries = load_query_only_manifest(args.queries)
    query_digest = query_manifest_sha256(queries)
    expected = expected_prediction_keys(
        queries,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        sample_count=args.sample_count,
    )
    checkpoint = Path(args.adapter) if args.adapter else None
    adapter_method = args.method != "base"
    if adapter_method != (checkpoint is not None):
        raise HeldoutProtocolError(
            "Every non-base method requires exactly one --adapter"
        )
    checkpoint_hash = tree_sha256(checkpoint) if checkpoint else "base"
    generation_config = {
            "schema_version": "trsd-heldout-generation-config-v2",
            "evaluation_prompt_version": EVALUATION_PROMPT_VERSION,
            "method": args.method,
            "checkpoint_episode": args.checkpoint_episode,
            "checkpoint_sha256": checkpoint_hash,
            "query_manifest_sha256": query_digest,
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
            "shard": {"count": args.num_shards, "index": args.shard_index},
        }
    generation_digest = generation_config_sha256(generation_config)
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
        expected_branch=args.expected_branch,
        expected_method_id=args.expected_method_id,
    )
    if args.method == "base" and args.checkpoint_episode != 0:
        raise HeldoutProtocolError("Base must be evaluated at checkpoint episode 0")
    runtime = collect_runtime_metadata(
        model, model_path=args.model_id or args.model, revision=args.revision or ""
    )
    key_to_global = {
        (query["query_id"], sample_index): (global_index, query)
        for global_index, query in enumerate(queries)
        if global_index % args.num_shards == args.shard_index
        for sample_index in range(args.sample_count)
    }
    active_memory_baseline = 0
    for query_id, sample_index in expected[len(rows) :]:
        global_index, query = key_to_global[(query_id, sample_index)]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            active_memory_baseline = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        else:
            active_memory_baseline = 0
        prompt = evaluation_problem_prompt(
            tokenizer,
            query["problem"],
            max_new_tokens=args.max_new_tokens,
        )
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_started = time.perf_counter()
        response, _, response_ids = generate_response(
            model,
            tokenizer,
            query["problem"],
            max_new_tokens=args.max_new_tokens,
            temperature=EVAL_TEMPERATURE,
            top_p=EVAL_TOP_P,
            top_k=EVAL_TOP_K,
            seed=seed,
            prompt_override=prompt,
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
        evaluation_seconds = generation_seconds
        resource_usage = {
            "schema_version": "trsd-query-resource-v1",
            "generation_seconds": generation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "method_end_to_end_seconds": evaluation_seconds,
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
        behavioral_diagnostics = {
            "fabricated_reference_hallucination": bool(_REFERENCE_RE.search(response)),
            "hedging_token_count": len(_HEDGE_RE.findall(response)),
            "response_tokens": generated,
            "mean_entropy": None,
            "truncated": truncated,
        }
        training_audit = checkpoint_audit
        rows.append(
            {
                "schema_version": "clean-self-distill-heldout-prediction-v1",
                "method": args.method,
                "checkpoint_episode": args.checkpoint_episode,
                "checkpoint_sha256": checkpoint_hash,
                "query_manifest_sha256": query_digest,
                "generation_config_sha256": generation_digest,
                "evaluation_prompt_version": EVALUATION_PROMPT_VERSION,
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
                "trajectory_metrics": None,
                "behavioral_diagnostics": behavioral_diagnostics,
                "resource_usage": resource_usage,
                "runtime": runtime,
            }
        )
        _atomic_write_jsonl(Path(args.output), rows)


def generate_vllm(args: argparse.Namespace) -> None:
    """Run the same held-out protocol with batched, paged-KV inference."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from src.clean_self_distill.generation import (
        EVALUATION_PROMPT_VERSION,
        evaluation_problem_prompt,
    )
    from src.clean_self_distill.runtime import collect_runtime_metadata

    queries = load_query_only_manifest(args.queries)
    query_digest = query_manifest_sha256(queries)
    expected = expected_prediction_keys(
        queries,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        sample_count=args.sample_count,
    )
    checkpoint = Path(args.adapter) if args.adapter else None
    adapter_method = args.method != "base"
    if adapter_method != (checkpoint is not None):
        raise HeldoutProtocolError(
            "Every non-base method requires exactly one --adapter"
        )
    checkpoint_hash = tree_sha256(checkpoint) if checkpoint else "base"
    generation_digest = generation_config_sha256(
        {
            "schema_version": "trsd-heldout-generation-config-v2",
            "evaluation_prompt_version": EVALUATION_PROMPT_VERSION,
            "inference_engine": "vllm-batched-v1",
            "method": args.method,
            "checkpoint_episode": args.checkpoint_episode,
            "checkpoint_sha256": checkpoint_hash,
            "query_manifest_sha256": query_digest,
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
    checkpoint_audit = _load_training_audit(
        checkpoint,
        method=args.method,
        checkpoint_episode=args.checkpoint_episode,
        model_id=args.model_id,
        revision=args.revision,
        expected_branch=args.expected_branch,
        expected_method_id=args.expected_method_id,
    )
    if args.method == "base" and args.checkpoint_episode != 0:
        raise HeldoutProtocolError("Base must be evaluated at checkpoint episode 0")
    if len(rows) == len(expected):
        print(json.dumps({"status": "complete", "rows": len(rows), "output": args.output}))
        return

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True
    )
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.context_window,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=checkpoint is not None,
        max_lora_rank=args.max_lora_rank,
    )
    lora_request = (
        None
        if checkpoint is None
        else LoRARequest(args.method, 1, str(checkpoint))
    )
    runtime = collect_runtime_metadata(
        None, model_path=args.model_id or args.model, revision=args.revision or ""
    )
    runtime.update(
        {
            "inference_engine": "vllm",
            "vllm_model_path": str(args.model),
            "vllm_batch_size": args.batch_size,
            "vllm_tensor_parallel_size": args.tensor_parallel_size,
        }
    )
    key_to_global = {
        (query["query_id"], sample_index): (global_index, query)
        for global_index, query in enumerate(queries)
        if global_index % args.num_shards == args.shard_index
        for sample_index in range(args.sample_count)
    }
    pending = expected[len(rows) :]
    for offset in range(0, len(pending), args.batch_size):
        batch_keys = pending[offset : offset + args.batch_size]
        batch_items = [key_to_global[key] for key in batch_keys]
        prompts = [
            evaluation_problem_prompt(
                tokenizer, query["problem"], max_new_tokens=args.max_new_tokens
            )
            for _, query in batch_items
        ]
        prompt_token_ids = [
            tokenizer(prompt, add_special_tokens=True)["input_ids"] for prompt in prompts
        ]
        prompt_tokens = [len(token_ids) for token_ids in prompt_token_ids]
        offenders = [
            query["query_id"]
            for (_, query), length in zip(batch_items, prompt_tokens)
            if length > args.max_prompt_tokens
            or length + args.max_new_tokens > args.context_window
        ]
        if offenders:
            raise HeldoutProtocolError(
                f"Prompts cannot receive the preregistered token opportunity: {offenders}"
            )
        sampling = [
            SamplingParams(
                temperature=EVAL_TEMPERATURE,
                top_p=EVAL_TOP_P,
                top_k=EVAL_TOP_K,
                max_tokens=args.max_new_tokens,
                seed=paired_sample_seed(
                    args.seed,
                    global_index,
                    sample_index,
                    sample_count=args.sample_count,
                ),
            )
            for (global_index, _), (_, sample_index) in zip(batch_items, batch_keys)
        ]
        started = time.perf_counter()
        generated = llm.generate(
            prompts,
            sampling,
            use_tqdm=True,
            lora_request=lora_request,
        )
        elapsed = time.perf_counter() - started
        amortized_seconds = elapsed / len(batch_items)
        for key, (global_index, query), prompt_length, result in zip(
            batch_keys, batch_items, prompt_tokens, generated
        ):
            query_id, sample_index = key
            candidate = result.outputs[0]
            response = candidate.text.strip()
            generated_count = len(candidate.token_ids)
            truncated = candidate.finish_reason == "length"
            seed = paired_sample_seed(
                args.seed,
                global_index,
                sample_index,
                sample_count=args.sample_count,
            )
            resource_usage = {
                "schema_version": "trsd-query-resource-v1",
                "generation_seconds": amortized_seconds,
                "evaluation_seconds": amortized_seconds,
                "method_end_to_end_seconds": amortized_seconds,
                "batch_generation_seconds": elapsed,
                "batch_size": len(batch_items),
                "cuda_memory_baseline_bytes": 0,
                "cuda_peak_memory_allocated_bytes": 0,
                "cuda_peak_memory_delta_bytes": 0,
                "cuda_peak_memory_reserved_bytes": 0,
                "process_peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                ),
            }
            rows.append(
                {
                    "schema_version": "clean-self-distill-heldout-prediction-v1",
                    "method": args.method,
                    "checkpoint_episode": args.checkpoint_episode,
                    "checkpoint_sha256": checkpoint_hash,
                    "query_manifest_sha256": query_digest,
                    "generation_config_sha256": generation_digest,
                    "evaluation_prompt_version": EVALUATION_PROMPT_VERSION,
                    "query_id": query_id,
                    "problem_sha256": query["problem_sha256"],
                    "source": query["source"],
                    "sample_index": sample_index,
                    "seed": seed,
                    "temperature": EVAL_TEMPERATURE,
                    "top_p": EVAL_TOP_P,
                    "top_k": EVAL_TOP_K,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_tokens": prompt_length,
                    "generated_tokens": generated_count,
                    "truncated": truncated,
                    "response": response,
                    "training_audit": checkpoint_audit,
                    "trajectory_metrics": None,
                    "behavioral_diagnostics": {
                        "fabricated_reference_hallucination": bool(
                            _REFERENCE_RE.search(response)
                        ),
                        "hedging_token_count": len(_HEDGE_RE.findall(response)),
                        "response_tokens": generated_count,
                        "mean_entropy": None,
                        "truncated": truncated,
                    },
                    "resource_usage": resource_usage,
                    "runtime": runtime,
                }
            )
        _atomic_write_jsonl(Path(args.output), rows)
        print(
            json.dumps(
                {
                    "method": args.method,
                    "shard": args.shard_index,
                    "complete": len(rows),
                    "total": len(expected),
                    "last_batch_seconds": round(elapsed, 3),
                }
            ),
            flush=True,
        )


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
    generate_parser.add_argument("--method", required=True)
    generate_parser.add_argument("--expected-branch")
    generate_parser.add_argument("--expected-method-id")
    generate_parser.add_argument("--checkpoint-episode", type=int, required=True)
    generate_parser.add_argument("--dtype", default="bfloat16")
    generate_parser.add_argument("--device-map", default="auto")
    generate_parser.add_argument("--attn-implementation", default="sdpa")
    generate_parser.add_argument("--engine", choices=("hf", "vllm"), default="hf")
    generate_parser.add_argument("--batch-size", type=int, default=64)
    generate_parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    generate_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    generate_parser.add_argument("--max-lora-rank", type=int, default=8)
    generate_parser.add_argument("--max-new-tokens", type=int, default=32768)
    generate_parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    generate_parser.add_argument("--context-window", type=int, default=40960)
    generate_parser.add_argument("--num-shards", type=int, default=1)
    generate_parser.add_argument("--shard-index", type=int, default=0)
    generate_parser.add_argument("--sample-count", type=int, default=EVAL_SAMPLE_COUNT)
    generate_parser.add_argument("--seed", type=int, default=0)
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
