#!/usr/bin/env python3
"""Run label-blind, query-local CSD-SD or pre-decision Privileged-SD."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from src.clean_self_distill.heldout import (
    EVAL_SAMPLE_COUNT,
    EVAL_TEMPERATURE,
    EVAL_TOP_K,
    EVAL_TOP_P,
    _atomic_write_jsonl,
    expected_prediction_keys,
    generation_config_sha256,
    load_query_only_manifest,
    load_resumable_predictions,
    paired_sample_seed,
    query_manifest_sha256,
)
from src.clean_self_distill.persistent import (
    PersistentConfig,
    capture_trainable_state,
    load_persistent_inputs,
    restore_trainable_state,
    train_one_episode,
)
from src.clean_self_distill.runtime import collect_runtime_metadata, load_hf_model
from src.clean_self_distill.ridge import FRONTIER_TARGET_MARGIN, problem_prompt
from src.clean_self_distill.train_eval import generate_response


def _config(args: argparse.Namespace, *, episodes: int) -> PersistentConfig:
    branch = "clean" if args.method == "csd_sd" else "privileged"
    return PersistentConfig(
        branch=branch,
        variant=args.support_variant,
        model=args.model,
        model_id=args.model_id,
        revision=args.revision,
        episodes=episodes,
        scientific_checkpoints=(0, episodes),
        rolling_checkpoint_interval=episodes,
        max_sequence_tokens=args.max_sequence_tokens,
        max_rollout_tokens=args.max_sequence_tokens,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        train_temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        top_k=EVAL_TOP_K,
        distill_top_k=args.distill_top_k,
        distill_temperature=args.distill_temperature,
        ridge_lambda=args.ridge_lambda,
        residual_step_size=args.residual_step_size,
        max_tokens_per_candidate=args.max_tokens_per_candidate,
        max_support_tokens=args.max_support_tokens,
        hard_negatives=args.hard_negatives,
        ridge_max_length=args.max_sequence_tokens,
        reasoning_token_weight=args.reasoning_token_weight,
        answer_token_weight=args.answer_token_weight,
        frontier_positive_weight=args.frontier_positive_weight,
        frontier_negative_weight=args.frontier_negative_weight,
        frontier_max_tokens=args.frontier_max_tokens,
        frontier_negative_probability_floor=args.frontier_negative_probability_floor,
        frontier_target_margin=args.frontier_target_margin,
        max_update_norm=args.max_update_norm,
    )


def run(args: argparse.Namespace) -> None:
    if args.method == "csd_sd" and args.support_variant != "correct_wrong_signed":
        raise ValueError("Formal CSD-SD must use correct_wrong_signed support")
    clean_queries = load_query_only_manifest(args.queries)
    queries, proposals, input_hashes = load_persistent_inputs(
        args.queries, args.proposals, episodes=len(clean_queries)
    )
    config = _config(args, episodes=len(queries))
    config.validate()
    query_digest = query_manifest_sha256(queries)
    generation_digest = generation_config_sha256(
        {
            "schema_version": "clean-self-distill-short-generation-config-v1",
            "method": args.method,
            "model_id": args.model_id,
            "revision": args.revision,
            "query_manifest_sha256": query_digest,
            "proposal_manifest_sha256": input_hashes["proposal_manifest_sha256"],
            "persistent_episode_config": config.identity_payload(),
            "evaluation": {
                "sample_count": EVAL_SAMPLE_COUNT,
                "temperature": EVAL_TEMPERATURE,
                "top_p": EVAL_TOP_P,
                "top_k": EVAL_TOP_K,
                "max_new_tokens": args.max_new_tokens,
                "max_prompt_tokens": args.max_prompt_tokens,
                "context_window": args.context_window,
                "seed": args.seed,
            },
            "shard": {"count": args.num_shards, "index": args.shard_index},
        }
    )
    expected = expected_prediction_keys(
        queries, num_shards=args.num_shards, shard_index=args.shard_index
    )
    rows = load_resumable_predictions(
        args.output,
        expected,
        method=args.method,
        checkpoint_episode=0,
        checkpoint_sha256="base",
        generation_config_sha256=generation_digest,
    )
    completed_keys = {
        (str(row["query_id"]), int(row["sample_index"])) for row in rows
    }

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    local = Path(args.model)
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        training=True,
        revision=None if local.exists() else args.revision,
    )
    runtime = collect_runtime_metadata(
        model, model_path=args.model_id, revision=args.revision
    )
    initial_state = capture_trainable_state(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for global_index, query in enumerate(queries):
        if global_index % args.num_shards != args.shard_index:
            continue
        query_keys = {
            (query["query_id"], sample_index)
            for sample_index in range(EVAL_SAMPLE_COUNT)
        }
        if query_keys <= completed_keys:
            continue
        prompt = problem_prompt(tokenizer, query["problem"])
        eval_prompt_tokens = int(
            tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
                "input_ids"
            ].numel()
        )
        if eval_prompt_tokens > args.max_prompt_tokens:
            raise ValueError(
                f"{query['query_id']} prompt has {eval_prompt_tokens}>{args.max_prompt_tokens} tokens"
            )
        if eval_prompt_tokens + args.max_new_tokens > args.context_window:
            raise ValueError("Evaluation budget does not fit the context window")
        restore_trainable_state(model, initial_state)
        optimizer = torch.optim.AdamW(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
        trace = train_one_episode(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            query=query,
            proposal=proposals[query["query_id"]],
            stream_index=global_index,
            config=config,
            run_identity_sha256=(
                f"query-local:{args.method}:{input_hashes['query_manifest_sha256']}"
            ),
        )
        measured_proposal_seconds = (
            float(
                proposals[query["query_id"]]
                .get("cost_audit", {})
                .get("end_to_end_seconds", 0.0)
            )
            if args.method == "csd_sd"
            else 0.0
        )
        measured_adaptation_seconds = measured_proposal_seconds + float(
            trace["episode_seconds"]
        )
        previous_query_rows = [
            row for row in rows if row.get("query_id") == query["query_id"]
        ]
        if previous_query_rows:
            first = previous_query_rows[0]
            query_training_audit = first["training_audit"]
            query_adaptation_seconds = float(first["adaptation_seconds"])
            query_proposal_seconds = float(first["proposal_end_to_end_seconds"])
            query_specialization_metrics = first["specialization_metrics"]
            query_distillation_trace = first["distillation_trace"]
        else:
            query_training_audit = trace["audit"]
            query_adaptation_seconds = measured_adaptation_seconds
            query_proposal_seconds = measured_proposal_seconds
            query_specialization_metrics = trace["ridge_metrics"]
            query_distillation_trace = trace
        for sample_index in range(EVAL_SAMPLE_COUNT):
            key = (query["query_id"], sample_index)
            if key in completed_keys:
                continue
            seed = paired_sample_seed(args.seed, global_index, sample_index)
            response, prompt_ids, response_ids = generate_response(
                model,
                tokenizer,
                query["problem"],
                adapter=None,
                max_new_tokens=args.max_new_tokens,
                temperature=EVAL_TEMPERATURE,
                top_p=EVAL_TOP_P,
                top_k=EVAL_TOP_K,
                seed=seed,
            )
            prompt_tokens = int(prompt_ids.numel())
            generated_tokens = int(response_ids.numel())
            if prompt_tokens != eval_prompt_tokens:
                raise ValueError("Evaluation prompt tokenization changed within one query")
            ended_by_eos = bool(
                generated_tokens
                and tokenizer.eos_token_id is not None
                and int(response_ids[0, -1].item()) == int(tokenizer.eos_token_id)
            )
            rows.append(
                {
                    "schema_version": "clean-self-distill-heldout-prediction-v1",
                    "method": args.method,
                    "checkpoint_episode": 0,
                    "checkpoint_sha256": "base",
                    "query_manifest_sha256": query_digest,
                    "generation_config_sha256": generation_digest,
                    "query_id": query["query_id"],
                    "problem_sha256": query["problem_sha256"],
                    "source": query["source"],
                    "sample_index": sample_index,
                    "seed": seed,
                    "temperature": EVAL_TEMPERATURE,
                    "top_p": EVAL_TOP_P,
                    "top_k": EVAL_TOP_K,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "truncated": (
                        generated_tokens >= args.max_new_tokens and not ended_by_eos
                    ),
                    "response": response,
                    "training_audit": query_training_audit,
                    "adaptation_seconds": query_adaptation_seconds,
                    "proposal_end_to_end_seconds": query_proposal_seconds,
                    "specialization_metrics": query_specialization_metrics,
                    "distillation_trace": query_distillation_trace,
                    "runtime": runtime,
                }
            )
            completed_keys.add(key)
            _atomic_write_jsonl(Path(args.output), rows)
        restore_trainable_state(model, initial_state)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--method", choices=("csd_sd", "privileged_sd"), required=True)
    value.add_argument("--queries", required=True)
    value.add_argument("--proposals", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--model-id", default="Qwen/Qwen3-8B")
    value.add_argument("--revision", required=True)
    value.add_argument("--dtype", default="bfloat16")
    value.add_argument("--device-map", default="auto")
    value.add_argument("--attn-implementation")
    value.add_argument("--max-sequence-tokens", type=int, default=16384)
    value.add_argument("--max-new-tokens", type=int, default=32768)
    value.add_argument("--max-prompt-tokens", type=int, default=8192)
    value.add_argument("--context-window", type=int, default=40960)
    value.add_argument("--num-shards", type=int, default=1)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--learning-rate", type=float, default=2e-5)
    value.add_argument("--weight-decay", type=float, default=0.0)
    value.add_argument("--lora-rank", type=int, default=8)
    value.add_argument("--lora-alpha", type=int, default=16)
    value.add_argument("--distill-top-k", type=int, default=64)
    value.add_argument("--distill-temperature", type=float, default=1.0)
    value.add_argument(
        "--support-variant",
        choices=("correct_only", "correct_wrong_signed"),
        default="correct_wrong_signed",
    )
    value.add_argument("--ridge-lambda", type=float, default=0.1)
    value.add_argument("--residual-step-size", type=float, default=0.8)
    value.add_argument("--max-tokens-per-candidate", type=int, default=96)
    value.add_argument("--max-support-tokens", type=int, default=768)
    value.add_argument("--hard-negatives", type=int, default=8)
    value.add_argument("--reasoning-token-weight", type=float, default=0.25)
    value.add_argument("--answer-token-weight", type=float, default=1.0)
    value.add_argument("--frontier-positive-weight", type=float, default=8.0)
    value.add_argument("--frontier-negative-weight", type=float, default=8.0)
    value.add_argument("--frontier-max-tokens", type=int, default=24)
    value.add_argument(
        "--frontier-negative-probability-floor", type=float, default=0.25
    )
    value.add_argument(
        "--frontier-target-margin", type=float, default=FRONTIER_TARGET_MARGIN
    )
    value.add_argument("--max-update-norm", type=float, default=2.0)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards>0 and a valid shard_index")
    run(args)


if __name__ == "__main__":
    main()
