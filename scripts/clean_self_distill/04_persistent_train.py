#!/usr/bin/env python3
"""Train one restartable persistent Clean/Privileged LoRA branch.

The command intentionally has no labels argument.  ``--queries`` must be the
physically target-free DeepMath stream and ``--proposals`` must be its verified,
target-disjoint corrective-support manifest.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import torch

from src.clean_self_distill.ridge import FRONTIER_TARGET_MARGIN
from src.clean_self_distill.heldout import load_query_only_manifest
from src.clean_self_distill.io import canonical_json_sha256
from src.clean_self_distill.persistent import (
    REQUEUE_EXIT_CODE,
    PersistentConfig,
    load_persistent_inputs,
    parse_scientific_checkpoints,
    run_persistent_training,
)
from src.clean_self_distill.runtime import collect_runtime_metadata, load_hf_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch",
        choices=("clean", "privileged", "proposed_privileged"),
        required=True,
    )
    parser.add_argument(
        "--variant",
        choices=("correct_only", "correct_wrong_signed"),
        required=True,
    )
    parser.add_argument("--queries", required=True)
    parser.add_argument("--proposals")
    parser.add_argument("--model", required=True, help="Pinned local model snapshot")
    parser.add_argument("--model-id", required=True, help="Canonical Hugging Face id")
    parser.add_argument("--revision", required=True, help="Pinned model revision")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument(
        "--scientific-checkpoints", default="0,250,500,750,1000"
    )
    parser.add_argument("--rolling-checkpoint-interval", type=int, default=5)
    parser.add_argument("--max-sequence-tokens", type=int, default=16_384)
    parser.add_argument(
        "--max-rollout-tokens",
        type=int,
        default=16_384,
        help="Upper bound further clipped so both branch contexts fit the total cap",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--train-temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--distill-top-k", type=int, default=64)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--distill-token-clip", type=float, default=0.0)
    parser.add_argument("--distill-token-chunk-size", type=int, default=128)

    parser.add_argument("--ridge-lambda", type=float, default=0.1)
    parser.add_argument("--residual-step-size", type=float, default=0.8)
    parser.add_argument("--max-tokens-per-candidate", type=int, default=96)
    parser.add_argument("--max-support-tokens", type=int, default=768)
    parser.add_argument("--num-specialization-candidates", type=int)
    parser.add_argument("--hard-negatives", type=int, default=8)
    parser.add_argument("--ridge-max-length", type=int, default=8_192)
    parser.add_argument("--reasoning-token-weight", type=float, default=0.25)
    parser.add_argument("--answer-token-weight", type=float, default=1.0)
    parser.add_argument("--frontier-positive-weight", type=float, default=8.0)
    parser.add_argument("--frontier-negative-weight", type=float, default=8.0)
    parser.add_argument("--frontier-max-tokens", type=int, default=24)
    parser.add_argument(
        "--frontier-negative-probability-floor", type=float, default=0.25
    )
    parser.add_argument(
        "--frontier-target-margin", type=float, default=FRONTIER_TARGET_MARGIN
    )
    parser.add_argument("--max-update-norm", type=float, default=2.0)

    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation")
    return parser


def config_from_args(args: argparse.Namespace) -> PersistentConfig:
    return PersistentConfig(
        branch=args.branch,
        variant=args.variant,
        model=args.model,
        model_id=args.model_id,
        revision=args.revision,
        episodes=args.episodes,
        scientific_checkpoints=parse_scientific_checkpoints(
            args.scientific_checkpoints
        ),
        rolling_checkpoint_interval=args.rolling_checkpoint_interval,
        max_sequence_tokens=args.max_sequence_tokens,
        max_rollout_tokens=args.max_rollout_tokens,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        train_temperature=args.train_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_grad_norm=args.max_grad_norm,
        distill_top_k=args.distill_top_k,
        distill_temperature=args.distill_temperature,
        distill_token_clip=args.distill_token_clip,
        distill_token_chunk_size=args.distill_token_chunk_size,
        ridge_lambda=args.ridge_lambda,
        residual_step_size=args.residual_step_size,
        max_tokens_per_candidate=args.max_tokens_per_candidate,
        max_support_tokens=args.max_support_tokens,
        num_specialization_candidates=args.num_specialization_candidates,
        hard_negatives=args.hard_negatives,
        ridge_max_length=args.ridge_max_length,
        reasoning_token_weight=args.reasoning_token_weight,
        answer_token_weight=args.answer_token_weight,
        frontier_positive_weight=args.frontier_positive_weight,
        frontier_negative_weight=args.frontier_negative_weight,
        frontier_max_tokens=args.frontier_max_tokens,
        frontier_negative_probability_floor=(
            args.frontier_negative_probability_floor
        ),
        frontier_target_margin=args.frontier_target_margin,
        max_update_norm=args.max_update_norm,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    config.validate()

    # Clean and the self-proposed privileged baseline consume the same verified
    # proposal schema.  Only the legacy fixed-prompt baseline has no probes.
    if config.branch in {"clean", "proposed_privileged"}:
        if not args.proposals:
            raise ValueError(f"{config.branch} persistent training requires --proposals")
        queries, proposals, hashes = load_persistent_inputs(
            args.queries, args.proposals, episodes=config.episodes
        )
    else:
        queries = load_query_only_manifest(args.queries)
        if len(queries) < config.episodes:
            raise ValueError(
                f"need at least {config.episodes} privileged episodes, "
                f"found {len(queries)}"
            )
        queries = queries[: config.episodes]
        proposals = {}
        hashes = {
            "query_manifest_sha256": canonical_json_sha256(queries),
            "proposal_manifest_sha256": canonical_json_sha256(
                {"mode": "privileged-no-clean-proposals-v1"}
            ),
        }
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    local_model = Path(args.model)
    load_revision = None if local_model.exists() else args.revision
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        use_lora=True,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        training=True,
        revision=load_revision,
    )
    runtime = collect_runtime_metadata(
        model, model_path=args.model_id, revision=args.revision
    )
    result = run_persistent_training(
        model=model,
        tokenizer=tokenizer,
        queries=queries,
        proposals=proposals,
        config=config,
        output_dir=args.output_dir,
        input_hashes=hashes,
        resume=args.resume,
        runtime_metadata=runtime,
    )
    if result.get("status") == "interrupted":
        # EX_TEMPFAIL lets the Slurm wrapper request requeue while the trainer
        # has already published a fully consistent restart checkpoint.
        return REQUEUE_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
