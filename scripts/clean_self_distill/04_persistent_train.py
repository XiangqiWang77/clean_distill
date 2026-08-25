#!/usr/bin/env python3
"""Train one restartable persistent LGSD, OPSD, or Veto LoRA branch.

The command intentionally has no labels, answers, or reference solutions.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import torch

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
        "--branch", choices=("clean", "privileged", "veto"), required=True
    )
    parser.add_argument("--queries", required=True)
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
    parser.add_argument("--trust-region-kl-budget", type=float, default=0.004)
    parser.add_argument("--trust-region-binary-search-steps", type=int, default=5)
    parser.add_argument(
        "--projection-scope",
        choices=("trajectory", "token", "fixed"),
        default="trajectory",
    )
    parser.add_argument(
        "--projection-path",
        choices=("exponential", "arithmetic"),
        default="exponential",
    )
    parser.add_argument("--fixed-projection-alpha", type=float, default=0.5595703125)
    parser.add_argument(
        "--student-kl-direction",
        choices=("reverse", "forward"),
        default="forward",
        help=(
            "forward is canonical LGSD/OPSD and required by Veto; reverse "
            "reproduces the legacy TRSD objective"
        ),
    )
    parser.add_argument(
        "--veto-beta-start",
        type=float,
        default=0.8,
        help="Veto product-of-experts beta at global step zero",
    )
    parser.add_argument(
        "--veto-beta-end",
        type=float,
        default=0.0,
        help="Veto beta approached at the end of training",
    )
    parser.add_argument(
        "--veto-beta-schedule",
        choices=("linear", "const"),
        default="linear",
    )
    parser.add_argument(
        "--disable-same-prefix-scoring",
        action="store_true",
        help="Score the teacher under its own position-aligned generated prefix",
    )
    parser.add_argument(
        "--update-guard",
        action="store_true",
        help="Skip an update whose projected target lowers realized trajectory log-probability",
    )

    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation")
    return parser


def config_from_args(args: argparse.Namespace) -> PersistentConfig:
    return PersistentConfig(
        branch=args.branch,
        variant=(
            "adaptive_target_reformulation"
            if args.branch == "veto"
            else "trust_region"
        ),
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
        trust_region_kl_budget=args.trust_region_kl_budget,
        trust_region_binary_search_steps=args.trust_region_binary_search_steps,
        projection_scope=args.projection_scope,
        projection_path=args.projection_path,
        fixed_projection_alpha=args.fixed_projection_alpha,
        student_kl_direction=args.student_kl_direction,
        same_prefix_scoring=not args.disable_same_prefix_scoring,
        update_guard=args.update_guard,
        veto_beta_start=args.veto_beta_start,
        veto_beta_end=args.veto_beta_end,
        veto_beta_schedule=args.veto_beta_schedule,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    config.validate()

    queries, hashes = load_persistent_inputs(
        args.queries, episodes=config.episodes
    )
    if config.branch == "veto":
        teacher_signal = {
            "mode": "veto-adaptive-target-reformulation-v1",
            "target": "teacher_student_product_of_experts_v1",
            "beta_schedule": config.veto_beta_schedule,
            "beta_start": config.veto_beta_start,
            "beta_end": config.veto_beta_end,
            "same_prefix_scoring": config.same_prefix_scoring,
            "distillation_kl_direction": config.distillation_kl_direction,
        }
    elif config.branch == "clean" and (
        config.projection_scope == "trajectory"
        and config.projection_path == "exponential"
        and config.student_kl_direction == "forward"
        and config.same_prefix_scoring
        and not config.update_guard
    ):
        teacher_signal = {
            "mode": "predecision-geometric-kl-ball-forward-distillation-v2"
        }
    elif config.branch == "clean" and (
        config.projection_scope == "trajectory"
        and config.projection_path == "exponential"
        and config.student_kl_direction == "reverse"
        and config.same_prefix_scoring
        and not config.update_guard
    ):
        teacher_signal = {"mode": "predecision-exponential-projection-v1"}
    elif config.branch == "clean":
        teacher_signal = {
            "mode": "predecision-projection-ablation-v1",
            "projection_scope": config.projection_scope,
            "projection_path": config.projection_path,
            "fixed_projection_alpha": (
                config.fixed_projection_alpha
                if config.projection_scope == "fixed"
                else None
            ),
            "student_kl_direction": config.student_kl_direction,
            "same_prefix_scoring": config.same_prefix_scoring,
            "update_guard": config.update_guard,
        }
    elif config.student_kl_direction == "forward":
        teacher_signal = {
            "mode": "raw-predecision-teacher-forward-distillation-v2"
        }
    else:
        teacher_signal = {"mode": "raw-predecision-teacher-v1"}
    hashes["teacher_signal_sha256"] = canonical_json_sha256(teacher_signal)
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
