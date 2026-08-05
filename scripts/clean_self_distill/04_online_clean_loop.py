#!/usr/bin/env python3
"""Run the core Clean-SD loop directly on a query-only DeepMath stream.

Each episode proposes verified right/wrong domain probes with the current
student, builds the signed ridge teacher, and immediately transfers that
teacher on the student's own prefix.  No target label is accepted by this CLI.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from src.clean_self_distill.heldout import load_query_only_manifest
from src.clean_self_distill.io import canonical_json_sha256, load_proposal_map
from src.clean_self_distill.persistent import (
    REQUEUE_EXIT_CODE,
    PersistentConfig,
    _atomic_write_jsonl,
    parse_scientific_checkpoints,
    run_persistent_training,
)
from src.clean_self_distill.propose import propose_for_query
from src.clean_self_distill.runtime import (
    HFGenerator,
    collect_runtime_metadata,
    load_hf_model,
    unwrap_causal_lm,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--scientific-checkpoints", default="0,250,500,750,1000")
    parser.add_argument("--rolling-checkpoint-interval", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--proposal-oversample", type=int, default=2)
    parser.add_argument("--proposal-max-rounds", type=int, default=3)
    parser.add_argument("--min-accepted-candidates", type=int, default=4)
    parser.add_argument("--proposal-max-new-tokens", type=int, default=1536)
    parser.add_argument("--stage-max-attempts", type=int, default=2)
    parser.add_argument("--max-fourgram-overlap-rate", type=float, default=0.20)
    parser.add_argument("--max-fourgram-overlap-count", type=int, default=4)

    parser.add_argument("--max-sequence-tokens", type=int, default=16_384)
    parser.add_argument("--max-rollout-tokens", type=int, default=16_384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--distill-token-chunk-size", type=int, default=128)
    parser.add_argument("--ridge-lambda", type=float, default=0.1)
    parser.add_argument("--residual-step-size", type=float, default=0.8)
    parser.add_argument("--max-support-tokens", type=int, default=768)
    parser.add_argument("--max-tokens-per-candidate", type=int, default=96)
    parser.add_argument("--hard-negatives", type=int, default=8)
    parser.add_argument("--frontier-positive-weight", type=float, default=8.0)
    parser.add_argument("--frontier-negative-weight", type=float, default=8.0)
    parser.add_argument("--frontier-target-margin", type=float, default=1.0)
    parser.add_argument("--max-update-norm", type=float, default=2.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.min_accepted_candidates <= args.num_candidates:
        raise ValueError("min accepted candidates must be within candidate count")
    queries = load_query_only_manifest(args.queries)
    if len(queries) < args.episodes:
        raise ValueError(
            f"need at least {args.episodes} query-only episodes, found {len(queries)}"
        )
    queries = queries[: args.episodes]

    config = PersistentConfig(
        branch="clean",
        variant="correct_wrong_signed",
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
        weight_decay=0.0,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        train_temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_grad_norm=1.0,
        distill_top_k=64,
        distill_temperature=1.0,
        distill_token_clip=0.0,
        distill_token_chunk_size=args.distill_token_chunk_size,
        ridge_lambda=args.ridge_lambda,
        residual_step_size=args.residual_step_size,
        max_tokens_per_candidate=args.max_tokens_per_candidate,
        max_support_tokens=args.max_support_tokens,
        hard_negatives=args.hard_negatives,
        ridge_max_length=args.max_sequence_tokens,
        reasoning_token_weight=0.25,
        answer_token_weight=1.0,
        frontier_positive_weight=args.frontier_positive_weight,
        frontier_negative_weight=args.frontier_negative_weight,
        frontier_max_tokens=24,
        frontier_negative_probability_floor=0.25,
        frontier_target_margin=args.frontier_target_margin,
        max_update_norm=args.max_update_norm,
    )
    config.validate()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = output_dir / "online_proposals.jsonl"
    proposals = load_proposal_map(proposal_path) if proposal_path.exists() else {}
    if proposals and not args.resume:
        raise ValueError("online proposals already exist; use --resume")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    local_model = Path(args.model)
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        training=True,
        revision=None if local_model.exists() else args.revision,
    )
    runtime = collect_runtime_metadata(
        model, model_path=args.model_id, revision=args.revision
    )

    proposer = HFGenerator(
        model,
        tokenizer,
        max_new_tokens=args.proposal_max_new_tokens,
        temperature=0.8,
        top_p=0.95,
        enable_thinking=False,
    )
    solver = HFGenerator(
        model,
        tokenizer,
        max_new_tokens=args.proposal_max_new_tokens,
        temperature=0.3,
        top_p=0.95,
        enable_thinking=False,
    )
    verifier = HFGenerator(
        model,
        tokenizer,
        max_new_tokens=args.proposal_max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    )

    proposal_rows = [dict(row) for row in proposals.values()]

    def commit_proposal(row):
        proposal_rows.append(dict(row))
        _atomic_write_jsonl(proposal_path, proposal_rows)

    def provide_proposal(query, stream_index):
        # Proposal sampling is episode-seeded and isolated from persistent
        # training RNG, so a saved pending proposal can be reused after requeue.
        python_state = random.getstate()
        base = unwrap_causal_lm(model)
        prior_training = model.training
        prior_cache = bool(getattr(base.config, "use_cache", False))
        model.eval()
        base.config.use_cache = True
        seed = args.seed + (stream_index + 1) * 100_003 + 17
        try:
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                row = propose_for_query(
                    dict(query),
                    proposer,
                    solver,
                    verifier,
                    num_candidates=args.num_candidates,
                    proposal_oversample=args.proposal_oversample,
                    max_rounds=args.proposal_max_rounds,
                    min_accepted_candidates=args.min_accepted_candidates,
                    max_literal_overlap=0.0,
                    max_fourgram_overlap=args.max_fourgram_overlap_rate,
                    accept_verifier_corrections=False,
                    stage_max_attempts=args.stage_max_attempts,
                    max_fourgram_overlap_count=args.max_fourgram_overlap_count,
                )
        finally:
            random.setstate(python_state)
            base.config.use_cache = prior_cache
            model.train(prior_training)
        row["model"] = args.model_id
        row["model_revision"] = args.revision
        row["online_episode"] = stream_index + 1
        return row

    proposal_policy = {
        "mode": "online-current-student-right-wrong-frontier-v1",
        "num_candidates": args.num_candidates,
        "proposal_oversample": args.proposal_oversample,
        "max_rounds": args.proposal_max_rounds,
        "min_accepted_candidates": args.min_accepted_candidates,
        "max_new_tokens": args.proposal_max_new_tokens,
        "stage_max_attempts": args.stage_max_attempts,
        "max_fourgram_overlap_rate": args.max_fourgram_overlap_rate,
        "max_fourgram_overlap_count": args.max_fourgram_overlap_count,
        "seed": args.seed,
    }
    hashes = {
        "query_manifest_sha256": canonical_json_sha256(queries),
        # This binds the online generator policy; every realized proposal has
        # its own proposal_training_sha256 in the episode ridge audit.
        "proposal_manifest_sha256": canonical_json_sha256(proposal_policy),
    }
    result = run_persistent_training(
        model=model,
        tokenizer=tokenizer,
        queries=queries,
        proposals=proposals,
        config=config,
        output_dir=output_dir,
        input_hashes=hashes,
        resume=args.resume,
        runtime_metadata={**runtime, "online_proposal_policy": proposal_policy},
        proposal_provider=provide_proposal,
        proposal_committer=commit_proposal,
    )
    print(json.dumps(result, sort_keys=True))
    return REQUEUE_EXIT_CODE if result.get("status") == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
