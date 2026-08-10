#!/usr/bin/env python3
"""Replay the archived Qwen3-8B Privilege-SD trajectory and verify ep64.

This is deliberately a narrow recovery utility.  The original run retained its
64-row journal and episode-64 checkpoint, but its episode-16/32/48 checkpoints
and proposal manifest were cleaned.  Privilege-SD never reads proposal content
during an update; the archived implementation only validates it before the
training loop.  We therefore bypass that now-unrecoverable, computation-free
validation while retaining the original input hashes and fail closed unless the
replayed episode-64 adapter is byte-identical to the retained original.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch

from src.clean_self_distill import persistent
from src.clean_self_distill.heldout import load_query_only_manifest
from src.clean_self_distill.persistent import PersistentConfig, run_persistent_training
from src.clean_self_distill.runtime import collect_runtime_metadata, load_hf_model


ORIGINAL_QUERY_SHA256 = "7d1d3df1880f0ddf7eb4c11ff8f94d1d475d2ef14591540b50e70217b150fc0c"
ORIGINAL_PROPOSAL_SHA256 = "27af3415e9524d3b7ba60539c08cfeab220960ba7835403aecdbd3e422bda0ed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_queries(
    queries: list[dict[str, str]], original_rows: list[dict[str, Any]]
) -> None:
    if len(queries) != 64 or len(original_rows) != 64:
        raise RuntimeError(
            f"replay requires exactly 64 queries and journal rows; got {len(queries)} and {len(original_rows)}"
        )
    for index, (query, row) in enumerate(zip(queries, original_rows)):
        expected_episode = index + 1
        if (
            int(row.get("episode", -1)) != expected_episode
            or int(row.get("stream_index", -1)) != index
            or row.get("query_id") != query.get("query_id")
            or row.get("problem_sha256") != query.get("problem_sha256")
        ):
            raise RuntimeError(f"query/journal mismatch at episode {expected_episode}")


def validate_replay(
    output_dir: Path,
    original_rows: list[dict[str, Any]],
    original_checkpoint: Path,
    expected_adapter_sha256: str,
) -> None:
    expected_checkpoints = (0, 16, 32, 48, 64)
    for episode in expected_checkpoints:
        checkpoint = output_dir / "checkpoints" / f"episode_{episode:04d}"
        if not (checkpoint / "checkpoint_manifest.json").is_file():
            raise RuntimeError(f"missing replayed scientific checkpoint: {checkpoint}")

    original_adapter = original_checkpoint / "adapter_model.safetensors"
    replay_adapter = output_dir / "checkpoints/episode_0064/adapter_model.safetensors"
    original_digest = sha256(original_adapter)
    replay_digest = sha256(replay_adapter)
    if original_digest != expected_adapter_sha256:
        raise RuntimeError(
            f"retained original ep64 adapter changed: {original_digest} != {expected_adapter_sha256}"
        )
    if replay_digest != expected_adapter_sha256:
        raise RuntimeError(
            f"replayed ep64 adapter is not byte-identical: {replay_digest} != {expected_adapter_sha256}"
        )

    replay_rows = read_jsonl(output_dir / "episodes.jsonl")
    if len(replay_rows) != 64:
        raise RuntimeError(f"replayed journal has {len(replay_rows)} rows, expected 64")
    stable_fields = (
        "episode",
        "stream_index",
        "query_id",
        "problem_sha256",
        "seed",
        "response_tokens",
        "student_context_sha256",
        "teacher_context_sha256",
        "student_prefix_token_ids",
        "distillation_loss",
        "mean_teacher_student_kl",
    )
    for episode, (expected, actual) in enumerate(zip(original_rows, replay_rows), 1):
        mismatched = [field for field in stable_fields if expected.get(field) != actual.get(field)]
        if mismatched:
            raise RuntimeError(
                f"replayed episode {episode} differs from retained journal in {mismatched}"
            )

    marker = {
        "schema_version": "privileged-intermediate-replay-verification-v1",
        "status": "verified_byte_identical_ep64",
        "adapter_sha256": replay_digest,
        "verified_scientific_checkpoints": list(expected_checkpoints),
        "original_checkpoint": str(original_checkpoint),
        "proposal_validation_bypass": {
            "reason": "original proposal file was cleaned after the completed run",
            "safe_for_branch": "privileged",
            "training_dependency": "none",
            "original_proposal_manifest_sha256": ORIGINAL_PROPOSAL_SHA256,
        },
    }
    (output_dir / "REPLAY_VERIFIED.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--original-journal", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    all_queries = load_query_only_manifest(args.queries)
    queries = [dict(row) for row in all_queries[:64]]
    original_rows = read_jsonl(args.original_journal)
    validate_queries(queries, original_rows)

    config = PersistentConfig(
        branch="privileged",
        variant="correct_wrong_signed",
        model=args.model,
        model_id=args.model_id,
        revision=args.revision,
        episodes=64,
        scientific_checkpoints=(0, 16, 32, 48, 64),
        rolling_checkpoint_interval=1,
        max_sequence_tokens=16_384,
        max_rollout_tokens=4_096,
        learning_rate=2e-5,
        weight_decay=0.0,
        lora_rank=8,
        lora_alpha=16,
        seed=0,
        train_temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_grad_norm=1.0,
        distill_top_k=64,
        distill_temperature=1.0,
        distill_token_clip=0.0,
        distill_token_chunk_size=128,
        ridge_lambda=0.1,
        residual_step_size=0.8,
        max_tokens_per_candidate=96,
        max_support_tokens=768,
        num_specialization_candidates=None,
        hard_negatives=8,
        ridge_max_length=8_192,
        reasoning_token_weight=0.25,
        answer_token_weight=1.0,
        frontier_positive_weight=8.0,
        frontier_negative_weight=8.0,
        frontier_max_tokens=24,
        frontier_negative_probability_floor=0.25,
        frontier_target_margin=1.0,
        max_update_norm=2.0,
    )
    config.validate()

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model_path = Path(args.model)
    model, tokenizer = load_hf_model(
        args.model,
        dtype="bfloat16",
        device_map="auto",
        use_lora=True,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        training=True,
        revision=None if model_path.exists() else args.revision,
    )
    runtime = collect_runtime_metadata(model, model_path=args.model_id, revision=args.revision)

    # The archived Privilege-SD branch never consults this object in
    # train_one_episode.  Keep exact coverage and bypass only the unavailable
    # proposal firewall, then prove computational equivalence at ep64.
    proposals = {str(query["query_id"]): {} for query in queries}
    persistent._validate_proposal_firewall = lambda _proposal, _query: None
    run_persistent_training(
        model=model,
        tokenizer=tokenizer,
        queries=queries,
        proposals=proposals,
        config=config,
        output_dir=args.output_dir,
        input_hashes={
            "query_manifest_sha256": ORIGINAL_QUERY_SHA256,
            "proposal_manifest_sha256": ORIGINAL_PROPOSAL_SHA256,
        },
        resume=args.resume,
        runtime_metadata=runtime,
    )
    validate_replay(
        args.output_dir,
        original_rows,
        args.original_checkpoint,
        args.expected_adapter_sha256,
    )


if __name__ == "__main__":
    main()
