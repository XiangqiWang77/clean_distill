#!/usr/bin/env python3
"""Collect gold-anchored nuisance and surrogate-reliability evidence on DeepMath 10%."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from statistics import pvariance
from typing import Any

import pyarrow.parquet as pq
import torch

from src.clean_self_distill.generation import problem_prompt
from src.clean_self_distill.heldout import _atomic_write_jsonl
from src.clean_self_distill.runtime import backbone_forward, input_device, load_hf_model
from src.clean_self_distill.trust_region_mechanism import (
    WRAPPER_IDS,
    build_privileged_prompt_wrapper,
    evaluate_projection_alphas,
    solve_epsilon_alphas,
    token_categories,
)


SCHEMA_VERSION = "trsd-deepmath10-query-evidence-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Malformed JSONL: {path}")
    return rows


def mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("mean requires finite nonempty values")
    return math.fsum(values) / len(values)


def selected_mean(values: list[float], categories: list[str], category: str) -> float | None:
    chosen = [value for value, item in zip(values, categories, strict=True) if item == category]
    return mean(chosen) if chosen else None


def prompt_ids(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    value = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"]
    return value.to(device)


def score_query(
    *,
    model,
    tokenizer,
    query_id: str,
    study_index: int,
    problem: str,
    solution: str,
    model_id: str,
    revision: str,
    manifest_sha256: str,
    token_cap: int,
    max_prompt_tokens: int,
    context_window: int,
    epsilon: float,
    binary_search_steps: int,
    chunk_size: int,
) -> dict[str, Any]:
    device = input_device(model)
    normal_text = problem_prompt(tokenizer, problem)
    normal_ids = prompt_ids(tokenizer, normal_text, device)
    wrapper_texts = {
        wrapper: build_privileged_prompt_wrapper(tokenizer, problem, wrapper)
        for wrapper in WRAPPER_IDS
    }
    wrapper_ids = {
        wrapper: prompt_ids(tokenizer, text, device)
        for wrapper, text in wrapper_texts.items()
    }
    longest_prompt = max(
        int(normal_ids.shape[1]), *(int(value.shape[1]) for value in wrapper_ids.values())
    )
    if longest_prompt > max_prompt_tokens:
        raise ValueError(f"{query_id}: prompt has {longest_prompt}>{max_prompt_tokens} tokens")

    reference_ids = tokenizer(
        solution.strip(), add_special_tokens=False, return_tensors="pt"
    )["input_ids"][:, :token_cap].to(device)
    token_count = int(reference_ids.shape[1])
    if token_count <= 0:
        raise ValueError(f"{query_id}: empty reference solution")
    if longest_prompt + token_count > context_window:
        raise ValueError(f"{query_id}: prompt plus reference exceeds context window")
    categories, _token_text = token_categories(tokenizer, reference_ids)

    student_full = torch.cat([normal_ids, reference_ids], dim=1)
    with torch.inference_mode():
        student_all, _ = backbone_forward(
            model,
            input_ids=student_full,
            attention_mask=torch.ones_like(student_full),
            use_cache=False,
        )
    student_start = int(normal_ids.shape[1]) - 1
    student_hidden = student_all[:, student_start : student_start + token_count].detach()
    if int(student_hidden.shape[1]) != token_count:
        raise ValueError(f"{query_id}: student hidden-state alignment failed")

    wrappers: list[dict[str, Any]] = []
    raw_vectors: list[list[float]] = []
    projected_vectors: list[list[float]] = []
    student_logprobs: list[float] | None = None
    for wrapper in WRAPPER_IDS:
        teacher_full = torch.cat([wrapper_ids[wrapper], reference_ids], dim=1)
        with torch.inference_mode():
            teacher_all, _ = backbone_forward(
                model,
                input_ids=teacher_full,
                attention_mask=torch.ones_like(teacher_full),
                use_cache=False,
            )
        teacher_start = int(wrapper_ids[wrapper].shape[1]) - 1
        teacher_hidden = teacher_all[:, teacher_start : teacher_start + token_count].detach()
        if teacher_hidden.shape != student_hidden.shape:
            raise ValueError(f"{query_id}: {wrapper} hidden-state alignment failed")

        endpoints = evaluate_projection_alphas(
            model=model,
            student_hidden=student_hidden,
            privileged_hidden=teacher_hidden,
            labels=reference_ids,
            categories=categories,
            alphas=(0.0, 1.0),
            chunk_size=chunk_size,
            capture_trace=False,
        )
        alpha = solve_epsilon_alphas(
            model=model,
            student_hidden=student_hidden,
            privileged_hidden=teacher_hidden,
            labels=reference_ids,
            categories=categories,
            alpha_evaluation=endpoints,
            epsilon_grid=(epsilon,),
            chunk_size=chunk_size,
            binary_search_steps=binary_search_steps,
        )[epsilon]
        final = evaluate_projection_alphas(
            model=model,
            student_hidden=student_hidden,
            privileged_hidden=teacher_hidden,
            labels=reference_ids,
            categories=categories,
            alphas=tuple(sorted({alpha, 1.0})),
            chunk_size=chunk_size,
            capture_trace=True,
        )
        raw = dict(final.summaries[1.0])
        projected = dict(final.summaries[alpha])
        raw_trace = final.traces[1.0]
        projected_trace = final.traces[alpha]
        raw_vector = [float(value) for value in raw_trace["logratio"]]
        projected_vector = [float(value) for value in projected_trace["logratio"]]
        if student_logprobs is None:
            student_logprobs = [float(value) for value in raw_trace["student_logprob"]]
        raw_vectors.append(raw_vector)
        projected_vectors.append(projected_vector)
        wrappers.append(
            {
                "wrapper_id": wrapper,
                "prompt_sha256": hashlib.sha256(wrapper_texts[wrapper].encode()).hexdigest(),
                "prompt_tokens": int(wrapper_ids[wrapper].shape[1]),
                "raw": raw,
                "projected": projected,
                "constraint_active": alpha < 1.0 - 1e-12,
                "gold_positive_token_fraction_raw": sum(value > 0 for value in raw_vector)
                / token_count,
                "gold_positive_token_fraction_projected": sum(
                    value > 0 for value in projected_vector
                )
                / token_count,
            }
        )
        del teacher_full, teacher_all, teacher_hidden, endpoints, final
        gc.collect()
        torch.cuda.empty_cache()

    raw_position_variance = [
        pvariance([vector[position] for vector in raw_vectors])
        for position in range(token_count)
    ]
    projected_position_variance = [
        pvariance([vector[position] for vector in projected_vectors])
        for position in range(token_count)
    ]
    neutral = wrappers[0]
    neutral_raw = raw_vectors[0]
    neutral_projected = projected_vectors[0]
    if student_logprobs is None:
        raise ValueError(f"{query_id}: no wrapper traces were collected")
    row = {
        "schema_version": SCHEMA_VERSION,
        "query_id": query_id,
        "study_index": study_index,
        "split": "audit" if study_index < 1558 else "confirmation",
        "model_id": model_id,
        "model_revision": revision,
        "manifest_sha256": manifest_sha256,
        "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        "reference_solution_sha256": hashlib.sha256(solution.encode()).hexdigest(),
        "reference_tokens": token_count,
        "reference_token_cap": token_cap,
        "task_tokens": categories.count("task"),
        "style_tokens": categories.count("style"),
        "other_tokens": categories.count("other"),
        "primary_epsilon": epsilon,
        "wrappers": wrappers,
        "nuisance": {
            "raw_prompt_variance_mean": mean(raw_position_variance),
            "projected_prompt_variance_mean": mean(projected_position_variance),
            "raw_style_shift_neutral": neutral["raw"]["style_abs_logprob_shift"],
            "projected_style_shift_neutral": neutral["projected"][
                "style_abs_logprob_shift"
            ],
        },
        "useful_signal": {
            "student_reference_nll": -mean(student_logprobs),
            "raw_reference_nll_neutral": -mean(
                [left + right for left, right in zip(student_logprobs, neutral_raw, strict=True)]
            ),
            "projected_reference_nll_neutral": -mean(
                [
                    left + right
                    for left, right in zip(student_logprobs, neutral_projected, strict=True)
                ]
            ),
            "raw_gold_logprob_gain_neutral": mean(neutral_raw),
            "projected_gold_logprob_gain_neutral": mean(neutral_projected),
            "raw_task_gold_logprob_gain_neutral": selected_mean(
                neutral_raw, categories, "task"
            ),
            "projected_task_gold_logprob_gain_neutral": selected_mean(
                neutral_projected, categories, "task"
            ),
            "raw_all_wrapper_min_gold_gain": min(
                float(wrapper["raw"]["normalized_logratio"]) for wrapper in wrappers
            ),
            "projected_all_wrapper_min_gold_gain": min(
                float(wrapper["projected"]["normalized_logratio"])
                for wrapper in wrappers
            ),
        },
    }
    del student_full, student_all, student_hidden, reference_ids
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepmath", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--reference-token-cap", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--context-window", type=int, default=40960)
    parser.add_argument("--epsilon", type=float, default=0.004)
    parser.add_argument("--binary-search-steps", type=int, default=6)
    parser.add_argument("--full-vocab-chunk-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    query_ids = [str(value) for value in manifest["query_ids"]]
    if manifest.get("schema_version") != "trsd-deepmath10-surrogate-study-v1":
        raise ValueError("Wrong DeepMath-10 manifest schema")
    if len(query_ids) != 3116 or len(set(query_ids)) != 3116:
        raise ValueError("DeepMath-10 manifest must contain 3,116 unique queries")
    assigned = {
        query_id: index
        for index, query_id in enumerate(query_ids)
        if index % args.num_shards == args.shard_index
    }
    rows = read_jsonl(args.output)
    completed = {str(row.get("query_id")) for row in rows}
    if len(completed) != len(rows) or not completed <= set(assigned):
        raise ValueError("Existing output contains duplicate or foreign queries")
    for row in rows:
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("manifest_sha256") != args.manifest_sha256
        ):
            raise ValueError("Existing output belongs to another protocol")

    local_model = Path(args.model)
    model, tokenizer = load_hf_model(
        args.model,
        dtype="bfloat16",
        device_map="auto",
        training=False,
        revision=None if local_model.exists() else args.revision,
        attn_implementation="sdpa",
    )
    model.eval()
    pending_since_checkpoint = 0
    parquet = pq.ParquetFile(args.deepmath)
    columns = [
        "extra_info.index",
        "extra_info.problem",
        "extra_info.solution",
    ]
    for batch in parquet.iter_batches(batch_size=4, columns=columns):
        for source in batch.to_pylist():
            extra = source["extra_info"]
            query_id = str(extra["index"])
            if query_id not in assigned or query_id in completed:
                continue
            row = score_query(
                model=model,
                tokenizer=tokenizer,
                query_id=query_id,
                study_index=assigned[query_id],
                problem=str(extra["problem"]),
                solution=str(extra["solution"]),
                model_id=args.model_id,
                revision=args.revision,
                manifest_sha256=args.manifest_sha256,
                token_cap=args.reference_token_cap,
                max_prompt_tokens=args.max_prompt_tokens,
                context_window=args.context_window,
                epsilon=args.epsilon,
                binary_search_steps=args.binary_search_steps,
                chunk_size=args.full_vocab_chunk_size,
            )
            rows.append(row)
            completed.add(query_id)
            pending_since_checkpoint += 1
            if pending_since_checkpoint >= args.checkpoint_every:
                _atomic_write_jsonl(args.output, rows)
                pending_since_checkpoint = 0
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "completed": len(rows),
                            "expected": len(assigned),
                            "query_id": query_id,
                        }
                    ),
                    flush=True,
                )
    _atomic_write_jsonl(args.output, rows)
    if completed != set(assigned):
        raise ValueError(f"Shard incomplete: {len(completed)}/{len(assigned)}")
    print(json.dumps({"status": "complete", "shard": args.shard_index, "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
