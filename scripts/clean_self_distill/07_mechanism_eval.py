#!/usr/bin/env python3
"""Generate pre-decision or post-outcome Teacher mechanism trajectories."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from src.clean_self_distill.heldout import (
    _atomic_write_jsonl,
    generation_config_sha256,
    load_query_only_manifest,
    load_sealed_labels,
    paired_sample_seed,
    query_manifest_sha256,
)
from src.clean_self_distill.io import iter_rows
from src.clean_self_distill.persistent import (
    build_privileged_prompt,
    file_sha256,
    style_task_error_from_trace,
)
from src.clean_self_distill.ridge import problem_prompt
from src.clean_self_distill.runtime import (
    backbone_forward,
    collect_runtime_metadata,
    input_device,
    load_hf_model,
    project_logits,
    render_chat,
)
from src.clean_self_distill.train_eval import _sample_token
from src.opsd_format import extract_boxed_answer, grade_boxed_answer


TEACHER_TYPES = ("pre_decision_privilege", "post_outcome_privilege")
_REFERENCE_RE = re.compile(
    r"\b(?:answer\s*key|given\s+(?:answer|solution)|reference\s+(?:answer|solution)|"
    r"according\s+to\s+the\s+(?:answer|reference))\b",
    flags=re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(?:perhaps|maybe|possibly|likely|seems?|appears?|I\s+think|not\s+sure)\b",
    flags=re.IGNORECASE,
)


def _post_outcome_prompt(tokenizer, problem: str, *, previous_correct: bool) -> str:
    verdict = "correct" if previous_correct else "incorrect"
    messages = [
        {
            "role": "system",
            "content": (
                "Post-outcome feedback for a privileged mechanism control: an "
                f"external evaluator judged a previous attempt {verdict}. Re-solve "
                "the problem carefully, check alternatives and boundary cases, and "
                "do not assume access to any reference answer or reference solution."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{problem.strip()}\n\nPlease reason step by step, and put your final "
                "answer within \\boxed{{}}."
            ),
        },
    ]
    return render_chat(tokenizer, messages, add_generation_prompt=True)


def _load_base_predictions(
    paths: Sequence[str], queries: Sequence[Mapping[str, str]]
) -> dict[str, dict[str, Any]]:
    expected = {query["query_id"]: query for query in queries}
    selected: dict[str, dict[str, Any]] = {}
    for path in paths:
        for raw in iter_rows(path):
            if raw.get("method") != "base" or raw.get("sample_index") != 0:
                continue
            query_id = str(raw.get("query_id", ""))
            if query_id not in expected or query_id in selected:
                raise ValueError(f"Unexpected/duplicate Base sample-0 prediction {query_id!r}")
            if "correct" in raw or "profile" in raw:
                raise ValueError("Mechanism generation requires blind Base predictions")
            query = expected[query_id]
            if (
                raw.get("problem_sha256") != query["problem_sha256"]
                or str(raw.get("source", "")).casefold() != query["source"]
            ):
                raise ValueError(f"Base prediction binding mismatch for {query_id}")
            selected[query_id] = dict(raw)
    if set(selected) != set(expected):
        raise ValueError("Base sample-0 predictions do not cover the held-out manifest")
    return selected


def _load_resume(
    path: Path,
    expected_queries: Sequence[Mapping[str, str]],
    teacher_type: str,
    *,
    manifest_sha256: str,
    generation_sha256: str,
    seed_by_query: Mapping[str, int],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [dict(row) for row in iter_rows(path)]
    if len(rows) > len(expected_queries):
        raise ValueError("Mechanism resume file is longer than its shard")
    for index, row in enumerate(rows):
        query = expected_queries[index]
        if (
            row.get("query_id") != query["query_id"]
            or row.get("teacher_type") != teacher_type
            or row.get("record_type") != "trajectory"
            or row.get("problem_sha256") != query["problem_sha256"]
            or str(row.get("source", "")).casefold() != query["source"]
            or row.get("query_manifest_sha256") != manifest_sha256
            or row.get("generation_config_sha256") != generation_sha256
            or row.get("seed") != seed_by_query[query["query_id"]]
        ):
            raise ValueError("Mechanism resume file is not the exact expected prefix")
    return rows


@torch.inference_mode()
def _dual_context_generate(
    model,
    tokenizer,
    *,
    problem: str,
    teacher_prompt: str,
    max_new_tokens: int,
    max_prompt_tokens: int,
    context_window: int,
    seed: int,
) -> tuple[str, int, bool, list[dict[str, Any]], float, int, int]:
    device = input_device(model)
    student_prompt = problem_prompt(tokenizer, problem)
    student_ids = tokenizer(
        student_prompt, add_special_tokens=True, return_tensors="pt"
    )["input_ids"].to(device)
    teacher_ids = tokenizer(
        teacher_prompt, add_special_tokens=True, return_tensors="pt"
    )["input_ids"].to(device)
    if max(int(student_ids.numel()), int(teacher_ids.numel())) > max_prompt_tokens:
        raise ValueError("Mechanism prompt exceeds the preregistered prompt cap")
    if (
        max(int(student_ids.numel()), int(teacher_ids.numel())) + max_new_tokens
        > context_window
    ):
        raise ValueError("Mechanism prompt cannot receive the full evaluation opportunity")
    student_next = student_ids
    teacher_next = teacher_ids
    student_cache = teacher_cache = None
    generator: torch.Generator | None = None
    generated: list[torch.Tensor] = []
    trace: list[dict[str, Any]] = []
    entropy_sum = 0.0
    ended_by_eos = False
    model.eval()
    for _ in range(max_new_tokens):
        student_hidden, student_cache = backbone_forward(
            model,
            input_ids=student_next,
            past_key_values=student_cache,
            use_cache=True,
        )
        teacher_hidden, teacher_cache = backbone_forward(
            model,
            input_ids=teacher_next,
            past_key_values=teacher_cache,
            use_cache=True,
        )
        student_logits = project_logits(model, student_hidden[:, -1, :]).float()
        teacher_logits = project_logits(model, teacher_hidden[:, -1, :]).float()
        if generator is None:
            generator = torch.Generator(device=teacher_logits.device)
            generator.manual_seed(seed)
        token = _sample_token(
            teacher_logits,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            generator=generator,
        )
        token_id = int(token.item())
        student_log = F.log_softmax(student_logits, dim=-1)
        teacher_log = F.log_softmax(teacher_logits, dim=-1)
        teacher_probability = teacher_log.exp()
        entropy_sum += float(
            -(teacher_probability * teacher_log).sum(dim=-1).item()
        )
        trace.append(
            {
                "token_id": token_id,
                "student_logprob": float(student_log[0, token_id].item()),
                "teacher_logprob": float(teacher_log[0, token_id].item()),
            }
        )
        generated.append(token)
        student_next = teacher_next = token[:, None].to(device)
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            ended_by_eos = True
            break
    response_ids = (
        torch.stack(generated, dim=1)
        if generated
        else torch.empty((1, 0), dtype=torch.long, device=device)
    )
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
    length = int(response_ids.numel())
    return (
        response,
        length,
        length >= max_new_tokens and not ended_by_eos,
        trace,
        entropy_sum / max(length, 1),
        int(student_ids.numel()),
        int(teacher_ids.numel()),
    )


def run(args: argparse.Namespace) -> None:
    queries = load_query_only_manifest(args.queries)
    base = _load_base_predictions(args.base_predictions, queries)
    labels = None
    if args.teacher_type == "post_outcome_privilege":
        if not args.labels:
            raise ValueError("Post-outcome privilege requires --labels")
        labels = load_sealed_labels(args.labels)
        if set(labels) != {query["query_id"] for query in queries}:
            raise ValueError("Post-outcome label coverage mismatch")
        for query in queries:
            if labels[query["query_id"]]["problem_sha256"] != query["problem_sha256"]:
                raise ValueError(
                    f"Post-outcome label binding mismatch for {query['query_id']}"
                )
    elif args.labels:
        raise ValueError("Pre-decision privilege must not receive a label path")

    selected = [
        query
        for index, query in enumerate(queries)
        if index % args.num_shards == args.shard_index
    ]
    manifest_hash = query_manifest_sha256(queries)
    global_indices = {query["query_id"]: index for index, query in enumerate(queries)}
    for query in queries:
        base_row = base[query["query_id"]]
        expected_seed = paired_sample_seed(
            args.seed, global_indices[query["query_id"]], 0
        )
        if (
            base_row.get("query_manifest_sha256") != manifest_hash
            or base_row.get("checkpoint_episode") != 0
            or base_row.get("checkpoint_sha256") != "base"
            or base_row.get("seed") != expected_seed
            or float(base_row.get("temperature", -1.0)) != 0.6
            or float(base_row.get("top_p", -1.0)) != 0.95
            or int(base_row.get("top_k", -1)) != 20
            or int(base_row.get("max_new_tokens", -1)) != 32768
        ):
            raise ValueError(
                f"Base prediction protocol mismatch for {query['query_id']}"
            )
    seed_by_query = {
        query["query_id"]: paired_sample_seed(
            args.seed, global_indices[query["query_id"]], 0
        )
        for query in selected
    }
    generation_digest = generation_config_sha256(
        {
            "schema_version": "clean-self-distill-mechanism-generation-config-v1",
            "teacher_type": args.teacher_type,
            "query_manifest_sha256": manifest_hash,
            "base_prediction_sha256": [file_sha256(path) for path in args.base_predictions],
            "sealed_label_sha256": file_sha256(args.labels) if args.labels else None,
            "model_id": args.model_id,
            "revision": args.revision,
            "sampling": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_new_tokens": args.max_new_tokens,
                "max_prompt_tokens": args.max_prompt_tokens,
                "context_window": args.context_window,
                "seed": args.seed,
            },
            "shard": {"count": args.num_shards, "index": args.shard_index},
        }
    )
    rows = _load_resume(
        Path(args.output),
        selected,
        args.teacher_type,
        manifest_sha256=manifest_hash,
        generation_sha256=generation_digest,
        seed_by_query=seed_by_query,
    )
    local = Path(args.model)
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        training=False,
        revision=None if local.exists() else args.revision,
    )
    runtime = collect_runtime_metadata(
        model, model_path=args.model_id, revision=args.revision
    )
    for query in selected[len(rows) :]:
        query_id = query["query_id"]
        previous_correct: bool | None = None
        if args.teacher_type == "pre_decision_privilege":
            teacher_prompt = build_privileged_prompt(tokenizer, query["problem"])
        else:
            assert labels is not None
            parsed = extract_boxed_answer(str(base[query_id].get("response", "")))
            previous_correct = bool(
                grade_boxed_answer(parsed, labels[query_id]["answer"])
            )
            teacher_prompt = _post_outcome_prompt(
                tokenizer, query["problem"], previous_correct=previous_correct
            )
        seed = paired_sample_seed(args.seed, global_indices[query_id], 0)
        (
            response,
            token_count,
            truncated,
            trace,
            mean_entropy,
            student_prompt_tokens,
            teacher_prompt_tokens,
        ) = _dual_context_generate(
            model,
            tokenizer,
            problem=query["problem"],
            teacher_prompt=teacher_prompt,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            context_window=args.context_window,
            seed=seed,
        )
        exposed = token_count if args.teacher_type == "post_outcome_privilege" else 0
        metrics = {
            "token_count": token_count,
            "student_logprob_sum": sum(float(item["student_logprob"]) for item in trace),
            "teacher_logprob_sum": sum(float(item["teacher_logprob"]) for item in trace),
            **style_task_error_from_trace(tokenizer, trace),
        }
        rows.append(
            {
                "schema_version": "clean-self-distill-mechanism-trajectory-v1",
                "record_type": "trajectory",
                "teacher_type": args.teacher_type,
                "query_id": query_id,
                "problem_sha256": query["problem_sha256"],
                "source": query["source"],
                "trajectory_id": f"{query_id}:sample0",
                "query_manifest_sha256": manifest_hash,
                "generation_config_sha256": generation_digest,
                "seed": seed,
                "response": response,
                "trajectory_metrics": metrics,
                "training_audit": {
                    "teacher_positions": token_count,
                    "hindsight_exposed_positions": exposed,
                    "compared_positions": token_count,
                    "exact_context_positions": 0,
                },
                "previous_attempt_feedback": (
                    None if previous_correct is None else ("correct" if previous_correct else "incorrect")
                ),
                "behavioral_diagnostics": {
                    "fabricated_reference_hallucination": bool(_REFERENCE_RE.search(response)),
                    "hedging_token_count": len(_HEDGE_RE.findall(response)),
                    "response_tokens": token_count,
                    "mean_entropy": mean_entropy if math.isfinite(mean_entropy) else None,
                    "truncated": truncated,
                },
                "runtime": runtime,
            }
        )
        _atomic_write_jsonl(Path(args.output), rows)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--teacher-type", choices=TEACHER_TYPES, required=True)
    value.add_argument("--queries", required=True)
    value.add_argument("--base-predictions", action="append", required=True)
    value.add_argument("--labels")
    value.add_argument("--output", required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--model-id", default="Qwen/Qwen3-8B")
    value.add_argument("--revision", required=True)
    value.add_argument("--dtype", default="bfloat16")
    value.add_argument("--device-map", default="auto")
    value.add_argument("--max-new-tokens", type=int, default=32768)
    value.add_argument("--max-prompt-tokens", type=int, default=8192)
    value.add_argument("--context-window", type=int, default=40960)
    value.add_argument("--num-shards", type=int, default=1)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--seed", type=int, default=0)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards>0 and a valid shard_index")
    run(args)


if __name__ == "__main__":
    main()
