#!/usr/bin/env python3
"""Teacher-force held-out human-preference pairs under a model checkpoint.

For every existing LMArena pair this scorer computes the mean token
log-probability of the human-preferred and human-rejected responses.  It does
not generate responses, call an LLM judge, or fit a Bradley--Terry model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.clean_self_distill.arena_preference import (
    SCORE_SCHEMA_VERSION,
    ArenaPreferenceError,
    load_preference_pairs,
    read_jsonl,
    summarize_score_rows,
    validate_score_row,
)


PROMPT_PROFILE = "arena-human-preference-ordinary-chat-v1"
TRUNCATION_POLICY = "prompt-prefix_then-response-prefix-v1"


class PreferenceScoringError(ArenaPreferenceError):
    """Raised when a score run violates the fixed evaluation protocol."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _summary_path(path: Path) -> Path:
    return path.with_suffix(".summary.json")


def _single_token_ids(encoded: Any, *, context: str) -> list[int]:
    try:
        values = encoded["input_ids"]
    except (KeyError, TypeError) as exc:
        raise PreferenceScoringError(
            f"tokenizer did not return input_ids for {context}"
        ) from exc
    detach = getattr(values, "detach", None)
    if callable(detach):
        values = detach()
    cpu = getattr(values, "cpu", None)
    if callable(cpu):
        values = cpu()
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    if isinstance(values, tuple):
        values = list(values)
    if isinstance(values, list) and values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise PreferenceScoringError(
                f"tokenizer returned multiple sequences for {context}"
            )
        values = list(values[0])
    if not isinstance(values, list) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in values
    ):
        raise PreferenceScoringError(
            f"tokenizer returned invalid input_ids for {context}"
        )
    return list(values)


def _text_token_ids(tokenizer: Any, text: str) -> list[int]:
    return _single_token_ids(
        tokenizer(text, add_special_tokens=False), context="plain text"
    )


def _render_prompt(tokenizer: Any, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None):
        # This is the same ordinary one-user template used by Arena training.
        # No system or privileged instruction is inserted.
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return f"USER: {content}\n\nASSISTANT:"


def _prompt_token_ids(tokenizer: Any, rendered: str) -> list[int]:
    return _single_token_ids(
        tokenizer(rendered, add_special_tokens=True), context="rendered prompt"
    )


def _decode_prefix(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))


def bounded_prompt(
    tokenizer: Any, prompt: str, *, max_prompt_tokens: int
) -> tuple[list[int], dict[str, Any]]:
    """Keep the longest user-token prefix whose re-rendered chat fits."""
    if max_prompt_tokens <= 0:
        raise PreferenceScoringError("max_prompt_tokens must be positive")
    content = str(prompt).strip()
    if not content:
        raise PreferenceScoringError("preference prompt cannot be empty")
    user_ids = _text_token_ids(tokenizer, content)
    rendered = _render_prompt(tokenizer, content)
    original_ids = _prompt_token_ids(tokenizer, rendered)
    retained_user_tokens = len(user_ids)
    prompt_ids = original_ids
    applied = len(prompt_ids) > max_prompt_tokens
    if applied:
        low = 0
        high = len(user_ids)
        retained_user_tokens = 0
        empty_ids = _prompt_token_ids(tokenizer, _render_prompt(tokenizer, ""))
        if not empty_ids or len(empty_ids) > max_prompt_tokens:
            raise PreferenceScoringError(
                "chat-template overhead exceeds max_prompt_tokens"
            )
        prompt_ids = empty_ids
        while low <= high:
            middle = (low + high) // 2
            candidate_content = _decode_prefix(tokenizer, user_ids[:middle])
            candidate_ids = _prompt_token_ids(
                tokenizer, _render_prompt(tokenizer, candidate_content)
            )
            if len(candidate_ids) <= max_prompt_tokens:
                retained_user_tokens = middle
                prompt_ids = candidate_ids
                low = middle + 1
            else:
                high = middle - 1
    if not prompt_ids or len(prompt_ids) > max_prompt_tokens:
        raise PreferenceScoringError("could not construct a bounded prompt")
    return prompt_ids, {
        "policy": TRUNCATION_POLICY,
        "applied": applied,
        "original_user_tokens": len(user_ids),
        "retained_user_tokens": retained_user_tokens,
        "removed_user_tokens": len(user_ids) - retained_user_tokens,
        "original_prompt_tokens": len(original_ids),
        "retained_prompt_tokens": len(prompt_ids),
        "prompt_token_limit": max_prompt_tokens,
    }


def bounded_response(
    tokenizer: Any,
    response: str,
    *,
    prompt_token_count: int,
    max_response_tokens: int,
    context_window: int,
) -> tuple[list[int], dict[str, Any]]:
    if max_response_tokens <= 0 or context_window <= 1:
        raise PreferenceScoringError("response/context token limits must be positive")
    original = _text_token_ids(tokenizer, str(response).strip())
    if not original:
        raise PreferenceScoringError("response tokenization is empty")
    available = context_window - prompt_token_count
    if available <= 0:
        raise PreferenceScoringError("prompt leaves no context for response tokens")
    retained_count = min(len(original), max_response_tokens, available)
    retained = original[:retained_count]
    return retained, {
        "policy": TRUNCATION_POLICY,
        "applied": retained_count < len(original),
        "original_response_tokens": len(original),
        "retained_response_tokens": retained_count,
        "removed_response_tokens": len(original) - retained_count,
        "response_token_limit": max_response_tokens,
        "context_available_tokens": available,
    }


def token_chunk_bounds(token_count: int, chunk_size: int) -> list[tuple[int, int]]:
    if token_count <= 0 or chunk_size <= 0:
        raise PreferenceScoringError("token_count and chunk_size must be positive")
    return [
        (start, min(start + chunk_size, token_count))
        for start in range(0, token_count, chunk_size)
    ]


def _score_response(
    *,
    torch: Any,
    model: Any,
    prompt_token_ids: Sequence[int],
    response_token_ids: Sequence[int],
    chunk_size: int,
) -> tuple[float, float]:
    """Return sum and mean log-probability for every response token."""
    from src.clean_self_distill.runtime import (
        backbone_forward,
        input_device,
        project_logits,
    )

    prompt_count = len(prompt_token_ids)
    response_count = len(response_token_ids)
    if prompt_count <= 0 or response_count <= 0:
        raise PreferenceScoringError("scoring requires non-empty prompt/response IDs")
    device = input_device(model)
    scoring_ids = list(prompt_token_ids) + list(response_token_ids[:-1])
    input_ids = torch.tensor([scoring_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden_all, _ = backbone_forward(
            model,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        )
    start = prompt_count - 1
    stop = start + response_count
    response_hidden = hidden_all[:, start:stop].detach()
    del hidden_all, input_ids
    if int(response_hidden.shape[1]) != response_count:
        raise PreferenceScoringError("hidden states do not cover every response token")

    total = 0.0
    for chunk_start, chunk_stop in token_chunk_bounds(response_count, chunk_size):
        labels = torch.tensor(
            [list(response_token_ids[chunk_start:chunk_stop])],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            logits = project_logits(
                model, response_hidden[:, chunk_start:chunk_stop]
            ).float()
            selected = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            log_probs = selected - torch.logsumexp(logits, dim=-1)
        value = float(log_probs.sum().item())
        if not math.isfinite(value):
            raise PreferenceScoringError("model produced non-finite log-probability")
        total += value
        del labels, logits, selected, log_probs
    del response_hidden
    return total, total / response_count


def _validate_shard(num_shards: int, shard_index: int) -> None:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise PreferenceScoringError("invalid shard index/count")


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    _validate_shard(args.num_shards, args.shard_index)
    if args.checkpoint < 0:
        raise PreferenceScoringError("checkpoint must be non-negative")
    adapter: str | None = None
    if args.adapter is not None:
        adapter_path = args.adapter.resolve()
        if not adapter_path.is_dir():
            raise PreferenceScoringError(f"adapter directory does not exist: {adapter_path}")
        adapter = str(adapter_path)
    if args.method == "Base":
        if adapter is not None or args.checkpoint != 0:
            raise PreferenceScoringError("Base must use checkpoint=0 and no adapter")
    elif adapter is None:
        raise PreferenceScoringError("trained methods require --adapter")
    return {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "model_id": args.model_id,
        "revision": args.revision,
        "adapter": adapter,
        "pair_file_sha256": _file_sha256(args.pairs.resolve()),
        "prompt_profile": PROMPT_PROFILE,
        "truncation_policy": TRUNCATION_POLICY,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_response_tokens": args.max_response_tokens,
        "context_window": args.context_window,
        "logit_chunk_size": args.logit_chunk_size,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "normalization": "mean_over_retained_response_tokens_per_answer",
        "include_eos": False,
        "external_llm_judge_used": False,
        "bradley_terry_used": False,
        "shard": {"count": args.num_shards, "index": args.shard_index},
    }


def _expected_pairs(
    pairs: Sequence[Mapping[str, Any]], *, num_shards: int, shard_index: int
) -> list[tuple[int, Mapping[str, Any]]]:
    _validate_shard(num_shards, shard_index)
    return [
        (index, pair)
        for index, pair in enumerate(pairs)
        if index % num_shards == shard_index
    ]


def _load_resume(
    path: Path,
    *,
    identity: Mapping[str, Any],
    expected: Sequence[tuple[int, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [
        validate_score_row(row, row_number=index)
        for index, row in enumerate(read_jsonl(path), 1)
    ]
    if len(rows) > len(expected):
        raise PreferenceScoringError(f"{path} has too many resumable rows")
    for index, row in enumerate(rows):
        global_index, pair = expected[index]
        if (
            row.get("evaluation_identity") != dict(identity)
            or row["global_query_index"] != global_index
            or row["query_id"] != pair["query_id"]
            or row.get("prompt_sha256") != pair["prompt_sha256"]
            or row.get("preferred_response_sha256")
            != pair["preferred_response_sha256"]
            or row.get("rejected_response_sha256")
            != pair["rejected_response_sha256"]
        ):
            raise PreferenceScoringError(
                f"{path} is not an exact prefix of this score run"
            )
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_text(path, "".join(_canonical_json(dict(row)) + "\n" for row in rows))


def score(args: argparse.Namespace) -> dict[str, Any]:
    pairs = load_preference_pairs(args.pairs)
    identity = _identity(args)
    expected = _expected_pairs(
        pairs, num_shards=args.num_shards, shard_index=args.shard_index
    )
    rows = _load_resume(args.output, identity=identity, expected=expected)
    if len(rows) == len(expected):
        summary = summarize_score_rows(rows)
        _atomic_text(
            _summary_path(args.output),
            json.dumps({**summary, "complete": True, "identity": identity}, indent=2, sort_keys=True)
            + "\n",
        )
        print(_canonical_json(summary), flush=True)
        return summary

    # Delay the GPU stack until validation and the already-complete path pass.
    import torch

    from src.clean_self_distill.runtime import load_hf_model

    model_path = args.model.resolve() if args.model.exists() else args.model
    model, tokenizer = load_hf_model(
        str(model_path),
        dtype=args.dtype,
        device_map=args.device_map,
        training=False,
        revision=None if isinstance(model_path, Path) and model_path.exists() else args.revision,
        attn_implementation=args.attn_implementation,
    )
    if identity["adapter"] is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(identity["adapter"]), is_trainable=False
        )
    model.eval()

    for local_index, (global_index, pair) in enumerate(expected[len(rows) :], len(rows)):
        prompt_ids, prompt_truncation = bounded_prompt(
            tokenizer,
            str(pair["prompt"]),
            max_prompt_tokens=args.max_prompt_tokens,
        )
        preferred_ids, preferred_truncation = bounded_response(
            tokenizer,
            str(pair["preferred_response"]),
            prompt_token_count=len(prompt_ids),
            max_response_tokens=args.max_response_tokens,
            context_window=args.context_window,
        )
        rejected_ids, rejected_truncation = bounded_response(
            tokenizer,
            str(pair["rejected_response"]),
            prompt_token_count=len(prompt_ids),
            max_response_tokens=args.max_response_tokens,
            context_window=args.context_window,
        )
        preferred_sum, preferred_mean = _score_response(
            torch=torch,
            model=model,
            prompt_token_ids=prompt_ids,
            response_token_ids=preferred_ids,
            chunk_size=args.logit_chunk_size,
        )
        rejected_sum, rejected_mean = _score_response(
            torch=torch,
            model=model,
            prompt_token_ids=prompt_ids,
            response_token_ids=rejected_ids,
            chunk_size=args.logit_chunk_size,
        )
        margin = preferred_mean - rejected_mean
        row = {
            "schema_version": SCORE_SCHEMA_VERSION,
            "external_judge_used": False,
            "bradley_terry_used": False,
            "evaluation_identity": identity,
            "method": args.method,
            "checkpoint": args.checkpoint,
            "query_id": pair["query_id"],
            "global_query_index": global_index,
            "prompt_sha256": pair["prompt_sha256"],
            "preferred_response_sha256": pair["preferred_response_sha256"],
            "rejected_response_sha256": pair["rejected_response_sha256"],
            "domains": pair["domains"],
            "prompt_token_count": len(prompt_ids),
            "preferred_token_count": len(preferred_ids),
            "rejected_token_count": len(rejected_ids),
            "preferred_logprob_sum": preferred_sum,
            "rejected_logprob_sum": rejected_sum,
            "preferred_mean_logprob": preferred_mean,
            "rejected_mean_logprob": rejected_mean,
            "preference_margin": margin,
            "preference_correct": preferred_mean > rejected_mean,
            "prompt_truncation": prompt_truncation,
            "preferred_truncation": preferred_truncation,
            "rejected_truncation": rejected_truncation,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        }
        rows.append(validate_score_row(row, row_number=global_index + 1))
        completed = local_index + 1
        if completed % args.write_every == 0 or completed == len(expected):
            _write_rows(args.output, rows)
        if completed % args.progress_every == 0 or completed == len(expected):
            print(
                _canonical_json(
                    {
                        "method": args.method,
                        "checkpoint": args.checkpoint,
                        "shard": args.shard_index,
                        "complete": completed,
                        "total": len(expected),
                    }
                ),
                flush=True,
            )

    summary = summarize_score_rows(rows)
    _atomic_text(
        _summary_path(args.output),
        json.dumps({**summary, "complete": True, "identity": identity}, indent=2, sort_keys=True)
        + "\n",
    )
    print(_canonical_json(summary), flush=True)
    return summary


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    pairs = load_preference_pairs(args.pairs)
    all_rows: list[dict[str, Any]] = []
    identities: list[Mapping[str, Any]] = []
    for path in args.input:
        rows = [
            validate_score_row(row, row_number=index)
            for index, row in enumerate(read_jsonl(path), 1)
        ]
        identity = rows[0].get("evaluation_identity")
        if not isinstance(identity, Mapping):
            raise PreferenceScoringError(f"{path} lacks evaluation_identity")
        identities.append(identity)
        all_rows.extend(rows)
    common = [{key: value for key, value in identity.items() if key != "shard"} for identity in identities]
    if not common or any(identity != common[0] for identity in common[1:]):
        raise PreferenceScoringError("score shards mix evaluation identities")
    indices = [int(row["global_query_index"]) for row in all_rows]
    if len(indices) != len(set(indices)):
        raise PreferenceScoringError("score shards overlap")
    if set(indices) != set(range(len(pairs))):
        missing = sorted(set(range(len(pairs))) - set(indices))[:10]
        extra = sorted(set(indices) - set(range(len(pairs))))[:10]
        raise PreferenceScoringError(
            f"score shards do not cover pairs: missing={missing}, extra={extra}"
        )
    all_rows.sort(key=lambda row: int(row["global_query_index"]))
    for index, (row, pair) in enumerate(zip(all_rows, pairs, strict=True)):
        if row["query_id"] != pair["query_id"] or row["global_query_index"] != index:
            raise PreferenceScoringError("merged score order disagrees with pair file")
    _write_rows(args.output, all_rows)
    summary = summarize_score_rows(all_rows)
    aggregate_identity = {
        **common[0],
        "shards_aggregated": len(args.input),
        "input_paths": [str(path.resolve()) for path in args.input],
    }
    _atomic_text(
        _summary_path(args.output),
        json.dumps(
            {**summary, "complete": True, "identity": aggregate_identity},
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(_canonical_json(summary), flush=True)
    return summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score one model checkpoint")
    score_parser.add_argument("--pairs", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--model", type=Path, required=True)
    score_parser.add_argument("--model-id", required=True)
    score_parser.add_argument("--revision", required=True)
    score_parser.add_argument("--method", required=True)
    score_parser.add_argument("--checkpoint", type=int, required=True)
    score_parser.add_argument("--adapter", type=Path)
    score_parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    score_parser.add_argument("--max-response-tokens", type=int, default=4096)
    score_parser.add_argument("--context-window", type=int, default=8192)
    score_parser.add_argument("--logit-chunk-size", type=int, default=32)
    score_parser.add_argument("--num-shards", type=int, default=1)
    score_parser.add_argument("--shard-index", type=int, default=0)
    score_parser.add_argument("--write-every", type=int, default=1)
    score_parser.add_argument("--progress-every", type=int, default=10)
    score_parser.add_argument("--dtype", default="bfloat16")
    score_parser.add_argument("--device-map", default="auto")
    score_parser.add_argument("--attn-implementation", default="sdpa")

    aggregate_parser = subparsers.add_parser("aggregate", help="merge score shards")
    aggregate_parser.add_argument("--pairs", type=Path, required=True)
    aggregate_parser.add_argument("--input", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "score":
        if args.write_every <= 0 or args.progress_every <= 0:
            raise PreferenceScoringError("write/progress intervals must be positive")
        score(args)
    else:
        aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
