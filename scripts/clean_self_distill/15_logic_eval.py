#!/usr/bin/env python3
"""Run and report full SATQuest + LogicSkills adapter evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from src.clean_self_distill.logic_evaluation import (
    extract_binary_answer,
    grouped_accuracy,
    score_logicskills,
)


METHOD_LABELS = {
    "base": "Base",
    "privileged_sd_16": "Privilege-SD 16",
    "trsd_16": "TRSD 16",
    "privileged_sd_64": "Privilege-SD 64",
    "trsd_64": "TRSD 64",
}


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(value)
    return rows


def parse_shard_indices(value: str | None, num_shards: int) -> tuple[int, ...]:
    """Parse an optional comma-separated subset of source shard indices."""
    if value is None:
        return tuple(range(num_shards))
    try:
        indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError(f"Invalid shard indices: {value!r}") from error
    if not indices:
        raise ValueError("At least one shard index is required")
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate shard indices: {indices}")
    if any(index < 0 or index >= num_shards for index in indices):
        raise ValueError(
            f"Shard indices must be in [0, {num_shards - 1}]: {indices}"
        )
    return indices


def load_benchmark_rows(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in ("satquest/val.parquet", "logicskills/val.parquet"):
        path = data_root / relative
        source_sha256 = sha256_file(path)
        for local_index, source in enumerate(pq.read_table(path).to_pylist()):
            extra = source["extra_info"]
            rows.append(
                {
                    "global_index": len(rows),
                    "local_index": local_index,
                    "query_id": extra["index"],
                    "dataset": source["data_source"],
                    "problem": source["prompt"][0]["content"],
                    "target": source["reward_model"]["ground_truth"],
                    "style": source["reward_model"]["style"],
                    "metadata": extra,
                    "source_sha256": source_sha256,
                }
            )
    if len(rows) != 4860 or len({row["query_id"] for row in rows}) != 4860:
        raise ValueError("Expected 4,860 unique full-benchmark queries")
    return rows


def validate_adapter(
    adapter: Path,
    *,
    method: str,
    model_id: str,
    revision: str,
    expected_branch: str | None = None,
    expected_method_id: str | None = None,
    checkpoint_episode: int | None = None,
) -> int:
    manifest_path = adapter / "checkpoint_manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    known_method = method.startswith("privileged_sd_") or method.startswith("trsd_")
    if known_method:
        expected_branch = expected_branch or (
            "privileged" if method.startswith("privileged_sd_") else "clean"
        )
        expected_method_id = expected_method_id or (
            "privileged:predecision_method"
            if method.startswith("privileged_sd_")
            else "trsd:exponential_teacher_projection"
        )
        checkpoint_episode = checkpoint_episode or int(method.rsplit("_", 1)[-1])
    if expected_branch is None or expected_method_id is None or checkpoint_episode is None:
        raise ValueError(
            "Custom adapter methods require --expected-branch, "
            "--expected-method-id, and --checkpoint-episode"
        )
    expected = {
        "schema_version": "clean-self-distill-persistent-checkpoint-v1",
        "branch": expected_branch,
        "checkpoint_episode": checkpoint_episode,
        "completed_episodes": checkpoint_episode,
        "method_id": expected_method_id,
        "model_id": model_id,
        "model_revision": revision,
    }
    mismatches = {key: (value.get(key), wanted) for key, wanted in expected.items() if value.get(key) != wanted}
    if mismatches:
        raise ValueError(f"Adapter identity mismatch at {manifest_path}: {mismatches}")
    return int(checkpoint_episode)


def render_prompt(tokenizer: Any, problem: str) -> str:
    messages = [{"role": "user", "content": problem.strip()}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from src.clean_self_distill.runtime import collect_runtime_metadata

    benchmark_rows = load_benchmark_rows(args.data_root)
    selected = [row for row in benchmark_rows if row["global_index"] % args.num_shards == args.shard_index]
    existing = read_jsonl(args.output)
    expected_ids = [row["query_id"] for row in selected]
    if [row.get("query_id") for row in existing] != expected_ids[: len(existing)]:
        raise ValueError(f"Existing prediction prefix does not match shard {args.shard_index}")
    if any(row.get("method") != args.method for row in existing):
        raise ValueError("Existing predictions use a different method")
    if len(existing) == len(selected):
        print(json.dumps({"status": "complete", "rows": len(existing), "output": str(args.output)}))
        return

    if args.method == "base":
        if args.adapter is not None:
            raise ValueError("Base evaluation does not accept an adapter")
    else:
        if args.adapter is None:
            raise ValueError(f"{args.method} requires --adapter")
        checkpoint_episode = validate_adapter(
            args.adapter,
            method=args.method,
            model_id=args.model_id,
            revision=args.revision,
            expected_branch=args.expected_branch,
            expected_method_id=args.expected_method_id,
            checkpoint_episode=args.checkpoint_episode,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True
    )
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_prompt_tokens + args.max_new_tokens,
        gpu_memory_utilization=0.88,
        enable_lora=args.adapter is not None,
        max_lora_rank=8,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=20260808,
    )
    lora_request = (
        None
        if args.adapter is None
        else LoRARequest(args.method, 1, str(args.adapter))
    )
    runtime = collect_runtime_metadata(None, model_path=args.model_id, revision=args.revision)
    runtime.update(
        {
            "inference_engine": "vllm",
            "vllm_model_path": str(args.model),
            "vllm_tensor_parallel_size": args.tensor_parallel_size,
        }
    )
    output_rows = list(existing)
    pending = selected[len(existing) :]
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        prompts = [render_prompt(tokenizer, row["problem"]) for row in batch]
        prompt_tokens = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
        if max(prompt_tokens) > args.max_prompt_tokens:
            offenders = [row["query_id"] for row, length in zip(batch, prompt_tokens) if length > args.max_prompt_tokens]
            raise ValueError(f"Prompts exceed {args.max_prompt_tokens} tokens: {offenders}")
        started = time.perf_counter()
        generated = llm.generate(
            prompts,
            sampling,
            use_tqdm=True,
            lora_request=lora_request,
        )
        elapsed = time.perf_counter() - started
        for row, result, prompt_length in zip(batch, generated, prompt_tokens):
            candidate = result.outputs[0]
            response = candidate.text.strip()
            output_rows.append(
                {
                    "schema_version": "trsd-logic-prediction-v1",
                    "method": args.method,
                    "checkpoint_episode": (
                        0 if args.method == "base" else checkpoint_episode
                    ),
                    "query_id": row["query_id"],
                    "global_index": row["global_index"],
                    "dataset": row["dataset"],
                    "prompt_tokens": int(prompt_length),
                    "generated_tokens": len(candidate.token_ids),
                    "max_new_tokens": args.max_new_tokens,
                    "response": response,
                    "batch_generation_seconds": elapsed,
                    "batch_size": len(batch),
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "runtime": runtime,
                }
            )
        atomic_jsonl(args.output, output_rows)
        print(
            json.dumps(
                {
                    "method": args.method,
                    "shard": args.shard_index,
                    "complete": len(output_rows),
                    "total": len(selected),
                    "last_batch_seconds": round(elapsed, 3),
                }
            ),
            flush=True,
        )


def score(args: argparse.Namespace) -> None:
    benchmark = {row["query_id"]: row for row in load_benchmark_rows(args.data_root)}
    predictions = read_jsonl(args.predictions)
    expected = [
        row for row in benchmark.values() if row["global_index"] % args.num_shards == args.shard_index
    ]
    if [row["query_id"] for row in predictions] != [row["query_id"] for row in expected]:
        raise ValueError("Predictions are not the complete ordered shard")
    scored: list[dict[str, Any]] = []
    for prediction in predictions:
        source = benchmark[prediction["query_id"]]
        metadata = source["metadata"]
        payload = json.loads(metadata["verifier_payload"])
        extracted: str | None
        if source["dataset"] == "satquest":
            import sys

            for path in (args.satquest_deps, args.satquest_repo):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            from satquest import CNF, create_problem

            problem = create_problem(metadata["problem_type"], CNF(dimacs=metadata["cnf_dimacs"]))
            required_length = 1 if metadata["problem_type"].startswith("SATDP") else (
                int(metadata["num_variable"])
                if metadata["problem_type"] in {"SATSP", "MaxSAT"}
                else int(metadata["num_clause"])
            )
            extracted = extract_binary_answer(prediction["response"], required_length)
            correct = extracted is not None and bool(problem.check(extracted))
        else:
            correct, extracted = score_logicskills(
                prediction["response"],
                task=metadata["task"],
                target=source["target"],
                payload=payload,
                logic_repo=args.logicskills_repo,
            )
        scored.append(
            {
                **{key: value for key, value in prediction.items() if key != "runtime"},
                "schema_version": "trsd-logic-scored-v1",
                "correct": bool(correct),
                "extracted_answer": extracted,
                "eval_regime": metadata["eval_regime"],
                "problem_type": metadata.get("problem_type"),
                "question_type": metadata.get("question_type"),
                "num_variable": int(metadata["num_variable"]) if metadata.get("num_variable") else None,
                "task": metadata.get("task"),
                "language": metadata.get("language"),
                "verifier": payload["verifier"],
            }
        )
    atomic_jsonl(args.output, scored)
    print(json.dumps({"scored": len(scored), "correct": sum(row["correct"] for row in scored), "output": str(args.output)}))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def report(args: argparse.Namespace) -> None:
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    if not methods or any(method not in METHOD_LABELS for method in methods):
        raise ValueError(f"Invalid report methods: {methods}")
    if "base" not in methods:
        raise ValueError("Logical report methods must include base")
    shard_indices = parse_shard_indices(args.shard_indices, args.num_shards)
    all_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for method in methods:
        for shard in shard_indices:
            path = args.run_root / "scored" / method / f"shard_{shard:02d}.jsonl"
            rows = read_jsonl(path)
            expected_rows = len(range(shard, 4860, args.num_shards))
            if len(rows) != expected_rows:
                missing.append(str(path))
            all_rows.extend(rows)
    if missing:
        print(json.dumps({"status": "waiting", "incomplete": missing}, indent=2))
        return
    keys = {(row["method"], row["query_id"]) for row in all_rows}
    rows_per_method = sum(len(range(shard, 4860, args.num_shards)) for shard in shard_indices)
    if len(keys) != len(methods) * rows_per_method:
        raise ValueError("Merged scored rows are not complete for every requested method")
    query_ids_by_method = {
        method: {row["query_id"] for row in all_rows if row["method"] == method}
        for method in methods
    }
    if any(query_ids_by_method[method] != query_ids_by_method["base"] for method in methods):
        raise ValueError("Methods do not contain the same evaluated query subset")
    generation_budgets = {int(row["max_new_tokens"]) for row in all_rows}
    if len(generation_budgets) != 1:
        raise ValueError(f"Methods do not share one generation budget: {generation_budgets}")
    summary = grouped_accuracy(all_rows, ("method", "dataset"))
    detail = grouped_accuracy(
        all_rows,
        ("method", "dataset", "eval_regime", "problem_type", "question_type", "task", "language"),
    )
    overall = grouped_accuracy(all_rows, ("method",))
    base_rows = [row for row in all_rows if row["method"] == "base"]
    dataset_counts = Counter(row["dataset"] for row in base_rows)
    base_accuracy = {row["dataset"]: row["accuracy"] for row in summary if row["method"] == "base"}
    for row in summary:
        row["delta_vs_base"] = row["accuracy"] - base_accuracy[row["dataset"]]
        row["method_label"] = METHOD_LABELS[row["method"]]
    for row in overall:
        row["method_label"] = METHOD_LABELS[row["method"]]
    result = {
        "schema_version": "trsd-logic-report-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "model": args.model_label,
            "methods": {method: METHOD_LABELS[method] for method in methods},
            "satquest_rows_per_method": dataset_counts["satquest"],
            "logicskills_rows_per_method": dataset_counts["logicskills"],
            "total_rows_per_method": rows_per_method,
            "generation": "greedy pass@1 with the model-native chat template",
            "max_new_tokens": next(iter(generation_budgets)),
            "scoring": "official PySAT/Z3 verifier semantics with final-channel-only answer extraction",
            "num_source_shards": args.num_shards,
            "evaluated_shard_indices": list(shard_indices),
            "partial_evaluation": len(shard_indices) != args.num_shards,
        },
        "overall": overall,
        "dataset_summary": summary,
        "detail": detail,
    }
    reports = args.run_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_text(reports / "logic_results.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_csv(reports / "logic_dataset_summary.csv", summary)
    write_csv(reports / "logic_detailed_slices.csv", detail)
    atomic_jsonl(reports / "all_scored.jsonl", sorted(all_rows, key=lambda row: (row["method"], row["global_index"])))

    scope = (
        "full verifier-backed evaluation"
        if len(shard_indices) == args.num_shards
        else f"partial verifier-backed evaluation ({len(shard_indices)}/{args.num_shards} shards)"
    )
    lines = [
        f"# {args.model_label} logical reasoning: {scope}",
        "",
        f"| Method | SATQuest | Delta | LogicSkills external OOD | Delta | All {rows_per_method:,} |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_method_dataset = {(row["method"], row["dataset"]): row for row in summary}
    by_method_all = {row["method"]: row for row in overall}
    for method in methods:
        label = METHOD_LABELS[method]
        sat = by_method_dataset[(method, "satquest")]
        logic = by_method_dataset[(method, "logicskills")]
        total = by_method_all[method]
        lines.append(
            f"| {label} | {sat['accuracy']:.2%} | {sat['delta_vs_base']:+.2%} | "
            f"{logic['accuracy']:.2%} | {logic['delta_vs_base']:+.2%} | {total['accuracy']:.2%} |"
        )
    lines.extend(
        [
            "",
            (
                f"This table evaluates source shards {','.join(str(index) for index in shard_indices)} "
                f"({dataset_counts['satquest']:,} SATQuest and "
                f"{dataset_counts['logicskills']:,} LogicSkills examples per method). "
                "Every score is verifier-derived; omitted shards are not included in denominators."
            ),
            "",
        ]
    )
    atomic_text(reports / "LOGIC_RESULTS.md", "\n".join(lines))
    atomic_text(args.run_root / "LOGIC_EVAL_COMPLETE", datetime.now(timezone.utc).isoformat() + "\n")
    print(json.dumps({"status": "complete", "report": str(reports / "LOGIC_RESULTS.md")}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate")
    generation.add_argument("--data-root", type=Path, required=True)
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--model", type=Path, required=True)
    generation.add_argument("--model-id", required=True)
    generation.add_argument("--revision", required=True)
    generation.add_argument("--method", required=True)
    generation.add_argument("--adapter", type=Path)
    generation.add_argument("--expected-branch")
    generation.add_argument("--expected-method-id")
    generation.add_argument("--checkpoint-episode", type=int)
    generation.add_argument("--num-shards", type=int, default=2)
    generation.add_argument("--shard-index", type=int, required=True)
    generation.add_argument("--batch-size", type=int, default=512)
    generation.add_argument("--max-new-tokens", type=int, default=512)
    generation.add_argument("--max-prompt-tokens", type=int, default=8192)
    generation.add_argument("--tensor-parallel-size", type=int, default=1)
    scoring = commands.add_parser("score")
    scoring.add_argument("--data-root", type=Path, required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--satquest-repo", type=Path, required=True)
    scoring.add_argument("--satquest-deps", type=Path, required=True)
    scoring.add_argument("--logicskills-repo", type=Path, required=True)
    scoring.add_argument("--num-shards", type=int, default=2)
    scoring.add_argument("--shard-index", type=int, required=True)
    reporting = commands.add_parser("report")
    reporting.add_argument("--run-root", type=Path, required=True)
    reporting.add_argument("--num-shards", type=int, default=2)
    reporting.add_argument(
        "--shard-indices",
        help="Optional comma-separated source shard subset; defaults to every shard",
    )
    reporting.add_argument(
        "--methods",
        default="base,privileged_sd_64,trsd_64",
        help="Comma-separated completed methods to merge",
    )
    reporting.add_argument("--model-label", default="Qwen3-8B")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    {"generate": generate, "score": score, "report": report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
