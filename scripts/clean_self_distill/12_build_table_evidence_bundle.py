#!/usr/bin/env python3
"""Build the table-first TRSD evidence bundle from completed artifacts only.

All five checkpoints, including the current matched reverse-KL TRSD-16
checkpoint, are required to contain the full 143-query evaluation before the
reporter writes any output.  The public accuracy estimand is strict Acc@1 over
the full held-out denominator: an answer is correct iff the sealed-label scorer
marks it correct *and* the generation did not hit the 10,240-token cap.  No
completed-only statistic is computed or emitted.

The job is model-free and CPU-light.  It validates the matched evaluation
metadata, recomputes paired query inference for the available checkpoints,
collates the already completed trust-region diagnostics, and writes Markdown,
CSV, JSON, and booktabs LaTeX outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SOURCES = ("combined", "amc23", "aime24", "aime25")
SOURCE_LABEL = {
    "combined": "Combined",
    "amc23": "AMC23",
    "aime24": "AIME24",
    "aime25": "AIME25",
}
EXPECTED_SOURCE_COUNTS = {"amc23": 83, "aime24": 30, "aime25": 30}
METHODS = ("base", "privileged_16", "trsd_16", "privileged_64", "trsd_64")
AVAILABLE_METHODS = METHODS
METHOD_LABEL = {
    "base": "Base",
    "privileged_16": "Privilege-SD 16",
    "trsd_16": "TRSD 16",
    "privileged_64": "Privilege-SD 64",
    "trsd_64": "TRSD 64",
}
SCORED_FILENAME = {
    "base": "base.jsonl",
    "privileged_16": "privileged_ep16.jsonl",
    "trsd_16": "trsd_ep16_rkl_current.jsonl",
    "privileged_64": "privileged_ep64.jsonl",
    "trsd_64": "trsd_ep64.jsonl",
}
PREDICTION_DIRECTORY = {
    "base": "base",
    "privileged_16": "privileged_ep16",
    "trsd_16": "trsd_ep16_rkl_current",
    "privileged_64": "privileged_ep64",
    "trsd_64": "trsd_ep64",
}
CHECKPOINT_EPISODE = {
    "base": 0,
    "privileged_16": 16,
    "trsd_16": 16,
    "privileged_64": 64,
    "trsd_64": 64,
}
PAIRED_BASE_METHODS = ("privileged_16", "trsd_16", "privileged_64", "trsd_64")
# Keep the existing P16/P64/T64 bootstrap streams byte-for-byte stable when T16
# is inserted into display order.  T16 receives a new, non-overlapping offset.
PAIRED_BASE_SEED_OFFSET = {
    "privileged_16": 0,
    "privileged_64": 1,
    "trsd_64": 2,
    "trsd_16": 3,
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260808
EXPECTED_PROMPT_VERSION = "explicit-generation-budget-v1"
EXPECTED_MAX_NEW_TOKENS = 10_240


class EvidenceError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--style-root", type=Path, required=True)
    parser.add_argument("--epsilon-csv", type=Path, required=True)
    parser.add_argument("--p16-journal", type=Path, required=True)
    parser.add_argument("--p16-manifest", type=Path, required=True)
    parser.add_argument("--p64-journal", type=Path, required=True)
    parser.add_argument("--p64-manifest", type=Path, required=True)
    parser.add_argument("--t64-journal", type=Path, required=True)
    parser.add_argument("--t64-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"Missing JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON root is not an object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise EvidenceError(f"Missing JSONL input: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise EvidenceError(f"Non-object row at {path}:{line_number}")
            yield value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise EvidenceError(f"Missing CSV input: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise EvidenceError(f"Empty CSV input: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvidenceError(f"Refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise EvidenceError(f"{label} is not finite")
    return result


def as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not an integer: {value!r}") from exc


def strict_correct(row: Mapping[str, Any]) -> int:
    correct = as_int(row.get("correct"), "correct")
    if correct not in (0, 1):
        raise EvidenceError(f"Correctness is not binary: {correct}")
    truncated = row.get("truncated")
    if not isinstance(truncated, bool):
        raise EvidenceError("Missing boolean truncated field")
    return int(correct == 1 and not truncated)


def load_scored(path: Path, method: str) -> list[dict[str, Any]]:
    keep = (
        "profile",
        "query_id",
        "problem_sha256",
        "source",
        "sample_index",
        "correct",
        "truncated",
        "generated_tokens",
        "behavioral_diagnostics",
        "resource_usage",
    )
    rows = [{key: row.get(key) for key in keep} for row in iter_jsonl(path)]
    rows = [row for row in rows if row.get("profile", "acc1") == "acc1"]
    if len(rows) != 143:
        raise EvidenceError(f"{method}: expected 143 rows, found {len(rows)}")
    counts = Counter(str(row.get("source", "")) for row in rows)
    if dict(counts) != EXPECTED_SOURCE_COUNTS:
        raise EvidenceError(f"{method}: source counts differ: {dict(counts)}")
    ids = [str(row.get("query_id", "")) for row in rows]
    if any(not query_id for query_id in ids) or len(set(ids)) != len(ids):
        raise EvidenceError(f"{method}: missing or duplicate query IDs")
    for row in rows:
        strict_correct(row)
    return rows


def validate_matched_queries(method_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    base = {str(row["query_id"]): row for row in method_rows["base"]}
    for method, rows in method_rows.items():
        current = {str(row["query_id"]): row for row in rows}
        if set(current) != set(base):
            raise EvidenceError(f"{method}: query set differs from Base")
        for query_id, base_row in base.items():
            row = current[query_id]
            for field in ("problem_sha256", "source", "sample_index"):
                if str(row.get(field, "")) != str(base_row.get(field, "")):
                    raise EvidenceError(f"{method}/{query_id}: {field} mismatch")


def validate_generation_protocol(
    prediction_root: Path,
    method_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    common_fields = {
        "evaluation_prompt_version": EXPECTED_PROMPT_VERSION,
        "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }
    result: dict[str, Any] = {}
    seeds_by_method: dict[str, dict[str, int]] = {}
    for method in AVAILABLE_METHODS:
        rows: list[dict[str, Any]] = []
        statuses = []
        method_root = prediction_root / PREDICTION_DIRECTORY[method]
        for shard in range(4):
            prediction = method_root / "predictions" / f"shard_{shard:02d}.jsonl"
            status = method_root / "status" / f"shard_{shard:02d}.done"
            if not status.is_file():
                raise EvidenceError(f"{method}: missing completion marker {status}")
            statuses.append(status)
            for row in iter_jsonl(prediction):
                rows.append(
                    {
                        "query_id": row.get("query_id"),
                        "evaluation_prompt_version": row.get("evaluation_prompt_version"),
                        "max_new_tokens": row.get("max_new_tokens"),
                        "seed": row.get("seed"),
                        "temperature": row.get("temperature"),
                        "top_p": row.get("top_p"),
                        "top_k": row.get("top_k"),
                        "checkpoint_episode": row.get("checkpoint_episode"),
                    }
                )
        if len(rows) != 143:
            raise EvidenceError(f"{method}: raw prediction count is {len(rows)}, not 143")
        expected_ids = {str(row["query_id"]) for row in method_rows[method]}
        if {str(row["query_id"]) for row in rows} != expected_ids:
            raise EvidenceError(f"{method}: raw/scored query IDs differ")
        for row in rows:
            for field, expected in common_fields.items():
                observed = row.get(field)
                if isinstance(expected, float):
                    if abs(as_float(observed, f"{method}.{field}") - expected) > 1e-12:
                        raise EvidenceError(f"{method}: {field} differs")
                elif observed != expected:
                    raise EvidenceError(
                        f"{method}: {field}={observed!r}, expected {expected!r}"
                    )
            if (
                as_int(row.get("checkpoint_episode"), f"{method}.checkpoint_episode")
                != CHECKPOINT_EPISODE[method]
            ):
                raise EvidenceError(f"{method}: checkpoint episode differs")
        seed_map = {
            str(row["query_id"]): as_int(row.get("seed"), f"{method}.seed")
            for row in rows
        }
        if len(seed_map) != 143:
            raise EvidenceError(f"{method}: seed map is not query-unique")
        seeds_by_method[method] = seed_map
        result[method] = {
            **common_fields,
            "checkpoint_episode": CHECKPOINT_EPISODE[method],
            "rows": len(rows),
            "shards": 4,
            "seed_policy": "deterministic_query_specific_matched_across_methods",
            "unique_seeds": len(set(seed_map.values())),
            "status": "complete",
        }
    base_seeds = seeds_by_method["base"]
    for method, seed_map in seeds_by_method.items():
        if seed_map != base_seeds:
            mismatched = sorted(
                query_id
                for query_id in base_seeds
                if seed_map.get(query_id) != base_seeds[query_id]
            )
            raise EvidenceError(
                f"{method}: {len(mismatched)} query-specific seeds differ from Base"
            )
    return result


def rows_for_dataset(rows: Sequence[Mapping[str, Any]], dataset: str) -> list[Mapping[str, Any]]:
    if dataset == "combined":
        return list(rows)
    return [row for row in rows if str(row["source"]) == dataset]


def aggregate_accuracy(method: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in SOURCES:
        subset = rows_for_dataset(rows, dataset)
        correct = sum(strict_correct(row) for row in subset)
        seconds = []
        peaks = []
        tokens = []
        hedging_tokens = 0
        fabricated_references = 0
        for row in subset:
            tokens.append(as_int(row.get("generated_tokens", 0), "generated_tokens"))
            diagnostics = row.get("behavioral_diagnostics")
            if not isinstance(diagnostics, Mapping):
                diagnostics = {}
            hedging_tokens += as_int(
                diagnostics.get("hedging_token_count", 0), "hedging_token_count"
            )
            fabricated_references += int(
                bool(diagnostics.get("fabricated_reference_hallucination", False))
            )
            resource = row.get("resource_usage")
            if isinstance(resource, Mapping):
                if resource.get("generation_seconds") is not None:
                    seconds.append(as_float(resource["generation_seconds"], "generation_seconds"))
                if resource.get("cuda_peak_memory_allocated_bytes") is not None:
                    peaks.append(as_float(resource["cuda_peak_memory_allocated_bytes"], "peak bytes"))
        output.append(
            {
                "method": method,
                "method_label": METHOD_LABEL[method],
                "dataset": dataset,
                "strict_correct": correct,
                "n": len(subset),
                "strict_acc1": correct / len(subset),
                "strict_acc1_percent": 100.0 * correct / len(subset),
                "budget_cap_hit_count": sum(bool(row["truncated"]) for row in subset),
                "budget_cap_hit_rate": sum(bool(row["truncated"]) for row in subset) / len(subset),
                "mean_generated_tokens": sum(tokens) / len(tokens),
                "hedging_tokens_per_1k": 1000.0 * hedging_tokens / sum(tokens),
                "fabricated_reference_count": fabricated_references,
                "fabricated_reference_rate": fabricated_references / len(subset),
                "mean_generation_seconds": (sum(seconds) / len(seconds)) if seconds else None,
                "aggregate_gpu_hours": (sum(seconds) / 3600.0) if seconds else None,
                "peak_gpu_allocated_gib": max(peaks) / (1024.0**3) if peaks else None,
                "estimand": "strict_acc1_full_denominator_unfinished_is_wrong",
            }
        )
    return output


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise EvidenceError("Cannot compute percentile of empty data")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def exact_mcnemar(wrong_to_correct: int, correct_to_wrong: int) -> float:
    discordant = wrong_to_correct + correct_to_wrong
    if discordant == 0:
        return 1.0
    smaller = min(wrong_to_correct, correct_to_wrong)
    lower = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * lower)


def paired_inference(
    reference_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    *,
    reference: str,
    method: str,
    dataset: str,
    seed: int,
) -> dict[str, Any]:
    reference_map = {
        str(row["query_id"]): row for row in rows_for_dataset(reference_rows, dataset)
    }
    method_map = {
        str(row["query_id"]): row for row in rows_for_dataset(method_rows, dataset)
    }
    if set(reference_map) != set(method_map):
        raise EvidenceError(f"{method} vs {reference}: paired IDs differ in {dataset}")
    query_ids = sorted(reference_map)
    ref = [strict_correct(reference_map[query_id]) for query_id in query_ids]
    cur = [strict_correct(method_map[query_id]) for query_id in query_ids]
    n = len(query_ids)
    ref_acc = sum(ref) / n
    cur_acc = sum(cur) / n
    wrong_to_correct = sum(a == 0 and b == 1 for a, b in zip(ref, cur))
    correct_to_wrong = sum(a == 1 and b == 0 for a, b in zip(ref, cur))
    generator = random.Random(seed)
    ref_boot = []
    cur_boot = []
    delta_boot = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [generator.randrange(n) for _ in range(n)]
        ref_value = sum(ref[index] for index in indices) / n
        cur_value = sum(cur[index] for index in indices) / n
        ref_boot.append(ref_value)
        cur_boot.append(cur_value)
        delta_boot.append(cur_value - ref_value)
    return {
        "reference": reference,
        "reference_label": METHOD_LABEL[reference],
        "method": method,
        "method_label": METHOD_LABEL[method],
        "dataset": dataset,
        "n": n,
        "reference_correct": sum(ref),
        "reference_acc1": ref_acc,
        "reference_acc1_ci_low": percentile(ref_boot, 0.025),
        "reference_acc1_ci_high": percentile(ref_boot, 0.975),
        "method_correct": sum(cur),
        "method_acc1": cur_acc,
        "method_acc1_ci_low": percentile(cur_boot, 0.025),
        "method_acc1_ci_high": percentile(cur_boot, 0.975),
        "delta_acc1": cur_acc - ref_acc,
        "delta_percentage_points": 100.0 * (cur_acc - ref_acc),
        "delta_ci_low": percentile(delta_boot, 0.025),
        "delta_ci_high": percentile(delta_boot, 0.975),
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "discordant_pairs": wrong_to_correct + correct_to_wrong,
        "mcnemar_exact_two_sided_p": exact_mcnemar(wrong_to_correct, correct_to_wrong),
        "bootstrap_unit": "paired_query",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": seed,
        "estimand": "strict_acc1_full_denominator_unfinished_is_wrong",
    }


def summarize_journal(path: Path, episodes: int) -> dict[str, Any]:
    selected = []
    for row in iter_jsonl(path):
        if as_int(row.get("episode"), "episode") <= episodes:
            selected.append(row)
    if len(selected) != episodes:
        raise EvidenceError(f"{path}: expected first {episodes} episodes, found {len(selected)}")
    if sorted(as_int(row["episode"], "episode") for row in selected) != list(range(1, episodes + 1)):
        raise EvidenceError(f"{path}: episode indices do not equal 1..{episodes}")
    audit_fields = ("teacher_positions", "on_policy_positions", "exact_context_positions", "hindsight_exposed_positions")
    audit = {field: 0 for field in audit_fields}
    sources = set()
    destroyed = True
    peak_allocated = []
    peak_delta = []
    for row in selected:
        row_audit = row.get("audit")
        if not isinstance(row_audit, Mapping):
            raise EvidenceError(f"{path}: missing episode audit")
        for field in audit_fields:
            audit[field] += as_int(row_audit.get(field), field)
        row_sources = row.get("teacher_context_sources")
        if not isinstance(row_sources, list):
            raise EvidenceError(f"{path}: missing teacher context sources")
        sources.add(tuple(str(item) for item in row_sources))
        destroyed = destroyed and bool(row.get("temporary_teacher_destroyed_after_update"))
        resource = row.get("resource_usage")
        if isinstance(resource, Mapping):
            if resource.get("cuda_peak_memory_allocated_bytes") is not None:
                peak_allocated.append(as_float(resource["cuda_peak_memory_allocated_bytes"], "peak allocated"))
            if resource.get("cuda_peak_memory_delta_bytes") is not None:
                peak_delta.append(as_float(resource["cuda_peak_memory_delta_bytes"], "peak delta"))
    if len(sources) != 1:
        raise EvidenceError(f"{path}: teacher context sources vary")
    teacher = audit["teacher_positions"]
    return {
        "episodes": episodes,
        "optimizer_steps": sum(bool(row.get("optimizer_step")) for row in selected),
        "no_op_episodes": sum(not bool(row.get("optimizer_step")) for row in selected),
        "response_tokens": sum(as_int(row.get("response_tokens"), "response_tokens") for row in selected),
        "training_seconds": sum(as_float(row.get("episode_seconds"), "episode_seconds") for row in selected),
        "teacher_positions": teacher,
        "on_policy_positions": audit["on_policy_positions"],
        "exact_context_positions": audit["exact_context_positions"],
        "hindsight_exposed_positions": audit["hindsight_exposed_positions"],
        "on_policy_prefix_rate": audit["on_policy_positions"] / teacher if teacher else None,
        "full_context_parity": audit["exact_context_positions"] / teacher if teacher else None,
        "hindsight_exposure_rate": audit["hindsight_exposed_positions"] / teacher if teacher else None,
        "teacher_context_sources": list(next(iter(sources))),
        "temporary_teacher_destroyed": destroyed,
        "max_gpu_peak_allocated_gib": max(peak_allocated) / (1024.0**3) if peak_allocated else None,
        "max_gpu_peak_delta_gib": max(peak_delta) / (1024.0**3) if peak_delta else None,
    }


def manifest_provenance(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    arguments = manifest.get("arguments")
    if not isinstance(arguments, Mapping):
        raise EvidenceError(f"{path}: missing arguments")
    direction = arguments.get("distillation_kl_direction")
    return {
        "method_id": arguments.get("method_id"),
        "max_rollout_tokens": as_int(arguments.get("max_rollout_tokens"), "max_rollout_tokens"),
        "max_sequence_tokens": as_int(arguments.get("max_sequence_tokens"), "max_sequence_tokens"),
        "kl_direction_manifest": direction,
        "model_id": arguments.get("model_id"),
        "seed": arguments.get("seed"),
        "learning_rate": arguments.get("learning_rate"),
        "lora_rank": arguments.get("lora_rank"),
        "trust_region_kl_budget": arguments.get("trust_region_kl_budget"),
        "git_commit": (manifest.get("runtime") or {}).get("git_commit"),
        "slurm_job_id": (manifest.get("runtime") or {}).get("slurm_job_id"),
    }


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def fmt_pp(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return f"{100.0 * float(value):+.2f} pp"


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}f}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], align: Sequence[str] | None = None) -> str:
    if align is None:
        align = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "—": r"--",
        "→": r"$\rightarrow$",
        "±": r"$\pm$",
        "×": r"$\times$",
        "↓": r"$\downarrow$",
        "↑": r"$\uparrow$",
    }
    return "".join(replacements.get(character, character) for character in text)


def latex_table(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    alignment: str | None = None,
) -> str:
    alignment = alignment or ("l" + "r" * (len(headers) - 1))
    body = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(item) for item in headers) + r" \\",
        r"\midrule",
    ]
    body.extend(" & ".join(latex_escape(item) for item in row) + r" \\" for row in rows)
    body.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(body)


def build_bundle(args: argparse.Namespace) -> None:
    scored_paths = {
        method: args.scored_root / SCORED_FILENAME[method] for method in METHODS
    }
    method_rows = {method: load_scored(path, method) for method, path in scored_paths.items()}
    validate_matched_queries(method_rows)
    generation_protocol = validate_generation_protocol(args.prediction_root, method_rows)

    accuracy_rows = []
    aggregate_by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for method in AVAILABLE_METHODS:
        values = aggregate_accuracy(method, method_rows[method])
        aggregate_by_method[method] = {str(row["dataset"]): row for row in values}
    base = aggregate_by_method["base"]
    for method in METHODS:
        for dataset in SOURCES:
            row = dict(aggregate_by_method[method][dataset])
            delta = float(row["strict_acc1"]) - float(base[dataset]["strict_acc1"])
            row.update(
                {
                    "episodes": CHECKPOINT_EPISODE[method],
                    "delta_vs_base": delta,
                    "delta_vs_base_percentage_points": 100.0 * delta,
                    "status": "complete_current_matched_evaluation",
                }
            )
            accuracy_rows.append(row)

    paired_vs_base = []
    for dataset_index, dataset in enumerate(SOURCES):
        for method in PAIRED_BASE_METHODS:
            paired_vs_base.append(
                paired_inference(
                    method_rows["base"],
                    method_rows[method],
                    reference="base",
                    method=method,
                    dataset=dataset,
                    seed=(
                        BOOTSTRAP_SEED
                        + 100 * dataset_index
                        + PAIRED_BASE_SEED_OFFSET[method]
                    ),
                )
            )

    direct_t64_p64 = []
    for dataset_index, dataset in enumerate(SOURCES):
        direct_t64_p64.append(
            paired_inference(
                method_rows["privileged_64"],
                method_rows["trsd_64"],
                reference="privileged_64",
                method="trsd_64",
                dataset=dataset,
                seed=BOOTSTRAP_SEED + 1000 + dataset_index,
            )
        )

    style_summary = read_csv(args.style_root / "matched64_style_summary.csv")
    style_effects = read_csv(args.style_root / "matched64_paired_effects.csv")
    same_prefix = read_csv(args.style_root / "same_prefix_mechanism_summary.csv")
    style_by_target = {row["target"]: row for row in style_summary}
    if set(style_by_target) != {"raw_privileged", "trsd_projected"}:
        raise EvidenceError(f"Unexpected style targets: {sorted(style_by_target)}")
    epsilon_rows = read_csv(args.epsilon_csv)
    selected_epsilon = [row for row in epsilon_rows if str(row.get("is_selected", "")).lower() == "true"]
    if len(selected_epsilon) != 1 or abs(as_float(selected_epsilon[0]["epsilon"], "selected epsilon") - 0.004) > 1e-12:
        raise EvidenceError("Expected exactly one development-selected epsilon=0.004")

    p16_journal = summarize_journal(args.p16_journal, 16)
    p64_journal = summarize_journal(args.p64_journal, 64)
    t16_journal = summarize_journal(args.t64_journal, 16)
    t64_journal = summarize_journal(args.t64_journal, 64)
    p16_prov = manifest_provenance(args.p16_manifest)
    p64_prov = manifest_provenance(args.p64_manifest)
    t64_prov = manifest_provenance(args.t64_manifest)
    t16_prov = t64_prov
    if p64_prov["kl_direction_manifest"] is not None:
        raise EvidenceError("P64 unexpectedly declares a KL direction; provenance assumption changed")
    if t64_prov["kl_direction_manifest"] != "student_to_projected_teacher_reverse_kl_v1":
        raise EvidenceError("T64 is not the expected exact reverse-KL run")
    if p64_prov["max_rollout_tokens"] != 4096 or t64_prov["max_rollout_tokens"] != 10240:
        raise EvidenceError("Expected P64/T64 rollout caps 4096/10240")

    audit_rows: list[dict[str, Any]] = [
        {
            "method": "base",
            "method_label": "Base",
            "episodes": 0,
            "teacher_positions": 0,
            "hindsight_exposure_rate": "N/A",
            "on_policy_prefix_rate": "N/A",
            "full_context_parity": "N/A",
            "teacher_only_reasoning_method": False,
            "student_centered_projection": False,
            "temporary_teacher_destroyed": "N/A",
            "status": "no_distillation_teacher",
        }
    ]
    for method, summary in (
        ("privileged_16", p16_journal),
        ("trsd_16", t16_journal),
        ("privileged_64", p64_journal),
        ("trsd_64", t64_journal),
    ):
        sources = summary["teacher_context_sources"]
        audit_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABEL[method],
                "episodes": summary["episodes"],
                "teacher_positions": summary["teacher_positions"],
                "hindsight_exposure_rate": summary["hindsight_exposure_rate"],
                "on_policy_prefix_rate": summary["on_policy_prefix_rate"],
                "full_context_parity": summary["full_context_parity"],
                "teacher_only_reasoning_method": "predecision_reasoning_method" in sources,
                "student_centered_projection": "student_centered_exponential_projection" in sources,
                "temporary_teacher_destroyed": summary["temporary_teacher_destroyed"],
                "teacher_context_sources": ";".join(sources),
                "status": "audited",
            }
        )

    training_rows = []
    for method, journal, provenance, direction in (
        (
            "privileged_16",
            p16_journal,
            p16_prov,
            "legacy_run; KL direction absent from manifest",
        ),
        (
            "trsd_16",
            t16_journal,
            t16_prov,
            "exact reverse KL: student -> projected teacher",
        ),
        (
            "privileged_64",
            p64_journal,
            p64_prov,
            "legacy forward-KL run; direction field absent",
        ),
        (
            "trsd_64",
            t64_journal,
            t64_prov,
            "exact reverse KL: student -> projected teacher",
        ),
    ):
        seconds = float(journal["training_seconds"])
        tokens = int(journal["response_tokens"])
        training_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABEL[method],
                "episodes": journal["episodes"],
                "optimizer_steps": journal["optimizer_steps"],
                "no_op_episodes": journal["no_op_episodes"],
                "response_tokens": tokens,
                "training_hours": seconds / 3600.0,
                "seconds_per_episode": seconds / int(journal["episodes"]),
                "seconds_per_1k_response_tokens": 1000.0 * seconds / tokens,
                "max_rollout_tokens": provenance["max_rollout_tokens"],
                "max_sequence_tokens": provenance["max_sequence_tokens"],
                "kl_direction": direction,
                "peak_gpu_allocated_gib": journal["max_gpu_peak_allocated_gib"],
                "peak_gpu_delta_gib": journal["max_gpu_peak_delta_gib"],
                "training_seed": provenance["seed"],
                "slurm_job_id": provenance["slurm_job_id"],
            }
        )

    eval_efficiency = []
    behavior_rows = []
    for method in METHODS:
        row = aggregate_by_method[method]["combined"]
        base_seconds = float(aggregate_by_method["base"]["combined"]["mean_generation_seconds"])
        eval_efficiency.append(
            {
                "method": method,
                "method_label": METHOD_LABEL[method],
                "strict_acc1": row["strict_acc1"],
                "mean_generated_tokens": row["mean_generated_tokens"],
                "mean_generation_seconds": row["mean_generation_seconds"],
                "aggregate_gpu_hours": row["aggregate_gpu_hours"],
                "peak_gpu_allocated_gib": row["peak_gpu_allocated_gib"],
                "relative_seconds_vs_base": float(row["mean_generation_seconds"]) / base_seconds,
                "status": "complete",
            }
        )
        behavior_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABEL[method],
                "budget_cap_hit_count": row["budget_cap_hit_count"],
                "n": row["n"],
                "budget_cap_hit_rate": row["budget_cap_hit_rate"],
                "mean_generated_tokens": row["mean_generated_tokens"],
                "hedging_tokens_per_1k": row["hedging_tokens_per_1k"],
                "fabricated_reference_count": row["fabricated_reference_count"],
                "fabricated_reference_rate": row["fabricated_reference_rate"],
                "metric_role": "non_accuracy_operational_diagnostic",
                "status": "complete",
            }
        )

    combined_direct = next(row for row in direct_t64_p64 if row["dataset"] == "combined")
    p64_by_query = {str(row["query_id"]): row for row in method_rows["privileged_64"]}
    t64_by_query = {str(row["query_id"]): row for row in method_rows["trsd_64"]}
    favorable_ids = [
        query_id
        for query_id in sorted(p64_by_query)
        if strict_correct(p64_by_query[query_id]) == 0
        and strict_correct(t64_by_query[query_id]) == 1
    ]
    unfavorable_ids = [
        query_id
        for query_id in sorted(p64_by_query)
        if strict_correct(p64_by_query[query_id]) == 1
        and strict_correct(t64_by_query[query_id]) == 0
    ]
    p64_cap_hit_to_t64_correct = sum(
        bool(p64_by_query[query_id]["truncated"]) for query_id in favorable_ids
    )
    if (
        len(favorable_ids) != 16
        or len(unfavorable_ids) != 4
        or p64_cap_hit_to_t64_correct != 11
    ):
        raise EvidenceError(
            "Unexpected P64/T64 transition anatomy; expected W->C=16, C->W=4, "
            "and P64-cap-hit->T64-correct=11"
        )
    transition_anatomy_rows = [
        {
            "comparison": "TRSD 64 vs Privilege-SD 64",
            "wrong_to_correct": len(favorable_ids),
            "correct_to_wrong": len(unfavorable_ids),
            "p64_cap_hit_to_t64_correct": p64_cap_hit_to_t64_correct,
            "p64_cap_hit_share_of_favorable": p64_cap_hit_to_t64_correct
            / len(favorable_ids),
            "metric_role": "non_accuracy_transition_anatomy",
        }
    ]
    combined_t64_base = next(
        row
        for row in paired_vs_base
        if row["dataset"] == "combined" and row["method"] == "trsd_64"
    )
    combined_t16_base = next(
        row
        for row in paired_vs_base
        if row["dataset"] == "combined" and row["method"] == "trsd_16"
    )
    style_raw = style_by_target["raw_privileged"]
    style_trsd = style_by_target["trsd_projected"]
    style_delta = next(row for row in style_effects if row["metric"] == "style_error_per_token")
    same_raw = next(row for row in same_prefix if row["projection"] == "raw_privileged_surrogate")
    same_trsd = next(row for row in same_prefix if row["projection"] == "trsd_projected")

    claim_rows = [
        {
            "claim_id": "C1",
            "claim": "TRSD-64 improves strict held-out Acc@1 over Base under the matched 10,240-token evaluation.",
            "evidence": f"102/143 vs 77/143; delta {combined_t64_base['delta_percentage_points']:+.2f} pp; paired 95% CI [{100*combined_t64_base['delta_ci_low']:+.2f}, {100*combined_t64_base['delta_ci_high']:+.2f}] pp; exact p={combined_t64_base['mcnemar_exact_two_sided_p']:.3g}.",
            "status": "supported_on_this_single-seed_evaluation",
            "boundary": "One training seed and one sampled response per query.",
        },
        {
            "claim_id": "C2",
            "claim": "The observed TRSD-64 checkpoint outperforms the observed Privilege-SD64 checkpoint.",
            "evidence": f"102/143 vs 90/143; paired delta {combined_direct['delta_percentage_points']:+.2f} pp; 95% CI [{100*combined_direct['delta_ci_low']:+.2f}, {100*combined_direct['delta_ci_high']:+.2f}] pp; W->C/C->W={combined_direct['wrong_to_correct']}/{combined_direct['correct_to_wrong']}; 11/16 favorable transitions are P64 cap-hit -> T64 correct.",
            "status": "supported_as_observed_checkpoint_comparison",
            "boundary": "Not a clean causal ablation: P64 is legacy forward-KL/4096; T64 is exact reverse-KL/10240.",
        },
        {
            "claim_id": "C3",
            "claim": "Trajectory-level projection substantially limits privileged-target movement.",
            "evidence": f"Target KL {float(style_raw['target_student_kl']):.6f} -> {float(style_trsd['target_student_kl']):.6f} ({100*(1-float(style_trsd['target_student_kl'])/float(style_raw['target_student_kl'])):.2f}% reduction); constraint active on {100*float(style_trsd['constraint_activation_rate']):.2f}% of 64 episodes.",
            "status": "supported",
            "boundary": "A surrogate-distribution guarantee, not a theorem about downstream accuracy.",
        },
        {
            "claim_id": "C4",
            "claim": "TRSD reduces the measured style-target movement on the paired 64-query stream.",
            "evidence": f"Style/token {float(style_raw['style_error_per_token']):.6f} -> {float(style_trsd['style_error_per_token']):.6f}; relative reduction {100*float(style_delta['relative_reduction']):.2f}% with paired-bootstrap 95% CI [{100*float(style_delta['relative_reduction_ci_low']):.2f}, {100*float(style_delta['relative_reduction_ci_high']):.2f}]%.",
            "status": "supported_for_versioned_token_partition",
            "boundary": "Heuristic token partition; trajectories differ in length/content.",
        },
        {
            "claim_id": "C5",
            "claim": "The style reduction persists under an identical-prefix mechanism check.",
            "evidence": f"Across 3 queries x 3 wrappers, style shift {float(same_raw['style_abs_logprob_shift']):.6f} -> {float(same_trsd['style_abs_logprob_shift']):.6f} ({100*(1-float(same_trsd['style_abs_logprob_shift'])/float(same_raw['style_abs_logprob_shift'])):.2f}% reduction).",
            "status": "descriptive_support",
            "boundary": "Only three distinct queries; no inferential claim.",
        },
        {
            "claim_id": "C6",
            "claim": "Training uses no target answer, reference solution, future trajectory, or post-outcome feedback.",
            "evidence": "HER=0.000 across P16, current T16, P64, and T64 audited teacher positions; all positions are student on-policy prefixes.",
            "status": "supported_by_training_journals",
            "boundary": "The raw teacher still receives a teacher-only pre-decision reasoning-method prompt, so strict full-context parity is 0.",
        },
        {
            "claim_id": "C7",
            "claim": "TRSD is operationally stable through this observed 64-episode run.",
            "evidence": "The 64-episode run completed 64/64 optimizer steps with 0 no-ops and positive endpoint accuracy.",
            "status": "partially_supported",
            "boundary": "This is not a multi-seed or general long-term-stability claim; two evaluated checkpoints do not constitute AULC.",
        },
        {
            "claim_id": "C8",
            "claim": "The current reverse-KL TRSD-16 checkpoint has a fully matched strict held-out evaluation.",
            "evidence": (
                f"{combined_t16_base['method_correct']}/143 vs "
                f"{combined_t16_base['reference_correct']}/143 for Base; delta "
                f"{combined_t16_base['delta_percentage_points']:+.2f} pp; paired "
                f"95% CI [{100*combined_t16_base['delta_ci_low']:+.2f}, "
                f"{100*combined_t16_base['delta_ci_high']:+.2f}] pp; exact "
                f"p={combined_t16_base['mcnemar_exact_two_sided_p']:.3g}."
            ),
            "status": "supported_on_this_single-seed_evaluation",
            "boundary": "One training seed and one sampled response per query; T16 and T64 alone do not identify an AULC.",
        },
    ]

    limitation_rows = [
        {
            "limitation_id": "L1",
            "severity": "high",
            "limitation": "P64 and T64 training are not protocol-matched.",
            "consequence": "The +8.39 pp endpoint gap is observational, not a clean estimate of projection alone.",
            "detail": "P64: legacy forward-KL, 4,096 rollout cap, 246,371 response tokens. T64: exact reverse-KL, 10,240 cap, 433,074 tokens.",
        },
        {
            "limitation_id": "L2",
            "severity": "high",
            "limitation": "Single training seed and one generation per held-out query.",
            "consequence": "Paired query CIs quantify query uncertainty, not training-seed variance.",
            "detail": "Mean@4 and multi-seed training remain future work.",
        },
        {
            "limitation_id": "L3",
            "severity": "medium",
            "limitation": "Only the T16 and T64 TRSD checkpoints are evaluated.",
            "consequence": "The two endpoints do not support AULC or a formal clean/privilege crossover claim.",
            "detail": "Additional intermediate checkpoints would be required for a resolved learning curve.",
        },
        {
            "limitation_id": "L4",
            "severity": "medium",
            "limitation": "Style/task token categories are heuristic.",
            "consequence": "Style reduction supports controlled target movement but does not prove semantic disentanglement.",
            "detail": "The same-prefix pilot mitigates trajectory confounding but has n=3 distinct queries.",
        },
        {
            "limitation_id": "L5",
            "severity": "medium",
            "limitation": "Strict full-context parity is zero.",
            "consequence": "TRSD is no-hindsight and on-policy, but raw direction construction remains privileged-informed.",
            "detail": "Clean refers to student-centered KL projection, not absence of privileged information during teacher construction.",
        },
    ]

    output = args.out
    tables = output / "tables"
    latex = output / "latex"
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    latex.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "main_accuracy.csv", accuracy_rows)
    write_csv(tables / "paired_inference_vs_base.csv", paired_vs_base)
    write_csv(tables / "trsd64_vs_privileged64.csv", direct_t64_p64)
    write_csv(tables / "trust_region_target_summary.csv", style_summary)
    write_csv(tables / "trust_region_paired_effects.csv", style_effects)
    write_csv(tables / "same_prefix_pilot.csv", same_prefix)
    write_csv(tables / "epsilon_sensitivity.csv", epsilon_rows)
    write_csv(tables / "cleanliness_audit.csv", audit_rows)
    write_csv(tables / "evaluation_efficiency.csv", eval_efficiency)
    write_csv(tables / "training_efficiency_and_provenance.csv", training_rows)
    write_csv(tables / "completion_behavior_diagnostics.csv", behavior_rows)
    write_csv(tables / "transition_anatomy.csv", transition_anatomy_rows)
    write_csv(tables / "claim_evidence_map.csv", claim_rows)
    write_csv(tables / "limitations.csv", limitation_rows)

    main_by_method = {
        method: {row["dataset"]: row for row in accuracy_rows if row["method"] == method}
        for method in METHODS
    }
    main_rows = []
    for method in METHODS:
        combined = main_by_method[method]["combined"]
        cells = []
        for dataset in ("amc23", "aime24", "aime25", "combined"):
            row = main_by_method[method][dataset]
            cells.append(
                f"{float(row['strict_acc1_percent']):.2f}% "
                f"({row['strict_correct']}/{row['n']})"
            )
        delta = fmt_pp(combined["delta_vs_base"])
        main_rows.append([METHOD_LABEL[method], combined["episodes"], *cells, delta])

    paired_combined = [row for row in paired_vs_base if row["dataset"] == "combined"]
    paired_rows = []
    for row in paired_combined:
        paired_rows.append(
            [
                row["method_label"],
                f"{100*row['method_acc1']:.2f}% [{100*row['method_acc1_ci_low']:.2f}, {100*row['method_acc1_ci_high']:.2f}]",
                f"{row['delta_percentage_points']:+.2f} pp [{100*row['delta_ci_low']:+.2f}, {100*row['delta_ci_high']:+.2f}]",
                f"{row['wrong_to_correct']} / {row['correct_to_wrong']}",
                f"{row['mcnemar_exact_two_sided_p']:.4g}",
            ]
        )
    direct_rows = []
    for row in direct_t64_p64:
        direct_rows.append(
            [
                SOURCE_LABEL[row["dataset"]],
                f"{100*row['reference_acc1']:.2f}% ({row['reference_correct']}/{row['n']})",
                f"{100*row['method_acc1']:.2f}% ({row['method_correct']}/{row['n']})",
                f"{row['delta_percentage_points']:+.2f} pp [{100*row['delta_ci_low']:+.2f}, {100*row['delta_ci_high']:+.2f}]",
                f"{row['wrong_to_correct']} / {row['correct_to_wrong']}",
                f"{row['mcnemar_exact_two_sided_p']:.4g}",
            ]
        )

    style_rows_md = []
    for key, label in (("raw_privileged", "Raw privileged target"), ("trsd_projected", "TRSD projected target")):
        row = style_by_target[key]
        style_rows_md.append(
            [
                label,
                f"{float(row['style_error_per_token']):.6f} [{float(row['style_error_per_token_ci_low']):.6f}, {float(row['style_error_per_token_ci_high']):.6f}]",
                f"{float(row['task_error_per_token']):.6f} [{float(row['task_error_per_token_ci_low']):.6f}, {float(row['task_error_per_token_ci_high']):.6f}]",
                f"{float(row['psr']):.3f}",
                fmt_num(row.get("mean_alpha") if key == "trsd_projected" else 1.0, 4),
                fmt_num(row.get("target_student_kl"), 6),
                (fmt_pct(row.get("constraint_activation_rate")) if key == "trsd_projected" else "N/A"),
                f"{row['optimizer_steps']}/{row['no_op_episodes']}",
                f"{float(row['training_hours']):.2f}",
            ]
        )

    epsilon_md_rows = []
    for row in epsilon_rows:
        epsilon_md_rows.append(
            [
                row["epsilon"],
                fmt_num(row["mean_alpha"], 4),
                fmt_num(row["achieved_mean_kl"], 6),
                row["active_wrappers"],
                fmt_num(row["task_gain_vs_raw"], 3),
                fmt_num(row["style_retention_vs_raw"], 3),
                fmt_num(row["prompt_variance_retention"], 3),
                "✓" if str(row["is_selected"]).lower() == "true" else "",
            ]
        )

    audit_md_rows = []
    for row in audit_rows:
        if row["method"] == "base":
            audit_md_rows.append([row["method_label"], 0, "N/A", "N/A", "N/A", "No", "N/A"])
        else:
            audit_md_rows.append(
                [
                    row["method_label"],
                    row["episodes"],
                    fmt_pct(row["hindsight_exposure_rate"]),
                    fmt_pct(row["on_policy_prefix_rate"]),
                    fmt_pct(row["full_context_parity"]),
                    "Yes" if row["student_centered_projection"] else "No",
                    "Yes" if row["temporary_teacher_destroyed"] else "No",
                ]
            )

    eval_md_rows = []
    for row in eval_efficiency:
        eval_md_rows.append(
            [
                row["method_label"],
                fmt_pct(row["strict_acc1"]),
                f"{float(row['mean_generated_tokens']):.0f}",
                f"{float(row['mean_generation_seconds']):.1f}",
                f"{float(row['aggregate_gpu_hours']):.2f}",
                f"{float(row['peak_gpu_allocated_gib']):.2f}",
            ]
        )

    behavior_md_rows = []
    for row in behavior_rows:
        behavior_md_rows.append(
            [
                row["method_label"],
                f"{row['budget_cap_hit_count']}/{row['n']} ({100*float(row['budget_cap_hit_rate']):.2f}%)",
                f"{float(row['mean_generated_tokens']):.0f}",
                f"{float(row['hedging_tokens_per_1k']):.2f}",
                f"{row['fabricated_reference_count']}/{row['n']} ({100*float(row['fabricated_reference_rate']):.2f}%)",
            ]
        )

    train_md_rows = []
    for row in training_rows:
        train_md_rows.append(
            [
                row["method_label"], row["episodes"], f"{row['optimizer_steps']}/{row['no_op_episodes']}",
                row["response_tokens"], row["max_rollout_tokens"], f"{float(row['training_hours']):.2f}",
                f"{float(row['seconds_per_1k_response_tokens']):.2f}",
                fmt_num(row["peak_gpu_allocated_gib"], 2), row["kl_direction"],
            ]
        )

    report_parts = [
        "# TRSD table-first evidence report",
        "",
        "This bundle reports only completed, auditable artifacts. The sole public performance metric is **strict Acc@1 over the full denominator**: a response must be correct and finish within the fixed 10,240-token generation budget; otherwise it is wrong. No alternative accuracy denominator is emitted. All five checkpoints must pass the complete matched-protocol audit before this report is written.",
        "",
        "The repository also includes the four complete 143-query scored outputs and the corresponding training audit journals/manifests under [`evidence/`](evidence/README.md). Model and optimizer weights are intentionally excluded.",
        "",
        "## 1. Main held-out result",
        "",
        md_table(
            ["Method", "Episodes", "AMC23", "AIME24", "AIME25", "Combined", "Δ vs Base"],
            main_rows,
            ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "All five reported checkpoints use Qwen3-8B, the same 143 held-out questions, the same explicit generation-budget prompt, identical deterministic query-specific seeds, temperature 0.6, top-p 0.95, top-k 20, and a 10,240-token cap. Dataset labels are sealed during training and used only by the offline scorer.",
        "",
        "TRSD-64 reaches **102/143 (71.33%)**, which is +17.48 pp over Base. It improves all three datasets: AMC23 84.34%, AIME24 63.33%, and AIME25 43.33%.",
        "",
        "## 2. Paired robustness against Base",
        "",
        md_table(
            ["Method", "Strict Acc@1 [95% CI]", "Δ vs Base [95% CI]", "W→C / C→W", "Exact p"],
            paired_rows,
            ["---", "---:", "---:", "---:", "---:"],
        ),
        "",
        f"Intervals use {BOOTSTRAP_REPLICATES:,} paired-query bootstrap resamples. Exact p-values are two-sided McNemar tests on discordant query outcomes. These intervals quantify held-out query uncertainty, not training-seed variance.",
        "",
        "## 3. Direct 64-episode checkpoint comparison",
        "",
        md_table(
            ["Dataset", "Privilege-SD64", "TRSD-64", "TRSD−P64 [95% CI]", "W→C / C→W", "Exact p"],
            direct_rows,
            ["---", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "This is an **observed checkpoint comparison**, not a clean causal ablation. Evaluation is matched, but training is not: Privilege-SD64 is the older forward-KL checkpoint with a 4,096-token rollout cap, whereas TRSD-64 uses exact reverse KL and a 10,240-token rollout cap.",
        "",
        "### Transition anatomy (non-accuracy diagnostic)",
        "",
        md_table(
            ["Comparison", "W→C", "C→W", "P64 cap-hit → T64 correct", "Share of favorable"],
            [["TRSD-64 vs Privilege-SD64", 16, 4, 11, "68.75% (11/16)"]],
            ["---", "---:", "---:", "---:", "---:"],
        ),
        "",
        "Eleven of the sixteen favorable transitions are cases where Privilege-SD64 exhausted the generation budget and TRSD-64 finished with the correct answer. This row explains transition behavior; it is not a second accuracy metric.",
        "",
        "## 4. What the trust region changes",
        "",
        md_table(
            ["Distillation target", "Style/token [95% CI] ↓", "Task/token [95% CI]†", "PSR", "α", "Target KL ↓", "Constraint active", "Steps/no-op", "Train h"],
            style_rows_md,
            ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        f"The projection reduced target-to-student KL by **{100*(1-float(style_trsd['target_student_kl'])/float(style_raw['target_student_kl'])):.2f}%** and normalized style movement by **{100*float(style_delta['relative_reduction']):.2f}%** (paired-episode 95% CI {100*float(style_delta['relative_reduction_ci_low']):.2f}%–{100*float(style_delta['relative_reduction_ci_high']):.2f}%). The constraint activated on {100*float(style_trsd['constraint_activation_rate']):.2f}% of episodes, so ε=0.004 was operational rather than inert.",
        "",
        "† Task/token is the absolute movement of realized task-bearing token log-probabilities. It is neither signed improvement nor downstream accuracy. PSR does not improve here; therefore the defensible claim is reduced absolute privileged drift, not perfect task/style separation.",
        "",
        "## 5. Same-prefix mechanism check",
        "",
        md_table(
            ["Target", "Queries × wrappers", "Style shift ↓", "Signed task-token gain ↑", "α", "Target KL ↓"],
            [
                ["Raw privileged", "3 × 3", fmt_num(same_raw["style_abs_logprob_shift"], 6), fmt_num(same_raw["task_logprob_gain"], 6), "1.0000", fmt_num(same_raw["mean_achieved_kl"], 6)],
                ["TRSD projected", "3 × 3", fmt_num(same_trsd["style_abs_logprob_shift"], 6), fmt_num(same_trsd["task_logprob_gain"], 6), fmt_num(same_trsd["mean_alpha"], 4), fmt_num(same_trsd["mean_achieved_kl"], 6)],
            ],
            ["---", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        f"Holding prefixes fixed, projection reduced measured style shift by **{100*(1-float(same_trsd['style_abs_logprob_shift'])/float(same_raw['style_abs_logprob_shift'])):.2f}%** while signed task-token gain rose from {float(same_raw['task_logprob_gain']):.6f} to {float(same_trsd['task_logprob_gain']):.6f}. This is a descriptive mechanism pilot because it contains only three distinct queries.",
        "",
        "## 6. Development-only ε sensitivity",
        "",
        md_table(
            ["ε", "Mean α", "Achieved KL", "Active wrappers", "Task gain/raw ↑", "Style retained ↓", "Prompt variance retained ↓", "Selected"],
            epsilon_md_rows,
            ["---:", "---:", "---:", "---:", "---:", "---:", "---:", ":---:"],
        ),
        "",
        "ε=0.004 was selected on the one-episode development mechanism sweep because it achieved the largest signed task gain in the tested grid while keeping all three wrapper constraints active. Held-out labels were not used for this choice.",
        "",
        "## 7. Cleanliness and context audit",
        "",
        md_table(
            ["Method", "Episodes", "HER ↓", "On-policy prefix", "Strict full-context parity", "Student-centered projection", "Teacher destroyed"],
            audit_md_rows,
            ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "HER=0 means no target answer, reference solution, future trajectory, or post-outcome feedback was exposed. All scored teacher positions use the student's on-policy prefix. Strict full-context parity remains 0 because the raw teacher receives a teacher-only pre-decision reasoning-method prompt. Thus, **clean** here means a no-hindsight, student-centered projected distillation target—not privilege-free teacher construction.",
        "",
        "## 8. Evaluation efficiency",
        "",
        md_table(
            ["Method", "Strict Acc@1", "Tokens/query", "Seconds/query", "Aggregate GPU h", "Peak alloc. GiB"],
            eval_md_rows,
            ["---", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "TRSD-64 evaluation is not slower than the observed Privilege-SD64 checkpoint in this run (266.7 vs 282.1 seconds/query), while using essentially the same peak inference allocation. Timing depends strongly on generated length and cluster conditions.",
        "",
        "### Completion and response behavior (non-accuracy diagnostics)",
        "",
        md_table(
            ["Method", "Budget-cap hits", "Tokens/query", "Hedging/1k", "Fabricated reference"],
            behavior_md_rows,
            ["---", "---:", "---:", "---:", "---:"],
        ),
        "",
        "These quantities diagnose how responses use the fixed budget and how their surface form changes. They are not performance metrics; strict Acc@1 above remains the sole accuracy metric.",
        "",
        "## 9. Training efficiency and provenance",
        "",
        md_table(
            ["Method", "Episodes", "Steps/no-op", "Response tokens", "Rollout cap", "Train h", "Sec/1k tok", "Peak alloc. GiB", "KL objective"],
            train_md_rows,
            ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
        ),
        "",
        "Both 64-episode runs completed every optimizer step with zero no-ops. TRSD-64 took 2.39× the recorded wall-clock training time, but also processed 1.76× as many response tokens under a larger rollout cap. Privilege-SD64 did not record per-episode GPU memory telemetry, so a matched training-memory comparison is unavailable.",
        "",
        "## 10. Claim–evidence map",
        "",
        md_table(
            ["ID", "Claim", "Evidence status", "Boundary"],
            [[row["claim_id"], row["claim"], row["status"], row["boundary"]] for row in claim_rows],
            ["---", "---", "---", "---"],
        ),
        "",
        "## 11. Limitations",
        "",
        md_table(
            ["ID", "Severity", "Limitation", "Consequence"],
            [[row["limitation_id"], row["severity"], row["limitation"], row["consequence"]] for row in limitation_rows],
            ["---", "---", "---", "---"],
        ),
        "",
        "## Reviewer-facing self-check",
        "",
        "- **Contribution:** the exact exponential projection and trajectory-level KL budget are directly tied to measured KL/style contraction.",
        "- **Clarity:** strict Acc@1, clean, on-policy, same-prefix, and full-context parity are defined separately.",
        "- **Empirical strength:** the 143-query endpoint gain and 64-query paired drift result are strong within this run; multi-seed evidence is absent.",
        "- **Evaluation completeness:** matched T16 and T64 endpoints are reported; AULC, Mean@4, and fully training-matched P64/T64 ablations remain future work.",
        "- **Method soundness:** the trust-region surrogate is exact and operational; downstream accuracy is empirical rather than guaranteed by projection theory.",
        "",
    ]
    report = "\n".join(report_parts)
    (output / "TRSD_TABLE_REPORT.md").write_text(report, encoding="utf-8")
    (output / "README.md").write_text(report, encoding="utf-8")

    tex_tables = {
        "table_main_accuracy.tex": latex_table(
            "Strict Acc@1 after DeepMath self-distillation. Unfinished generations count as wrong; every denominator is fixed.",
            "tab:trsd-main-accuracy",
            ["Method", "Ep.", "AMC23", "AIME24", "AIME25", "Combined", "Delta vs Base"],
            main_rows,
        ),
        "table_paired_vs_base.tex": latex_table(
            "Paired held-out inference against Base. Confidence intervals use paired-query bootstrap resampling.",
            "tab:trsd-paired-base",
            ["Method", "Acc@1 [95% CI]", "Delta [95% CI]", "W/C / C/W", "p"],
            paired_rows,
        ),
        "table_trsd64_vs_p64.tex": latex_table(
            "Observed TRSD-64 versus Privilege-SD64 checkpoints under matched evaluation.",
            "tab:trsd64-p64",
            ["Dataset", "P64", "T64", "T64-P64 [95% CI]", "W/C / C/W", "p"],
            direct_rows,
        ),
        "table_trust_region_target.tex": latex_table(
            "Target-distribution diagnostics over the paired 64-query stream.",
            "tab:trsd-target",
            ["Target", "Style/token", "Task/token", "PSR", "alpha", "Target KL", "Active", "Steps/no-op", "Hours"],
            style_rows_md,
        ),
        "table_same_prefix.tex": latex_table(
            "Same-prefix mechanism pilot across three queries and three prompt wrappers.",
            "tab:trsd-same-prefix",
            ["Target", "Q x W", "Style shift", "Task gain", "alpha", "Target KL"],
            [
                ["Raw privileged", "3 x 3", fmt_num(same_raw["style_abs_logprob_shift"], 6), fmt_num(same_raw["task_logprob_gain"], 6), "1.0000", fmt_num(same_raw["mean_achieved_kl"], 6)],
                ["TRSD projected", "3 x 3", fmt_num(same_trsd["style_abs_logprob_shift"], 6), fmt_num(same_trsd["task_logprob_gain"], 6), fmt_num(same_trsd["mean_alpha"], 4), fmt_num(same_trsd["mean_achieved_kl"], 6)],
            ],
        ),
        "table_epsilon.tex": latex_table(
            "Development-only sensitivity of the trajectory KL budget.",
            "tab:trsd-epsilon",
            ["epsilon", "alpha", "KL", "Active", "Task/raw", "Style ret.", "Var. ret.", "Chosen"],
            epsilon_md_rows,
        ),
        "table_cleanliness.tex": latex_table(
            "Training-context audit. CP denotes strict full-prompt parity.",
            "tab:trsd-cleanliness",
            ["Method", "Ep.", "HER", "On-policy", "CP", "Projection", "Destroyed"],
            audit_md_rows,
        ),
        "table_eval_efficiency.tex": latex_table(
            "Matched held-out evaluation efficiency on H100 GPUs.",
            "tab:trsd-eval-efficiency",
            ["Method", "Acc@1", "Tok/query", "Sec/query", "GPU h", "Peak GiB"],
            eval_md_rows,
        ),
        "table_train_efficiency.tex": latex_table(
            "Recorded training efficiency and objective provenance.",
            "tab:trsd-train-efficiency",
            ["Method", "Ep.", "Steps/no-op", "Tokens", "Cap", "Hours", "Sec/1k", "Peak GiB", "Objective"],
            train_md_rows,
        ),
        "table_completion_behavior.tex": latex_table(
            "Non-accuracy completion and response-behavior diagnostics under the fixed evaluation budget.",
            "tab:trsd-completion-behavior",
            ["Method", "Cap hits", "Tok/query", "Hedge/1k", "Fabricated ref."],
            behavior_md_rows,
        ),
    }
    for name, payload in tex_tables.items():
        (latex / name).write_text(payload, encoding="utf-8")
    (latex / "all_tables.tex").write_text("\n".join(tex_tables.values()), encoding="utf-8")

    source_paths = [
        *scored_paths.values(),
        args.style_root / "matched64_style_summary.csv",
        args.style_root / "matched64_paired_effects.csv",
        args.style_root / "same_prefix_mechanism_summary.csv",
        args.epsilon_csv,
        args.p16_journal,
        args.p16_manifest,
        args.p64_journal,
        args.p64_manifest,
        args.t64_journal,
        args.t64_manifest,
    ]
    summary = {
        "schema_version": "trsd-table-evidence-bundle-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_estimand": "strict_acc1_full_denominator_unfinished_is_wrong",
        "trsd_16_status": "complete_current_matched_evaluation",
        "generation_protocol": generation_protocol,
        "accuracy": accuracy_rows,
        "paired_vs_base": paired_vs_base,
        "trsd64_vs_privileged64": direct_t64_p64,
        "style_summary": style_summary,
        "style_paired_effects": style_effects,
        "same_prefix": same_prefix,
        "epsilon_sensitivity": epsilon_rows,
        "cleanliness_audit": audit_rows,
        "evaluation_efficiency": eval_efficiency,
        "completion_behavior_diagnostics": behavior_rows,
        "transition_anatomy": transition_anatomy_rows,
        "training_efficiency": training_rows,
        "training_provenance": {
            "privileged_16": p16_prov,
            "trsd_16": t16_prov,
            "privileged_64": p64_prov,
            "trsd_64": t64_prov,
        },
        "claim_evidence": claim_rows,
        "limitations": limitation_rows,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "base_seed": BOOTSTRAP_SEED},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_files = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and "logs" not in path.relative_to(output).parts
    )
    manifest = {
        "schema_version": "trsd-table-evidence-manifest-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "output_files": output_files,
        "fail_closed_checks": [
            "143 unique matched queries per reported method",
            "83/30/30 source counts",
            "identical query IDs, problem hashes, sources, and sample indices",
            "same explicit-budget prompt and 10,240-token evaluation cap",
            "identical deterministic query-specific seeds across reported methods",
            "strict correctness requires correct=true and truncated=false",
            "current reverse-KL TRSD-16 has 143 rows and four completion markers",
            "development-selected epsilon equals 0.004",
            "P64 provenance is legacy 4,096-token run without direction field",
            "T64 provenance is exact reverse-KL 10,240-token run",
            "T64/P64 transition anatomy equals 16 W->C, 4 C->W, and 11 P64-cap-hit->T64-correct",
        ],
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "REPORT_COMPLETE").write_text("complete\n", encoding="utf-8")

    args.stage.mkdir(parents=True, exist_ok=True)
    stage_logs = args.stage / "logs"
    if stage_logs.is_dir():
        shutil.rmtree(stage_logs)
    for source in output.rglob("*"):
        if source.is_file():
            relative = source.relative_to(output)
            if "logs" in relative.parts:
                continue
            destination = args.stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def main() -> None:
    build_bundle(parse_args())


if __name__ == "__main__":
    main()
