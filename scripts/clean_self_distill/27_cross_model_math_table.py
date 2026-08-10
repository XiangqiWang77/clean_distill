#!/usr/bin/env python3
"""Build the complete three-model AMC/AIME accuracy and mechanism table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCES = ("amc23", "aime24", "aime25")
EXPECTED_SOURCE_COUNTS = {"amc23": 83, "aime24": 30, "aime25": 30}


@dataclass(frozen=True)
class Entry:
    model: str
    label: str
    method: str
    episode: int
    scored: Path
    journal: Path | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_metrics(path: Path, episode: int, method: str) -> tuple[float, float]:
    rows = read_jsonl(path)
    selected = [row for row in rows if int(row.get("episode", -1)) <= episode]
    if [int(row.get("episode", -1)) for row in selected] != list(range(1, episode + 1)):
        raise ValueError(f"{path} does not contain the exact episode prefix 1..{episode}")
    for row in selected:
        partition = row.get("style_task_error")
        if not isinstance(partition, dict):
            raise ValueError(f"{path} lacks style_task_error")
        if partition.get("partition_version") != "rlcsd-style-task-v1":
            raise ValueError(f"{path} uses an unexpected style partition")
    if method == "privileged_sd":
        target_kl = statistics.fmean(float(row["mean_teacher_student_kl"]) for row in selected)
    elif method == "trsd":
        achieved = [row.get("trust_region_achieved_kl") for row in selected]
        if any(value is None for value in achieved):
            raise ValueError(f"{path} has a missing projected Target KL")
        target_kl = statistics.fmean(float(value) for value in achieved)
    else:
        raise ValueError(method)
    style_sum = sum(float(row["style_task_error"]["style_abs_error_sum"]) for row in selected)
    style_count = sum(int(row["style_task_error"]["style_token_count"]) for row in selected)
    if style_count <= 0:
        raise ValueError(f"{path} has no style tokens")
    return target_kl, style_sum / style_count


def scored_metrics(entry: Entry) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    rows = [row for row in read_jsonl(entry.scored) if row.get("profile") == "acc1"]
    if len(rows) != 143:
        raise ValueError(f"{entry.scored} has {len(rows)} Acc@1 rows, expected 143")
    if len({str(row.get("query_id", "")) for row in rows}) != 143:
        raise ValueError(f"{entry.scored} has duplicate or missing query IDs")
    if {str(row.get("method")) for row in rows} != {entry.method}:
        raise ValueError(f"{entry.scored} method does not equal {entry.method}")
    if {int(row.get("checkpoint_episode", -1)) for row in rows} != {entry.episode}:
        raise ValueError(f"{entry.scored} checkpoint episode does not equal {entry.episode}")
    counts = {source: sum(str(row.get("source")) == source for row in rows) for source in SOURCES}
    if counts != EXPECTED_SOURCE_COUNTS:
        raise ValueError(f"{entry.scored} source counts are {counts}")
    for row in rows:
        if (
            int(row.get("max_new_tokens", -1)) != 10_240
            or int(row.get("sample_index", -1)) != 0
            or abs(float(row.get("temperature", -1)) - 0.6) > 1e-12
            or abs(float(row.get("top_p", -1)) - 0.95) > 1e-12
            or int(row.get("top_k", -1)) != 20
        ):
            raise ValueError(f"{entry.scored} violates the common 10,240-token sampling protocol")
        hit = int(row.get("generated_tokens", -1)) >= 10_240
        if bool(row.get("truncated")) != hit:
            raise ValueError(f"{entry.scored}/{row.get('query_id')} cap-hit flag disagrees with token count")

    def group(source: str | None) -> tuple[int, int]:
        chosen = rows if source is None else [row for row in rows if row["source"] == source]
        correct = sum(bool(row.get("correct")) and not bool(row.get("truncated")) for row in chosen)
        return correct, len(chosen)

    by_source = {source: group(source) for source in SOURCES}
    combined = group(None)
    cap_hits = sum(int(row["generated_tokens"]) >= 10_240 for row in rows)
    target_kl: float | None = None
    style_drift: float | None = None
    if entry.journal is not None:
        target_kl, style_drift = training_metrics(entry.journal, entry.episode, entry.method)
    record: dict[str, Any] = {
        "model": entry.model,
        "method": entry.label,
        "episodes": entry.episode,
        "amc23_correct": by_source["amc23"][0],
        "amc23_total": by_source["amc23"][1],
        "amc23_strict_acc1_percent": 100 * by_source["amc23"][0] / by_source["amc23"][1],
        "aime24_correct": by_source["aime24"][0],
        "aime24_total": by_source["aime24"][1],
        "aime24_strict_acc1_percent": 100 * by_source["aime24"][0] / by_source["aime24"][1],
        "aime25_correct": by_source["aime25"][0],
        "aime25_total": by_source["aime25"][1],
        "aime25_strict_acc1_percent": 100 * by_source["aime25"][0] / by_source["aime25"][1],
        "combined_correct": combined[0],
        "combined_total": combined[1],
        "combined_strict_acc1_percent": 100 * combined[0] / combined[1],
        "cap_hits": cap_hits,
        "target_kl": target_kl,
        "style_drift": style_drift,
        "scored_path": str(entry.scored),
        "journal_path": None if entry.journal is None else str(entry.journal),
        "scored_sha256": file_sha256(entry.scored),
        "journal_sha256": None if entry.journal is None else file_sha256(entry.journal),
    }
    protocol = {
        str(row["query_id"]): {
            "problem_sha256": str(row["problem_sha256"]),
            "source": str(row["source"]),
            "seed": str(row["seed"]),
        }
        for row in rows
    }
    return record, protocol


def cell(correct: int, total: int) -> str:
    return f"{100 * correct / total:.2f}% ({correct}/{total})"


def mechanism(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def build_entries(args: argparse.Namespace) -> list[Entry]:
    q8_existing = args.qwen8_existing_eval_root
    q8_new = args.qwen8_new_eval_root
    q8_p = args.qwen8_privileged_journal
    q8_t = args.qwen8_trsd_journal
    entries = [
        Entry("Qwen3-8B", "Base", "base", 0, q8_existing / "base.jsonl", None),
        Entry("Qwen3-8B", "Privilege-SD", "privileged_sd", 16, q8_existing / "privileged_ep16.jsonl", q8_p),
        Entry("Qwen3-8B", "TRSD", "trsd", 16, q8_existing / "trsd_ep16_rkl_current.jsonl", q8_t),
        Entry("Qwen3-8B", "Privilege-SD", "privileged_sd", 32, q8_new / "privileged_32/scored.jsonl", q8_p),
        Entry("Qwen3-8B", "TRSD", "trsd", 32, q8_new / "trsd_32/scored.jsonl", q8_t),
        Entry("Qwen3-8B", "Privilege-SD", "privileged_sd", 48, q8_new / "privileged_48/scored.jsonl", q8_p),
        Entry("Qwen3-8B", "TRSD", "trsd", 48, q8_new / "trsd_48/scored.jsonl", q8_t),
        Entry("Qwen3-8B", "Privilege-SD", "privileged_sd", 64, q8_existing / "privileged_ep64.jsonl", q8_p),
        Entry("Qwen3-8B", "TRSD", "trsd", 64, q8_existing / "trsd_ep64.jsonl", q8_t),
    ]
    for model, root in (("Qwen3-1.7B", args.qwen17_root), ("GPT-OSS-20B", args.gptoss20_root)):
        entries.append(Entry(model, "Base", "base", 0, root / "eval/base/scored.jsonl", None))
        for method, label in (("privileged", "Privilege-SD"), ("trsd", "TRSD")):
            eval_method = "privileged_sd" if method == "privileged" else "trsd"
            for episode in (16, 64):
                tag = f"{method}_{episode}"
                entries.append(
                    Entry(
                        model,
                        label,
                        eval_method,
                        episode,
                        root / f"eval/{tag}/scored.jsonl",
                        root / f"train/{tag}/episodes.jsonl",
                    )
                )
    if args.srpo_root is not None:
        for model, key in (
            ("Qwen3-8B", "q8"),
            ("Qwen3-1.7B", "q17"),
            ("GPT-OSS-20B", "go20"),
        ):
            for episode in (16, 64):
                entries.append(
                    Entry(
                        model,
                        "SRPO",
                        "srpo",
                        episode,
                        args.srpo_root
                        / f"eval/{key}/episode_{episode:04d}/scored.jsonl",
                        None,
                    )
                )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    runs = Path("/home/da839/scratch_pi_mg269/da839/clean_distill/runs")
    parser.add_argument("--qwen8-existing-eval-root", type=Path, default=runs / "budget-prompt-eval-20260807/partial_scored")
    parser.add_argument("--qwen8-new-eval-root", type=Path, required=True)
    parser.add_argument("--qwen8-privileged-journal", type=Path, default=runs / "csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/episodes.jsonl")
    parser.add_argument("--qwen8-trsd-journal", type=Path, default=runs / "reverse-kl-matched64-20260807/trsd/train/episodes.jsonl")
    parser.add_argument("--qwen17-root", type=Path, default=runs / "qwen3-1.7b-fourway-20260808")
    parser.add_argument("--gptoss20-root", type=Path, default=runs / "gptoss20b-fiveway-20260809")
    parser.add_argument(
        "--srpo-root",
        type=Path,
        help="optional SRPO three-model run root; adds episode-16/64 rows",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    reference_protocol: dict[str, dict[str, str]] | None = None
    for entry in build_entries(args):
        record, protocol = scored_metrics(entry)
        protocol_without_seed = {
            query_id: {key: value for key, value in fields.items() if key != "seed"}
            for query_id, fields in protocol.items()
        }
        if reference_protocol is None:
            reference_protocol = protocol_without_seed
        elif protocol_without_seed != reference_protocol:
            raise ValueError(f"{entry.scored} does not use the common 143 questions")
        records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_fields = [key for key in records[0] if not key.endswith("_path") and not key.endswith("_sha256")]
    with (args.output_dir / "complete_math_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in records)

    lines = [
        "# Complete AMC/AIME table",
        "",
        "| Model | Method | Ep. | AMC23 | AIME24 | AIME25 | Combined Strict Acc@1 | Cap hits | Target KL ↓ | StyleDrift ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['episodes']} | "
            f"{cell(row['amc23_correct'], row['amc23_total'])} | "
            f"{cell(row['aime24_correct'], row['aime24_total'])} | "
            f"{cell(row['aime25_correct'], row['aime25_total'])} | "
            f"{cell(row['combined_correct'], row['combined_total'])} | "
            f"{row['cap_hits']}/143 | {mechanism(row['target_kl'])} | {mechanism(row['style_drift'])} |"
        )
    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- **Cap hits** is the number of the 143 responses that reach the 10,240-token generation limit. A cap-hit response is incorrect under Strict Acc@1 even if the offline parser finds a boxed answer.",
            "- **Target KL** is the full-vocabulary KL between the distillation target distribution q and the current ordinary student distribution p on a fixed identical prefix, averaged over token positions. The table follows the registered run summary convention: per-episode token-position means are averaged through the reported checkpoint.",
            "- **StyleDrift** fixes the pre-registered style-token position set S and reports the pooled |log q_t(y_t) - log p_t(y_t)| over those positions, where y_t is the ordinary student's token on the identical prefix.",
            "- Lower Target KL means a more local target; lower StyleDrift means less transfer of privileged-context expression style. Base has no distillation target, so both fields are undefined.",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "three-model-complete-math-table-v1",
                "queries": 143,
                "sources": EXPECTED_SOURCE_COUNTS,
                "max_new_tokens": 10_240,
                "strict_metric": "boxed_answer_correct AND cap_hit=false",
                "rows": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    qwen8_rows = [row for row in records if row["model"] == "Qwen3-8B"]
    with (args.output_dir / "qwen3_8b_intermediate_checkpoints.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in qwen8_rows)


if __name__ == "__main__":
    main()
