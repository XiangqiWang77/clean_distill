#!/usr/bin/env python3
"""Measure matched pre-update student NLL on one common ordinary rollout.

At each training step, the retained Privilege-SD ordinary (non-privileged)
on-policy response is the frozen common sequence.  Its journaled NLL is the
Privilege-SD score.  A deterministic TRSD replay scores that exact prompt and
response before its corresponding update.  Thus both curves use identical
tokens and prefixes instead of comparing method-specific training objectives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping

import torch

from src.clean_self_distill import persistent
from src.clean_self_distill.generation import problem_prompt
from src.clean_self_distill.io import canonical_json_sha256
from src.clean_self_distill.persistent import PersistentConfig, load_persistent_inputs
from src.clean_self_distill.runtime import (
    backbone_forward,
    collect_runtime_metadata,
    input_device,
    load_hf_model,
    project_logits,
)
from src.clean_self_distill.streaming_distill import _realized_logprob_sum


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def token_ids_sha256(value: torch.Tensor) -> str:
    ids = value.detach().cpu().reshape(-1).tolist()
    return hashlib.sha256(",".join(map(str, ids)).encode("utf-8")).hexdigest()


@torch.inference_mode()
def common_sequence_nll(
    model,
    tokenizer,
    query: Mapping[str, str],
    response_token_ids: list[int],
    *,
    chunk_size: int = 128,
) -> tuple[float, int, str]:
    was_training = model.training
    model.eval()
    device = input_device(model)
    prompt = problem_prompt(tokenizer, str(query["problem"]))
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
        "input_ids"
    ].to(device)
    response_ids = torch.tensor(
        response_token_ids, dtype=torch.long, device=device
    ).unsqueeze(0)
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    hidden, _ = backbone_forward(
        model,
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        use_cache=False,
    )
    start = int(prompt_ids.shape[1]) - 1
    response_hidden = hidden[:, start : start + response_ids.shape[1]]
    logprob_sum = 0.0
    for offset in range(0, int(response_ids.shape[1]), chunk_size):
        stop = min(offset + chunk_size, int(response_ids.shape[1]))
        logits = project_logits(model, response_hidden[:, offset:stop])
        logprob_sum += _realized_logprob_sum(logits, response_ids[:, offset:stop])
        del logits
    nll = -logprob_sum / int(response_ids.shape[1])
    context_digest = token_ids_sha256(full_ids)
    del response_hidden, hidden, full_ids, response_ids, prompt_ids
    model.train(was_training)
    return nll, len(response_token_ids), context_digest


def rolling_mean(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
        else:
            current = values[index + 1 - window : index + 1]
            result.append(sum(current) / window)
    return result


def write_outputs(
    output_dir: Path,
    privileged_rows: list[dict[str, Any]],
    trsd_rows: list[dict[str, Any]],
) -> None:
    indexed = {int(row["episode"]): row for row in trsd_rows}
    if sorted(indexed) != list(range(1, 65)):
        raise RuntimeError("common-evaluation TRSD metrics are not complete for steps 1..64")
    privilege_nll = [-float(row["student_normalized_logprob"]) for row in privileged_rows]
    trsd_nll = [float(indexed[step]["trsd_common_nll"]) for step in range(1, 65)]
    p_roll = rolling_mean(privilege_nll, 8)
    t_roll = rolling_mean(trsd_nll, 8)
    rows: list[dict[str, Any]] = []
    for step, (p_row, t_row) in enumerate(zip(privileged_rows, trsd_rows), 1):
        if p_row["query_id"] != t_row["query_id"]:
            raise RuntimeError(f"common-evaluation query mismatch at step {step}")
        rows.append(
            {
                "training_step": step,
                "query_id": p_row["query_id"],
                "common_sequence_source": "privilege_sd_ordinary_on_policy_pre_update",
                "common_response_tokens": int(p_row["response_tokens"]),
                "privilege_sd_student_token_nll": privilege_nll[step - 1],
                "trsd_student_token_nll": trsd_nll[step - 1],
                "privilege_sd_nll_8step_mean": p_roll[step - 1],
                "trsd_nll_8step_mean": t_roll[step - 1],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "common_evaluation_nll.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
        }
    )
    steps = list(range(1, 65))
    colors = {"Privilege-SD": "#2878B5", "TRSD": "#D9534F"}
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ax.plot(steps, privilege_nll, color=colors["Privilege-SD"], alpha=0.20, linewidth=1.0)
    ax.plot(steps, trsd_nll, color=colors["TRSD"], alpha=0.20, linewidth=1.0)
    ax.plot(steps, [math.nan if value is None else value for value in p_roll], color=colors["Privilege-SD"], linewidth=2.8, label="Privilege-SD · 8-step mean")
    ax.plot(steps, [math.nan if value is None else value for value in t_roll], color=colors["TRSD"], linewidth=2.8, label="TRSD · 8-step mean")
    ax.set_xlabel("Training step (pre-update policy)")
    ax.set_ylabel("Student-token NLL on common ordinary rollout ↓")
    ax.set_title("Common-evaluation NLL rebound", loc="left", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"common_evaluation_nll_rebound.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)

    p_valid = [value for value in p_roll if value is not None]
    t_valid = [value for value in t_roll if value is not None]
    summary = {
        "schema_version": "common-evaluation-nll-dynamics-v1",
        "steps": 64,
        "rolling_window": 8,
        "evaluation_timing": "pre_update",
        "common_sequence": "Privilege-SD ordinary on-policy prompt/response, held fixed across methods at each matched step",
        "privilege_sd": {
            "best_8step_nll": min(p_valid),
            "final_8step_nll": p_valid[-1],
            "rebound_fraction": p_valid[-1] / min(p_valid) - 1.0,
        },
        "trsd": {
            "best_8step_nll": min(t_valid),
            "final_8step_nll": t_valid[-1],
            "rebound_fraction": t_valid[-1] / min(t_valid) - 1.0,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Common-evaluation NLL rebound\n\n"
        "At each matched training step, both pre-update students are scored on the "
        "exact same ordinary, non-privileged prompt and response token sequence. "
        "The common sequence is the retained Privilege-SD ordinary on-policy rollout; "
        "teacher prompts and method-specific distillation losses are never used in this comparison.\n\n"
        "Thin curves are the 64 raw per-step values. Thick curves are trailing 8-step means.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--privileged-journal", type=Path, required=True)
    parser.add_argument("--original-trsd-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-trsd-adapter-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    privileged_rows = read_jsonl(args.privileged_journal)
    if len(privileged_rows) != 64:
        raise RuntimeError("Privilege-SD journal must contain exactly 64 rows")
    queries, hashes = load_persistent_inputs(args.queries, episodes=64)
    for index, (query, row) in enumerate(zip(queries, privileged_rows), 1):
        if query["query_id"] != row.get("query_id"):
            raise RuntimeError(f"query mismatch at step {index}")

    metric_path = args.output_dir / "trsd_common_nll.jsonl"
    existing = read_jsonl(metric_path) if metric_path.is_file() else []
    if len(existing) == 64:
        write_outputs(args.output_dir, privileged_rows, existing)
        (args.output_dir / "DYNAMICS_COMPLETE").write_text(
            "complete\n", encoding="utf-8"
        )
        return

    config = PersistentConfig(
        branch="clean",
        variant="trust_region",
        model=args.model,
        model_id=args.model_id,
        revision=args.revision,
        episodes=64,
        scientific_checkpoints=(0, 16, 32, 48, 64),
        rolling_checkpoint_interval=8,
        max_sequence_tokens=10_240,
        max_rollout_tokens=10_240,
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
        trust_region_kl_budget=0.004,
        trust_region_binary_search_steps=6,
    )
    config.validate()
    hashes["teacher_signal_sha256"] = canonical_json_sha256(
        {"mode": "predecision-exponential-projection-v1"}
    )

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model, tokenizer = load_hf_model(
        args.model,
        dtype="bfloat16",
        device_map="auto",
        attn_implementation="sdpa",
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        training=True,
        revision=None if Path(args.model).exists() else args.revision,
    )
    runtime = collect_runtime_metadata(model, model_path=args.model_id, revision=args.revision)

    metrics = {int(row["episode"]): row for row in existing}
    original_train_one = persistent.train_one_episode

    def measured_train_one_episode(**kwargs: Any) -> dict[str, Any]:
        stream_index = int(kwargs["stream_index"])
        episode = stream_index + 1
        query = kwargs["query"]
        reference = privileged_rows[stream_index]
        nll, token_count, context_digest = common_sequence_nll(
            kwargs["model"],
            kwargs["tokenizer"],
            query,
            [int(value) for value in reference["student_prefix_token_ids"]],
        )
        if context_digest != reference["student_context_sha256"]:
            raise RuntimeError(f"common sequence context hash mismatch at episode {episode}")
        if episode == 1:
            expected = -float(reference["student_normalized_logprob"])
            if abs(nll - expected) > 1e-4:
                raise RuntimeError(f"step-1 common NLL gate failed: {nll} != {expected}")
        metrics[episode] = {
            "schema_version": "common-evaluation-nll-row-v1",
            "episode": episode,
            "query_id": query["query_id"],
            "problem_sha256": query["problem_sha256"],
            "common_context_sha256": context_digest,
            "common_response_tokens": token_count,
            "trsd_common_nll": nll,
            "privilege_sd_common_nll": -float(reference["student_normalized_logprob"]),
        }
        atomic_jsonl(metric_path, [metrics[key] for key in sorted(metrics)])
        return original_train_one(**kwargs)

    persistent.train_one_episode = measured_train_one_episode
    persistent.run_persistent_training(
        model=model,
        tokenizer=tokenizer,
        queries=queries,
        config=config,
        output_dir=args.replay_dir,
        input_hashes=hashes,
        resume=args.resume,
        runtime_metadata=runtime,
    )

    trsd_rows = [metrics[key] for key in sorted(metrics)]
    write_outputs(args.output_dir, privileged_rows, trsd_rows)
    (args.output_dir / "DYNAMICS_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
