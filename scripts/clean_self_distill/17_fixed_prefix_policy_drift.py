#!/usr/bin/env python3
"""Measure lexicon-free anchored policy drift with full-vocabulary JSD.

For each method, this evaluates the model on an identical set of ordinary-
context prefixes and compares its complete next-token distribution with the
frozen initialization.  Dividing Jensen--Shannon divergence by log(2) gives a
bounded [0, 1] quantity.  No lexical style list or task-token regex is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class Method:
    label: str
    adapter_path: Path


def unwrap_causal_lm(model):
    """Resolve a PEFT-wrapped causal LM without discarding active adapters."""
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def backbone_forward(model, *, input_ids: torch.Tensor) -> torch.Tensor:
    causal_lm = unwrap_causal_lm(model)
    decoder = getattr(causal_lm, "model", None)
    if decoder is None:
        outputs = causal_lm(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]
    outputs = decoder(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    )
    return outputs.last_hidden_state


def project_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    head = unwrap_causal_lm(model).get_output_embeddings()
    if head is None:
        raise ValueError("causal LM has no output projection")
    return head(hidden)


def evaluation_problem_prompt(tokenizer, problem: str, *, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                f"{problem.strip()}\n\nPlease reason step by step. You have a strict "
                f"response budget of at most {int(max_new_tokens):,} generated tokens. "
                "Finish the reasoning and put the final answer within \\boxed{} before "
                "reaching that limit; prioritize completing the answer over extending "
                "the analysis."
            ),
        }
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"USER: {messages[0]['content']}\nASSISTANT:"


def parse_method(value: str) -> Method:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must be LABEL=/path/to/adapter")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser()
    if not label.strip() or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid method adapter: {value}")
    return Method(label.strip(), path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_prediction_files(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            query_id = str(row["query_id"])
            if query_id in rows:
                raise ValueError(f"duplicate base prediction for {query_id}")
            rows[query_id] = row
    return rows


def js_divergence_from_logits(
    base_logits: torch.Tensor, method_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized JSD, base entropy, and method entropy per position."""
    base_logp = F.log_softmax(base_logits.float(), dim=-1)
    method_logp = F.log_softmax(method_logits.float(), dim=-1)
    base_p = base_logp.exp()
    method_p = method_logp.exp()
    log_m = torch.logaddexp(base_logp, method_logp) - math.log(2.0)
    js = 0.5 * (
        (base_p * (base_logp - log_m)).sum(dim=-1)
        + (method_p * (method_logp - log_m)).sum(dim=-1)
    )
    base_entropy = -(base_p * base_logp).sum(dim=-1)
    method_entropy = -(method_p * method_logp).sum(dim=-1)
    return js / math.log(2.0), base_entropy, method_entropy


def selected_logits(model, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    hidden = backbone_forward(model, input_ids=input_ids)
    return project_logits(model, hidden[:, positions, :]).squeeze(0)


def stable_subset(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(str(row["query_id"]).encode()).hexdigest(),
    )
    return ordered[:limit]


def bootstrap_mean_ci(values: list[float], *, seed: int, draws: int) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    return means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]


def score(
    *,
    model,
    tokenizer,
    queries: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    methods: list[Method],
    response_tokens: int,
    stride: int,
) -> list[dict[str, Any]]:
    device = input_device(model)
    results: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries, 1):
        query_id = str(query["query_id"])
        prediction = predictions.get(query_id)
        if prediction is None:
            raise ValueError(f"missing base response for {query_id}")
        max_new_tokens = int(prediction["max_new_tokens"])
        prompt = evaluation_problem_prompt(
            tokenizer, str(query["problem"]), max_new_tokens=max_new_tokens
        )
        prompt_ids = tokenizer(
            prompt, add_special_tokens=True, return_tensors="pt"
        )["input_ids"]
        continuation = tokenizer(
            str(prediction["response"]), add_special_tokens=False, return_tensors="pt"
        )["input_ids"][:, :response_tokens]
        if continuation.shape[1] < stride:
            continue
        input_ids = torch.cat([prompt_ids, continuation], dim=1).to(device)
        prompt_length = int(prompt_ids.shape[1])
        # Logit at prompt_length - 1 + t predicts continuation token t.  The
        # fixed positions are identical for Base and every adapter.
        offsets = list(range(stride - 1, int(continuation.shape[1]), stride))
        positions = torch.tensor(
            [prompt_length - 1 + offset for offset in offsets],
            device=device,
            dtype=torch.long,
        )

        with torch.inference_mode(), model.disable_adapter():
            base_logits = selected_logits(model, input_ids, positions).detach()

        for method in methods:
            model.set_adapter(method.label)
            with torch.inference_mode():
                method_logits = selected_logits(model, input_ids, positions)
                js, base_h, method_h = js_divergence_from_logits(
                    base_logits, method_logits
                )
            results.append(
                {
                    "query_id": query_id,
                    "source": str(query["source"]),
                    "method": method.label,
                    "anchor_positions": len(offsets),
                    "normalized_jsd": float(js.mean().item()),
                    "base_entropy_nats": float(base_h.mean().item()),
                    "method_entropy_nats": float(method_h.mean().item()),
                    "entropy_retention": float(
                        (method_h.mean() / base_h.mean()).item()
                    ),
                }
            )
            del method_logits, js, base_h, method_h
        del base_logits, input_ids
        if query_index % 4 == 0:
            print(f"scored {query_index}/{len(queries)} anchor queries", flush=True)
    return results


def summarize(rows: list[dict[str, Any]], *, draws: int, seed: int) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    for index, method in enumerate(methods):
        subset = [row for row in rows if row["method"] == method]
        js_values = [float(row["normalized_jsd"]) for row in subset]
        entropy_values = [float(row["entropy_retention"]) for row in subset]
        low, high = bootstrap_mean_ci(js_values, seed=seed + index, draws=draws)
        e_low, e_high = bootstrap_mean_ci(
            entropy_values, seed=seed + 100 + index, draws=draws
        )
        summaries.append(
            {
                "method": method,
                "queries": len(subset),
                "anchor_positions": sum(int(row["anchor_positions"]) for row in subset),
                "normalized_jsd_mean": float(np.mean(js_values)),
                "normalized_jsd_ci_low": low,
                "normalized_jsd_ci_high": high,
                "entropy_retention_mean": float(np.mean(entropy_values)),
                "entropy_retention_ci_low": e_low,
                "entropy_retention_ci_high": e_high,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, summaries: list[dict[str, Any]]) -> None:
    colors = ["#B97925", "#D5533D", "#4169A1", "#557A46"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    x = np.arange(len(summaries))
    labels = [str(row["method"]) for row in summaries]

    js = np.asarray([row["normalized_jsd_mean"] for row in summaries])
    js_low = np.asarray([row["normalized_jsd_ci_low"] for row in summaries])
    js_high = np.asarray([row["normalized_jsd_ci_high"] for row in summaries])
    bars = axes[0].bar(
        x,
        js,
        color=colors[: len(x)],
        width=0.62,
        yerr=np.vstack([js - js_low, js_high - js]),
        capsize=4,
    )
    axes[0].bar_label(bars, labels=[f"{value:.4f}" for value in js], padding=5)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Normalized JS divergence [0, 1] ↓")
    axes[0].set_title("Anchored policy drift", loc="left", fontweight="bold")

    entropy = np.asarray([row["entropy_retention_mean"] for row in summaries])
    e_low = np.asarray([row["entropy_retention_ci_low"] for row in summaries])
    e_high = np.asarray([row["entropy_retention_ci_high"] for row in summaries])
    bars = axes[1].bar(
        x,
        entropy,
        color=colors[: len(x)],
        width=0.62,
        yerr=np.vstack([entropy - e_low, e_high - entropy]),
        capsize=4,
    )
    axes[1].axhline(1.0, color="#5D6874", linestyle="--", linewidth=1.2)
    axes[1].bar_label(bars, labels=[f"{value:.3f}×" for value in entropy], padding=5)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel(r"Entropy retention  $H(\pi_s)/H(\pi_0)$")
    axes[1].set_title("Exploration retained", loc="left", fontweight="bold")

    for axis in axes:
        axis.grid(axis="y", color="#D8DDE3", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Lexicon-free behavioral drift on fixed ordinary-context prefixes",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--base-prediction",
        type=Path,
        action="append",
        required=True,
        help="Base prediction JSONL; repeat for shards",
    )
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--max-queries", type=int, default=32)
    parser.add_argument("--response-tokens", type=int, default=512)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_queries, args.response_tokens, args.stride, args.bootstrap_draws) <= 0:
        raise ValueError("numeric limits must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    first = args.method[0]
    model = PeftModel.from_pretrained(
        base, first.adapter_path, adapter_name=first.label, is_trainable=False
    )
    for method in args.method[1:]:
        model.load_adapter(method.adapter_path, adapter_name=method.label, is_trainable=False)
    model.eval()

    queries = stable_subset(read_jsonl(args.queries), args.max_queries)
    predictions = read_prediction_files(args.base_prediction)
    rows = score(
        model=model,
        tokenizer=tokenizer,
        queries=queries,
        predictions=predictions,
        methods=args.method,
        response_tokens=args.response_tokens,
        stride=args.stride,
    )
    summaries = summarize(rows, draws=args.bootstrap_draws, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "anchored_policy_jsd_per_query.csv", rows)
    write_csv(args.output_dir / "anchored_policy_jsd_summary.csv", summaries)
    plot(args.output_dir / "anchored_policy_jsd.png", summaries)
    manifest = {
        "metric": "anchored_policy_jsd_v1",
        "definition": "mean JS(method||base)/log(2) over fixed full-vocabulary next-token distributions",
        "range": [0.0, 1.0],
        "anchors": "base-generated ordinary-context prefixes",
        "model": str(args.model),
        "queries": str(args.queries),
        "base_predictions": [str(path) for path in args.base_prediction],
        "methods": {method.label: str(method.adapter_path) for method in args.method},
        "max_queries": args.max_queries,
        "response_tokens": args.response_tokens,
        "stride": args.stride,
        "bootstrap_draws": args.bootstrap_draws,
        "summary": summaries,
    }
    (args.output_dir / "anchored_policy_jsd_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
