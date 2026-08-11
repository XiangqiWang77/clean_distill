#!/usr/bin/env python3
"""Fast Qwen3-8B verified-CoT target-path study.

This is a frozen-checkpoint mechanism diagnostic, not a training run.  For a
held-out problem i with canonical correct-answer suffix y_i, it compares the
ordinary distribution p_S with three verified-CoT privileged distributions
p_{P,w}.  Raw OPSD uses p_{P,w}; TRSD uses the exponential projection

    q_{alpha,w} \propto p_S**(1-alpha) p_{P,w}**alpha,

where the largest per-query/per-wrapper alpha satisfying mean tokenwise
KL(q || p_S) <= epsilon is selected.  Correct-answer gain is teacher-forced
mean log q(y_i) - log p_S(y_i).  Because the privileged prompt contains a
verified solution and final answer, this is deliberately an oracle positive
control; it is not evidence that answer-free privilege improves accuracy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import pvariance
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from src.clean_self_distill.runtime import (
    backbone_forward,
    collect_runtime_metadata,
    input_device,
    load_hf_model,
    render_chat,
)
from src.clean_self_distill.trust_region_mechanism import (
    evaluate_projection_alphas,
    solve_epsilon_alphas,
)
from src.opsd_format import extract_boxed_answer, strip_legacy_math_prompt


SCHEMA_VERSION = "verified-cot-target-path-v1"
WRAPPER_SET_VERSION = "verified-cot-answer-probe-paraphrases-v1"
WRAPPER_IDS = ("neutral", "terse", "verbose")
ANSWER_INSTRUCTION = (
    "Output only the final answer within \\boxed{}; do not include reasoning or "
    "any other text."
)
WRAPPER_FRAMINGS = {
    "neutral": "Use the following verified worked solution as private guidance.",
    "terse": "Private verified derivation:",
    "verbose": (
        "The private material below is a complete and verified derivation; "
        "consult it only as evidence for the correct result."
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Malformed JSONL: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise ValueError("mean requires finite, nonempty values")
    return math.fsum(numbers) / len(numbers)


def canonical_answer(answer: str) -> str:
    """Return an unwrapped answer to be placed in one canonical boxed suffix."""
    value = str(answer).strip()
    boxed = extract_boxed_answer(value)
    if boxed is not None:
        value = boxed
    value = value.strip().strip("$").strip()
    if not value:
        raise ValueError("empty canonical answer")
    return value


def truncate_reference(tokenizer, solution: str, token_cap: int) -> tuple[str, int, bool]:
    """Keep both derivation start and answer-bearing end under a fixed token cap."""
    ids = tokenizer(str(solution).strip(), add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("empty verified reference solution")
    if len(ids) <= token_cap:
        return str(solution).strip(), len(ids), False
    tail = min(max(32, token_cap // 4), token_cap // 2)
    head = token_cap - tail
    start = tokenizer.decode(ids[:head], skip_special_tokens=False)
    end = tokenizer.decode(ids[-tail:], skip_special_tokens=False)
    text = f"{start}\n[... middle of verified derivation omitted ...]\n{end}"
    return text, token_cap, True


def prompt_templates() -> dict[str, str]:
    ordinary = f"Problem: {{problem}}\n\n{ANSWER_INSTRUCTION}"
    privileged = {
        wrapper: (
            "Problem: {problem}\n\n"
            f"{framing}\n"
            "=== Verified Solution Begin ===\n"
            "{reference_solution}\n\n"
            "Verified final answer: {answer}\n"
            "=== Verified Solution End ===\n\n"
            f"{ANSWER_INSTRUCTION}"
        )
        for wrapper, framing in WRAPPER_FRAMINGS.items()
    }
    return {"ordinary": ordinary, **privileged}


def build_prompts(
    tokenizer,
    *,
    problem: str,
    solution: str,
    answer: str,
) -> tuple[str, dict[str, str]]:
    templates = prompt_templates()
    ordinary_message = templates["ordinary"].format(problem=problem)
    ordinary = render_chat(
        tokenizer,
        [{"role": "user", "content": ordinary_message}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    privileged = {
        wrapper: render_chat(
            tokenizer,
            [
                {
                    "role": "user",
                    "content": templates[wrapper].format(
                        problem=problem,
                        reference_solution=solution,
                        answer=answer,
                    ),
                }
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for wrapper in WRAPPER_IDS
    }
    return ordinary, privileged


def encode_prompt(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    return tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(
        device
    )


@torch.inference_mode()
def answer_hidden(model, prompt_ids: torch.Tensor, answer_ids: torch.Tensor) -> torch.Tensor:
    full = torch.cat([prompt_ids, answer_ids], dim=1)
    hidden, _ = backbone_forward(
        model,
        input_ids=full,
        attention_mask=torch.ones_like(full),
        use_cache=False,
    )
    start = int(prompt_ids.shape[1]) - 1
    result = hidden[:, start : start + int(answer_ids.shape[1])].detach()
    if result.shape[:2] != answer_ids.shape:
        raise ValueError("answer hidden-state alignment failed")
    return result


def evaluate_wrapper(
    *,
    model,
    student_hidden: torch.Tensor,
    privileged_hidden: torch.Tensor,
    answer_ids: torch.Tensor,
    alpha_grid: Sequence[float],
    epsilon: float,
    binary_search_steps: int,
    chunk_size: int,
) -> dict[str, Any]:
    categories = ["task"] * int(answer_ids.shape[1])
    grid = evaluate_projection_alphas(
        model=model,
        student_hidden=student_hidden,
        privileged_hidden=privileged_hidden,
        labels=answer_ids,
        categories=categories,
        alphas=alpha_grid,
        chunk_size=chunk_size,
        capture_trace=True,
    )
    selected_alpha = solve_epsilon_alphas(
        model=model,
        student_hidden=student_hidden,
        privileged_hidden=privileged_hidden,
        labels=answer_ids,
        categories=categories,
        alpha_evaluation=grid,
        epsilon_grid=(epsilon,),
        chunk_size=chunk_size,
        binary_search_steps=binary_search_steps,
    )[epsilon]
    if selected_alpha in grid.summaries:
        projected_summary = grid.summaries[selected_alpha]
        projected_trace = grid.traces[selected_alpha]
    else:
        selected = evaluate_projection_alphas(
            model=model,
            student_hidden=student_hidden,
            privileged_hidden=privileged_hidden,
            labels=answer_ids,
            categories=categories,
            alphas=(selected_alpha,),
            chunk_size=chunk_size,
            capture_trace=True,
        )
        projected_summary = selected.summaries[selected_alpha]
        projected_trace = selected.traces[selected_alpha]
    raw_summary = grid.summaries[1.0]
    raw_trace = grid.traces[1.0]
    return {
        "raw": dict(raw_summary),
        "projected": dict(projected_summary),
        "selected_alpha": float(selected_alpha),
        "constraint_active": bool(selected_alpha < 1.0 - 1e-12),
        "raw_logratio": [float(value) for value in raw_trace["logratio"]],
        "projected_logratio": [
            float(value) for value in projected_trace["logratio"]
        ],
        "student_logprob": [
            float(value) for value in raw_trace["student_logprob"]
        ],
        "curve": {
            str(alpha): {
                "gain": float(grid.summaries[alpha]["normalized_logratio"]),
                "mean_kl": float(grid.summaries[alpha]["mean_kl"]),
            }
            for alpha in alpha_grid
        },
    }


def score_query(
    *,
    model,
    tokenizer,
    query: dict[str, Any],
    label: dict[str, Any],
    study_index: int,
    reference_token_cap: int,
    answer_token_cap: int,
    max_prompt_tokens: int,
    context_window: int,
    alpha_grid: Sequence[float],
    epsilon: float,
    binary_search_steps: int,
    chunk_size: int,
) -> dict[str, Any]:
    query_id = str(query["query_id"])
    problem = strip_legacy_math_prompt(str(query["problem"]))
    answer = canonical_answer(str(label["answer"]))
    solution, reference_tokens, reference_truncated = truncate_reference(
        tokenizer, str(label["reference_solution"]), reference_token_cap
    )
    ordinary_text, privileged_texts = build_prompts(
        tokenizer,
        problem=problem,
        solution=solution,
        answer=answer,
    )
    device = input_device(model)
    ordinary_ids = encode_prompt(tokenizer, ordinary_text, device)
    privileged_ids = {
        wrapper: encode_prompt(tokenizer, text, device)
        for wrapper, text in privileged_texts.items()
    }
    prompt_lengths = {
        "ordinary": int(ordinary_ids.shape[1]),
        **{
            wrapper: int(ids.shape[1])
            for wrapper, ids in privileged_ids.items()
        },
    }
    longest_prompt = max(prompt_lengths.values())
    if longest_prompt > max_prompt_tokens:
        raise ValueError(
            f"{query_id}: prompt has {longest_prompt}>{max_prompt_tokens} tokens"
        )

    answer_suffix = f"\\boxed{{{answer}}}"
    answer_ids = tokenizer(
        answer_suffix, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    answer_tokens = int(answer_ids.shape[1])
    if answer_tokens <= 0 or answer_tokens > answer_token_cap:
        raise ValueError(
            f"{query_id}: canonical answer has {answer_tokens} tokens; cap={answer_token_cap}"
        )
    if longest_prompt + answer_tokens > context_window:
        raise ValueError(f"{query_id}: prompt plus answer exceeds context window")

    student_hidden = answer_hidden(model, ordinary_ids, answer_ids)
    wrappers: list[dict[str, Any]] = []
    raw_vectors: list[list[float]] = []
    projected_vectors: list[list[float]] = []
    student_logprobs: list[float] | None = None
    for wrapper in WRAPPER_IDS:
        privileged_hidden = answer_hidden(model, privileged_ids[wrapper], answer_ids)
        result = evaluate_wrapper(
            model=model,
            student_hidden=student_hidden,
            privileged_hidden=privileged_hidden,
            answer_ids=answer_ids,
            alpha_grid=alpha_grid,
            epsilon=epsilon,
            binary_search_steps=binary_search_steps,
            chunk_size=chunk_size,
        )
        if student_logprobs is None:
            student_logprobs = result.pop("student_logprob")
        else:
            result.pop("student_logprob")
        raw_vectors.append(result.pop("raw_logratio"))
        projected_vectors.append(result.pop("projected_logratio"))
        wrappers.append(
            {
                "wrapper_id": wrapper,
                "prompt_sha256": hashlib.sha256(
                    privileged_texts[wrapper].encode()
                ).hexdigest(),
                "prompt_tokens": prompt_lengths[wrapper],
                **result,
            }
        )
        del privileged_hidden

    if student_logprobs is None:
        raise ValueError(f"{query_id}: no wrapper was evaluated")
    raw_position_variance = [
        pvariance([vector[position] for vector in raw_vectors])
        for position in range(answer_tokens)
    ]
    projected_position_variance = [
        pvariance([vector[position] for vector in projected_vectors])
        for position in range(answer_tokens)
    ]
    raw_gains = [float(wrapper["raw"]["normalized_logratio"]) for wrapper in wrappers]
    projected_gains = [
        float(wrapper["projected"]["normalized_logratio"]) for wrapper in wrappers
    ]
    row = {
        "schema_version": SCHEMA_VERSION,
        "wrapper_set_version": WRAPPER_SET_VERSION,
        "query_id": query_id,
        "study_index": int(study_index),
        "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        "reference_solution_sha256": hashlib.sha256(
            str(label["reference_solution"]).encode()
        ).hexdigest(),
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "answer_suffix_sha256": hashlib.sha256(answer_suffix.encode()).hexdigest(),
        "answer_tokens": answer_tokens,
        "reference_tokens_retained": reference_tokens,
        "reference_truncated": reference_truncated,
        "prompt_tokens": prompt_lengths,
        "primary_epsilon": float(epsilon),
        "student_answer_nll": -mean(student_logprobs),
        "wrappers": wrappers,
        "query_summary": {
            "raw_mean_gain": mean(raw_gains),
            "projected_mean_gain": mean(projected_gains),
            "raw_worst_wrapper_gain": min(raw_gains),
            "projected_worst_wrapper_gain": min(projected_gains),
            "raw_mean_kl": mean(
                float(wrapper["raw"]["mean_kl"]) for wrapper in wrappers
            ),
            "projected_mean_kl": mean(
                float(wrapper["projected"]["mean_kl"]) for wrapper in wrappers
            ),
            "mean_selected_alpha": mean(
                float(wrapper["selected_alpha"]) for wrapper in wrappers
            ),
            "raw_wrapper_variance": mean(raw_position_variance),
            "projected_wrapper_variance": mean(projected_position_variance),
            "all_wrappers_positive_raw": all(value > 0.0 for value in raw_gains),
            "all_wrappers_positive_projected": all(
                value > 0.0 for value in projected_gains
            ),
        },
    }
    del ordinary_ids, privileged_ids, answer_ids, student_hidden
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def bootstrap_mean(
    values: Sequence[float], *, resamples: int, seed: int
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap requires a finite nonempty vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    block = 1000
    for start in range(0, resamples, block):
        stop = min(start + block, resamples)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        estimates[start:stop] = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }


def wrapper_value(row: dict[str, Any], wrapper_id: str, method: str, field: str) -> float:
    wrapper = next(
        item for item in row["wrappers"] if item["wrapper_id"] == wrapper_id
    )
    return float(wrapper[method][field])


def historical_log_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    rows = read_jsonl(path)
    if not rows:
        return None
    row = rows[-1]
    metrics = row.get("objective_metrics", {})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "episodes": len(rows),
        "episode": row.get("episode"),
        "raw_target_kl": metrics.get("raw_target_kl"),
        "projected_target_kl": metrics.get("projected_target_kl"),
        "projection_alpha": metrics.get("projection_alpha"),
        "realized_target_logprob_advantage": metrics.get(
            "realized_target_logprob_advantage"
        ),
        "gradient_norm": row.get("gradient_norm"),
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    alpha_grid: Sequence[float],
    epsilon: float,
    resamples: int,
    seed: int,
    historical_raw: dict[str, Any] | None,
    historical_projected: dict[str, Any] | None,
) -> dict[str, Any]:
    summary_fields = (
        "raw_mean_gain",
        "projected_mean_gain",
        "raw_worst_wrapper_gain",
        "projected_worst_wrapper_gain",
        "raw_mean_kl",
        "projected_mean_kl",
        "mean_selected_alpha",
        "raw_wrapper_variance",
        "projected_wrapper_variance",
    )
    estimates = {
        field: bootstrap_mean(
            [float(row["query_summary"][field]) for row in rows],
            resamples=resamples,
            seed=seed + index,
        )
        for index, field in enumerate(summary_fields)
    }
    rates = {
        "all_wrappers_positive_raw": bootstrap_mean(
            [float(row["query_summary"]["all_wrappers_positive_raw"]) for row in rows],
            resamples=resamples,
            seed=seed + 101,
        ),
        "all_wrappers_positive_projected": bootstrap_mean(
            [
                float(row["query_summary"]["all_wrappers_positive_projected"])
                for row in rows
            ],
            resamples=resamples,
            seed=seed + 102,
        ),
    }
    curves: dict[str, Any] = {}
    for index, alpha in enumerate(alpha_grid):
        gains = []
        kls = []
        for row in rows:
            points = [wrapper["curve"][str(alpha)] for wrapper in row["wrappers"]]
            gains.append(mean(float(point["gain"]) for point in points))
            kls.append(mean(float(point["mean_kl"]) for point in points))
        curves[str(alpha)] = {
            "gain": bootstrap_mean(gains, resamples=resamples, seed=seed + 200 + index),
            "mean_kl": bootstrap_mean(
                kls, resamples=resamples, seed=seed + 300 + index
            ),
        }
    wrapper_estimates: dict[str, Any] = {}
    for index, wrapper_id in enumerate(WRAPPER_IDS):
        wrapper_estimates[wrapper_id] = {
            method: bootstrap_mean(
                [
                    wrapper_value(row, wrapper_id, method, "normalized_logratio")
                    for row in rows
                ],
                resamples=resamples,
                seed=seed + 400 + 10 * index + (method == "projected"),
            )
            for method in ("raw", "projected")
        }
    raw_variance = estimates["raw_wrapper_variance"]["mean"]
    projected_variance = estimates["projected_wrapper_variance"]["mean"]
    retained = projected_variance / raw_variance if raw_variance > 0 else None
    claims = {
        "verified_cot_raw_mean_gain_positive": estimates["raw_mean_gain"]["ci95_low"]
        > 0.0,
        "raw_deviation_exceeds_budget": estimates["raw_mean_kl"]["ci95_low"]
        > epsilon,
        "trsd_mean_gain_positive": estimates["projected_mean_gain"]["ci95_low"]
        > 0.0,
        "trsd_reduces_wrapper_variance": retained is not None and retained < 1.0,
        "all_requested_pattern_holds": False,
    }
    claims["all_requested_pattern_holds"] = all(
        claims[key]
        for key in (
            "verified_cot_raw_mean_gain_positive",
            "raw_deviation_exceeds_budget",
            "trsd_mean_gain_positive",
            "trsd_reduces_wrapper_variance",
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "estimand": {
            "benefit": (
                "mean_t[log q(y*_t | prompt,y*_<t) - "
                "log p_student(y*_t | ordinary,y*_<t)] in nats/token"
            ),
            "deviation": "mean_t KL(q_t || p_student,t), exact full vocabulary",
            "trsd": (
                "q_alpha proportional to p_student^(1-alpha) p_privileged^alpha; "
                "largest per-query/per-wrapper alpha with mean KL <= epsilon"
            ),
            "wrapper_variance": (
                "mean_t population-variance_w of correct-answer token log-prob gain"
            ),
            "positive_query": "correct-answer gain > 0 under all three wrappers",
        },
        "scope": (
            "Frozen-checkpoint oracle positive-control: privilege contains a verified "
            "solution and final answer; no optimizer update and no accuracy claim."
        ),
        "queries": len(rows),
        "answer_tokens": sum(int(row["answer_tokens"]) for row in rows),
        "wrappers": list(WRAPPER_IDS),
        "epsilon": float(epsilon),
        "estimates": estimates,
        "positive_query_rates": rates,
        "wrapper_estimates": wrapper_estimates,
        "curve": curves,
        "wrapper_variance_retained": retained,
        "claims": claims,
        "historical_one_step_sanity_check": {
            "scope_warning": (
                "These existing training logs score the on-policy rollout, not the "
                "canonical correct-answer suffix, and are reported separately."
            ),
            "opsd_raw": historical_raw,
            "opsd_projected": historical_projected,
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csvs(run_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    per_query = run_root / "per_query.csv"
    fields = [
        "query_id",
        "answer_tokens",
        "raw_mean_gain",
        "projected_mean_gain",
        "raw_worst_wrapper_gain",
        "projected_worst_wrapper_gain",
        "raw_mean_kl",
        "projected_mean_kl",
        "mean_selected_alpha",
        "raw_wrapper_variance",
        "projected_wrapper_variance",
        "all_wrappers_positive_raw",
        "all_wrappers_positive_projected",
    ]
    with per_query.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "answer_tokens": row["answer_tokens"],
                    **row["query_summary"],
                }
            )
    curve_path = run_root / "target_path.csv"
    with curve_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alpha",
                "gain_mean",
                "gain_ci95_low",
                "gain_ci95_high",
                "kl_mean",
                "kl_ci95_low",
                "kl_ci95_high",
            ]
        )
        for alpha, point in summary["curve"].items():
            writer.writerow(
                [
                    alpha,
                    point["gain"]["mean"],
                    point["gain"]["ci95_low"],
                    point["gain"]["ci95_high"],
                    point["mean_kl"]["mean"],
                    point["mean_kl"]["ci95_low"],
                    point["mean_kl"]["ci95_high"],
                ]
            )


def ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def plot_figures(run_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
        }
    )
    raw_color = "#D55E00"
    trsd_color = "#0072B2"
    gray = "#6B7280"
    alphas = [float(value) for value in summary["curve"]]
    gain = [summary["curve"][str(alpha)]["gain"]["mean"] for alpha in alphas]
    gain_low = [
        summary["curve"][str(alpha)]["gain"]["ci95_low"] for alpha in alphas
    ]
    gain_high = [
        summary["curve"][str(alpha)]["gain"]["ci95_high"] for alpha in alphas
    ]
    kl = [summary["curve"][str(alpha)]["mean_kl"]["mean"] for alpha in alphas]
    kl_low = [
        summary["curve"][str(alpha)]["mean_kl"]["ci95_low"] for alpha in alphas
    ]
    kl_high = [
        summary["curve"][str(alpha)]["mean_kl"]["ci95_high"] for alpha in alphas
    ]
    trsd_gain = summary["estimates"]["projected_mean_gain"]["mean"]
    trsd_kl = summary["estimates"]["projected_mean_kl"]["mean"]
    raw_gain = summary["estimates"]["raw_mean_gain"]["mean"]
    raw_kl = summary["estimates"]["raw_mean_kl"]["mean"]
    epsilon = float(summary["epsilon"])

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.55))
    axes[0].plot(alphas, gain, marker="o", color=raw_color, label="OPSD target path")
    axes[0].fill_between(alphas, gain_low, gain_high, color=raw_color, alpha=0.16)
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].scatter(
        [summary["estimates"]["mean_selected_alpha"]["mean"]],
        [trsd_gain],
        s=75,
        marker="*",
        color=trsd_color,
        zorder=5,
        label="TRSD (per-query $\\alpha^*$)",
    )
    axes[0].set(xlabel="Privileged target strength $\\alpha$", ylabel="Correct-answer gain (nats/token)")
    axes[0].set_title("(a) Verified CoT supplies benefit")
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(alphas, kl, marker="o", color=raw_color)
    axes[1].fill_between(alphas, kl_low, kl_high, color=raw_color, alpha=0.16)
    axes[1].axhline(epsilon, color=trsd_color, linestyle="--", label=f"TRSD budget $\\epsilon$={epsilon:g}")
    axes[1].scatter(
        [summary["estimates"]["mean_selected_alpha"]["mean"]],
        [trsd_kl],
        s=75,
        marker="*",
        color=trsd_color,
        zorder=5,
    )
    axes[1].set_yscale("symlog", linthresh=max(epsilon / 20.0, 1e-8))
    axes[1].set(xlabel="Privileged target strength $\\alpha$", ylabel="$\\mathrm{KL}(q_\\alpha\\,\\|\\,p_S)$ (nats/token)")
    axes[1].set_title("(b) Unconstrained deviation grows")
    axes[1].legend(frameon=False, loc="best")

    axes[2].plot(kl, gain, marker="o", color=gray, linewidth=1.4, label="Shared target path")
    for alpha, x, y in zip(alphas, kl, gain, strict=True):
        if alpha in {0.0, 0.25, 0.5, 1.0}:
            axes[2].annotate(f"$\\alpha$={alpha:g}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[2].scatter([raw_kl], [raw_gain], s=62, color=raw_color, label="OPSD ($\\alpha$=1)", zorder=5)
    axes[2].scatter([trsd_kl], [trsd_gain], s=90, marker="*", color=trsd_color, label="TRSD", zorder=6)
    axes[2].axvline(epsilon, color=trsd_color, linestyle="--", linewidth=1)
    axes[2].set_xscale("symlog", linthresh=max(epsilon / 20.0, 1e-8))
    axes[2].set(xlabel="Deviation (mean KL)", ylabel="Correct-answer gain (nats/token)")
    axes[2].set_title("(c) TRSD stops on the stable frontier")
    axes[2].legend(frameon=False, loc="best")
    fig.suptitle(
        f"Frozen Qwen3-8B · verified-CoT oracle control · N={len(rows)} held-out queries",
        y=1.03,
        fontsize=12,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(run_root / f"figure_positive_cot_target_path.{suffix}")
    plt.close(fig)

    raw_variance = np.asarray(
        [row["query_summary"]["raw_wrapper_variance"] for row in rows], dtype=float
    )
    projected_variance = np.asarray(
        [row["query_summary"]["projected_wrapper_variance"] for row in rows],
        dtype=float,
    )
    floor = 1e-12
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.55))
    axes[0].scatter(
        raw_variance + floor,
        projected_variance + floor,
        s=18,
        alpha=0.55,
        color=trsd_color,
        linewidths=0,
    )
    limits = [
        min(float((raw_variance + floor).min()), float((projected_variance + floor).min())),
        max(float((raw_variance + floor).max()), float((projected_variance + floor).max())),
    ]
    axes[0].plot(limits, limits, linestyle="--", color=gray, linewidth=1, label="equal variance")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="OPSD wrapper variance", ylabel="TRSD wrapper variance")
    axes[0].set_title("(a) Query-level paired comparison")
    below = float(np.mean(projected_variance < raw_variance))
    axes[0].text(0.04, 0.96, f"{100 * below:.1f}% below diagonal", transform=axes[0].transAxes, va="top")
    axes[0].legend(frameon=False, loc="lower right")

    for values, label, color in (
        (raw_variance + floor, "OPSD", raw_color),
        (projected_variance + floor, "TRSD", trsd_color),
    ):
        x, y = ecdf(values)
        axes[1].step(x, y, where="post", label=label, color=color, linewidth=2)
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Across-wrapper variance", ylabel="Fraction of queries ≤ x")
    axes[1].set_title("(b) Stability across the population")
    axes[1].legend(frameon=False, loc="lower right")
    retained = summary["wrapper_variance_retained"]
    retained_text = "undefined" if retained is None else f"{100 * retained:.1f}%"
    axes[1].text(0.04, 0.96, f"mean variance retained: {retained_text}", transform=axes[1].transAxes, va="top")

    positions = np.arange(len(WRAPPER_IDS), dtype=float)
    for index, wrapper_id in enumerate(WRAPPER_IDS):
        raw = summary["wrapper_estimates"][wrapper_id]["raw"]
        projected = summary["wrapper_estimates"][wrapper_id]["projected"]
        axes[2].plot(
            [positions[index] - 0.08, positions[index] + 0.08],
            [raw["mean"], projected["mean"]],
            color=gray,
            linewidth=1,
            alpha=0.8,
        )
        axes[2].errorbar(
            positions[index] - 0.08,
            raw["mean"],
            yerr=[[raw["mean"] - raw["ci95_low"]], [raw["ci95_high"] - raw["mean"]]],
            fmt="o",
            color=raw_color,
            capsize=3,
            label="OPSD" if index == 0 else None,
        )
        axes[2].errorbar(
            positions[index] + 0.08,
            projected["mean"],
            yerr=[
                [projected["mean"] - projected["ci95_low"]],
                [projected["ci95_high"] - projected["mean"]],
            ],
            fmt="o",
            color=trsd_color,
            capsize=3,
            label="TRSD" if index == 0 else None,
        )
    axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axes[2].set_xticks(positions, WRAPPER_IDS)
    axes[2].set(ylabel="Correct-answer gain (nats/token)")
    axes[2].set_title("(c) Three prompt phrasings")
    axes[2].legend(frameon=False, loc="best")
    fig.suptitle(
        f"Multiple-prompt stability of TRSD · N={len(rows)} × {len(WRAPPER_IDS)} wrappers",
        y=1.03,
        fontsize=12,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(run_root / f"figure_multiple_prompt_stability.{suffix}")
    plt.close(fig)


def write_summary_markdown(run_root: Path, summary: dict[str, Any]) -> None:
    estimates = summary["estimates"]
    rates = summary["positive_query_rates"]
    retained = summary["wrapper_variance_retained"]
    retained_text = "undefined" if retained is None else f"{100 * retained:.1f}%"

    def interval(name: str) -> str:
        value = estimates[name]
        return (
            f"{value['mean']:.5f} "
            f"[{value['ci95_low']:.5f}, {value['ci95_high']:.5f}]"
        )

    raw_rate = rates["all_wrappers_positive_raw"]
    projected_rate = rates["all_wrappers_positive_projected"]
    historical = summary["historical_one_step_sanity_check"]
    lines = [
        "# Verified-CoT target-path empirical study",
        "",
        f"Frozen Qwen3-8B; {summary['queries']} held-out queries; "
        f"{summary['answer_tokens']} teacher-forced correct-answer tokens; three wrappers.",
        "",
        "This is an **oracle positive-control mechanism diagnostic**: each privileged "
        "prompt contains a verified reference derivation and the correct final answer. "
        "It does not estimate answer-free generalization or post-training accuracy.",
        "",
        "## Primary estimates (query bootstrap 95% CI)",
        "",
        f"- OPSD correct-answer gain: {interval('raw_mean_gain')} nats/token.",
        f"- TRSD correct-answer gain: {interval('projected_mean_gain')} nats/token.",
        f"- OPSD deviation: {interval('raw_mean_kl')} mean KL.",
        f"- TRSD deviation: {interval('projected_mean_kl')} mean KL "
        f"at epsilon={summary['epsilon']:g}.",
        f"- Mean TRSD alpha: {interval('mean_selected_alpha')}.",
        f"- Across-wrapper variance retained: {retained_text}.",
        f"- Queries positive under all wrappers: OPSD {100 * raw_rate['mean']:.1f}% "
        f"[{100 * raw_rate['ci95_low']:.1f}, {100 * raw_rate['ci95_high']:.1f}]%; "
        f"TRSD {100 * projected_rate['mean']:.1f}% "
        f"[{100 * projected_rate['ci95_low']:.1f}, {100 * projected_rate['ci95_high']:.1f}]%.",
        "",
        "## Predeclared claim checks",
        "",
    ]
    lines.extend(
        f"- {key}: **{'PASS' if value else 'FAIL'}**"
        for key, value in summary["claims"].items()
    )
    lines.extend(
        [
            "",
            "## Reused one-step training logs",
            "",
            historical["scope_warning"],
            "",
            "```json",
            json.dumps(
                {
                    "opsd_raw": historical["opsd_raw"],
                    "opsd_projected": historical["opsd_projected"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Figure captions",
            "",
            "**Figure 1 — Benefit–deviation path.** Along the frozen-model exponential "
            "target path, stronger use of verified CoT changes both correct-answer "
            "log probability and full-vocabulary KL. Raw OPSD is alpha=1; TRSD chooses "
            "the largest per-query/per-wrapper alpha within the fixed KL budget. Error "
            "bands are query-bootstrap 95% confidence intervals.",
            "",
            "**Figure 2 — Multiple-prompt stability.** Each query is evaluated under "
            "three semantically matched wrappers around the same verified solution. "
            "Variance is computed across wrappers at each answer-token position and "
            "then averaged within query. Points and intervals in panel (c) are "
            "query-bootstrap means and 95% confidence intervals.",
            "",
        ]
    )
    (run_root / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--num-queries", type=int, default=128)
    parser.add_argument("--reference-token-cap", type=int, default=512)
    parser.add_argument("--answer-token-cap", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--alpha-grid", default="0,0.125,0.25,0.5,0.75,1")
    parser.add_argument("--epsilon", type=float, default=0.004)
    parser.add_argument("--binary-search-steps", type=int, default=6)
    parser.add_argument("--full-vocab-chunk-size", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--historical-raw-log", type=Path)
    parser.add_argument("--historical-projected-log", type=Path)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reuse run_root/evidence.jsonl and rebuild summaries/figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be positive")
    alpha_grid = tuple(float(piece) for piece in args.alpha_grid.split(","))
    if (
        not alpha_grid
        or alpha_grid[0] != 0.0
        or alpha_grid[-1] != 1.0
        or any(right <= left for left, right in zip(alpha_grid, alpha_grid[1:]))
    ):
        raise ValueError("--alpha-grid must be increasing and include 0 and 1")
    args.run_root.mkdir(parents=True, exist_ok=True)
    evidence_path = args.run_root / "evidence.jsonl"
    queries = sorted(read_jsonl(args.queries), key=lambda row: str(row["query_id"]))
    labels = {str(row["query_id"]): row for row in read_jsonl(args.labels)}
    selected_queries = queries[: args.num_queries]
    if len(selected_queries) != args.num_queries:
        raise ValueError(
            f"requested {args.num_queries} queries, found {len(selected_queries)}"
        )
    missing = [row["query_id"] for row in selected_queries if row["query_id"] not in labels]
    if missing:
        raise ValueError(f"missing labels for {len(missing)} selected queries")

    existing_rows = read_jsonl(evidence_path) if evidence_path.exists() else []
    by_id = {str(row["query_id"]): row for row in existing_rows}
    selected_ids = [str(row["query_id"]) for row in selected_queries]
    unexpected = sorted(set(by_id) - set(selected_ids))
    if unexpected:
        raise ValueError(f"evidence contains {len(unexpected)} unexpected query IDs")

    model = None
    tokenizer = None
    runtime: dict[str, Any] | None = None
    if not args.plot_only and len(by_id) < len(selected_queries):
        model, tokenizer = load_hf_model(
            args.model,
            dtype=args.dtype,
            device_map="auto",
            attn_implementation=args.attn_implementation,
            revision=args.revision,
        )
        runtime = collect_runtime_metadata(
            model, model_path=args.model, revision=args.revision
        )
        started = time.monotonic()
        with evidence_path.open("a") as handle:
            for index, query in enumerate(selected_queries):
                query_id = str(query["query_id"])
                if query_id in by_id:
                    continue
                row = score_query(
                    model=model,
                    tokenizer=tokenizer,
                    query=query,
                    label=labels[query_id],
                    study_index=index,
                    reference_token_cap=args.reference_token_cap,
                    answer_token_cap=args.answer_token_cap,
                    max_prompt_tokens=args.max_prompt_tokens,
                    context_window=args.context_window,
                    alpha_grid=alpha_grid,
                    epsilon=args.epsilon,
                    binary_search_steps=args.binary_search_steps,
                    chunk_size=args.full_vocab_chunk_size,
                )
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                by_id[query_id] = row
                completed = len(by_id)
                elapsed = time.monotonic() - started
                if completed == 1 or completed % 4 == 0:
                    rate = elapsed / max(1, completed - len(existing_rows))
                    remaining = rate * (len(selected_queries) - completed)
                    print(
                        f"[{completed}/{len(selected_queries)}] {query_id} "
                        f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m",
                        flush=True,
                    )
    elif args.plot_only and len(by_id) < len(selected_queries):
        raise ValueError(
            f"--plot-only requires {len(selected_queries)} rows; found {len(by_id)}"
        )

    rows = [by_id[query_id] for query_id in selected_ids]
    if len(rows) != len(selected_queries):
        raise ValueError("incomplete evidence after scoring")
    historical_raw = historical_log_summary(args.historical_raw_log)
    historical_projected = historical_log_summary(args.historical_projected_log)
    summary = summarize(
        rows,
        alpha_grid=alpha_grid,
        epsilon=args.epsilon,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        historical_raw=historical_raw,
        historical_projected=historical_projected,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "model_id": args.model_id,
        "model_path": args.model,
        "model_revision": args.revision,
        "queries_path": str(args.queries),
        "queries_sha256": sha256_file(args.queries),
        "labels_path": str(args.labels),
        "labels_sha256": sha256_file(args.labels),
        "selection": "lexicographic query_id prefix; label-independent",
        "selected_query_ids_sha256": hashlib.sha256(
            ("\n".join(selected_ids) + "\n").encode()
        ).hexdigest(),
        "num_queries": len(rows),
        "alpha_grid": list(alpha_grid),
        "epsilon": args.epsilon,
        "binary_search_steps": args.binary_search_steps,
        "full_vocabulary_exact": True,
        "reference_token_cap": args.reference_token_cap,
        "answer_token_cap": args.answer_token_cap,
        "wrapper_set_version": WRAPPER_SET_VERSION,
        "prompt_templates": prompt_templates(),
        "runtime": runtime,
    }
    write_json(args.run_root / "manifest.json", manifest)
    write_json(args.run_root / "summary.json", summary)
    write_csvs(args.run_root, rows, summary)
    plot_figures(args.run_root, rows, summary)
    write_summary_markdown(args.run_root, summary)
    write_json(
        args.run_root / "COMPLETE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "queries": len(rows),
            "claims": summary["claims"],
            "summary_sha256": sha256_file(args.run_root / "summary.json"),
            "figure_1_sha256": sha256_file(
                args.run_root / "figure_positive_cot_target_path.pdf"
            ),
            "figure_2_sha256": sha256_file(
                args.run_root / "figure_multiple_prompt_stability.pdf"
            ),
        },
    )
    print(json.dumps(summary["claims"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
