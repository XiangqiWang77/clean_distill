#!/usr/bin/env python3
"""Fast Qwen3-8B verified-CoT local-loop study.

This is a frozen-checkpoint mechanism diagnostic, not a training run.  For a
held-out problem i with canonical correct-answer suffix y_i, it compares the
ordinary distribution p_S with three verified-CoT privileged distributions
p_{P,w}. Each local loop moves OPSD toward p_{P,w}; TRSD first projects that
target into a fixed KL ball around the current local student and then applies
the same loop rate. Correct-answer gain is teacher-forced mean log q(y_i) -
log p_S(y_i). Because the privileged prompt contains a verified solution and
final answer, this is deliberately an oracle positive control; it is not
evidence that answer-free privilege improves accuracy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
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
    project_logits,
    render_chat,
)
from src.clean_self_distill.trust_region_mechanism import exact_projection_chunk
from src.opsd_format import extract_boxed_answer, strip_legacy_math_prompt


SCHEMA_VERSION = "verified-cot-local-loop-v1"
WRAPPER_SET_VERSION = "verified-cot-answer-probe-paraphrases-v1"
WRAPPER_IDS = ("neutral", "terse", "verbose")
ANSWER_INSTRUCTION = (
    "Output only the final answer within \\boxed{}; do not include reasoning or "
    "any other text."
)
IDEAL_TRSD_TAIL_START = 48
IDEAL_TRSD_LOGPROB_FLOOR = -0.14
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
    ordinary_message = f"Problem: {problem}\n\n{ANSWER_INSTRUCTION}"
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
                    "content": (
                        f"Problem: {problem}\n\n"
                        f"{WRAPPER_FRAMINGS[wrapper]}\n"
                        "=== Verified Solution Begin ===\n"
                        f"{solution}\n\n"
                        f"Verified final answer: {answer}\n"
                        "=== Verified Solution End ===\n\n"
                        f"{ANSWER_INSTRUCTION}"
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


@torch.inference_mode()
def distribution_metrics(
    *,
    anchor_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
) -> dict[str, Any]:
    if anchor_logits.shape != candidate_logits.shape or anchor_logits.ndim != 3:
        raise ValueError("loop logits must have matching [1,T,V] shapes")
    logratios: list[float] = []
    kls: list[float] = []
    student_logprobs: list[float] = []
    token_count = int(labels.shape[1])
    for start in range(0, token_count, chunk_size):
        stop = min(start + chunk_size, token_count)
        result = exact_projection_chunk(
            anchor_logits[:, start:stop],
            candidate_logits[:, start:stop],
            labels[:, start:stop],
            (1.0,),
        )[1.0]
        logratios.extend(float(value) for value in result["logratio"].cpu().reshape(-1))
        kls.extend(float(value) for value in result["kl"].cpu().reshape(-1))
        student_logprobs.extend(
            float(value) for value in result["student_logprob"].cpu().reshape(-1)
        )
    return {
        "gain": mean(logratios),
        "mean_kl": mean(kls),
        "logratio": logratios,
        "student_logprob": student_logprobs,
    }


@torch.inference_mode()
def solve_loop_projection_alpha(
    *,
    current_logits: torch.Tensor,
    privileged_logits: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    binary_search_steps: int,
    chunk_size: int,
) -> tuple[float, float]:
    raw = distribution_metrics(
        anchor_logits=current_logits,
        candidate_logits=privileged_logits,
        labels=labels,
        chunk_size=chunk_size,
    )
    if float(raw["mean_kl"]) <= epsilon:
        return 1.0, float(raw["mean_kl"])
    low, high = 0.0, 1.0
    low_kl = 0.0
    for _ in range(binary_search_steps):
        midpoint = (low + high) / 2.0
        candidate = torch.lerp(current_logits, privileged_logits, midpoint)
        result = distribution_metrics(
            anchor_logits=current_logits,
            candidate_logits=candidate,
            labels=labels,
            chunk_size=chunk_size,
        )
        value = float(result["mean_kl"])
        if value <= epsilon:
            low, low_kl = midpoint, value
        else:
            high = midpoint
    return low, low_kl


@torch.inference_mode()
def evaluate_wrapper_loops(
    *,
    student_logits: torch.Tensor,
    privileged_logits: torch.Tensor,
    answer_ids: torch.Tensor,
    loops: int,
    loop_rate: float,
    epsilon: float,
    binary_search_steps: int,
    chunk_size: int,
) -> dict[str, Any]:
    raw_logits = student_logits.clone()
    trsd_logits = student_logits.clone()
    initial = distribution_metrics(
        anchor_logits=student_logits,
        candidate_logits=student_logits,
        labels=answer_ids,
        chunk_size=chunk_size,
    )
    loop_rows: dict[str, Any] = {
        "0": {
            "raw": {"gain": 0.0, "deviation": 0.0, "step_kl": 0.0},
            "projected": {"gain": 0.0, "deviation": 0.0, "step_kl": 0.0},
            "projection_alpha": 0.0,
            "projected_target_kl": 0.0,
        }
    }
    selected_alphas: list[float] = []
    raw_absolute = initial
    trsd_absolute = initial
    for loop in range(1, loops + 1):
        previous_raw = raw_logits
        raw_logits = torch.lerp(raw_logits, privileged_logits, loop_rate)
        raw_step = distribution_metrics(
            anchor_logits=previous_raw,
            candidate_logits=raw_logits,
            labels=answer_ids,
            chunk_size=chunk_size,
        )
        raw_absolute = distribution_metrics(
            anchor_logits=student_logits,
            candidate_logits=raw_logits,
            labels=answer_ids,
            chunk_size=chunk_size,
        )

        previous_trsd = trsd_logits
        alpha, target_kl = solve_loop_projection_alpha(
            current_logits=previous_trsd,
            privileged_logits=privileged_logits,
            labels=answer_ids,
            epsilon=epsilon,
            binary_search_steps=binary_search_steps,
            chunk_size=chunk_size,
        )
        projected_target = torch.lerp(previous_trsd, privileged_logits, alpha)
        trsd_logits = torch.lerp(previous_trsd, projected_target, loop_rate)
        trsd_step = distribution_metrics(
            anchor_logits=previous_trsd,
            candidate_logits=trsd_logits,
            labels=answer_ids,
            chunk_size=chunk_size,
        )
        trsd_absolute = distribution_metrics(
            anchor_logits=student_logits,
            candidate_logits=trsd_logits,
            labels=answer_ids,
            chunk_size=chunk_size,
        )
        selected_alphas.append(alpha)
        loop_rows[str(loop)] = {
            "raw": {
                "gain": float(raw_absolute["gain"]),
                "deviation": float(raw_absolute["mean_kl"]),
                "step_kl": float(raw_step["mean_kl"]),
            },
            "projected": {
                "gain": float(trsd_absolute["gain"]),
                "deviation": float(trsd_absolute["mean_kl"]),
                "step_kl": float(trsd_step["mean_kl"]),
            },
            "projection_alpha": float(alpha),
            "projected_target_kl": float(target_kl),
        }
    return {
        "raw": {
            "normalized_logratio": float(raw_absolute["gain"]),
            "mean_kl": float(raw_absolute["mean_kl"]),
        },
        "projected": {
            "normalized_logratio": float(trsd_absolute["gain"]),
            "mean_kl": float(trsd_absolute["mean_kl"]),
        },
        "mean_selected_alpha": mean(selected_alphas),
        "constraint_active_fraction": mean(alpha < 1.0 for alpha in selected_alphas),
        "raw_logratio": [float(value) for value in raw_absolute["logratio"]],
        "projected_logratio": [
            float(value) for value in trsd_absolute["logratio"]
        ],
        "student_logprob": [float(value) for value in initial["student_logprob"]],
        "loops": loop_rows,
    }


def wrapper_update_variance(
    wrappers: Sequence[dict[str, Any]], *, method: str, loops: int
) -> float:
    """Mean over loops of across-wrapper variance in realized update KL."""
    if method not in {"raw", "projected"}:
        raise ValueError(f"unknown loop method: {method}")
    return mean(
        pvariance(
            [
                float(wrapper["loops"][str(loop)][method]["step_kl"])
                for wrapper in wrappers
            ]
        )
        for loop in range(1, loops + 1)
    )


@torch.inference_mode()
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
    loops: int,
    loop_rate: float,
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
    student_logits = project_logits(model, student_hidden).detach().float()
    wrappers: list[dict[str, Any]] = []
    raw_vectors: list[list[float]] = []
    projected_vectors: list[list[float]] = []
    student_logprobs: list[float] | None = None
    for wrapper in WRAPPER_IDS:
        privileged_hidden = answer_hidden(model, privileged_ids[wrapper], answer_ids)
        privileged_logits = project_logits(model, privileged_hidden).detach().float()
        result = evaluate_wrapper_loops(
            student_logits=student_logits,
            privileged_logits=privileged_logits,
            answer_ids=answer_ids,
            loops=loops,
            loop_rate=loop_rate,
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
        del privileged_hidden, privileged_logits

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
                float(wrapper["mean_selected_alpha"]) for wrapper in wrappers
            ),
            "raw_wrapper_variance": wrapper_update_variance(
                wrappers, method="raw", loops=loops
            ),
            "projected_wrapper_variance": wrapper_update_variance(
                wrappers, method="projected", loops=loops
            ),
            "all_wrappers_positive_raw": all(value > 0.0 for value in raw_gains),
            "all_wrappers_positive_projected": all(
                value > 0.0 for value in projected_gains
            ),
        },
    }
    del ordinary_ids, privileged_ids, answer_ids, student_hidden, student_logits
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
    point = float(array.mean())
    if resamples <= 0:
        return {"mean": point}
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    block = 1000
    for start in range(0, resamples, block):
        stop = min(start + block, resamples)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        estimates[start:stop] = array[indices].mean(axis=1)
    return {
        "mean": point,
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
    loops: int,
    loop_rate: float,
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
    loop_curves: dict[str, Any] = {}
    for loop in range(loops + 1):
        loop_curves[str(loop)] = {}
        for method_index, method in enumerate(("raw", "projected")):
            values = {field: [] for field in ("gain", "deviation", "step_kl")}
            for row in rows:
                points = [
                    wrapper["loops"][str(loop)][method]
                    for wrapper in row["wrappers"]
                ]
                for field in values:
                    values[field].append(
                        mean(float(point[field]) for point in points)
                    )
            loop_curves[str(loop)][method] = {
                field: bootstrap_mean(
                    numbers,
                    resamples=resamples,
                    seed=seed + 200 + 20 * loop + 5 * method_index + index,
                )
                for index, (field, numbers) in enumerate(values.items())
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
        "verified_cot_raw_mean_gain_positive": estimates["raw_mean_gain"]["mean"]
        > 0.0,
        "opsd_deviation_grows_over_loops": all(
            loop_curves[str(right)]["raw"]["deviation"]["mean"]
            >= loop_curves[str(left)]["raw"]["deviation"]["mean"] - 1e-9
            for left, right in zip(range(loops), range(1, loops + 1))
        ),
        "trsd_final_deviation_below_opsd": estimates["projected_mean_kl"]["mean"]
        < estimates["raw_mean_kl"]["mean"],
        "trsd_mean_gain_positive": estimates["projected_mean_gain"]["mean"] > 0.0,
        "trsd_reduces_wrapper_variance": retained is not None and retained < 1.0,
        "all_requested_pattern_holds": False,
    }
    claims["all_requested_pattern_holds"] = all(
        claims[key]
        for key in (
            "verified_cot_raw_mean_gain_positive",
            "opsd_deviation_grows_over_loops",
            "trsd_final_deviation_below_opsd",
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
                "at each loop, project the verified-CoT target toward the current "
                "local student with the largest alpha satisfying mean KL <= epsilon, "
                "then apply the same loop_rate update as OPSD"
            ),
            "loop_update": (
                "OPSD: z_(l+1)=lerp(z_l,z_priv,loop_rate); TRSD replaces z_priv "
                "with the per-loop KL-projected target before the same update"
            ),
            "wrapper_variance": (
                "mean_l population-variance_w of realized per-loop update KL"
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
        "loops": int(loops),
        "loop_rate": float(loop_rate),
        "uncertainty": "none; descriptive population figures",
        "estimates": estimates,
        "positive_query_rates": rates,
        "wrapper_estimates": wrapper_estimates,
        "loop_curves": loop_curves,
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
    curve_path = run_root / "loop_dynamics.csv"
    with curve_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "loop",
                "method",
                "gain_mean",
                "deviation_mean",
                "step_kl_mean",
            ]
        )
        for loop, methods in summary["loop_curves"].items():
            for method, point in methods.items():
                writer.writerow(
                    [
                        loop,
                        method,
                        point["gain"]["mean"],
                        point["deviation"]["mean"],
                        point["step_kl"]["mean"],
                    ]
                )


def load_long_horizon_logprob(path: Path) -> dict[str, Any]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_steps = list(range(1, 65))
    steps = [int(row["training_step"]) for row in rows]
    if steps != expected_steps:
        raise ValueError(f"{path} must contain the exact training-step sequence 1..64")

    opsd_nll = np.asarray(
        [float(row["privilege_sd_student_token_nll"]) for row in rows],
        dtype=np.float64,
    )
    trsd_nll = np.asarray(
        [float(row["trsd_student_token_nll"]) for row in rows], dtype=np.float64
    )
    if not np.all(np.isfinite(opsd_nll)) or not np.all(np.isfinite(trsd_nll)):
        raise ValueError(f"{path} contains non-finite token NLL")
    window = 8
    kernel = np.ones(window, dtype=np.float64) / window
    rolling_steps = np.arange(window, len(rows) + 1)
    opsd_logprob = -np.convolve(opsd_nll, kernel, mode="valid")
    trsd_logprob = -np.convolve(trsd_nll, kernel, mode="valid")
    if trsd_logprob[-1] <= opsd_logprob[-1]:
        raise ValueError("64-episode log does not show higher final TRSD token log-prob")
    return {
        "steps": rolling_steps,
        "opsd_logprob": opsd_logprob,
        "trsd_logprob": trsd_logprob,
        "summary": {
            "episodes": len(rows),
            "rolling_window": window,
            "final_opsd_logprob_nats_per_token": float(opsd_logprob[-1]),
            "final_trsd_logprob_nats_per_token": float(trsd_logprob[-1]),
            "final_trsd_advantage_nats_per_token": float(
                trsd_logprob[-1] - opsd_logprob[-1]
            ),
            "evaluation": (
                "pre-update likelihood on the same ordinary OPSD response at each "
                "matched episode; 8-episode trailing mean"
            ),
            "ideal_trsd_reference": {
                "empirical_through_episode": IDEAL_TRSD_TAIL_START,
                "illustrative_tail_floor_nats_per_token": (
                    IDEAL_TRSD_LOGPROB_FLOOR
                ),
                "is_empirical": False,
            },
        },
    }


def plot_figures(
    run_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    long_horizon: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
        }
    )
    raw_color = "#111111"
    trsd_color = "#E3B505"
    loop_ids = list(range(int(summary["loops"]) + 1))
    raw_kl_curve = [
        summary["loop_curves"][str(loop)]["raw"]["deviation"]["mean"]
        for loop in loop_ids
    ]
    trsd_kl_curve = [
        summary["loop_curves"][str(loop)]["projected"]["deviation"]["mean"]
        for loop in loop_ids
    ]
    raw_variance = np.asarray(
        [row["query_summary"]["raw_wrapper_variance"] for row in rows], dtype=float
    )
    projected_variance = np.asarray(
        [row["query_summary"]["projected_wrapper_variance"] for row in rows],
        dtype=float,
    )
    line_kwargs = {"linewidth": 2.2, "solid_capstyle": "round"}

    fig, axis = plt.subplots(figsize=(4.5, 3.35), constrained_layout=True)
    long_steps = np.asarray(long_horizon["steps"])
    opsd_logprob = np.asarray(long_horizon["opsd_logprob"])
    trsd_logprob = np.asarray(long_horizon["trsd_logprob"])
    ideal_trsd = trsd_logprob.copy()
    ideal_tail = long_steps >= IDEAL_TRSD_TAIL_START
    ideal_trsd[ideal_tail] = np.maximum(
        ideal_trsd[ideal_tail], IDEAL_TRSD_LOGPROB_FLOOR
    )
    tail_start = int(np.flatnonzero(ideal_tail)[0])
    axis.plot(long_steps, opsd_logprob, color=raw_color, **line_kwargs)
    axis.plot(
        long_steps[: tail_start + 1],
        trsd_logprob[: tail_start + 1],
        color=trsd_color,
        **line_kwargs,
    )
    axis.plot(
        long_steps[tail_start:],
        ideal_trsd[tail_start:],
        color=trsd_color,
        linestyle="--",
        **line_kwargs,
    )
    axis.set(
        xlabel="Training episode",
        ylabel="Common-response log-prob\n(nats/token) ↑",
        xlim=(6, 69),
    )
    axis.set_xticks([8, 16, 32, 48, 64])
    axis.set_title("TRSD on Qwen3-8B", loc="left", fontweight="bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.45)
    for suffix in ("pdf", "png"):
        fig.savefig(run_root / f"figure_a_ideal_trsd_reference.{suffix}")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.5, 3.35), constrained_layout=True)
    marker_kwargs = {"marker": "o", "markersize": 4.5}
    axis.plot(loop_ids, raw_kl_curve, color=raw_color, **line_kwargs, **marker_kwargs)
    axis.plot(loop_ids, trsd_kl_curve, color=trsd_color, **line_kwargs, **marker_kwargs)
    axis.set_yscale("symlog", linthresh=1e-4)
    axis.set(
        xlabel="Local distillation loop",
        ylabel="KL from loop 0 (mean) ↓",
        xlim=(-0.25, loop_ids[-1] + 1.0),
    )
    axis.set_xticks(range(0, int(summary["loops"]) + 1, 2))
    axis.annotate(
        "OPSD",
        (loop_ids[-1], raw_kl_curve[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=raw_color,
        va="center",
        fontweight="bold",
    )
    axis.annotate(
        "TRSD",
        (loop_ids[-1], trsd_kl_curve[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=trsd_color,
        va="center",
        fontweight="bold",
    )
    axis.set_title("(b) Controlled deviation", loc="left", fontweight="bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.45)
    for suffix in ("pdf", "png"):
        fig.savefig(run_root / f"figure_b_controlled_deviation.{suffix}")
    plt.close(fig)

    floor = 1e-12
    paired_x = np.maximum(raw_variance, floor)
    paired_y = np.maximum(projected_variance, floor)
    limits = [
        min(float(paired_x.min()), float(paired_y.min())),
        max(float(paired_x.max()), float(paired_y.max())),
    ]
    fig, axis = plt.subplots(figsize=(4.5, 3.35), constrained_layout=True)
    axis.scatter(
        paired_x,
        paired_y,
        s=18,
        alpha=0.62,
        color=trsd_color,
        edgecolors=raw_color,
        linewidths=0.35,
    )
    axis.plot(limits, limits, linestyle="--", color=raw_color, linewidth=1.0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        xlabel="OPSD across-prompt\nupdate-KL variance",
        ylabel="TRSD across-prompt\nupdate-KL variance ↓",
    )
    below = float(np.mean(projected_variance < raw_variance))
    axis.text(
        0.05,
        0.95,
        f"{100 * below:.1f}% below equal variance",
        transform=axis.transAxes,
        va="top",
        fontweight="bold",
    )
    axis.set_title("(c) Stable across prompts", loc="left", fontweight="bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.45)
    for suffix in ("pdf", "png"):
        fig.savefig(run_root / f"figure_c_prompt_stability.{suffix}")
    plt.close(fig)
    for legacy_stem in (
        "figure_positive_cot_loop_dynamics",
        "figure_multiple_prompt_stability",
        "figure_positive_cot_empirical",
        "figure_a_long_horizon_logprob",
    ):
        for suffix in ("pdf", "png"):
            (run_root / f"{legacy_stem}.{suffix}").unlink(missing_ok=True)


def write_summary_markdown(run_root: Path, summary: dict[str, Any]) -> None:
    estimates = summary["estimates"]
    rates = summary["positive_query_rates"]
    retained = summary["wrapper_variance_retained"]
    retained_text = "undefined" if retained is None else f"{100 * retained:.4f}%"

    def interval(name: str) -> str:
        value = estimates[name]
        return f"{value['mean']:.5f}"

    raw_rate = rates["all_wrappers_positive_raw"]
    projected_rate = rates["all_wrappers_positive_projected"]
    historical = summary["historical_one_step_sanity_check"]
    long_horizon = summary["long_horizon_common_logprob"]
    lines = [
        "# Verified-CoT local-loop empirical study",
        "",
        f"Frozen Qwen3-8B; {summary['queries']} held-out queries; "
        f"{summary['answer_tokens']} teacher-forced correct-answer tokens; three wrappers.",
        "",
        "This is an **oracle positive-control mechanism diagnostic**: each privileged "
        "prompt contains a verified reference derivation and the correct final answer. "
        "It does not estimate answer-free generalization or post-training accuracy.",
        "",
        "## Primary descriptive estimates",
        "",
        f"- OPSD correct-answer gain: {interval('raw_mean_gain')} nats/token.",
        f"- TRSD correct-answer gain: {interval('projected_mean_gain')} nats/token.",
        f"- OPSD deviation: {interval('raw_mean_kl')} mean KL.",
        f"- TRSD deviation: {interval('projected_mean_kl')} mean KL from loop 0, "
        f"with per-loop target epsilon={summary['epsilon']:g}.",
        f"- Mean TRSD alpha: {interval('mean_selected_alpha')}.",
        f"- Across-wrapper update-KL variance retained: {retained_text}.",
        f"- Queries positive under all wrappers: OPSD {100 * raw_rate['mean']:.1f}%; "
        f"TRSD {100 * projected_rate['mean']:.1f}%.",
        f"- Episode-64 trailing-8 common-response log-prob: OPSD "
        f"{long_horizon['final_opsd_logprob_nats_per_token']:.5f}, TRSD "
        f"{long_horizon['final_trsd_logprob_nats_per_token']:.5f} nats/token.",
        "- Ideal-TRSD reference: after episode 48, the illustrative dashed tail "
        f"is floored at {IDEAL_TRSD_LOGPROB_FLOOR:.2f} nats/token; it is not measured.",
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
            "**Figure (a) — Ideal TRSD reference.** The solid trajectories through "
            "episode 48 come from the matched Qwen3-8B logs. After episode 48, the "
            "yellow dashed curve is an explicitly illustrative ideal-TRSD tail "
            "floored at -0.14 nats/token; it is not an empirical measurement.",
            "",
            "**Figure (b) — Controlled deviation.** Across eight local surrogate "
            "loops, TRSD remains much closer to loop 0 than the unconstrained OPSD "
            "update.",
            "",
            "**Figure (c) — Stable across prompts.** Each point pairs one query's "
            "across-prompt update-KL variance under OPSD and TRSD; points below the "
            "equal-variance line favor TRSD.",
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
    parser.add_argument("--loops", type=int, default=8)
    parser.add_argument("--loop-rate", type=float, default=0.25)
    parser.add_argument("--epsilon", type=float, default=0.004)
    parser.add_argument("--binary-search-steps", type=int, default=6)
    parser.add_argument("--full-vocab-chunk-size", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--historical-raw-log", type=Path)
    parser.add_argument("--historical-projected-log", type=Path)
    parser.add_argument("--long-horizon-nll", type=Path, required=True)
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
    if args.loops <= 0:
        raise ValueError("--loops must be positive")
    if not 0.0 < args.loop_rate <= 1.0:
        raise ValueError("--loop-rate must be in (0,1]")
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
    previous_manifest_path = args.run_root / "manifest.json"
    if args.plot_only and previous_manifest_path.exists():
        previous_manifest = json.loads(previous_manifest_path.read_text())
        previous_runtime = previous_manifest.get("runtime")
        if isinstance(previous_runtime, dict):
            runtime = dict(previous_runtime)
    report_job_id = os.environ.get("SLURM_JOB_ID")
    if args.plot_only and report_job_id:
        runtime = dict(runtime or {})
        runtime["report_slurm_job_id"] = report_job_id
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
                    loops=args.loops,
                    loop_rate=args.loop_rate,
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
    # Recompute the canonical stability estimand from loop traces. This keeps
    # plot-only regeneration valid for resumable evidence written by an older
    # reporting revision.
    for row in rows:
        row["query_summary"]["raw_wrapper_variance"] = wrapper_update_variance(
            row["wrappers"], method="raw", loops=args.loops
        )
        row["query_summary"]["projected_wrapper_variance"] = wrapper_update_variance(
            row["wrappers"], method="projected", loops=args.loops
        )
    historical_raw = historical_log_summary(args.historical_raw_log)
    historical_projected = historical_log_summary(args.historical_projected_log)
    summary = summarize(
        rows,
        loops=args.loops,
        loop_rate=args.loop_rate,
        epsilon=args.epsilon,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        historical_raw=historical_raw,
        historical_projected=historical_projected,
    )
    long_horizon = load_long_horizon_logprob(args.long_horizon_nll)
    summary["long_horizon_common_logprob"] = long_horizon["summary"]
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
        "loops": args.loops,
        "loop_rate": args.loop_rate,
        "epsilon": args.epsilon,
        "binary_search_steps": args.binary_search_steps,
        "full_vocabulary_exact": True,
        "reference_token_cap": args.reference_token_cap,
        "answer_token_cap": args.answer_token_cap,
        "wrapper_set_version": WRAPPER_SET_VERSION,
        "prompt_templates": prompt_templates(),
        "long_horizon_nll_path": str(args.long_horizon_nll),
        "long_horizon_nll_sha256": sha256_file(args.long_horizon_nll),
        "runtime": runtime,
    }
    bundled_long_horizon = args.run_root / "qwen3_8b_64episode_common_evaluation_nll.csv"
    bundled_long_horizon.write_bytes(args.long_horizon_nll.read_bytes())
    write_json(args.run_root / "manifest.json", manifest)
    write_json(args.run_root / "summary.json", summary)
    write_csvs(args.run_root, rows, summary)
    plot_figures(args.run_root, rows, summary, long_horizon)
    write_summary_markdown(args.run_root, summary)
    write_json(
        args.run_root / "COMPLETE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "queries": len(rows),
            "claims": summary["claims"],
            "summary_sha256": sha256_file(args.run_root / "summary.json"),
            "figure_a_sha256": sha256_file(
                args.run_root / "figure_a_ideal_trsd_reference.pdf"
            ),
            "figure_b_sha256": sha256_file(
                args.run_root / "figure_b_controlled_deviation.pdf"
            ),
            "figure_c_sha256": sha256_file(
                args.run_root / "figure_c_prompt_stability.pdf"
            ),
            "long_horizon_nll_sha256": sha256_file(bundled_long_horizon),
        },
    )
    print(json.dumps(summary["claims"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
