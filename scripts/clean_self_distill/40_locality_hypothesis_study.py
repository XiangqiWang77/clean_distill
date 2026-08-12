#!/usr/bin/env python3
"""Frozen Qwen3-8B empirical chain for the locality hypothesis.

Study 1 holds the model and teacher-forced answer prefix fixed.  For each of
three verified-CoT privileged prompts it follows the exponential path

    q_alpha = softmax((1-alpha) z_student + alpha z_privileged)

and measures correct-answer log-probability gain, exact full-vocabulary KL,
and full-vocabulary total variation (TV).  TV is the primary locality axis:
unlike KL, it is first order near the student and therefore does not create a
linear-gain-versus-quadratic-distance artifact.  Study 2 reuses the existing
controlled distribution-loop traces.  Study 3 reuses frozen strict-Acc@1
training evaluations; it is downstream behavior, not a projection-only
causal ablation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from statistics import pvariance
from types import ModuleType
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.clean_self_distill.runtime import (
    collect_runtime_metadata,
    input_device,
    load_hf_model,
    project_logits,
)


SCHEMA_VERSION = "qwen3-8b-locality-hypothesis-v1"
DEFAULT_ALPHA_GRID = (
    0.0,
    0.0078125,
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.1875,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
)
RAW_COLOR = "#111111"
LOCAL_COLOR = "#E3B505"


def load_positive_cot_helpers() -> ModuleType:
    path = Path(__file__).with_name("39_positive_cot_target_study.py")
    spec = importlib.util.spec_from_file_location("_positive_cot_study", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import positive-CoT helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PCOT = load_positive_cot_helpers()
WRAPPER_IDS = tuple(PCOT.WRAPPER_IDS)


def mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise ValueError("mean requires finite nonempty values")
    return math.fsum(numbers) / len(numbers)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_alpha_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(piece.strip()) for piece in value.split(",") if piece.strip())
    if len(values) < 3 or values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError("alpha grid must include 0 and 1 and at least one interior point")
    if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in values):
        raise ValueError("alpha values must be finite and in [0,1]")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("alpha grid must be strictly increasing")
    return values


def _empty_accumulators(
    wrapper_ids: Sequence[str], alphas: Sequence[float]
) -> tuple[dict[str, dict[float, dict[str, float]]], dict[float, float]]:
    metrics = {
        wrapper: {
            float(alpha): {"gold_gain_sum": 0.0, "kl_sum": 0.0, "tv_sum": 0.0}
            for alpha in alphas
        }
        for wrapper in wrapper_ids
    }
    pairwise_tv = {float(alpha): 0.0 for alpha in alphas}
    return metrics, pairwise_tv


@torch.inference_mode()
def evaluate_common_alpha_path(
    *,
    student_logits: torch.Tensor,
    privileged_logits: dict[str, torch.Tensor],
    labels: torch.Tensor,
    alphas: Sequence[float],
    chunk_size: int,
) -> dict[str, Any]:
    """Score a common-alpha path with exact full-vocabulary KL and TV."""
    if list(privileged_logits) != list(WRAPPER_IDS):
        raise ValueError("privileged logits must use canonical wrapper order")
    if student_logits.ndim != 3 or student_logits.shape[0] != 1:
        raise ValueError("student logits must have shape [1,T,V]")
    if labels.shape != student_logits.shape[:2]:
        raise ValueError("labels do not align with logits")
    if any(value.shape != student_logits.shape for value in privileged_logits.values()):
        raise ValueError("privileged logits do not align with student logits")
    token_count = int(labels.shape[1])
    accumulators, pairwise_tv = _empty_accumulators(WRAPPER_IDS, alphas)
    for start in range(0, token_count, chunk_size):
        stop = min(token_count, start + chunk_size)
        student = student_logits[:, start:stop].float()
        label_chunk = labels[:, start:stop].long()
        student_logz = torch.logsumexp(student, dim=-1)
        student_gold = (
            student.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1) - student_logz
        )
        student_probability = F.softmax(student, dim=-1)
        for alpha in alphas:
            projected_probabilities: dict[str, torch.Tensor] = {}
            for wrapper in WRAPPER_IDS:
                projected = torch.lerp(
                    student,
                    privileged_logits[wrapper][:, start:stop].float(),
                    float(alpha),
                )
                projected_logz = torch.logsumexp(projected, dim=-1)
                projected_gold = (
                    projected.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
                    - projected_logz
                )
                probability = F.softmax(projected, dim=-1)
                log_density_ratio = (
                    projected
                    - student
                    + student_logz.unsqueeze(-1)
                    - projected_logz.unsqueeze(-1)
                )
                per_token_kl = torch.sum(
                    probability * log_density_ratio, dim=-1
                ).clamp_min(0.0)
                per_token_tv = 0.5 * torch.sum(
                    torch.abs(probability - student_probability), dim=-1
                )
                cell = accumulators[wrapper][float(alpha)]
                cell["gold_gain_sum"] += float(
                    torch.sum(projected_gold - student_gold).item()
                )
                cell["kl_sum"] += float(torch.sum(per_token_kl).item())
                cell["tv_sum"] += float(torch.sum(per_token_tv).item())
                projected_probabilities[wrapper] = probability
                del projected, projected_logz, projected_gold, log_density_ratio
                del per_token_kl, per_token_tv
            pairs = (("neutral", "terse"), ("neutral", "verbose"), ("terse", "verbose"))
            pairwise_tv[float(alpha)] += mean(
                float(
                    torch.sum(
                        0.5
                        * torch.sum(
                            torch.abs(
                                projected_probabilities[left]
                                - projected_probabilities[right]
                            ),
                            dim=-1,
                        )
                    ).item()
                )
                for left, right in pairs
            )
            del projected_probabilities
        del student, label_chunk, student_logz, student_gold, student_probability
    wrappers: dict[str, list[dict[str, float]]] = {}
    for wrapper in WRAPPER_IDS:
        wrappers[wrapper] = [
            {
                "alpha": float(alpha),
                "gold_gain": accumulators[wrapper][float(alpha)]["gold_gain_sum"]
                / token_count,
                "mean_kl": accumulators[wrapper][float(alpha)]["kl_sum"]
                / token_count,
                "mean_tv": accumulators[wrapper][float(alpha)]["tv_sum"]
                / token_count,
            }
            for alpha in alphas
        ]
    return {
        "wrappers": wrappers,
        "pairwise_wrapper_tv": [
            {
                "alpha": float(alpha),
                "mean_pairwise_tv": pairwise_tv[float(alpha)] / token_count,
            }
            for alpha in alphas
        ],
        "token_count": token_count,
    }


def row_for_alpha(rows: Sequence[dict[str, float]], alpha: float) -> dict[str, float]:
    matches = [row for row in rows if math.isclose(row["alpha"], alpha, abs_tol=1e-12)]
    if len(matches) != 1:
        raise ValueError(f"alpha {alpha} is missing or duplicated")
    return matches[0]


@torch.inference_mode()
def solve_epsilon_alpha(
    *,
    student_logits: torch.Tensor,
    privileged_logits: torch.Tensor,
    labels: torch.Tensor,
    grid_rows: Sequence[dict[str, float]],
    epsilon: float,
    chunk_size: int,
    binary_search_steps: int,
) -> tuple[float, dict[str, float]]:
    if grid_rows[-1]["mean_kl"] <= epsilon:
        return 1.0, dict(grid_rows[-1])
    low, high = 0.0, 1.0
    for left, right in zip(grid_rows, grid_rows[1:]):
        if left["mean_kl"] <= epsilon:
            low, high = float(left["alpha"]), float(right["alpha"])
        else:
            break
    low_row = row_for_alpha(grid_rows, low)
    for _ in range(binary_search_steps):
        midpoint = (low + high) / 2.0
        # A single-wrapper pass avoids materializing unrelated privileged paths.
        token_count = int(labels.shape[1])
        gain_sum = kl_sum = tv_sum = 0.0
        for start in range(0, token_count, chunk_size):
            stop = min(token_count, start + chunk_size)
            student = student_logits[:, start:stop].float()
            privileged = privileged_logits[:, start:stop].float()
            label_chunk = labels[:, start:stop].long()
            student_logz = torch.logsumexp(student, dim=-1)
            projected = torch.lerp(student, privileged, midpoint)
            projected_logz = torch.logsumexp(projected, dim=-1)
            student_gold = (
                student.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
                - student_logz
            )
            projected_gold = (
                projected.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
                - projected_logz
            )
            student_probability = F.softmax(student, dim=-1)
            projected_probability = F.softmax(projected, dim=-1)
            log_density_ratio = (
                projected
                - student
                + student_logz.unsqueeze(-1)
                - projected_logz.unsqueeze(-1)
            )
            gain_sum += float(torch.sum(projected_gold - student_gold).item())
            kl_sum += float(
                torch.sum(
                    torch.sum(projected_probability * log_density_ratio, dim=-1)
                    .clamp_min(0.0)
                ).item()
            )
            tv_sum += float(
                torch.sum(
                    0.5
                    * torch.sum(
                        torch.abs(projected_probability - student_probability), dim=-1
                    )
                ).item()
            )
            del student, privileged, label_chunk, projected
            del student_probability, projected_probability, log_density_ratio
        midpoint_row = {
            "alpha": midpoint,
            "gold_gain": gain_sum / token_count,
            "mean_kl": kl_sum / token_count,
            "mean_tv": tv_sum / token_count,
        }
        if midpoint_row["mean_kl"] <= epsilon:
            low, low_row = midpoint, midpoint_row
        else:
            high = midpoint
    return low, low_row


@torch.inference_mode()
def selected_pairwise_tv(
    *,
    student_logits: torch.Tensor,
    privileged_logits: dict[str, torch.Tensor],
    selected_alphas: dict[str, float],
    chunk_size: int,
) -> float:
    token_count = int(student_logits.shape[1])
    total = 0.0
    for start in range(0, token_count, chunk_size):
        stop = min(token_count, start + chunk_size)
        student = student_logits[:, start:stop].float()
        probabilities = {
            wrapper: F.softmax(
                torch.lerp(
                    student,
                    privileged_logits[wrapper][:, start:stop].float(),
                    float(selected_alphas[wrapper]),
                ),
                dim=-1,
            )
            for wrapper in WRAPPER_IDS
        }
        total += mean(
            float(
                torch.sum(
                    0.5 * torch.sum(torch.abs(probabilities[left] - probabilities[right]), dim=-1)
                ).item()
            )
            for left, right in (("neutral", "terse"), ("neutral", "verbose"), ("terse", "verbose"))
        )
        del student, probabilities
    return total / token_count


@torch.inference_mode()
def score_query(
    *,
    model,
    tokenizer,
    query: dict[str, Any],
    label: dict[str, Any],
    study_index: int,
    alphas: Sequence[float],
    epsilon: float,
    binary_search_steps: int,
    reference_token_cap: int,
    answer_token_cap: int,
    max_prompt_tokens: int,
    context_window: int,
    chunk_size: int,
) -> dict[str, Any]:
    query_id = str(query["query_id"])
    problem = PCOT.strip_legacy_math_prompt(str(query["problem"]))
    answer = PCOT.canonical_answer(str(label["answer"]))
    solution, reference_tokens, reference_truncated = PCOT.truncate_reference(
        tokenizer, str(label["reference_solution"]), reference_token_cap
    )
    ordinary_text, privileged_texts = PCOT.build_prompts(
        tokenizer, problem=problem, solution=solution, answer=answer
    )
    device = input_device(model)
    ordinary_ids = PCOT.encode_prompt(tokenizer, ordinary_text, device)
    privileged_ids = {
        wrapper: PCOT.encode_prompt(tokenizer, privileged_texts[wrapper], device)
        for wrapper in WRAPPER_IDS
    }
    prompt_lengths = {
        "ordinary": int(ordinary_ids.shape[1]),
        **{wrapper: int(ids.shape[1]) for wrapper, ids in privileged_ids.items()},
    }
    if max(prompt_lengths.values()) > max_prompt_tokens:
        raise ValueError(f"{query_id}: prompt exceeds {max_prompt_tokens} tokens")
    answer_suffix = f"\\boxed{{{answer}}}"
    answer_ids = tokenizer(
        answer_suffix, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    answer_tokens = int(answer_ids.shape[1])
    if not 0 < answer_tokens <= answer_token_cap:
        raise ValueError(f"{query_id}: invalid answer-token count {answer_tokens}")
    if max(prompt_lengths.values()) + answer_tokens > context_window:
        raise ValueError(f"{query_id}: prompt plus answer exceeds context window")

    student_hidden = PCOT.answer_hidden(model, ordinary_ids, answer_ids)
    student_logits = project_logits(model, student_hidden).detach().float()
    privileged_logits: dict[str, torch.Tensor] = {}
    for wrapper in WRAPPER_IDS:
        hidden = PCOT.answer_hidden(model, privileged_ids[wrapper], answer_ids)
        privileged_logits[wrapper] = project_logits(model, hidden).detach().float()
        del hidden
    path = evaluate_common_alpha_path(
        student_logits=student_logits,
        privileged_logits=privileged_logits,
        labels=answer_ids,
        alphas=alphas,
        chunk_size=chunk_size,
    )
    selected_alphas: dict[str, float] = {}
    selected_rows: dict[str, dict[str, float]] = {}
    for wrapper in WRAPPER_IDS:
        selected_alpha, selected_row = solve_epsilon_alpha(
            student_logits=student_logits,
            privileged_logits=privileged_logits[wrapper],
            labels=answer_ids,
            grid_rows=path["wrappers"][wrapper],
            epsilon=epsilon,
            chunk_size=chunk_size,
            binary_search_steps=binary_search_steps,
        )
        selected_alphas[wrapper] = selected_alpha
        selected_rows[wrapper] = selected_row
    epsilon_pairwise_tv = selected_pairwise_tv(
        student_logits=student_logits,
        privileged_logits=privileged_logits,
        selected_alphas=selected_alphas,
        chunk_size=chunk_size,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "study_index": study_index,
        "query_id": query_id,
        "problem_sha256": hashlib.sha256(problem.encode()).hexdigest(),
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "answer_tokens": answer_tokens,
        "reference_solution_sha256": hashlib.sha256(
            str(label["reference_solution"]).encode()
        ).hexdigest(),
        "reference_tokens_retained": reference_tokens,
        "reference_truncated": reference_truncated,
        "prompt_tokens": prompt_lengths,
        "alpha_grid": [float(alpha) for alpha in alphas],
        "alpha_sweep": path,
        "epsilon_projection": {
            "epsilon": epsilon,
            "wrappers": {
                wrapper: {**selected_rows[wrapper], "alpha": selected_alphas[wrapper]}
                for wrapper in WRAPPER_IDS
            },
            "mean_pairwise_wrapper_tv": epsilon_pairwise_tv,
        },
    }
    del ordinary_ids, privileged_ids, answer_ids
    del student_hidden, student_logits, privileged_logits, path
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize_locality(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    alphas = [float(value) for value in rows[0]["alpha_grid"]]
    if any([float(value) for value in row["alpha_grid"]] != alphas for row in rows):
        raise ValueError("evidence alpha grids are inconsistent")
    curve: list[dict[str, float]] = []
    for alpha in alphas:
        wrapper_rows = [
            row_for_alpha(row["alpha_sweep"]["wrappers"][wrapper], alpha)
            for row in rows
            for wrapper in WRAPPER_IDS
        ]
        per_query_variance = [
            pvariance(
                row_for_alpha(row["alpha_sweep"]["wrappers"][wrapper], alpha)[
                    "gold_gain"
                ]
                for wrapper in WRAPPER_IDS
            )
            for row in rows
        ]
        pairwise = [
            row_for_alpha(row["alpha_sweep"]["pairwise_wrapper_tv"], alpha)[
                "mean_pairwise_tv"
            ]
            for row in rows
        ]
        curve.append(
            {
                "alpha": alpha,
                "mean_gold_gain": mean(item["gold_gain"] for item in wrapper_rows),
                "mean_kl": mean(item["mean_kl"] for item in wrapper_rows),
                "mean_tv": mean(item["mean_tv"] for item in wrapper_rows),
                "positive_wrapper_rate": mean(
                    float(item["gold_gain"] > 0.0) for item in wrapper_rows
                ),
                "all_wrappers_positive_query_rate": mean(
                    float(
                        all(
                            row_for_alpha(
                                row["alpha_sweep"]["wrappers"][wrapper], alpha
                            )["gold_gain"]
                            > 0.0
                            for wrapper in WRAPPER_IDS
                        )
                    )
                    for row in rows
                ),
                "mean_gold_gain_wrapper_variance": mean(per_query_variance),
                "mean_pairwise_wrapper_tv": mean(pairwise),
            }
        )
    full = curve[-1]
    if full["mean_gold_gain"] <= 0.0 or full["mean_tv"] <= 0.0:
        raise ValueError("full privileged endpoint must have positive gain and movement")
    for item in curve:
        item["gold_gain_retained"] = item["mean_gold_gain"] / full["mean_gold_gain"]
        item["tv_retained"] = item["mean_tv"] / full["mean_tv"]
        item["kl_retained"] = (
            item["mean_kl"] / full["mean_kl"] if full["mean_kl"] > 0 else 0.0
        )
        item["pairwise_wrapper_tv_retained"] = (
            item["mean_pairwise_wrapper_tv"] / full["mean_pairwise_wrapper_tv"]
            if full["mean_pairwise_wrapper_tv"] > 0
            else 0.0
        )
    epsilon_rows = [
        row["epsilon_projection"]["wrappers"][wrapper]
        for row in rows
        for wrapper in WRAPPER_IDS
    ]
    epsilon_summary = {
        "epsilon": float(rows[0]["epsilon_projection"]["epsilon"]),
        "mean_alpha": mean(item["alpha"] for item in epsilon_rows),
        "mean_gold_gain": mean(item["gold_gain"] for item in epsilon_rows),
        "mean_kl": mean(item["mean_kl"] for item in epsilon_rows),
        "mean_tv": mean(item["mean_tv"] for item in epsilon_rows),
        "mean_pairwise_wrapper_tv": mean(
            row["epsilon_projection"]["mean_pairwise_wrapper_tv"] for row in rows
        ),
        "positive_wrapper_rate": mean(
            float(item["gold_gain"] > 0.0) for item in epsilon_rows
        ),
        "all_wrappers_positive_query_rate": mean(
            float(
                all(
                    row["epsilon_projection"]["wrappers"][wrapper]["gold_gain"] > 0.0
                    for wrapper in WRAPPER_IDS
                )
            )
            for row in rows
        ),
    }
    epsilon_summary["gold_gain_retained"] = (
        epsilon_summary["mean_gold_gain"] / full["mean_gold_gain"]
    )
    epsilon_summary["tv_retained"] = epsilon_summary["mean_tv"] / full["mean_tv"]
    epsilon_summary["kl_retained"] = epsilon_summary["mean_kl"] / full["mean_kl"]
    epsilon_summary["pairwise_wrapper_tv_retained"] = (
        epsilon_summary["mean_pairwise_wrapper_tv"] / full["mean_pairwise_wrapper_tv"]
        if full["mean_pairwise_wrapper_tv"] > 0
        else 0.0
    )
    epsilon_query_map: list[dict[str, Any]] = []
    for row in rows:
        endpoint_rows = {
            wrapper: row_for_alpha(
                row["alpha_sweep"]["wrappers"][wrapper], 1.0
            )
            for wrapper in WRAPPER_IDS
        }
        projected_rows = row["epsilon_projection"]["wrappers"]
        endpoint_gold = math.fsum(
            endpoint_rows[wrapper]["gold_gain"] for wrapper in WRAPPER_IDS
        )
        endpoint_tv = math.fsum(
            endpoint_rows[wrapper]["mean_tv"] for wrapper in WRAPPER_IDS
        )
        if endpoint_gold <= 0.0 or endpoint_tv <= 0.0:
            raise ValueError("per-query privileged endpoint must have positive gain and TV")
        epsilon_query_map.append(
            {
                "query_id": str(row["query_id"]),
                "mean_alpha": mean(
                    projected_rows[wrapper]["alpha"] for wrapper in WRAPPER_IDS
                ),
                "useful_fidelity_retained": math.fsum(
                    projected_rows[wrapper]["gold_gain"] for wrapper in WRAPPER_IDS
                )
                / endpoint_gold,
                "distribution_movement_retained": math.fsum(
                    projected_rows[wrapper]["mean_tv"] for wrapper in WRAPPER_IDS
                )
                / endpoint_tv,
            }
        )
    query_above_equal = mean(
        float(
            item["useful_fidelity_retained"]
            > item["distribution_movement_retained"]
        )
        for item in epsilon_query_map
    )
    x = np.asarray([item["tv_retained"] for item in curve], dtype=float)
    y = np.asarray([item["gold_gain_retained"] for item in curve], dtype=float)
    if np.any(np.diff(x) < -1e-7):
        raise ValueError("aggregate TV curve is unexpectedly non-monotone")
    auc = float(np.trapezoid(y, x))
    return {
        "estimand": {
            "path": "q_alpha=softmax((1-alpha)z_student+alpha*z_privileged)",
            "task_signal": (
                "mean correct-answer log-probability gain in nats/token, first "
                f"within query then equally over {len(rows)} queries and 3 wrappers"
            ),
            "primary_movement": (
                "mean exact full-vocabulary total variation TV(q_alpha,p_student); "
                "TV is used instead of KL for the retained-fraction claim because "
                "both TV and log-probability gain are first order locally"
            ),
            "secondary_movement": "exact KL(q_alpha||p_student)",
            "prompt_sensitivity": (
                "mean pairwise full-vocabulary TV across neutral/terse/verbose paths"
            ),
        },
        "alpha_curve": curve,
        "epsilon_projection": epsilon_summary,
        "epsilon_query_map": epsilon_query_map,
        "epsilon_query_fraction_above_equal_retention": query_above_equal,
        "epsilon_queries_above_equal_retention": sum(
            item["useful_fidelity_retained"]
            > item["distribution_movement_retained"]
            for item in epsilon_query_map
        ),
        "locality_auc": auc,
        "locality_auc_excess_over_equal_retention": auc - 0.5,
        "hypothesis_holds_at_epsilon": (
            epsilon_summary["gold_gain_retained"] > epsilon_summary["tv_retained"]
        ),
        "pairwise_wrapper_tv_is_nonmonotone": any(
            right["mean_pairwise_wrapper_tv"]
            < left["mean_pairwise_wrapper_tv"] - 1e-12
            for left, right in zip(curve, curve[1:])
        ),
    }


def summarize_loops(path: Path) -> dict[str, Any]:
    rows = PCOT.read_jsonl(path)
    loop_ids = sorted(int(key) for key in rows[0]["wrappers"][0]["loops"])
    output: list[dict[str, Any]] = []
    for loop in loop_ids:
        cell: dict[str, Any] = {"loop": loop}
        for method in ("raw", "projected"):
            deviations = [
                float(wrapper["loops"][str(loop)][method]["deviation"])
                for row in rows
                for wrapper in row["wrappers"]
            ]
            wrapper_means = {
                wrapper_id: mean(
                    next(
                        wrapper
                        for wrapper in row["wrappers"]
                        if wrapper["wrapper_id"] == wrapper_id
                    )["loops"][str(loop)][method]["deviation"]
                    for row in rows
                )
                for wrapper_id in WRAPPER_IDS
            }
            sensitivity = mean(
                pvariance(
                    float(wrapper["loops"][str(loop)][method]["step_kl"])
                    for wrapper in row["wrappers"]
                )
                for row in rows
            )
            cell[method] = {
                "mean_deviation": mean(deviations),
                "wrapper_mean_deviation": wrapper_means,
                "wrapper_sensitivity": sensitivity,
            }
        output.append(cell)
    final = output[-1]
    raw_sensitivity = mean(row["raw"]["wrapper_sensitivity"] for row in output[1:])
    projected_sensitivity = mean(
        row["projected"]["wrapper_sensitivity"] for row in output[1:]
    )
    return {
        "definition": {
            "deviation": "mean KL(q_loop||p_student) over queries and wrappers",
            "wrapper_sensitivity": (
                "mean over queries of population variance across three wrappers "
                "in realized per-loop update KL"
            ),
        },
        "loop_curve": output,
        "final_deviation_ratio_local_over_raw": (
            final["projected"]["mean_deviation"] / final["raw"]["mean_deviation"]
        ),
        "mean_wrapper_sensitivity_raw": raw_sensitivity,
        "mean_wrapper_sensitivity_local": projected_sensitivity,
        "wrapper_sensitivity_retained": projected_sensitivity / raw_sensitivity,
    }


def load_horizon(path: Path) -> dict[str, Any]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["model"] == "Qwen3-8B"
        and row["method"] in {"Privilege-SD", "TRSD"}
        and int(row["episodes"]) in {16, 32, 48, 64}
    ]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for method in ("Privilege-SD", "TRSD"):
        method_rows = sorted(
            (row for row in selected if row["method"] == method),
            key=lambda row: int(row["episodes"]),
        )
        if [int(row["episodes"]) for row in method_rows] != [16, 32, 48, 64]:
            raise ValueError(f"incomplete Qwen3-8B horizon for {method}")
        by_method[method] = [
            {
                "episodes": int(row["episodes"]),
                "strict_correct": int(row["combined_correct"]),
                "total": int(row["combined_total"]),
                "strict_acc1_percent": float(row["combined_strict_acc1_percent"]),
            }
            for row in method_rows
        ]
    early_delta = (
        by_method["TRSD"][0]["strict_acc1_percent"]
        - by_method["Privilege-SD"][0]["strict_acc1_percent"]
    )
    late_delta = (
        by_method["TRSD"][-1]["strict_acc1_percent"]
        - by_method["Privilege-SD"][-1]["strict_acc1_percent"]
    )
    return {
        "metric": "strict Acc@1 on the frozen 143-question AMC23/AIME24/AIME25 set",
        "qwen3_8b": by_method,
        "trsd_minus_raw_pp_at_16": early_delta,
        "trsd_minus_raw_pp_at_64": late_delta,
        "rank_reversal": early_delta < 0.0 < late_delta,
        "caveat": (
            "historical matched-horizon behavior, not a projection-only causal "
            "ablation; source protocols are documented in EXPERIMENTAL_SETTINGS.md"
        ),
    }


def _style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.5)
    axis.tick_params(labelsize=10)


def draw_locality(axis, locality: dict[str, Any]) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    query_map = locality["epsilon_query_map"]
    movement = 100.0 * np.asarray(
        [item["distribution_movement_retained"] for item in query_map]
    )
    fidelity = 100.0 * np.asarray(
        [item["useful_fidelity_retained"] for item in query_map]
    )
    limit = 70.0
    density_cmap = LinearSegmentedColormap.from_list(
        "fidelity_density",
        ("#FFFDF2", "#F5DE72", LOCAL_COLOR),
    )
    density = axis.hexbin(
        movement,
        fidelity,
        gridsize=(10, 10),
        extent=(0.0, limit, 0.0, limit),
        mincnt=1,
        cmap=density_cmap,
        edgecolors=RAW_COLOR,
        linewidths=0.25,
    )
    colorbar = axis.figure.colorbar(density, ax=axis, pad=0.02, fraction=0.05)
    colorbar.set_label("Queries per bin", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    axis.scatter(
        movement,
        fidelity,
        s=11,
        color=RAW_COLOR,
        alpha=0.28,
        linewidths=0.0,
        zorder=2,
    )
    axis.plot([0.0, limit], [0.0, limit], color=RAW_COLOR, linewidth=1.8)
    selected = locality["epsilon_projection"]
    selected_x = 100.0 * selected["tv_retained"]
    selected_y = 100.0 * selected["gold_gain_retained"]
    axis.scatter(
        [selected_x],
        [selected_y],
        s=120,
        marker="*",
        color=LOCAL_COLOR,
        edgecolor=RAW_COLOR,
        linewidth=0.8,
        zorder=5,
    )
    axis.annotate(
        (
            rf"Pooled TRSD $\epsilon={selected['epsilon']:.3f}$"
            "\n"
            rf"{selected_y:.1f}% fidelity / {selected_x:.1f}% movement"
        ),
        (selected_x, selected_y),
        xytext=(10, 5),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
    )
    axis.annotate(
        "equal retention",
        (0.70 * limit, 0.70 * limit),
        xytext=(7, -12),
        textcoords="offset points",
        color=RAW_COLOR,
        fontsize=8.5,
        fontweight="bold",
    )
    above = locality["epsilon_queries_above_equal_retention"]
    axis.text(
        0.98,
        0.04,
        f"{above}/{len(query_map)} queries above equal retention",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )
    axis.set(xlim=(0.0, limit), ylim=(0.0, limit))
    axis.set_xlabel("Full-distribution movement retained (%)", fontsize=11)
    axis.set_ylabel("Correct-answer gain retained (%)", fontsize=11)
    axis.set_title("Useful fidelity gathers locally", loc="left", fontweight="bold")
    axis.set_aspect("equal", adjustable="box")
    _style_axis(axis)
    axis.grid(False)


def draw_loops(axis, loops: dict[str, Any]) -> None:
    rows = loops["loop_curve"][1:]
    x = np.asarray([row["loop"] for row in rows])
    raw = np.asarray([row["raw"]["mean_deviation"] for row in rows])
    local = np.asarray([row["projected"]["mean_deviation"] for row in rows])
    for method, color in (("raw", RAW_COLOR), ("projected", LOCAL_COLOR)):
        for wrapper_id in WRAPPER_IDS:
            values = [
                row[method]["wrapper_mean_deviation"][wrapper_id] for row in rows
            ]
            axis.plot(x, values, color=color, linewidth=0.8, alpha=0.25)
    axis.plot(x, raw, color=RAW_COLOR, linewidth=3.0, marker="o", markersize=4.5)
    axis.plot(x, local, color=LOCAL_COLOR, linewidth=3.0, marker="o", markersize=4.5)
    axis.set_yscale("log")
    axis.set_xlabel("Controlled distribution-space loop", fontsize=11)
    axis.set_ylabel("KL from initial student ↓", fontsize=11)
    axis.set_title("Local targets slow accumulated drift", loc="left", fontweight="bold")
    axis.set_xlim(0.7, 9.0)
    axis.text(8.16, raw[-1], "OPSD", color=RAW_COLOR, va="center", fontweight="bold")
    axis.text(8.16, local[-1], "TRSD", color=LOCAL_COLOR, va="center", fontweight="bold")
    _style_axis(axis)

    inset = axis.inset_axes([0.44, 0.54, 0.48, 0.28])
    raw_sensitivity = np.asarray([row["raw"]["wrapper_sensitivity"] for row in rows])
    local_sensitivity = np.asarray(
        [row["projected"]["wrapper_sensitivity"] for row in rows]
    )
    inset.plot(x, raw_sensitivity, color=RAW_COLOR, linewidth=1.8)
    inset.plot(x, local_sensitivity, color=LOCAL_COLOR, linewidth=1.8)
    inset.set_yscale("log")
    inset.set_title("across-wrapper update variance", fontsize=8, loc="left")
    inset.tick_params(labelsize=6.5, length=2)
    inset.set_xticks([1, 4, 8])
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)


def draw_horizon(axis, horizon: dict[str, Any]) -> None:
    raw_rows = horizon["qwen3_8b"]["Privilege-SD"]
    local_rows = horizon["qwen3_8b"]["TRSD"]
    x = np.asarray([row["episodes"] for row in raw_rows])
    raw = np.asarray([row["strict_acc1_percent"] for row in raw_rows])
    local = np.asarray([row["strict_acc1_percent"] for row in local_rows])
    axis.plot(x, raw, color=RAW_COLOR, linewidth=3.0, marker="o", markersize=5)
    axis.plot(x, local, color=LOCAL_COLOR, linewidth=3.0, marker="o", markersize=5)
    axis.set_xlabel("Training episodes", fontsize=11)
    axis.set_ylabel("Strict Acc@1 (%) ↑", fontsize=11)
    axis.set_title("Long-horizon rank reversal", loc="left", fontweight="bold")
    axis.set_xticks([16, 32, 48, 64])
    axis.set_xlim(12, 71)
    lower = min(float(raw.min()), float(local.min())) - 3.0
    upper = max(float(raw.max()), float(local.max())) + 4.0
    axis.set_ylim(lower, upper)
    axis.text(65.5, raw[-1], "OPSD", color=RAW_COLOR, va="center", fontweight="bold")
    axis.text(65.5, local[-1], "TRSD", color=LOCAL_COLOR, va="center", fontweight="bold")
    _style_axis(axis)


def save_figures(
    run_root: Path,
    locality: dict[str, Any],
    loops: dict[str, Any],
    horizon: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlepad": 8,
            "savefig.dpi": 220,
        }
    )
    renderers = (
        ("figure_a_locality_concentration", draw_locality, locality),
        ("figure_b_repeated_updates", draw_loops, loops),
        ("figure_c_horizon_effect", draw_horizon, horizon),
    )
    for stem, renderer, payload in renderers:
        figure, axis = plt.subplots(figsize=(6.2, 5.1))
        renderer(axis, payload)
        figure.tight_layout()
        figure.savefig(run_root / f"{stem}.png", bbox_inches="tight")
        figure.savefig(run_root / f"{stem}.pdf", bbox_inches="tight")
        plt.close(figure)


def write_tables(run_root: Path, summary: dict[str, Any]) -> None:
    with (run_root / "locality_curve.csv").open("w", newline="") as handle:
        fieldnames = list(summary["study_1_locality"]["alpha_curve"][0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["study_1_locality"]["alpha_curve"])
    with (run_root / "fidelity_drift_query_map.csv").open(
        "w", newline=""
    ) as handle:
        fieldnames = list(summary["study_1_locality"]["epsilon_query_map"][0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["study_1_locality"]["epsilon_query_map"])
    with (run_root / "loop_curve.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "loop",
                "raw_deviation",
                "local_deviation",
                "raw_wrapper_sensitivity",
                "local_wrapper_sensitivity",
            ]
        )
        for row in summary["study_2_loops"]["loop_curve"]:
            writer.writerow(
                [
                    row["loop"],
                    row["raw"]["mean_deviation"],
                    row["projected"]["mean_deviation"],
                    row["raw"]["wrapper_sensitivity"],
                    row["projected"]["wrapper_sensitivity"],
                ]
            )


def write_readme(run_root: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    locality = summary["study_1_locality"]
    selected = summary["study_1_locality"]["epsilon_projection"]
    loops = summary["study_2_loops"]
    horizon = summary["study_3_horizon"]
    text = f"""# Locality-hypothesis empirical chain

Frozen Qwen3-8B Study 1: {manifest['num_queries']} held-out verified-CoT positive controls, three prompt wrappers, exact full-vocabulary scoring, and no parameter update.

![Useful correction is student-local](figure_a_locality_concentration.png)

![Local targets slow accumulated drift](figure_b_repeated_updates.png)

![Long-horizon rank reversal](figure_c_horizon_effect.png)

## Result

At the pre-specified TRSD radius $\\epsilon={selected['epsilon']}$, the projected point retains **{100*selected['gold_gain_retained']:.1f}%** of full privileged useful-answer fidelity while retaining **{100*selected['tv_retained']:.1f}%** of full-distribution TV movement.  The density map aggregates the three wrappers within each query; **{locality['epsilon_queries_above_equal_retention']}/{len(locality['epsilon_query_map'])}** queries lie above equal retention.  Useful-answer fidelity is the retained fraction of the verified-answer log-probability gain, not a claim that every other distributional change is nuisance.

Across eight controlled loops, local targets end at {loops['final_deviation_ratio_local_over_raw']:.4f}x the raw-target deviation.  Wrapper sensitivity is {100*loops['wrapper_sensitivity_retained']:.6f}% of raw when averaged over loops; raw sensitivity spikes at the first loop rather than increasing monotonically.

In the historical matched-horizon Qwen3-8B evaluation, TRSD-minus-OPSD changes from {horizon['trsd_minus_raw_pp_at_16']:+.2f} points at 16 episodes to {horizon['trsd_minus_raw_pp_at_64']:+.2f} points at 64 episodes.  This is downstream long-horizon behavior, not a projection-only causal ablation.

## Figure caption

**Locality hypothesis and its empirical consequences.** Study 1: the fidelity--movement density map contains one point per held-out query after aggregating its three verified-CoT wrappers.  The horizontal coordinate is the fraction of endpoint full-vocabulary TV retained by the adaptive TRSD target; the vertical coordinate is the fraction of endpoint verified-answer log-probability gain retained.  The black diagonal denotes equal retention, the heatmap shows query density, and the star is the ratio-of-global-sums target at $\\epsilon={selected['epsilon']}$. Study 2: raw privileged targets rapidly leave the student neighborhood in controlled distribution-space loops, while KL-bounded local targets accumulate deviation slowly; the inset is across-wrapper variance of per-loop update KL. This does not assume nuisance-free supervision: it measures that context-specific variation cannot enter arbitrarily strongly in one bounded update and therefore accumulates more slowly. Study 3: on the frozen 143-question strict-Acc@1 evaluation, raw/direct OPSD is stronger early, whereas TRSD is stronger at the 64-episode horizon.

## Scope

- Study 1 is an oracle positive-control mechanism diagnostic because the privileged prompts contain verified derivations and final answers.
- Pairwise wrapper TV is retained in `summary.json` as a secondary diagnostic, but it is nonmonotone along the fixed-alpha path and is not used for the headline locality claim; wrapper robustness is tested by the filtered-loop estimand in Study 2.
- Study 2 is distribution-space simulation with no optimizer step.
- Study 3 reuses completed training/evaluation logs and supports a horizon association, not projection-only causality.
"""
    (run_root / "README.md").write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--loop-evidence", type=Path, required=True)
    parser.add_argument("--horizon-table", type=Path, required=True)
    parser.add_argument("--num-queries", type=int, default=128)
    parser.add_argument("--selection-offset", type=int, default=0)
    parser.add_argument("--reference-token-cap", type=int, default=512)
    parser.add_argument("--answer-token-cap", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--epsilon", type=float, default=0.004)
    parser.add_argument("--binary-search-steps", type=int, default=10)
    parser.add_argument("--full-vocab-chunk-size", type=int, default=16)
    parser.add_argument(
        "--alpha-grid",
        default=",".join(str(value) for value in DEFAULT_ALPHA_GRID),
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--compute-slurm-job-id",
        default="",
        help="Job that produced evidence; kept distinct from a plot-only report job.",
    )
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = parse_alpha_grid(args.alpha_grid)
    if args.num_queries <= 0 or args.binary_search_steps <= 0:
        raise ValueError("query count and binary-search steps must be positive")
    if args.selection_offset < 0:
        raise ValueError("selection offset must be nonnegative")
    if args.epsilon <= 0.0 or args.full_vocab_chunk_size <= 0:
        raise ValueError("epsilon and chunk size must be positive")
    args.run_root.mkdir(parents=True, exist_ok=True)
    evidence_path = args.run_root / "evidence.jsonl"
    queries = sorted(PCOT.read_jsonl(args.queries), key=lambda row: str(row["query_id"]))
    labels = {str(row["query_id"]): row for row in PCOT.read_jsonl(args.labels)}
    selection_stop = args.selection_offset + args.num_queries
    selected_queries = queries[args.selection_offset : selection_stop]
    if len(selected_queries) != args.num_queries:
        raise ValueError("insufficient selected queries")
    selected_ids = [str(row["query_id"]) for row in selected_queries]
    if any(query_id not in labels for query_id in selected_ids):
        raise ValueError("selected query is missing a label")
    existing = PCOT.read_jsonl(evidence_path) if evidence_path.exists() else []
    by_id = {str(row["query_id"]): row for row in existing}
    if set(by_id) - set(selected_ids):
        raise ValueError("evidence contains unexpected query IDs")
    runtime: dict[str, Any] | None = None
    model = tokenizer = None
    if not args.plot_only and len(by_id) < len(selected_ids):
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
                    alphas=alphas,
                    epsilon=args.epsilon,
                    binary_search_steps=args.binary_search_steps,
                    reference_token_cap=args.reference_token_cap,
                    answer_token_cap=args.answer_token_cap,
                    max_prompt_tokens=args.max_prompt_tokens,
                    context_window=args.context_window,
                    chunk_size=args.full_vocab_chunk_size,
                )
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                by_id[query_id] = row
                completed = len(by_id)
                elapsed = time.monotonic() - started
                if completed == 1 or completed % 4 == 0:
                    rate = elapsed / max(1, completed - len(existing))
                    eta = rate * (len(selected_ids) - completed)
                    print(
                        f"[{completed}/{len(selected_ids)}] elapsed={elapsed/60:.1f}m "
                        f"eta={eta/60:.1f}m",
                        flush=True,
                    )
    if args.plot_only and len(by_id) < len(selected_ids):
        raise ValueError("plot-only requested before evidence is complete")
    rows = [by_id[query_id] for query_id in selected_ids]
    if len(rows) != len(selected_ids):
        raise ValueError("locality evidence is incomplete")
    if runtime is None and (args.run_root / "manifest.json").exists():
        previous = json.loads((args.run_root / "manifest.json").read_text())
        runtime = previous.get("runtime")
    report_slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if runtime is not None and args.compute_slurm_job_id:
        runtime = dict(runtime or {})
        runtime["slurm_job_id"] = args.compute_slurm_job_id

    locality = summarize_locality(rows)
    loops = summarize_loops(args.loop_evidence)
    horizon = load_horizon(args.horizon_table)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_1_locality": locality,
        "study_2_loops": loops,
        "study_3_horizon": horizon,
        "claims": {
            "useful_correction_is_disproportionately_student_local": locality[
                "hypothesis_holds_at_epsilon"
            ],
            "kl_bounded_targets_control_accumulation_rate": (
                loops["final_deviation_ratio_local_over_raw"] < 1.0
            ),
            "qwen3_8b_long_horizon_rank_reversal": horizon["rank_reversal"],
        },
    }
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
        "selection": (
            "lexicographic query_id slice; label-independent; "
            f"offset={args.selection_offset}; count={args.num_queries}"
        ),
        "selection_offset": args.selection_offset,
        "selected_query_ids_sha256": hashlib.sha256(
            ("\n".join(selected_ids) + "\n").encode()
        ).hexdigest(),
        "num_queries": len(rows),
        "wrappers": list(WRAPPER_IDS),
        "alpha_grid": list(alphas),
        "epsilon": args.epsilon,
        "binary_search_steps": args.binary_search_steps,
        "full_vocabulary_exact": True,
        "primary_distance": "total_variation",
        "secondary_distance": "KL(q_alpha||p_student)",
        "loop_evidence_path": str(args.loop_evidence),
        "loop_evidence_sha256": sha256_file(args.loop_evidence),
        "horizon_table_path": str(args.horizon_table),
        "horizon_table_sha256": sha256_file(args.horizon_table),
        "runtime": runtime,
        "compute_slurm_job_id": args.compute_slurm_job_id,
        "report_slurm_job_id": report_slurm_job_id,
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(args.run_root / "summary.json", summary)
    write_json(args.run_root / "manifest.json", manifest)
    write_tables(args.run_root, summary)
    save_figures(args.run_root, locality, loops, horizon)
    write_readme(args.run_root, summary, manifest)
    figures = [
        "figure_a_locality_concentration",
        "figure_b_repeated_updates",
        "figure_c_horizon_effect",
    ]
    complete = {
        "schema_version": SCHEMA_VERSION,
        "queries": len(rows),
        "claims": summary["claims"],
        "evidence_sha256": sha256_file(evidence_path),
        "summary_sha256": sha256_file(args.run_root / "summary.json"),
        "figures": {
            stem: {
                "pdf_sha256": sha256_file(args.run_root / f"{stem}.pdf"),
                "png_sha256": sha256_file(args.run_root / f"{stem}.png"),
            }
            for stem in figures
        },
    }
    write_json(args.run_root / "COMPLETE.json", complete)
    print(json.dumps(summary["claims"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
