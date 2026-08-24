#!/usr/bin/env python3
"""Collect one label-free, post-hoc LGSD locality mechanism trajectory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from src.clean_self_distill.generation import generate_response, problem_prompt
from src.clean_self_distill.heldout import (
    _atomic_write_jsonl,
    generation_config_sha256,
    load_query_only_manifest,
    paired_sample_seed,
    query_manifest_sha256,
    tree_sha256,
)
from src.clean_self_distill.persistent import CHECKPOINT_SCHEMA_VERSION
from src.clean_self_distill.runtime import (
    backbone_forward,
    collect_runtime_metadata,
    input_device,
    load_hf_model,
)
from src.clean_self_distill.trust_region_mechanism import (
    MECHANISM_SCHEMA_VERSION,
    WRAPPER_IDS,
    WRAPPER_SET_VERSION,
    TrustRegionMechanismError,
    build_privileged_prompt_wrapper,
    evaluate_projection_alphas,
    parse_float_grid,
    solve_epsilon_alphas,
    summarize_wrapper_robustness,
    token_categories,
)


DEFAULT_ALPHA_GRID = ",".join(f"{index / 20:.2f}" for index in range(21))
DEFAULT_EPSILON_GRID = "0.001,0.002,0.004,0.008,0.016,0.032,0.08"


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrustRegionMechanismError(f"Cannot read JSON object {path}") from exc
    if not isinstance(value, dict):
        raise TrustRegionMechanismError(f"{path} must contain one JSON object")
    return value


def _validate_final_lgsd_checkpoint(
    adapter: Path, *, model_id: str, revision: str
) -> dict[str, Any]:
    """Fail closed on the completed protocol or the adopted rolling-36 endpoint.

    The historical H100 run was deliberately stopped at its last loadable
    rolling checkpoint (episode 36), before the replacement persistent runner
    acquired its current ``COMPLETE.json`` convention.  That endpoint is
    accepted only by exact episode/type/latest-pointer checks and is reported
    as rolling, never mislabeled as a completed scientific checkpoint.
    """
    manifest_path = adapter / "checkpoint_manifest.json"
    if not adapter.is_dir() or not manifest_path.is_file():
        raise TrustRegionMechanismError(
            f"Adapter lacks checkpoint_manifest.json: {adapter}"
        )
    manifest = _read_object(manifest_path)
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "branch": "clean",
        "model_id": model_id,
        "model_revision": revision,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise TrustRegionMechanismError(
            f"Checkpoint is not the requested final LGSD adapter: {mismatches}"
        )
    try:
        checkpoint_episode = int(manifest["checkpoint_episode"])
        completed_episodes = int(manifest["completed_episodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrustRegionMechanismError("Checkpoint has invalid episode metadata") from exc
    if checkpoint_episode <= 0 or checkpoint_episode != completed_episodes:
        raise TrustRegionMechanismError("Checkpoint episode metadata is inconsistent")
    method_id = str(manifest.get("method_id", ""))
    checkpoint_type = str(manifest.get("checkpoint_type", ""))
    current_method = method_id in {
        "lgsd:geometric_kl_ball_projection:forward_kl_v1",
        "trsd:exponential_teacher_projection",
    }
    historical_method = method_id.endswith(":trust_region")
    if not current_method and not historical_method:
        raise TrustRegionMechanismError("Checkpoint method is not an LGSD projection")
    if method_id.startswith("lgsd:") and manifest.get(
        "distillation_kl_direction"
    ) != "projected_teacher_to_student_forward_kl_v1":
        raise TrustRegionMechanismError(
            "LGSD checkpoint does not use projected-target forward KL"
        )

    run_root = adapter.parent.parent
    run_manifest = _read_object(run_root / "run_manifest.json")
    arguments = run_manifest.get("arguments")
    if not isinstance(arguments, dict):
        raise TrustRegionMechanismError("LGSD run manifest lacks arguments")
    if (
        run_manifest.get("run_identity_sha256")
        != manifest.get("run_identity_sha256")
        or arguments.get("branch") != "clean"
    ):
        raise TrustRegionMechanismError(
            "Adapter disagrees with its clean-branch LGSD run manifest"
        )
    complete_path = run_root / "COMPLETE.json"
    completed_final = False
    complete_sha256: str | None = None
    if complete_path.is_file():
        complete = _read_object(complete_path)
        completed_final = bool(
            checkpoint_type == "scientific"
            and current_method
            and manifest.get("variant") == "trust_region"
            and complete.get("run_identity_sha256")
            == manifest.get("run_identity_sha256")
            and complete.get("status") == "complete"
            and int(complete.get("completed_episodes", -1)) == checkpoint_episode
            and int(arguments.get("episodes", -1)) == checkpoint_episode
        )
        complete_sha256 = hashlib.sha256(complete_path.read_bytes()).hexdigest()

    latest_path = adapter.parent / "LATEST.json"
    historical_rolling_36 = False
    if latest_path.is_file():
        latest = _read_object(latest_path)
        historical_rolling_36 = bool(
            checkpoint_type == "rolling"
            and historical_method
            and checkpoint_episode == 36
            and adapter.name == "rolling_episode_0036"
            and latest.get("checkpoint_dir") == adapter.name
            and int(latest.get("completed_episodes", -1)) == checkpoint_episode
            and latest.get("run_identity_sha256")
            == manifest.get("run_identity_sha256")
        )
    if not completed_final and not historical_rolling_36:
        raise TrustRegionMechanismError(
            "Checkpoint is neither a completed final LGSD checkpoint nor the "
            "exact adopted latest rolling_episode_0036 endpoint"
        )
    return {
        "checkpoint_episode": checkpoint_episode,
        "checkpoint_type": checkpoint_type,
        "method_id": method_id,
        "variant": str(manifest.get("variant", "")),
        "selection_status": (
            "completed_scientific_final"
            if completed_final
            else "historical_latest_rolling_endpoint"
        ),
        "run_identity_sha256": str(manifest["run_identity_sha256"]),
        "checkpoint_manifest": manifest,
        "checkpoint_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "run_manifest_sha256": hashlib.sha256(
            (run_root / "run_manifest.json").read_bytes()
        ).hexdigest(),
        "complete_sha256": complete_sha256,
    }


def _validate_pinned_base_checkpoint(*, model_id: str, revision: str) -> dict[str, Any]:
    """Return auditable provenance for a base-model mechanism calibration."""
    identity_payload = json.dumps(
        {
            "checkpoint_type": "base",
            "model_id": model_id,
            "model_revision": revision,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    identity = hashlib.sha256(identity_payload).hexdigest()
    return {
        "checkpoint_episode": 0,
        "checkpoint_type": "base",
        "method_id": "base:pinned_model",
        "variant": "base",
        "selection_status": "pinned_base_model",
        "run_identity_sha256": identity,
        "checkpoint_manifest_sha256": identity,
        "run_manifest_sha256": identity,
        "complete_sha256": None,
        "checkpoint_content_sha256": identity,
    }


def _tokenize(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")[
        "input_ids"
    ]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise TrustRegionMechanismError("Tokenizer did not return one prompt")
    return ids.to(device)


def _find_epsilon_row(rows: list[dict[str, Any]], epsilon: float) -> dict[str, Any]:
    found = [
        row
        for row in rows
        if math.isclose(float(row["epsilon"]), epsilon, abs_tol=1e-12)
    ]
    if len(found) != 1:
        raise TrustRegionMechanismError(f"Missing unique epsilon row for {epsilon}")
    return found[0]


def _existing_row(path: Path, generation_digest: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise TrustRegionMechanismError(
            "Mechanism output must contain exactly one resumable query row"
        )
    value = json.loads(lines[0])
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != MECHANISM_SCHEMA_VERSION
        or value.get("generation_config_sha256") != generation_digest
        or value.get("labels_loaded") is not False
    ):
        raise TrustRegionMechanismError(
            "Existing mechanism output belongs to another protocol"
        )
    return value


def collect(args: argparse.Namespace) -> dict[str, Any]:
    alpha_grid = parse_float_grid(
        args.alpha_grid,
        name="alpha grid",
        minimum=0.0,
        maximum=1.0,
        require_endpoints=True,
    )
    epsilon_grid = parse_float_grid(
        args.epsilon_grid,
        name="epsilon grid",
        minimum=0.0,
        maximum=float("inf"),
    )
    if epsilon_grid[0] <= 0:
        raise TrustRegionMechanismError("epsilon grid values must be positive")
    for requested, name in (
        (args.selected_epsilon, "selected epsilon"),
        (args.training_epsilon, "training epsilon"),
    ):
        if not any(math.isclose(requested, item, abs_tol=1e-12) for item in epsilon_grid):
            raise TrustRegionMechanismError(f"{name} must occur in --epsilon-grid")

    queries = load_query_only_manifest(args.queries)
    if not 0 <= args.query_index < len(queries):
        raise TrustRegionMechanismError(
            f"query-index {args.query_index} outside manifest of size {len(queries)}"
        )
    query = queries[args.query_index]
    if args.base_only:
        checkpoint = _validate_pinned_base_checkpoint(
            model_id=args.model_id, revision=args.revision
        )
        checkpoint_sha256 = checkpoint["checkpoint_content_sha256"]
    else:
        if not args.adapter:
            raise TrustRegionMechanismError("--adapter is required without --base-only")
        checkpoint = _validate_final_lgsd_checkpoint(
            Path(args.adapter), model_id=args.model_id, revision=args.revision
        )
        checkpoint_sha256 = tree_sha256(args.adapter)
    manifest_sha256 = query_manifest_sha256(queries)
    seed = paired_sample_seed(args.seed, args.query_index, 0, sample_count=1)
    generation_digest = generation_config_sha256(
        {
            "schema_version": MECHANISM_SCHEMA_VERSION,
            "query_manifest_sha256": manifest_sha256,
            "query_index": args.query_index,
            "query_id": query["query_id"],
            "checkpoint_sha256": checkpoint_sha256,
            "model_id": args.model_id,
            "revision": args.revision,
            "wrapper_set_version": WRAPPER_SET_VERSION,
            "wrappers": list(WRAPPER_IDS),
            "alpha_grid": list(alpha_grid),
            "epsilon_grid": list(epsilon_grid),
            "selected_epsilon": args.selected_epsilon,
            "training_epsilon": args.training_epsilon,
            "binary_search_steps": args.binary_search_steps,
            "full_vocab_chunk_size": args.full_vocab_chunk_size,
            "sampling": {
                "seed": seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
                "max_prompt_tokens": args.max_prompt_tokens,
                "context_window": args.context_window,
            },
        }
    )
    existing = _existing_row(Path(args.output), generation_digest)
    if existing is not None:
        return existing

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    local_model = Path(args.model)
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        training=False,
        revision=None if local_model.exists() else args.revision,
        attn_implementation=args.attn_implementation,
    )
    if not args.base_only:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, args.adapter, is_trainable=False
        )
    model.eval()
    runtime = collect_runtime_metadata(
        model, model_path=args.model_id, revision=args.revision
    )
    device = input_device(model)
    normal_prompt = problem_prompt(tokenizer, query["problem"])
    normal_prompt_ids = _tokenize(tokenizer, normal_prompt, device)
    wrapper_prompts = {
        wrapper_id: build_privileged_prompt_wrapper(
            tokenizer, query["problem"], wrapper_id
        )
        for wrapper_id in WRAPPER_IDS
    }
    wrapper_prompt_ids = {
        wrapper_id: _tokenize(tokenizer, prompt, device)
        for wrapper_id, prompt in wrapper_prompts.items()
    }
    longest_prompt = max(
        [int(normal_prompt_ids.shape[1])]
        + [int(ids.shape[1]) for ids in wrapper_prompt_ids.values()]
    )
    if longest_prompt > args.max_prompt_tokens:
        raise TrustRegionMechanismError(
            f"Prompt has {longest_prompt}>{args.max_prompt_tokens} tokens"
        )
    if longest_prompt + args.max_new_tokens > args.context_window:
        raise TrustRegionMechanismError(
            "Prompts cannot receive the full preregistered rollout opportunity"
        )

    response, generated_prompt_ids, response_ids = generate_response(
        model,
        tokenizer,
        query["problem"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=seed,
        prompt_override=normal_prompt,
    )
    token_count = int(response_ids.numel())
    if token_count <= 0:
        raise TrustRegionMechanismError("On-policy rollout is empty")
    if not torch.equal(normal_prompt_ids, generated_prompt_ids):
        raise TrustRegionMechanismError("Generated prompt disagrees with scored prompt")
    ended_by_eos = bool(
        tokenizer.eos_token_id is not None
        and int(response_ids[0, -1].item()) == int(tokenizer.eos_token_id)
    )
    student_full_ids = torch.cat([normal_prompt_ids, response_ids], dim=1)
    with torch.inference_mode():
        student_hidden_all, _ = backbone_forward(
            model,
            input_ids=student_full_ids,
            attention_mask=torch.ones_like(student_full_ids),
            use_cache=False,
        )
    student_start = int(normal_prompt_ids.shape[1]) - 1
    student_hidden = student_hidden_all[
        :, student_start : student_start + token_count
    ].detach()
    if int(student_hidden.shape[1]) != token_count:
        raise TrustRegionMechanismError("Student hidden states do not cover rollout")
    categories, token_texts = token_categories(tokenizer, response_ids)
    label_ids = response_ids.to(student_hidden.device)

    wrappers: list[dict[str, Any]] = []
    for wrapper_id in WRAPPER_IDS:
        teacher_prompt_ids = wrapper_prompt_ids[wrapper_id]
        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids], dim=1)
        if int(teacher_full_ids.shape[1]) > args.context_window:
            raise TrustRegionMechanismError(
                f"{wrapper_id} teacher sequence exceeds context window"
            )
        with torch.inference_mode():
            teacher_hidden_all, _ = backbone_forward(
                model,
                input_ids=teacher_full_ids,
                attention_mask=torch.ones_like(teacher_full_ids),
                use_cache=False,
            )
        teacher_start = int(teacher_prompt_ids.shape[1]) - 1
        teacher_hidden = teacher_hidden_all[
            :, teacher_start : teacher_start + token_count
        ].detach()
        if teacher_hidden.shape != student_hidden.shape:
            raise TrustRegionMechanismError(
                f"{wrapper_id} privileged states do not align with response prefix"
            )

        alpha_evaluation = evaluate_projection_alphas(
            model=model,
            student_hidden=student_hidden,
            privileged_hidden=teacher_hidden,
            labels=label_ids,
            categories=categories,
            alphas=alpha_grid,
            chunk_size=args.full_vocab_chunk_size,
            capture_trace=False,
        )
        epsilon_alphas = solve_epsilon_alphas(
            model=model,
            student_hidden=student_hidden,
            privileged_hidden=teacher_hidden,
            labels=label_ids,
            categories=categories,
            alpha_evaluation=alpha_evaluation,
            epsilon_grid=epsilon_grid,
            chunk_size=args.full_vocab_chunk_size,
            binary_search_steps=args.binary_search_steps,
        )
        trace_alphas = sorted(set([1.0, *epsilon_alphas.values()]))
        final_evaluation = evaluate_projection_alphas(
            model=model,
            student_hidden=student_hidden,
            privileged_hidden=teacher_hidden,
            labels=label_ids,
            categories=categories,
            alphas=trace_alphas,
            chunk_size=args.full_vocab_chunk_size,
            capture_trace=True,
        )
        raw = dict(final_evaluation.summaries[1.0])
        raw["projection"] = "unconstrained_privileged_surrogate"
        epsilon_sweep: list[dict[str, Any]] = []
        for epsilon in epsilon_grid:
            alpha = epsilon_alphas[epsilon]
            summary = dict(final_evaluation.summaries[alpha])
            achieved = float(summary.pop("mean_kl"))
            summary.update(
                {
                    "epsilon": epsilon,
                    "achieved_mean_kl": achieved,
                    "epsilon_slack": epsilon - achieved,
                    "constraint_active": alpha < 1.0 - 1e-12,
                    "is_training_budget": math.isclose(
                        epsilon, args.training_epsilon, abs_tol=1e-12
                    ),
                    "is_posthoc_stress_test": math.isclose(
                        epsilon, args.selected_epsilon, abs_tol=1e-12
                    ),
                }
            )
            epsilon_sweep.append(summary)

        raw_trace = final_evaluation.traces[1.0]
        token_trace: list[dict[str, Any]] = []
        for position, token_id in enumerate(response_ids.detach().cpu()[0].tolist()):
            projections = []
            for epsilon in epsilon_grid:
                alpha = epsilon_alphas[epsilon]
                values = final_evaluation.traces[alpha]
                projections.append(
                    {
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "projected_logprob": values["projected_logprob"][position],
                        "projected_surrogate_logratio": values["logratio"][position],
                        "projected_kl": values["kl"][position],
                    }
                )
            selected = _find_epsilon_row(projections, args.selected_epsilon)
            token_trace.append(
                {
                    "position": position,
                    "normalized_position": (
                        position / (token_count - 1) if token_count > 1 else 0.0
                    ),
                    "token_id": int(token_id),
                    "token_text": token_texts[position],
                    "token_category": categories[position],
                    "student_logprob": raw_trace["student_logprob"][position],
                    "raw_teacher_logprob": raw_trace["projected_logprob"][position],
                    "raw_surrogate_logratio": raw_trace["logratio"][position],
                    "raw_kl": raw_trace["kl"][position],
                    "projected_alpha": selected["alpha"],
                    "projected_logprob": selected["projected_logprob"],
                    "projected_surrogate_logratio": selected[
                        "projected_surrogate_logratio"
                    ],
                    "projected_kl": selected["projected_kl"],
                    "epsilon_projections": projections,
                }
            )
        wrappers.append(
            {
                "wrapper_id": wrapper_id,
                "wrapper_set_version": WRAPPER_SET_VERSION,
                "prompt_sha256": hashlib.sha256(
                    wrapper_prompts[wrapper_id].encode("utf-8")
                ).hexdigest(),
                "prompt_tokens": int(teacher_prompt_ids.shape[1]),
                "raw": raw,
                "alpha_sweep": [
                    dict(alpha_evaluation.summaries[alpha]) for alpha in alpha_grid
                ],
                "epsilon_sweep": epsilon_sweep,
                "token_trace": token_trace,
            }
        )
        del teacher_hidden_all, teacher_hidden, teacher_full_ids
        del alpha_evaluation, final_evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    robustness = summarize_wrapper_robustness(
        wrappers, selected_epsilon=args.selected_epsilon
    )
    training_rows = [
        _find_epsilon_row(wrapper["epsilon_sweep"], args.training_epsilon)
        for wrapper in wrappers
    ]
    row = {
        "schema_version": MECHANISM_SCHEMA_VERSION,
        "record_type": "query_mechanism",
        "labels_loaded": False,
        "label_paths_accepted_by_cli": False,
        "query_id": query["query_id"],
        "query_index": args.query_index,
        "problem_sha256": query["problem_sha256"],
        "source": query["source"],
        "query_manifest_sha256": manifest_sha256,
        "generation_config_sha256": generation_digest,
        "checkpoint_episode": checkpoint["checkpoint_episode"],
        "checkpoint_sha256": checkpoint_sha256,
        "run_identity_sha256": checkpoint["run_identity_sha256"],
        "checkpoint_validation": {
            "selection_status": checkpoint["selection_status"],
            "checkpoint_type": checkpoint["checkpoint_type"],
            "method_id": checkpoint["method_id"],
            "variant": checkpoint["variant"],
            "checkpoint_manifest_sha256": checkpoint[
                "checkpoint_manifest_sha256"
            ],
            "run_manifest_sha256": checkpoint["run_manifest_sha256"],
            "complete_sha256": checkpoint["complete_sha256"],
        },
        "seed": seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": token_count,
        "truncated": token_count >= args.max_new_tokens and not ended_by_eos,
        "student_prompt_tokens": int(normal_prompt_ids.shape[1]),
        "response": response,
        "response_token_ids": response_ids.detach().cpu()[0].tolist(),
        "full_vocabulary_kl": True,
        "full_vocab_chunk_size": args.full_vocab_chunk_size,
        "alpha_grid": list(alpha_grid),
        "epsilon_grid": list(epsilon_grid),
        "training_epsilon": args.training_epsilon,
        "training_epsilon_role": (
            "historical_reference_not_base_checkpoint_budget"
            if args.base_only
            else "checkpoint_training_budget"
        ),
        "training_constraint_active_by_wrapper": {
            wrapper["wrapper_id"]: bool(result["constraint_active"])
            for wrapper, result in zip(wrappers, training_rows)
        },
        "selected_epsilon": args.selected_epsilon,
        "selected_epsilon_role": "posthoc_stress_test_not_training_budget",
        "binary_search_steps": args.binary_search_steps,
        "wrappers": wrappers,
        "wrapper_robustness": robustness,
        "runtime": runtime,
    }
    _atomic_write_jsonl(Path(args.output), [row])
    del model, student_hidden_all, student_hidden
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--query-index", type=int, default=22)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--context-window", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-grid", default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--epsilon-grid", default=DEFAULT_EPSILON_GRID)
    parser.add_argument("--selected-epsilon", type=float, default=0.008)
    parser.add_argument("--training-epsilon", type=float, default=0.004)
    parser.add_argument("--binary-search-steps", type=int, default=8)
    parser.add_argument("--full-vocab-chunk-size", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser


def main() -> None:
    collect(build_parser().parse_args())


if __name__ == "__main__":
    main()
