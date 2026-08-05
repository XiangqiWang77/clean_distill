#!/usr/bin/env python3
"""Validate the exact 16k Clean/Privileged training paths on a real H100.

This is a full-sequence memory and backward validation, not a reduced model or
short-context smoke test.  It consumes only a query-only Dev record and never
loads a target answer, reference solution, reward, or feedback.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

from src.clean_self_distill.heldout import load_query_only_manifest
from src.clean_self_distill.ridge import SparseRidgeAdapter, problem_prompt
from src.clean_self_distill.runtime import (
    backbone_forward,
    collect_runtime_metadata,
    input_device,
    load_hf_model,
    project_logits,
    unwrap_causal_lm,
)
from src.clean_self_distill.streaming_distill import stream_distillation_chunks


SCHEMA_VERSION = "clean-self-distill-full-sequence-h100-validation-v2"
FORBIDDEN_QUERY_KEYS = frozenset(
    {
        "answer",
        "feedback",
        "ground_truth",
        "label",
        "reference",
        "reference_answer",
        "reference_solution",
        "reward",
        "solution",
        "target_answer",
        "target_solution",
    }
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_binding(args: argparse.Namespace) -> dict[str, Any]:
    value = {
        "git_commit": args.git_commit,
        "code_tree_sha256": args.code_tree_sha256,
        "run_config_sha256": _sha256_file(args.run_config),
        "query_manifest_sha256": _sha256_file(args.queries),
        "model_id": args.model_id,
        "model_revision": args.revision,
        "model_path": str(Path(args.model).resolve()),
        "max_sequence_tokens": args.max_sequence_tokens,
        "distilled_positions": args.max_sequence_tokens - 1,
        "distill_token_chunk_size": args.distill_token_chunk_size,
        "distill_top_k": args.distill_top_k,
        "distill_temperature": args.distill_temperature,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "max_grad_norm": args.max_grad_norm,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "expected_gpu_name_regex": args.expected_gpu_name_regex,
        "expected_gpu_capability": [
            args.expected_gpu_capability_major,
            args.expected_gpu_capability_minor,
        ],
        "expected_gpu_arch_flag": args.expected_gpu_arch_flag,
        "minimum_reserved_headroom_bytes": args.minimum_reserved_headroom_bytes,
    }
    value["binding_sha256"] = _canonical_sha256(value)
    return value


def _model_device_placement(model) -> dict[str, Any]:
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if parameter_devices != ["cuda:0"]:
        raise RuntimeError(
            f"formal validation forbids parameter offload: {parameter_devices}"
        )
    maps = []
    for candidate in (model, unwrap_causal_lm(model)):
        mapping = getattr(candidate, "hf_device_map", None)
        if isinstance(mapping, dict):
            maps.append({str(key): str(value) for key, value in mapping.items()})
    offloaded = sorted(
        {
            value
            for mapping in maps
            for value in mapping.values()
            if value not in {"0", "cuda", "cuda:0"}
        }
    )
    if offloaded:
        raise RuntimeError(f"formal validation forbids HF CPU/disk offload: {offloaded}")
    return {
        "parameter_devices": parameter_devices,
        "hf_device_maps": maps,
        "cpu_or_disk_offload": False,
    }


def _validate_artifact(payload: dict[str, Any], args: argparse.Namespace) -> None:
    """Fail closed on both newly written and marker-reused gate artifacts."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("full-sequence artifact has the wrong schema")
    if payload.get("status") != "complete" or payload.get("target_fields_loaded") is not False:
        raise ValueError("full-sequence artifact status/firewall is invalid")
    expected_binding = _validation_binding(args)
    if payload.get("binding") != expected_binding:
        raise ValueError("full-sequence artifact binding disagrees with immutable inputs")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("full-sequence artifact runtime is missing")
    if runtime.get("git_commit") != args.git_commit:
        raise ValueError("full-sequence runtime commit mismatch")
    if runtime.get("code_tree_sha256") != args.code_tree_sha256:
        raise ValueError("full-sequence runtime code-tree mismatch")
    if runtime.get("requested_model_revision") != args.revision or runtime.get(
        "resolved_model_revision"
    ) != args.revision:
        raise ValueError("full-sequence runtime model revision mismatch")
    if runtime.get("model_use_cache") is not False:
        raise ValueError("full-sequence runtime cache must be disabled")
    if runtime.get("gradient_checkpointing_enabled") is not True or runtime.get(
        "gradient_checkpointing_use_reentrant"
    ) is not False:
        raise ValueError("full-sequence runtime checkpointing mode is invalid")
    gpus = runtime.get("gpus")
    if runtime.get("gpu_count") != 1 or not isinstance(gpus, list) or len(gpus) != 1:
        raise ValueError("full-sequence artifact must bind exactly one GPU")
    gpu = gpus[0]
    if not re.search(args.expected_gpu_name_regex, str(gpu.get("name", "")), re.I):
        raise ValueError("full-sequence artifact was not produced on the expected H100")
    if gpu.get("capability") != [
        args.expected_gpu_capability_major,
        args.expected_gpu_capability_minor,
    ]:
        raise ValueError("full-sequence artifact GPU capability mismatch")
    if args.expected_gpu_arch_flag not in runtime.get("torch_arch_flags", []):
        raise ValueError("full-sequence artifact lacks the expected CUDA arch")
    total_memory = int(gpu.get("total_memory_bytes", 0))
    if total_memory <= args.minimum_reserved_headroom_bytes:
        raise ValueError("full-sequence artifact GPU memory metadata is invalid")
    placement = payload.get("device_placement")
    if not isinstance(placement, dict) or placement.get("parameter_devices") != [
        "cuda:0"
    ] or placement.get("cpu_or_disk_offload") is not False:
        raise ValueError("full-sequence artifact permits model offload")
    branches = payload.get("branches")
    if not isinstance(branches, list) or {
        item.get("branch") for item in branches if isinstance(item, dict)
    } != {"clean", "privileged"} or len(branches) != 2:
        raise ValueError("full-sequence artifact must contain exactly two branches")
    for item in branches:
        if item.get("sequence_tokens") != args.max_sequence_tokens:
            raise ValueError("full-sequence branch used the wrong sequence length")
        if item.get("distilled_positions") != args.max_sequence_tokens - 1:
            raise ValueError("full-sequence branch used the wrong distilled length")
        if item.get("chunk_size") != args.distill_token_chunk_size or int(
            item.get("max_projected_chunk_tokens", 0)
        ) > args.distill_token_chunk_size:
            raise ValueError("full-sequence branch violated its vocabulary chunk cap")
        if item.get("optimizer_changed_trainable_parameter") is not True:
            raise ValueError("full-sequence branch did not execute a real optimizer step")
        for key in ("loss", "mean_teacher_student_kl", "gradient_norm_before_clip"):
            value = float(item.get(key, float("nan")))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"full-sequence branch has invalid {key}")
        allocated = int(item.get("peak_memory_allocated_bytes", 0))
        reserved = int(item.get("peak_memory_reserved_bytes", 0))
        if allocated <= 0 or reserved < allocated:
            raise ValueError("full-sequence branch memory counters are invalid")
        if reserved > total_memory - args.minimum_reserved_headroom_bytes:
            raise ValueError("full-sequence branch lacks the frozen H100 memory headroom")


def _full_length_ids(tokenizer, problem: str, length: int, device) -> torch.Tensor:
    if length < 2:
        raise ValueError("full-sequence validation requires at least two tokens")
    prompt_ids = tokenizer(
        problem_prompt(tokenizer, problem),
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"].reshape(-1)
    continuation = tokenizer(
        (
            "We decompose the problem, track every invariant, check boundary "
            "cases, verify each algebraic transformation, and independently "
            "recheck the resulting argument. "
        ),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].reshape(-1)
    source = torch.cat([prompt_ids, continuation])
    if source.numel() == 0:
        raise ValueError("tokenizer produced an empty validation sequence")
    repeats = (length + int(source.numel()) - 1) // int(source.numel())
    return source.repeat(repeats)[:length].reshape(1, -1).to(device)


def _clean_teacher(hidden: torch.Tensor, token_id: int) -> SparseRidgeAdapter:
    support = hidden[0, 0].detach().float()
    support = support / support.norm().clamp_min(1e-6)
    return SparseRidgeAdapter(
        support_hidden=support.reshape(1, -1),
        coefficients=torch.tensor([[2.0]], device=hidden.device),
        vocab_ids=torch.tensor([token_id], dtype=torch.long, device=hidden.device),
        hidden_scale=1.0,
        ridge_lambda_effective=0.1,
        metadata={"validation_only": True},
    )


def _run_branch(
    *,
    model,
    input_ids: torch.Tensor,
    branch: str,
    chunk_size: int,
    top_k: int,
    temperature: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    device = input_ids.device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("full-sequence validation requires trainable LoRA parameters")
    output_head = unwrap_causal_lm(model).get_output_embeddings()
    if output_head is None or any(
        parameter.requires_grad for parameter in output_head.parameters()
    ):
        raise RuntimeError("full-sequence validation requires a frozen LM head")
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in parameters]

    # Training mode is required for Transformers gradient checkpointing.  The
    # pinned Qwen3 config and LoRA config both use zero dropout.
    model.train()
    student_all, _ = backbone_forward(
        model,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
    )
    student_hidden = student_all[:, :-1]
    labels = input_ids[:, 1:]

    teacher_hidden_all = None
    if branch == "clean":
        adapter = _clean_teacher(student_hidden, int(labels[0, 0].item()))

        def teacher_for_chunk(student_logits, hidden_chunk, _start, _stop):
            return adapter.apply_to_logits(
                student_logits.detach(), hidden_chunk.detach()
            )

    elif branch == "privileged":
        teacher_ids = input_ids.detach().clone()
        prefix_width = min(32, int(teacher_ids.shape[1]))
        teacher_ids[:, :prefix_width] = torch.roll(
            teacher_ids[:, :prefix_width], shifts=1, dims=1
        )
        model.eval()
        with torch.no_grad():
            teacher_hidden_all, _ = backbone_forward(
                model,
                input_ids=teacher_ids,
                attention_mask=torch.ones_like(teacher_ids),
                use_cache=False,
            )
        model.train()
        teacher_hidden = teacher_hidden_all[:, :-1]

        def teacher_for_chunk(_student_logits, _hidden_chunk, start, stop):
            return project_logits(model, teacher_hidden[:, start:stop]).detach()

    else:
        raise ValueError(f"unknown validation branch {branch!r}")

    result = stream_distillation_chunks(
        student_hidden=student_hidden,
        labels=labels,
        project_student=lambda hidden: project_logits(model, hidden),
        teacher_for_chunk=teacher_for_chunk,
        chunk_size=chunk_size,
        top_k=top_k,
        temperature=temperature,
        token_clip=0.0,
        backward=True,
    )
    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm).item())
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError(f"{branch} produced invalid gradient norm {grad_norm}")
    optimizer.step()
    changed = any(
        not torch.equal(prior, parameter.detach())
        for prior, parameter in zip(before, parameters)
    )
    if not changed:
        raise RuntimeError(f"{branch} optimizer step did not change a LoRA parameter")
    if not math.isfinite(result.loss) or result.loss <= 0:
        raise RuntimeError(f"{branch} produced invalid loss {result.loss}")
    if result.token_count != int(input_ids.shape[1]) - 1:
        raise RuntimeError("streamed token count disagrees with the full sequence")
    if result.max_chunk_tokens > chunk_size:
        raise RuntimeError("a vocabulary projection exceeded the frozen chunk cap")
    torch.cuda.synchronize(device)
    payload = {
        "branch": branch,
        "sequence_tokens": int(input_ids.shape[1]),
        "distilled_positions": result.token_count,
        "chunk_size": chunk_size,
        "max_projected_chunk_tokens": result.max_chunk_tokens,
        "loss": result.loss,
        "mean_teacher_student_kl": result.mean_kl,
        "gradient_norm_before_clip": grad_norm,
        "optimizer_changed_trainable_parameter": changed,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    del result, optimizer, before, labels, student_hidden, student_all
    if teacher_hidden_all is not None:
        del teacher_hidden_all, teacher_hidden
    if branch == "clean":
        del adapter
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--code-tree-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-sequence-tokens", type=int, required=True)
    parser.add_argument("--distill-token-chunk-size", type=int, required=True)
    parser.add_argument("--distill-top-k", type=int, default=64)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--expected-gpu-name-regex", required=True)
    parser.add_argument("--expected-gpu-capability-major", type=int, required=True)
    parser.add_argument("--expected-gpu-capability-minor", type=int, required=True)
    parser.add_argument("--expected-gpu-arch-flag", required=True)
    parser.add_argument("--minimum-reserved-headroom-bytes", type=int, required=True)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()

    if args.max_sequence_tokens != 16_384:
        raise ValueError("formal memory validation must use exactly 16,384 tokens")
    if not re.fullmatch(r"[0-9a-f]{40}", args.git_commit):
        raise ValueError("formal memory validation requires a full git commit")
    if not re.fullmatch(r"[0-9a-f]{64}", args.code_tree_sha256):
        raise ValueError("formal memory validation requires a code-tree SHA256")
    if args.minimum_reserved_headroom_bytes < 1_000_000_000:
        raise ValueError("formal H100 validation requires at least 1 GB headroom")
    queries = load_query_only_manifest(args.queries)
    if not queries:
        raise ValueError("query-only validation manifest is empty")
    query = dict(queries[0])
    exposed = sorted(FORBIDDEN_QUERY_KEYS & {str(key).casefold() for key in query})
    if exposed:
        raise ValueError(f"query-only validation row exposes forbidden fields {exposed}")
    if args.validate_existing:
        with Path(args.output).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("full-sequence artifact must be a JSON object")
        _validate_artifact(payload, args)
        return

    local_model = Path(args.model)
    model, tokenizer = load_hf_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        training=True,
        revision=None if local_model.exists() else args.revision,
    )
    runtime = collect_runtime_metadata(
        model, model_path=args.model_id, revision=args.revision
    )
    if runtime.get("model_use_cache") is not False:
        raise RuntimeError("training runtime did not disable the model cache")
    if runtime.get("gradient_checkpointing_enabled") is not True:
        raise RuntimeError("training runtime did not enable gradient checkpointing")
    if runtime.get("gradient_checkpointing_use_reentrant") is not False:
        raise RuntimeError("formal runtime requires non-reentrant checkpointing")
    if runtime.get("git_commit") != args.git_commit or runtime.get(
        "code_tree_sha256"
    ) != args.code_tree_sha256:
        raise RuntimeError("runtime provenance disagrees with the immutable archive")
    device = input_device(model)
    if device.type != "cuda" or torch.cuda.device_count() != 1:
        raise RuntimeError("full-sequence validation requires exactly one visible GPU")
    input_ids = _full_length_ids(
        tokenizer, str(query["problem"]), args.max_sequence_tokens, device
    )
    results = [
        _run_branch(
            model=model,
            input_ids=input_ids,
            branch=branch,
            chunk_size=args.distill_token_chunk_size,
            top_k=args.distill_top_k,
            temperature=args.distill_temperature,
            max_grad_norm=args.max_grad_norm,
        )
        for branch in ("clean", "privileged")
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "query_id": query["query_id"],
        "problem_sha256": query["problem_sha256"],
        "target_fields_loaded": False,
        "binding": _validation_binding(args),
        "runtime": runtime,
        "device_placement": _model_device_placement(model),
        "branches": results,
    }
    _validate_artifact(payload, args)
    _atomic_json(Path(args.output), payload)
    with Path(args.output).open("r", encoding="utf-8") as stream:
        persisted = json.load(stream)
    _validate_artifact(persisted, args)


if __name__ == "__main__":
    main()
