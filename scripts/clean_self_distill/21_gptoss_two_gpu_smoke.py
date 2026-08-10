#!/usr/bin/env python3
"""Verify that GPT-OSS training is genuinely sharded across two CUDA devices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.clean_self_distill.runtime import (
    backbone_forward,
    input_device,
    load_hf_model,
    unwrap_causal_lm,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--revision", required=True)
    result.add_argument("--sequence-tokens", type=int, default=10_240)
    result.add_argument("--output", type=Path, required=True)
    return result


def cuda_indices(model) -> set[int]:
    candidates = [model, unwrap_causal_lm(model)]
    indices: set[int] = set()
    for candidate in candidates:
        device_map = getattr(candidate, "hf_device_map", {}) or {}
        for value in device_map.values():
            device = torch.device(f"cuda:{value}" if isinstance(value, int) else value)
            if device.type == "cuda" and device.index is not None:
                indices.add(device.index)
    if not indices:
        for parameter in model.parameters():
            if parameter.device.type == "cuda" and parameter.device.index is not None:
                indices.add(parameter.device.index)
    return indices


def main() -> int:
    args = parser().parse_args()
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected exactly two CUDA devices, got {torch.cuda.device_count()}")
    model, tokenizer = load_hf_model(
        str(args.model),
        dtype="bfloat16",
        device_map="balanced",
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        training=True,
        revision=None,
    )
    devices = cuda_indices(model)
    if devices != {0, 1}:
        raise RuntimeError(f"Balanced placement did not use both GPUs: {sorted(devices)}")

    token_id = tokenizer.eos_token_id
    if token_id is None:
        raise RuntimeError("Tokenizer has no EOS token")
    input_ids = torch.full(
        (1, args.sequence_tokens),
        int(token_id),
        dtype=torch.long,
        device=input_device(model),
    )
    hidden, _ = backbone_forward(
        model,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
    )
    loss = hidden[:, -128:].float().square().mean()
    loss.backward()
    peak = {
        str(index): int(torch.cuda.max_memory_allocated(index))
        for index in range(torch.cuda.device_count())
    }
    payload = {
        "status": "complete",
        "cuda_devices": sorted(devices),
        "sequence_tokens": args.sequence_tokens,
        "attention_implementation": "eager",
        "peak_memory_bytes": peak,
        "loss": float(loss.detach().cpu()),
        "hf_device_map": getattr(unwrap_causal_lm(model), "hf_device_map", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
