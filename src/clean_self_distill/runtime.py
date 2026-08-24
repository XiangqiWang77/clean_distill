"""Shared Hugging Face runtime utilities."""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
import time
from typing import Any, Optional


_GRADIENT_CHECKPOINTING_ENABLED_ATTR = "_csd_gradient_checkpointing_enabled"
_GRADIENT_CHECKPOINTING_REENTRANT_ATTR = (
    "_csd_gradient_checkpointing_use_reentrant"
)


def resolve_dtype(name: str):
    import torch

    normalized = str(name).strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(mapping)}")
    return mapping[normalized]


def load_hf_model(
    model_path: str,
    *,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
    use_lora: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    training: bool = False,
    revision: Optional[str] = None,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = resolve_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, revision=revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": device_map,
        "revision": revision,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("--use-lora requires peft (pip install peft).") from exc
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules(model),
        )
        model = get_peft_model(model, lora_config)

    if training:
        _enable_gradient_checkpointing(model)
    model.train(training)
    if training and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def lora_target_modules(model) -> list[str]:
    """Select trainable linear projections supported by the base architecture."""
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    attention = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if model_type == "gpt_oss":
        # GPT-OSS stores MoE expert matrices in MXFP4 block tensors rather than
        # ordinary nn.Linear modules.  Attention projections remain standard
        # linear layers and are supported by PEFT/vLLM LoRA end to end.
        return attention
    return attention + ["gate_proj", "up_proj", "down_proj"]


def _underlying_config(model):
    """Resolve the causal LM config through wrappers such as ``PeftModel``."""
    base_model = unwrap_causal_lm(model)
    config = getattr(base_model, "config", None)
    if config is None:
        config = getattr(model, "config", None)
    return config


def _record_gradient_checkpointing_mode(model, *, use_reentrant: Optional[bool]) -> None:
    """Keep the resolved call mode available for runtime provenance."""
    base_model = unwrap_causal_lm(model)
    for candidate in (model, base_model):
        try:
            setattr(candidate, _GRADIENT_CHECKPOINTING_ENABLED_ATTR, True)
            setattr(
                candidate,
                _GRADIENT_CHECKPOINTING_REENTRANT_ATTR,
                use_reentrant,
            )
        except (AttributeError, TypeError):
            # Some third-party wrappers restrict arbitrary attributes. The
            # model's own ``is_gradient_checkpointing`` flag remains usable.
            pass


def _enable_gradient_checkpointing(model) -> None:
    """Enable memory-saving training with old/new Transformers compatibility."""
    config = _underlying_config(model)
    if config is None:
        raise AttributeError(
            f"{type(model).__name__} has no underlying model config to disable caching"
        )
    config.use_cache = False

    base_model = unwrap_causal_lm(model)
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        enable = getattr(base_model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise AttributeError(
            f"{type(model).__name__} does not support gradient checkpointing"
        )

    try:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError as modern_error:
        # Transformers <4.35 did not accept gradient_checkpointing_kwargs.
        # If the no-argument call also fails, preserve the original error,
        # which identifies the unsupported modern call more clearly.
        try:
            enable()
        except TypeError:
            raise modern_error
        _record_gradient_checkpointing_mode(model, use_reentrant=None)
    else:
        _record_gradient_checkpointing_mode(model, use_reentrant=False)


def _gradient_checkpointing_enabled(model) -> bool:
    """Resolve the effective checkpointing flag across wrapper boundaries."""
    base_model = unwrap_causal_lm(model)
    observed: list[bool] = []
    for candidate in (model, base_model):
        value = getattr(candidate, "is_gradient_checkpointing", None)
        if callable(value):
            value = value()
        if value is not None:
            observed.append(bool(value))
    if observed:
        return any(observed)
    return any(
        bool(getattr(candidate, _GRADIENT_CHECKPOINTING_ENABLED_ATTR, False))
        for candidate in (model, base_model)
    )


def _gradient_checkpointing_use_reentrant(model) -> Optional[bool]:
    """Return the explicit checkpoint mode, or ``None`` for a legacy call."""
    base_model = unwrap_causal_lm(model)
    for candidate in (model, base_model):
        if hasattr(candidate, _GRADIENT_CHECKPOINTING_REENTRANT_ATTR):
            return getattr(candidate, _GRADIENT_CHECKPOINTING_REENTRANT_ATTR)
    return None


def collect_runtime_metadata(model=None, *, model_path: str = "", revision: str = "") -> dict[str, Any]:
    """Collect the small, reproducibility-relevant runtime manifest."""
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "torch_overlay": os.environ.get("CSD_TORCH_OVERLAY", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "requested_gpu_partition": os.environ.get("CSD_GPU_PARTITION", ""),
        "requested_gpu_gres": os.environ.get("CSD_GPU_GRES", ""),
        "expected_gpu_name_regex": os.environ.get("CSD_GPU_NAME_REGEX", ""),
        "expected_gpu_capability": [
            int(os.environ.get("CSD_GPU_CAPABILITY_MAJOR", "-1")),
            int(os.environ.get("CSD_GPU_CAPABILITY_MINOR", "-1")),
        ],
        "expected_gpu_arch_flag": os.environ.get("CSD_GPU_ARCH_FLAG", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "model": model_path,
        "requested_model_revision": revision,
        "code_tree_sha256": os.environ.get("CSD_CODE_TREE_SHA256", ""),
    }
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        metadata["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        # Formal Slurm runs execute an immutable ``git archive`` snapshot, so
        # there is intentionally no .git directory on the compute node.  The
        # submitter exports the verified source commit and deterministic tree
        # hash through the immutable run configuration.
        metadata["git_commit"] = os.environ.get("CSD_GIT_COMMIT", "")
        metadata["git_dirty"] = False if metadata["git_commit"] else None
    try:
        import torch

        torch_module_path = getattr(torch, "__file__", "") or ""
        try:
            torch_arch_flags = torch._C._cuda_getArchFlags().split()
        except (AttributeError, RuntimeError):
            torch_arch_flags = []
        gpu_count = torch.cuda.device_count()
        metadata.update(
            {
                "torch": torch.__version__,
                "torch_module_path": str(torch_module_path),
                "torch_arch_flags": torch_arch_flags,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": gpu_count,
                "gpus": [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                        "capability": list(torch.cuda.get_device_capability(index)),
                    }
                    for index in range(gpu_count)
                ],
            }
        )
    except (ImportError, RuntimeError) as exc:
        metadata["torch_error"] = repr(exc)
    if model is not None:
        config = _underlying_config(model)
        metadata["resolved_model_revision"] = str(
            getattr(config, "_commit_hash", "") or revision
        )
        metadata["model_class"] = type(model).__name__
        use_cache = getattr(config, "use_cache", None)
        metadata["model_use_cache"] = (
            None if use_cache is None else bool(use_cache)
        )
        metadata["gradient_checkpointing_enabled"] = (
            _gradient_checkpointing_enabled(model)
        )
        metadata["gradient_checkpointing_use_reentrant"] = (
            _gradient_checkpointing_use_reentrant(model)
        )
    return metadata


def input_device(model):
    """Return the embedding device, including accelerate-sharded models."""
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, RuntimeError):
        return next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")


def unwrap_causal_lm(model):
    """Return the underlying causal LM while preserving injected LoRA modules."""
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except (AttributeError, TypeError):
            pass
    return model


def backbone_forward(
    model,
    *,
    input_ids,
    attention_mask=None,
    past_key_values=None,
    use_cache: bool = False,
):
    """Run the decoder without projecting every prompt token to the vocabulary.

    Qwen/Olmo expose the decoder as ``causal_lm.model``. A conservative full-LM
    fallback keeps the scripts usable with less common architectures.
    """
    causal_lm = unwrap_causal_lm(model)
    decoder = getattr(causal_lm, "model", None)
    if decoder is not None:
        outputs = decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        return outputs.last_hidden_state, getattr(outputs, "past_key_values", None)

    outputs = causal_lm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_hidden_states=True,
        return_dict=True,
    )
    return outputs.hidden_states[-1], getattr(outputs, "past_key_values", None)


def project_logits(model, hidden):
    causal_lm = unwrap_causal_lm(model)
    output_head = causal_lm.get_output_embeddings()
    if output_head is None:
        raise ValueError(f"{type(causal_lm).__name__} has no output embedding / LM head")
    return output_head(hidden)


def render_chat(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: Optional[bool] = None,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(messages, **kwargs)
    parts = [f"{message['role'].upper()}: {message['content']}" for message in messages]
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n\n".join(parts)


class HFGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        enable_thinking: Optional[bool] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_generation_seconds = 0.0

    def __call__(self, messages: list[dict[str, str]]) -> str:
        import torch

        prompt = render_chat(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(encoded["input_ids"].numel())
        encoded = {key: value.to(input_device(self.model)) for key, value in encoded.items()}
        do_sample = self.temperature > 0
        generation_kwargs = {
            **encoded,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=self.temperature, top_p=self.top_p)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(**generation_kwargs)
        elapsed = time.perf_counter() - started
        completion_ids = generated[0, encoded["input_ids"].shape[1] :]
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += int(completion_ids.numel())
        self.total_generation_seconds += elapsed
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    def counters(self) -> dict[str, float]:
        return {
            "prompt_tokens": float(self.total_prompt_tokens),
            "completion_tokens": float(self.total_completion_tokens),
            "generation_seconds": float(self.total_generation_seconds),
        }


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse strict JSON with a defensive fenced-text fallback."""
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidate = fenced.group(1) if fenced else None
        if candidate is None:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                candidate = text[start : end + 1]
        if candidate is None:
            raise ValueError(f"Model did not return a JSON object: {text[:300]!r}")
        value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from the model")
    return value
