"""Shared Hugging Face runtime utilities."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional


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
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = resolve_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": device_map,
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
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    model.train(training)
    if training and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


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


def render_chat(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_generation_seconds = 0.0

    def __call__(self, messages: list[dict[str, str]]) -> str:
        import torch

        prompt = render_chat(self.tokenizer, messages, add_generation_prompt=True)
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
