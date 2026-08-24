"""Focused, download-free tests for model training runtime configuration."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

from src.clean_self_distill.runtime import (
    collect_runtime_metadata,
    lora_target_modules,
    load_hf_model,
)


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = None
        self.pad_token = None
        self.eos_token = "<eos>"
        self.padding_side = "right"


class ModernFakeModel:
    def __init__(self):
        self.config = SimpleNamespace(use_cache=True, _commit_hash="resolved-revision")
        self.is_gradient_checkpointing = False
        self.gradient_checkpointing_kwargs = None
        self.train_modes = []
        self.input_require_grads_enabled = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs
        self.is_gradient_checkpointing = True

    def train(self, mode):
        self.train_modes.append(mode)
        return self

    def enable_input_require_grads(self):
        self.input_require_grads_enabled = True


class LegacyFakeWrapper:
    def __init__(self, base_model):
        self.base_model = base_model
        self.config = SimpleNamespace(use_cache=True)
        self.gradient_checkpointing_calls = 0
        self.train_modes = []
        self.input_require_grads_enabled = False

    @property
    def is_gradient_checkpointing(self):
        return self.base_model.is_gradient_checkpointing

    def get_base_model(self):
        return self.base_model

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_calls += 1
        self.base_model.is_gradient_checkpointing = True

    def train(self, mode):
        self.train_modes.append(mode)
        return self

    def enable_input_require_grads(self):
        self.input_require_grads_enabled = True


def _fake_torch():
    return SimpleNamespace(
        __version__="test",
        __file__="/fake/torch/__init__.py",
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        ),
        _C=SimpleNamespace(_cuda_getArchFlags=lambda: ""),
    )


def _fake_transformers(model, tokenizer):
    return SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model
        ),
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: tokenizer
        ),
    )


def _metadata(model, fake_torch):
    with (
        mock.patch.dict(sys.modules, {"torch": fake_torch}),
        mock.patch(
            "src.clean_self_distill.runtime.platform.platform",
            return_value="test-platform",
        ),
        mock.patch(
            "src.clean_self_distill.runtime.subprocess.check_output",
            side_effect=["commit\n", ""],
        ),
    ):
        return collect_runtime_metadata(model, revision="requested-revision")


def test_training_enables_non_reentrant_checkpointing_and_disables_cache():
    model = ModernFakeModel()
    tokenizer = FakeTokenizer()
    fake_torch = _fake_torch()
    fake_transformers = _fake_transformers(model, tokenizer)

    with mock.patch.dict(
        sys.modules,
        {"torch": fake_torch, "transformers": fake_transformers},
    ):
        loaded_model, loaded_tokenizer = load_hf_model("fake/model", training=True)

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert model.config.use_cache is False
    assert model.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert model.is_gradient_checkpointing is True
    assert model.train_modes == [True]
    assert model.input_require_grads_enabled is True
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.padding_side == "left"

    metadata = _metadata(model, fake_torch)
    assert metadata["model_use_cache"] is False
    assert metadata["gradient_checkpointing_enabled"] is True
    assert metadata["gradient_checkpointing_use_reentrant"] is False


def test_lora_wrapper_uses_underlying_config_and_legacy_signature_fallback():
    base_model = ModernFakeModel()
    wrapper = LegacyFakeWrapper(base_model)
    tokenizer = FakeTokenizer()
    fake_torch = _fake_torch()
    fake_transformers = _fake_transformers(base_model, tokenizer)
    fake_peft = SimpleNamespace(
        LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        get_peft_model=lambda model, config: wrapper,
    )

    with mock.patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "transformers": fake_transformers,
            "peft": fake_peft,
        },
    ):
        loaded_model, _ = load_hf_model(
            "fake/model",
            training=True,
            use_lora=True,
        )

    assert loaded_model is wrapper
    assert base_model.config.use_cache is False
    assert wrapper.config.use_cache is True
    assert wrapper.gradient_checkpointing_calls == 1
    assert base_model.gradient_checkpointing_kwargs is None
    assert wrapper.train_modes == [True]
    assert wrapper.input_require_grads_enabled is True

    metadata = _metadata(wrapper, fake_torch)
    assert metadata["resolved_model_revision"] == "resolved-revision"
    assert metadata["model_use_cache"] is False
    assert metadata["gradient_checkpointing_enabled"] is True
    assert metadata["gradient_checkpointing_use_reentrant"] is None


def test_inference_leaves_cache_and_checkpointing_untouched():
    model = ModernFakeModel()
    tokenizer = FakeTokenizer()
    fake_torch = _fake_torch()
    fake_transformers = _fake_transformers(model, tokenizer)

    with mock.patch.dict(
        sys.modules,
        {"torch": fake_torch, "transformers": fake_transformers},
    ):
        loaded_model, _ = load_hf_model("fake/model", training=False)

    assert loaded_model is model
    assert model.config.use_cache is True
    assert model.gradient_checkpointing_kwargs is None
    assert model.is_gradient_checkpointing is False
    assert model.train_modes == [False]
    assert model.input_require_grads_enabled is False

    metadata = _metadata(model, fake_torch)
    assert metadata["model_use_cache"] is True
    assert metadata["gradient_checkpointing_enabled"] is False
    assert metadata["gradient_checkpointing_use_reentrant"] is None


def test_gpt_oss_lora_targets_only_standard_attention_linears():
    model = SimpleNamespace(config=SimpleNamespace(model_type="gpt_oss"))
    assert lora_target_modules(model) == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_dense_lora_targets_keep_attention_and_mlp_projections():
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
    assert lora_target_modules(model) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
