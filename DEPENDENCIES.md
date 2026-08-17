# Dependency stack

TRSD is self-contained at the algorithm level. It does not wrap `verl`, TRL,
or another RL trainer. The persistent on-policy loop, privileged-teacher
scoring, trajectory-level projection, streaming distillation objective,
checkpointing, and resume logic live in `src/clean_self_distill`.

| Layer | Dependency |
|---|---|
| Tensor operations and optimization | PyTorch |
| Qwen model and tokenizer | Transformers |
| LoRA adapters | PEFT and safetensors |
| Device placement | Accelerate |
| Exact math answer scoring | math-verify |
| Dataset preparation | datasets and PyArrow |
| Optional batched evaluation | vLLM |
| Optional attention kernel | FlashAttention |
| Tests | pytest |

The dependency files are split by use case:

```text
requirements/core.txt       training and Hugging Face inference
requirements/data.txt       parquet and dataset preparation
requirements/eval.txt       optional vLLM evaluation
requirements/attention.txt  optional FlashAttention build
requirements/dev.txt        unit tests
```

The version ranges cover the validated stack while allowing compatible patch
updates. Install a CUDA-matched PyTorch wheel before FlashAttention or vLLM.
