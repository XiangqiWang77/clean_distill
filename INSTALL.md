# Installation

Use Python 3.10 or newer. Training Qwen3-8B requires a CUDA GPU with enough
memory for the base model, LoRA state, and optimizer state.

```bash
git clone <repository-url> trsd
cd trsd

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the PyTorch build appropriate for the host CUDA driver first. For
example, consult the [PyTorch installer](https://pytorch.org/get-started/locally/),
then install the repository:

```bash
python -m pip install -e .
```

Optional extras are isolated from the core installation:

```bash
python -m pip install -e '.[data]'   # parquet data preparation
python -m pip install -e '.[vllm]'   # batched vLLM evaluation
python -m pip install -e '.[dev]'    # tests
```

The pinned backbone can be used directly by repository ID or downloaded once
to shared storage:

```bash
export MODEL_ID=Qwen/Qwen3-8B
export MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
export MODEL_DIR=/path/to/shared/models/Qwen3-8B-$MODEL_REVISION

hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
```

No API token is required for the public base model or published adapters.
