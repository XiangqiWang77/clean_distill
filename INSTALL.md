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

Install the PyTorch build appropriate for the host CUDA driver first, then the
core training stack:

```bash
python -m pip install -r requirements/core.txt
python -m pip install -e . --no-deps
```

Install only the additional layer needed by the workflow:

```bash
python -m pip install -r requirements/data.txt       # parquet preparation
python -m pip install -r requirements/eval.txt       # vLLM evaluation
python -m pip install -r requirements/attention.txt  # optional FlashAttention
python -m pip install -r requirements/dev.txt        # tests
```

See [DEPENDENCIES.md](DEPENDENCIES.md) for the full component mapping and
version policy. Unlike the RL pipeline, TRSD does not depend on `verl`: the
persistent self-distillation trainer and streaming loss are implemented in
this repository.

The pinned backbone can be used directly by repository ID or downloaded once
to shared storage:

```bash
export MODEL_ID=Qwen/Qwen3-8B
export MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
export MODEL_DIR=/path/to/shared/models/Qwen3-8B-$MODEL_REVISION

hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
```

The two released LoRA checkpoints are downloaded from GitHub with the `gh`
CLI; see [RUN.md](RUN.md).
