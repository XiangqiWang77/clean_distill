# Run TRSD and OPSD

Set the pinned Qwen3-8B identity. `MODEL_DIR` may be either a local snapshot or
the public model ID when using the Hugging Face engine.

```bash
export MODEL_ID=Qwen/Qwen3-8B
export MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
export MODEL_DIR=Qwen/Qwen3-8B
```

## Download a checkpoint from GitHub

```bash
python scripts/download_checkpoints.py --method trsd
# or
python scripts/download_checkpoints.py --method opsd
```

The downloader infers the GitHub repository from `origin`. Use
`--repository OWNER/REPOSITORY` when running outside a Git checkout.

## Load the local adapter

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

method = "trsd"
adapter_dir = f"checkpoints/qwen3-8b-{method}-ep64"
tokenizer = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REVISION)
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    revision=BASE_REVISION,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    base,
    adapter_dir,
    is_trainable=False,
)
model.eval()

messages = [{"role": "user", "content": "Solve the problem and box the final answer."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
with torch.inference_mode():
    output = model.generate(inputs, max_new_tokens=1024)
print(tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True))
```

## Input format

Training and generation consume target-free JSONL. Each row must contain only
the query identity, exact problem text, its SHA-256 digest, and a source label:

```json
{"query_id":"custom:0001","problem":"...","problem_sha256":"<sha256-of-exact-utf8-problem>","source":"custom"}
```

Do not place answers, solutions, references, rewards, or feedback in this
file. Use one row per requested training episode.

## Train

The CLI names TRSD `clean` and OPSD `privileged`. The two commands otherwise
share the same initialization, query order, optimizer, and rollout settings.

TRSD:

```bash
python scripts/clean_self_distill/04_persistent_train.py \
  --branch clean \
  --queries data/train_queries.jsonl \
  --model "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --output-dir outputs/trsd \
  --episodes 64 \
  --scientific-checkpoints 0,16,32,48,64 \
  --rolling-checkpoint-interval 8 \
  --max-sequence-tokens 10240 \
  --max-rollout-tokens 10240 \
  --learning-rate 2e-5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --distill-token-chunk-size 128 \
  --trust-region-kl-budget 0.004 \
  --trust-region-binary-search-steps 6 \
  --seed 0 \
  --attn-implementation sdpa
```

OPSD:

```bash
python scripts/clean_self_distill/04_persistent_train.py \
  --branch privileged \
  --queries data/train_queries.jsonl \
  --model "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --output-dir outputs/opsd \
  --episodes 64 \
  --scientific-checkpoints 0,16,32,48,64 \
  --rolling-checkpoint-interval 8 \
  --max-sequence-tokens 10240 \
  --max-rollout-tokens 10240 \
  --learning-rate 2e-5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --distill-token-chunk-size 128 \
  --seed 0 \
  --attn-implementation sdpa
```

Add `--resume` to the identical command to resume from the newest validated
checkpoint. Final adapters are written under
`OUTPUT_DIR/checkpoints/episode_0064`.

## Generate and score

Use a local training checkpoint or download the GitHub-hosted adapter first:

```bash
python scripts/download_checkpoints.py --method trsd

python scripts/clean_self_distill/05_heldout_eval.py generate \
  --queries data/eval_queries.jsonl \
  --output outputs/trsd_predictions.jsonl \
  --model "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --adapter checkpoints/qwen3-8b-trsd-ep64 \
  --method trsd \
  --checkpoint-episode 64 \
  --sample-count 1 \
  --seed 0
```

Scoring is a separate CPU-only step. The sealed label file is joined only
after generation finishes:

```bash
python scripts/clean_self_distill/05_heldout_eval.py score \
  --predictions outputs/trsd_predictions.jsonl \
  --labels data/eval_labels.sealed.jsonl \
  --output outputs/trsd_scored.jsonl \
  --sample-count 1
```
