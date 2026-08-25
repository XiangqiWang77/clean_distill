# Run LGSD, OPSD, and Veto

Set the pinned Qwen3-8B identity. `MODEL_DIR` may be either a local snapshot or
the public model ID when using the Hugging Face engine.

```bash
export MODEL_ID=Qwen/Qwen3-8B
export MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
export MODEL_DIR=Qwen/Qwen3-8B
```

## Download a legacy checkpoint from GitHub

```bash
python scripts/download_checkpoints.py --method legacy-trsd
```

This is the old reverse-KL checkpoint and is not a checkpoint for the forward-KL
implementation below. The downloader infers the GitHub repository from
`origin`; use `--repository OWNER/REPOSITORY` outside a Git checkout.

## Load the local adapter

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

method = "lgsd"
adapter_dir = "outputs/lgsd/checkpoints/episode_0064"
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

The CLI names LGSD `clean`, OPSD `privileged`, and the Veto baseline `veto`.
Matched commands otherwise share initialization, query order, optimizer, and
rollout settings.

LGSD:

```bash
python scripts/clean_self_distill/04_persistent_train.py \
  --branch clean \
  --queries data/train_queries.jsonl \
  --model "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --output-dir outputs/lgsd \
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
  --student-kl-direction forward \
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
  --student-kl-direction forward \
  --seed 0 \
  --attn-implementation sdpa
```

Veto (published reasoning default, `beta: 0.8 -> 0`):

```bash
python scripts/clean_self_distill/04_persistent_train.py \
  --branch veto \
  --queries data/train_queries.jsonl \
  --model "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --output-dir outputs/veto \
  --episodes 64 \
  --scientific-checkpoints 0,16,32,48,64 \
  --rolling-checkpoint-interval 8 \
  --max-sequence-tokens 10240 \
  --max-rollout-tokens 10240 \
  --learning-rate 2e-5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --distill-token-chunk-size 128 \
  --student-kl-direction forward \
  --veto-beta-start 0.8 \
  --veto-beta-end 0.0 \
  --veto-beta-schedule linear \
  --seed 0 \
  --attn-implementation sdpa
```

This is a matched DeepMath adaptation of Veto's published target equation, not
a reproduction of its original Qwen2-0.5B/Qwen2-7B experiment. See
[VETO_BASELINE.md](VETO_BASELINE.md) before reporting comparisons.

Add `--resume` to the identical command to resume from the newest validated
checkpoint. Final adapters are written under
`OUTPUT_DIR/checkpoints/episode_0064`.

## Generate and score

Use the local checkpoint produced by the forward-KL training command above:

```bash
python scripts/clean_self_distill/05_heldout_eval.py generate \
  --queries data/eval_queries.jsonl \
  --output outputs/lgsd_predictions.jsonl \
  --model "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --adapter outputs/lgsd/checkpoints/episode_0064 \
  --method lgsd \
  --checkpoint-episode 64 \
  --sample-count 1 \
  --seed 0
```

Scoring is a separate CPU-only step. The sealed label file is joined only
after generation finishes:

```bash
python scripts/clean_self_distill/05_heldout_eval.py score \
  --predictions outputs/lgsd_predictions.jsonl \
  --labels data/eval_labels.sealed.jsonl \
  --output outputs/lgsd_scored.jsonl \
  --sample-count 1
```
