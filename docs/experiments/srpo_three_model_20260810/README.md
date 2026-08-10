# Fast-budget SRPO results

All reported accuracies use the same frozen 143-question AMC23/AIME24/AIME25
scorer and a 10,240-token evaluation limit. Training uses 64 DeepMath episodes,
four rollouts per query, and a 2,048-token rollout limit. This is the requested
speed-prioritized SRPO baseline, not a full-budget SRPO reproduction.

| Model | Method | AMC23 | AIME24 | AIME25 | Combined | Cap hits |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | Base | 56.63% (47/83) | 33.33% (10/30) | 23.33% (7/30) | 44.76% (64/143) | 71/143 |
| Qwen3-1.7B | SRPO 16 | 55.42% (46/83) | 30.00% (9/30) | 26.67% (8/30) | 44.06% (63/143) | 70/143 |
| Qwen3-1.7B | SRPO 64 | 55.42% (46/83) | 33.33% (10/30) | 26.67% (8/30) | 44.76% (64/143) | 71/143 |
| Qwen3-8B | Base | 65.06% (54/83) | 50.00% (15/30) | 26.67% (8/30) | 53.85% (77/143) | 65/143 |
| Qwen3-8B | SRPO 16 | 65.06% (54/83) | 43.33% (13/30) | 23.33% (7/30) | 51.75% (74/143) | 69/143 |
| Qwen3-8B | SRPO 64 | 68.67% (57/83) | 36.67% (11/30) | 23.33% (7/30) | 52.45% (75/143) | 67/143 |
| GPT-OSS-20B | Base | 81.93% (68/83) | 70.00% (21/30) | 46.67% (14/30) | 72.03% (103/143) | 37/143 |
| GPT-OSS-20B | SRPO 16 | 86.75% (72/83) | 46.67% (14/30) | 60.00% (18/30) | 72.73% (104/143) | 33/143 |
| GPT-OSS-20B | SRPO 64 | 84.34% (70/83) | 63.33% (19/30) | 60.00% (18/30) | 74.83% (107/143) | 34/143 |

Relative to Base, SRPO-64 changes combined Strict Acc@1 by +0.00 points on
Qwen3-1.7B, -1.40 points on Qwen3-8B, and +2.80 points on GPT-OSS-20B.

## Training-route audit

| Model | Rollouts | Correct | SDPO-routed | GRPO-routed | Training cap hits |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 256 | 6 | 10 | 246 | 256 |
| Qwen3-8B | 256 | 9 | 3 | 253 | 254 |
| GPT-OSS-20B | 256 | 65 | 27 | 229 | 163 |

The short training horizon makes the two Qwen runs predominantly GRPO-routed;
the GPT-OSS run receives substantially more usable self-distillation routes.
The complete cross-method table is in `complete_math_table.csv`.
