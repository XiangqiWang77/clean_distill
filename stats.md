# Math evaluation statistics

Last updated: 2026-08-13

## Evaluation protocol

- Common held-out set: 143 problems: AMC 2022/2023 (83), AIME 2024 (30), and AIME 2025 (30).
- Metric: full-response `math_verify==0.9.0` Accuracy@1. Gold answers and complete generated responses are parsed with `LatexExtractionConfig` plus `ExprExtractionConfig`, then checked with `verify(..., strict=True)`.
- A generation reaching the 10,240-token cap is **not** automatically marked wrong; `Cap hits` is reported separately.
- Only existing completed generations were rescored. No model inference or regeneration was used.
- The main table contains 31 completed model/method/checkpoint rows. Together with seven Qwen3-8B ablations and five evaluation repeats, 6,149 responses were rescored. Two empty parses were counted as incorrect.
- Main TRSD runs use the global trust-region budget `epsilon=0.004`.
- `N/A` means that no completed 143-response generation artifact exists; it does not mean zero accuracy.

## Main results

### Qwen3-1.7B

| Method | Episodes | AMC 2022/2023 | AIME 2024 | AIME 2025 | Combined | Cap hits |
|---|---:|---:|---:|---:|---:|---:|
| Base | — | 63.86% (53/83) | 43.33% (13/30) | 26.67% (8/30) | **51.75% (74/143)** | 71/143 |
| OPSD | 16 | 69.88% (58/83) | 33.33% (10/30) | 30.00% (9/30) | **53.85% (77/143)** | 65/143 |
| SRPO | 16 | 57.83% (48/83) | 40.00% (12/30) | 30.00% (9/30) | **48.25% (69/143)** | 70/143 |
| TRSD | 16 | 68.67% (57/83) | 33.33% (10/30) | 33.33% (10/30) | **53.85% (77/143)** | 63/143 |
| DemoPSD | 64 | 69.88% (58/83) | 40.00% (12/30) | 30.00% (9/30) | **55.24% (79/143)** | 68/143 |
| Outcome-GRPO | 64 | 56.63% (47/83) | 40.00% (12/30) | 30.00% (9/30) | **47.55% (68/143)** | 68/143 |
| GRPO-PRM | 64 | 61.45% (51/83) | 23.33% (7/30) | 26.67% (8/30) | **46.15% (66/143)** | 72/143 |
| OPSD | 64 | 24.10% (20/83) | 6.67% (2/30) | 3.33% (1/30) | **16.08% (23/143)** | 10/143 |
| SRPO | 64 | 65.06% (54/83) | 36.67% (11/30) | 26.67% (8/30) | **51.05% (73/143)** | 71/143 |
| TRSD | 64 | 22.89% (19/83) | 0.00% (0/30) | 0.00% (0/30) | **13.29% (19/143)** | 8/143 |

Unavailable: DemoPSD-16, Outcome-GRPO-16, and GRPO-PRM-16.

### Qwen3-8B

| Method | Episodes | AMC 2022/2023 | AIME 2024 | AIME 2025 | Combined | Cap hits |
|---|---:|---:|---:|---:|---:|---:|
| Base | — | 74.70% (62/83) | 60.00% (18/30) | 40.00% (12/30) | **64.34% (92/143)** | 65/143 |
| DemoPSD | 16 | 75.90% (63/83) | 63.33% (19/30) | 43.33% (13/30) | **66.43% (95/143)** | 67/143 |
| Outcome-GRPO | 16 | 79.52% (66/83) | 46.67% (14/30) | 43.33% (13/30) | **65.03% (93/143)** | 62/143 |
| OPSD | 16 | 75.90% (63/83) | 70.00% (21/30) | 50.00% (15/30) | **69.23% (99/143)** | 60/143 |
| SRPO | 16 | 75.90% (63/83) | 56.67% (17/30) | 33.33% (10/30) | **62.94% (90/143)** | 69/143 |
| TRSD | 16 | 77.11% (64/83) | 60.00% (18/30) | 43.33% (13/30) | **66.43% (95/143)** | 65/143 |
| OPSD | 32 | 79.52% (66/83) | 56.67% (17/30) | 43.33% (13/30) | **67.13% (96/143)** | 59/143 |
| TRSD | 32 | 69.88% (58/83) | 46.67% (14/30) | 50.00% (15/30) | **60.84% (87/143)** | 63/143 |
| OPSD | 48 | 83.13% (69/83) | 56.67% (17/30) | 50.00% (15/30) | **70.63% (101/143)** | 51/143 |
| TRSD | 48 | 78.31% (65/83) | 63.33% (19/30) | 43.33% (13/30) | **67.83% (97/143)** | 54/143 |
| GRPO-PRM | 64 | 77.11% (64/83) | 56.67% (17/30) | 40.00% (12/30) | **65.03% (93/143)** | 68/143 |
| OPSD | 64 | 81.93% (68/83) | 63.33% (19/30) | 40.00% (12/30) | **69.23% (99/143)** | 43/143 |
| SRPO | 64 | 78.31% (65/83) | 50.00% (15/30) | 36.67% (11/30) | **63.64% (91/143)** | 67/143 |
| TRSD (`epsilon=0.004`) | 64 | 85.54% (71/83) | 70.00% (21/30) | 43.33% (13/30) | **73.43% (105/143)** | 25/143 |

Unavailable: DemoPSD-64, Outcome-GRPO-64, and GRPO-PRM-16.

### GPT-OSS-20B

| Method | Episodes | AMC 2022/2023 | AIME 2024 | AIME 2025 | Combined | Cap hits |
|---|---:|---:|---:|---:|---:|---:|
| Base | — | 86.75% (72/83) | 73.33% (22/30) | 53.33% (16/30) | **76.92% (110/143)** | 37/143 |
| OPSD | 16 | 87.95% (73/83) | 56.67% (17/30) | 60.00% (18/30) | **75.52% (108/143)** | 17/143 |
| SRPO | 16 | 87.95% (73/83) | 50.00% (15/30) | 60.00% (18/30) | **74.13% (106/143)** | 33/143 |
| TRSD | 16 | 81.93% (68/83) | 76.67% (23/30) | 70.00% (21/30) | **78.32% (112/143)** | 22/143 |
| OPSD | 64 | 37.35% (31/83) | 6.67% (2/30) | 6.67% (2/30) | **24.48% (35/143)** | 0/143 |
| SRPO | 64 | 90.36% (75/83) | 73.33% (22/30) | 60.00% (18/30) | **80.42% (115/143)** | 34/143 |
| TRSD | 64 | 75.90% (63/83) | 53.33% (16/30) | 33.33% (10/30) | **62.24% (89/143)** | 2/143 |

Unavailable: DemoPSD-16/64, Outcome-GRPO-16/64, and GRPO-PRM-16/64.

## Qwen3-8B 64-episode controls and projection ablations

| Method | AMC 2022/2023 | AIME 2024 | AIME 2025 | Combined | Cap hits |
|---|---:|---:|---:|---:|---:|
| OPSD, LR=1e-5 | 75.90% (63/83) | 53.33% (16/30) | 46.67% (14/30) | **65.03% (93/143)** | 61/143 |
| OPSD, LoRA r4/a8 | 78.31% (65/83) | 60.00% (18/30) | 43.33% (13/30) | **67.13% (96/143)** | 55/143 |
| OPSD, gradient clip=0.5 | 78.31% (65/83) | 56.67% (17/30) | 43.33% (13/30) | **66.43% (95/143)** | 44/143 |
| OPSD, policy-KL beta=1 | 73.49% (61/83) | 60.00% (18/30) | 36.67% (11/30) | **62.94% (90/143)** | 61/143 |
| OPSD, update scale=0.5 | 79.52% (66/83) | 46.67% (14/30) | 46.67% (14/30) | **65.73% (94/143)** | 64/143 |
| TRSD, fixed global alpha | 80.72% (67/83) | 63.33% (19/30) | 50.00% (15/30) | **70.63% (101/143)** | 45/143 |
| TRSD, tokenwise `alpha_t` | 75.90% (63/83) | 56.67% (17/30) | 50.00% (15/30) | **66.43% (95/143)** | 45/143 |

## Qwen3-8B evaluation repeat

The following vLLM rows are independent evaluation generations from the same checkpoint family. They are reported for audit and are not averaged with the canonical main-table rows.

| Method | Episodes | AMC 2022/2023 | AIME 2024 | AIME 2025 | Combined | Cap hits |
|---|---:|---:|---:|---:|---:|---:|
| Base | — | 74.70% (62/83) | 56.67% (17/30) | 43.33% (13/30) | **64.34% (92/143)** | 62/143 |
| OPSD | 16 | 78.31% (65/83) | 63.33% (19/30) | 50.00% (15/30) | **69.23% (99/143)** | 60/143 |
| TRSD | 16 | 78.31% (65/83) | 60.00% (18/30) | 50.00% (15/30) | **68.53% (98/143)** | 66/143 |
| OPSD | 64 | 81.93% (68/83) | 56.67% (17/30) | 40.00% (12/30) | **67.83% (97/143)** | 48/143 |
| TRSD | 64 | 75.90% (63/83) | 53.33% (16/30) | 46.67% (14/30) | **65.03% (93/143)** | 40/143 |

## Notes

- The canonical Qwen3-8B TRSD-64 output scores **105/143**, not 120/143, under the full-response Math-Verify protocol above. Its older last-box strict score was 102/143. The independent vLLM repeat scores 93/143. No currently available complete 143-response TRSD-64 artifact reproduces 120/143.
- Accuracy rows from different baseline runs are directly comparable on query IDs and evaluation set, but are not all query-seed paired. Do not describe every cross-method difference as a paired comparison.
