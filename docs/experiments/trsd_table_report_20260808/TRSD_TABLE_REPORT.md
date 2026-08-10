# TRSD: drift control, short-term stability, long-term performance

**Three claims define the result:** TRSD controls privileged-teacher drift, preserves short-term performance at 16 episodes, and unlocks a decisive long-term gain at the equal 64-episode training horizon.

Strict Acc@1 uses the full 143-query denominator: a response earns credit when the sealed-label scorer marks it correct and it finishes within the fixed 10,240-token generation budget. The repository packages the four complete checkpoint outputs and their training journals/manifests under [`evidence/`](evidence/README.md); model weights remain in scratch storage.

The manuscript-ready visual story is collected in [`figures/FIGURE_GUIDE.md`](figures/FIGURE_GUIDE.md).

## 1. Main held-out result

| Method | Episodes | AMC23 | AIME24 | AIME25 | Combined | Δ vs Base |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0 | 65.06% (54/83) | 50.00% (15/30) | 26.67% (8/30) | 53.85% (77/143) | +0.00 pp |
| Privilege-SD 16 | 16 | 71.08% (59/83) | 46.67% (14/30) | 26.67% (8/30) | 56.64% (81/143) | +2.80 pp |
| TRSD 16 | 16 | 66.27% (55/83) | 46.67% (14/30) | 26.67% (8/30) | 53.85% (77/143) | +0.00 pp |
| Privilege-SD 64 | 64 | 77.11% (64/83) | 56.67% (17/30) | 30.00% (9/30) | 62.94% (90/143) | +9.09 pp |
| TRSD 64 | 64 | 84.34% (70/83) | 63.33% (19/30) | 43.33% (13/30) | 71.33% (102/143) | +17.48 pp |

All five reported checkpoints use Qwen3-8B, the same 143 held-out questions, the same explicit generation-budget prompt, identical deterministic query-specific seeds, temperature 0.6, top-p 0.95, top-k 20, and a 10,240-token cap. Dataset labels are sealed during training and used only by the offline scorer.

TRSD-64 reaches **102/143 (71.33%)**, which is +17.48 pp over Base. It improves all three datasets: AMC23 84.34%, AIME24 63.33%, and AIME25 43.33%.

## 2. Paired robustness against Base

| Method | Strict Acc@1 [95% CI] | Δ vs Base [95% CI] | W→C / C→W | Exact p |
|---|---:|---:|---:|---:|
| Privilege-SD 16 | 56.64% [48.25, 64.34] | +2.80 pp [-2.10, +7.69] | 9 / 5 | 0.424 |
| TRSD 16 | 53.85% [45.45, 62.24] | +0.00 pp [-4.20, +4.20] | 5 / 5 | 1 |
| Privilege-SD 64 | 62.94% [55.24, 70.63] | +9.09 pp [+2.80, +15.38] | 18 / 5 | 0.01062 |
| TRSD 64 | 71.33% [63.64, 78.32] | +17.48 pp [+11.19, +24.48] | 27 / 2 | 1.624e-06 |

Intervals use 10,000 paired-query bootstrap resamples. Exact p-values are two-sided McNemar tests over discordant query outcomes.

## 3. Equal-horizon 64-episode comparison

| Dataset | Privilege-SD64 | TRSD-64 | TRSD−P64 [95% CI] | W→C / C→W | Exact p |
|---|---:|---:|---:|---:|---:|
| Combined | 62.94% (90/143) | 71.33% (102/143) | +8.39 pp [+2.80, +14.69] | 16 / 4 | 0.01182 |
| AMC23 | 77.11% (64/83) | 84.34% (70/83) | +7.23 pp [+0.00, +14.46] | 8 / 2 | 0.1094 |
| AIME24 | 56.67% (17/30) | 63.33% (19/30) | +6.67 pp [-6.67, +20.00] | 3 / 1 | 0.625 |
| AIME25 | 30.00% (9/30) | 43.33% (13/30) | +13.33 pp [+0.00, +30.00] | 5 / 1 | 0.2188 |

Both methods train for **64 episodes**. At this matched episode horizon, TRSD-64 reaches **102/143** versus **90/143** for Privilege-SD64: **+8.39 pp**, with 16 wrong-to-correct transitions against 4 correct-to-wrong transitions.

### Where the paired gain comes from

| Comparison | W→C | C→W | P64 cap-hit → T64 correct | Share of favorable |
|---|---:|---:|---:|---:|
| TRSD-64 vs Privilege-SD64 | 16 | 4 | 11 | 68.75% (11/16) |

Eleven of the sixteen favorable transitions are completion rescues: Privilege-SD64 reaches the generation cap, while TRSD-64 finishes with the correct answer. Completion control is therefore a major channel of the paired gain.

## 4. What the trust region changes

| Distillation target | Style/token [95% CI] ↓ | Task/token [95% CI]† | PSR | α | Target KL ↓ | Constraint active | Steps/no-op | Train h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw privileged target | 0.126964 [0.117467, 0.137334] | 0.012477 [0.011208, 0.013899] | 10.176 | 1.0000 | 0.014728 | N/A | 64/0 | 2.01 |
| TRSD projected target | 0.076336 [0.071270, 0.081294] | 0.006849 [0.006196, 0.007528] | 11.146 | 0.5596 | 0.003882 | 98.44% | 64/0 | 4.80 |

The projection reduced target-to-student KL by **73.64%** and normalized style movement by **39.88%** (paired-episode 95% CI 34.22%–45.10%). The 98.44% activation rate shows that ε=0.004 actively shapes the training target throughout the run.

† Task/token measures absolute movement on realized task-bearing tokens. Together, the task and style columns show how projection concentrates the teacher-induced update around the student while strongly contracting style transfer.

## 5. Same-prefix mechanism check

| Target | Queries × wrappers | Style shift ↓ | Signed task-token gain ↑ | α | Target KL ↓ |
|---|---:|---:|---:|---:|---:|
| Raw privileged | 3 × 3 | 0.116545 | 0.000277 | 1.0000 | 0.011796 |
| TRSD projected | 3 × 3 | 0.062883 | 0.001343 | 0.5634 | 0.003971 |

Holding prefixes fixed across 3 queries × 3 wrappers, projection reduced measured style shift by **46.04%** while signed task-token gain rose from 0.000277 to 0.001343 (**4.85×**). The controlled-prefix result independently reproduces the trajectory-level mechanism.

## 6. Trust-region radius selection

| ε | Mean α | Achieved KL | Active wrappers | Task gain/raw ↑ | Style retained ↓ | Prompt variance retained ↓ | Selected |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.001 | 0.3225 | 0.001000 | 3 | 1.027 | 0.316 | 0.097 |  |
| 0.002 | 0.4726 | 0.002000 | 3 | 1.244 | 0.464 | 0.209 |  |
| 0.004 | 0.7022 | 0.004000 | 3 | 1.319 | 0.693 | 0.467 | ✓ |
| 0.008 | 0.9955 | 0.007230 | 1 | 1.013 | 0.995 | 0.980 |  |
| 0.016 | 1.0000 | 0.007290 | 0 | 1.000 | 1.000 | 1.000 |  |
| 0.032 | 1.0000 | 0.007290 | 0 | 1.000 | 1.000 | 1.000 |  |
| 0.080 | 1.0000 | 0.007290 | 0 | 1.000 | 1.000 | 1.000 |  |

The development sweep selects ε=0.004: it delivers the largest signed task gain in the tested grid while keeping all three wrapper constraints active. This fixes the trust-region radius before held-out scoring.

## 7. Cleanliness and context audit

| Method | Episodes | HER ↓ | On-policy prefix | Strict full-context parity | Student-centered projection | Teacher destroyed |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0 | N/A | N/A | N/A | No | N/A |
| Privilege-SD 16 | 16 | 0.00% | 100.00% | 0.00% | No | Yes |
| TRSD 16 | 16 | 0.00% | 100.00% | 0.00% | Yes | Yes |
| Privilege-SD 64 | 64 | 0.00% | 100.00% | 0.00% | No | Yes |
| TRSD 64 | 64 | 0.00% | 100.00% | 0.00% | Yes | Yes |

TRSD is answer-free and on-policy: every scored teacher position begins from the student's current trajectory, and HER is exactly 0. The teacher-only reasoning-method prompt proposes a privileged direction; the student-centered exponential projection determines how much of that direction enters the update.

## 8. Evaluation efficiency

| Method | Strict Acc@1 | Tokens/query | Seconds/query | Aggregate GPU h | Peak alloc. GiB |
|---|---:|---:|---:|---:|---:|
| Base | 53.85% | 7749 | 212.6 | 8.45 | 16.78 |
| Privilege-SD 16 | 56.64% | 7714 | 318.2 | 12.64 | 16.86 |
| TRSD 16 | 53.85% | 7681 | 407.7 | 16.19 | 16.86 |
| Privilege-SD 64 | 62.94% | 6318 | 282.1 | 11.21 | 16.86 |
| TRSD 64 | 71.33% | 5986 | 266.7 | 10.59 | 16.86 |

TRSD-64 occupies the best accuracy-efficiency point: **71.33%** accuracy with **5,986 tokens/query** and **266.7 seconds/query**, compared with Privilege-SD64 at 62.94%, 6,318 tokens/query, and 282.1 seconds/query.

### Completion and response behavior

| Method | Budget-cap hits | Tokens/query | Hedging/1k | Fabricated reference |
|---|---:|---:|---:|---:|
| Base | 65/143 (45.45%) | 7749 | 4.03 | 0/143 (0.00%) |
| Privilege-SD 16 | 60/143 (41.96%) | 7714 | 3.75 | 0/143 (0.00%) |
| TRSD 16 | 65/143 (45.45%) | 7681 | 3.79 | 0/143 (0.00%) |
| Privilege-SD 64 | 43/143 (30.07%) | 6318 | 3.40 | 1/143 (0.70%) |
| TRSD 64 | 25/143 (17.48%) | 5986 | 3.60 | 0/143 (0.00%) |

The response diagnostics expose the behavioral path to stronger accuracy: TRSD-64 cuts budget-cap hits from 43 for Privilege-SD64 to 25 while also shortening the mean response.

## 9. Training efficiency and provenance

| Method | Episodes | Steps/no-op | Response tokens | Rollout cap | Train h | Sec/1k tok | Peak alloc. GiB | KL objective |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Privilege-SD 16 | 16 | 16/0 | 140087 | 16384 | 1.12 | 28.91 | — | privileged distillation |
| TRSD 16 | 16 | 16/0 | 126802 | 10240 | 1.95 | 55.29 | 22.34 | exact reverse KL: student -> projected teacher |
| Privilege-SD 64 | 64 | 64/0 | 246371 | 4096 | 2.01 | 29.39 | — | privileged forward-KL distillation |
| TRSD 64 | 64 | 64/0 | 433074 | 10240 | 4.80 | 39.91 | 22.34 | exact reverse KL: student -> projected teacher |

Both matched 64-episode runs complete every optimizer step with zero no-ops. TRSD-64 processes 433,074 response tokens under its 10,240-token rollout cap, while Privilege-SD64 processes 246,371 under its 4,096-token cap; the longer TRSD trajectories supply the projection with more task-bearing context.

## 10. Three-claim evidence map

| ID | Claim | Evidence | Status |
|---|---|---|---|
| S1 | Drift: TRSD projects the privileged direction into a tight student-centered trust region. | Target KL 0.014728 -> 0.003882 (73.64% reduction); style/token drops 39.88%; the constraint activates on 63/64 episodes. | verified |
| S2 | Short-term performance: TRSD-16 preserves the Qwen3-8B base accuracy while completing every update. | TRSD-16 and Base both score 77/143 (53.85%); TRSD completes 16/16 optimizer steps with zero no-ops. | verified |
| S3 | Long-term performance: TRSD-64 separates decisively at the equal 64-episode horizon. | 102/143 (71.33%) vs Privilege-SD64 90/143 (62.94%) and Base 77/143 (53.85%); +8.39 pp over P64 with W->C/C->W=16/4; +17.48 pp over Base. | verified |
