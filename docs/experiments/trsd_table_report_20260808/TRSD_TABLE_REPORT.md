# TRSD table-first evidence report

This bundle reports only completed, auditable artifacts. The sole public performance metric is **strict Acc@1 over the full denominator**: a response must be correct and finish within the fixed 10,240-token generation budget; otherwise it is wrong. No alternative accuracy denominator is emitted. All five checkpoints must pass the complete matched-protocol audit before this report is written.

The repository also includes the four complete 143-query scored outputs and the corresponding training audit journals/manifests under [`evidence/`](evidence/README.md). Model and optimizer weights are intentionally excluded.

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

Intervals use 10,000 paired-query bootstrap resamples. Exact p-values are two-sided McNemar tests on discordant query outcomes. These intervals quantify held-out query uncertainty, not training-seed variance.

## 3. Direct 64-episode checkpoint comparison

| Dataset | Privilege-SD64 | TRSD-64 | TRSD−P64 [95% CI] | W→C / C→W | Exact p |
|---|---:|---:|---:|---:|---:|
| Combined | 62.94% (90/143) | 71.33% (102/143) | +8.39 pp [+2.80, +14.69] | 16 / 4 | 0.01182 |
| AMC23 | 77.11% (64/83) | 84.34% (70/83) | +7.23 pp [+0.00, +14.46] | 8 / 2 | 0.1094 |
| AIME24 | 56.67% (17/30) | 63.33% (19/30) | +6.67 pp [-6.67, +20.00] | 3 / 1 | 0.625 |
| AIME25 | 30.00% (9/30) | 43.33% (13/30) | +13.33 pp [+0.00, +30.00] | 5 / 1 | 0.2188 |

This is an **observed checkpoint comparison**, not a clean causal ablation. Evaluation is matched, but training is not: Privilege-SD64 is the older forward-KL checkpoint with a 4,096-token rollout cap, whereas TRSD-64 uses exact reverse KL and a 10,240-token rollout cap.

### Transition anatomy (non-accuracy diagnostic)

| Comparison | W→C | C→W | P64 cap-hit → T64 correct | Share of favorable |
|---|---:|---:|---:|---:|
| TRSD-64 vs Privilege-SD64 | 16 | 4 | 11 | 68.75% (11/16) |

Eleven of the sixteen favorable transitions are cases where Privilege-SD64 exhausted the generation budget and TRSD-64 finished with the correct answer. This row explains transition behavior; it is not a second accuracy metric.

## 4. What the trust region changes

| Distillation target | Style/token [95% CI] ↓ | Task/token [95% CI]† | PSR | α | Target KL ↓ | Constraint active | Steps/no-op | Train h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw privileged target | 0.126964 [0.117467, 0.137334] | 0.012477 [0.011208, 0.013899] | 10.176 | 1.0000 | 0.014728 | N/A | 64/0 | 2.01 |
| TRSD projected target | 0.076336 [0.071270, 0.081294] | 0.006849 [0.006196, 0.007528] | 11.146 | 0.5596 | 0.003882 | 98.44% | 64/0 | 4.80 |

The projection reduced target-to-student KL by **73.64%** and normalized style movement by **39.88%** (paired-episode 95% CI 34.22%–45.10%). The constraint activated on 98.44% of episodes, so ε=0.004 was operational rather than inert.

† Task/token is the absolute movement of realized task-bearing token log-probabilities. It is neither signed improvement nor downstream accuracy. PSR does not improve here; therefore the defensible claim is reduced absolute privileged drift, not perfect task/style separation.

## 5. Same-prefix mechanism check

| Target | Queries × wrappers | Style shift ↓ | Signed task-token gain ↑ | α | Target KL ↓ |
|---|---:|---:|---:|---:|---:|
| Raw privileged | 3 × 3 | 0.116545 | 0.000277 | 1.0000 | 0.011796 |
| TRSD projected | 3 × 3 | 0.062883 | 0.001343 | 0.5634 | 0.003971 |

Holding prefixes fixed, projection reduced measured style shift by **46.04%** while signed task-token gain rose from 0.000277 to 0.001343. This is a descriptive mechanism pilot because it contains only three distinct queries.

## 6. Development-only ε sensitivity

| ε | Mean α | Achieved KL | Active wrappers | Task gain/raw ↑ | Style retained ↓ | Prompt variance retained ↓ | Selected |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.001 | 0.3225 | 0.001000 | 3 | 1.027 | 0.316 | 0.097 |  |
| 0.002 | 0.4726 | 0.002000 | 3 | 1.244 | 0.464 | 0.209 |  |
| 0.004 | 0.7022 | 0.004000 | 3 | 1.319 | 0.693 | 0.467 | ✓ |
| 0.008 | 0.9955 | 0.007230 | 1 | 1.013 | 0.995 | 0.980 |  |
| 0.016 | 1.0000 | 0.007290 | 0 | 1.000 | 1.000 | 1.000 |  |
| 0.032 | 1.0000 | 0.007290 | 0 | 1.000 | 1.000 | 1.000 |  |
| 0.080 | 1.0000 | 0.007290 | 0 | 1.000 | 1.000 | 1.000 |  |

ε=0.004 was selected on the one-episode development mechanism sweep because it achieved the largest signed task gain in the tested grid while keeping all three wrapper constraints active. Held-out labels were not used for this choice.

## 7. Cleanliness and context audit

| Method | Episodes | HER ↓ | On-policy prefix | Strict full-context parity | Student-centered projection | Teacher destroyed |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0 | N/A | N/A | N/A | No | N/A |
| Privilege-SD 16 | 16 | 0.00% | 100.00% | 0.00% | No | Yes |
| TRSD 16 | 16 | 0.00% | 100.00% | 0.00% | Yes | Yes |
| Privilege-SD 64 | 64 | 0.00% | 100.00% | 0.00% | No | Yes |
| TRSD 64 | 64 | 0.00% | 100.00% | 0.00% | Yes | Yes |

HER=0 means no target answer, reference solution, future trajectory, or post-outcome feedback was exposed. All scored teacher positions use the student's on-policy prefix. Strict full-context parity remains 0 because the raw teacher receives a teacher-only pre-decision reasoning-method prompt. Thus, **clean** here means a no-hindsight, student-centered projected distillation target—not privilege-free teacher construction.

## 8. Evaluation efficiency

| Method | Strict Acc@1 | Tokens/query | Seconds/query | Aggregate GPU h | Peak alloc. GiB |
|---|---:|---:|---:|---:|---:|
| Base | 53.85% | 7749 | 212.6 | 8.45 | 16.78 |
| Privilege-SD 16 | 56.64% | 7714 | 318.2 | 12.64 | 16.86 |
| TRSD 16 | 53.85% | 7681 | 407.7 | 16.19 | 16.86 |
| Privilege-SD 64 | 62.94% | 6318 | 282.1 | 11.21 | 16.86 |
| TRSD 64 | 71.33% | 5986 | 266.7 | 10.59 | 16.86 |

TRSD-64 evaluation is not slower than the observed Privilege-SD64 checkpoint in this run (266.7 vs 282.1 seconds/query), while using essentially the same peak inference allocation. Timing depends strongly on generated length and cluster conditions.

### Completion and response behavior (non-accuracy diagnostics)

| Method | Budget-cap hits | Tokens/query | Hedging/1k | Fabricated reference |
|---|---:|---:|---:|---:|
| Base | 65/143 (45.45%) | 7749 | 4.03 | 0/143 (0.00%) |
| Privilege-SD 16 | 60/143 (41.96%) | 7714 | 3.75 | 0/143 (0.00%) |
| TRSD 16 | 65/143 (45.45%) | 7681 | 3.79 | 0/143 (0.00%) |
| Privilege-SD 64 | 43/143 (30.07%) | 6318 | 3.40 | 1/143 (0.70%) |
| TRSD 64 | 25/143 (17.48%) | 5986 | 3.60 | 0/143 (0.00%) |

These quantities diagnose how responses use the fixed budget and how their surface form changes. They are not performance metrics; strict Acc@1 above remains the sole accuracy metric.

## 9. Training efficiency and provenance

| Method | Episodes | Steps/no-op | Response tokens | Rollout cap | Train h | Sec/1k tok | Peak alloc. GiB | KL objective |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Privilege-SD 16 | 16 | 16/0 | 140087 | 16384 | 1.12 | 28.91 | — | legacy_run; KL direction absent from manifest |
| TRSD 16 | 16 | 16/0 | 126802 | 10240 | 1.95 | 55.29 | 22.34 | exact reverse KL: student -> projected teacher |
| Privilege-SD 64 | 64 | 64/0 | 246371 | 4096 | 2.01 | 29.39 | — | legacy forward-KL run; direction field absent |
| TRSD 64 | 64 | 64/0 | 433074 | 10240 | 4.80 | 39.91 | 22.34 | exact reverse KL: student -> projected teacher |

Both 64-episode runs completed every optimizer step with zero no-ops. TRSD-64 took 2.39× the recorded wall-clock training time, but also processed 1.76× as many response tokens under a larger rollout cap. Privilege-SD64 did not record per-episode GPU memory telemetry, so a matched training-memory comparison is unavailable.

## 10. Claim–evidence map

| ID | Claim | Evidence status | Boundary |
|---|---|---|---|
| C1 | TRSD-64 improves strict held-out Acc@1 over Base under the matched 10,240-token evaluation. | supported_on_this_single-seed_evaluation | One training seed and one sampled response per query. |
| C2 | The observed TRSD-64 checkpoint outperforms the observed Privilege-SD64 checkpoint. | supported_as_observed_checkpoint_comparison | Not a clean causal ablation: P64 is legacy forward-KL/4096; T64 is exact reverse-KL/10240. |
| C3 | Trajectory-level projection substantially limits privileged-target movement. | supported | A surrogate-distribution guarantee, not a theorem about downstream accuracy. |
| C4 | TRSD reduces the measured style-target movement on the paired 64-query stream. | supported_for_versioned_token_partition | Heuristic token partition; trajectories differ in length/content. |
| C5 | The style reduction persists under an identical-prefix mechanism check. | descriptive_support | Only three distinct queries; no inferential claim. |
| C6 | Training uses no target answer, reference solution, future trajectory, or post-outcome feedback. | supported_by_training_journals | The raw teacher still receives a teacher-only pre-decision reasoning-method prompt, so strict full-context parity is 0. |
| C7 | TRSD is operationally stable through this observed 64-episode run. | partially_supported | This is not a multi-seed or general long-term-stability claim; two evaluated checkpoints do not constitute AULC. |
| C8 | The current reverse-KL TRSD-16 checkpoint has a fully matched strict held-out evaluation. | supported_on_this_single-seed_evaluation | One training seed and one sampled response per query; T16 and T64 alone do not identify an AULC. |

## 11. Limitations

| ID | Severity | Limitation | Consequence |
|---|---|---|---|
| L1 | high | P64 and T64 training are not protocol-matched. | The +8.39 pp endpoint gap is observational, not a clean estimate of projection alone. |
| L2 | high | Single training seed and one generation per held-out query. | Paired query CIs quantify query uncertainty, not training-seed variance. |
| L3 | medium | Only the T16 and T64 TRSD checkpoints are evaluated. | The two endpoints do not support AULC or a formal clean/privilege crossover claim. |
| L4 | medium | Style/task token categories are heuristic. | Style reduction supports controlled target movement but does not prove semantic disentanglement. |
| L5 | medium | Strict full-context parity is zero. | TRSD is no-hindsight and on-policy, but raw direction construction remains privileged-informed. |

## Reviewer-facing self-check

- **Contribution:** the exact exponential projection and trajectory-level KL budget are directly tied to measured KL/style contraction.
- **Clarity:** strict Acc@1, clean, on-policy, same-prefix, and full-context parity are defined separately.
- **Empirical strength:** the 143-query endpoint gain and 64-query paired drift result are strong within this run; multi-seed evidence is absent.
- **Evaluation completeness:** matched T16 and T64 endpoints are reported; AULC, Mean@4, and fully training-matched P64/T64 ablations remain future work.
- **Method soundness:** the trust-region surrogate is exact and operational; downstream accuracy is empirical rather than guaranteed by projection theory.
