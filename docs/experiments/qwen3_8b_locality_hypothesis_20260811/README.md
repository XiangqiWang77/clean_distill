# Locality-hypothesis empirical chain

Frozen Qwen3-8B Study 1: 72 held-out verified-CoT positive controls, three prompt wrappers, exact full-vocabulary scoring, and no parameter update.

![Useful correction is student-local](figure_a_locality_concentration.png)

![Local targets slow accumulated drift](figure_b_repeated_updates.png)

![Long-horizon rank reversal](figure_c_horizon_effect.png)

## Result

At the pre-specified TRSD radius $\epsilon=0.004$, the projected point retains **11.4%** of the full privileged correct-answer gain while retaining **5.2%** of full-distribution TV movement.  This is the direct test of the locality hypothesis; KL retention (0.196%) is secondary because KL is locally quadratic.

Across eight controlled loops, local targets end at 0.0036x the raw-target deviation.  Wrapper sensitivity is 0.000491% of raw when averaged over loops; raw sensitivity spikes at the first loop rather than increasing monotonically.

In the historical matched-horizon Qwen3-8B evaluation, TRSD-minus-OPSD changes from -2.80 points at 16 episodes to +8.39 points at 64 episodes.  This is downstream long-horizon behavior, not a projection-only causal ablation.

## Figure caption

**Locality hypothesis and its empirical consequences.** Study 1: along the exact exponential path from the deployable Qwen3-8B student to three verified-CoT privileged distributions, endpoint-normalized correct-answer gain is plotted against same-order full-vocabulary movement; the star is the independently aggregated adaptive TRSD target at $\epsilon=0.004$, not a common-$\alpha$ curve point. Study 2: raw privileged targets rapidly leave the student neighborhood in controlled distribution-space loops, while KL-bounded local targets accumulate deviation slowly; the inset is across-wrapper variance of per-loop update KL. This does not assume nuisance-free supervision: it measures that context-specific variation cannot enter arbitrarily strongly in one bounded update and therefore accumulates more slowly. Study 3: on the frozen 143-question strict-Acc@1 evaluation, raw/direct OPSD is stronger early, whereas TRSD is stronger at the 64-episode horizon.

## Scope

- Study 1 is an oracle positive-control mechanism diagnostic because the privileged prompts contain verified derivations and final answers.
- Pairwise wrapper TV is retained in `summary.json` as a secondary diagnostic, but it is nonmonotone along the fixed-alpha path and is not used for the headline locality claim; wrapper robustness is tested by the filtered-loop estimand in Study 2.
- Study 2 is distribution-space simulation with no optimizer step.
- Study 3 reuses completed training/evaluation logs and supports a horizon association, not projection-only causality.
