# Qwen3-8B ablation and rollout-shift figures

## Figure 1: core projection ablations

`math_ablation_dynamics` contains the requested four panels: `(a) Math:
entropy`, `(b) Math: length`, `(c) Math: reward`, and `(d) Math: accuracy`.
Pale curves are per-rollout observations and thick curves are trailing
8-step means. The dynamics compare only completed variants with the
same 64-query DeepMath stream and 10,240-token rollout cap. Panel (d) uses the
frozen 143-question AMC23/AIME24/AIME25 scorer. The registered full TRSD result
is 71.33%, versus
66.43% for fixed global alpha and
62.94% for independent
token budgets.

## Figure 2: correct/incorrect rollout target-shift distributions

`rollout_target_shift_distribution` is a large six-panel distributional
diagnostic. Panels (a–d, f) show the outcome-conditioned target-KL and
StyleDrift diagnostics, query-matched target shift, and TRSD alpha distribution.
Panel (e) isolates the paper's style-transfer claim in a clean paired slope
plot: 64 thin gray lines show every matched query, two large dark points report
the token-pooled formal-table means, and the gold arrow reports their relative
reduction.

**Panel (e) caption.** TRSD suppresses privileged-context style transfer across
rollouts. Across 64 query-matched DeepMath rollouts, trajectory projection
reduces token-pooled mean StyleDrift from 0.126964 to
0.076336, a 39.9%
reduction. Thin lines connect the trajectory-level raw privileged and projected
targets associated with the same query; targets are evaluated on their recorded
on-policy prefixes. Large points report the token-pooled means used in the
formal table.

Mean Target KL falls from 0.014728 to 0.003882.
Training-rollout correctness is
19/64 versus
43/64, with
24 query-level W→C transitions and
0 C→W transitions.

The raw privileged and TRSD journals use the same query order but each method's
own on-policy prefix. Therefore the query-matched slope panels are descriptive
distributional comparisons; they are not labeled as same-prefix causal
measurements. Exact plotted values are in the accompanying CSV files.
