# Routed-distillation and outcome-RL baselines

This directory contains the matched Qwen3-8B, Qwen3-1.7B, and GPT-OSS-20B
implementations. SRPO follows the sample-routing, EMA self-teacher, entropy-
weighted SDPO, and GRPO definitions in
[SRPO](https://arxiv.org/abs/2604.02288). Its teacher-selected top-k plus tail
bucket follows the released [SDPO implementation](https://github.com/lasgroup/SDPO).
OPSD follows the
released generalized-JSD objective and fixed-teacher protocol in
[OPSD](https://arxiv.org/abs/2601.18734). Outcome GRPO follows equations
(3)--(4) in [DeepSeekMath](https://arxiv.org/abs/2402.03300).

Routed methods use verifiable DeepMath answers only during training. Evaluation
continues to use the repository's sealed-label AMC23/AIME24/AIME25 protocol.
Every run records prompt, rollout, and generated-token counts because grouped
baselines consume more trajectories per prompt than the existing one-rollout
TRSD study.

## Projection and robustness studies

Archived projection studies remain reproducible through their original
launchers, but they are not part of the current SRPO paper comparison. SRPO is
kept unprojected so that sample routing, rather than a changed teacher target,
is the controlled baseline distinction from TRSD.

The Table 15 source study uses current-policy contextual teachers and exact
reverse KL under the same projection. It covers verified solutions,
answer-free methods, verifier critiques, execution feedback, equivalent prompt
wrappers, style-only instructions, and deterministically permuted context.
Wrapper sensitivity is measured on fixed ordinary-context prefixes with three
semantically equivalent outer wrappers.

Table 16 reports synchronized wall time for rollout generation, teacher
scoring, target construction, and optimization, plus peak allocated GPU
memory. `submit_projection_tables.sh` serializes the stages and caps the study
at two concurrent H100s.
