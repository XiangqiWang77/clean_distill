# Qwen3-8B Math training dynamics

Status: **preview (checkpoints 32/48 pending)**.

The four panels are `(a) Math: entropy`, `(b) Math: length`,
`(c) Math: reward`, and `(d) Math: accuracy`. Pale curves in panels (a)--(c)
are per-rollout values; thick curves are trailing 8-step means.
The 64 training observations use the exact matched DeepMath query order.

The entropy panel reports realized-token surprisal
`-mean(log p_student(y_t | prefix))` as an on-policy entropy proxy. The
journals do not store exact full-vocabulary categorical entropy, so the figure
does not claim that stronger quantity. Reward is the frozen boxed-answer
verifier's binary score on each training rollout. Accuracy is Strict Acc@1 on
the common 143-question AMC23/AIME24/AIME25 scorer; a 10,240-token cap hit is
incorrect.

See `summary.json` for input hashes and exact definitions, and the CSV files
for every plotted value.
