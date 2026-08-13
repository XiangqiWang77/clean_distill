# Qwen3-8B Math training dynamics

Status: **complete**.

The three panels are `(a) Entropy`, `(b) Response length`, and
`(c) Verifier reward`. Pale curves are per-rollout values; thick curves are
trailing 8-episode means. The 64 training observations use the
exact matched DeepMath query order. The figure uses an 18:5 canvas, a
black/yellow palette, and a shared legend below the panels.

The entropy panel reports realized-token surprisal
`-mean(log p_student(y_t | prefix))` as an on-policy entropy proxy. The
journals do not store exact full-vocabulary categorical entropy, so the figure
does not claim that stronger quantity. Reward is the frozen boxed-answer
verifier's binary score on each training rollout.

See `summary.json` for exact definitions and `math_training_dynamics.csv` for
every plotted value.
