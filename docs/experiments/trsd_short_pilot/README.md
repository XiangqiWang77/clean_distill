# TRSD short empirical bundle

This directory contains the completed, H100-recorded mechanism and short-run
diagnostics for trajectory-level Trust-Region Self-Distillation (TRSD). Every
number is recomputed from saved JSONL artifacts; missing values remain `N/A`.

- [Full report](report.md)
- [Answer-free style controls](multiquery_style_controls.png)
- [Actual optimization and resource accounting](training_efficiency_resources.png)
- [Short checkpoint dynamics](checkpoint_long_horizon.png)

Scope: the mechanism panel covers three DeepMath queries with neutral, terse,
and verbose answer-free wrappers; the resource panel covers two executed TRSD
updates. The short six-query checkpoint diagnostic is not the full benchmark
result. The final DeepMath-distilled AMC23/AIME24/AIME25 table is intentionally
reported separately after its 143-query evaluation completes.
