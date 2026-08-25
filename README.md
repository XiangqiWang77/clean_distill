# Locality-Guided Self-Distillation

This repository is a compact implementation of Locality-Guided Self-Distillation
(LGSD), its unprojected privileged-teacher counterpart (OPSD), and a matched
implementation of the published Veto adaptive-target baseline. It contains
only the implementation and usage instructions.

For the pre-update student distribution `p_t` and raw privileged proposal
`q_t^P`, LGSD constructs the geometric path

```text
q_t^C(alpha) proportional to p_t^(1-alpha) (q_t^P)^alpha
```

and selects the largest trajectory-level `alpha` in `[0, 1]` satisfying
`mean_t KL(q_t^C(alpha) || p_t) <= epsilon`. The projected distribution is
detached, then fitted with the forward distillation objective
`mean_t KL(q_t^C || pi_theta)`. Thus the projected target, rather than the
current student, weights the token-level cross-entropy. OPSD uses `alpha=1`
and fits the raw privileged target with the same forward-KL direction.

The legacy reverse objective `KL(pi_theta || q_t^C)` remains available only as
`--student-kl-direction reverse` for reproduction. It is not the canonical LGSD
objective because, on the geometric path, it has an exact adaptive-anchoring
rewrite.

Veto constructs `Q_beta proportional to P_T P_S^beta`, linearly schedules
`beta` from `0.8` toward zero, detaches that target, and applies the same
forward-KL fitting direction. It is exposed as `--branch veto` with a distinct
checkpoint identity. See [VETO_BASELINE.md](VETO_BASELINE.md) for the formula,
paper provenance, fair-comparison boundary, and command.

## Start here

- [Installation](INSTALL.md)
- [Dependency stack](DEPENDENCIES.md)
- [Training, inference, and evaluation](RUN.md)
- [Veto baseline and provenance](VETO_BASELINE.md)

## Qwen3-8B checkpoints

The code in this revision changes the optimization objective, so it requires a
new training run. No reverse-KL adapter is relabeled as forward-KL LGSD.

Release `qwen3-8b-checkpoints-v1` contains the earlier reverse-KL TRSD/OPSD
adapters and remains available only for provenance and exact reproduction.
Use `python scripts/download_checkpoints.py --method legacy-trsd` if that is
specifically what you need. New checkpoints produced by this code record both
`projection_kl_direction` and `distillation_kl_direction` in their manifest.

For a beginner-oriented walk-through of the implementation and the reason old
weights cannot be reused, see [LGSD_IMPLEMENTATION.md](LGSD_IMPLEMENTATION.md).

## Code layout

```text
src/clean_self_distill/                 core generation and distillation
src/clean_self_distill/veto.py          Veto target and beta schedule
scripts/clean_self_distill/04_persistent_train.py
scripts/clean_self_distill/05_heldout_eval.py
scripts/clean_self_distill/prepare_empirical_data.py
scripts/download_checkpoints.py
tests/                                  lightweight unit tests
```

## License

MIT. The Qwen3 base model and published adapters retain their own licenses.
