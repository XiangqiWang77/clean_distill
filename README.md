# Trust-Region Self-Distillation

Trust-Region Self-Distillation (TRSD) uses a teacher-only, pre-decision
reasoning-method prompt to propose a useful policy direction, projects that
direction into a student-centered trajectory-level KL ball, and distills the
projected distribution on the student's on-policy response prefixes.

For student distribution `p_t` and raw pre-decision teacher `q_t^P`, TRSD uses

```text
q_t^TR ∝ p_t^(1-α) (q_t^P)^α,
```

where the largest trajectory-level `α ∈ [0,1]` satisfying the configured mean
`KL(q_t^TR || p_t) ≤ ε` is found by bisection. If the unprojected teacher is
already inside the KL ball, the exact KKT solution is `α=1`.

## Experiment

- Model: pinned `Qwen/Qwen3-8B`.
- Distillation stream: DeepMath difficulty 7–10, without target answers or
  reference solutions in the training API.
- Held-out evaluation: AMC23, AIME24, and AIME25.
- Reported methods: Base, raw CoT-Privileged SD, and TRSD.
- Primary evidence: held-out Acc@1, learning across checkpoints, hindsight
  exposure, style-token log-prob shift, KL drift, time, and memory.
- Large datasets, model weights, checkpoints, and generations stay in scratch.

Core entry points:

```text
scripts/clean_self_distill/04_persistent_train.py
scripts/clean_self_distill/05_heldout_eval.py
scripts/clean_self_distill/slurm/trsd_loop.slurm
scripts/clean_self_distill/slurm/privileged_loop.slurm
scripts/clean_self_distill/slurm/trust_region_checkpoint_eval.slurm
```

The held-out scorer sees sealed labels only after generation finishes.

## License

MIT
