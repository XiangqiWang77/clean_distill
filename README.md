# Clean Self-Distillation

Clean Self-Distillation (CSD) constructs a temporary query-specific teacher by
parameter specialization rather than teacher-only target context.  The current
paper proof of concept uses Qwen3-8B, a persistent 1,000-episode DeepMath
difficulty-7--10 stream, and held-out AMC23/AIME24/AIME25 evaluation.

The authoritative claim/evidence boundary is
[EMPIRICAL_CLAIM_CONTRACT.md](EMPIRICAL_CLAIM_CONTRACT.md).  Method details are
in [CLEAN_SELF_DISTILL.md](CLEAN_SELF_DISTILL.md), and the exact study matrix is
in [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md).

## Current pipeline

1. **Self-proposed corrective set.**  A proposer sees only a sanitized skill
   card.  Every accepted target-disjoint support item contains a verified
   correct trajectory, an independently generated wrong trajectory, and a
   verified first-error/corrective-action frontier.
2. **Signed lazy specialization.**  A frozen-backbone, closed-form LM-head ridge
   solve boosts the corrective action and suppresses the wrong action at their
   first divergent token.  It builds a temporary, query-local teacher and is
   destroyed after use.
3. **Same-context distillation.**  Teacher and student are compared on the exact
   same query, prompt, and student-generated prefix.  The persistent student is
   updated without target-answer hindsight.

## Formal proof-of-concept

The committed configuration is
[`configs/clean_self_distill/empirical_poc.env`](configs/clean_self_distill/empirical_poc.env):

- pinned `Qwen/Qwen3-8B` revision;
- 1,000 DeepMath distillation queries and a disjoint Dev-200 audit;
- AMC23 (83), AIME24 (30), and AIME25 (30) held out for scoring;
- persistent checkpoints at `0,250,500,750,1000`;
- 16,384-token training cap and 32,768-token held-out generation opportunity;
- exact 128-token vocabulary chunks, a frozen LM head, and one checkpointed
  backbone backward, guarded by a real full-16k H100 validation stage;
- paired Acc@1/sample-0 and Mean@4 evaluation;
- Base, Privileged SD, CSD-T, CSD-SD, and the matched Correct-only ridge
  control;
- short-term, long-horizon, HER/CP/HFG, mechanism, and signed-ridge ablation
  reports;
- at most four typed H100 tasks at once, with restart-safe three-hour slices.

Large datasets, model weights, checkpoints, and responses belong under the
configured task scratch root, never in this repository.

Submit the complete dependency chain only from a clean committed tree:

```bash
RUN_ID=<new-unique-run-id> \
  bash scripts/clean_self_distill/slurm/submit_empirical_poc.sh
```

The launcher archives and hashes the exact commit into scratch.  Every stage is
fail-closed on model revision, accelerator type, split identity, label
firewall, resume identity, and expected output coverage.

## Reporting discipline

Successful execution is not evidence of positive accuracy.  Structural claims
such as target exclusion, closed-form fitting, update destruction, `HER=0`, and
`CP=1` are audited separately from performance hypotheses such as positive
STG-T/STG-S, long-horizon gain, or Clean-over-Privilege crossover.  Missing or
negative evidence remains missing or negative in the report.

## Legacy code

The older Qwen3-4B query-reset wrappers, 4,096-token prefix analysis, legacy
paper-suite YAML, and smoke configurations remain only for reproducibility of
excluded runs.  They are not part of the current main table and must not be
used to support the persistent Qwen3-8B claims.

## Validation

The protocol test suite is under [`tests/`](tests).  Cluster validation uses
[`scripts/clean_self_distill/slurm/empirical_validate.slurm`](scripts/clean_self_distill/slurm/empirical_validate.slurm)
on a compute node; the login session should not load the model or dataset.

## License

MIT
