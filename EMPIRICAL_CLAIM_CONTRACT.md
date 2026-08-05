# Empirical claim contract

This file is the authoritative claim-to-evidence contract for the current
Qwen3-8B proof of concept.  It prevents method properties, hoped-for outcomes,
and measured results from being conflated.  The formal run is allowed to
support, fail to support, or contradict a performance hypothesis; reports must
never insert an expected positive value.

## Scope

- Model: `Qwen/Qwen3-8B`, pinned revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Persistent stream: 1,000 deterministic, unique DeepMath difficulty-7--10
  queries.
- Development split: a disjoint 200-query DeepMath split.  For this first PoC
  it is a label-free coverage and frozen-configuration audit, not a tuning
  sweep; no claim of dev-optimal hyperparameters is permitted.
- Held-out test: AMC23 (83), AIME24 (30), and AIME25 (30), for 143 queries.
- Persistent checkpoints: `0, 250, 500, 750, 1000` episodes.
- Training cap: 16,384 total tokens.  Held-out opportunity: 32,768 generated
  tokens inside a 40,960-token context window.
- Training materializes full-vocabulary logits in exact 128-token chunks, then
  propagates the assembled hidden-state gradient through the checkpointed
  backbone once.  The LM head remains frozen; chunking changes neither the
  token-mean KL objective nor its LoRA gradient.  A real 16,384-token H100
  forward/backward for both Clean and Privileged paths must pass before any
  proposal GPU job is released.
- Evaluation: four paired samples at temperature `0.6`, top-p `0.95`, top-k
  `20`; Acc@1 is sample 0 and Mean@4 is the mean of all four binary scores.

The clean proposal, teacher construction, and distillation processes receive
query-only manifests.  Answers and reference solutions are physically sealed
and may enter only offline scoring or an explicitly declared privileged
control.

## Claim 1: self-proposed corrective specialization set

Allowed claim:

> From a sanitized skill card, the model can construct target-disjoint support
> items containing a verified correct trajectory, an independently generated
> wrong trajectory, and a verified first-error/corrective-action frontier,
> without loading the target answer or reference solution.

Required evidence:

- raw proposal coverage and ready/no-op counts over all 1,343 queries;
- accepted-candidate count and rejection reasons;
- nonempty correct and wrong trajectories plus a verifier-valid frontier for
  every accepted candidate;
- proposer/solver/wrong-generator/verifier provenance and target-disjoint
  lexical audits;
- raw firewall booleans showing that target answers and solutions were not
  loaded.

Coverage is a measured result, not a guaranteed method property.  A low ready
rate weakens this claim and must be reported; no-op queries remain Base.

## Claim 2: contrastive closed-form lazy specialization

Allowed structural claim:

> A single frozen-backbone LM-head ridge solve uses correct support and signed
> wrong-frontier supervision to construct a query-local, reversible temporary
> teacher.

The signed variant must compare the first divergent correct and wrong token at
the same hidden state, target a `correct - wrong = +1.0` logit margin, and use
true weighted least squares.  Ordinary reasoning, answer, frontier-positive,
and frontier-negative rows have weights `0.25`, `1`, `8`, and `8`.  The
temporary update is norm-capped at `2` and destroyed after use.

Required evidence:

- correct-only versus correct+wrong signed ablation matched on query,
  candidate count, actual support rows/tokens, ridge dimension, frontier
  identity/base margin, decoding, and seeds;
- exact pre-update support-target NLL, objective-aligned support logit gain,
  target STG-T, paired wrong-to-correct/correct-to-wrong flips, RLRS,
  DBCR/regression rate, and latency;
- separate proposal, feature-extraction, closed-form-solve/ridge, distillation,
  and end-to-end timing where applicable.

`Closed-form`, `query-local`, and `reversible` are implementation properties.
`Fast` must be stated with measured seconds (or a matched speed baseline), and
`stronger` is supported only when held-out STG-T is positive.  A positive
support-objective logit gain or frontier DBCR alone does not establish an
accuracy gain.  The pre-update NLL is a scale diagnostic and must never be
described as adapted support NLL.

## Claim 3: same-context hindsight-free self-distillation

Allowed structural claim:

> The clean temporary teacher and student are compared on the identical query,
> prompt template, and student-generated causal prefix; the teacher has no
> target answer, future token, post-outcome feedback, or teacher-only text.

Required evidence:

- raw teacher-position, exposed-position, compared-position, and exact-context
  counts;
- clean `HER=0` and `CP=1` derived from those counts, not assigned as labels;
- an on-policy-prefix audit and explicit teacher-destruction marker;
- STG-S and teacher-gain retention for query-local transfer;
- persistent Clean and Privileged students started from the same base state,
  trained on the same ordered episodes with the same update opportunities,
  training-token limits, and evaluation budget, and evaluated at every
  registered checkpoint.  Realized optimizer-step counts and Clean no-op
  counts must be disclosed rather than assumed equal.

The accuracy-aware clean metric is

```text
HFG = (1 - HER) * CP * (Acc_method - Acc_Base).
```

Cleanliness is established when `HER=0, CP=1`; benefit is established only
when HFG is positive.  A clean but zero/negative gain must be reported as such.

## Performance hypotheses

The following are preregistered hypotheses, not claims until the report supplies
their evidence:

1. CSD-T has positive STG-T and more wrong-to-correct than
   correct-to-wrong flips.
2. CSD-SD retains a positive fraction of CSD-T's gain.
3. Persistent Clean SD has positive final LHG and AULC.
4. Clean SD eventually exceeds Privileged SD at an observed checkpoint; if it
   never does, `K*=N/A`.
5. Correct+Wrong signed ridge improves DBCR and target performance relative to
   the matched Correct-only control.

No report may convert these hypotheses into conclusions merely because the
protocol ran successfully.

## Required studies and outputs

1. **Short term:** Base, Privileged SD, CSD-T, and CSD-SD; Acc@1, Mean@4,
   STG-T, STG-S, retention, paired flips, truncation, and timing.
2. **Long horizon:** persistent Clean and Privileged learning curves at
   `0/250/500/750/1000`; final LHG, normalized trapezoidal AULC, and first
   observed crossover `K*` or `N/A`.
3. **Mechanism:** pre-decision privilege (`HER=0, CP=0`), post-outcome
   privilege (`HER=1, CP=0`), and clean teacher (`HER=0, CP=1`); RLRS, a
   versioned PSR partition, behavior diagnostics, and clean-teacher DBCR.
4. **Minimal causal ablation:** Correct-only versus Correct+Wrong signed ridge
   with the matching rules above.  Runtime exclusions and the full held-out
   denominator must be disclosed.

Optional OOD evaluation is outside this one-seed PoC and must be labeled
missing rather than silently treated as completed.

## Excluded evidence

The earlier AMC-only distillation stream, Qwen3-4B query-reset runs, 4,096-token
prefix study, 8,192-token evaluation, answer-conditioned private-CoT control,
and any smoke runs are not evidence for this contract and must not appear in
the current main table.  Their artifacts may remain only as an audit trail.
