# Clean Self-Distillation experiment memory

This file records durable experiment decisions and failed-run evidence. It is
not a substitute for the immutable run artifacts or the final audited report.

## 2026-08-04: privileged-context definition

The formal privileged control must measure a *reasoning advantage*, not the
trivial benefit of revealing the ground-truth final answer.

- Use a target-specific correct reasoning method or chain-of-thought-style
  advantage text.
- Remove the final answer, boxed-answer spans, direct answer declarations, and
  any equivalent answer-only cue before the evaluated model receives it.
- Store both the pre-redaction and post-redaction hashes, the redaction audit,
  the context source, and the exact text actually supplied to the model.
- If the reasoning text comes from a gold solution or any future correct
  trajectory, it remains a privileged control: `HER=1`, `HFS=0`, even after
  the final answer is redacted. Removing the answer does not make a future
  correct trajectory hindsight-free.
- If the reasoning text is generated only from the target question, sanitized
  skill card, or target-disjoint surrogate/surrounding questions and their
  self-proposed solutions, it is a separate reasoning-advantage baseline:
  `HER=0`. It must not be mislabeled as privileged hindsight.
- The clean CSD proposer, ridge teacher, and distillation path may never consume
  either form of privileged context.

The next formal four-method table should therefore name the control precisely,
for example `Privileged-CoT (answer redacted)`, and report both
`Uses Target Answer = No` and `Uses Future Correct Trajectory = Yes` when the
source is a gold/correct trajectory.

## 2026-08-04: run07 partial-result obstacles

Run:
`/home/da839/scratch_pi_mg269/da839/clean_distill/runs/csd-qwen3-4b-full-cu128-scav-07`

Run07 was canceled and excluded from final reporting after 36/143 queries had
complete proposal, adapter, Task 1, and Task 2 artifacts. The partial paired
set passed the production binding and grading validators and is retained only
for diagnosis.

Observed partial results:

- Base: 21/36 (58.33%).
- Direct-answer privileged control: 28/36 (77.78%, +19.44 pp), but this is the
  superseded control design and must not be used as the next formal control.
- CSD-T: 20/36 (55.56%, -2.78 pp).
- CSD-SD: 21/36 (58.33%, +0.00 pp).
- Clean Task 1/Task 2 contexts had zero forbidden exposure; all 54 active
  distillation comparisons used identical student/teacher prefixes. The
  temporary teacher was destroyed and the student reset was verified on every
  completed Task 2 row.

Primary obstacles:

1. **Candidate yield / no-op rate.** Only 18/36 queries reached the minimum
   verified-candidate count; 18/36 were explicit no-ops. Of 725 unique
   proposals, 137 were accepted (18.90% aggregate yield). The largest reject
   classes were solver parse errors (226), literal overlap (196), and 4-gram
   overlap (144).
2. **Soft gain did not cross the decoding boundary.** On the 18 active ridge
   queries, 17/18 improved target-answer NLL (mean gain 0.08333), but there were
   zero Base-wrong to teacher-correct flips and one correct-to-wrong flip.
3. **Evaluation and prefix truncation.** Base and CSD-T each hit the 8192-token
   generation cap on 18/36 rows; CSD-SD did so on 16/36. All 54 active
   distillation prefixes hit the 512-token prefix cap. The sole CSD-T negative
   accuracy flip was also truncated.
4. **No positive teacher gain to transfer.** On active queries, accuracy moved
   13/18 Base -> 12/18 CSD-T -> 13/18 CSD-SD. Distillation recovered the ridge
   regression but produced no new correct answers; mean distilled target-answer
   NLL gain was -0.00347.
5. **End-to-end speed is proposal-bound.** Median proposal time was 62.12 s,
   median ridge specialization was 0.157 s, and median distillation was
   30.02 s. The closed-form solve is fast, but the whole method cannot be
   described as millisecond end-to-end adaptation.

Required direction before the next formal run:

- Generate and solve target-disjoint surrogate questions with structured,
  parser-stable outputs; retry adaptively until the minimum verified support is
  met or an explicitly audited attempt budget is exhausted.
- Preserve the target-disjoint firewall while separating harmless mathematical
  structure from target-instance leakage; do not relax leakage checks merely
  to improve yield.
- Make the ridge signal strong enough to affect decoded trajectories using an
  AMC-only, answer-blind selection rule; do not select on final correctness.
- Prevent reasoning from consuming the entire output/prefix budget, and record
  completion/EOS evidence rather than treating a cap hit as a finished answer.
- Give same-prefix distillation a measurable teacher distribution gap before
  expecting transfer; keep teacher destruction and exact-prefix invariants.

## 2026-08-04: authoritative three-sellpoint redesign

The next method version replaces answer-level supervision with process-level
corrective supervision. These three components form one required pipeline; a
run that implements only a renamed version of the old answer-token method is
not the requested experiment.

### Sellpoint 1: Self-Proposed Corrective Specialization Set

For each target query, derive a sanitized skill graph that contains:

- primitive skills;
- their composition/order;
- likely reasoning failure modes.

The proposer then creates a target-disjoint mixture of atomic, compositional,
and failure-focused surrogate questions. Every accepted candidate must retain:

1. a verified correct reasoning trajectory;
2. an independently generated model wrong trajectory;
3. the first error frontier in that wrong trajectory;
4. an explanation of why that step is wrong;
5. the correct next action or replacement step at the frontier.

The target answer, target solution, target-verifier feedback, and future target
tokens remain forbidden. Candidate answers and feedback are allowed because
they belong to target-disjoint surrogate questions. Parsing must be structured
and retryable. The desired candidate-ready rate is 80--90%, but the attempt
budget and all no-ops must remain visible rather than silently weakening the
firewall.

### Sellpoint 2: Fast Frontier-Weighted Lazy Specialization

Ridge fitting must use reasoning-trajectory hidden states, not only final-answer
tokens. Token supervision is weighted by decision relevance:

- ordinary language: low weight;
- mathematical operations and intermediate conclusions: medium weight;
- the first error frontier: high weight;
- the corrective next action: high weight;
- final answer: medium weight.

At the frontier, the residual must both raise the verified corrective action
and suppress the observed wrong action. The first implementation remains a
closed-form sparse LM-head update. It must log frontier coverage, positive and
negative residual norms, decision/top-k flips, candidate-only validation, solve
time, and reversibility. If a validated LM-head implementation still improves
NLL without changing decision positions, the next permitted extension is a
query-local low-rank Jacobian ridge over the last 2--4 layers; it must not be
silently substituted without being named and audited.

### Sellpoint 3: Hindsight-Free Delta-Selective Same-Prefix Distillation

The student must learn the temporary teacher's specialization delta rather than
the almost-identical full distribution:

```
delta_logits = teacher_logits - base_logits
```

Only same-prefix positions selected without target labels may contribute, for
example positions where the ridge delta is large, changes the top-k ordering,
or the base student is uncertain. The loss must explicitly align the student's
post-update delta with the detached teacher delta at those positions. It must
log the selection rule, selected/eligible positions, selection coverage,
teacher delta magnitude, top-k flips, and loss components. Exact serialized
student/teacher prefixes, causal scoring, zero forbidden exposure, teacher
destruction before evaluation, and exact per-query student reset remain hard
invariants.

### Required evidence

- Sellpoint 1: high candidate-ready rate; complete correct/wrong/frontier
  artifacts; target-disjoint and provenance audits pass.
- Sellpoint 2: decision-level changes are measured; wrong-to-correct flips
  exceed regressions; CSD-T gains roughly 3--4 percentage points while the
  closed-form specialization remains fast.
- Sellpoint 3: `HER` near 0, `CPP` near 1, teacher destroyed, and CSD-SD retains
  a positive portion of CSD-T gain.

Concise pitch:

> The model proposes its own target-disjoint corrective curriculum, converts
> verified error frontiers into a fast reversible query-specific teacher, and
> transfers only that teacher's specialization delta under identical on-policy
> prefixes without target-answer hindsight.

### Immediate implementation scope

The user subsequently narrowed the next runnable revision to two method changes:

1. Every accepted surrogate candidate must contain both a verified right
   trajectory and a model-produced wrong trajectory, plus a verified first
   error and correction.
2. The closed-form LM-head ridge update must weight that correction frontier
   aggressively: boost the correct next action and explicitly suppress the
   wrong action.

For this revision, keep the existing clean same-prefix student protocol so the
effect of a stronger teacher is identifiable. Do not simultaneously introduce
last-layer Jacobian ridge or a new delta-distillation loss. Those remain future
fallbacks only if the dual-trajectory frontier-weighted teacher still fails.
The answer-redacted Privileged-CoT change is a separate control-definition fix.
