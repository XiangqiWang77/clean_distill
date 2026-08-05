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

## 2026-08-04: v5-08 canceled; pre-registered horizon correction

The formal array `21283358` and dependent report `21283359` for
`csd-qwen3-4b-full-frontier-v5-08` were canceled and excluded.  The reason was
not a cluster failure: it was a scientific configuration error discovered from
the first completed Task 2 artifact.  Formal evaluation allowed 8192 output
tokens, while same-prefix distillation used three independently regenerated
prefixes capped at 512 tokens.  The first ready row had prefix lengths
`[512, 512, 512]`, but Base/CSD-T/CSD-SD final responses used
3346/3058/4270 tokens.  Summing three independent rollouts gives 1536 sampled
positions, but the maximum contiguous causal training horizon remains 512.
This run therefore cannot support a long-prefix self-distillation claim.

The replacement experiment must report three distinct evidence groups:

1. **Immediate/short-term performance.**  While auxiliary state is active,
   compare Privileged Control and CSD-T with Base using paired Acc@1, target
   answer NLL where defined, Base-wrong to method-correct flips,
   Base-correct to method-wrong regressions, output length, and truncation.
2. **Post-teacher retained and long-prefix performance.**  After destroying
   the ridge teacher, evaluate CSD-SD alone with the same 8192-token budget and
   report Acc@1, target-answer NLL gain, regression, and teacher-gain
   retention.  Separately audit the contiguous distillation horizon and
   early/late same-prefix stability.  The fixed target-length diagnostic uses
   only Base output length: short `<=2048` versus long `>2048` tokens (or Base
   truncation), never CSD correctness.
3. **Hindsight quality.**  Report raw audit denominators plus HER, CPP, HFS,
   and HFAG.  Clean protocol properties and empirical accuracy are separate;
   cleanliness does not by itself guarantee a raw-accuracy gain.

For the replacement scaled run, Task 2 is pre-registered as three independent
on-policy rollouts of up to 4096 tokens each, with final decoding still capped
at 8192.  `H_train` is the maximum contiguous prefix length, never the sum of
rollouts.  Each ready query must have at least one trajectory that either
reaches 4096 tokens or terminates naturally, and traces must preserve exact
same-prefix hashes plus windowed diagnostics for 0--512, 512--1024,
1024--2048, and 2048--4096.  Long-prefix stability may be claimed only if at
least ten held-out AIME ready queries enter the post-2048 window.  Otherwise
the report must label that evidence insufficient rather than silently changing
the subset.

This remains a per-query-reset experiment.  “Long-term” means query-local
post-teacher retention and stability at late reasoning prefixes; it does not
mean cross-query continual learning.  The immediate method scope remains the
user-approved two changes (right+wrong corrective proposals and hard signed
frontier weighting); no Jacobian ridge or delta-selective loss is introduced
in this replacement.

## 2026-08-04: persistent empirical protocol adopted; v6-09 excluded

The user superseded the preceding v6 protocol with a paper-level empirical
design.  Array `21291557` and dependent report `21291558` were canceled at
2026-08-04 14:30 EDT and are excluded from every new main table.  Their logs
remain only as an audit trail.  In particular, the old 3x4096 query-local
prefix experiment must never be reported as long-horizon accumulation.

The authoritative proof-of-concept now uses:

- `Qwen/Qwen3-8B` at pinned revision
  `b968826d9c46dd6066d109eabc6255188de91218`;
- a deterministic 1,000-query distillation stream and disjoint 200-query
  development set from the RLCSD DeepMath difficulty-7--10 parquet;
- AMC23 (83), AIME24 (30), and AIME25 (30) as one 143-query held-out test set;
- physically label-free query manifests for Clean proposal, training, and GPU
  held-out generation; sealed answers/solutions are available only to the
  offline scorer and explicitly privileged baselines;
- a persistent student and optimizer across DeepMath episodes, with scientific
  checkpoints at 0/250/500/750/1000 and restart checkpoints between them;
- separate Clean and Privileged student trajectories starting from the exact
  same base/LoRA initialization, episode order, update budget, and evaluation
  seeds;
- a total training sequence cap of 16,384 tokens and a 32,768-token held-out
  generation opportunity within Qwen3-8B's 40,960-token context;
- four paired stochastic samples (`temperature=0.6`, `top_p=0.95`, `top_k=20`):
  Acc@1 is paired sample 0 and Mean@4 is the mean of all four binary scores;
- at most two concurrent B200s, no GPU smoke test, and restart-safe short
  allocations rather than a 48-hour queue request.

The three primary evidence groups are now:

1. short-term query-specific STG-T/STG-S, retention, paired flips, and latency;
2. persistent `A_k`, `LHG=A_1000-A_0`, normalized trapezoidal AULC, and the
   first *observed* Clean-over-Privilege checkpoint (or N/A, with no interpolation);
3. position-weighted HER and CP with
   `HFG=(1-HER)*CP*(Acc_method-Acc_base)`, always accompanied by raw numerators
   and denominators.

The formal mechanism study additionally requires RLRS, a versioned RLCSD-style
task/style partition for PSR, behavioral diagnostics, and a preregistered
decoding-boundary crossing rate.  The minimal causal ablation is a matched
Correct-only ridge versus Correct+Wrong signed ridge with identical candidate,
actual support-token, model, decoding, and seed budgets.

The storage authorization was updated to at most 20,000,000,000 bytes of new
downloads and at most 100,000,000,000 bytes under the task scratch root.  All
large assets and outputs stay under
`/home/da839/scratch_pi_mg269/da839/clean_distill`; the repository contains only
small source/config/test files.  The pinned DeepMath parquet (587,568,701 bytes,
SHA-256 `611d3030a2a74eaea9514ab732fc33aa6a35d668c7f804c383772939b159f2a0`)
and Qwen3-8B snapshot (16,397,461,266 bytes) were downloaded there.  Heavy data
preparation must run streaming on a high-memory compute node and release it
immediately; it must not load the full parquet on a login/small CPU node.

Implementation freeze for the first empirical PoC:

- the signed ridge objective acts at the exact first divergent corrective/wrong
  token and targets a versioned `correct - wrong = +1.0` logit margin;
- its weights are true weighted least squares, and the Correct-only control has
  the same candidate count, actual support-row budget, and ridge rank while
  never reading wrong/frontier content during fitting;
- Dev-200 is a label-free coverage/configuration-freeze audit in this PoC. No
  dev hyperparameter sweep or claim of dev-optimality is made;
- formal jobs execute a read-only archived commit from task scratch, use only
  RTX Pro 6000 nodes for preparation/scoring and at most two dedicated B200s
  for model work, and are linked in a strictly sequential restart-safe DAG.

High-memory RTX validation job `21333331` completed successfully before the
formal commit: 152 tests and 181 subtests passed. It performed code/protocol
validation only and did not load Qwen3-8B or run a model smoke experiment.

## 2026-08-04: empirical PoC run 01 excluded; deterministic duplicate quarantine

Formal run `csd-qwen3-8b-deepmath-empirical-poc-01` used commit `13e11d7`.
Its preparation job `21334336` failed closed after 92 seconds, before any B200
work, because an exact DeepMath problem appeared with multiple reference
solutions.  All downstream jobs `21334337`--`21334344` were canceled by their
`afterok` dependencies and the run produced no predictions, checkpoints, or
metrics; it is excluded from every empirical table.

An independent high-memory RTX audit (`21336328`) streamed the pinned parquet
and found 31,164 eligible rows, 30,932 unique exact problem texts, and 218
duplicate groups (450 rows).  Every duplicate group had one identical exact
answer; the variation was only in reference-solution text.  Thus the failure
was not an answer-label conflict.  The small audit report is stored under task
scratch at `diagnostics/deepmath_duplicate_audit.json`.

The replacement data schema is
`clean-self-distill-empirical-data-v2-conflict-filtered`.  Selection uses two
streaming passes, groups whitespace/case-insensitive duplicate problems,
quarantines the entire group if any answer-or-reference-solution fingerprint
differs, deterministically canonicalizes fully consistent duplicates, and then
selects by stable exact-problem SHA-256.  Quarantining the 218 solution-variant
groups avoids a post-hoc choice of privileged reference and still leaves far
more than the required 1,200 DeepMath queries.  The manifest records counts
and a digest of quarantined groups.  A replacement formal run may be submitted
only after the full RTX validation suite passes on the new immutable commit.

The user has superseded the earlier one-check-only monitoring instruction:
after resubmission, monitor until the corrected preparation completes and the
formal B200 work is demonstrably running and producing valid artifacts; on any
failure, diagnose, repair, validate, and resubmit rather than stopping at
successful `sbatch` acceptance.

## 2026-08-04: run 02 prep passed; formal accelerator changed to H100

Replacement run `csd-qwen3-8b-deepmath-empirical-poc-02` used commit
`6bd1f2b`.  Full RTX validation job `21336477` passed 154 tests and 181
subtests.  Prep job `21336864` then completed successfully and published the
exact 1,000/200/143 split under the v2 conflict-filtered schema: every pairwise
exact and whitespace/casefold overlap was zero, query files physically excluded
labels, and 227 normalized solution-variant groups (468 rows) were quarantined,
leaving 30,696 eligible groups.

No model task from run 02 started.  The dedicated B200 proposal array had a
dynamic scheduler estimate of 2026-08-05 07:24 EDT; adding `gpu_devel` in place
was rejected by the cluster accounting policy.  A scheduler test estimated
H200 availability only on 2026-08-08, while H100 was immediately schedulable.
The still-pending run-02 jobs `21336865`--`21336872` were therefore canceled and
are excluded; the successful prep artifact remains only as an audit record.

At the user's direction, the next formal run pins every model task to exactly
one typed H100 (`gpu_h100`, `gpu:h100:1`, CUDA capability 9.0 / `sm_90`) and
allows at most four concurrent H100 tasks for faster query/shard throughput.
The persistent Clean and Privileged array has only two branches and therefore
still uses at most two GPUs.  Hardware type, GRES, capability, architecture
flag, and concurrency cap must be recorded in immutable run/runtime metadata;
resume validation must reject accelerator drift.  The scientific datasets,
model revision, methods, seeds, sequence lengths, checkpoint schedule, and
metrics remain unchanged.  Submit a new run ID only after the complete RTX
validation suite passes on this H100-aware commit, then monitor until real
model artifacts are produced.

### Run 03 excluded before model work

Run `csd-qwen3-8b-deepmath-empirical-poc-03` was submitted from H100-aware
commit `9e5e523` after RTX validation `21341154` passed 158 tests and 181
subtests.  Prep `21341373` failed in 17 seconds before data preparation or any
H100 allocation.  The cause was a launcher-helper regression: the new exact
H100 assertion had been placed inside the generic GPU-name helper also used by
RTX prep/merge/dev/report jobs, so the RTX prep name was incorrectly compared
against `H100`.  All dependent jobs `21341374`--`21341381` were automatically
canceled and run 03 produced no model artifact or metric.

The required correction is to keep `csd_assert_gpu(regex)` as a generic
single-device name check for RTX stages and introduce a separate
`csd_assert_model_gpu` for H100 name, capability 9.0, and `sm_90` checks.  Every
model script must call only the latter and every RTX script only the former;
static tests enforce this separation.  Revalidate the corrected immutable
commit and use a new run ID.

### Run 04 successfully entered formal H100 model work

Run `csd-qwen3-8b-deepmath-empirical-poc-04` uses immutable commit `64e1c4c`.
RTX validation `21341565` passed 158 tests and 181 subtests.  Prep `21341882`
completed with the exact 1,000/200/143 counts, zero exact or normalized split
overlap, the physical label firewall, and the same 30,696 eligible DeepMath
groups.  Proposal array `21341883` then allocated four typed H100s and passed
the exact H100 name/capability/architecture gate.

The first formal proposal artifacts were structurally validated rather than
accepted from file existence alone.  At the audit read there were 21 completed
rows: 16 ready and 5 safe no-ops (76.2% early ready rate), 109 accepted
candidates from 322 unique proposals, and 6.625 candidates per ready row on
average.  Every accepted candidate checked had a nonempty independently solved
correct trajectory, independently generated wrong trajectory, verified error
frontier, safe target-disjoint audit, and valid proposal-training binding;
target answer and target solution exposure flags were both false.  The five
no-ops comprised two skill-card safety failures, two 0/8 verified-candidate
rows, and one 3/8 row.  This is an early coverage diagnostic, not a final
performance result: it improves over the old 50% coverage but has not yet
established the desired 80--90% full-run rate.

Transformers emitted a warning that sampling flags may be ignored.  Code and
artifact provenance confirmed this warning comes from the intentionally greedy
`temperature=0` verifier: proposer (`0.8`) and solver (`0.3`) explicitly set
`do_sample=True`.  It is therefore not a protocol deviation.  No short-term,
long-horizon, or HFG accuracy metric exists until the downstream proposal
merge/dev gate/evaluation stages run.

## 2026-08-04: authoritative claim/evidence contract

The user supplied the final method pitch and empirical-study reference.  The
current Qwen3-8B run remains the authoritative experiment because it implements
the required 1,000-query persistent DeepMath stream, disjoint Dev-200 audit,
AMC23/AIME24/AIME25 held-out set, 16,384/32,768 token protocol, four paired
samples, short/long/HFG studies, three-teacher mechanism comparison, and matched
Correct-only versus Correct+Wrong signed-ridge ablation.

The three sellpoints now have explicit evidence boundaries:

1. self-proposal is supported by complete correct/wrong/frontier artifacts,
   target-disjoint provenance, raw firewall flags, and disclosed ready/no-op
   coverage;
2. closed-form signed specialization is a structural property, while `fast`
   requires measured ridge-only time and `stronger` requires positive held-out
   STG-T/paired flips rather than support NLL alone;
3. same-context cleanliness requires raw-count-derived `HER=0, CP=1` and teacher
   destruction, while beneficial transfer separately requires positive
   STG-S/HFG and long-run LHG/AULC.

Expected positive STG, retention, long-run gain, crossover, and signed-ridge
ablation advantage are preregistered hypotheses, not guaranteed conclusions.
If a metric is zero or negative, or crossover never occurs, the report must say
so (`K*=N/A`) rather than altering the protocol or selecting a favorable subset.

`EMPIRICAL_CLAIM_CONTRACT.md`, `README.md`, `CLEAN_SELF_DISTILL.md`,
`PAPER_EXPERIMENTS.md`, and `scripts/clean_self_distill/README.md` replace the
stale Qwen3-4B/query-reset/4,096-token descriptions in the primary documentation.
Legacy wrappers and their artifacts remain explicitly excluded from the current
main table.

## 2026-08-04: run 04 canceled; model-JSON TeX repair corrected

Run 04 was canceled after 47--50 minutes of H100 proposal work, together with
all pending dependents, before merge, training, evaluation, or scoring.  At the
decision read it had 50 proposal rows, 36 ready rows, 14 safe no-ops, and 259
accepted candidates.  The low 72% early ready rate exposed a concrete parser
defect rather than merely low model quality.

Several zero-candidate rows contained three otherwise valid batches of ten
support problems each.  The model emitted ordinary TeX such as `\subseteq`,
`\mathbb`, `\{`, and `\}` with single backslashes inside JSON strings.  The
existing tolerant parser repaired letter commands but not punctuation TeX, so
`json.loads` rejected the entire batch at `\{`/`\}`.  Consequently valid
proposals never reached the solver/verifier and were silently counted as
`0/8` no-ops.

Parser version `literal-math-backslash-v2` preserves every lone mathematical
backslash except the JSON structural escapes for quote, slash, and an already
doubled backslash.  It does not infer fields, alter target-disjoint filters, or
relax solver/frontier verification.  Replaying all 33 proposal rounds that run
04 had recorded as parse failures made 32 parseable; the one remaining response
was genuinely truncated mid-string and correctly remains invalid.  Focused
proposal/reporter regression tests pass.  Run 04 is excluded from all paper
tables and a new immutable run is required after full compute-node validation.

The broader replay covered 1,953 recorded raw-response attempts.  The TeX JSON
repair recovered 107 previously rejected responses, including 32 proposal
batches (288 structurally valid proposed candidates), 74 correct-verifier
objects, and one frontier-verifier object.  A second syntax-only repair accepts
whitespace inside otherwise exact closing tags such as `</ WRONG_STEP>`; it
recovered 26 wrong trajectories while still rejecting missing `STEP_TEXT`
closures or missing final answers.  Together the two repairs raise structural
parsing from 1,802/1,953 (92.27%) to 1,935/1,953 (99.08%).  The remaining 18
responses are missing required fields/tags or are truly truncated and remain
fail-closed.  These are parser-coverage diagnostics, not accepted-candidate or
accuracy results; every recovered artifact still faces all original solver,
verifier, firewall, and target-disjoint gates.

The claim-evidence audit also made the final reporter require the clean
on-policy-position equality and explicit temporary-teacher destruction marker,
and added output-length/truncation plus disjoint latency components.  Ablation
reporting now names the exact pre-update support-target NLL and the
objective-aligned support logit gain; it no longer mislabels the former as an
adapted support NLL.  Ridge specialization wall time covers the complete fit
call, while feature extraction and closed-form solve remain separately
reported.

Before the replacement run, total `da839` scratch usage was 105.199 decimal GB,
above the user's 100 GB cap even though Clean Distill assets were only 17.053
GB.  With no Slurm jobs active, only recoverable caches were removed: a 5.981
GB vLLM/Torch compile cache and two duplicate 4.080 GB Qwen3-1.7B Hugging Face
caches.  Models, datasets, checkpoints, run-04 diagnostics, and current
validation evidence were preserved.  Exact scratch usage afterward was 91.069
GB, leaving roughly 8.9 GB for the new formal artifacts.

## 2026-08-04: run 05 canceled; skill-card method-phrase false rejection

Run `csd-qwen3-8b-deepmath-empirical-poc-05` used immutable commit `735c4f4`
after full RTX validation job `21347077` passed 166 tests and 181 subtests.
Prep `21347391` completed with the required 1,000/200/143 query counts and
zero exact or normalized split overlap.  Four typed H100 proposal workers were
allocated, but the proposal job and every dependent were deliberately canceled
after a deterministic skill-card gate defect was identified.  The final
partial snapshot contains only 10/1,343 proposal rows: seven ready rows, three
safe no-ops, and 54 accepted candidates.  All 54 have complete independently
generated correct and wrong trajectories, verifier-valid error frontiers,
target-disjoint provenance, and valid state/hash bindings.  Merge, Dev,
short-term, persistent, long-term, mechanism, and reporting never ran; run 05
therefore supplies no empirical result and is excluded from all paper tables.

Two of the three partial no-ops were false skill-card rejections.  Every raw
attempt parsed and contained no target literal, entity, symbol, formula,
English number word, or answer cue; its four-gram overlap rate was only
0.0179--0.0339.  Each retained one generic five-token method or constraint
phrase, which mechanically creates two overlapping four-grams.  The old
absolute cap of one rejected all three attempts.  The same two deterministic
rows were the only skill-card failures among 52 run-04 rows, so this was a
systematic Sellpoint-1 coverage defect rather than random generation quality.

The corrected skill-card-only audit permits at most two unique overlapping
four-grams only when they form one genuine contiguous phrase of at most five
tokens.  It also requires overlap rate at most 0.05, at most one overlap
component, zero target literals/entities, zero shared single symbols, zero
symbolic details, zero English number words, and zero direct-answer cues.
Candidate-problem, trajectory, solver, verifier, and frontier gates are
unchanged.  A periodic six-token copy remains rejected by the exact
longest-common-span check even when it has only two unique four-grams, and two
separated four-token copies remain rejected by the component check.  Audit
artifacts now record a named version and all thresholds, and a
parsed-but-unsafe skill card is recorded as `parsed=true, accepted=false` with
its post-sanitize audit rather than being mislabeled as a parser failure.
Replaying run-04 and run-05 raw skill-card attempts recovers both false-failure
rows in each run; truly unsafe longer or separated overlaps remain unsafe.
This replay is a coverage/firewall diagnostic, not an accuracy result.  A
fresh immutable run may be submitted to at most four H100s only after the
corrected commit passes the complete high-memory RTX validation suite.
