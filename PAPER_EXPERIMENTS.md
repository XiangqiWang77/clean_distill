# Clean Self-Distillation empirical study

This is the experiment matrix for the current Qwen3-8B proof of concept.  It
implements the claim boundary in
[`EMPIRICAL_CLAIM_CONTRACT.md`](EMPIRICAL_CLAIM_CONTRACT.md).  Older query-reset
and multi-backbone plans are excluded from this main study.

## Experimental setup

| Role | Data | Count | Label use |
|---|---|---:|---|
| Persistent distillation | DeepMath difficulty 7--10 | 1,000 | sealed from Clean GPU path |
| Development audit | disjoint DeepMath difficulty 7--10 | 200 | label-free coverage/config freeze |
| Held-out test | AMC23 | 83 | offline scoring only |
| Held-out test | AIME24 | 30 | offline scoring only |
| Held-out test | AIME25 | 30 | offline scoring only |
| Combined held-out | all three test sets | 143 | offline scoring only |

The model is the pinned `Qwen/Qwen3-8B` snapshot.  Clean generation receives
only `(query_id, problem, source, problem_sha256)` plus target-disjoint proposed
support.  Answers and reference solutions are physically separated into sealed
files.  The explicitly post-outcome mechanism control is the only GPU stage
allowed to consume a label file, and its entire trajectory is counted as
`HER=1`.

The persistent branches share the same base/LoRA initialization, ordered
DeepMath stream, 1,000 episode/update opportunities, training-token limits,
checkpoints, and paired held-out seeds.  A Clean episode with insufficient
verified support deliberately skips its optimizer step; therefore the report
discloses realized optimizer-step and Clean no-op counts instead of claiming
that the realized update counts are identical.  Training sequences are capped
at 16,384 total tokens.  Evaluation offers up to 32,768 generated tokens within
a 40,960-token context.  Exact token-chunked vocabulary projection preserves
the same mean KL and LoRA gradient while bounding H100 memory; the frozen LM
head permits one assembled hidden-gradient backward through the checkpointed
backbone rather than one decoder recomputation per chunk.

Four paired samples are generated with temperature `0.6`, top-p `0.95`, and
top-k `20`.  Acc@1 is exactly sample index 0; Mean@4 is the mean of all four
binary outcomes.  The report validates those aliases and refuses unpaired
coverage or seeds.

## Primary metrics

### Short-term performance

```text
STG-T = Acc(CSD-T)  - Acc(Base)
STG-S = Acc(CSD-SD) - Acc(Base)
Retention = STG-S / STG-T
```

Retention is `N/A` when STG-T is zero.  Both Acc@1 and Mean@4 report accuracy,
paired wrong-to-correct and correct-to-wrong counts, truncation, and timing.
Timing separates proposal construction, ridge specialization, the student
distillation episode, and end-to-end adaptation; a fast-ridge claim must use
the ridge component rather than hide proposal cost.

### Long-horizon accumulation

Let `A_k` be held-out accuracy after `k` persistent distillation episodes, for
`k in {0,250,500,750,1000}`.

```text
LHG   = A_1000 - A_0
AULC  = normalized trapezoidal integral of (A_k - A_0) over 0..1000
K*    = first observed checkpoint where A_clean > A_privileged, else N/A
```

No interpolation is used for `K*`.  Long horizon means accumulation across
ordered distillation episodes, not the token length of one answer.

### Hindsight-free gain

HER and CP are reconstructed from raw token-position counts:

```text
HER = hindsight_exposed_positions / teacher_positions
CP  = exact_context_positions / compared_positions
HFG = (1 - HER) * CP * (Acc_method - Acc_Base)
```

The expected structural values are:

| Teacher | HER | CP |
|---|---:|---:|
| Pre-decision reasoning-method privilege | 0 | 0 |
| Post-outcome previous-attempt feedback | 1 | 0 |
| Clean temporary teacher | 0 | 1 |

`HER=0, CP=1` establishes cleanliness, not benefit.  Positive benefit requires
positive HFG.

## Study 1: short-term teacher construction and transfer

Methods:

1. Base;
2. pre-decision Privileged SD;
3. CSD-T with Correct+Wrong signed ridge;
4. CSD-SD after the temporary teacher is destroyed.

The held-out set is the complete 143-query AMC/AIME union.  The primary outputs
are Acc@1, Mean@4, STG-T, STG-S, retention, paired flips, truncation, and latency
components.  The study tests whether the clean temporary teacher crosses the
decoding boundary and whether its gain transfers; it does not assume either
answer is positive.

## Study 2: persistent long-horizon accumulation

Two non-reset students traverse the same 1,000-query DeepMath stream:

- **Clean SD:** each episode constructs target-disjoint corrective support,
  fits a temporary signed-ridge teacher, distills on the student's exact
  on-policy prefix, destroys the teacher, and retains the student update;
- **Privileged SD:** only the teacher receives a fixed pre-decision
  reasoning-method instruction; the student sees the original prompt.

Both branches are evaluated on the same held-out samples at
`0/250/500/750/1000`.  Report `A_k`, LHG, AULC, `K*`, teacher-student KL/log-ratio,
gain retention, raw HER/CP counts, and checkpoint/resume identity.

## Study 3: privilege and hindsight mechanism

The complete 143-query held-out set is evaluated with:

1. pre-decision reasoning-method privilege (`HER=0, CP=0`);
2. post-outcome feedback stating only whether the previous Base attempt was
   correct or incorrect (`HER=1, CP=0`);
3. clean parameter-specialized teacher (`HER=0, CP=1`).

The mechanism report contains:

- reward log-ratio separation (RLRS), normalized by trajectory length;
- the versioned RLCSD-style task/style partition and PSR;
- fabricated-reference diagnostics, hedge counts, response length, entropy,
  and truncation;
- clean-teacher decision-boundary crossing and regression rates.

OOD accuracy is optional in this PoC and must remain explicitly missing when no
OOD dataset is run.

## Study 4: matched minimal ablation

| Variant | Correct support | Wrong support | Signed frontier ridge |
|---|---:|---:|---:|
| Correct-only | yes | no | no |
| Correct+Wrong | yes | yes | yes |

Both variants are generated for the entire held-out set.  The causal comparison
uses only runtime-matched ready pairs and discloses the full denominator and
every exclusion reason.  Each retained pair must match on candidate count,
actual support rows/tokens, ridge dimension, frontier identity and Base margin,
model, decoding configuration, and four seeds.

Report exact pre-update support-target NLL, objective-aligned support logit
gain, Acc@1/Mean@4 gain, paired flips, RLRS, DBCR/regression, and
ridge/adaptation latency.  The pre-update NLL is only a scale diagnostic; the
implementation does not mislabel it as adapted NLL.  This is the direct test
of whether right/wrong contrast—not merely more support compute—causes the
change.

## Main table

The final JSON and CSV contain one Acc@1 row and one Mean@4 row for each method:

| Method | Short Acc | STG-T | Student Acc | STG-S | Retention | Final Long Acc | LHG | AULC | HER | CP | HFG | Sec/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | measured | -- | measured | -- | -- | measured | 0 | 0 | -- | -- | -- | 0 |
| Privileged SD | measured | -- | measured | -- | -- | measured | measured | measured | 0 | 0 | 0 | measured |
| CSD-T | measured | measured | -- | -- | -- | -- | -- | -- | 0 | 1 | measured | measured |
| CSD-SD | measured | -- | measured | measured | measured/N/A | measured | measured | measured | 0 | 1 | measured | measured |

No placeholder in this table is replaced until the corresponding real-model
artifacts pass the fail-closed scorer/reporter.

## Claim-evidence map

| Claim | Direct evidence | Supported only if |
|---|---|---|
| Corrective self-proposal | full proposal audit, firewall, provenance, ready/no-op counts | every accepted item has verified right/wrong/frontier and zero target exposure; coverage is disclosed |
| Fast signed-ridge teacher | matched ablation, DBCR, ridge-only seconds, STG-T | structural part passes; `fast` is quantified and `stronger` requires STG-T > 0 |
| Same-context clean transfer | raw HER/CP, on-policy prefix, destruction marker, STG-S/retention | HER=0, CP=1; benefit separately requires positive STG-S/HFG |
| Stable long accumulation | paired learning curves, LHG, AULC, K* | positive metrics are observed; otherwise report failure or `K*=N/A` |

## Formal artifacts

The dependency chain is

```text
prepare -> propose -> merge -> dev audit -> short term
        -> persistent train -> checkpoint evaluation -> mechanism -> report
```

The final artifacts are written beneath the immutable run's scratch directory:

```text
report/main_table.json
report/main_table.csv
report/dev_audit.json
report/aux/mechanism.jsonl
report/aux/ablation.jsonl
report/aux/audit.json
```

Failed, canceled, smoke, Qwen3-4B, or legacy query-reset runs are excluded from
these artifacts and from all paper claims.
