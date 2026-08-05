# Clean Self-Distillation implementation

This document describes the current persistent Qwen3-8B empirical protocol.
The claim/evidence rules are in
[`EMPIRICAL_CLAIM_CONTRACT.md`](EMPIRICAL_CLAIM_CONTRACT.md).  The older
Qwen3-4B query-reset scripts remain in the tree only as excluded legacy code.

## Method overview

```text
query-only target
  -> sanitized skill card
  -> target-disjoint support questions
  -> verified correct trajectory
  -> independently generated wrong trajectory
  -> verified first-error frontier and corrective action
  -> signed closed-form LM-head ridge teacher
  -> teacher/student logits on the same on-policy prefix
  -> one persistent student update
  -> destroy the temporary teacher
```

The method has no fixed Fit/Check mock exam and no target-answer runtime gate.
Every accepted candidate is used.  A query with fewer than the registered
minimum of verified candidates is an explicit no-op and falls back to Base.

## 1. Self-proposed corrective specialization set

The model first converts the target problem into a sanitized skill card.  The
card may describe primitive skills, their composition, and likely failure
modes, but must not retain unique target numerals, entities, expressions,
answers, or reference reasoning.  The proposer sees only that card.

Each accepted support candidate is built with isolated model calls:

1. generate a target-disjoint support problem;
2. solve it and independently verify the correct trajectory;
3. generate a wrong trajectory without showing that generator the verified
   solution;
4. compare the verified correct and wrong trajectories to certify the first
   invalid step and a local corrective action.

An accepted v5 candidate contains `correct_trajectory`, `final_answer`,
`wrong_trajectory`, `wrong_final_answer`, and an `error_frontier` with
`wrong_step_index`, `wrong_step_text`, `error_explanation`,
`corrective_action`, and verifier-valid flags.  Literal, expression, numeric,
and four-gram target-overlap filters plus provenance hashes are recorded.

The formal proposal stage consumes only
`prepared/proposal_queries.jsonl`.  It never opens a sealed answer or solution
file.  The per-row firewall must record
`target_answer_loaded=false` and `target_solution_loaded=false`.

## 2. Signed closed-form lazy specialization

For frozen final-layer support states `H` and sparse desired logit residuals
`R`, the implementation solves

```text
C = (H H^T + lambda I)^-1 R
delta_logits(h) = (h H^T) C
```

This is the selected-vocabulary, low-rank representation of an LM-head update;
the dense hidden-by-vocabulary matrix is not materialized.  Feature extraction
and the linear solve require no optimizer state or backbone backpropagation.

The registered PoC uses these row types:

| Supervision row | Weight | Direction |
|---|---:|---|
| ordinary verified reasoning | 0.25 | boost correct next token |
| verified final answer | 1 | boost answer token |
| frontier corrective action | 8 | boost |
| frontier wrong action | 8 | suppress |

For every signed frontier, the first divergent correct and wrong tokens are
scored at the exact same hidden state.  If the Base margin is

```text
m_B = z_B(correct) - z_B(wrong),
```

the pairwise residual asks for margin `+1.0`.  Weighted least squares applies
the square root of each registered row weight to both `H` and `R`.  The final
temporary update is norm-capped at `2`.

The matched Correct-only control uses the same candidates and actual support
row/rank budget but never reads wrong/frontier content during fitting.  The
frontier is scored only afterward for a matched DBCR diagnostic.

Every adapter records candidate count, support rows/tokens, effective ridge,
rank, norm, base/teacher frontier margins, boundary crossings/regressions,
feature-extraction time, solve time, and total ridge specialization time.

## 3. Same-context persistent distillation

At DeepMath episode `k`, the current persistent student produces an on-policy
response from the original query.  For every response token, Clean Teacher and
Student receive exactly the same tokenized

```text
original query + student-generated prefix.
```

The distributions differ only because the teacher applies the temporary ridge
adapter.  A top-k forward-KL objective updates the student's rank-8 LoRA.  The
LoRA and AdamW state persist into episode `k+1`; they are not reset between
queries.  The query-local ridge adapter is deleted immediately after its one
episode update.

The Clean audit counts teacher positions, exposed positions, compared
positions, exact-context positions, and on-policy positions.  A valid Clean
episode has `HER=0` and `CP=1`.  An insufficient-support no-op skips the
optimizer step rather than fabricating teacher signal.

The Privileged branch starts from the same base/LoRA state and traverses the
same episodes.  Its teacher alone receives a fixed pre-decision
reasoning-method instruction; it receives no answer, solution, future token, or
post-outcome feedback.  It therefore has `HER=0, CP=0`.  This is the long-run
projection-mismatch comparator, not an answer-conditioned oracle.

## Data and sequence protocol

The formal configuration is
[`configs/clean_self_distill/empirical_poc.env`](configs/clean_self_distill/empirical_poc.env).
Data preparation produces:

```text
prepared/distill_queries.jsonl          1,000 query-only DeepMath items
prepared/dev_queries.jsonl                200 query-only DeepMath items
prepared/heldout_queries.jsonl             143 query-only AMC/AIME items
prepared/proposal_queries.jsonl          1,343 query-only items
prepared/*.sealed.jsonl                  labels/solutions for offline use
prepared/manifest.json                   split/firewall/hash audit
```

DeepMath is the pinned difficulty-7--10 parquet.  Exact and normalized duplicate
groups with conflicting answer or solution fingerprints are quarantined before
deterministic selection.  Distill, Dev, and held-out splits must have zero exact
and whitespace/casefold overlap.

Persistent training uses a 16,384-token total sequence cap.  Held-out decoding
offers 32,768 generated tokens within Qwen3-8B's 40,960-token context.  Each
method uses paired seeds for four samples at temperature `0.6`, top-p `0.95`,
and top-k `20`.

## Formal stages

The restart-safe Slurm launcher submits one sequential DAG:

```text
prepare
  -> 36-way proposal array (at most four H100s)
  -> merge and split proposals
  -> label-free Dev-200 coverage/configuration audit
  -> short-term Base/CSD-T/Correct-only/CSD-SD/Privileged-SD evaluation
  -> persistent Clean and Privileged branches
  -> held-out evaluation at 250/500/750/1000
  -> pre-decision and post-outcome mechanism controls
  -> offline scoring and fail-closed report
```

Every model stage is pinned to one typed H100 and an exact CUDA capability
check.  Three-hour allocations publish ordered-prefix or episode-safe state and
requeue before the walltime.  At most four H100 tasks run concurrently.  Large
artifacts stay below the configured task-scratch cap.

Submit from a clean committed checkout:

```bash
RUN_ID=<new-unique-run-id> \
  bash scripts/clean_self_distill/slurm/submit_empirical_poc.sh
```

The launcher archives a read-only copy of that commit into the run directory,
hashes it, writes an immutable run configuration, and records every job ID.

## Metrics

Short-term metrics:

- Acc@1 (paired sample 0) and Mean@4;
- `STG-T = Acc_CSD-T - Acc_Base`;
- `STG-S = Acc_CSD-SD - Acc_Base`;
- gain retention, paired flips, output length, and truncation;
- proposal, ridge, distillation-episode, and end-to-end seconds/query.

Persistent metrics:

- `A_k` at `0/250/500/750/1000`;
- `LHG = A_1000 - A_0`;
- normalized trapezoidal AULC of `A_k-A_0`;
- first observed Clean-over-Privilege checkpoint `K*`, or `N/A`;
- realized optimizer-step and Clean insufficient-support no-op counts.

Cleanliness metric:

```text
HFG = (1 - HER) * CP * (Acc_method - Acc_Base).
```

Mechanism metrics include RLRS, the versioned task/style PSR partition,
behavioral diagnostics, and DBCR/regression.  The post-outcome control receives
only the binary verdict on a prior Base attempt, is explicitly `HER=1, CP=0`,
and never enters Clean training.

The report reconstructs all headline metrics from raw rows and fails on missing
coverage, unpaired seeds, impossible HER/CP counts, incomplete checkpoint
curves, unmatched ablations, or inconsistent runtime identity.  It never fills
missing cells with hypothetical values.

## Interpretation boundary

The protocol guarantees neither positive Acc@1 nor a Clean-over-Privilege
crossover.  It can establish that the support is clean, the ridge objective is
closed-form and signed, and the distillation contexts match.  Claims of a
stronger teacher, retained gain, or stable long-run improvement require the
corresponding measured STG, HFG, LHG, AULC, flips, and ablation results.

## Legacy exclusions

The following remain only to reproduce old, excluded runs:

- `train_task1_fast_teacher.sh` and `train_task2_clean_distillation.sh`;
- `submit_b200_poc.sh` and the Qwen3-4B smoke/full configs;
- the older `paper_suite.yaml` multi-model query-reset matrix;
- the old 4,096-token prefix/horizon reporter.

Those artifacts cannot be mixed with the persistent Qwen3-8B main table.
