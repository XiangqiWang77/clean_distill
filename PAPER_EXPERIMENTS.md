# Clean Self-Distillation paper experiment matrix

Task 1 (CSD-T) and Task 2 (CSD-SD) are method outputs. They are not the
complete evaluation. The paper suite below runs every result family from real
model logs; the analyzer never inserts hypothetical values.

The corrective-v5 method supplies a verified correct trajectory, a separately
generated wrong trajectory, and a verified first-error/corrective-action
frontier for every accepted candidate, then adapts on every selected candidate.
There is no mock exam, Fit/Check split, or runtime gate.

## Research questions and artifacts

| RQ | Reviewer question | Runs | Paper artifacts |
|---|---|---|---|
| RQ1 | Does CSD improve AMC/AIME under fair compute? | Base, Maj@8, Support ICL, Head SGD, Support LoRA, answer-redacted privileged-CoT control, CSD-T, CSD-SD; 4 backbones x 5 seeds | Table 1, Table 2, Figure 5 |
| RQ2 | Are proposed candidates target-disjoint and useful? | correct/wrong trajectory provenance, first-error verification, literal and 4-gram overlap, verifier validity, fit-signal/target-transfer relation | Table 3, Figures 2--3 |
| RQ3 | Is the temporary teacher both stronger and faster? | CSD ridge vs Head SGD vs Support LoRA; NLL gain, success rate, latency, peak memory, FATE | Tables 4--5, Figure 4 |
| RQ4 | Is improvement hindsight-free and transferable? | answer-conditioned/answer-redacted privileged-CoT control, structural provenance and redaction audits, same-prefix audit, CSD-SD step sweep | Table 4, Figures 6--8 |
| RQ5 | Is the method robust? | ridge lambda, support-token cap, support count, distillation steps, optional MATH-500/OlympiadBench/GSM8K | Tables 6--7, Figure 9 |
| Reporting | Can the run be reproduced and costed? | shared YAML plus query/role timing and token logs | Tables 8--9 |

Figure 1 is the pipeline in the LaTeX paper. Figures 2--9 are generated as
vector PDFs by `analyze_paper_suite.py`.

## Paper-specific metrics

- **HER (Hindsight Exposure Rate):** fraction of teacher construction/scoring
  events whose provenance includes a forbidden target answer, target solution,
  future target token, or target-verifier feedback.
- **CPP (Context--Prefix Parity):** fraction of student/teacher comparisons
  whose serialized causal inputs are identical.
- **SPF (Same-Prefix Fidelity):** token-position-weighted version of CPP.
- **CAR (Clean Advantage Retention):** clean CSD-T accuracy gain divided by the
  answer-redacted privileged-CoT gain, clipped to `[0, 1]` when that gain is
  positive.
- **Privileged-CoT provenance audit:** the private CoT constructor uses the
  target answer and therefore has `HER=1` and `HFS=0`. Before evaluation, the
  final answer and literal-equivalent spellings are removed with a fail-closed
  audit, so the evaluated-context flag must separately report
  `literal_target_answer=false`. Redaction does not relabel the ancestry as
  hindsight-free.
- **HFTG (Hindsight-Free Transfer Gain):** target NLL gain multiplied by the
  no-exposure and context-parity audit factors.
- **FATE (Fast-Adaptation Teacher Efficiency):** HFTG per adaptation second.
- **Accuracy teacher-gain retention:**
  `(Acc_CSD-SD - Acc_Base) / (Acc_CSD-T - Acc_Base)` after the ridge state is
  destroyed. It is reported as N/A whenever the CSD-T accuracy gain is zero or
  negative, rather than averaging per-query NLL ratios.

Standard benchmark metrics remain Acc@1, Mean@N, Pass@N, and Majority@N. They
must not be substituted for one another.

## Immediate v5 method constants

The frozen-backbone closed-form LM-head ridge uses a requested 768-token total
support budget and 96-token per-candidate budget. Correct reasoning rows have
weight `0.25`, correct-answer rows `1`, verified frontier corrective-action
rows `8`, and verified frontier wrong-action rows `8` with an explicit
logit-suppression direction. The frontier is therefore weighted `32x` relative
to ordinary reasoning, and the update norm is capped at `2`.

Task 2 retains the existing identical-on-policy-prefix distillation objective.
No last-layer Jacobian update and no new delta-selective distillation loss are
part of this revision.

## Commands

Preview the complete matrix without allocating a GPU:

```bash
python scripts/clean_self_distill/run_paper_suite.py --dry-run
```

Run every enabled experiment:

```bash
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,main,budget,hindsight,transfer,sensitivity,ood
```

Run by research question:

```bash
# RQ1: main table, iterative baselines, and compute curve
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,main,budget

# RQ2: candidate-set audit plus fit-to-target diagnostics
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,main

# RQ3: ridge vs Head-SGD/Support-LoRA strength-speed scaling
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,main

# RQ4: privileged-CoT/redaction audit and write-back step curve
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,hindsight,transfer

# RQ5: hyperparameter sensitivity and cross-benchmark transfer
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,sensitivity,ood
```

Generate tables and figures only after jobs finish:

```bash
python scripts/clean_self_distill/analyze_paper_suite.py
```

For a two-item infrastructure smoke test:

```bash
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,main,transfer,sensitivity \
  --models qwen3_4b --seeds 0 --max-eval-samples 2
```

To enable Table 7, place verl-compatible parquet files locally and uncomment
their paths under `ood_datasets` in
`configs/clean_self_distill/paper_suite.yaml`.

## Output contract

All jobs write under `outputs/clean_self_distill/paper_suite`. A job is skipped
only when its `summary.json` already exists; use `--force` to rerun it. Each job
also writes a hidden `.job.json` command/status record.

The final analyzer writes:

```text
analysis/
  manifest.json
  tables/table1_main.csv ... table9_cost_breakdown.csv
  figures/fig2_support_hygiene.pdf ... fig9_sensitivity.pdf
```

Missing runs stay missing: their table is empty or their figure is omitted.
