# Clean Self-Distillation paper experiment matrix

Task 1 (CSD-T) and Task 2 (CSD-SD) are method outputs. They are not the
complete evaluation. The paper suite below runs every result family from real
model logs; the analyzer never inserts hypothetical values.

The corrected method proposes and verifies a specialization set, then adapts
on every selected candidate. There is no mock exam, Fit/Check split, or runtime
gate.

## Research questions and artifacts

| RQ | Reviewer question | Runs | Paper artifacts |
|---|---|---|---|
| RQ1 | Does CSD improve AMC/AIME under fair compute? | Base, Maj@8, Support ICL, Head SGD, Support LoRA, privileged hint control, CSD-T, CSD-SD; 4 backbones x 5 seeds | Table 1, Table 2, Figure 5 |
| RQ2 | Are proposed candidates target-disjoint and useful? | proposer/solver/verifier audit, literal and 4-gram overlap, verifier validity, fit-signal/target-transfer relation | Table 3, Figures 2--3 |
| RQ3 | Is the temporary teacher both stronger and faster? | CSD ridge vs Head SGD vs Support LoRA; NLL gain, success rate, latency, peak memory, FATE | Tables 4--5, Figure 4 |
| RQ4 | Is improvement hindsight-free and transferable? | correct-vs-wrong answer-hint intervention, structural provenance audit, same-prefix audit, CSD-SD step sweep | Table 4, Figures 6--8 |
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
- **CHS (Counterfactual Hindsight Sensitivity):** JSD between distributions
  under format-matched correct and wrong answer hints on fixed student response
  tokens. CSD consumes neither hint, so its CHS is structurally zero; the
  privileged control is measured by an actual intervention.
- **CAR (Clean Advantage Retention):** clean CSD-T accuracy gain divided by the
  privileged-hint gain, clipped to `[0, 1]` when the privileged gain is
  positive.
- **HFTG (Hindsight-Free Transfer Gain):** target NLL gain multiplied by the
  no-exposure and context-parity audit factors.
- **FATE (Fast-Adaptation Teacher Efficiency):** HFTG per adaptation second.
- **TAT (Teacher-Advantage Transfer):** fraction of positive clean-teacher
  target-NLL advantage retained by the distilled student after the ridge state
  is destroyed.

Standard benchmark metrics remain Acc@1, Mean@N, Pass@N, and Majority@N. They
must not be substituted for one another.

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

# RQ4: hindsight intervention and write-back step curve
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
