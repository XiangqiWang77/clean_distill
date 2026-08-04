# Clean Self-Distillation

Reference implementation of **Clean Self-Distillation (CSD)** for query-local mathematical reasoning specialization.

CSD builds a target-disjoint corrective specialization set from a sanitized
skill card, fits a temporary closed-form ridge LM-head teacher, and transfers
the activated capability through hindsight-free same-prefix on-policy
distillation.

## Core pipeline

1. **Corrective candidate proposal (v5)** — every accepted surrogate contains a verified correct trajectory, a separately generated wrong trajectory, and an independently verified first-error frontier with a corrective action. Proposer, solver, and verifier use isolated contexts; target answers and solutions never enter clean-teacher construction.
2. **Frontier-weighted ridge specialization (CSD-T)** — a frozen backbone and closed-form LM-head update weight ordinary reasoning tokens `0.25`, answer tokens `1`, the correct frontier action `8`, and explicit suppression of the wrong action `8` (a `32x` frontier-to-reasoning ratio). The update is query-local, norm-capped at `2`, and exactly reset.
3. **Clean write-back (CSD-SD)** — the existing rank-8 query-local LoRA is trained on the student's exact on-policy prefixes, followed by teacher destruction and student-only evaluation. This revision does not add a Jacobian update or a delta-selective distillation loss.

There is no mock exam, Fit/Check split, or runtime gate. Every selected verified candidate is used for specialization.

## Install

```bash
pip install -r requirements.txt
python scripts/download_data.py --dataset amc23+aime24+aime25 --split val
```

## Paper tasks

```bash
bash scripts/clean_self_distill/train_task1_fast_teacher.sh
bash scripts/clean_self_distill/train_task2_clean_distillation.sh
```

For the pinned, restart-safe multi-B200 PoC (Qwen3-4B, 8192-token evaluation,
AMC23+AIME24+AIME25), use the dependency-chain launcher documented in
[`scripts/clean_self_distill/slurm/README.md`](scripts/clean_self_distill/slurm/README.md).

## Full paper suite

```bash
python scripts/clean_self_distill/run_paper_suite.py --dry-run
python scripts/clean_self_distill/run_paper_suite.py   --tasks supports,main,budget,hindsight,transfer,sensitivity,ood
python scripts/clean_self_distill/analyze_paper_suite.py
```

The suite covers four backbones, five synthesis seeds, Base/Maj@8/Support-ICL/Head-SGD/Support-LoRA baselines, hindsight interventions, teacher-to-student transfer, sensitivity sweeps, and optional OOD evaluation. It produces Tables 1--9 and vector PDF Figures 2--9 from real JSONL logs only.

See [CLEAN_SELF_DISTILL.md](CLEAN_SELF_DISTILL.md) for the method protocol and [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) for the RQ-to-artifact matrix and metric definitions.

## Hindsight and fast-teacher audit

The evaluator logs HER, CPP, HFS, HFAG, same-prefix fidelity, CAR, HFTG, FATE,
and accuracy-based CSD-SD teacher-gain retention in addition to Acc@1, Mean@N,
Pass@N, and Majority@N. The evaluation-only privileged control is an
answer-conditioned correct-CoT advantage whose final answer and literal
equivalents are removed before it reaches the evaluated model. Its ancestry is
still privileged (`HER=1`, `HFS=0`) even though the evaluated context is
certified not to contain a literal target answer.

## Validation

```bash
python -m unittest tests.test_clean_self_distill tests.test_csd_invariants \
  tests.test_poc_report tests.test_paper_suite_analysis -v
```

## License

MIT
