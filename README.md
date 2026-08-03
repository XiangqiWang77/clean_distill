# Clean Self-Distillation

Reference implementation of **Clean Self-Distillation (CSD)** for query-local mathematical reasoning specialization.

CSD builds a target-disjoint specialization set from a sanitized skill card, fits a temporary closed-form ridge LM-head teacher, and transfers the activated capability through hindsight-free same-prefix on-policy distillation.

## Core pipeline

1. **Skill-card candidate proposal** — proposer, solver, and verifier use isolated contexts; target answers and solutions never enter teacher construction.
2. **Fast ridge specialization (CSD-T)** — frozen backbone, sparse closed-form top-layer update, exact per-query reset.
3. **Clean write-back (CSD-SD)** — rank-8 query-local LoRA trained on the student's exact on-policy prefixes, followed by teacher destruction and student-only evaluation.

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

## Full paper suite

```bash
python scripts/clean_self_distill/run_paper_suite.py --dry-run
python scripts/clean_self_distill/run_paper_suite.py   --tasks supports,main,budget,hindsight,transfer,sensitivity,ood
python scripts/clean_self_distill/analyze_paper_suite.py
```

The suite covers four backbones, five synthesis seeds, Base/Maj@8/Support-ICL/Head-SGD/Support-LoRA baselines, hindsight interventions, teacher-to-student transfer, sensitivity sweeps, and optional OOD evaluation. It produces Tables 1--9 and vector PDF Figures 2--9 from real JSONL logs only.

See [CLEAN_SELF_DISTILL.md](CLEAN_SELF_DISTILL.md) for the method protocol and [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) for the RQ-to-artifact matrix and metric definitions.

## Hindsight and fast-teacher metrics

The evaluator logs HER, CPP, same-prefix fidelity, CHS, CAR, HFTG, FATE, and TAT in addition to Acc@1, Mean@N, Pass@N, and Majority@N.

## Validation

```bash
python -m unittest tests.test_clean_self_distill tests.test_paper_suite_analysis -v
```

## License

MIT
