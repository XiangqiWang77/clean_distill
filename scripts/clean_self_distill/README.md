# Paper run scripts

Run from the repository root after `pip install -r requirements.txt` and
downloading the evaluation parquet.

```bash
python scripts/download_data.py --dataset amc23+aime24+aime25 --split val
```

## Task 1: CSD-T / fast temporary teacher

```bash
bash scripts/clean_self_distill/train_task1_fast_teacher.sh
```

This performs candidate proposal, ridge specialization, and direct temporary
teacher evaluation. It does not train persistent model parameters.

## Task 2: CSD-SD / clean write-back

```bash
bash scripts/clean_self_distill/train_task2_clean_distillation.sh
```

For each benchmark item this trains a fresh rank-8 LoRA for three AdamW steps
using same-prefix KL, evaluates it after deleting the ridge teacher, and resets
the LoRA before the next item.

Both wrappers accept `MODEL_PATH`, `EVAL_DATA`, `RUN_SEED`, `OUTPUT_ROOT`,
`PROPOSALS`, `NUM_CANDIDATES`, `MAX_EVAL_SAMPLES`, and `FORCE_PROPOSE`. Task 1
also accepts `ADAPTER_DIR`, `FORCE_SPECIALIZE`, and `PRIVILEGED_CONTROL`
(enabled by default as an evaluation-only hindsight upper bound).

Example:

```bash
MODEL_PATH=Qwen/Qwen3-8B RUN_SEED=3 MAX_EVAL_SAMPLES=2 \
bash scripts/clean_self_distill/train_task2_clean_distillation.sh
```

See [`CLEAN_SELF_DISTILL.md`](../../CLEAN_SELF_DISTILL.md) for the exact
protocol, metrics, low-level commands, and five-seed loop.

## Full paper figures and tables

```bash
python scripts/clean_self_distill/run_paper_suite.py --dry-run
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,main,budget,hindsight,transfer,sensitivity,ood
python scripts/clean_self_distill/analyze_paper_suite.py
```

The suite YAML is `configs/clean_self_distill/paper_suite.yaml`. The analyzer
creates Tables 1--9 as CSV and Figures 2--9 as vector PDFs from real logs only.
See [`PAPER_EXPERIMENTS.md`](../../PAPER_EXPERIMENTS.md) for the five RQs, all
baseline jobs, paper-specific hindsight/teacher metrics, and artifact mapping.
