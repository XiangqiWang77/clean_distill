# Clean Self-Distillation scripts

The current paper proof of concept is the persistent Qwen3-8B empirical DAG,
not the older query-reset Task-1/Task-2 wrappers.

## Authoritative entry points

Run code/protocol validation on a high-memory compute node:

```bash
sbatch scripts/clean_self_distill/slurm/empirical_validate.slurm
```

After validation passes and the worktree is clean and committed, submit the
complete restart-safe study:

```bash
RUN_ID=<new-unique-run-id> \
  bash scripts/clean_self_distill/slurm/submit_empirical_poc.sh
```

That launcher executes:

```text
prepare -> propose -> merge -> Dev-200 audit -> short-term evaluation
        -> persistent training -> checkpoint evaluation -> mechanism -> report
```

The formal stages are implemented by:

- `prepare_empirical_data.py`: deterministic 1,000/200/143 split and physical
  label firewall;
- `01_propose.py`: corrective-v5 right/wrong/frontier support generation;
- `04_persistent_train.py`: non-reset Clean and Privileged student branches;
- `05_heldout_eval.py`: paired four-sample label-blind generation and offline
  scoring;
- `06_short_term_eval.py`: query-local CSD-SD and Privileged-SD evaluation;
- `07_mechanism_eval.py`: pre-decision and post-outcome mechanism controls;
- `08_build_empirical_aux.py`: scored mechanism and matched ablation artifacts;
- `09_build_dev_audit.py`: label-free Dev-200 coverage/configuration audit;
- `report_persistent_metrics.py`: fail-closed short/long/HFG/mechanism/ablation
  report.

See [`../../EMPIRICAL_CLAIM_CONTRACT.md`](../../EMPIRICAL_CLAIM_CONTRACT.md),
[`../../CLEAN_SELF_DISTILL.md`](../../CLEAN_SELF_DISTILL.md), and
[`../../PAPER_EXPERIMENTS.md`](../../PAPER_EXPERIMENTS.md) for the claim,
method, and evaluation contracts.

## Excluded legacy entry points

`train_task1_fast_teacher.sh`, `train_task2_clean_distillation.sh`,
`submit_b200_poc.sh`, `run_paper_suite.py`, and `analyze_paper_suite.py` reproduce
older Qwen3-4B query-reset/smoke/multi-backbone plans.  They are retained only
for their historical artifacts and cannot contribute to the current persistent
Qwen3-8B main table.
