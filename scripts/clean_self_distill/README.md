# Entry points

- `04_persistent_train.py`: restartable LGSD (`--branch clean`), OPSD
  (`--branch privileged`), or matched Veto (`--branch veto`) LoRA training.
- `05_heldout_eval.py`: target-free generation followed by separate offline
  scoring.
- `06_trust_region_mechanism.py`: optional projection inspection for one
  trajectory.
- `prepare_empirical_data.py`: optional deterministic parquet-to-JSONL data
  preparation.

See [`INSTALL.md`](../../INSTALL.md), [`RUN.md`](../../RUN.md), and
[`VETO_BASELINE.md`](../../VETO_BASELINE.md) for commands.
