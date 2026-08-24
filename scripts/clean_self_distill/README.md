# Entry points

- `04_persistent_train.py`: restartable LGSD (`--branch clean`) or OPSD
  (`--branch privileged`) LoRA training.
- `05_heldout_eval.py`: target-free generation followed by separate offline
  scoring.
- `06_trust_region_mechanism.py`: optional projection inspection for one
  trajectory.
- `prepare_empirical_data.py`: optional deterministic parquet-to-JSONL data
  preparation.

See [`INSTALL.md`](../../INSTALL.md) and [`RUN.md`](../../RUN.md) for commands.
