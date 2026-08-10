# Slurm entry points

- `trsd_loop.slurm`: persistent TRSD training on one H100.
- `privileged_loop.slurm`: matched raw pre-decision privileged baseline.
- `trust_region_checkpoint_eval.slurm`: Base, Privileged SD, and TRSD held-out
  evaluation with four H100s maximum.
- `empirical_prep.slurm`: prepare target-free DeepMath queries and sealed
  AMC23/AIME24/AIME25 labels.

All large assets and outputs are written under the configured scratch run root.
