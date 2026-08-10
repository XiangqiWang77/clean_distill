# TRSD scripts

- `prepare_empirical_data.py` creates the target-free DeepMath stream and the
  sealed AMC23/AIME24/AIME25 evaluation split.
- `04_persistent_train.py` trains either TRSD or the matched raw
  pre-decision-privileged baseline.
- `05_heldout_eval.py` generates label-blind responses and scores them offline.
- `slurm/trsd_loop.slurm` and `slurm/privileged_loop.slurm` run restartable H100
  training.
- `slurm/trust_region_checkpoint_eval.slurm` evaluates Base, Privileged SD, and
  TRSD checkpoints with at most four H100s.
