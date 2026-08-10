# Qwen3-8B aligned training dynamics

All trajectories use the same 64 optimizer updates. Each plotted dot is one
post-update observation from the journal; the curves are causal 8-update means
or prefix-cumulative means. The checkpoint-performance figure contains only the
two checkpoints that were actually evaluated (16 and 64).

## Figures

1. `fig1_objective_convergence`: within-method normalized optimization. TRSD
   reduces its objective by 18.7%
   from the first to final 8-step window, versus
   3.3% for
   Privileged-SD.
2. `fig2_late_stage_nll_rebound`: the Privileged-SD student NLL rebounds
   73.8% from its best
   8-step block to steps 57–64; TRSD rebounds
   22.6%.
3. `fig3_vocabulary_style_drift`: the original vocabulary-based diagnostic is
   preserved and shown throughout training. TRSD finishes with
   39.9% less privileged-marker
   transfer.
4. `fig4_phase_difference_heatmap`: eight matched training windows summarize
   where TRSD's advantage appears across objective, student NLL, style drift,
   and task-token error.
5. `fig5_checkpoint_accuracy`: Privileged-SD leads by
   2.80 pp at step 16, but
   TRSD leads by 8.39 pp at step 64.
6. `fig6_compute_indexed_loss`, `fig7_compute_indexed_nll`, and
   `fig8_compute_indexed_style_drift`: the same process is indexed by actual
   cumulative 128-token chunks, reaching
   1,932 microsteps
   for Privileged-SD and
   3,413 for TRSD. Shading is
   the causal 8-update local SEM within each trajectory, not a multi-seed CI.

## Paper-level interpretation

The matched process supports a two-timescale story. Raw privileged distillation
has the short-horizon performance edge, but its fitting signal plateaus and its
student NLL rebounds sharply late in training while privileged-style markers
continue to accumulate. TRSD is deliberately slower at step 16, continues to
improve across the 64-step horizon, and converts that stability into the best
long-horizon strict accuracy.

We operationalize *collapse-like late regression* as the NLL rebound plus
accumulating style drift. Strict accuracy itself does not collapse:
Privileged-SD rises from 56.64%
to 62.94%, while TRSD rises from
53.85% to
71.33% and overtakes it.

## Reproduction

```bash
python scripts/clean_self_distill/20_plot_aligned_training_dynamics.py
```

PNG, PDF, and SVG versions are in `docs/figures/qwen3_8b_training_dynamics/`.
The exact plotted values and phase-level comparisons are stored beside this
README. Older token-chunk visualizations remain untouched, but these aligned
figures are the appropriate evidence for optimizer-step claims.
