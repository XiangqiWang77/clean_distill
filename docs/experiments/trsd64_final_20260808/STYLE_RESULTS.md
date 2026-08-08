# Matched 64-episode style report

The two journals are paired exactly by episode, query ID, stream index, and problem hash.

| Target | Style/token [95% CI] | Task/token [95% CI] | PSR [95% CI] | Alpha | Target KL | Steps/no-op | Train h |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_privileged | 0.126964 [0.117467, 0.137334] | 0.012477 [0.011208, 0.013899] | 10.1757 [9.0461, 11.3916] | 1.000000 | 0.014728 | 64/0 | 2.011 |
| trsd_projected | 0.076336 [0.071270, 0.081294] | 0.006849 [0.006196, 0.007528] | 11.1460 [9.7165, 12.7275] | 0.559570 | 0.003882 | 64/0 | 4.801 |

## Paired effects (TRSD minus raw privilege)

| Metric | Delta [95% CI] | Relative change | Relative reduction [95% CI] |
|---|---:|---:|---:|
| style_error_per_token | -0.050629 [-0.060851, -0.040923] | -39.88% | +39.88% [+34.22, +45.10] |
| task_error_per_token | -0.005629 [-0.006916, -0.004464] | -45.11% | +45.11% [+38.86, +51.11] |
| psr | +0.970306 [-0.296875, +2.379896] | +9.54% | -9.54% [-24.35, +2.76] |

Target-to-student KL was reduced by 73.64% (0.014728 → 0.003882); TRSD's trajectory-level constraint activated on 98.44% of episodes. Recorded training time was 2.39× the historical privileged run, noting the runs generated different total token counts.

## Same-prefix mechanism pilot

Across 3 queries × neutral/terse/verbose wrappers, style shift changed from 0.116545 to 0.062883 (46.04% reduction). Task-token log-probability gain changed from 0.000277 to 0.001343; projected alpha was 0.563368 and projected KL was 0.003971.

The pilot is descriptive because it contains three distinct queries. The 64-episode confidence intervals use a paired episode/query bootstrap.
