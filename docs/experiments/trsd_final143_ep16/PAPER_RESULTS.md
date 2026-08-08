# TRSD empirical result bundle

## Full 10,240-token held-out table

| Method | Combined | Δ vs Base | AMC23 | AIME24 | AIME25 | Trunc. | Sec/query | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 53.85% | +0.00 pp | 67.47% | 40.00% | 30.00% | 51.05% | 130.9 | 16.78 |
| Privileged-SD | 65.73% | +11.89 pp | 75.90% | 56.67% | 46.67% | 33.57% | 204.5 | 16.86 |
| TRSD | 54.55% | +0.70 pp | 65.06% | 50.00% | 30.00% | 46.15% | 321.9 | 16.86 |

## Paired changes

Privileged-SD produces 20 wrong→correct and 3 correct→wrong transitions relative to Base.

TRSD produces 9 wrong→correct and 8 correct→wrong transitions relative to Base.

The short pilot and mechanism diagnostics remain separate supporting evidence; the TRSD row above is the full 143-query final-checkpoint evaluation.
