# Drift metrics: lexical diagnostic and lexicon-free policy drift

| Method | Training steps | Lexicon-conditioned style drift ↓ | AP-JSD | Entropy retention |
|---|---:|---:|---:|---:|
| Privilege-SD 64 | 64 | 0.1270 [0.1175, 0.1373] | 0.03847 [0.03057, 0.04689] | 0.907 [0.821, 0.993] |
| TRSD 64 | 64 | 0.0763 [0.0713, 0.0813] | 0.04749 [0.03870, 0.05711] | 0.647 [0.564, 0.735] |

- Paired AP-JSD difference (TRSD64 − Privilege-SD64): +0.00903 [+0.00659, +0.01150], with 28/32 query-level differences positive.

- Lexicon-conditioned style drift is the original realized-token absolute log-probability shift on the frozen style-word partition.
- AP-JSD is mean full-vocabulary Jensen-Shannon divergence from the Base policy on identical ordinary-context anchor prefixes, normalized by log(2) to [0, 1].
- Intervals are query/episode bootstrap 95% confidence intervals.
- Joint reading: TRSD transfers fewer predefined style markers while moving the overall policy farther from Base; its effect is selective rather than globally conservative.
