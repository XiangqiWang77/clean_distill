# Same-prefix prompt sensitivity

This figure asks whether changing only the wording of an answer-free
privileged prompt changes the distillation target. Each of three DeepMath
student trajectories is scored under the fixed neutral, terse, and verbose
wrappers. At every fixed token position, the plotted coordinates are

\[
x_t=\operatorname{Var}_{m}\!\left[\log q^P_{t,m}(y_t)-\log p_t(y_t)\right],
\qquad
y_t=\operatorname{Var}_{m}\!\left[\log q^C_{t,m}(y_t)-\log p_t(y_t)\right].
\]

The identity line is unchanged prompt sensitivity. Points below it have lower
across-wrapper variance after projection; points above it have higher variance.
The analysis contains 24,140 paired token positions from three query
replicates. Tokens are repeated observations within trajectories, not
independent query replicates.

Main descriptive result: mean variance falls from 0.000710778 to 0.000215645
(69.7%). Query-level reductions are 47.7%, 75.0%, and 85.4%.

Files:

- `prompt_sensitivity_multiquery.png`, `.pdf`, `.svg`: paper figure.
- `prompt_sensitivity_multiquery.csv`: all paired token coordinates and the
  three raw/projected shifts.
- `prompt_sensitivity_multiquery_summary.csv`: aggregate and query-level
  summaries.
- `prompt_sensitivity_multiquery_example.csv`: the real “Therefore” example
  shown in panel (a).

Regenerate on a CPU node with
`scripts/clean_self_distill/slurm/plot_prompt_sensitivity_cpu.slurm`.
