# TRSD final held-out evaluation

**Table 1. Main results at the completed DeepMath checkpoints.** AMC23,
AIME24, and AIME25 are Acc@1 (%) with one deterministic 10,240-token rollout
per problem; Macro is their unweighted mean and Combined is accuracy over all
143 problems. Evaluation is unprivileged. TRSD uses the development-fixed
trajectory KL budget `epsilon=0.004`. Results that were not run are not imputed.

| Backbone | Method | DeepMath rounds | AMC23 | AIME24 | AIME25 | Macro | Combined Acc@1 | LHG | Target KL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | Base | 0 | 67.47 | 40.00 | 30.00 | 45.82 | 53.85 | -- | -- |
| Qwen3-8B | Privileged-SD | 64 | 75.90 | 56.67 | 46.67 | 59.75 | 65.73 | +11.89 pp | unconstrained |
| Qwen3-8B | TRSD (`epsilon=0.004`) | 16 | 65.06 | 50.00 | 30.00 | 48.35 | 54.55 | +0.70 pp | 0.00390 |

`LHG` is final Combined Acc@1 minus the Base Combined Acc@1. The privileged
and TRSD rows are available final checkpoints with different episode counts;
they are not presented as a round-matched comparison. Mean@4, 1,000-round
results, Qwen3-1.7B/Qwen3-4B, and GRPO/OPSD/SDPO/SRPO/RLSD/RLCSD were not
measured by this run.

## Paired behavior and operating profile

Relative to Base, Privileged-SD records 20 wrong-to-correct and 3
correct-to-wrong transitions. TRSD records 9 wrong-to-correct and 8
correct-to-wrong transitions, for a net gain of one correct response.

| Method | Truncation | Mean tokens | Seconds/query | Peak allocated |
|---|---:|---:|---:|---:|
| Base | 51.05% | 8,266.7 | 130.9 | 16.78 GiB |
| Privileged-SD | 33.57% | 7,069.3 | 204.5 | 16.86 GiB |
| TRSD | 46.15% | 7,934.9 | 321.9 | 16.86 GiB |

The complete machine-readable table, paired transitions, summary JSON, and
PNG/PDF figures are stored alongside this README. The answer-free three-query
mechanism study remains in
[`../trsd_short_pilot`](../trsd_short_pilot/README.md) and is explicitly
separate from this full 143-query accuracy evaluation.
