# TRSD-64 final evidence

This bundle records the final TRSD-64 point estimate under the current unprivileged evaluation protocol. Every method receives one fixed 10,240-token generation opportunity; the primary Acc@1 metric counts an unfinished response as wrong and retains all 143 held-out questions.

## Current-protocol Acc@1

| Method | Correct / 143 | Acc@1 |
|---|---:|---:|
| Base | 77 / 143 | 53.85% |
| Privilege-SD 16 | 81 / 143 | 56.64% |
| TRSD 16 (current matched run) | pending | pending |
| Privilege-SD 64 | 90 / 143 | 62.94% |
| **TRSD 64** | **102 / 143** | **71.33%** |

Among the four currently available matched rows, TRSD-64 has the highest strict Acc@1 point estimate: +17.48 percentage points over Base, +14.69 over Privilege-SD 16, and +8.39 over Privilege-SD 64.

## TRSD-64 by dataset

| Dataset | Correct / total | Acc@1 |
|---|---:|---:|
| AMC23 | 70 / 83 | 84.34% |
| AIME24 | 19 / 30 | 63.33% |
| AIME25 | 13 / 30 | 43.33% |
| **Combined** | **102 / 143** | **71.33%** |

## Paired transitions into TRSD-64

The following counts compare strict per-query outcomes on identical held-out questions. `W→C` is a gain for TRSD-64 and `C→W` is a regression relative to the named comparator.

| Comparator | W→C | C→W | Net correct |
|---|---:|---:|---:|
| Base | 27 | 2 | +25 |
| Privilege-SD 16 | 24 | 3 | +21 |
| Privilege-SD 64 | 16 | 4 | +12 |

## Completion-conditioned diagnostic

TRSD-64 is correct on 102 of all 143 questions under the primary unfinished-as-wrong definition. Conditional on the 118 responses that completed before the token limit, it is correct on 102/118 (86.44%). The completed-only rate is descriptive and can be selection-biased because completion is method-dependent.

## Trust-region mechanism evidence

The paired 64-episode style study supports the intended distribution-control mechanism:

- Normalized absolute style-target movement fell by **39.88%**, with a paired-bootstrap 95% reduction interval of **[34.22%, 45.10%]**.
- Target-to-student KL fell by **73.64%**, from 0.014728 for the raw privileged target to 0.003882 for the TRSD projection.
- In the separate same-prefix pilot, style shift fell by **46.04%**, from 0.116545 to 0.062883. This pilot is descriptive because it uses three distinct queries with neutral/terse/verbose wrappers.
- PSR changed from 10.1757 to 11.1460 (+9.54%), but its paired delta interval, [-0.2969, 2.3799], includes zero. The PSR change is therefore not statistically resolved and is not used as positive evidence.

See [STYLE_RESULTS.md](STYLE_RESULTS.md) and the accompanying CSV/figure artifacts for the full definitions, confidence intervals, and operational measurements.

## Protocol boundary

- `TRSD 16` in the current matched reverse-KL protocol is still pending, so this bundle does not claim a matched T16→T64 learning-curve effect.
- The previously saved `TRSD 16†` artifact predates the current explicit-budget evaluation prompt and exact reverse-KL implementation. It is historical appendix evidence only and must not be mixed into the current five-row inference table.
- No raw responses, target labels, model checkpoints, or training state are included in this published folder.

## Claim–evidence check

| Claim | Evidence | Status |
|---|---|---|
| TRSD-64 has the strongest available current-protocol Acc@1 point estimate | 102/143 versus 77/143, 81/143, and 90/143 | Supported |
| Projection substantially reduces absolute style-target movement | 39.88% reduction; paired 95% interval [34.22%, 45.10%] | Supported |
| Projection constrains target distribution shift | 73.64% lower target-to-student KL | Supported |
| PSR improves significantly | Paired delta interval includes zero | Not supported |
| Accuracy improves from matched current T16 to T64 | Current matched T16 is pending | Pending |
