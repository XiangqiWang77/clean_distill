# Complete TRSD and Privilege-SD evidence

This directory makes the reported 16- and 64-episode results self-contained at the per-query and training-audit levels. It packages scored responses, checkpoint identities, training journals, run manifests, and mechanism traces; model weights remain in scratch storage and evaluation labels remain sealed.

## Evaluation outputs

Each file under `evaluation/` has exactly 143 scored rows (AMC23: 83, AIME24: 30, AIME25: 30). Each row includes the generated response, parsed answer, correctness, truncation flag, decoding identity, checkpoint identity, behavioral diagnostics, and resource telemetry.

| Checkpoint | File | Strict Acc@1 |
|---|---|---:|
| Privilege-SD 16 | `evaluation/privileged_sd_16.scored.jsonl` | 81/143 (56.64%) |
| TRSD 16 | `evaluation/trsd_16.scored.jsonl` | 77/143 (53.85%) |
| Privilege-SD 64 | `evaluation/privileged_sd_64.scored.jsonl` | 90/143 (62.94%) |
| TRSD 64 | `evaluation/trsd_64.scored.jsonl` | 102/143 (71.33%) |

Strict Acc@1 counts a response as correct only when the offline sealed-label scorer marks it correct and the response did not exhaust the 10,240-token evaluation budget.

## Training evidence

- `training/privileged_sd_16/source_run_episodes_30.jsonl` is the complete 30-episode source run. The reported P16 checkpoint uses its first 16 rows, bound by `checkpoint_manifest.json`.
- `training/privileged_sd_64/episodes.jsonl` contains the complete 64-episode P64 journal.
- `training/trsd_16_64/episodes.jsonl` contains the complete shared 64-episode TRSD run. T16 is the first 16 rows and T64 is all 64 rows; both checkpoint manifests are included.
- Every training directory includes its immutable run manifest. The SHA-256 inventory is in `MANIFEST.json`.

The aggregate report, paired-query inference, efficiency tables, mechanism diagnostics, CSVs, and LaTeX tables are in the parent directory.
