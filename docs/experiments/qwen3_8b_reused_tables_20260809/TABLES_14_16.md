# Qwen3-8B reused-run Tables 14--16

Every value below is regenerated from completed runs; no new training or evaluation is used.

## Table 14

Trajectory projection at the matched 64-episode horizon on Qwen3-8B. Math is AMC23/AIME24/AIME25 combined; SATQuest is logical ID/shifted-format evaluation; LogicSkills is external OOD.

| Base method | Variant | Math Acc@1 | SATQuest | LogicSkills OOD | Target KL | Cap hits | Style shift↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Privilege-SD | raw privileged target | 60.84 | 28.93 | 58.93 | 0.01473 | — | 0.11655 |
| Privilege-SD | + trajectory projection (TRSD) | 61.54 | 29.64 | 61.13 | 0.00388 | 63 | 0.06288 |

## Table 15

Robustness across completed answer-free same-prefix probes. Task gain Δ is projected minus raw task-token log-probability gain. All rows probe the same TRSD-64 checkpoint, so Acc@1 is shared.

| Privilege source / probe | Raw KL | Projected KL | Task gain Δ | Style shift↓ | Wrapper var.↓ | Shared Acc@1 |
| --- | --- | --- | --- | --- | --- | --- |
| Answer-free reasoning method | 0.01159 | 0.00396 | +0.00141 | 0.06288 | — | 61.54 |
| Style-only directive (terse) | 0.01290 | 0.00398 | +0.00102 | 0.06493 | — | 61.54 |
| Style-only directive (verbose) | 0.01090 | 0.00397 | +0.00077 | 0.06085 | — | 61.54 |
| Equivalent prompt wrappers | 0.01180 | 0.00397 | +0.00107 | 0.06288 | 0.00000002 | 61.54 |

## Table 16

Recorded end-to-end training cost from completed Qwen3-8B runs. GRPO and DemoPSD are the completed 16-episode baselines; Privilege-SD and TRSD are the matched 64-episode long-horizon runs.

| Method | Episodes | Rollouts | Generated tokens | Teacher positions | Update steps | Total s | s / episode | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GRPO | 16 | 128 | 1056908 | 0 | 16 | 8828.7 | 551.8 | — |
| DemoPSD | 16 | 128 | 1051071 | 890949 | 14 | 8962.7 | 560.2 | — |
| Privilege-SD | 64 | 64 | 246371 | 246371 | 64 | 7239.9 | 113.1 | — |
| TRSD | 64 | 64 | 433074 | 433074 | 64 | 17284.7 | 270.1 | 22.34 |
