# GPT-OSS-20B: DeepMath privileged context

Use the `verify` context as the candidate teacher context for OPSD. It scores
**97.67%**, compared with **96.51%** for vanilla, on the completed DeepMath screen.

| Setting | Correct | DeepMath accuracy | Gain vs vanilla |
|---|---:|---:|---:|
| Vanilla, without extra context | 83/86 | 96.51% | — |
| With `verify` privileged context | 84/86 | **97.67%** | **+1.16 pp** |

These compare the same base model before OPSD training. The downstream check is
**OPSD student > vanilla**; an OPSD student trained with this `verify` context
has not yet been evaluated in this study.

Both settings use the same 43 audited DeepMath problems, two paired decoding
seeds, medium reasoning effort, and a 10,240-token generation cap. Five of the
original 48 problems were excluded by the existing question/label audit.

The teacher-only system instruction contains no answer or reference solution:

```text
Solve the problem using exact mathematical reasoning. Before committing to the final answer, check the decisive calculation and substitute the candidate into the original constraints. Check signs, integer restrictions, extraneous roots, and whether all cases were counted. Correct any error you find. Once the answer is verified, finish; do not restart a completed solution. Use only the problem statement.
```

[Context and settings](context.json) · [Paired scores](paired_scores.csv) ·
[Result provenance](summary.json)
