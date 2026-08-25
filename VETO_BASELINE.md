# Veto baseline

Veto is a directly competing adaptive-target baseline from Jang et al.,
“Stable On-Policy Distillation through Adaptive Target Reformulation,” formally
published in *Findings of ACL 2026*. This repository implements its published
target equation independently and exposes it for evaluation under the same
DeepMath protocol as LGSD.

## What Veto computes

For the current on-policy student distribution `P_S` and the same-prefix
teacher distribution `P_T`, Veto constructs

```text
Q_beta(y | x) proportional to P_T(y | x) P_S(y | x)^beta
```

or, equivalently,

```python
target_log_probs = log_softmax(
    teacher_log_probs + beta * student_log_probs,
    dim=-1,
)
```

The target is detached and the student minimizes

```text
KL(Q_beta || P_S).
```

For reasoning tasks, the published setting starts at `beta=0.8` and decays
linearly toward zero. The released implementation uses
`progress = global_step / total_steps`; consequently, the last finite update
approaches but does not exactly reach zero. This repository preserves that
step convention.

Veto and LGSD are not the same interpolation:

| Method | Target | Control variable | Meaning |
|---|---|---|---|
| Veto | `Q ∝ P_T P_S^beta` | global scheduled `beta` | suppress teacher mass where the student is currently unlikely |
| LGSD | `qC ∝ P_S^(1-alpha) P_T^alpha` | per-trajectory solved `alpha` | take the largest teacher-directed movement inside a fixed KL budget |

At `beta=0`, Veto becomes the raw teacher target. This is different from
LGSD’s `alpha=0`, which becomes the current student.

## Fair LGSD comparison

The ACL paper reports its original experiment with a Qwen2-0.5B student and a
Qwen2-7B SFT teacher. That is not the model setup used by this repository.
Therefore `--branch veto` is a **formula-faithful matched baseline**, not a
claim to reproduce the paper’s reported accuracy.

For a controlled comparison, Veto, LGSD, and OPSD must share:

- the same base checkpoint and LoRA initialization;
- the same ordered DeepMath query stream and rollout seeds;
- the same pre-decision privileged teacher prompt and on-policy prefix;
- the same optimizer, learning rate, token cap, and checkpoint episodes;
- the same forward-KL implementation and held-out decoding seeds.

Only target reformulation differs. Use the same checkpoints (for example,
episodes 16 and 64) and evaluate all methods with the existing held-out math
and logic scorers. Do not copy the ACL paper’s GSM8K number into an LGSD table;
it comes from a different student, teacher, and training protocol.

Suggested baseline-status row before matched results finish:

| Method | Paper | Publication status | Matched DeepMath status |
|---|---|---|---|
| Veto | Jang et al., *Stable On-Policy Distillation through Adaptive Target Reformulation* | Findings of ACL 2026 | Implemented; results must come from a new matched run |

## Run the matched baseline

Use the same values passed to the matched LGSD run:

```bash
python scripts/clean_self_distill/04_persistent_train.py \
  --branch veto \
  --queries data/train_queries.jsonl \
  --model "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --output-dir outputs/veto \
  --episodes 64 \
  --scientific-checkpoints 0,16,32,48,64 \
  --rolling-checkpoint-interval 8 \
  --max-sequence-tokens 10240 \
  --max-rollout-tokens 10240 \
  --learning-rate 2e-5 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --distill-token-chunk-size 128 \
  --student-kl-direction forward \
  --veto-beta-start 0.8 \
  --veto-beta-end 0.0 \
  --veto-beta-schedule linear \
  --seed 0 \
  --attn-implementation sdpa
```

Each run, checkpoint, and episode journal records:

```text
method_id = veto:adaptive_target_reformulation:forward_kl_v1
target_reformulation = teacher_student_product_of_experts_v1
distillation_kl_direction = adaptive_target_to_student_forward_kl_v1
veto_beta = the coefficient used for that episode
```

These fields prevent a Veto adapter from being silently reported as LGSD or
OPSD.

Generate held-out predictions with the same decoding configuration used for
the matched methods:

```bash
python scripts/clean_self_distill/05_heldout_eval.py generate \
  --queries data/eval_queries.jsonl \
  --output outputs/veto_predictions.jsonl \
  --model "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --adapter outputs/veto/checkpoints/episode_0064 \
  --method veto \
  --checkpoint-episode 64 \
  --sample-count 1 \
  --seed 0
```

The evaluator rejects a checkpoint unless both its Veto target identity and
forward-KL direction are present in the manifest.

## Provenance

- Paper: <https://aclanthology.org/2026.findings-acl.2094/>
- DOI: <https://doi.org/10.18653/v1/2026.findings-acl.2094>
- Official code: <https://github.com/jjun-0824/Veto>
- Reference commit inspected: `0ff04a0de21e93bb7e13beaa55d37fd6975dd70e`
- ACL PDF SHA-256:
  `b28dba5bcc09ebb38d23af53da069374a11ed6d26ecd040454ebee7daf5f3e4b`

The inspected official repository did not contain a license file, so its
source is not copied into this MIT repository. `src/clean_self_distill/veto.py`
is an independent implementation of the public equation and schedule.

```bibtex
@inproceedings{jang-etal-2026-stable,
  title = {Stable On-Policy Distillation through Adaptive Target Reformulation},
  author = {Jang, Ijun and Yeom, Jewon and Yeo, Juan and Lim, Hyunggyu and Kim, Taesup},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  month = jul,
  year = {2026},
  address = {San Diego, California, United States},
  publisher = {Association for Computational Linguistics},
  pages = {42217--42227},
  doi = {10.18653/v1/2026.findings-acl.2094},
  url = {https://aclanthology.org/2026.findings-acl.2094/}
}
```
