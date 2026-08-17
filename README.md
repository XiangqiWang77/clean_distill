# Trust-Region Self-Distillation

This repository is a compact implementation of Trust-Region Self-Distillation
(TRSD) and its unprojected privileged-teacher counterpart (OPSD). It contains
only the implementation and usage instructions.

For student distribution `p_t` and privileged teacher distribution `q_t`, TRSD
uses the exponential path

```text
q_t(alpha) proportional to p_t^(1-alpha) q_t^alpha
```

and selects the largest trajectory-level `alpha` in `[0, 1]` satisfying
`mean_t KL(q_t(alpha) || p_t) <= epsilon`. OPSD uses the unprojected privileged
teacher target.

## Start here

- [Installation](INSTALL.md)
- [Training, inference, and evaluation](RUN.md)

## Public Qwen3-8B adapters

Both checkpoints are public PEFT LoRA adapters and require the pinned
`Qwen/Qwen3-8B` base revision
`b968826d9c46dd6066d109eabc6255188de91218`.

| Method | Adapter | Immutable revision |
|---|---|---|
| TRSD, episode 64 | [`qisein/Qwen3-8B-TRSD-ep64`](https://huggingface.co/qisein/Qwen3-8B-TRSD-ep64) | `52e00776e9c47295e1ef1d7d515a60595c3210ce` |
| OPSD, episode 64 | [`qisein/Qwen3-8B-OPSD-ep64`](https://huggingface.co/qisein/Qwen3-8B-OPSD-ep64) | `c467875ece536bcb629fd66d45f92138953f7c1a` |

## Code layout

```text
src/clean_self_distill/                 core generation and distillation
scripts/clean_self_distill/04_persistent_train.py
scripts/clean_self_distill/05_heldout_eval.py
scripts/clean_self_distill/prepare_empirical_data.py
tests/                                  lightweight unit tests
```

## License

MIT. The Qwen3 base model and published adapters retain their own licenses.
