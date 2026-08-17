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
- [Dependency stack](DEPENDENCIES.md)
- [Training, inference, and evaluation](RUN.md)

## Qwen3-8B checkpoints

The episode-64 TRSD and OPSD PEFT adapters are hosted directly in this
repository's GitHub Release. Download either checkpoint into the standard
local layout with:

```bash
python scripts/download_checkpoints.py --method trsd
python scripts/download_checkpoints.py --method opsd
```

Each download contains `adapter_model.safetensors`, `adapter_config.json`, and
`checkpoint_manifest.json`. The base model is `Qwen/Qwen3-8B`.

## Code layout

```text
src/clean_self_distill/                 core generation and distillation
scripts/clean_self_distill/04_persistent_train.py
scripts/clean_self_distill/05_heldout_eval.py
scripts/clean_self_distill/prepare_empirical_data.py
scripts/download_checkpoints.py
tests/                                  lightweight unit tests
```

## License

MIT. The Qwen3 base model and published adapters retain their own licenses.
