# LGSD implementation guide

This guide states exactly what the code implements. It deliberately separates
target construction from target fitting and labels the old objective as
legacy.

## 1. Construct the local target

At one training episode, let `p` be the current student distribution and let
`qP` be the distribution produced with the privileged, answer-free prompt.
LGSD solves

```text
min_q KL(q || qP)  subject to  mean_token KL(q || p) <= epsilon.
```

Along the exponential path, the solution has the form

```text
qC(alpha) proportional to p^(1-alpha) qP^alpha,
```

where `alpha` is the largest feasible value in `[0, 1]`. The implementation
interpolates logits,

```python
projected_logits = (1 - alpha) * student_logits + alpha * privileged_logits
```

which is distributionally identical to the normalized geometric mixture. A
trajectory-level binary search chooses `alpha` using the full-vocabulary mean
`KL(qC || p)`. This is the operation that removes proposal movement beyond the
budget; the final distillation-KL direction does not change that locality.

## 2. Detach and fit the projected target

After `qC` is constructed, it is detached from autograd. Canonical LGSD then
optimizes

```text
mean_token KL(qC || pi_theta).
```

Ignoring the target-only entropy constant, this is ordinary soft-target
cross-entropy:

```python
with torch.no_grad():
    target_log_probs = torch.log_softmax(projected_logits, dim=-1)
    target_probs = target_log_probs.exp()

student_log_probs = torch.log_softmax(student_logits, dim=-1)
loss = -(target_probs * student_log_probs).sum(dim=-1).mean()
```

The repository computes the equivalent full KL so that the reported loss and
diagnostics remain explicit. The target distribution decides which vocabulary
entries receive weight. The gradient with respect to the unscaled student
logits is `pi_theta - qC`.

## 3. Why reverse KL is legacy

For the same geometric target, the old objective has the exact identity

```text
KL(pi || qC)
  = (1-alpha) KL(pi || p)
  + alpha KL(pi || qP)
  + log Z.
```

`log Z` does not depend on the optimized policy. Therefore the old objective
is exactly an adaptive old-policy anchor plus raw-target fitting. The code keeps
it behind `--student-kl-direction reverse` only so old runs remain reproducible.
It is not the default and its checkpoint method ID remains the legacy `trsd:`
identity.

Forward KL generally cannot be rewritten as a weighted sum of forward KLs from
`p` and `qP`, because `qC` is a geometric mixture rather than the arithmetic
mixture required by that rewrite.

## 4. Where to check the implementation

- `src/clean_self_distill/persistent.py` constructs the geometric target,
  solves the KL budget, detaches it, and records method identity.
- `src/clean_self_distill/streaming_distill.py` computes
  `KL(qC || pi_theta)` in memory-bounded token chunks.
- `scripts/clean_self_distill/04_persistent_train.py` defaults to forward KL.
- `tests/test_persistent_training.py` checks the geometric mixture and the
  reverse-KL equivalence.
- `tests/test_streaming_distill.py` checks the forward-KL value, gradient, and
  detached target.

Every newly written checkpoint manifest includes:

```text
method_id = lgsd:geometric_kl_ball_projection:forward_kl_v1
projection_kl_direction = projected_teacher_to_pre_update_student_forward_kl_v1
distillation_kl_direction = projected_teacher_to_student_forward_kl_v1
```

Evaluation fails closed if an adapter labeled `lgsd` lacks this method identity
or direction metadata.

## 5. Do not confuse Veto with an LGSD radius

The Veto baseline uses `Q proportional to P_T P_S^beta`, with a globally
scheduled `beta`; LGSD uses `qC proportional to P_S^(1-alpha) P_T^alpha`, with
a per-trajectory `alpha` solved from the KL budget. The exponents and control
rules differ, so Veto is implemented as its own `--branch veto`, not as a fixed
LGSD alpha. See [VETO_BASELINE.md](VETO_BASELINE.md).

## 6. Checkpoint compatibility

Changing the KL direction changes finite-step optimization, so old weights are
not valid evidence for the new objective. The GitHub `v1` Qwen3-8B adapters are
kept as reverse-KL legacy artifacts and are never renamed. Train a new adapter
with this revision before reporting forward-KL LGSD results.

The defensible method claim is: LGSD keeps the privileged direction but limits
its magnitude with a student-centered KL budget, then directly fits the
detached projected target. Locality alone does not prove that the retained
signal is correct or that the removed movement is purely stylistic.
