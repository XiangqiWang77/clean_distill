# Experimental Settings and Reproducibility Disclosure

This document records the complete experimental protocol used by the current
TRSD study. It distinguishes the primary Qwen3-8B experiment from scale
extensions, and it distinguishes configuration intent from the settings stored
in the completed run artifacts. Unless stated otherwise, all reported accuracy
is strict single-sample accuracy (`Acc@1`).

Last audited: 2026-08-09.

## 1. Experimental scope

The primary experiment starts from `Qwen/Qwen3-8B`, performs persistent
self-distillation on a frozen stream of DeepMath problems, and evaluates the
resulting adapters on AMC 2023, AIME 2024, and AIME 2025. SATQuest and
LogicSkills are external transfer evaluations of these math-trained adapters;
the Qwen3-8B results in the main logical-transfer table do not include further
logic training.

| Axis | Primary setting |
|---|---|
| Base model | `Qwen/Qwen3-8B` |
| Train data | DeepMath levels 7--10 |
| Training horizons | 16 and 64 episodes |
| Main methods | Base, Privilege-SD, TRSD |
| Optimization unit | one on-policy trajectory and one optimizer update per episode |
| Math evaluation | AMC23 (83), AIME24 (30), AIME25 (30) |
| Logical transfer | SATQuest (3,360), LogicSkills (1,500) |
| Main metric | strict `Acc@1` |
| Drift diagnostics | lexical realized-token shift and lexicon-free AP-JSD |
| Training seed count | one (`seed=0`) |

The scale extensions use the same data stream and core method with
`Qwen/Qwen3-1.7B` and `openai/gpt-oss-20b`. SRPO, GRPO, OPSD, privilege-source
controls, and component ablations are documented separately below because some
of them require training labels or use different objectives.

## 2. Model identities

Model revisions are pinned rather than resolved from a moving branch.

| Model | Hugging Face revision | LoRA target modules |
|---|---|---|
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| GPT-OSS-20B | `6cee5e81ee83917806bbde320786a8fb61efebee` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

GPT-OSS uses attention-only LoRA because its MoE expert matrices are stored as
MXFP4 block tensors rather than ordinary linear modules supported by the shared
PEFT path. The LoRA target selection is implemented in
[`src/clean_self_distill/runtime.py`](../src/clean_self_distill/runtime.py).

## 3. Data

### 3.1 DeepMath training stream

The source is pinned as follows.

| Field | Value |
|---|---|
| Repository | `Leyiii/RLCSD` |
| Revision | `33d7de919af5b03257ff92c30303fddf9afdda4a` |
| File | `deepmath_filtered_level7_10/train.parquet` |
| Declared difficulty | levels 7--10 |
| File bytes | 587,568,701 |
| SHA-256 | `611d3030a2a74eaea9514ab732fc33aa6a35d668c7f804c383772939b159f2a0` |

The deterministic preprocessing procedure is:

1. Require a nonempty problem, answer, and reference solution in the source
   record. These labels are used to validate and deduplicate the source, not as
   inputs to TRSD or Privilege-SD.
2. Normalize problem text by collapsing whitespace and applying case folding.
3. Group records by normalized problem text.
4. Exclude an entire group when its answer-plus-solution fingerprints conflict.
5. For a consistent duplicate group, choose the exact problem text with the
   smallest SHA-256 as its canonical representative.
6. Sort canonical representatives by problem SHA-256.
7. Select 1,000 distillation queries followed by 200 development queries.
8. Use the first 16 or 64 rows of the same frozen distillation order for the
   corresponding horizon.

The GPU-facing query records contain only:

```json
{
  "query_id": "deepmath:<problem_sha256>",
  "problem": "<problem text>",
  "problem_sha256": "<sha256>",
  "source": "deepmath"
}
```

Answers and reference solutions are stored in separate permission-restricted
`*.sealed.jsonl` files. The preprocessing step also audits normalized-problem
overlap among the distillation, development, and held-out splits. The
implementation is in
[`scripts/clean_self_distill/prepare_empirical_data.py`](../scripts/clean_self_distill/prepare_empirical_data.py).

### 3.2 Held-out mathematics

The held-out file is from the same pinned source revision.

| Field | Value |
|---|---|
| File | `amc23+aime24+aime25/val.parquet` |
| File bytes | 105,032 |
| SHA-256 | `42e7c50d0511fb52680ae6fc6cbfc46ff6c361771378dd5c6d228acb61be1cbf` |
| AMC 2023 | 83 queries |
| AIME 2024 | 30 queries |
| AIME 2025 | 30 queries |
| Total | 143 queries |

The canonical evaluation order is AMC23, then AIME24, then AIME25, with
problem-hash order within each source.

### 3.3 Logical benchmarks

The logical datasets were converted to the same top-level parquet contract as
DeepMath: `prompt`, `reward_model`, `data_source`, `ability`, and `extra_info`.

| Asset | Source identity | Rows | Converted SHA-256 |
|---|---|---:|---|
| SATQuest evaluation | Git commit `8932b68688165c49bce4acbac94e2d3999cd8195`; official HF source SHA-256 `4082bd9e95c75b20ab5a211eab1a171cdca20f1686f6e881985045c470b80508` | 3,360 | `01b65a6514952a0dfdbbd7290b5f9968ae115a465c8ee256727139c0e0deb56f` |
| SATQuest training conversion | official HF source SHA-256 `c5a2544959811c74186e810996008841d73e0a9a47c51438c35d9a5f1854ed6e` | 36,000 | `94f423c8de33b6db55e6824816e0630f32149e8233e4cb174ed4865ffb95266c` |
| LogicSkills evaluation | Git commit `1f23e684d6b1a465047f8b0d833d9b9b3388441a` | 1,500 | `f414193c1ed1eca75156c3124d96fedf815f992e6e552570137de67c5dc03024` |

SATQuest evaluation contains six equally sized problem types: MCS, MUS,
MaxSAT, SATDP-SAT, SATDP-UNSAT, and SATSP, with 560 instances of each. It is
partitioned as:

| Regime | Rows | Formats |
|---|---:|---|
| ID | 240 | Math and Story |
| Format OOD | 240 | DIMACS and DualStory |
| Size OOD | 1,440 | Math and Story |
| Size + format OOD | 1,440 | DIMACS and DualStory |

The converted SATQuest training set has 3--4 variables, uses only Math and
Story formats, and has zero source-ID overlap with the evaluation set. It is
available for separate logic-training studies but is not used to train the
math-to-logic transfer checkpoints reported in the primary table.

LogicSkills contains:

| Task | Rows | Language composition |
|---|---:|---|
| Countermodel construction | 300 | formal |
| First-order symbolization | 600 | 300 English, 300 Carroll |
| Logical validity | 600 | 300 English, 300 Carroll |

## 4. Exact prompt templates

Every message below is shown before the tokenizer's model-specific chat
template is applied. `add_generation_prompt=True` is used throughout. For the
main Qwen mathematics runs, `enable_thinking` is not overridden, so the pinned
Qwen chat template's default is used. Logic evaluation explicitly sets
`enable_thinking=True`.

### 4.1 Ordinary student training prompt

There is no system message. The single user message is:

```text
{problem}

Please reason step by step, and put your final answer within \boxed{}.
```

This prompt is used to sample the on-policy trajectory. The Base model,
Privilege-SD student, and TRSD student all use the same ordinary prompt.

### 4.2 Privileged teacher prompt

The teacher-only system message is exactly:

```text
Private reasoning-method instruction for the teacher: decompose the problem into explicit subgoals, track constraints and invariants, check boundary cases, and verify the chosen route against an independent alternative when possible. Use only the problem statement.
```

The accompanying user message is exactly:

```text
{problem}

Please reason step by step, and put your final answer within \boxed{}.
```

The privileged context is therefore an answer-free reasoning-method
instruction. It contains no target answer, reference solution, correctness
label, reward, verifier output, previous-attempt feedback, future response
token, or post-outcome information.

The student and privileged teacher are the same current model. The privileged
teacher scores the exact response prefixes already sampled by the ordinary
student; it does not generate a replacement solution in the main setting. At
position \(t\), both distributions condition on the same realized
\(y_{<t}\), while the teacher additionally conditions on the system message.
Thus on-policy prefix parity is 100%, while literal prompt/context parity is
0% by construction.

The implementation is in
[`src/clean_self_distill/persistent.py`](../src/clean_self_distill/persistent.py).

### 4.3 Mathematics evaluation prompt

For the reported 10,240-token evaluations, the single user message is:

```text
{problem}

Please reason step by step. You have a strict response budget of at most 10,240 generated tokens. Finish the reasoning and put the final answer within \boxed{} before reaching that limit; prioritize completing the answer over extending the analysis.
```

Its version identifier is `explicit-generation-budget-v1`. This wording is
shared across all methods in the current main mathematics table.

### 4.4 Answer-free wrapper robustness prompts

The neutral wrapper is byte-for-byte the training-time privileged prompt in
Section 4.2. The two paraphrased teacher-only system messages are:

Terse:

```text
Private reasoning-method instruction for the teacher: use only the problem statement. Split the task into explicit subgoals, track constraints and invariants, check boundary cases, and when possible verify the route independently.
```

Verbose:

```text
Private reasoning-method instruction for the teacher: rely exclusively on the stated problem, with no answer, solution, outcome, or external feedback. Organize the reasoning as explicit subgoals; keep all constraints and invariants visible; examine relevant boundary cases; and, whenever feasible, validate the selected route by an independent alternative calculation or argument.
```

Both use the same user message as Section 4.2. The three wrappers are scored on
the same fixed ordinary-context prefixes. Their implementation is in
[`src/clean_self_distill/trust_region_mechanism.py`](../src/clean_self_distill/trust_region_mechanism.py).

### 4.5 Logic evaluation prompt

No additional task-generic instruction is appended. The benchmark-provided
problem text is sent as the only user message:

```text
{benchmark_problem}
```

The pinned model chat template is applied with `enable_thinking=True`.

### 4.6 SRPO correct-sibling prompt

SRPO samples eight ordinary-prompt rollouts. If the deterministic answer
checker accepts at least one rollout, one accepted rollout is selected using
`Random(episode_seed + 91337)`. Incorrect rollouts use that accepted sibling as
teacher information in the paper's single user message:

```text
{problem}
Correct solution:
{selected_correct_rollout}
Correctly solve the original question.
```

The EMA self-teacher re-scores each incorrect student's own trajectory under
this prompt. Correct rollouts use GRPO. If no accepted sibling exists, every
rollout falls back to GRPO; no reference solution or held-out label is inserted
into the prompt.

### 4.7 OPSD reference-solution prompt

The OPSD implementation uses this single user message:

```text
Problem: {problem}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the reasoning behind each step; do not copy or paraphrase it. Using your own words and independent reasoning, derive the same final answer. Think step by step, and put your final answer within \boxed{}.
```

### 4.8 Privilege-source study prompts

Table 15-style source controls use this common message structure:

```text
SYSTEM:
{source_specific_instruction}

{source_specific_private_text}

USER:
{problem}

Reason step by step and put the final answer within \boxed{}.
```

The exact source-specific values are:

| Source | Instruction | Private text |
|---|---|---|
| Verified reference solution | `Use the following verified reference solution as private guidance, then independently derive the answer.` | verified reference solution |
| Answer-free reasoning method | `Apply the following private answer-free reasoning method.` | `Decompose the problem into explicit subgoals; track constraints and invariants; check boundary cases; verify the route independently. Use only the problem statement and do not assume an answer.` |
| Verifier critique | `Use this private verifier critique to revise the reasoning.` | accepted/rejected status followed by an instruction to locate the earliest unsupported step; no reference answer |
| Execution/solver feedback | `Use this private execution/verifier feedback to reconstruct a correct solution.` | parsed candidate and verified answer |
| Equivalent prompt wrappers | one of three verified-solution paraphrases | verified reference solution |
| Style-only directive | `Follow this private response-style directive; it contains no task answer.` | `Write in concise academic prose. Use explicit transitions, numbered steps, and a final verification sentence. Do not add task facts.` |
| Permuted non-inferable context | states that the context comes from a different problem | deterministically permuted reference solution from another problem |

The three exact equivalent-wrapper instructions are:

```text
Use the following verified worked solution as private guidance. Derive the answer independently.

Consult this correct solution privately, then solve the problem in your own reasoning and wording.

The private material below is a verified derivation. Internalize it and produce an independent solution.
```

The remaining variable private-text templates are exactly:

```text
# Verifier critique, where verdict is "accepted" or "rejected"
The deterministic answer verifier {verdict} the candidate. Re-check the derivation, locate the earliest unsupported step, and recompute the final answer. No reference answer is supplied.

# Execution/solver feedback
Deterministic checker result: candidate={parsed_candidate!r}; verified answer={answer!r}. Reconstruct a valid derivation that reaches the verified result.

# Style-only directive
Write in concise academic prose. Use explicit transitions, numbered steps, and a final verification sentence. Do not add task facts.
```

The exact permuted-context instruction is:

```text
The following private context was deterministically permuted from a different problem and contains no inferable answer to this task.
```

All baseline and privilege-source templates are implemented in
[`baselines/train.py`](../baselines/train.py).

## 5. TRSD training algorithm

For one problem \(x\), the current student first samples one trajectory
\(y\sim p_\theta(\cdot\mid x)\) from the ordinary prompt. At every realized
prefix \(y_{<t}\), the ordinary and privileged next-token distributions are

\[
p_t(v)=p_\theta(v\mid x,y_{<t}),
\qquad
q_t^P(v)=p_\theta(v\mid m,x,y_{<t}),
\]

where \(m\) is the teacher-only instruction in Section 4.2.

TRSD uses the exponential path

\[
q_{t,\alpha}(v)
=
\frac{p_t(v)^{1-\alpha}q_t^P(v)^\alpha}
{\sum_u p_t(u)^{1-\alpha}q_t^P(u)^\alpha}
=
\operatorname{softmax}\!\left((1-\alpha)z_t^S+\alpha z_t^P\right)_v.
\]

A single \(\alpha\) is shared by every token in the trajectory. TRSD chooses
the largest feasible value:

\[
\alpha^*=
\max\left\{\alpha\in[0,1]:
\frac{1}{T}\sum_{t=1}^T
D_{\mathrm{KL}}(q_{t,\alpha}\Vert p_t)
\leq\epsilon
\right\}.
\]

The primary settings are:

| Projection field | Value |
|---|---:|
| KL budget \(\epsilon\) | 0.004 |
| Constraint KL direction | \(D_{\mathrm{KL}}(q\Vert p)\) |
| Scope | one shared coefficient per trajectory |
| Path | exponential/geometric |
| Bisection steps | 6 |
| Raw teacher already feasible | use \(\alpha=1\) |
| Vocabulary | complete model vocabulary |
| Vocabulary-tensor token chunk | 128 |

The projected target is detached. The current primary TRSD objective is exact
full-vocabulary reverse KL:

\[
\mathcal L_{\mathrm{TRSD}}
=
\frac1T\sum_t
D_{\mathrm{KL}}\!\left(
p_\theta(\cdot\mid x,y_{<t})
\Vert
\operatorname{stopgrad}(q_{t,\alpha^*})
\right).
\]

The `distill_top_k=64` field remains in the run schema for compatibility with
older checkpoints. Current TRSD computes the projection and loss over the full
vocabulary and does not approximate the objective with top-k.

After one backward pass and AdamW update, the temporary privileged activations
and target tensors are destroyed. Model and optimizer states persist across
episodes.

## 6. Main optimization and generation hyperparameters

| Field | Qwen3-8B TRSD setting |
|---|---:|
| Numeric precision | BF16 |
| Attention implementation | SDPA |
| Adaptation | LoRA |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| LoRA bias | none |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Weight decay | 0 |
| Adam betas | `(0.9, 0.999)` |
| Adam epsilon | `1e-8` |
| Maximum gradient norm | 1.0 |
| Distillation temperature | 1.0 |
| Per-token loss clipping | disabled (`0`) |
| Gradient accumulation | one trajectory per update |
| Gradient checkpointing | enabled |
| Checkpointing reentrant mode | false |
| Training `use_cache` | false |
| Train seed | 0 |
| Episode seed | `seed + stream_index * 100003` |
| Rollout temperature | 0.6 |
| Rollout top-p | 0.95 |
| Rollout top-k | 20 |
| Scientific checkpoints | 0, 16, 32, 48, 64 |
| Rolling checkpoint interval for TRSD-64 | 8 episodes |

One episode contains exactly one DeepMath query, one ordinary on-policy
rollout, target construction on that rollout, and one optimizer update unless
an explicitly named update-guard ablation rejects the update.

## 7. Actual Qwen3-8B run-level settings

The completed artifacts used for the present 16/64-horizon figures have the
following realized settings.

| Run | Source relationship | Max sequence | Configured max rollout | Realized response tokens | 128-token chunks | Optimizer updates |
|---|---|---:|---:|---:|---:|---:|
| Privilege-SD-16 | checkpoint at episode 16 of its source run | 16,384 | 16,384 | 140,087 | 1,102 | 16 |
| TRSD-16 | checkpoint at episode 16 of the TRSD-64 run | 10,240 | 10,240 | 126,802 | 998 | 16 |
| Privilege-SD-64 | independent 64-episode run | 16,384 | 4,096 | 246,371 | 1,932 | 64 |
| TRSD-64 | full 64-episode run | 10,240 | 10,240 | 433,074 | 3,413 | 64 |

The privileged prompt is longer than the ordinary prompt. TRSD therefore uses
the smaller of the configured rollout cap and `max_sequence_tokens -
max(student_prompt_tokens, teacher_prompt_tokens)`. The realized TRSD-64
per-episode rollout budgets range from 9,891 to 10,146 tokens.

The 64-episode Privilege-SD and TRSD artifacts use the same 64 query IDs in the
same order, the same initialization, LoRA size, learning rate, and training
seed, and the same number of optimizer updates. They are matched by episode
horizon, not by generated-token count.

The historical Qwen3-8B Privilege-SD checkpoints used
\(D_{\mathrm{KL}}(q^P\Vert p_\theta)\), evaluated on the teacher's top-64
vocabulary entries plus one explicit aggregated `other` bucket. The current
TRSD checkpoint uses the full-vocabulary reverse-KL objective in Section 5.
Consequently, the existing `direct OPSD`/Privilege-SD-64 row is the completed
historical direct-distillation baseline rather than a strict one-line removal
of projection from the current TRSD code path.

## 8. Mathematics evaluation

The current main evaluation uses one sample for each of the 143 held-out
queries.

| Field | Value |
|---|---:|
| Engine | batched vLLM |
| Precision | BF16 |
| Samples per query | 1 |
| Temperature | 0.6 |
| Top-p | 0.95 |
| Top-k | 20 |
| Max generated tokens | 10,240 |
| Max prompt tokens | 8,192 |
| Context window | 40,960 |
| Base evaluation seed | 0 |
| Per-query seed | `base_seed + global_query_index * 1009 + sample_index` |
| Number of shards | 4 |
| Batch size per shard | 64 |
| Tensor parallel size | 1 |
| vLLM GPU-memory utilization | 0.88 |
| Maximum LoRA rank | 8 |

The seed range for the 143 one-sample queries is 0--143,278. All compared
methods use the same per-query seed.

### 8.1 Strict answer scoring

The scorer extracts the last balanced `\boxed{...}` expression. It then:

1. applies direct normalized numeric comparison when both answers are simple
   numbers;
2. otherwise parses the prediction and ground truth with `math_verify` using
   `fallback_mode="no_fallback"`;
3. calls symbolic `verify(..., timeout_seconds=5)`;
4. falls back to normalized string equality only if symbolic parsing or
   verification raises an exception.

A query is strictly correct only when the boxed answer is accepted and the
generation did not terminate because it reached the token cap. Missing or
malformed boxed answers and length-truncated outputs count as incorrect. There
is no LLM judge. Scoring is implemented in
[`src/opsd_format.py`](../src/opsd_format.py) and
[`src/clean_self_distill/heldout.py`](../src/clean_self_distill/heldout.py).

## 9. Logic evaluation

The full logical evaluation contains all 4,860 converted benchmark items and
uses:

| Field | Value |
|---|---:|
| Decoding | greedy pass@1 |
| Temperature | 0 |
| Qwen thinking mode | explicitly enabled |
| Max generated tokens | 10,240 |
| Max prompt tokens | 8,192 |
| Seed | 20260808 |
| Precision | BF16 |
| Batch size | 512 |
| Number of Qwen3-8B shards | 2 |
| Hardware | one H100 per shard |

SATQuest answers are parsed as binary certificates of the exact requested
length and checked with the official `Problem.check`/PySAT path. LogicSkills
validity answers are exact option sets. Symbolization is parsed and checked by
Z3 entailment against the expected formula. Countermodels are parsed into the
documented finite-domain format and validated with Z3. No heuristic LLM repair
or LLM judge is used. The implementation is in
[`scripts/clean_self_distill/15_logic_eval.py`](../scripts/clean_self_distill/15_logic_eval.py)
and
[`src/clean_self_distill/logic_evaluation.py`](../src/clean_self_distill/logic_evaluation.py).

## 10. Original lexical drift metric

The original vocabulary-based diagnostic is versioned as
`rlcsd-style-task-v1`. Its precise name is **lexicon-conditioned realized-token
absolute log-probability shift**. It does not count how frequently the words
occur in the final generated text and it is not a full-policy divergence.

### 10.1 Complete style vocabulary

The fixed 22-word style lexicon is:

```text
accordingly
alternatively
answer
clearly
consequently
finally
first
hence
however
indeed
next
note
now
perhaps
second
similarly
step
suppose
therefore
thus
verify
we
```

This vocabulary covers sequencing, discourse transitions, metareasoning, and
verification language. In particular, `step` and `verify` directly probe
language emphasized by the privileged reasoning-method instruction.

### 10.2 Complete task-token rule

Task tokens are detected first using this exact case-insensitive regular
expression:

```regex
(?:\d|[=+\-*/^<>%{}\[\]()]|\\(?:frac|sqrt|boxed|sum|prod|mod|equiv|binom|gcd|lcm|sin|cos|tan|log|ln|pi|theta|alpha|beta))
```

It includes every digit, the listed mathematical punctuation, and the listed
LaTeX commands. Task matching has priority over style matching.

### 10.3 Token classification

For each realized response token ID:

1. use `convert_ids_to_tokens` when available, otherwise decode the individual
   token;
2. replace SentencePiece `▁` and GPT-style BPE `Ġ` with spaces;
3. assign `task` if the task regex matches;
4. otherwise extract case-folded alphabetic substrings with `[A-Za-z]+` and
   assign `style` if any substring is in the 22-word lexicon;
5. assign every remaining token to `other`.

### 10.4 Shift definition

For the token \(y_t\) actually sampled by the ordinary student, the pre-update
shift is

\[
d_t=
\left|
\log q_t(y_t)-\log p_t(y_t)
\right|.
\]

For Privilege-SD, \(q_t=q_t^P\). For TRSD,
\(q_t=q_{t,\alpha^*}\). The aggregate metrics are token-weighted:

\[
\mathrm{StyleDrift}
=\frac{\sum_{t\in\mathcal S}d_t}{|\mathcal S|},
\qquad
\mathrm{TaskMovement}
=\frac{\sum_{t\in\mathcal T}d_t}{|\mathcal T|},
\]

\[
\mathrm{PSR}
=\frac{\mathrm{StyleDrift}}{\mathrm{TaskMovement}}.
\]

The main 64-episode intervals use 10,000 paired episode/query bootstrap
resamples with seed `20260807`. Each resample selects the same episode indices
for Privilege-SD and TRSD and recomputes the aggregate ratio from summed shifts
and token counts. The branches use matched queries but generate their own
on-policy trajectories; the descriptive three-query by three-wrapper probe
additionally evaluates both targets on identical fixed prefixes.

The implementation is in
[`src/clean_self_distill/persistent.py`](../src/clean_self_distill/persistent.py),
and the reported summaries are in
[`docs/results/drift_metrics_side_by_side.md`](results/drift_metrics_side_by_side.md).

## 11. Lexicon-free anchored policy drift

Anchored Policy Jensen--Shannon Divergence (AP-JSD) measures complete-policy
movement without a style lexicon or task-token regex.

### 11.1 Anchor construction

1. Deterministically sort held-out AMC/AIME queries by `SHA256(query_id)` and
   select the first 32.
2. Use the Base model's existing ordinary-context evaluation continuation for
   each query.
3. Retain the first 512 continuation tokens.
4. Place anchors every 32 tokens at zero-based continuation offsets
   `31, 63, 95, ..., 511`.
5. At each anchor, feed the identical ordinary prompt and identical Base
   continuation prefix to the frozen Base policy and the adapted policy.
6. Compare their complete next-token vocabularies.

There are 16 anchors per query and 512 anchors in total.

### 11.2 Metric

For Base distribution \(p_0\), method distribution \(p_k\), and
\(m=(p_0+p_k)/2\),

\[
\mathrm{AP\text{-}JSD}(p_0,p_k)
=
\frac{
\tfrac12D_{\mathrm{KL}}(p_0\Vert m)
+\tfrac12D_{\mathrm{KL}}(p_k\Vert m)
}{\log 2}
\in[0,1].
\]

The 16 anchors are first averaged within each query, then the 32 query means
are averaged. This gives every query equal weight. Ninety-five-percent
intervals use 5,000 query bootstrap resamples with seed `20260809`.

Entropy retention is computed at the same anchors as

\[
\frac{H(p_k)}{H(p_0)},
\]

first averaged within query and then across queries. AP-JSD measures total
policy movement; it is not labeled as style drift and is not interpreted as a
metric that must always decrease. The implementation is in
[`scripts/clean_self_distill/17_fixed_prefix_policy_drift.py`](../scripts/clean_self_distill/17_fixed_prefix_policy_drift.py).

## 12. Epsilon sensitivity and wrapper pilot

The answer-free mechanism probe uses three development queries and the neutral,
terse, and verbose prompts in Section 4.4. It evaluates:

```text
alpha grid   = 0.00, 0.05, 0.10, ..., 0.95, 1.00
epsilon grid = 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.080
bisection    = 8 steps
```

The development sensitivity table marks `epsilon=0.004` as the training
setting. It uses realized-token task gain, style shift, and wrapper variance;
it does not use held-out answer correctness. Because it contains only three
distinct queries, it is reported as a descriptive mechanism probe rather than
as a benchmark estimate.

## 13. SRPO routed baseline

The SRPO comparison trains for 64 DeepMath episodes and evaluates the episode
16 and 64 checkpoints for Qwen3-1.7B, Qwen3-8B, and GPT-OSS-20B. It uses the
same frozen 64-query stream and the same 143 AMC23/AIME24/AIME25 scorer as the
main study; it does not use the logical-reasoning datasets.

| Field | Setting |
|---|---:|
| Episodes / reported checkpoints | 64 / {16, 64} |
| Group size | 8 |
| Max prompt / rollout tokens | 2,048 / 8,192 |
| Max ordinary / teacher sequence | 10,240 / 40,960 |
| Learning rate / warmup | `5e-6` / 10 updates |
| LoRA rank / alpha | 8 / 16 |
| Weight decay | 0.01 |
| Train temperature / top-p / top-k | 1.0 / 1.0 / disabled |
| EMA update rate | 0.05 |
| SDPO support | EMA-teacher top-100 plus one tail bucket |
| Generalized JSD coefficient | 0.5 |
| Entropy coefficient | 1.0 |
| GRPO clip low / high | 0.20 / 0.28 |
| GRPO reference-KL coefficient | 0 |
| Rollout importance clip | 2.0 (inert at the on-policy ratio of one) |
| Train seed | 0 |

For correctness indicator (c_i) and availability (m_i) of a correct
sibling, routing is (z_i^{\mathrm{SDPO}}=(1-c_i)m_i) and
(z_i^{\mathrm{GRPO}}=1-z_i^{\mathrm{SDPO}}). The EMA teacher re-scores the
student's own tokens under the correct-sibling prompt. Its uncertainty weight
is (\exp[-H(q_t)]), normalized to mean one over all routed SDPO tokens. The
final update is normalized once over tokens from both routes, without a manual
loss-mixing coefficient.

This is an objective-faithful local implementation of the SRPO paper rather
than a claim of systems-level reproduction: the paper reports a `verl`/FSDP2/
SGLang stack, whereas this study retains the existing LoRA and streamed-logit
infrastructure for matched comparison. The sealed DeepMath answer is used only
to compute training outcome rewards and routing. Held-out labels remain offline.

The implementations and objective tests are in
[`baselines/train.py`](../baselines/train.py) and
[`baselines/objectives.py`](../baselines/objectives.py).

## 14. Qwen3-8B component ablations

The 64-episode component jobs use the same Qwen3-8B revision, 64-query stream,
BF16/SDPA model path, LoRA 8/16, learning rate `2e-5`, weight decay 0,
10,240-token training cap, 128-token vocabulary chunks, `epsilon=0.004`, six
bisection steps, and seed 0.

| Variant | Exact change from full TRSD |
|---|---|
| Independent token budgets \(\alpha_t\) | each position independently chooses the largest \(\alpha_t\) with token KL at most 0.004 |
| Fixed global \(\alpha\) | use `0.5595703125` for every token and trajectory |
| Arithmetic probability path | use \(q=(1-\alpha)p+\alpha q^P\) instead of exponential interpolation |
| Forward-KL student loss | optimize \(D_{\mathrm{KL}}(q^{TR}\Vert p_\theta)\) |
| Without same-prefix scoring | independently generate the privileged teacher trajectory with seed `episode_seed + 50000003`, then truncate or EOS-pad it to the student trajectory length and align by position |
| Realized-update guard | skip the optimizer update when the target-minus-student normalized log-probability of the realized trajectory is nonpositive |

Full TRSD uses a shared trajectory coefficient, exponential interpolation,
full-vocabulary reverse-KL student loss, same-prefix scoring, and no update
guard.

The component-ablation evaluator uses one shard, batch size 64, one sample,
10,240 generated tokens, an 8,192-token prompt cap, a 20,480-token context,
and evaluation seed `20260808`. These settings are recorded separately from
the four-shard main-table evaluation in Section 8.

## 15. Scale-extension settings

### 15.1 Qwen3-1.7B

The four Privilege-SD/TRSD runs at 16 and 64 episodes are independent runs with
matched settings:

```text
max sequence / rollout = 10,240 / 10,240
learning rate          = 2e-5
LoRA rank / alpha      = 8 / 16
weight decay           = 0
chunk size             = 128
KL budget              = 0.004
bisection steps        = 6
rollout sampling       = temperature 0.6, top-p 0.95, top-k 20
train seed             = 0
```

Both Privilege-SD and TRSD use the current exact full-vocabulary reverse-KL
student loss. Math evaluation uses one sample, 10,240 generated tokens, a
40,960-token context, batch size 64, and seed `20260808`.

The separate 64-episode SRPO scale run uses group size 8, learning rate
`5e-6`, train temperature 1.0, and the routed settings in Section 13.

### 15.2 GPT-OSS-20B

The GPT-OSS-20B DeepMath runs use:

```text
max sequence / rollout = 10,240 / 10,240
learning rate          = 2e-5
LoRA rank / alpha      = 8 / 16
weight decay           = 0
chunk size             = 128
KL budget              = 0.004
TRSD bisection steps   = 6
Privilege-SD bisection field in completed manifests = 5
rollout sampling       = temperature 0.6, top-p 0.95, top-k 20
train seed             = 0
```

The model uses attention-only LoRA as described in Section 2. Math evaluation
uses batch size 64, one sample, 10,240 generated tokens, a 20,480-token
context, and seed `20260809`. GPT-OSS logical evaluation uses tensor parallel
size 2, two H100s per job, eight dataset shards, and batch size 128.

## 16. Statistical reporting

Accuracy is reported as exact counts and percentages. When an accuracy
difference interval is shown, it is generated by 10,000 paired bootstrap
resamples of the shared held-out query IDs. When reported, the two-sided
McNemar test is exact and uses only the two discordant counts: Base-wrong to
method-correct and Base-correct to method-wrong.

The uncertainty units are:

| Quantity | Resampling unit | Replicates | Seed |
|---|---|---:|---:|
| Accuracy difference | paired held-out query | 10,000 | report-specific recorded seed |
| Lexical drift and PSR | paired training episode/query | 10,000 | 20260807 |
| AP-JSD and entropy retention | held-out anchor query | 5,000 | 20260809 |

There is one training seed for the main runs. Query/episode bootstrap intervals
therefore quantify variation across matched examples, not variation across
independent training seeds.

## 17. Training-process figures

The study uses three distinct progress units:

| Name | Definition |
|---|---|
| Episode | one DeepMath query and on-policy trajectory |
| Optimizer step | one persistent AdamW update at the end of an episode |
| Token-chunk microstep | one contiguous block of at most 128 response tokens used to materialize the vocabulary projection and KL |

For the unguarded main runs, 64 episodes equal 64 optimizer updates. Figures
whose x-axis reaches several thousand use cumulative token-chunk microsteps:

\[
s_e=\sum_{i=1}^{e}\left\lceil\frac{T_i}{128}\right\rceil.
\]

The Qwen3-8B endpoints are 1,932 chunks for Privilege-SD-64 and 3,413 chunks
for TRSD-64. Trajectory-end markers identify optimizer updates on this expanded
axis. Raw curves use a five-trajectory moving mean where indicated. Aligned
dynamics use causal eight-update means or prefix-cumulative means; their shaded
bands are causal eight-update local standard errors, not multi-seed confidence
intervals.

## 18. Hardware and software

Primary Qwen3-8B training and each primary math-evaluation shard use one NVIDIA
H100 80GB HBM3. The recorded training environment is:

```text
Conda environment  TTT
Python             3.11.15
PyTorch            2.9.1+cu126
CUDA runtime       12.6
Transformers       4.57.6
PEFT               0.19.1
vLLM               0.12.0
math-verify        0.9.0
```

The historical Qwen3-8B Privilege-SD artifacts record PyTorch `2.9.1+cu128`
and CUDA runtime 12.8. Their device is also an H100 80GB. Gradient
checkpointing is enabled with `use_reentrant=False`, and `use_cache=False`
during all reported training runs.

## 19. Artifact and implementation map

| Item | Repository location |
|---|---|
| Student, privileged prompt, projection, drift partition | [`src/clean_self_distill/persistent.py`](../src/clean_self_distill/persistent.py) |
| Full-vocabulary streamed KL | [`src/clean_self_distill/streaming_distill.py`](../src/clean_self_distill/streaming_distill.py) |
| Ordinary and evaluation prompts | [`src/clean_self_distill/generation.py`](../src/clean_self_distill/generation.py) |
| Data preparation and firewall | [`scripts/clean_self_distill/prepare_empirical_data.py`](../scripts/clean_self_distill/prepare_empirical_data.py) |
| Held-out generation and scoring | [`scripts/clean_self_distill/05_heldout_eval.py`](../scripts/clean_self_distill/05_heldout_eval.py) |
| Wrapper mechanism probe | [`src/clean_self_distill/trust_region_mechanism.py`](../src/clean_self_distill/trust_region_mechanism.py) |
| AP-JSD | [`scripts/clean_self_distill/17_fixed_prefix_policy_drift.py`](../scripts/clean_self_distill/17_fixed_prefix_policy_drift.py) |
| SRPO, OPSD, GRPO | [`baselines/`](../baselines) |
| Logic generation and verification | [`scripts/clean_self_distill/15_logic_eval.py`](../scripts/clean_self_distill/15_logic_eval.py), [`src/clean_self_distill/logic_evaluation.py`](../src/clean_self_distill/logic_evaluation.py) |
| Completed Qwen3-8B run manifests | [`docs/experiments/trsd_table_report_20260808/evidence/training/`](experiments/trsd_table_report_20260808/evidence/training) |
| Expanded math and logical validation | [`docs/experiments/expanded_validation_20260809/`](experiments/expanded_validation_20260809) |
| Drift summaries | [`docs/results/drift_metrics_side_by_side.md`](results/drift_metrics_side_by_side.md) |

## 20. Compact paper-ready disclosure

The primary experiments initialize a persistent LoRA student from the pinned
Qwen3-8B revision above and train on the first 16 or 64 examples of a frozen,
conflict-filtered DeepMath level-7--10 stream. At each episode the student
samples one ordinary-context trajectory. The same current model then scores
the exact realized prefixes under a teacher-only, answer-free instruction to
decompose into subgoals, track constraints and invariants, check boundary
cases, and independently verify the route. TRSD geometrically interpolates the
ordinary and privileged distributions with one trajectory-level coefficient,
chosen by six-step bisection to satisfy mean
\(D_{\mathrm{KL}}(q\Vert p)\le0.004\), and minimizes exact full-vocabulary
reverse KL to the detached projected target. Training uses BF16 SDPA, LoRA
rank 8 and alpha 16, AdamW at `2e-5`, no weight decay, gradient clipping at
1.0, and one optimizer update per trajectory. Math evaluation uses one
10,240-token sample per AMC23/AIME24/AIME25 query with temperature 0.6,
top-p 0.95, top-k 20, paired query seeds, and deterministic boxed-answer
verification. Logical transfer is evaluated greedily on all SATQuest and
LogicSkills items with PySAT/Z3-backed verification. The lexical drift metric
is the realized-token absolute target-to-student log-probability shift over the
fixed vocabulary and regex in Section 10; AP-JSD independently measures
full-vocabulary policy movement on 512 identical ordinary-context anchors.
