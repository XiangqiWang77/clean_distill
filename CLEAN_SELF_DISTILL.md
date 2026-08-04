# Clean Self-Distillation implementation

This directory adds a standalone Clean Self-Distillation path on top of the
RLCSD repository. It reads the same JSON/parquet schemas as verl, reuses the
repository's AIME/AMC answer grader, and keeps the existing RLCSD/verl methods
unchanged.

The runtime pipeline is:

```text
target query -> sanitized skill card -> proposed candidates
             -> verified correct + independently generated wrong trajectories
             -> verify the first wrong step and its corrective action
             -> frontier-weighted closed-form ridge LM-head specialization
             -> temporary teacher scores the student's exact on-policy prefix
             -> top-k forward-KL update of the persistent student
```

There are no mock exams, no fixed `8 Fit + 2 Check` split, and no runtime check
gate. By default ten candidates are proposed and verified, and all ten are used
for specialization.

## Install

Use the repository environment:

```bash
pip install -r requirements.txt
```

The three entry scripts should be launched from the repository root.

## Paper tasks: one-command runs

The paper separates the two claims instead of conflating temporary-teacher
quality with successful write-back:

- **Task 1 / CSD-T:** construct the query-local ridge teacher and decode with
  it directly. No persistent student parameters are trained.
- **Task 2 / CSD-SD:** construct the same teacher, train a query-local rank-8
  student LoRA for three same-prefix KL steps, destroy the teacher, and decode
  with the student. Student LoRA state is reset to the identical base
  checkpoint before every AMC/AIME item.

Download the verl-format benchmark once:

```bash
python scripts/download_data.py --dataset amc23+aime24+aime25 --split val
```

Run Task 1:

```bash
bash scripts/clean_self_distill/train_task1_fast_teacher.sh
```

Run Task 2:

```bash
bash scripts/clean_self_distill/train_task2_clean_distillation.sh
```

The wrappers use the corrective-v5 protocol: ten proposed/verified candidates,
all accepted candidates used for specialization, configured support budgets of
768 tokens in total and 96 tokens per candidate, ridge `lambda=0.1`, residual
scale `eta=0.8`, float32 Cholesky, proposer/solver temperatures `0.8/0.3`, and
greedy Acc@1 decoding. Ridge row weights are `0.25` for ordinary reasoning, `1`
for answer tokens, `8` for the verified corrective action, and `8` for explicit
suppression of the verified wrong action; the LM-head update norm is capped at
`2`. On Task 2, the existing rank-8 LoRA receives three same-prefix AdamW steps
at learning rate `2e-5`. There is no Fit/Check split or gate.

Common overrides are environment variables:

```bash
MODEL_PATH=Qwen/Qwen3-4B \
EVAL_DATA=/data/amc_aime.parquet \
RUN_SEED=2 \
OUTPUT_ROOT=/checkpoints/csd/task2/seed_2 \
bash scripts/clean_self_distill/train_task2_clean_distillation.sh
```

The current single-model PoC pins Qwen3-4B because its approximately 8.06 GB
asset footprint fits the requested 10 GB download ceiling. The restart-safe
multi-B200 commands are in `scripts/clean_self_distill/slurm/README.md`.

For the paper's five synthesis seeds:

```bash
for seed in 0 1 2 3 4; do
  RUN_SEED=$seed \
  OUTPUT_ROOT=outputs/clean_self_distill/task1_fast_teacher/seed_$seed \
  bash scripts/clean_self_distill/train_task1_fast_teacher.sh
done

for seed in 0 1 2 3 4; do
  RUN_SEED=$seed \
  OUTPUT_ROOT=outputs/clean_self_distill/task2_clean_distillation/seed_$seed \
  bash scripts/clean_self_distill/train_task2_clean_distillation.sh
done
```

For a smoke test, set `MAX_EVAL_SAMPLES=2`. Set `FORCE_PROPOSE=1` to regenerate
the seed-specific support sets; Task 1 additionally accepts
`FORCE_SPECIALIZE=1` to rebuild cached ridge adapters and
`PRIVILEGED_CONTROL=0` to skip the evaluation-only hindsight upper bound.

Important outputs:

```text
Task 1: <OUTPUT_ROOT>/evaluation/metrics_task1_fast_teacher.json
        <OUTPUT_ROOT>/evaluation/eval_task1_fast_teacher.jsonl
        <OUTPUT_ROOT>/evaluation/summary.json
Task 2: <OUTPUT_ROOT>/evaluation/metrics_task2_clean_distillation.json
        <OUTPUT_ROOT>/evaluation/eval_task2_clean_distillation.jsonl
        <OUTPUT_ROOT>/evaluation/summary.json
```

## Full paper suite: Tables 1--9 and Figures 2--9

Task 1 and Task 2 are only the two method outputs. The full paper additionally
requires multi-model/seed main runs, support hygiene, compute controls,
hindsight interventions, transfer curves, sensitivity, efficiency scaling,
and OOD evaluation. These are declared in
`configs/clean_self_distill/paper_suite.yaml` and run by one orchestrator.
The exact RQ-to-artifact map and metric definitions are in
`PAPER_EXPERIMENTS.md`.

Preview every command without launching GPUs:

```bash
python scripts/clean_self_distill/run_paper_suite.py --dry-run
```

Run the complete real suite:

```bash
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,main,budget,hindsight,transfer,sensitivity,ood
```

The expensive default is four backbones times five seeds. Useful subsets:

```bash
# Two-item smoke test on Qwen3-4B, seed 0
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,main,transfer,sensitivity \
  --models qwen3_4b \
  --seeds 0 \
  --max-eval-samples 2

# Main table only, all models and seeds
python scripts/clean_self_distill/run_paper_suite.py --tasks supports,main

# Fig. 5 compute frontier
python scripts/clean_self_distill/run_paper_suite.py --tasks supports,budget

# Fig. 6 hindsight audit and Fig. 8 write-back curve
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,hindsight,transfer

# Fig. 9 lambda/token/support/step sensitivity
python scripts/clean_self_distill/run_paper_suite.py \
  --tasks supports,sensitivity,transfer
```

After the GPU jobs finish, generate CSV tables and vector PDF figures from the
real JSONL logs:

```bash
python scripts/clean_self_distill/analyze_paper_suite.py
```

Outputs are written under
`outputs/clean_self_distill/paper_suite/analysis/{tables,figures}`. Figure 1 is
the method pipeline drawn in the LaTeX source; the analysis script produces
Figures 2--9. The corrected Figure 7 is specialization-reliability calibration
without a runtime gate. Figure 9 sweeps ridge lambda, support-token cap,
support count, and distillation steps—no Fit/Check allocation.

The main suite also runs two query-local iterative baselines on the exact same
verified candidates and the same configured 768-token budget: 10-step LM-head SGD and
3-step rank-8 Support LoRA, each with an independently declared learning rate.
The privileged job builds a target-specific correct reasoning/CoT advantage
using the answer in a private construction prompt, then removes the final
answer and literal-equivalent spellings before the evaluated model sees the
context. Redaction is fail-closed and bound to provenance hashes. Because the
construction ancestry uses the target answer, this control is labeled `HER=1`
and `HFS=0`; the separate evaluated-context audit must report that no literal
target answer remains. This control is evaluation-only and never enters CSD
teacher construction.

OOD jobs are activated by adding local parquet paths under `ood_datasets` in
the YAML. The analyzer never inserts hypothetical numbers: missing jobs produce
empty CSVs or omit the corresponding figure.

## 1. Skill-card candidate proposal

Run proposal separately on the train and evaluation parquets. The input may be
JSONL, JSON, or a verl parquet. The proposer loader deliberately projects each
row down to `(query_id, problem, source)` before loading the model; target
answers and reference solutions do not enter this stage.

```bash
python scripts/clean_self_distill/01_propose.py \
  --input data/verl/deepmath_filtered_level6_8/train.parquet \
  --output outputs/csd/qwen3_4b/train_proposals.jsonl \
  --model Qwen/Qwen3-4B \
  --num-candidates 10

python scripts/clean_self_distill/01_propose.py \
  --input data/verl/amc23+aime24+aime25/val.parquet \
  --output outputs/csd/qwen3_4b/eval_proposals.jsonl \
  --model Qwen/Qwen3-4B \
  --num-candidates 10
```

The candidate proposer sees only the sanitized skill card. For every surrogate
problem, the v5 pipeline then (1) solves and independently verifies a correct
trajectory, (2) makes a separate model call that sees the surrogate problem
and sanitized failure modes, but not the verified solution, to generate a
wrong trajectory, and (3) asks the verifier to certify the first invalid step
and a local corrective action. An accepted candidate therefore contains
`correct_trajectory`, `final_answer`, `wrong_trajectory`,
`wrong_final_answer`, and an `error_frontier` with `wrong_step_index`,
`wrong_step_text`, `error_explanation`, `corrective_action`, and verifier-valid
flags. Each artifact passes the target-disjoint firewall, and each row stores
an auditable context-provenance manifest and lexical diagnostics.

## 2. Ridge-regression specialization

This optional materialization step builds checkpoint-specific temporary
adapters for inspection or evaluation:

```bash
python scripts/clean_self_distill/02_specialize.py \
  --proposals outputs/csd/qwen3_4b/eval_proposals.jsonl \
  --output-dir outputs/csd/qwen3_4b/eval_adapters \
  --model Qwen/Qwen3-4B \
  --ridge-lambda 0.1 \
  --residual-step-size 0.8 \
  --max-support-tokens 768 \
  --max-tokens-per-candidate 96 \
  --hard-negatives 8 \
  --reasoning-token-weight 0.25 \
  --answer-token-weight 1.0 \
  --frontier-positive-weight 8.0 \
  --frontier-negative-weight 8.0 \
  --frontier-max-tokens 24 \
  --frontier-negative-probability-floor 0.25 \
  --max-update-norm 2.0
```

Let `H` contain frozen final-layer hidden states for four supervision row
types: correct-trajectory reasoning, its answer tokens, the correct next action
at the verified error frontier, and the wrong action at that same frontier.
The desired sparse residual `R` boosts correct targets and explicitly reverses
direction for wrong-action rows, lowering their logits while redistributing
mass to alternatives. The row weights are:

| Supervision row | Weight | Direction |
|---|---:|---|
| Ordinary correct reasoning | `0.25` | boost |
| Correct answer | `1` | boost |
| Frontier corrective action | `8` | boost |
| Frontier wrong action | `8` | suppress |

Thus both frontier directions are weighted `32x` more heavily than ordinary
reasoning. The implementation solves

```text
C = (H H^T + lambda I)^-1 R
delta_logits(h) = (h H^T) C
```

It does not materialize a `hidden_size x vocabulary_size` matrix. The saved
support states and coefficients are exactly the corresponding low-rank
LM-head update for the selected vocabulary columns, capped at update norm `2`.
Total adaptation time includes the candidate feature forwards and the
closed-form solve; both are also logged separately.

## 3. Low-level same-prefix training and evaluation entry

The recommended paper commands are the Task 1/Task 2 wrappers above. The
following `train-eval` mode is a separate streaming/global-training ablation:
it lets student updates persist across training examples and therefore is not
the paper's isolated per-query reset protocol. LoRA is used by default; pass
`--full-finetune` only for that streaming ablation.

```bash
python scripts/clean_self_distill/03_train_eval.py \
  --mode train-eval \
  --train-data data/verl/deepmath_filtered_level6_8/train.parquet \
  --eval-data data/verl/amc23+aime24+aime25/val.parquet \
  --proposals outputs/csd/qwen3_4b/train_proposals.jsonl \
  --proposals outputs/csd/qwen3_4b/eval_proposals.jsonl \
  --model Qwen/Qwen3-4B \
  --output-dir outputs/csd/qwen3_4b/train_eval \
  --eval-samples 12 \
  --privileged-control
```

For cached-teacher evaluation only:

```bash
python scripts/clean_self_distill/03_train_eval.py \
  --mode eval \
  --eval-data data/verl/amc23+aime24+aime25/val.parquet \
  --proposals outputs/csd/qwen3_4b/eval_proposals.jsonl \
  --adapter-dir outputs/csd/qwen3_4b/eval_adapters \
  --model Qwen/Qwen3-4B \
  --output-dir outputs/csd/qwen3_4b/cached_eval \
  --eval-samples 12
```

The student rollout is generated before teacher scoring. Student and teacher
logits are computed on the exact same `original query + student-generated
prefix`; the teacher differs only by the query-local top-layer update.

The immediate v5 revision deliberately leaves this same-prefix distillation
objective unchanged. It does not introduce a last-layer Jacobian ridge update
or delta-selective distillation; those are outside the implemented scope being
evaluated here.

## Logged metrics

Standard benchmark metrics:

- `accuracy/base`, `accuracy/temporary_teacher`: mean@N on AMC/AIME.
- `accuracy/*_pass_at_n`: pass@N, reported separately from mean@N.
- Before/after-distillation summaries allow the persistent student gain to be
  computed without conflating it with the temporary teacher.

Teacher-strength and fast-adaptation metrics:

- `teacher/target_answer_nll_gain`: student NLL minus clean-teacher NLL on an
  offline target-answer continuation. The answer is an evaluation label, not
  an adapter or teacher-context input.
- `teacher/specialization_success_rate`: fraction of queries with positive
  target-answer NLL transfer.
- `speed/mean_specialization_seconds`: feature extraction plus ridge solve.
- `speed/mean_feature_extraction_seconds`,
  `speed/mean_closed_form_solve_seconds`, and the solve-time fraction separate
  model-forward cost from the actual closed-form linear algebra.
- **FATE** (`speed/fast_adaptation_teacher_efficiency`): hindsight-free target
  NLL gain divided by specialization seconds.
- `speed/fast_adaptation_accuracy_efficiency`: clean teacher accuracy points
  gained per specialization second.

Hindsight-specific metrics:

- **HER** (`hindsight/hindsight_exposure_rate`): fraction of teacher events
  whose provenance contains target answer, target solution, target verifier
  feedback, or future target tokens. Clean Self-Distillation should be `0`.
- **CPP** (`hindsight/context_parity_rate`): fraction of teacher/student
  comparisons with exactly identical tokenized contexts.
- **OP-SPR** (`hindsight/on_policy_same_prefix_rate`): same-prefix rate on
  student-generated trajectories specifically. It should be `1` during
  training.
- `hindsight/causal_scoring_rate`: fraction of teacher events using causal
  next-token scoring.
- **CAR** (`hindsight/clean_advantage_retention`): clean CSD-T accuracy gain
  retained relative to the answer-redacted privileged-CoT control.
- **HFTG** (`hindsight/hindsight_free_transfer_gain`):
  `(1 - HER) * CPP * target_answer_nll_gain`. It gives zero credit to a gain
  obtained with hindsight exposure or mismatched contexts.
- With `--privileged-control`, the evaluator also reports privileged accuracy,
  the **Hindsight Privilege Gap**, and the fraction of privileged gain recovered
  by the clean teacher. Its private correct-CoT construction is
  answer-conditioned, so it is always `HER=1` and `HFS=0`; the evaluated
  context separately certifies
  `privileged_context_contains_literal_target_answer=false`. The privileged
  branch is an evaluation-only baseline and never contributes training
  targets.

## Why this is not registered as a stock verl policy loss

The data files and grading are verl-compatible, but arbitrary per-query
closed-form LM-head updates are not a stock vLLM/verl adapter type: every query
has a different dense low-rank function over hidden states. This implementation
therefore uses Hugging Face forwards for the clean teacher and on-policy
distillation while leaving the existing Ray/vLLM trainers intact. A production
verl port should add the sparse ridge update inside the rollout worker before
registering a `clean_self_distill` policy loss; treating it as an ordinary LoRA
adapter would change the method.
