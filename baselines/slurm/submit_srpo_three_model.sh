#!/usr/bin/env bash
set -Eeuo pipefail

config=${CSD_SRPO_CONFIG:-/home/da839/clean_distill/configs/clean_self_distill/srpo_three_model.env}
# shellcheck disable=SC1090
source "$config"
start_dependency=${CSD_SRPO_START_DEPENDENCY:-}

mkdir -p "$CSD_SRPO_RUN_ROOT/logs" "$CSD_SRPO_SNAPSHOT_ROOT"
[[ "$(wc -l < "$CSD_SRPO_TRAIN_QUERIES")" == 64 ]]
[[ "$(wc -l < "$CSD_SRPO_HELDOUT_QUERIES")" == 143 ]]
[[ -s "$CSD_SRPO_TRAIN_LABELS" ]]
[[ -s "$CSD_SRPO_HELDOUT_LABELS" ]]
for path in "$CSD_SRPO_Q17_MODEL" "$CSD_SRPO_Q8_MODEL" "$CSD_SRPO_GO_MODEL"; do
  [[ -s "$path/config.json" ]]
done
for path in "$CSD_SRPO_Q17_BASE" "$CSD_SRPO_Q8_BASE" "$CSD_SRPO_GO_BASE"; do
  [[ "$(wc -l < "$path")" == 143 ]]
done

cd "$CSD_SRPO_CODE_ROOT"
python -m py_compile \
  baselines/objectives.py \
  baselines/train.py \
  baselines/build_comparison_table.py \
  scripts/clean_self_distill/05_heldout_eval.py \
  scripts/clean_self_distill/27_cross_model_math_table.py
for script in \
  baselines/slurm/srpo_qwen_train.slurm \
  baselines/slurm/srpo_gptoss_train.slurm \
  baselines/slurm/srpo_dispatch_eval.slurm \
  baselines/slurm/srpo_qwen_eval.slurm \
  baselines/slurm/srpo_gptoss_eval.slurm \
  baselines/slurm/srpo_score.slurm \
  baselines/slurm/srpo_final_report.slurm; do
  bash -n "$script"
done
if rg -n -i 'logic|satquest|logicskills' \
  baselines/slurm/srpo_qwen_train.slurm \
  baselines/slurm/srpo_gptoss_train.slurm \
  baselines/slurm/srpo_qwen_eval.slurm \
  baselines/slurm/srpo_gptoss_eval.slurm; then
  printf 'SRPO launchers unexpectedly reference a logical dataset\n' >&2
  exit 2
fi

rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  "$CSD_SRPO_CODE_ROOT/baselines/" "$CSD_SRPO_SNAPSHOT_ROOT/baselines/"
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  "$CSD_SRPO_CODE_ROOT/src/" "$CSD_SRPO_SNAPSHOT_ROOT/src/"
mkdir -p "$CSD_SRPO_SNAPSHOT_ROOT/scripts/clean_self_distill"
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  "$CSD_SRPO_CODE_ROOT/scripts/clean_self_distill/" \
  "$CSD_SRPO_SNAPSHOT_ROOT/scripts/clean_self_distill/"

common_export="ALL,CSD_SRPO_CONFIG=$config"
start_args=()
[[ -z "$start_dependency" ]] || start_args=(--dependency="afterany:$start_dependency")
q17_train=$(sbatch --parsable "${start_args[@]}" \
  --gres=gpu:h100:2 \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q17,CSD_SRPO_EXPECTED_GPUS=2" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_qwen_train.slurm")
q8_train=$(sbatch --parsable "${start_args[@]}" \
  --gres=gpu:h100:2 \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q8,CSD_SRPO_EXPECTED_GPUS=2" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_qwen_train.slurm")
go_train=$(sbatch --parsable "${start_args[@]}" \
  --gres=gpu:h100:2 \
  --export="$common_export,CSD_SRPO_EXPECTED_GPUS=2" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_gptoss_train.slurm")

dispatcher=$(sbatch --parsable \
  --export="$common_export,CSD_SRPO_Q17_TRAIN_JOB=$q17_train,CSD_SRPO_Q8_TRAIN_JOB=$q8_train,CSD_SRPO_GO_TRAIN_JOB=$go_train" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_dispatch_eval.slurm")

jq -n \
  --arg submitted_at "$(date -Is)" \
  --arg start_dependency "$start_dependency" \
  --arg q17_train "$q17_train" --arg q8_train "$q8_train" --arg go_train "$go_train" \
  --arg dispatcher "$dispatcher" \
  '{schema_version:"srpo-three-model-submission-v1", submitted_at:$submitted_at,
    scope:{train:"DeepMath-64", eval:"AMC23+AIME24+AIME25 (143 questions)", logical_eval:false,
           checkpoints:[16,64], global_h100_cap:12, train_h100:{q17:2,q8:2,go20:2},
           train_rollouts_per_query:4, train_rollout_cap:2048,
           eval_max_tokens:10240, eval_peak_h100:12, pipelined_eval:true,
           gpu_smoke_test:false},
    start_dependency:$start_dependency,
    jobs:{q17_train:$q17_train,q8_train:$q8_train,go20_train:$go_train,
          dispatcher:$dispatcher}}' \
  > "$CSD_SRPO_RUN_ROOT/SUBMISSION.json"

printf 'q17_train=%s\nq8_train=%s\ngo20_train=%s\ndispatcher=%s\n' \
  "$q17_train" "$q8_train" "$go_train" "$dispatcher"
