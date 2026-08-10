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
  --gres=gpu:h100:4 \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q17,CSD_SRPO_EXPECTED_GPUS=4" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_qwen_train.slurm")
q8_train=$(sbatch --parsable "${start_args[@]}" \
  --gres=gpu:h100:4 \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q8,CSD_SRPO_EXPECTED_GPUS=4" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_qwen_train.slurm")
go_train=$(sbatch --parsable "${start_args[@]}" \
  --gres=gpu:h100:4 \
  --export="$common_export,CSD_SRPO_EXPECTED_GPUS=4" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_gptoss_train.slurm")

q17_eval=$(sbatch --parsable --dependency="afterok:$q17_train" \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q17" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_qwen_eval.slurm")
q8_eval=$(sbatch --parsable --dependency="afterok:$q8_train" \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q8" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_qwen_eval.slurm")
go_eval=$(sbatch --parsable --dependency="afterok:$go_train" \
  --export="$common_export" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_gptoss_eval.slurm")

q17_score=$(sbatch --parsable --dependency="afterok:$q17_eval" \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q17" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_score.slurm")
q8_score=$(sbatch --parsable --dependency="afterok:$q8_eval" \
  --export="$common_export,CSD_SRPO_MODEL_KEY=q8" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_score.slurm")
go_score=$(sbatch --parsable --dependency="afterok:$go_eval" \
  --export="$common_export,CSD_SRPO_MODEL_KEY=go20" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_score.slurm")
final_report=$(sbatch --parsable \
  --dependency="afterok:$q17_score:$q8_score:$go_score" \
  --export="$common_export" \
  "$CSD_SRPO_SNAPSHOT_ROOT/baselines/slurm/srpo_final_report.slurm")

jq -n \
  --arg submitted_at "$(date -Is)" \
  --arg start_dependency "$start_dependency" \
  --arg q17_train "$q17_train" --arg q8_train "$q8_train" --arg go_train "$go_train" \
  --arg q17_eval "$q17_eval" --arg q8_eval "$q8_eval" --arg go_eval "$go_eval" \
  --arg q17_score "$q17_score" --arg q8_score "$q8_score" --arg go_score "$go_score" \
  --arg final_report "$final_report" \
  '{schema_version:"srpo-three-model-submission-v1", submitted_at:$submitted_at,
    scope:{train:"DeepMath-64", eval:"AMC23+AIME24+AIME25 (143 questions)", logical_eval:false,
           checkpoints:[16,64], global_h100_cap:12, train_h100:{q17:4,q8:4,go20:4},
           eval_peak_h100:12, gpu_smoke_test:false},
    start_dependency:$start_dependency,
    jobs:{q17_train:$q17_train,q8_train:$q8_train,go20_train:$go_train,
          q17_eval:$q17_eval,q8_eval:$q8_eval,go20_eval:$go_eval,
          q17_score:$q17_score,q8_score:$q8_score,go20_score:$go_score,
          final_report:$final_report}}' \
  > "$CSD_SRPO_RUN_ROOT/SUBMISSION.json"

printf 'q17_train=%s\nq8_train=%s\ngo20_train=%s\nq17_eval=%s\nq8_eval=%s\ngo20_eval=%s\nq17_score=%s\nq8_score=%s\ngo20_score=%s\nfinal_report=%s\n' \
  "$q17_train" "$q8_train" "$go_train" "$q17_eval" "$q8_eval" "$go_eval" \
  "$q17_score" "$q8_score" "$go_score" "$final_report"
