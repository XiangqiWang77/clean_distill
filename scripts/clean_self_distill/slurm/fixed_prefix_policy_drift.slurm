#!/usr/bin/env bash
#SBATCH --job-name=trsd-policy-jsd
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/da839/clean_distill/slurm-%j.out

set -euo pipefail

REPO=/home/da839/clean_distill
SCRATCH=/home/da839/scratch_pi_mg269/da839/clean_distill
MODEL="$SCRATCH/models/Qwen3-8B-b968826d9c46dd6066d109eabc6255188de91218"
RUN="$SCRATCH/runs/baseline-deepmath16-20260808-03"
P64="$SCRATCH/runs/csd-qwen3-8b-three-sellpoints-poc-07/timebox12h/privileged/checkpoints/episode_0064"
T64="$SCRATCH/runs/reverse-kl-matched64-20260807/trsd/train/checkpoints/episode_0064"
OUT="$SCRATCH/runs/policy-drift-20260809"

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

/home/da839/.conda/envs/TTT/bin/python \
  scripts/clean_self_distill/17_fixed_prefix_policy_drift.py \
  --model "$MODEL" \
  --queries "$SCRATCH/runs/csd-qwen3-8b-three-sellpoints-poc-07/prepared/heldout_queries.jsonl" \
  --base-prediction "$RUN/eval/base_vllm/predictions/shard_00.jsonl" \
  --base-prediction "$RUN/eval/base_vllm/predictions/shard_01.jsonl" \
  --base-prediction "$RUN/eval/base_vllm/predictions/shard_02.jsonl" \
  --base-prediction "$RUN/eval/base_vllm/predictions/shard_03.jsonl" \
  --method "Privilege-SD 64=$P64" \
  --method "TRSD 64=$T64" \
  --max-queries 32 \
  --response-tokens 512 \
  --stride 32 \
  --bootstrap-draws 5000 \
  --output-dir "$OUT"

cp "$OUT/anchored_policy_jsd.png" "$REPO/docs/figures/anchored_policy_jsd.png"
cp "$OUT/anchored_policy_jsd.pdf" "$REPO/docs/figures/anchored_policy_jsd.pdf"
cp "$OUT/anchored_policy_jsd_per_query.csv" "$REPO/docs/figures/anchored_policy_jsd_per_query.csv"
cp "$OUT/anchored_policy_jsd_summary.csv" "$REPO/docs/figures/anchored_policy_jsd_summary.csv"
cp "$OUT/anchored_policy_jsd_manifest.json" "$REPO/docs/figures/anchored_policy_jsd_manifest.json"

python scripts/clean_self_distill/18_drift_metric_table.py
