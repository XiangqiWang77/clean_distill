# Multi-B200 PoC launcher

This launcher runs one pinned `Qwen/Qwen3-4B` experiment as a dependency chain:

1. a CPU prefetch job downloads the AMC23+AIME24+AIME25 parquet and the pinned
   model revision into the shared da839 scratch tree;
2. a two-task `gpu_b200` array uses exactly one B200 per task;
3. each shard resumes proposal JSONL, then runs Task 1 (Base, Privileged, CSD-T)
   and Task 2 (CSD-SD), each with an 8192-token evaluation budget;
4. an `afterok` CPU job validates every shard and invokes `report_poc.py`.

The model, data, and scratch Python layer together may not reach 9.9 GB.
`HF_HUB_DISABLE_XET=1` is set during prefetch; GPU workers are offline and
share one HF cache. Run outputs are under
`/home/da839/scratch_pi_mg269/da839/clean_distill/runs/<RUN_ID>` with distinct
logs, status files, markers, proposals, Task 1 outputs, and Task 2 outputs.

Prepare the B200-compatible Python layer in da839 scratch. TTT remains the
activated Conda environment, while the site-provided CUDA 12.8 PyTorch module
supplies the `sm_100` build without downloading another torch wheel:

```bash
bash scripts/clean_self_distill/slurm/prepare_b200_env.sh
```

Validate without submitting:

```bash
bash -n scripts/clean_self_distill/slurm/{prepare_b200_env.sh,submit_b200_poc.sh,prefetch_assets.slurm,run_shard.slurm,merge_report.slurm}
python scripts/clean_self_distill/slurm/launcher_support.py --help
RUN_ID=csd-smoke-plan DRY_RUN=1 \
  bash scripts/clean_self_distill/slurm/submit_b200_poc.sh
```

Optionally warm the shared model/data cache first with a CPU-only job. Assets
are shared, so use a separate run ID and then submit the formal chain normally:

```bash
RUN_PROFILE=smoke RUN_ID=csd-assets-qwen3-4b PREFETCH_ONLY=1 \
  bash scripts/clean_self_distill/slurm/submit_b200_poc.sh
```

Submit the full chain (2 B200s, all 143 records, 10 candidates, 6-hour slice):

```bash
RUN_PROFILE=full RUN_ID=csd-qwen3-4b-full-01 \
  bash scripts/clean_self_distill/slurm/submit_b200_poc.sh
```

When the pinned model, dataset, and environment have already been validated,
skip another prefetch queue and submit the restart-safe formal chain directly:

```bash
ASSETS_READY=1 RESUBMIT=1 RUN_PROFILE=full RUN_ID=csd-qwen3-4b-full-01 \
  bash scripts/clean_self_distill/slurm/submit_b200_poc.sh
```

`ASSETS_READY=1` performs local offline manifest, dataset, dependency, and
combined-size validation before `sbatch`. Each formal array element then runs a
real CUDA kernel and requires capability `(10, 0)` before loading the model.

`MAX_EVAL_SAMPLES`, `NUM_CANDIDATES`, and `GPU_WALLTIME` override profile
defaults; `NUM_SHARDS` is fixed at two. `CPU_PARTITION` is configurable if
the site's CPU partition is not `day`. Reuse the exact `RUN_ID` with
`RESUBMIT=1` to resume: completed stage markers are skipped, a partial final
proposal record is archived/repaired, and incomplete evaluation stages rerun
the whole shard. The run config is immutable, so a changed configuration needs
a new `RUN_ID`.

The prefetch job is normally submitted automatically. Use `PREFETCH_ONLY=1`
with a distinct run ID to warm the shared cache without submitting GPU work.
Do not run model downloads concurrently against this dedicated cache.
