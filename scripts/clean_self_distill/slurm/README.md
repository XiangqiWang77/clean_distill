# Multi-B200 PoC launcher

This launcher runs one pinned `Qwen/Qwen3-4B` experiment as a dependency chain:

1. a CPU prefetch job downloads the AMC23+AIME24+AIME25 parquet and the pinned
   model revision into the shared da839 scratch tree;
2. a 2--8 task `gpu_b200` array uses exactly one B200 per task;
3. each shard resumes proposal JSONL, then runs Task 1 (Base, Privileged, CSD-T)
   and Task 2 (CSD-SD), each with an 8192-token evaluation budget;
4. an `afterok` CPU job validates every shard and invokes `report_poc.py`.

No download root may reach 9.9 GB. `HF_HUB_DISABLE_XET=1` is set during
prefetch; GPU workers are offline and share one HF cache. Run outputs are under
`/home/da839/scratch_pi_mg269/da839/clean_distill/runs/<RUN_ID>` with distinct
logs, status files, markers, proposals, Task 1 outputs, and Task 2 outputs.

Validate without submitting:

```bash
bash -n scripts/clean_self_distill/slurm/{submit_b200_poc.sh,prefetch_assets.slurm,run_shard.slurm,merge_report.slurm}
python scripts/clean_self_distill/slurm/launcher_support.py --help
RUN_ID=csd-smoke-plan DRY_RUN=1 \
  bash scripts/clean_self_distill/slurm/submit_b200_poc.sh
```

Submit the recommended smoke chain (2 B200s, 4 prefix records, 4 candidates):

```bash
RUN_PROFILE=smoke RUN_ID=csd-qwen3-4b-smoke-01 \
  bash scripts/clean_self_distill/slurm/submit_b200_poc.sh
```

Optionally warm the shared model/data cache first with a CPU-only job. Assets
are shared, so use a separate run ID and then submit the smoke chain normally:

```bash
RUN_PROFILE=smoke RUN_ID=csd-assets-qwen3-4b PREFETCH_ONLY=1 \
  bash scripts/clean_self_distill/slurm/submit_b200_poc.sh
```

Submit the full chain (8 B200s, all 143 records, 10 candidates):

```bash
RUN_PROFILE=full RUN_ID=csd-qwen3-4b-full-01 \
  bash scripts/clean_self_distill/slurm/submit_b200_poc.sh
```

To batch the full run behind a successful smoke report without launching a
second concurrent cache writer, pass the smoke report job as an upstream gate:

```bash
UPSTREAM_AFTEROK_JOB_ID=<SMOKE_REPORT_JOB_ID> \
RUN_PROFILE=full RUN_ID=csd-qwen3-4b-full-01 \
  bash scripts/clean_self_distill/slurm/submit_b200_poc.sh
```

The full B200 array then starts only after the smoke reporter has validated the
model, environment, artifacts, protocol, and authoritative scoring.

`NUM_SHARDS` (2--8), `MAX_EVAL_SAMPLES`, `NUM_CANDIDATES`, and
`GPU_WALLTIME` override profile defaults. `CPU_PARTITION` is configurable if
the site's CPU partition is not `day`. Reuse the exact `RUN_ID` with
`RESUBMIT=1` to resume: completed stage markers are skipped, a partial final
proposal record is archived/repaired, and incomplete evaluation stages rerun
the whole shard. The run config is immutable, so a changed configuration needs
a new `RUN_ID`.

The prefetch job is normally submitted automatically. Use `PREFETCH_ONLY=1`
with a distinct run ID to warm the shared cache without submitting GPU work.
Do not run model downloads concurrently against this dedicated cache.
