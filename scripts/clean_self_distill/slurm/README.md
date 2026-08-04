# Multi-B200 PoC launcher

This launcher runs one pinned `Qwen/Qwen3-4B` experiment as a dependency chain:

1. a CPU prefetch job downloads the AMC23+AIME24+AIME25 parquet and the pinned
   model revision into the shared da839 scratch tree;
2. a 16-task `scavenge_gpu` array requests one typed B200 per task and uses
   `%2` throttling, so no more than two B200s run concurrently;
3. each shard resumes proposal JSONL, then runs Task 1 (Base, Privileged, CSD-T)
   and Task 2 (CSD-SD); Task 2 uses three independent 4096-token same-prefix
   rollouts and every method has an 8192-token evaluation budget;
4. an `afterok` CPU job validates every shard, invokes `report_poc.py`, and
   derives the pre-registered short/long/hindsight tables post hoc.

This task's new model and dataset downloads together may not reach 9.9 GB.
The pre-existing read-only CUDA 12.8 overlay at
`/home/da839/scratch_pi_mg269/da839/mfspd/pydeps-cu128` is reused and excluded
from that task-download budget. `HF_HUB_DISABLE_XET=1` is set during prefetch;
GPU workers are offline and share one HF cache. Run outputs are under
`/home/da839/scratch_pi_mg269/da839/clean_distill/runs/<RUN_ID>` with distinct
logs, status files, markers, proposals, Task 1 outputs, and Task 2 outputs.

Validate the existing B200-compatible overlay before submitting. This script
does not create an environment, install packages, or download anything: it
runs the real `/home/da839/.conda/envs/TTT/bin/python`, places the overlay first
on `PYTHONPATH`, and checks `math_verify`, `peft`, `pyarrow`, `torch`, and
`transformers` imports plus the CUDA 12.8/`sm_100` build provenance:

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

Submit the full chain (16 restartable shards, at most 2 concurrent B200s, all
143 records, 8 accepted-candidate targets, 3 proposal rounds, 4096-token
same-prefix distillation, and a 3-hour restartable slice):

```bash
RUN_PROFILE=full RUN_ID=csd-qwen3-4b-full-01 \
  bash scripts/clean_self_distill/slurm/submit_b200_poc.sh
```

When the pinned model, dataset, TTT interpreter, and overlay have been validated,
skip another prefetch queue and submit the restart-safe formal chain directly:

```bash
ASSETS_READY=1 RESUBMIT=1 RUN_PROFILE=full RUN_ID=csd-qwen3-4b-full-01 \
  bash scripts/clean_self_distill/slurm/submit_b200_poc.sh
```

`ASSETS_READY=1` performs local offline manifest, dataset, dependency-overlay,
and task-download-size validation before `sbatch`. Each formal array element
then asserts the real TTT executable, `torch==2.9.1+cu128`, CUDA 12.8,
`sm_100`, and the overlay module path; it also runs a real CUDA kernel and
requires B200 capability `(10, 0)` before loading the model. Runtime manifests
record the executable, torch module path/architecture flags, overlay,
hostname, and Slurm array job/task identifiers. Submission requires a clean
worktree, pins its 40-character Git commit in the immutable run config, and
both GPU and report jobs fail closed if queued code later drifts from it.

`MAX_EVAL_SAMPLES`, `NUM_CANDIDATES`, and `GPU_WALLTIME` override profile
defaults. The full profile fixes `NUM_SHARDS=16` and submits array `0-15%2`;
the smoke profile keeps two shards. Each full array task requests exactly one
typed B200 for three hours. `CPU_PARTITION` is configurable if
the site's CPU partition is not `day`. Reuse the exact `RUN_ID` with
`RESUBMIT=1` to resume: completed stage markers are skipped, a partial final
proposal record is archived/repaired, and Task 1/Task 2 resume from their last
atomically committed, fully bound query row. The run config is immutable, so a
changed configuration needs a new `RUN_ID`.

The report treats the maximum contiguous prefix as `H_train`; it never sums
independent rollouts. It also reports immediate CSD-T/Privileged performance,
teacher-free CSD-SD retention, Base-only output-length strata split at 2048
tokens, and HER/CPP/HFS/HFAG. Long-prefix evidence fails closed unless the
formal 4096-token opportunity is present for every ready query and at least ten
held-out AIME queries enter the post-2048 diagnostic window.

`scavenge_gpu` uses Slurm `PreemptMode=REQUEUE`. The GPU launcher is submitted
with `--requeue`; completed per-shard stage markers, repaired proposal JSONL,
and validated per-query evaluation prefixes allow a preempted task to continue
on a later allocation without discarding finished queries. A watchdog also
self-requeues each task ten minutes before its three-hour limit, so proposal
rows and completed stages resume automatically instead of turning a TIMEOUT
into a failed dependency chain. The `%2` throttle remains the hard
experiment-level cap on simultaneous B200 use across requeues.

The prefetch job is normally submitted automatically. Use `PREFETCH_ONLY=1`
with a distinct run ID to warm the shared cache without submitting GPU work.
Do not run model downloads concurrently against this dedicated cache.
