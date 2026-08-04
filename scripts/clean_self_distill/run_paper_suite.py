#!/usr/bin/env python3
"""Run the real experiment matrix needed by the CSD paper figures/tables.

This orchestrator never fabricates results. It executes proposal, CSD-T, and
per-query CSD-SD jobs into a deterministic directory layout consumed by
``analyze_paper_suite.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PROPOSE = REPO_ROOT / "scripts/clean_self_distill/01_propose.py"
TRAIN_EVAL = REPO_ROOT / "scripts/clean_self_distill/03_train_eval.py"

RUN_TASKS = {
    "supports",
    "main",
    "budget",
    "hindsight",
    "transfer",
    "sensitivity",
    "ood",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def slug_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def apply_overrides(command: list[str], overrides: Iterable[str]) -> list[str]:
    """Replace existing CLI flags instead of relying on last-value wins."""
    result = list(command)
    values = list(overrides)
    index = 0
    while index < len(values):
        flag = values[index]
        if not flag.startswith("--"):
            raise ValueError(f"Expected an override flag, got {flag!r}")
        value = None
        if index + 1 < len(values) and not values[index + 1].startswith("--"):
            value = values[index + 1]
            index += 1
        while flag in result:
            position = result.index(flag)
            del result[position]
            if position < len(result) and not result[position].startswith("--"):
                del result[position]
        result.append(flag)
        if value is not None:
            result.append(value)
        index += 1
    return result


class SuiteRunner:
    def __init__(self, config: dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.python = args.python
        self.output_root = (REPO_ROOT / config["output_root"]).resolve()
        self.method = dict(config["method"])
        self.eval_data = (REPO_ROOT / config["eval_data"]).resolve()
        self.dtype = str(config.get("dtype", "bfloat16"))
        self.device_map = str(config.get("device_map", "auto"))
        self.executed = 0
        self.skipped = 0

    def run(
        self, command: list[str], *, output_dir: Path, kind: str, force: bool = False
    ) -> None:
        summary_path = output_dir / "summary.json"
        proposal_path = output_dir if output_dir.suffix == ".jsonl" else None
        completion_path = proposal_path or summary_path
        if completion_path.exists() and not (force or self.args.force):
            print(f"SKIP {kind}: {completion_path}")
            self.skipped += 1
            return

        printable = shlex.join(command)
        print(f"RUN  {kind}: {printable}")
        if self.args.dry_run:
            return
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        record = {
            "kind": kind,
            "command": command,
            "started_unix": started,
            "status": "running",
        }
        metadata_path = output_dir.parent / f".{output_dir.name}.job.json"
        write_json(metadata_path, record)
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            record.update(
                status="failed",
                returncode=exc.returncode,
                elapsed_seconds=time.time() - started,
            )
            write_json(metadata_path, record)
            raise
        record.update(
            status="complete", returncode=0, elapsed_seconds=time.time() - started
        )
        write_json(metadata_path, record)
        self.executed += 1

    def proposals(
        self, model_key: str, model_path: str, seed: int, data_path: Path | None = None
    ) -> Path:
        dataset_key = data_path.stem if data_path is not None else "headline"
        path = (
            self.output_root
            / "supports"
            / dataset_key
            / model_key
            / f"seed_{seed}"
            / "proposals.jsonl"
        )
        command = [
            self.python,
            str(PROPOSE),
            "--input",
            str(data_path or self.eval_data),
            "--output",
            str(path),
            "--model",
            model_path,
            "--num-candidates",
            str(self.method["num_candidates"]),
            "--proposal-oversample",
            str(self.method["proposal_oversample"]),
            "--max-rounds",
            str(self.method["proposal_rounds"]),
            "--temperature",
            str(self.method["proposer_temperature"]),
            "--solver-temperature",
            str(self.method["solver_temperature"]),
            "--verifier-temperature",
            str(self.method["verifier_temperature"]),
            "--stage-max-attempts",
            str(self.method["stage_max_attempts"]),
            "--dtype",
            self.dtype,
            "--device-map",
            self.device_map,
            "--seed",
            str(seed),
        ]
        if self.args.max_eval_samples is not None:
            command += ["--max-samples", str(self.args.max_eval_samples)]
        self.run(
            command,
            output_dir=path,
            kind=f"supports/{dataset_key}/{model_key}/seed{seed}",
        )
        return path

    def common_eval_command(
        self,
        *,
        mode: str,
        model_path: str,
        data_path: Path,
        proposals: Path,
        output_dir: Path,
        seed: int,
        overrides: Iterable[str] = (),
    ) -> list[str]:
        command = [
            self.python,
            str(TRAIN_EVAL),
            "--mode",
            mode,
            "--eval-data",
            str(data_path),
            "--proposals",
            str(proposals),
            "--model",
            model_path,
            "--output-dir",
            str(output_dir),
            "--dtype",
            self.dtype,
            "--device-map",
            self.device_map,
            "--ridge-lambda",
            str(self.method["ridge_lambda"]),
            "--residual-step-size",
            str(self.method["residual_step_size"]),
            "--max-support-tokens",
            str(self.method["max_support_tokens"]),
            "--max-tokens-per-candidate",
            str(self.method["max_tokens_per_candidate"]),
            "--hard-negatives",
            str(self.method["hard_negatives"]),
            "--reasoning-token-weight",
            str(self.method["reasoning_token_weight"]),
            "--answer-token-weight",
            str(self.method["answer_token_weight"]),
            "--frontier-positive-weight",
            str(self.method["frontier_positive_weight"]),
            "--frontier-negative-weight",
            str(self.method["frontier_negative_weight"]),
            "--frontier-max-tokens",
            str(self.method["frontier_max_tokens"]),
            "--frontier-negative-probability-floor",
            str(self.method["frontier_negative_probability_floor"]),
            "--max-update-norm",
            str(self.method["max_update_norm"]),
            "--lora-rank",
            str(self.method["lora_rank"]),
            "--lora-alpha",
            str(self.method["lora_alpha"]),
            "--distillation-steps",
            str(self.method["distillation_steps"]),
            "--learning-rate",
            str(self.method["learning_rate"]),
            "--baseline-steps",
            str(self.method["support_lora_steps"]),
            "--baseline-learning-rate",
            str(self.method["support_lora_learning_rate"]),
            "--train-temperature",
            str(self.method["train_temperature"]),
            "--eval-temperature",
            str(self.method["eval_temperature"]),
            "--eval-samples",
            str(self.method["eval_samples"]),
            "--seed",
            str(seed),
        ]
        if self.args.max_eval_samples is not None:
            command += ["--max-eval-samples", str(self.args.max_eval_samples)]
        return apply_overrides(command, overrides)

    def run_main(
        self, model_key: str, model_path: str, seed: int, proposals: Path
    ) -> None:
        for mode, label in (("task1", "csd_t"), ("task2", "csd_sd")):
            output = self.output_root / "main" / model_key / f"seed_{seed}" / label
            overrides = ["--privileged-control"] if mode == "task1" else []
            command = self.common_eval_command(
                mode=mode,
                model_path=model_path,
                data_path=self.eval_data,
                proposals=proposals,
                output_dir=output,
                seed=seed,
                overrides=overrides,
            )
            self.run(
                command, output_dir=output, kind=f"main/{model_key}/seed{seed}/{label}"
            )
        icl_output = (
            self.output_root / "main" / model_key / f"seed_{seed}" / "support_icl"
        )
        icl_command = self.common_eval_command(
            mode="support_icl",
            model_path=model_path,
            data_path=self.eval_data,
            proposals=proposals,
            output_dir=icl_output,
            seed=seed,
        )
        self.run(
            icl_command,
            output_dir=icl_output,
            kind=f"main/{model_key}/seed{seed}/support_icl",
        )
        for mode, label in (("head_sgd", "head_sgd"), ("support_lora", "support_lora")):
            output = self.output_root / "main" / model_key / f"seed_{seed}" / label
            baseline_overrides = (
                [
                    "--baseline-steps",
                    str(self.method["head_sgd_steps"]),
                    "--baseline-learning-rate",
                    str(self.method["head_sgd_learning_rate"]),
                ]
                if mode == "head_sgd"
                else []
            )
            command = self.common_eval_command(
                mode=mode,
                model_path=model_path,
                data_path=self.eval_data,
                proposals=proposals,
                output_dir=output,
                seed=seed,
                overrides=baseline_overrides,
            )
            self.run(
                command, output_dir=output, kind=f"main/{model_key}/seed{seed}/{label}"
            )
        sc_output = (
            self.output_root / "main" / model_key / f"seed_{seed}" / "self_consistency"
        )
        sc_command = self.common_eval_command(
            mode="task1",
            model_path=model_path,
            data_path=self.eval_data,
            proposals=proposals,
            output_dir=sc_output,
            seed=seed,
            overrides=[
                "--eval-samples",
                "8",
                "--eval-temperature",
                str(self.method["sampling_temperature"]),
            ],
        )
        self.run(
            sc_command,
            output_dir=sc_output,
            kind=f"main/{model_key}/seed{seed}/self_consistency",
        )

    def run_budget(
        self, model_key: str, model_path: str, seed: int, proposals: Path
    ) -> None:
        for samples in self.config["sweeps"]["budget_samples"]:
            output = (
                self.output_root
                / "budget"
                / model_key
                / f"seed_{seed}"
                / f"samples_{samples}"
            )
            command = self.common_eval_command(
                mode="task1",
                model_path=model_path,
                data_path=self.eval_data,
                proposals=proposals,
                output_dir=output,
                seed=seed,
                overrides=[
                    "--eval-samples",
                    str(samples),
                    "--eval-temperature",
                    str(self.method["sampling_temperature"]),
                ],
            )
            self.run(
                command,
                output_dir=output,
                kind=f"budget/{model_key}/seed{seed}/n{samples}",
            )

    def run_hindsight(
        self, model_key: str, model_path: str, seed: int, proposals: Path
    ) -> None:
        output = self.output_root / "hindsight" / model_key / f"seed_{seed}"
        command = self.common_eval_command(
            mode="task1",
            model_path=model_path,
            data_path=self.eval_data,
            proposals=proposals,
            output_dir=output,
            seed=seed,
            overrides=["--privileged-control"],
        )
        self.run(command, output_dir=output, kind=f"hindsight/{model_key}/seed{seed}")

    def run_transfer(
        self, model_key: str, model_path: str, seed: int, proposals: Path
    ) -> None:
        for steps in self.config["sweeps"]["distillation_steps"]:
            output = (
                self.output_root
                / "transfer"
                / model_key
                / f"seed_{seed}"
                / f"steps_{steps}"
            )
            command = self.common_eval_command(
                mode="task2",
                model_path=model_path,
                data_path=self.eval_data,
                proposals=proposals,
                output_dir=output,
                seed=seed,
                overrides=["--distillation-steps", str(steps)],
            )
            self.run(
                command,
                output_dir=output,
                kind=f"transfer/{model_key}/seed{seed}/steps{steps}",
            )

    def run_sensitivity(
        self, model_key: str, model_path: str, seed: int, proposals: Path
    ) -> None:
        sweep_specs = [
            ("ridge_lambda", "--ridge-lambda", self.config["sweeps"]["ridge_lambda"]),
            (
                "support_tokens",
                "--max-support-tokens",
                self.config["sweeps"]["support_tokens"],
            ),
            (
                "support_count",
                "--num-specialization-candidates",
                self.config["sweeps"]["support_count"],
            ),
        ]
        for sweep_name, flag, values in sweep_specs:
            for value in values:
                value_slug = slug_float(float(value))
                output = (
                    self.output_root
                    / "sensitivity"
                    / model_key
                    / f"seed_{seed}"
                    / sweep_name
                    / value_slug
                )
                command = self.common_eval_command(
                    mode="task1",
                    model_path=model_path,
                    data_path=self.eval_data,
                    proposals=proposals,
                    output_dir=output,
                    seed=seed,
                    overrides=[flag, str(value)],
                )
                self.run(
                    command,
                    output_dir=output,
                    kind=f"sensitivity/{model_key}/seed{seed}/{sweep_name}/{value}",
                )

    def run_ood(self, model_key: str, model_path: str, seed: int) -> None:
        for dataset_key, relative_path in (
            self.config.get("ood_datasets") or {}
        ).items():
            data_path = (REPO_ROOT / relative_path).resolve()
            proposals = self.proposals(model_key, model_path, seed, data_path=data_path)
            for mode, label in (
                ("task1", "csd_t"),
                ("task2", "csd_sd"),
                ("support_icl", "support_icl"),
                ("support_lora", "support_lora"),
            ):
                output = (
                    self.output_root
                    / "ood"
                    / dataset_key
                    / model_key
                    / f"seed_{seed}"
                    / label
                )
                command = self.common_eval_command(
                    mode=mode,
                    model_path=model_path,
                    data_path=data_path,
                    proposals=proposals,
                    output_dir=output,
                    seed=seed,
                )
                self.run(
                    command,
                    output_dir=output,
                    kind=f"ood/{dataset_key}/{model_key}/{label}",
                )


def parse_csv_filter(value: str | None, available: Iterable[str]) -> list[str]:
    available_list = list(available)
    if not value:
        return available_list
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(requested) - set(available_list)
    if unknown:
        raise ValueError(
            f"Unknown selection {sorted(unknown)}; available={available_list}"
        )
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/clean_self_distill/paper_suite.yaml"),
    )
    parser.add_argument(
        "--tasks", help=f"Comma-separated subset of {sorted(RUN_TASKS)}"
    )
    parser.add_argument("--models", help="Comma-separated model keys from the YAML")
    parser.add_argument("--seeds", help="Comma-separated integer seeds")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-eval-samples", type=int, help="Smoke-test item cap")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    tasks = parse_csv_filter(args.tasks, sorted(RUN_TASKS))
    model_keys = parse_csv_filter(args.models, config["models"].keys())
    seeds = (
        [int(value) for value in args.seeds.split(",")]
        if args.seeds
        else [int(value) for value in config["seeds"]]
    )
    runner = SuiteRunner(config, args)
    if (
        not args.dry_run
        and not runner.eval_data.exists()
        and any(task != "ood" for task in tasks)
    ):
        raise FileNotFoundError(
            f"Missing headline eval data: {runner.eval_data}. Run scripts/download_data.py first."
        )

    for model_key in model_keys:
        model_path = str(config["models"][model_key])
        for seed in seeds:
            proposals = (
                runner.proposals(model_key, model_path, seed)
                if set(tasks) - {"ood"}
                else None
            )
            if "main" in tasks:
                runner.run_main(model_key, model_path, seed, proposals)
            if "budget" in tasks:
                runner.run_budget(model_key, model_path, seed, proposals)
            if "hindsight" in tasks:
                runner.run_hindsight(model_key, model_path, seed, proposals)
            if "transfer" in tasks:
                runner.run_transfer(model_key, model_path, seed, proposals)
            if "sensitivity" in tasks:
                runner.run_sensitivity(model_key, model_path, seed, proposals)
            if "ood" in tasks:
                runner.run_ood(model_key, model_path, seed)

    print(
        json.dumps(
            {
                "executed": runner.executed,
                "skipped": runner.skipped,
                "dry_run": args.dry_run,
            }
        )
    )


if __name__ == "__main__":
    main()
