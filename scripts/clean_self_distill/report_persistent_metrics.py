#!/usr/bin/env python3
"""Build the preregistered persistent Clean Self-Distillation report.

The reporter is deliberately fail-closed.  It consumes only already-scored
artifacts, reconstructs every headline metric from raw per-sample or
per-trajectory observations, and refuses incomplete checkpoint curves,
unpaired decoding seeds, invalid cleanliness counters, or unmatched ablations.

Input contracts
---------------
``--heldout-scored``
    One or more JSONL files produced by ``05_heldout_eval.py score``.  The
    methods are ``base``, ``clean_sd``, and ``privileged_sd``.  Base is scored
    at episode 0; the persistent branches are scored at 250/500/750/1000 (an
    optional episode-0 branch score must be byte-for-byte behaviorally equal to
    Base).  Each cell contains four ``profile=mean4`` rows and the sample-0
    ``profile=acc1`` alias for every query.

``--short-term-scored``
    JSONL with the same four-sample/Acc@1 alias representation and methods
    ``base``, ``privileged_sd``, ``csd_t``, and ``csd_sd``.  Every sample also
    carries an explicit ``adaptation_seconds`` (or the preserved
    ``proposal_end_to_end_seconds`` + ``specialization_metrics`` components)
    and a ``cleanliness_audit``/``training_audit`` object with raw HER/CP
    numerators and denominators.

``--mechanism``
    JSONL containing ``trajectory`` and ``frontier`` records for each of
    ``pre_decision_privilege``, ``post_outcome_privilege``, and
    ``clean_teacher``.  Trajectories contain raw log-prob sums and versioned
    task/style absolute-error sums; frontiers contain Base/Teacher margins.

``--ablation``
    JSONL with one nested raw record per query and variant
    (``correct_only`` and ``correct_wrong_signed``).  Matching is checked per
    query, including actual support tokens—not merely configured budgets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "clean-self-distill-persistent-report-v1"
PSR_PARTITION_VERSION = "rlcsd-style-task-v1"
CHECKPOINTS = (0, 250, 500, 750, 1000)
SAMPLE_INDICES = (0, 1, 2, 3)
HELDOUT_METHODS = ("base", "clean_sd", "privileged_sd")
SHORT_METHODS = ("base", "privileged_sd", "csd_t", "csd_sd")
TEACHER_TYPES = (
    "pre_decision_privilege",
    "post_outcome_privilege",
    "clean_teacher",
)
ABLATION_VARIANTS = ("correct_only", "correct_wrong_signed")
DEFAULT_SOURCE_COUNTS = {"amc23": 83, "aime24": 30, "aime25": 30}
EPISODE_SCHEMA_VERSION = "clean-self-distill-persistent-episode-v1"
CHECKPOINT_SCHEMA_VERSION = "clean-self-distill-persistent-checkpoint-v1"
PERSISTENT_BRANCHES = ("clean", "privileged")
AUDIT_KEYS = (
    "teacher_positions",
    "hindsight_exposed_positions",
    "compared_positions",
    "exact_context_positions",
    "on_policy_positions",
)


class PersistentReportError(ValueError):
    """Raised when a formal empirical artifact violates the protocol."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise PersistentReportError(f"{context}.{key} is required")
    return mapping[key]


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PersistentReportError(f"{context} must be an object")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PersistentReportError(f"{context} must be a non-empty string")
    return value.strip()


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PersistentReportError(f"{context} must be an integer >= {minimum}")
    return value


def _integral_count(value: Any, context: str) -> int:
    """Accept JSON integer counts serialized as either ``3`` or ``3.0``."""
    number = _number(value, context, minimum=0.0)
    if not number.is_integer():
        raise PersistentReportError(f"{context} must be an integral count")
    return int(number)


def _number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PersistentReportError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise PersistentReportError(f"{context} is outside its finite numeric range")
    return result


def _binary(value: Any, context: str) -> int:
    number = _number(value, context)
    if number not in (0.0, 1.0):
        raise PersistentReportError(f"{context} must be binary")
    return int(number)


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise PersistentReportError(f"{context} must be boolean")
    return value


def _load_jsonl(paths: Sequence[str | Path], context: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, raw in enumerate(handle, 1):
                    if not raw.strip():
                        raise PersistentReportError(f"{path}:{line_number} is blank")
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise PersistentReportError(
                            f"{path}:{line_number} must be a JSON object"
                        )
                    rows.append(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PersistentReportError(f"Cannot read {context} artifact {path}: {exc}") from exc
    if not rows:
        raise PersistentReportError(f"{context} artifact is empty")
    return rows


def _load_json_objects(paths: Sequence[str | Path], context: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PersistentReportError(f"Cannot read {context} artifact {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise PersistentReportError(f"{context} artifact {path} must be a JSON object")
        values.append(value)
    if not values:
        raise PersistentReportError(f"{context} artifact is empty")
    return values


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _artifact_digest(paths: Sequence[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(item) for item in paths):
        digest.update(str(path.resolve()).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_source_universe(
    query_sources: Mapping[str, str],
    expected_source_counts: Mapping[str, int],
    context: str,
) -> None:
    counts = Counter(query_sources.values())
    if dict(counts) != dict(expected_source_counts):
        raise PersistentReportError(
            f"{context} source counts {dict(counts)} != expected {dict(expected_source_counts)}"
        )


def _validate_profile_aliases(
    rows: Sequence[Mapping[str, Any]],
    *,
    cell_fields: Sequence[str],
    context: str,
) -> list[dict[str, Any]]:
    """Return canonical mean4 rows after validating sample-0 Acc@1 aliases."""
    mean_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    acc_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for index, raw_row in enumerate(rows, 1):
        row = dict(raw_row)
        row_context = f"{context} row {index}"
        profile = _string(_required(row, "profile", row_context), f"{row_context}.profile")
        if profile not in {"mean4", "acc1"}:
            raise PersistentReportError(f"{row_context}.profile is not mean4/acc1")
        query_id = _string(_required(row, "query_id", row_context), f"{row_context}.query_id")
        source = _string(_required(row, "source", row_context), f"{row_context}.source").casefold()
        sample_index = _integer(
            _required(row, "sample_index", row_context),
            f"{row_context}.sample_index",
        )
        if sample_index not in SAMPLE_INDICES:
            raise PersistentReportError(f"{row_context}.sample_index must be 0..3")
        _binary(_required(row, "correct", row_context), f"{row_context}.correct")
        _integer(_required(row, "seed", row_context), f"{row_context}.seed")
        cell = tuple(_required(row, field, row_context) for field in cell_fields)
        key = (*cell, query_id, sample_index)
        row["query_id"] = query_id
        row["source"] = source
        target = mean_rows if profile == "mean4" else acc_rows
        if key in target:
            raise PersistentReportError(f"{context} duplicates {profile} key {key!r}")
        if profile == "acc1" and sample_index != 0:
            raise PersistentReportError(f"{row_context} Acc@1 alias must be sample 0")
        target[key] = row

    if set(acc_rows) != {key for key in mean_rows if key[-1] == 0}:
        raise PersistentReportError(f"{context} has incomplete or extra Acc@1 aliases")
    for key, alias in acc_rows.items():
        original = mean_rows[key]
        for field in ("correct", "seed", "response", "parsed_answer", "problem_sha256"):
            if field in alias or field in original:
                if alias.get(field) != original.get(field):
                    raise PersistentReportError(
                        f"{context} Acc@1 alias disagrees with mean4 for {key!r} field {field}"
                    )
    return list(mean_rows.values())


def _profile_rows(rows: Sequence[Mapping[str, Any]], profile: str) -> list[Mapping[str, Any]]:
    if profile == "acc1":
        return [row for row in rows if int(row["sample_index"]) == 0]
    if profile == "mean4":
        return list(rows)
    raise AssertionError(profile)


def _accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise PersistentReportError("Cannot compute accuracy from zero observations")
    return fmean(_binary(row["correct"], "correct") for row in rows)


def _scope_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    sources = sorted({str(row["source"]) for row in rows})
    return {
        "combined": list(rows),
        **{source: [row for row in rows if row["source"] == source] for source in sources},
    }


def _paired_flips(
    base_rows: Sequence[Mapping[str, Any]], method_rows: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    def index(values: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], int]:
        return {
            (str(row["query_id"]), int(row["sample_index"])): _binary(row["correct"], "correct")
            for row in values
        }

    base = index(base_rows)
    method = index(method_rows)
    if set(base) != set(method):
        raise PersistentReportError("Paired flip rows have different query/sample coverage")
    return {
        "wrong_to_correct": sum(base[key] == 0 and method[key] == 1 for key in base),
        "correct_to_wrong": sum(base[key] == 1 and method[key] == 0 for key in base),
        "paired_positions": len(base),
    }


def _validate_paired_seeds(
    rows_by_cell: Mapping[tuple[Any, ...], Sequence[Mapping[str, Any]]], context: str
) -> None:
    expected: dict[tuple[str, int], int] = {}
    for cell, rows in rows_by_cell.items():
        for row in rows:
            key = (str(row["query_id"]), int(row["sample_index"]))
            seed = _integer(row["seed"], f"{context} {cell} seed")
            if key in expected and expected[key] != seed:
                raise PersistentReportError(
                    f"{context} is not paired: seed mismatch for {key!r} in {cell!r}"
                )
            expected[key] = seed


def _raw_audit(value: Any, context: str) -> dict[str, int]:
    audit = _mapping(value, context)
    result = {
        key: _integer(_required(audit, key, context), f"{context}.{key}")
        for key in AUDIT_KEYS
    }
    if result["hindsight_exposed_positions"] > result["teacher_positions"]:
        raise PersistentReportError(f"{context} hindsight exposure exceeds teacher positions")
    if result["exact_context_positions"] > result["compared_positions"]:
        raise PersistentReportError(f"{context} exact context exceeds compared positions")
    if result["on_policy_positions"] > result["compared_positions"]:
        raise PersistentReportError(f"{context} on-policy positions exceed compared positions")
    return result


def _sum_audits(audits: Iterable[Mapping[str, int]]) -> dict[str, int]:
    totals = {key: 0 for key in AUDIT_KEYS}
    for audit in audits:
        for key in AUDIT_KEYS:
            totals[key] += int(audit[key])
    return totals


def _build_training_audit(
    raw_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove the two curves came from complete persistent 1,000-episode loops."""
    by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw_row in enumerate(raw_rows, 1):
        row = dict(raw_row)
        context = f"training journal row {index}"
        if _string(_required(row, "schema_version", context), f"{context}.schema_version") != EPISODE_SCHEMA_VERSION:
            raise PersistentReportError(f"{context} has an unsupported schema")
        branch = _string(_required(row, "branch", context), f"{context}.branch")
        if branch not in PERSISTENT_BRANCHES:
            raise PersistentReportError(f"{context} has unknown branch {branch!r}")
        by_branch[branch].append(row)
    if set(by_branch) != set(PERSISTENT_BRANCHES):
        raise PersistentReportError("Training journals need Clean and Privileged branches")

    branch_report: dict[str, Any] = {}
    branch_orders: dict[str, list[tuple[str, str, str]]] = {}
    branch_totals: dict[str, dict[str, int]] = {}
    for branch in PERSISTENT_BRANCHES:
        rows = sorted(
            by_branch[branch],
            key=lambda row: _integer(row.get("episode"), f"{branch}.episode", minimum=1),
        )
        episodes = [_integer(row["episode"], f"{branch}.episode", minimum=1) for row in rows]
        if episodes != list(range(1, CHECKPOINTS[-1] + 1)):
            raise PersistentReportError(
                f"{branch} journal must contain exactly episodes 1..{CHECKPOINTS[-1]}"
            )
        identities = {
            (
                _string(row.get("variant"), f"{branch}.variant"),
                _string(row.get("method_id"), f"{branch}.method_id"),
            )
            for row in rows
        }
        if len(identities) != 1:
            raise PersistentReportError(f"{branch} changes variant/method_id mid-stream")
        variant, method_id = next(iter(identities))
        if branch == "clean" and variant != "correct_wrong_signed":
            raise PersistentReportError("Formal Clean branch must use correct_wrong_signed")
        if branch == "privileged" and "predecision" not in method_id.casefold():
            raise PersistentReportError("Persistent Privileged branch must be pre-decision")

        audits: list[dict[str, int]] = []
        style_error = style_tokens = task_error = task_tokens = 0.0
        losses: list[float] = []
        normalized_log_ratios: list[float] = []
        optimizer_step_flags: list[bool] = []
        ridge_eligible = ridge_crossings = regression_eligible = regressions = 0
        order: list[tuple[str, str, str]] = []
        seen_queries: set[str] = set()
        for episode, row in zip(episodes, rows):
            context = f"{branch} episode {episode}"
            if _integer(row.get("stream_index"), f"{context}.stream_index") != episode - 1:
                raise PersistentReportError(f"{context} has noncanonical stream_index")
            query_id = _string(row.get("query_id"), f"{context}.query_id")
            if query_id in seen_queries:
                raise PersistentReportError(f"{branch} repeats distillation query {query_id!r}")
            seen_queries.add(query_id)
            problem_sha = _string(row.get("problem_sha256"), f"{context}.problem_sha256")
            if len(problem_sha) != 64:
                raise PersistentReportError(f"{context}.problem_sha256 must have 64 characters")
            source = _string(row.get("source"), f"{context}.source").casefold()
            order.append((query_id, problem_sha, source))
            response_tokens = _integer(row.get("response_tokens"), f"{context}.response_tokens", minimum=1)
            optimizer_step_flags.append(
                _boolean(row.get("optimizer_step"), f"{context}.optimizer_step")
            )
            losses.append(
                _number(row.get("distillation_loss"), f"{context}.distillation_loss", minimum=0.0)
            )
            teacher_sum = _number(row.get("teacher_logprob_sum"), f"{context}.teacher_logprob_sum")
            student_sum = _number(row.get("student_logprob_sum"), f"{context}.student_logprob_sum")
            teacher_normalized = _number(
                row.get("teacher_normalized_logprob"), f"{context}.teacher_normalized_logprob"
            )
            student_normalized = _number(
                row.get("student_normalized_logprob"), f"{context}.student_normalized_logprob"
            )
            if not math.isclose(teacher_normalized, teacher_sum / response_tokens, abs_tol=1e-8):
                raise PersistentReportError(f"{context} teacher normalized log-prob is inconsistent")
            if not math.isclose(student_normalized, student_sum / response_tokens, abs_tol=1e-8):
                raise PersistentReportError(f"{context} student normalized log-prob is inconsistent")
            normalized_log_ratios.append(teacher_normalized - student_normalized)

            partition = _mapping(row.get("style_task_error"), f"{context}.style_task_error")
            if partition.get("partition_version") != PSR_PARTITION_VERSION:
                raise PersistentReportError(f"{context} uses an unregistered PSR partition")
            style_error += _number(
                partition.get("style_abs_error_sum"), f"{context}.style_abs_error_sum", minimum=0.0
            )
            style_tokens += _integer(
                partition.get("style_token_count"), f"{context}.style_token_count"
            )
            task_error += _number(
                partition.get("task_abs_error_sum"), f"{context}.task_abs_error_sum", minimum=0.0
            )
            task_tokens += _integer(
                partition.get("task_token_count"), f"{context}.task_token_count"
            )
            audit = _raw_audit(row.get("audit"), f"{context}.audit")
            audits.append(audit)
            ridge = _mapping(row.get("ridge_metrics"), f"{context}.ridge_metrics")
            applicable = _boolean(ridge.get("applicable"), f"{context}.ridge_metrics.applicable")
            support_tokens = _integral_count(
                ridge.get("support_tokens"), f"{context}.ridge_metrics.support_tokens"
            )
            eligible = _integral_count(
                ridge.get("db_eligible_count"), f"{context}.ridge_metrics.db_eligible_count"
            )
            crossings = _integral_count(
                ridge.get("db_crossing_count"), f"{context}.ridge_metrics.db_crossing_count"
            )
            regression_denominator = _integral_count(
                ridge.get("regression_eligible_count"),
                f"{context}.ridge_metrics.regression_eligible_count",
            )
            regression_count = _integral_count(
                ridge.get("regression_count"), f"{context}.ridge_metrics.regression_count"
            )
            if crossings > eligible or regression_count > regression_denominator:
                raise PersistentReportError(f"{context} has impossible ridge counts")
            if branch == "privileged" and (
                applicable or support_tokens or eligible or crossings or regression_denominator or regression_count
            ):
                raise PersistentReportError(f"{context} Privileged ridge metrics must be inapplicable zeros")
            ridge_eligible += eligible
            ridge_crossings += crossings
            regression_eligible += regression_denominator
            regressions += regression_count

        completed_optimizer_steps = sum(optimizer_step_flags)
        if completed_optimizer_steps <= 0:
            raise PersistentReportError(f"{branch} never updated the persistent student")
        totals = _sum_audits(audits)
        if totals["teacher_positions"] == 0 or totals["compared_positions"] == 0:
            raise PersistentReportError(f"{branch} has empty persistent audit denominators")
        her = totals["hindsight_exposed_positions"] / totals["teacher_positions"]
        cp = totals["exact_context_positions"] / totals["compared_positions"]
        if branch == "clean" and not (her == 0.0 and cp == 1.0):
            raise PersistentReportError("Persistent Clean branch must have HER=0 and CP=1")
        if branch == "privileged" and not (her == 0.0 and cp == 0.0):
            raise PersistentReportError("Pre-decision Privileged branch must have HER=0 and CP=0")
        if style_tokens == 0 or task_tokens == 0:
            raise PersistentReportError(f"{branch} persistent PSR partitions are empty")
        style_mean = style_error / style_tokens
        task_mean = task_error / task_tokens
        branch_orders[branch] = order
        branch_totals[branch] = totals
        branch_report[branch] = {
            "variant": variant,
            "method_id": method_id,
            "completed_episodes": len(rows),
            "optimizer_steps_completed": completed_optimizer_steps,
            "mean_distillation_loss": fmean(losses),
            "mean_normalized_teacher_student_log_ratio": fmean(normalized_log_ratios),
            "HER": her,
            "CP": cp,
            "audit_totals": totals,
            "PSR": style_mean / task_mean if task_mean else None,
            "style_mean_abs_error": style_mean,
            "task_mean_abs_error": task_mean,
            "ridge_metrics": {
                "db_eligible_count": ridge_eligible,
                "db_crossing_count": ridge_crossings,
                "DBCR": ridge_crossings / ridge_eligible if ridge_eligible else None,
                "regression_eligible_count": regression_eligible,
                "regression_count": regressions,
                "regression_rate": regressions / regression_eligible if regression_eligible else None,
            },
        }
    if branch_orders["clean"] != branch_orders["privileged"]:
        raise PersistentReportError(
            "Clean and Privileged journals do not use the exact same episode order"
        )

    manifests: dict[tuple[str, int], Mapping[str, Any]] = {}
    shared_identities: dict[str, set[str]] = {
        "model_identity_sha256": set(),
        "query_manifest_sha256": set(),
    }
    for index, manifest in enumerate(checkpoint_manifests, 1):
        context = f"checkpoint manifest {index}"
        if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise PersistentReportError(f"{context} has an unsupported schema")
        branch = _string(manifest.get("branch"), f"{context}.branch")
        if branch not in PERSISTENT_BRANCHES:
            raise PersistentReportError(f"{context} has unknown branch")
        episode = _integer(manifest.get("checkpoint_episode"), f"{context}.checkpoint_episode")
        completed = _integer(manifest.get("completed_episodes"), f"{context}.completed_episodes")
        if episode != completed or episode not in CHECKPOINTS:
            raise PersistentReportError(f"{context} is not a scientific checkpoint")
        key = (branch, episode)
        if key in manifests:
            raise PersistentReportError(f"Duplicate checkpoint manifest {key!r}")
        manifests[key] = manifest
        for identity in shared_identities:
            digest = _string(manifest.get(identity), f"{context}.{identity}")
            if len(digest) != 64:
                raise PersistentReportError(f"{context}.{identity} must have 64 characters")
            shared_identities[identity].add(digest)
        _string(manifest.get("proposal_manifest_sha256"), f"{context}.proposal_manifest_sha256")
        _string(manifest.get("config_sha256"), f"{context}.config_sha256")
        cumulative = _raw_audit(manifest.get("cumulative_audit"), f"{context}.cumulative_audit")
        if episode == 0:
            expected_audit = {audit_key: 0 for audit_key in AUDIT_KEYS}
        else:
            rows = sorted(by_branch[branch], key=lambda row: int(row["episode"]))[:episode]
            expected_audit = _sum_audits(
                _raw_audit(row["audit"], f"{branch}.audit") for row in rows
            )
        if cumulative != expected_audit:
            raise PersistentReportError(
                f"{context} cumulative audit does not match journal prefix"
            )
    expected_manifest_keys = {
        (branch, episode) for branch in PERSISTENT_BRANCHES for episode in CHECKPOINTS
    }
    if set(manifests) != expected_manifest_keys:
        raise PersistentReportError(
            "Checkpoint manifests must cover both branches at 0/250/500/750/1000"
        )
    if any(len(values) != 1 for values in shared_identities.values()):
        raise PersistentReportError("Both persistent branches must share model/query identity")
    for branch in PERSISTENT_BRANCHES:
        manifest = manifests[(branch, CHECKPOINTS[-1])]
        if _raw_audit(manifest["cumulative_audit"], f"{branch}.final_audit") != branch_totals[branch]:
            raise PersistentReportError(f"{branch} final checkpoint audit is inconsistent")
    return {
        "status": "passed",
        "persistence_definition": "one non-reset student/optimizer trajectory over 1000 ordered episodes",
        "paired_episode_order": True,
        "scientific_checkpoints": list(CHECKPOINTS),
        "branches": branch_report,
    }


def _build_long_horizon(
    raw_rows: Sequence[Mapping[str, Any]], expected_source_counts: Mapping[str, int]
) -> dict[str, Any]:
    rows = _validate_profile_aliases(
        raw_rows,
        cell_fields=("method", "checkpoint_episode"),
        context="heldout",
    )
    cells: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, 1):
        method = _string(row["method"], f"heldout row {index}.method")
        if method not in HELDOUT_METHODS:
            raise PersistentReportError(f"Unknown heldout method {method!r}")
        episode = _integer(row["checkpoint_episode"], "heldout.checkpoint_episode")
        if episode not in CHECKPOINTS:
            raise PersistentReportError(f"Unexpected scientific checkpoint {episode}")
        _string(_required(row, "checkpoint_sha256", "heldout"), "checkpoint_sha256")
        cells[(method, episode)].append(row)

    if {episode for method, episode in cells if method == "base"} != {0}:
        raise PersistentReportError("Base must be evaluated exactly once at checkpoint 0")
    required_branch = set(CHECKPOINTS[1:])
    optional_zero: set[bool] = set()
    for method in ("clean_sd", "privileged_sd"):
        episodes = {episode for cell_method, episode in cells if cell_method == method}
        if episodes not in (required_branch, set(CHECKPOINTS)):
            raise PersistentReportError(
                f"{method} checkpoints {sorted(episodes)} do not complete {list(CHECKPOINTS)}"
            )
        optional_zero.add(0 in episodes)
    if len(optional_zero) != 1:
        raise PersistentReportError("Clean and Privileged must use the same checkpoint-0 policy")

    base = cells[("base", 0)]
    base_sources = {str(row["query_id"]): str(row["source"]) for row in base}
    if len(base_sources) * 4 != len(base):
        raise PersistentReportError("Base heldout cell does not contain exactly four samples/query")
    _validate_source_universe(base_sources, expected_source_counts, "heldout")
    expected_keys = {(str(row["query_id"]), int(row["sample_index"])) for row in base}
    for cell, values in cells.items():
        keys = {(str(row["query_id"]), int(row["sample_index"])) for row in values}
        if keys != expected_keys:
            raise PersistentReportError(f"Heldout cell {cell!r} has incomplete query/sample coverage")
        if any(base_sources[str(row["query_id"])] != row["source"] for row in values):
            raise PersistentReportError(f"Heldout cell {cell!r} changes query source")
    _validate_paired_seeds(cells, "heldout")

    if True in optional_zero:
        base_by_key = {
            (str(row["query_id"]), int(row["sample_index"])): row for row in base
        }
        for method in ("clean_sd", "privileged_sd"):
            for row in cells[(method, 0)]:
                key = (str(row["query_id"]), int(row["sample_index"]))
                base_row = base_by_key[key]
                for field in ("correct", "response", "parsed_answer"):
                    if row.get(field) != base_row.get(field):
                        raise PersistentReportError(
                            f"{method} checkpoint 0 differs from Base at {key!r} field {field}"
                        )

    report: dict[str, Any] = {}
    for profile in ("acc1", "mean4"):
        base_profile = _profile_rows(base, profile)
        scopes = _scope_rows(base_profile)
        profile_report: dict[str, Any] = {}
        for scope, scoped_base in scopes.items():
            source = None if scope == "combined" else scope
            a0 = _accuracy(scoped_base)
            branches: dict[str, Any] = {}
            curves: dict[str, list[dict[str, Any]]] = {}
            for method in ("clean_sd", "privileged_sd"):
                curve = [{"episode": 0, "accuracy": a0, "gain_vs_a0": 0.0}]
                for episode in CHECKPOINTS[1:]:
                    values = _profile_rows(cells[(method, episode)], profile)
                    if source is not None:
                        values = [row for row in values if row["source"] == source]
                    accuracy = _accuracy(values)
                    curve.append(
                        {
                            "episode": episode,
                            "accuracy": accuracy,
                            "gain_vs_a0": accuracy - a0,
                        }
                    )
                gains = [item["gain_vs_a0"] for item in curve]
                area = sum(
                    (curve[index]["episode"] - curve[index - 1]["episode"])
                    * (gains[index] + gains[index - 1])
                    / 2.0
                    for index in range(1, len(curve))
                ) / float(CHECKPOINTS[-1] - CHECKPOINTS[0])
                branches[method] = {
                    "A_k": curve,
                    "final_long_accuracy": curve[-1]["accuracy"],
                    "LHG": curve[-1]["gain_vs_a0"],
                    "AULC": area,
                }
                curves[method] = curve
            crossover = next(
                (
                    clean["episode"]
                    for clean, privileged in zip(
                        curves["clean_sd"][1:], curves["privileged_sd"][1:]
                    )
                    if clean["accuracy"] > privileged["accuracy"]
                ),
                None,
            )
            profile_report[scope] = {
                "n_queries": len({str(row["query_id"]) for row in scoped_base}),
                "A_0": a0,
                "branches": branches,
                "clean_privilege_crossover_K_star": crossover,
                "clean_privilege_crossover_display": (
                    str(crossover) if crossover is not None else "N/A"
                ),
            }
        report[profile] = profile_report
    return report


def _cleanliness(row: Mapping[str, Any], context: str) -> tuple[int, int, int, int]:
    audit_value = row.get("cleanliness_audit", row.get("training_audit"))
    audit = _mapping(
        audit_value, f"{context}.cleanliness_audit"
    )
    exposed = _integer(
        _required(audit, "hindsight_exposed_positions", context),
        f"{context}.hindsight_exposed_positions",
    )
    teacher = _integer(
        _required(audit, "teacher_positions", context), f"{context}.teacher_positions"
    )
    exact = _integer(
        _required(audit, "exact_context_positions", context),
        f"{context}.exact_context_positions",
    )
    compared = _integer(
        _required(audit, "compared_positions", context), f"{context}.compared_positions"
    )
    if exposed > teacher or exact > compared:
        raise PersistentReportError(f"{context} has impossible cleanliness counts")
    return exposed, teacher, exact, compared


def _adaptation_seconds(row: Mapping[str, Any], context: str) -> float:
    """Read one query-level latency while tolerating scorer-preserved components."""
    if "adaptation_seconds" in row:
        explicit = _number(
            row["adaptation_seconds"], f"{context}.adaptation_seconds", minimum=0.0
        )
        return explicit
    specialization = row.get("specialization_metrics", {})
    if specialization is None:
        specialization = {}
    metrics = _mapping(specialization, f"{context}.specialization_metrics")
    proposal = _number(
        row.get("proposal_end_to_end_seconds", 0.0),
        f"{context}.proposal_end_to_end_seconds",
        minimum=0.0,
    )
    if "total_adaptation_seconds" in metrics:
        return _number(
            metrics["total_adaptation_seconds"],
            f"{context}.specialization_metrics.total_adaptation_seconds",
            minimum=0.0,
        )
    if "specialization_seconds" in metrics:
        return proposal + _number(
            metrics["specialization_seconds"],
            f"{context}.specialization_metrics.specialization_seconds",
            minimum=0.0,
        )
    if metrics or proposal != 0.0:
        raise PersistentReportError(
            f"{context} carries incomplete specialization latency components"
        )
    return 0.0


def _build_short_term(
    raw_rows: Sequence[Mapping[str, Any]], expected_source_counts: Mapping[str, int]
) -> dict[str, Any]:
    rows = _validate_profile_aliases(
        raw_rows, cell_fields=("method",), context="short-term"
    )
    cells: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows, 1):
        method = _string(row["method"], f"short-term row {index}.method")
        if method not in SHORT_METHODS:
            raise PersistentReportError(f"Unknown short-term method {method!r}")
        _adaptation_seconds(row, f"short-term row {index}")
        _cleanliness(row, f"short-term row {index}")
        cells[method].append(row)
    if set(cells) != set(SHORT_METHODS):
        raise PersistentReportError(
            f"Short-term artifact methods {sorted(cells)} != {list(SHORT_METHODS)}"
        )
    base = cells["base"]
    query_sources = {str(row["query_id"]): str(row["source"]) for row in base}
    if len(query_sources) * 4 != len(base):
        raise PersistentReportError("Short-term Base does not have four samples/query")
    _validate_source_universe(query_sources, expected_source_counts, "short-term")
    expected_keys = {(str(row["query_id"]), int(row["sample_index"])) for row in base}
    for method, values in cells.items():
        if {(str(row["query_id"]), int(row["sample_index"])) for row in values} != expected_keys:
            raise PersistentReportError(f"Short-term {method} has incomplete paired coverage")
    _validate_paired_seeds({(method,): values for method, values in cells.items()}, "short-term")

    for method, values in cells.items():
        for query_id in query_sources:
            query_rows = [row for row in values if row["query_id"] == query_id]
            seconds = {
                _adaptation_seconds(row, "adaptation_seconds")
                for row in query_rows
            }
            if len(seconds) != 1:
                raise PersistentReportError(
                    f"Short-term {method}/{query_id} repeats inconsistent adaptation time"
                )

    for row in cells["base"]:
        if _cleanliness(row, "base") != (0, 0, 0, 0):
            raise PersistentReportError("Base cleanliness counters must all be zero")
        if _adaptation_seconds(row, "base") != 0.0:
            raise PersistentReportError("Base adaptation_seconds must be zero")
    for method in SHORT_METHODS[1:]:
        totals = [sum(values) for values in zip(*(_cleanliness(row, method) for row in cells[method]))]
        if totals[1] == 0 or totals[3] == 0:
            raise PersistentReportError(f"{method} has zero HER/CP denominator")
        if method in {"csd_t", "csd_sd"} and not (
            totals[0] == 0 and totals[2] == totals[3]
        ):
            raise PersistentReportError(
                f"{method} must demonstrate HER=0 and CP=1 from raw positions"
            )
        if method == "privileged_sd" and totals[2] != 0:
            raise PersistentReportError("Privileged SD must demonstrate CP=0")
        if method == "privileged_sd" and totals[0] != 0:
            raise PersistentReportError(
                "Pre-decision Privileged SD must demonstrate HER=0"
            )

    report: dict[str, Any] = {}
    for profile in ("acc1", "mean4"):
        base_profile = _profile_rows(base, profile)
        profile_report: dict[str, Any] = {}
        for scope, scoped_base in _scope_rows(base_profile).items():
            source = None if scope == "combined" else scope
            scoped_cells = {
                method: [
                    row
                    for row in _profile_rows(values, profile)
                    if source is None or row["source"] == source
                ]
                for method, values in cells.items()
            }
            base_accuracy = _accuracy(scoped_cells["base"])
            methods: dict[str, Any] = {}
            for method, values in scoped_cells.items():
                accuracy = _accuracy(values)
                query_seconds: dict[str, float] = {}
                for row in values:
                    query_seconds[str(row["query_id"])] = _adaptation_seconds(
                        row, "adaptation_seconds"
                    )
                if method == "base":
                    audit_result = {
                        "HER": None,
                        "CP": None,
                        "HFG": None,
                        "cleanliness_totals": {
                            "hindsight_exposed_positions": 0,
                            "teacher_positions": 0,
                            "exact_context_positions": 0,
                            "compared_positions": 0,
                        },
                    }
                else:
                    exposed, teacher, exact, compared = [
                        sum(values_) for values_ in zip(*(_cleanliness(row, method) for row in values))
                    ]
                    her = exposed / teacher
                    cp = exact / compared
                    audit_result = {
                        "HER": her,
                        "CP": cp,
                        "HFG": (1.0 - her) * cp * (accuracy - base_accuracy),
                        "cleanliness_totals": {
                            "hindsight_exposed_positions": exposed,
                            "teacher_positions": teacher,
                            "exact_context_positions": exact,
                            "compared_positions": compared,
                        },
                    }
                methods[method] = {
                    "accuracy": accuracy,
                    "gain_vs_base": None if method == "base" else accuracy - base_accuracy,
                    **(
                        {"wrong_to_correct": 0, "correct_to_wrong": 0, "paired_positions": len(values)}
                        if method == "base"
                        else _paired_flips(scoped_cells["base"], values)
                    ),
                    "adaptation_seconds_per_query": fmean(query_seconds.values()),
                    **audit_result,
                }
            stg_t = methods["csd_t"]["accuracy"] - base_accuracy
            stg_s = methods["csd_sd"]["accuracy"] - base_accuracy
            profile_report[scope] = {
                "n_queries": len({str(row["query_id"]) for row in scoped_base}),
                "methods": methods,
                "STG_T": stg_t,
                "STG_S": stg_s,
                "retention": stg_s / stg_t if stg_t != 0.0 else None,
            }
        report[profile] = profile_report
    return report


def _rlrs(rows: Sequence[Mapping[str, Any]], context: str) -> dict[str, Any]:
    values: dict[int, list[float]] = {0: [], 1: []}
    for index, row in enumerate(rows, 1):
        reward = _binary(_required(row, "reward", context), f"{context}[{index}].reward")
        teacher = _number(
            _required(row, "teacher_logprob_sum", context),
            f"{context}[{index}].teacher_logprob_sum",
        )
        student = _number(
            _required(row, "student_logprob_sum", context),
            f"{context}[{index}].student_logprob_sum",
        )
        tokens = _integer(
            _required(row, "token_count", context), f"{context}[{index}].token_count", minimum=1
        )
        values[reward].append((teacher - student) / tokens)
    correct_mean = fmean(values[1]) if values[1] else None
    incorrect_mean = fmean(values[0]) if values[0] else None
    return {
        "correct_trajectory_mean_normalized_log_ratio": correct_mean,
        "incorrect_trajectory_mean_normalized_log_ratio": incorrect_mean,
        "RLRS": (
            correct_mean - incorrect_mean
            if correct_mean is not None and incorrect_mean is not None
            else None
        ),
        "correct_trajectory_count": len(values[1]),
        "incorrect_trajectory_count": len(values[0]),
    }


def _psr(rows: Sequence[Mapping[str, Any]], context: str) -> dict[str, Any]:
    style_error = style_tokens = task_error = task_tokens = 0.0
    for index, row in enumerate(rows, 1):
        version = _string(
            _required(row, "partition_version", context),
            f"{context}[{index}].partition_version",
        )
        if version != PSR_PARTITION_VERSION:
            raise PersistentReportError(
                f"{context}[{index}] partition {version!r} != {PSR_PARTITION_VERSION!r}"
            )
        style_error += _number(
            _required(row, "style_abs_error_sum", context),
            f"{context}[{index}].style_abs_error_sum",
            minimum=0.0,
        )
        style_tokens += _integer(
            _required(row, "style_token_count", context),
            f"{context}[{index}].style_token_count",
        )
        task_error += _number(
            _required(row, "task_abs_error_sum", context),
            f"{context}[{index}].task_abs_error_sum",
            minimum=0.0,
        )
        task_tokens += _integer(
            _required(row, "task_token_count", context),
            f"{context}[{index}].task_token_count",
        )
    if style_tokens == 0 or task_tokens == 0:
        raise PersistentReportError(f"{context} needs nonempty style and task partitions")
    style_mean = style_error / style_tokens
    task_mean = task_error / task_tokens
    return {
        "partition_version": PSR_PARTITION_VERSION,
        "style_abs_error_sum": style_error,
        "style_token_count": int(style_tokens),
        "style_mean_abs_error": style_mean,
        "task_abs_error_sum": task_error,
        "task_token_count": int(task_tokens),
        "task_mean_abs_error": task_mean,
        "PSR": style_mean / task_mean if task_mean != 0.0 else None,
    }


def _frontier_rates(rows: Sequence[Mapping[str, Any]], context: str) -> dict[str, Any]:
    eligible = crossings = preserved_eligible = regressions = 0
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, 1):
        key = (
            _string(_required(row, "query_id", context), f"{context}[{index}].query_id"),
            _string(_required(row, "frontier_id", context), f"{context}[{index}].frontier_id"),
        )
        if key in seen:
            raise PersistentReportError(f"{context} duplicates frontier {key!r}")
        seen.add(key)
        base = _number(_required(row, "base_margin", context), f"{context}.base_margin")
        teacher = _number(
            _required(row, "teacher_margin", context), f"{context}.teacher_margin"
        )
        if base <= 0.0:
            eligible += 1
            crossings += teacher > 0.0
        else:
            preserved_eligible += 1
            regressions += teacher <= 0.0
    return {
        "decoding_boundary_eligible_count": eligible,
        "decoding_boundary_crossing_count": crossings,
        "DBCR": crossings / eligible if eligible else None,
        "positive_base_frontier_count": preserved_eligible,
        "positive_to_nonpositive_regression_count": regressions,
        "regression_rate": (
            regressions / preserved_eligible if preserved_eligible else None
        ),
    }


def _behavioral_diagnostics(
    rows: Sequence[Mapping[str, Any]], context: str
) -> dict[str, Any]:
    hallucinations = hedges = response_tokens = truncations = 0.0
    entropies: list[float] = []
    for index, row in enumerate(rows, 1):
        diagnostics = _mapping(
            _required(row, "behavioral_diagnostics", context),
            f"{context}[{index}].behavioral_diagnostics",
        )
        hallucinations += int(
            _boolean(
                _required(diagnostics, "fabricated_reference_hallucination", context),
                f"{context}[{index}].fabricated_reference_hallucination",
            )
        )
        hedges += _integer(
            _required(diagnostics, "hedging_token_count", context),
            f"{context}[{index}].hedging_token_count",
        )
        response_tokens += _integer(
            _required(diagnostics, "response_tokens", context),
            f"{context}[{index}].response_tokens",
        )
        truncations += int(
            _boolean(
                _required(diagnostics, "truncated", context),
                f"{context}[{index}].truncated",
            )
        )
        entropy = diagnostics.get("mean_entropy")
        if entropy is not None:
            entropies.append(
                _number(entropy, f"{context}[{index}].mean_entropy", minimum=0.0)
            )
    count = len(rows)
    return {
        "fabricated_reference_hallucination_rate": hallucinations / count,
        "mean_hedging_token_count": hedges / count,
        "mean_response_tokens": response_tokens / count,
        "mean_entropy": fmean(entropies) if entropies else None,
        "truncation_rate": truncations / count,
        "ood_accuracy": None,
    }


def _build_mechanism(raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        teacher_type: {"trajectory": [], "frontier": []} for teacher_type in TEACHER_TYPES
    }
    seen_trajectories: set[tuple[str, str, str]] = set()
    for index, row in enumerate(raw_rows, 1):
        context = f"mechanism row {index}"
        record_type = _string(_required(row, "record_type", context), f"{context}.record_type")
        if record_type not in {"trajectory", "frontier"}:
            raise PersistentReportError(f"Unknown mechanism record type {record_type!r}")
        teacher_type = _string(
            _required(row, "teacher_type", context), f"{context}.teacher_type"
        )
        if teacher_type not in grouped:
            raise PersistentReportError(f"Unknown mechanism teacher {teacher_type!r}")
        if record_type == "trajectory":
            key = (
                teacher_type,
                _string(_required(row, "query_id", context), f"{context}.query_id"),
                _string(
                    _required(row, "trajectory_id", context), f"{context}.trajectory_id"
                ),
            )
            if key in seen_trajectories:
                raise PersistentReportError(f"Mechanism duplicates trajectory {key!r}")
            seen_trajectories.add(key)
        grouped[teacher_type][record_type].append(row)

    query_sets: list[set[str]] = []
    report: dict[str, Any] = {}
    for teacher_type, records in grouped.items():
        if not records["trajectory"]:
            raise PersistentReportError(
                f"Mechanism {teacher_type} needs trajectory records"
            )
        trajectory_queries = {str(row["query_id"]) for row in records["trajectory"]}
        frontier_queries = {str(row["query_id"]) for row in records["frontier"]}
        if teacher_type == "clean_teacher" and not records["frontier"]:
            raise PersistentReportError(
                "Clean teacher needs measured correction frontiers"
            )
        if not frontier_queries <= trajectory_queries:
            raise PersistentReportError(
                f"Mechanism {teacher_type} has frontiers outside trajectory coverage"
            )
        if teacher_type != "clean_teacher" and records["frontier"]:
            raise PersistentReportError(
                f"Mechanism {teacher_type} must not fabricate ridge frontiers"
            )
        audits = [
            _cleanliness(row, f"mechanism.{teacher_type}.audit")
            for row in records["trajectory"]
        ]
        exposed, teacher, exact, compared = [
            sum(values) for values in zip(*audits)
        ]
        if teacher == 0 or compared == 0:
            raise PersistentReportError(
                f"Mechanism {teacher_type} has empty cleanliness denominators"
            )
        her = exposed / teacher
        cp = exact / compared
        expected = {
            "pre_decision_privilege": (0.0, 0.0),
            "post_outcome_privilege": (1.0, 0.0),
            "clean_teacher": (0.0, 1.0),
        }[teacher_type]
        if (her, cp) != expected:
            raise PersistentReportError(
                f"Mechanism {teacher_type} must demonstrate HER={expected[0]:g}/CP={expected[1]:g}"
            )
        query_sets.append(trajectory_queries)
        report[teacher_type] = {
            "trajectory_query_count": len(trajectory_queries),
            "frontier_query_count": len(frontier_queries),
            "frontier_query_coverage": len(frontier_queries) / len(trajectory_queries),
            "HER": her,
            "CP": cp,
            "cleanliness_totals": {
                "hindsight_exposed_positions": exposed,
                "teacher_positions": teacher,
                "exact_context_positions": exact,
                "compared_positions": compared,
            },
            **_rlrs(records["trajectory"], f"mechanism.{teacher_type}.trajectory"),
            **_psr(records["trajectory"], f"mechanism.{teacher_type}.trajectory"),
            "behavioral_diagnostics": _behavioral_diagnostics(
                records["trajectory"], f"mechanism.{teacher_type}.trajectory"
            ),
            **(
                _frontier_rates(
                    records["frontier"], f"mechanism.{teacher_type}.frontier"
                )
                if records["frontier"]
                else {
                    "decoding_boundary_eligible_count": None,
                    "decoding_boundary_crossing_count": None,
                    "DBCR": None,
                    "positive_base_frontier_count": None,
                    "positive_to_nonpositive_regression_count": None,
                    "regression_rate": None,
                }
            ),
        }
    if any(query_set != query_sets[0] for query_set in query_sets[1:]):
        raise PersistentReportError("Mechanism teacher types have unmatched query coverage")
    return {"n_queries": len(query_sets[0]), "teachers": report}


def _ablation_samples(row: Mapping[str, Any], context: str) -> list[dict[str, Any]]:
    samples = _required(row, "target_samples", context)
    if not isinstance(samples, list) or len(samples) != 4:
        raise PersistentReportError(f"{context}.target_samples must contain four samples")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(samples):
        item = _mapping(raw, f"{context}.target_samples[{index}]")
        sample_index = _integer(
            _required(item, "sample_index", context), f"{context}.sample_index"
        )
        normalized.append(
            {
                "sample_index": sample_index,
                "base_correct": _binary(
                    _required(item, "base_correct", context), f"{context}.base_correct"
                ),
                "method_correct": _binary(
                    _required(item, "method_correct", context), f"{context}.method_correct"
                ),
            }
        )
    if sorted(item["sample_index"] for item in normalized) != list(SAMPLE_INDICES):
        raise PersistentReportError(f"{context}.target_samples need indices 0..3")
    return sorted(normalized, key=lambda item: item["sample_index"])


def _validate_ablation_match(
    left: Mapping[str, Any], right: Mapping[str, Any], query_id: str
) -> None:
    fields = (
        "heldout_query_denominator",
        "runtime_subset_exclusions",
        "candidate_count",
        "actual_support_tokens",
        "ridge_dimension",
        "decode_config",
        "seeds",
    )
    for field in fields:
        if _canonical_json(_required(left, field, "ablation")) != _canonical_json(
            _required(right, field, "ablation")
        ):
            raise PersistentReportError(
                f"Ablation query {query_id!r} is not matched on {field}"
            )
    _integer(left["candidate_count"], "candidate_count", minimum=1)
    _integer(left["actual_support_tokens"], "actual_support_tokens", minimum=1)
    _integer(left["ridge_dimension"], "ridge_dimension", minimum=1)
    decode = _mapping(left["decode_config"], "decode_config")
    for key in ("temperature", "top_p", "top_k", "max_new_tokens", "num_samples"):
        _required(decode, key, "decode_config")
    if _integer(decode["num_samples"], "decode_config.num_samples", minimum=1) != 4:
        raise PersistentReportError("Ablation decoding must use four samples")
    seeds = left["seeds"]
    if not isinstance(seeds, list) or len(seeds) != 4 or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise PersistentReportError("Ablation seeds must be a four-integer list")
    frontier_identities = []
    for row, side in ((left, "correct_only"), (right, "correct_wrong_signed")):
        frontiers = _nested_rows(row, "frontiers", f"ablation.{side}")
        identity = sorted(
            (
                _string(frontier.get("frontier_id"), f"ablation.{side}.frontier_id"),
                _number(frontier.get("base_margin"), f"ablation.{side}.base_margin"),
            )
            for frontier in frontiers
        )
        frontier_identities.append(identity)
    if frontier_identities[0] != frontier_identities[1]:
        raise PersistentReportError(
            f"Ablation query {query_id!r} is not matched on frontier identity/base margin"
        )


def _nested_rows(row: Mapping[str, Any], key: str, context: str) -> list[dict[str, Any]]:
    values = _required(row, key, context)
    if not isinstance(values, list) or not values or any(not isinstance(item, dict) for item in values):
        raise PersistentReportError(f"{context}.{key} must be a nonempty object list")
    return [dict(item) for item in values]


def _build_ablation(raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw_row in enumerate(raw_rows, 1):
        row = dict(raw_row)
        context = f"ablation row {index}"
        query_id = _string(_required(row, "query_id", context), f"{context}.query_id")
        variant = _string(_required(row, "variant", context), f"{context}.variant")
        if variant not in ABLATION_VARIANTS:
            raise PersistentReportError(f"Unknown ablation variant {variant!r}")
        key = (query_id, variant)
        if key in indexed:
            raise PersistentReportError(f"Ablation duplicates {key!r}")
        _number(_required(row, "support_nll", context), f"{context}.support_nll", minimum=0.0)
        _number(
            _required(row, "adaptation_seconds", context),
            f"{context}.adaptation_seconds",
            minimum=0.0,
        )
        _ablation_samples(row, context)
        _nested_rows(row, "trajectories", context)
        _nested_rows(row, "frontiers", context)
        indexed[key] = row
    query_ids = sorted({query_id for query_id, _ in indexed})
    if not query_ids or set(indexed) != {
        (query_id, variant) for query_id in query_ids for variant in ABLATION_VARIANTS
    }:
        raise PersistentReportError("Ablation needs both variants for every query")
    for query_id in query_ids:
        _validate_ablation_match(
            indexed[(query_id, "correct_only")],
            indexed[(query_id, "correct_wrong_signed")],
            query_id,
        )
    denominators = {
        _integer(
            row.get("heldout_query_denominator"),
            "ablation.heldout_query_denominator",
            minimum=1,
        )
        for row in indexed.values()
    }
    exclusion_payloads = {
        _canonical_json(row.get("runtime_subset_exclusions"))
        for row in indexed.values()
    }
    if len(denominators) != 1 or len(exclusion_payloads) != 1:
        raise PersistentReportError("Ablation subset provenance is inconsistent")
    heldout_denominator = next(iter(denominators))
    if heldout_denominator < len(query_ids):
        raise PersistentReportError("Ablation matched subset exceeds its denominator")
    exclusions = _mapping(
        next(iter(indexed.values())).get("runtime_subset_exclusions"),
        "ablation.runtime_subset_exclusions",
    )
    normalized_exclusions = {
        _string(key, "ablation.exclusion_name"): _integer(
            value, f"ablation.exclusions.{key}"
        )
        for key, value in exclusions.items()
    }
    if sum(normalized_exclusions.values()) + len(query_ids) != heldout_denominator:
        raise PersistentReportError(
            "Ablation matched/excluded counts do not cover the held-out denominator"
        )

    variants: dict[str, Any] = {}
    for variant in ABLATION_VARIANTS:
        rows = [indexed[(query_id, variant)] for query_id in query_ids]
        samples = [
            sample
            for row in rows
            for sample in _ablation_samples(row, f"ablation.{variant}")
        ]
        trajectories = [
            trajectory
            for row in rows
            for trajectory in _nested_rows(row, "trajectories", f"ablation.{variant}")
        ]
        frontiers: list[dict[str, Any]] = []
        for row in rows:
            query_id = str(row["query_id"])
            for frontier in _nested_rows(row, "frontiers", f"ablation.{variant}"):
                frontiers.append({**frontier, "query_id": query_id})
        base_acc1 = fmean(sample["base_correct"] for sample in samples if sample["sample_index"] == 0)
        method_acc1 = fmean(sample["method_correct"] for sample in samples if sample["sample_index"] == 0)
        base_mean4 = fmean(sample["base_correct"] for sample in samples)
        method_mean4 = fmean(sample["method_correct"] for sample in samples)
        flips = {
            profile: {
                "wrong_to_correct": sum(
                    item["base_correct"] == 0 and item["method_correct"] == 1
                    for item in samples
                    if profile == "mean4" or item["sample_index"] == 0
                ),
                "correct_to_wrong": sum(
                    item["base_correct"] == 1 and item["method_correct"] == 0
                    for item in samples
                    if profile == "mean4" or item["sample_index"] == 0
                ),
            }
            for profile in ("acc1", "mean4")
        }
        variants[variant] = {
            "n_queries": len(rows),
            "mean_support_nll": fmean(float(row["support_nll"]) for row in rows),
            "acc1": method_acc1,
            "acc1_gain_vs_base": method_acc1 - base_acc1,
            "mean4": method_mean4,
            "mean4_gain_vs_base": method_mean4 - base_mean4,
            "flips": flips,
            **_rlrs(trajectories, f"ablation.{variant}.trajectories"),
            **_frontier_rates(frontiers, f"ablation.{variant}.frontiers"),
            "adaptation_seconds_per_query": fmean(
                float(row["adaptation_seconds"]) for row in rows
            ),
        }
    return {
        "matching": {
            "status": "passed",
            "fields": [
                "heldout_query_denominator",
                "runtime_subset_exclusions",
                "candidate_count",
                "actual_support_tokens",
                "ridge_dimension",
                "decode_config",
                "seeds",
                "frontier_identity_and_base_margin",
            ],
            "n_paired_queries": len(query_ids),
            "heldout_query_denominator": heldout_denominator,
            "matched_ready_rate": len(query_ids) / heldout_denominator,
            "exclusions": normalized_exclusions,
        },
        "variants": variants,
    }


def _main_table(
    short_term: Mapping[str, Any], long_horizon: Mapping[str, Any]
) -> list[dict[str, Any]]:
    labels = {
        "base": "Base",
        "privileged_sd": "Privileged SD",
        "csd_t": "CSD-T",
        "csd_sd": "CSD-SD",
    }
    rows: list[dict[str, Any]] = []
    for profile in ("acc1", "mean4"):
        short = short_term[profile]["combined"]
        long = long_horizon[profile]["combined"]
        for method in SHORT_METHODS:
            immediate = short["methods"][method]
            if method == "base":
                final_long = long["A_0"]
                lhg = aulc = 0.0
            elif method == "privileged_sd":
                branch = long["branches"]["privileged_sd"]
                final_long, lhg, aulc, final_long_hfg = (
                    branch["final_long_accuracy"],
                    branch["LHG"],
                    branch["AULC"],
                    branch["final_HFG"],
                )
            elif method == "csd_sd":
                branch = long["branches"]["clean_sd"]
                final_long, lhg, aulc, final_long_hfg = (
                    branch["final_long_accuracy"],
                    branch["LHG"],
                    branch["AULC"],
                    branch["final_HFG"],
                )
            else:
                final_long = lhg = aulc = final_long_hfg = None
            if method == "base":
                final_long_hfg = None
            rows.append(
                {
                    "profile": profile,
                    "method": labels[method],
                    "short_acc": immediate["accuracy"],
                    "STG_T": short["STG_T"] if method == "csd_t" else None,
                    "student_acc": (
                        immediate["accuracy"] if method in {"base", "privileged_sd", "csd_sd"} else None
                    ),
                    "STG_S": short["STG_S"] if method == "csd_sd" else None,
                    "retention": short["retention"] if method == "csd_sd" else None,
                    "final_long_acc": final_long,
                    "LHG": lhg,
                    "AULC": aulc,
                    "HER": immediate["HER"],
                    "CP": immediate["CP"],
                    "HFG": immediate["HFG"],
                    "final_long_HFG": final_long_hfg,
                    "seconds_per_query": immediate["adaptation_seconds_per_query"],
                }
            )
    return rows


def build_persistent_report(
    heldout_rows: Sequence[Mapping[str, Any]],
    short_term_rows: Sequence[Mapping[str, Any]],
    mechanism_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    *,
    training_rows: Sequence[Mapping[str, Any]],
    checkpoint_manifests: Sequence[Mapping[str, Any]],
    expected_source_counts: Mapping[str, int] = DEFAULT_SOURCE_COUNTS,
) -> dict[str, Any]:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in expected_source_counts.values()
    ):
        raise PersistentReportError("Expected source counts must be positive integers")
    training_audit = _build_training_audit(training_rows, checkpoint_manifests)
    long_horizon = _build_long_horizon(heldout_rows, expected_source_counts)
    branch_audits = {
        "clean_sd": training_audit["branches"]["clean"],
        "privileged_sd": training_audit["branches"]["privileged"],
    }
    for profile_report in long_horizon.values():
        for scope_report in profile_report.values():
            for method, branch in scope_report["branches"].items():
                audit = branch_audits[method]
                her = float(audit["HER"])
                cp = float(audit["CP"])
                branch["HER"] = her
                branch["CP"] = cp
                branch["final_HFG"] = (1.0 - her) * cp * float(branch["LHG"])
                for point in branch["A_k"]:
                    point["HFG"] = (
                        (1.0 - her) * cp * float(point["gain_vs_a0"])
                    )
    short_term = _build_short_term(short_term_rows, expected_source_counts)
    mechanism = _build_mechanism(mechanism_rows)
    heldout_query_ids = {
        str(row.get("query_id"))
        for row in heldout_rows
        if row.get("method") == "base" and row.get("profile") == "mean4"
    }
    mechanism_query_ids = {
        str(row.get("query_id"))
        for row in mechanism_rows
        if row.get("record_type") == "trajectory"
    }
    if (
        mechanism["n_queries"] != sum(expected_source_counts.values())
        or mechanism_query_ids != heldout_query_ids
    ):
        raise PersistentReportError(
            "Mechanism study must cover the complete held-out query universe"
        )
    ablation = _build_ablation(ablation_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "checkpoints": list(CHECKPOINTS),
            "samples": 4,
            "acc1_definition": "paired sample_index == 0",
            "mean4_definition": "mean of all four binary sample scores",
            "aulc_definition": "normalized trapezoidal integral of A_k-A_0 over episodes 0..1000",
            "crossover_definition": "first observed checkpoint with Clean accuracy > Privileged accuracy; no interpolation",
            "hfg_definition": "(1-HER)*CP*(method_accuracy-base_accuracy)",
            "long_hfg_definition": "the same HFG applied to each persistent checkpoint gain A_k-A_0",
            "rlrs_definition": "mean[(logp_T-logp_S)/tokens|R=1] - mean[(logp_T-logp_S)/tokens|R=0]",
            "psr_partition_version": PSR_PARTITION_VERSION,
            "dbcr_definition": "count(base_margin<=0 and teacher_margin>0) / count(base_margin<=0)",
            "expected_source_counts": dict(expected_source_counts),
        },
        "validation": {"status": "passed"},
        "persistent_training_audit": training_audit,
        "short_term": short_term,
        "long_horizon": long_horizon,
        "mechanism": mechanism,
        "ablation": ablation,
        "main_table": _main_table(short_term, long_horizon),
    }


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "profile",
        "method",
        "short_acc",
        "STG_T",
        "student_acc",
        "STG_S",
        "retention",
        "final_long_acc",
        "LHG",
        "AULC",
        "HER",
        "CP",
        "HFG",
        "final_long_HFG",
        "seconds_per_query",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row[key] for key in columns})
    return stream.getvalue()


def generate_persistent_report(
    *,
    training_journal_paths: Sequence[str | Path],
    checkpoint_manifest_paths: Sequence[str | Path],
    heldout_paths: Sequence[str | Path],
    short_term_paths: Sequence[str | Path],
    mechanism_paths: Sequence[str | Path],
    ablation_paths: Sequence[str | Path],
    output_json: str | Path,
    output_csv: str | Path,
    expected_source_counts: Mapping[str, int] = DEFAULT_SOURCE_COUNTS,
) -> dict[str, Any]:
    report = build_persistent_report(
        _load_jsonl(heldout_paths, "heldout"),
        _load_jsonl(short_term_paths, "short-term"),
        _load_jsonl(mechanism_paths, "mechanism"),
        _load_jsonl(ablation_paths, "ablation"),
        training_rows=_load_jsonl(training_journal_paths, "training journal"),
        checkpoint_manifests=_load_json_objects(
            checkpoint_manifest_paths, "checkpoint manifest"
        ),
        expected_source_counts=expected_source_counts,
    )
    report["input_sha256"] = {
        "training_journal": _artifact_digest(training_journal_paths),
        "checkpoint_manifests": _artifact_digest(checkpoint_manifest_paths),
        "heldout": _artifact_digest(heldout_paths),
        "short_term": _artifact_digest(short_term_paths),
        "mechanism": _artifact_digest(mechanism_paths),
        "ablation": _artifact_digest(ablation_paths),
    }
    _atomic_write(Path(output_json), json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _atomic_write(Path(output_csv), _csv_text(report["main_table"]))
    return report


def _source_count(value: str) -> tuple[str, int]:
    try:
        source, raw_count = value.split("=", 1)
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("source counts must be SOURCE=COUNT") from exc
    if not source.strip() or count <= 0:
        raise argparse.ArgumentTypeError("source counts must be SOURCE=positive_integer")
    return source.strip().casefold(), count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-journal", nargs="+", required=True)
    parser.add_argument("--checkpoint-manifest", nargs="+", required=True)
    parser.add_argument("--heldout-scored", nargs="+", required=True)
    parser.add_argument("--short-term-scored", nargs="+", required=True)
    parser.add_argument("--mechanism", nargs="+", required=True)
    parser.add_argument("--ablation", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--expected-source-count",
        action="append",
        type=_source_count,
        help="Override formal counts (repeat SOURCE=COUNT); tests only unless preregistered.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = (
        dict(args.expected_source_count)
        if args.expected_source_count
        else DEFAULT_SOURCE_COUNTS
    )
    report = generate_persistent_report(
        training_journal_paths=args.training_journal,
        checkpoint_manifest_paths=args.checkpoint_manifest,
        heldout_paths=args.heldout_scored,
        short_term_paths=args.short_term_scored,
        mechanism_paths=args.mechanism,
        ablation_paths=args.ablation,
        output_json=args.output_json,
        output_csv=args.output_csv,
        expected_source_counts=expected,
    )
    print(_canonical_json({"status": "passed", "schema_version": report["schema_version"]}))


if __name__ == "__main__":
    main()
