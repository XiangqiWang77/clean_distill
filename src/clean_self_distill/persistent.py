"""Persistent, label-blind Clean Self-Distillation training.

This module implements the long-horizon protocol used by the empirical study.
Unlike the legacy query-local evaluation path, the LoRA student and its AdamW
state are never reset between episodes.  Target answers and reference solutions
are physically absent from both the query stream and this trainer's API.

Two independently trained branches share the same query order, rollout budget,
initialization, and optimizer configuration:

* ``clean`` builds a temporary LM-head ridge teacher from target-disjoint
  correct/wrong support trajectories and scores it on the student's exact
  on-policy query and prefix (HER=0, CP=1).
* ``privileged`` gives only the teacher a fixed pre-decision reasoning-method
  instruction (HER=0, CP=0).  It never receives an answer, solution, feedback,
  or future target token.

Every committed episode is an atomically rewritten JSONL prefix.  Restartable
checkpoints contain the PEFT adapter, trainable tensors, optimizer state, Python
and Torch RNG state, and cumulative raw audits.  Scientific checkpoints are
published at the preregistered episode counts; a rolling pointer retains the
most recent restart checkpoint without accumulating unbounded copies.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

import torch
import torch.nn.functional as F

from .heldout import HeldoutProtocolError, load_query_only_manifest
from .io import (
    canonical_json_sha256,
    load_proposal_map,
    validate_proposal_training_binding,
    validate_specialization_state,
)
from .ridge import (
    FRONTIER_TARGET_MARGIN,
    SparseRidgeAdapter,
    fit_ridge_adapter,
    problem_prompt,
)
from .runtime import (
    backbone_forward,
    input_device,
    project_logits,
    render_chat,
    unwrap_causal_lm,
)
from .streaming_distill import stream_distillation_chunks
from .train_eval import generate_response


EPISODE_SCHEMA_VERSION = "clean-self-distill-persistent-episode-v1"
CHECKPOINT_SCHEMA_VERSION = "clean-self-distill-persistent-checkpoint-v1"
RUN_SCHEMA_VERSION = "clean-self-distill-persistent-run-v1"
STYLE_TASK_PARTITION_VERSION = "rlcsd-style-task-v1"
STYLE_TASK_ERROR_DEFINITION = (
    "abs_teacher_minus_student_realized_token_logprob_pre_update"
)
PRIVILEGED_PROMPT_VERSION = "predecision-reasoning-method-v1"
REQUEUE_EXIT_CODE = 75

BRANCHES = frozenset({"clean", "privileged"})
VARIANTS = frozenset({"correct_only", "correct_wrong_signed"})

# These are forbidden only at the target-query/proposal top level.  Support
# candidates necessarily contain their *own* verified solution and answer.
_FORBIDDEN_TARGET_TOP_LEVEL_KEYS = frozenset(
    {
        "answer",
        "feedback",
        "ground_truth",
        "label",
        "reference",
        "reference_answer",
        "reference_solution",
        "reward_model",
        "solution",
        "target",
        "target_answer",
        "target_solution",
    }
)

_STYLE_WORDS = frozenset(
    {
        "accordingly",
        "alternatively",
        "answer",
        "clearly",
        "consequently",
        "finally",
        "first",
        "hence",
        "however",
        "indeed",
        "next",
        "note",
        "now",
        "perhaps",
        "second",
        "similarly",
        "step",
        "suppose",
        "therefore",
        "thus",
        "verify",
        "we",
    }
)
_TASK_TOKEN_RE = re.compile(
    r"(?:\d|[=+\-*/^<>%{}\[\]()]|\\(?:frac|sqrt|boxed|sum|prod|mod|equiv|"
    r"binom|gcd|lcm|sin|cos|tan|log|ln|pi|theta|alpha|beta))",
    flags=re.IGNORECASE,
)


class PersistentProtocolError(ValueError):
    """Raised when an artifact violates the persistent-study contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_ids_sha256(token_ids: torch.Tensor) -> str:
    values = token_ids.detach().cpu().reshape(-1).tolist()
    return hashlib.sha256(",".join(map(str, values)).encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, _canonical_json(dict(value)) + "\n")


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(_canonical_json(dict(row)) + "\n" for row in rows),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PersistentProtocolError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PersistentProtocolError(f"{path}:{line_no} is not an object")
            rows.append(value)
    return rows


def parse_scientific_checkpoints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            checkpoints = tuple(int(item.strip()) for item in value.split(","))
        except ValueError as exc:
            raise PersistentProtocolError(
                "scientific checkpoints must be comma-separated integers"
            ) from exc
    else:
        checkpoints = tuple(int(item) for item in value)
    if not checkpoints or checkpoints != tuple(sorted(set(checkpoints))):
        raise PersistentProtocolError(
            "scientific checkpoints must be nonempty, unique, and increasing"
        )
    return checkpoints


@dataclass(frozen=True)
class PersistentConfig:
    branch: str
    variant: str
    model: str
    model_id: str
    revision: str
    episodes: int = 1_000
    scientific_checkpoints: tuple[int, ...] = (0, 250, 500, 750, 1_000)
    rolling_checkpoint_interval: int = 5
    max_sequence_tokens: int = 16_384
    max_rollout_tokens: int = 16_384
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    lora_rank: int = 8
    lora_alpha: int = 16
    seed: int = 0
    train_temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_grad_norm: float = 1.0
    distill_top_k: int = 64
    distill_temperature: float = 1.0
    distill_token_clip: float = 0.0
    # Bound every LM-head projection and full-vocabulary KL tensor along the
    # token axis.  This is an execution parameter only: chunks are weighted by
    # their exact token counts, so the objective and accumulated gradient equal
    # the unchunked mean loss.
    distill_token_chunk_size: int = 128
    teacher_projection_mode: str = "ridge"
    trust_region_kl_budget: float = 0.08
    trust_region_binary_search_steps: int = 5
    ridge_lambda: float = 0.1
    residual_step_size: float = 0.8
    max_tokens_per_candidate: int = 96
    max_support_tokens: int = 768
    num_specialization_candidates: Optional[int] = None
    hard_negatives: int = 8
    ridge_max_length: int = 8_192
    reasoning_token_weight: float = 0.25
    answer_token_weight: float = 1.0
    frontier_positive_weight: float = 8.0
    frontier_negative_weight: float = 8.0
    frontier_max_tokens: int = 24
    frontier_negative_probability_floor: float = 0.25
    frontier_target_margin: float = FRONTIER_TARGET_MARGIN
    max_update_norm: float = 2.0

    def validate(self) -> None:
        if self.branch not in BRANCHES:
            raise PersistentProtocolError(f"Unknown branch {self.branch!r}")
        if self.variant not in VARIANTS:
            raise PersistentProtocolError(f"Unknown variant {self.variant!r}")
        if not self.model or not self.model_id or not self.revision:
            raise PersistentProtocolError("model, model_id, and revision are required")
        if self.episodes <= 0:
            raise PersistentProtocolError("episodes must be positive")
        checkpoints = parse_scientific_checkpoints(self.scientific_checkpoints)
        if checkpoints[0] != 0 or checkpoints[-1] != self.episodes:
            raise PersistentProtocolError(
                "scientific checkpoints must include both 0 and final episodes"
            )
        if self.rolling_checkpoint_interval <= 0:
            raise PersistentProtocolError("rolling checkpoint interval must be positive")
        for name in (
            "max_sequence_tokens",
            "max_rollout_tokens",
            "lora_rank",
            "lora_alpha",
            "top_k",
            "distill_top_k",
            "distill_token_chunk_size",
            "trust_region_binary_search_steps",
            "frontier_target_margin",
            "max_tokens_per_candidate",
            "max_support_tokens",
            "hard_negatives",
            "ridge_max_length",
            "frontier_max_tokens",
        ):
            if int(getattr(self, name)) <= 0:
                raise PersistentProtocolError(f"{name} must be positive")
        for name in (
            "learning_rate",
            "max_grad_norm",
            "distill_temperature",
            "ridge_lambda",
            "residual_step_size",
            "reasoning_token_weight",
            "answer_token_weight",
            "frontier_positive_weight",
            "frontier_negative_weight",
            "frontier_target_margin",
            "max_update_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise PersistentProtocolError(f"{name} must be finite and positive")
        if self.weight_decay < 0 or self.distill_token_clip < 0:
            raise PersistentProtocolError("weight decay and token clip cannot be negative")
        if self.train_temperature < 0 or not 0 < self.top_p <= 1:
            raise PersistentProtocolError("invalid rollout sampling parameters")
        if (
            self.num_specialization_candidates is not None
            and self.num_specialization_candidates <= 0
        ):
            raise PersistentProtocolError(
                "num_specialization_candidates must be positive when provided"
            )
        if not 0 <= self.frontier_negative_probability_floor <= 1:
            raise PersistentProtocolError(
                "frontier negative probability floor must be in [0,1]"
            )
        if self.teacher_projection_mode not in {"ridge", "trust_region"}:
            raise PersistentProtocolError(
                "teacher_projection_mode must be ridge or trust_region"
            )
        if self.trust_region_kl_budget <= 0:
            raise PersistentProtocolError(
                "trust_region_kl_budget must be positive"
            )
        if not int(self.trust_region_binary_search_steps) > 0:
            raise PersistentProtocolError(
                "trust_region_binary_search_steps must be a positive integer"
            )
    @property
    def method_id(self) -> str:
        if self.branch == "clean":
            if self.teacher_projection_mode == "ridge":
                return f"clean:{self.variant}"
            return f"clean:{self.variant}:{self.teacher_projection_mode}"
        return "privileged:predecision_method"

    def identity_payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["scientific_checkpoints"] = list(self.scientific_checkpoints)
        value.update(
            {
                "run_schema_version": RUN_SCHEMA_VERSION,
                "method_id": self.method_id,
                "privileged_prompt_version": PRIVILEGED_PROMPT_VERSION,
                "style_task_partition_version": STYLE_TASK_PARTITION_VERSION,
            }
        )
        return value


def _validate_proposal_firewall(
    proposal: Mapping[str, Any], query: Mapping[str, Any]
) -> None:
    exposed = sorted(
        key
        for key in _FORBIDDEN_TARGET_TOP_LEVEL_KEYS
        if key in {str(item).strip().casefold() for item in proposal}
    )
    if exposed:
        raise PersistentProtocolError(
            f"Proposal {query['query_id']} exposes target-level fields {exposed}"
        )
    for key in ("query_id", "problem", "problem_sha256"):
        if str(proposal.get(key, "")).strip() != str(query[key]).strip():
            raise PersistentProtocolError(
                f"Proposal/query mismatch for {query['query_id']} field {key}"
            )
    if str(proposal.get("source", "")).strip().casefold() != str(
        query["source"]
    ).casefold():
        raise PersistentProtocolError(
            f"Proposal/query source mismatch for {query['query_id']}"
        )
    if proposal.get("schema_version") != "clean-self-distill-proposals-v5":
        raise PersistentProtocolError(
            f"Proposal {query['query_id']} is not the corrective v5 schema"
        )
    validate_proposal_training_binding(
        dict(proposal), context=f"Persistent proposal {query['query_id']}"
    )
    validate_specialization_state(
        dict(proposal), context=f"Persistent proposal {query['query_id']}"
    )
    audit = proposal.get("firewall_audit")
    if not isinstance(audit, Mapping):
        raise PersistentProtocolError(
            f"Proposal {query['query_id']} is missing its firewall audit"
        )
    if audit.get("target_answer_loaded") is not False:
        raise PersistentProtocolError(
            f"Proposal {query['query_id']} does not prove target-answer isolation"
        )
    if audit.get("target_solution_loaded") is not False:
        raise PersistentProtocolError(
            f"Proposal {query['query_id']} does not prove target-solution isolation"
        )
    if audit.get("all_accepted_candidate_artifacts_target_disjoint") is not True:
        raise PersistentProtocolError(
            f"Proposal {query['query_id']} failed target-disjoint artifact audit"
        )


def load_persistent_inputs(
    query_path: str | Path,
    proposal_path: str | Path,
    *,
    episodes: int,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]], dict[str, str]]:
    """Load only physically target-free queries and their bound proposals."""
    try:
        queries = load_query_only_manifest(query_path)
    except HeldoutProtocolError as exc:
        raise PersistentProtocolError(str(exc)) from exc
    if len(queries) < episodes:
        raise PersistentProtocolError(
            f"Query stream has {len(queries)} rows but {episodes} episodes were requested"
        )
    if len(queries) != episodes:
        # A formal branch must bind exactly the preregistered stream rather than
        # silently ignoring a suffix with potentially different composition.
        raise PersistentProtocolError(
            f"Query stream must contain exactly {episodes} rows, found {len(queries)}"
        )
    proposals = load_proposal_map(proposal_path)
    query_ids = [row["query_id"] for row in queries]
    if set(proposals) != set(query_ids):
        missing = sorted(set(query_ids) - set(proposals))
        extra = sorted(set(proposals) - set(query_ids))
        raise PersistentProtocolError(
            f"Proposal coverage mismatch missing={missing[:5]} extra={extra[:5]}"
        )
    for query in queries:
        _validate_proposal_firewall(proposals[query["query_id"]], query)
    hashes = {
        "query_manifest_sha256": file_sha256(query_path),
        "proposal_manifest_sha256": file_sha256(proposal_path),
    }
    return queries, proposals, hashes


def build_privileged_prompt(tokenizer, problem: str) -> str:
    """Fixed teacher-only methodology privilege with no outcome information."""
    messages = [
        {
            "role": "system",
            "content": (
                "Private reasoning-method instruction for the teacher: decompose the "
                "problem into explicit subgoals, track constraints and invariants, "
                "check boundary cases, and verify the chosen route against an "
                "independent alternative when possible. Use only the problem statement."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{problem.strip()}\n\nPlease reason step by step, and put your final "
                "answer within \\boxed{{}}."
            ),
        },
    ]
    return render_chat(tokenizer, messages, add_generation_prompt=True)


def _tokenize_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    )["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise PersistentProtocolError("Tokenizer did not return one prompt sequence")
    return ids.to(device)


def _trajectory_logprob(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    selected = _realized_token_logprobs(logits, labels)
    total = float(selected.sum().item())
    return total, total / max(int(selected.numel()), 1)


def _realized_token_logprobs(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Gather realized-token log-probabilities without full log-softmax output."""
    float_logits = logits.detach().float()
    label_ids = labels.to(logits.device, dtype=torch.long)
    selected_logits = float_logits.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1)
    return selected_logits - torch.logsumexp(float_logits, dim=-1)


def _decoded_token(tokenizer, token_id: int) -> str:
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if converter is not None:
        try:
            value = converter(int(token_id))
            if value is not None:
                return str(value).replace("\u2581", " ").replace("\u0120", " ")
        except (KeyError, TypeError, ValueError):
            pass
    return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def _token_partition(text: str) -> str:
    if _TASK_TOKEN_RE.search(text):
        return "task"
    words = set(re.findall(r"[A-Za-z]+", text.casefold()))
    if words & _STYLE_WORDS:
        return "style"
    return "other"


def style_task_error_accumulators(
    tokenizer,
    response_ids: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> dict[str, Any]:
    labels = response_ids.detach().reshape(-1)
    label_matrix = labels.reshape(1, -1)
    student_lp = _realized_token_logprobs(student_logits, label_matrix).reshape(-1)
    teacher_lp = _realized_token_logprobs(teacher_logits, label_matrix).reshape(-1)
    errors = (teacher_lp - student_lp).abs().cpu().tolist()
    result: dict[str, Any] = {
        "partition_version": STYLE_TASK_PARTITION_VERSION,
        "error_definition": STYLE_TASK_ERROR_DEFINITION,
        "style_abs_error_sum": 0.0,
        "style_token_count": 0,
        "task_abs_error_sum": 0.0,
        "task_token_count": 0,
        "other_abs_error_sum": 0.0,
        "other_token_count": 0,
    }
    for token_id, error in zip(labels.cpu().tolist(), errors):
        partition = _token_partition(_decoded_token(tokenizer, int(token_id)))
        result[f"{partition}_abs_error_sum"] += float(error)
        result[f"{partition}_token_count"] += 1
    return result


def style_task_error_from_trace(
    tokenizer, trace: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Aggregate the same versioned PSR partition from online scalar traces."""
    result: dict[str, Any] = {
        "partition_version": STYLE_TASK_PARTITION_VERSION,
        "error_definition": STYLE_TASK_ERROR_DEFINITION,
        "style_abs_error_sum": 0.0,
        "style_token_count": 0,
        "task_abs_error_sum": 0.0,
        "task_token_count": 0,
        "other_abs_error_sum": 0.0,
        "other_token_count": 0,
    }
    for index, row in enumerate(trace):
        try:
            token_id = int(row["token_id"])
            student = float(row["student_logprob"])
            teacher = float(row["teacher_logprob"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PersistentProtocolError(
                f"Invalid online log-prob trace row {index}"
            ) from exc
        if not math.isfinite(student) or not math.isfinite(teacher):
            raise PersistentProtocolError("Online log-prob trace must be finite")
        partition = _token_partition(_decoded_token(tokenizer, token_id))
        result[f"{partition}_abs_error_sum"] += abs(teacher - student)
        result[f"{partition}_token_count"] += 1
    return result


def _accumulate_style_task_error(
    cumulative: dict[str, Any], chunk: Mapping[str, Any]
) -> None:
    if (
        chunk.get("partition_version") != STYLE_TASK_PARTITION_VERSION
        or chunk.get("error_definition") != STYLE_TASK_ERROR_DEFINITION
    ):
        raise PersistentProtocolError("Style/task chunk uses an incompatible schema")
    for partition in ("style", "task", "other"):
        cumulative[f"{partition}_abs_error_sum"] += float(
            chunk[f"{partition}_abs_error_sum"]
        )
        cumulative[f"{partition}_token_count"] += int(
            chunk[f"{partition}_token_count"]
        )


def _ridge_metrics_not_applicable() -> dict[str, Any]:
    return {
        "applicable": False,
        "candidate_count": 0,
        "support_variant": "not_applicable",
        "specialization_status": "not_applicable",
        "specialization_no_op": False,
        "support_tokens": 0.0,
        "frontier_comparable_count": 0.0,
        "decision_boundary_crossing_count": 0.0,
        "decision_boundary_eligible_count": 0.0,
        "decision_boundary_crossing_rate": 0.0,
        "decision_boundary_regression_count": 0.0,
        "decision_boundary_regression_eligible_count": 0.0,
        "decision_boundary_regression_rate": 0.0,
        # Stable short aliases consumed by the persistent-study reporter.
        "db_crossing_count": 0.0,
        "db_eligible_count": 0.0,
        "regression_count": 0.0,
        "regression_eligible_count": 0.0,
    }


def _append_jsonl_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    payload = "".join(_canonical_json(row) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)


def _traced_mean_teacher_kl(
    student_logits: torch.Tensor,
    privileged_logits: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return per-position KL( (1-alpha)P + alphaQ || P ) for a fixed chunk."""
    if not (0.0 <= alpha <= 1.0):
        raise PersistentProtocolError("trust-region alpha must be in [0,1]")
    p = F.log_softmax(student_logits.float(), dim=-1)
    q = F.log_softmax(
        (1.0 - alpha) * student_logits.float() + alpha * privileged_logits.float(),
        dim=-1,
    )
    q_probs = torch.exp(q)
    return torch.sum(q_probs * (q - p), dim=-1)


def _trust_region_alpha(
    *,
    model,
    student_hidden: torch.Tensor,
    privileged_hidden: torch.Tensor,
    chunk_size: int,
    kl_budget: float,
    binary_search_steps: int,
) -> tuple[float, float]:
    """Find the largest alpha where KL(teacher||student)<=budget.

    The search is done over [0,1] using per-token KL from the exact full-vocab
    projected logits.  Returned tuple is ``(alpha, achieved_kl)``.
    """
    if student_hidden.shape != privileged_hidden.shape:
        raise PersistentProtocolError("Student/privileged hidden mismatch")

    def mean_kl(alpha_value: float) -> float:
        alpha_tensor = float(alpha_value)
        total_kl = 0.0
        total_tokens = 0
        with torch.no_grad():
            for start in range(0, student_hidden.shape[1], chunk_size):
                stop = min(start + chunk_size, student_hidden.shape[1])
                student_logits = project_logits(model, student_hidden[:, start:stop])
                privileged_logits = project_logits(
                    model, privileged_hidden[:, start:stop]
                )
                kl_chunk = _traced_mean_teacher_kl(
                    student_logits, privileged_logits, alpha_value=alpha_tensor
                ).sum().item()
                chunk_tokens = stop - start
                total_kl += float(kl_chunk)
                total_tokens += chunk_tokens
        if total_tokens <= 0:
            return 0.0
        return total_kl / float(total_tokens)

    low, high = 0.0, 1.0
    for _ in range(int(binary_search_steps)):
        mid = (low + high) / 2.0
        value = mean_kl(mid)
        if value <= kl_budget:
            low = mid
        else:
            high = mid
    achieved = mean_kl(low)
    return low, achieved


def _fit_current_student_teacher(
    model,
    tokenizer,
    proposal: Mapping[str, Any],
    config: PersistentConfig,
) -> tuple[SparseRidgeAdapter, dict[str, Any]]:
    status, reason, no_op = validate_specialization_state(
        dict(proposal), context=f"Persistent proposal {proposal.get('query_id')}"
    )
    candidates = list(proposal.get("specialization_candidates", []))
    if config.num_specialization_candidates is not None:
        candidates = candidates[: config.num_specialization_candidates]
        if not candidates and not no_op:
            raise PersistentProtocolError("Candidate limit removed every ready candidate")
    was_training = model.training
    model.eval()
    fit_fallback_reason = ""

    def fit(
        fit_candidates: Sequence[Mapping[str, Any]],
        *,
        fit_status: str,
        fit_reason: str,
        fit_no_op: bool,
    ) -> tuple[SparseRidgeAdapter, dict[str, Any]]:
        return fit_ridge_adapter(
            model,
            tokenizer,
            fit_candidates,
            ridge_lambda=config.ridge_lambda,
            residual_step_size=config.residual_step_size,
            max_tokens_per_candidate=config.max_tokens_per_candidate,
            max_support_tokens=config.max_support_tokens,
            hard_negatives=config.hard_negatives,
            max_length=config.ridge_max_length,
            reasoning_token_weight=config.reasoning_token_weight,
            answer_token_weight=config.answer_token_weight,
            frontier_positive_weight=config.frontier_positive_weight,
            frontier_negative_weight=config.frontier_negative_weight,
            frontier_max_tokens=config.frontier_max_tokens,
            frontier_negative_probability_floor=(
                config.frontier_negative_probability_floor
            ),
            frontier_target_margin=config.frontier_target_margin,
            signed_frontier=config.variant == "correct_wrong_signed",
            max_update_norm=config.max_update_norm,
            query_id=str(proposal["query_id"]),
            specialization_status=fit_status,
            specialization_failure_reason=fit_reason,
            specialization_no_op=fit_no_op,
        )

    try:
        try:
            adapter, metrics = fit(
                candidates,
                fit_status=status,
                fit_reason=reason,
                fit_no_op=no_op,
            )
        except RuntimeError as exc:
            if "frontier tokens were not scored at the same state" not in str(exc):
                raise
            fit_fallback_reason = f"incompatible verified frontier: {exc}"
            adapter, metrics = fit(
                [],
                fit_status="insufficient_verified_candidates",
                fit_reason=fit_fallback_reason,
                fit_no_op=True,
            )
    finally:
        model.train(was_training)
    metrics = dict(metrics)
    metrics["applicable"] = True
    metrics["proposal_training_sha256"] = validate_proposal_training_binding(
        dict(proposal), context=f"Persistent proposal {proposal.get('query_id')}"
    )
    metrics["teacher_anchor"] = "current_persistent_student"
    metrics["signed_frontier"] = config.variant == "correct_wrong_signed"
    metrics["candidate_count"] = len(candidates)
    metrics["ridge_fit_fallback"] = bool(fit_fallback_reason)
    metrics["ridge_fit_fallback_reason"] = fit_fallback_reason
    metrics["ridge_fit_used_candidate_count"] = (
        0 if fit_fallback_reason else len(candidates)
    )
    metrics["support_variant"] = config.variant
    metrics["db_crossing_count"] = float(
        metrics.get("decision_boundary_crossing_count", 0.0)
    )
    metrics["db_eligible_count"] = float(
        metrics.get("decision_boundary_eligible_count", 0.0)
    )
    metrics["regression_count"] = float(
        metrics.get("decision_boundary_regression_count", 0.0)
    )
    metrics["regression_eligible_count"] = float(
        metrics.get("decision_boundary_regression_eligible_count", 0.0)
    )
    return adapter.to(input_device(model)), metrics


def zero_cumulative_audit() -> dict[str, Any]:
    return {
        "episodes": 0,
        "teacher_positions": 0,
        "hindsight_exposed_positions": 0,
        "compared_positions": 0,
        "exact_context_positions": 0,
        "on_policy_positions": 0,
        "response_tokens": 0,
        "optimizer_steps": 0,
        "specialization_no_op_episodes": 0,
        "ridge_support_tokens": 0.0,
        "ridge_db_crossing_count": 0.0,
        "ridge_db_eligible_count": 0.0,
        "ridge_db_regression_count": 0.0,
        "ridge_db_regression_eligible_count": 0.0,
        "style_abs_error_sum": 0.0,
        "style_token_count": 0,
        "task_abs_error_sum": 0.0,
        "task_token_count": 0,
    }


def accumulate_episode_audit(
    cumulative: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(cumulative)
    audit = row["audit"]
    ridge = row["ridge_metrics"]
    style = row["style_task_error"]
    result["episodes"] += 1
    for key in (
        "teacher_positions",
        "hindsight_exposed_positions",
        "compared_positions",
        "exact_context_positions",
        "on_policy_positions",
    ):
        result[key] += int(audit[key])
    result["response_tokens"] += int(row["response_tokens"])
    result["optimizer_steps"] += int(bool(row["optimizer_step"]))
    result["specialization_no_op_episodes"] += int(
        bool(ridge.get("specialization_no_op", False))
    )
    result["ridge_support_tokens"] += float(ridge.get("support_tokens", 0.0))
    result["ridge_db_crossing_count"] += float(
        ridge.get("decision_boundary_crossing_count", 0.0)
    )
    result["ridge_db_eligible_count"] += float(
        ridge.get("decision_boundary_eligible_count", 0.0)
    )
    result["ridge_db_regression_count"] += float(
        ridge.get("decision_boundary_regression_count", 0.0)
    )
    result["ridge_db_regression_eligible_count"] += float(
        ridge.get("decision_boundary_regression_eligible_count", 0.0)
    )
    for key in (
        "style_abs_error_sum",
        "task_abs_error_sum",
    ):
        result[key] += float(style[key])
    for key in ("style_token_count", "task_token_count"):
        result[key] += int(style[key])
    return result


def _audit_with_rates(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    teacher = int(value["teacher_positions"])
    compared = int(value["compared_positions"])
    value["hindsight_exposure_rate"] = (
        float(value["hindsight_exposed_positions"]) / teacher if teacher else 0.0
    )
    value["context_parity"] = (
        float(value["exact_context_positions"]) / compared if compared else 0.0
    )
    eligible = float(value["ridge_db_eligible_count"])
    regression_eligible = float(value["ridge_db_regression_eligible_count"])
    value["ridge_db_crossing_rate"] = (
        float(value["ridge_db_crossing_count"]) / eligible if eligible else 0.0
    )
    value["ridge_db_regression_rate"] = (
        float(value["ridge_db_regression_count"]) / regression_eligible
        if regression_eligible
        else 0.0
    )
    return value


def _capture_trainable_state(model) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise PersistentProtocolError("Persistent student has no trainable parameters")
    return state


def capture_trainable_state(model) -> dict[str, torch.Tensor]:
    """Public snapshot primitive for label-blind query-local evaluations."""
    return _capture_trainable_state(model)


@torch.no_grad()
def _restore_trainable_state(model, state: Mapping[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    if set(state) != {name for name, parameter in named.items() if parameter.requires_grad}:
        raise PersistentProtocolError(
            "Checkpoint trainable-parameter names do not match the LoRA student"
        )
    for name, value in state.items():
        parameter = named[name]
        if tuple(parameter.shape) != tuple(value.shape):
            raise PersistentProtocolError(
                f"Checkpoint shape mismatch for trainable parameter {name}"
            )
        parameter.copy_(value.to(parameter.device, parameter.dtype))


def restore_trainable_state(
    model, state: Mapping[str, torch.Tensor]
) -> None:
    """Restore a snapshot made by :func:`capture_trainable_state`."""
    _restore_trainable_state(model, state)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older Torch does not expose ``weights_only``.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise PersistentProtocolError(f"{path} is not a trainer-state dictionary")
    return value


def _checkpoint_identity(
    config: PersistentConfig,
    hashes: Mapping[str, str],
) -> dict[str, str]:
    config_sha = canonical_json_sha256(config.identity_payload())
    model_sha = canonical_json_sha256(
        {"model_id": config.model_id, "revision": config.revision}
    )
    run_sha = canonical_json_sha256(
        {
            "config_sha256": config_sha,
            "model_identity_sha256": model_sha,
            **dict(hashes),
        }
    )
    return {
        "config_sha256": config_sha,
        "model_identity_sha256": model_sha,
        "query_manifest_sha256": hashes["query_manifest_sha256"],
        "proposal_manifest_sha256": hashes["proposal_manifest_sha256"],
        "run_identity_sha256": run_sha,
    }


def _checkpoint_dir_name(episode: int, *, scientific: bool) -> str:
    return (
        f"episode_{episode:04d}" if scientific else f"rolling_episode_{episode:04d}"
    )


def _publish_checkpoint(
    *,
    model,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    completed_episodes: int,
    scientific: bool,
    cumulative_audit: Mapping[str, Any],
    journal_rows: Sequence[Mapping[str, Any]],
    config: PersistentConfig,
    identity: Mapping[str, str],
) -> Path:
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    name = _checkpoint_dir_name(completed_episodes, scientific=scientific)
    target = checkpoints_dir / name
    if target.exists():
        manifest = _read_json(target / "checkpoint_manifest.json")
        if (
            manifest.get("run_identity_sha256") == identity["run_identity_sha256"]
            and manifest.get("completed_episodes") == completed_episodes
        ):
            return target
        raise PersistentProtocolError(f"Refusing to overwrite mismatched {target}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=checkpoints_dir))
    try:
        model.save_pretrained(temporary)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(temporary)
        trainer_state = {
            "completed_episodes": completed_episodes,
            "trainable_state": _capture_trainable_state(model),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _capture_rng_state(),
            "cumulative_audit": dict(cumulative_audit),
            **dict(identity),
        }
        torch.save(trainer_state, temporary / "trainer_state.pt")
        manifest: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "scientific" if scientific else "rolling",
            "checkpoint_episode": completed_episodes,
            "completed_episodes": completed_episodes,
            "branch": config.branch,
            "variant": config.variant,
            "method_id": config.method_id,
            "model_id": config.model_id,
            "model_revision": config.revision,
            "journal_prefix_sha256": canonical_json_sha256(list(journal_rows)),
            "cumulative_audit": _audit_with_rates(cumulative_audit),
            **dict(identity),
        }
        (temporary / "checkpoint_manifest.json").write_text(
            _canonical_json(manifest) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    pointer_path = checkpoints_dir / "LATEST.json"
    old_target: Optional[Path] = None
    if pointer_path.exists():
        old_value = _read_json(pointer_path)
        old_name = str(old_value.get("checkpoint_dir", ""))
        if old_name:
            old_target = checkpoints_dir / old_name
    _atomic_write_json(
        pointer_path,
        {
            "checkpoint_dir": target.name,
            "completed_episodes": completed_episodes,
            "run_identity_sha256": identity["run_identity_sha256"],
        },
    )
    if (
        old_target is not None
        and old_target != target
        and old_target.name.startswith("rolling_episode_")
    ):
        shutil.rmtree(old_target, ignore_errors=True)
    return target


_PUBLISHED_CHECKPOINT_RE = re.compile(
    r"^(?P<kind>episode|rolling_episode)_(?P<episode>[0-9]+)$"
)
_TEMPORARY_CHECKPOINT_RE = re.compile(
    r"^\.(?:(?:episode|rolling_episode)_[0-9]+|LATEST\.json)\."
)


def _checkpoint_protocol_error(path: Path, message: str) -> PersistentProtocolError:
    return PersistentProtocolError(f"Invalid restart checkpoint {path}: {message}")


def _load_checkpoint_json(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _checkpoint_protocol_error(path, "unreadable JSON") from exc


def _load_checkpoint_state(path: Path) -> dict[str, Any]:
    try:
        return _torch_load(path)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise _checkpoint_protocol_error(path, "unreadable trainer state") from exc


def _validate_checkpoint_candidate(
    checkpoint: Path,
    *,
    identity: Mapping[str, str],
    config: PersistentConfig,
    journal_rows: Sequence[Mapping[str, Any]],
) -> int:
    """Validate one published directory against all durable restart evidence."""
    match = _PUBLISHED_CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None or not checkpoint.is_dir():
        raise _checkpoint_protocol_error(checkpoint, "invalid published directory name")
    episode_from_name = int(match.group("episode"))
    expected_type = "scientific" if match.group("kind") == "episode" else "rolling"
    manifest_path = checkpoint / "checkpoint_manifest.json"
    state_path = checkpoint / "trainer_state.pt"
    if not manifest_path.is_file() or not state_path.is_file():
        raise _checkpoint_protocol_error(
            checkpoint, "missing checkpoint_manifest.json or trainer_state.pt"
        )

    manifest = _load_checkpoint_json(manifest_path)
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise _checkpoint_protocol_error(checkpoint, "unsupported manifest schema")
    try:
        manifest_episode = int(manifest.get("completed_episodes", -1))
        checkpoint_episode = int(manifest.get("checkpoint_episode", -1))
    except (TypeError, ValueError) as exc:
        raise _checkpoint_protocol_error(checkpoint, "non-integer episode metadata") from exc
    if not (
        episode_from_name == manifest_episode == checkpoint_episode
        and 0 <= manifest_episode <= config.episodes
    ):
        raise _checkpoint_protocol_error(checkpoint, "directory/manifest episode mismatch")
    if manifest.get("checkpoint_type") != expected_type:
        raise _checkpoint_protocol_error(checkpoint, "checkpoint type/name mismatch")
    if expected_type == "scientific" and manifest_episode not in set(
        config.scientific_checkpoints
    ):
        raise _checkpoint_protocol_error(checkpoint, "unregistered scientific episode")
    for key, expected in (
        ("branch", config.branch),
        ("variant", config.variant),
        ("method_id", config.method_id),
        ("model_id", config.model_id),
        ("model_revision", config.revision),
    ):
        if manifest.get(key) != expected:
            raise _checkpoint_protocol_error(checkpoint, f"manifest disagrees on {key}")
    for key, expected in identity.items():
        if manifest.get(key) != expected:
            raise _checkpoint_protocol_error(checkpoint, f"manifest disagrees on {key}")
    if manifest_episode > len(journal_rows):
        raise _checkpoint_protocol_error(
            checkpoint, "journal is shorter than the committed checkpoint"
        )
    journal_prefix = list(journal_rows[:manifest_episode])
    if manifest.get("journal_prefix_sha256") != canonical_json_sha256(journal_prefix):
        raise _checkpoint_protocol_error(checkpoint, "journal prefix digest mismatch")
    expected_cumulative = _cumulative_from_rows(journal_prefix)
    if manifest.get("cumulative_audit") != _audit_with_rates(expected_cumulative):
        raise _checkpoint_protocol_error(checkpoint, "manifest cumulative audit mismatch")

    state = _load_checkpoint_state(state_path)
    try:
        state_episode = int(state.get("completed_episodes", -1))
    except (TypeError, ValueError) as exc:
        raise _checkpoint_protocol_error(checkpoint, "invalid trainer-state episode") from exc
    if state_episode != manifest_episode:
        raise _checkpoint_protocol_error(checkpoint, "manifest/trainer-state episode mismatch")
    for key, expected in identity.items():
        if state.get(key) != expected:
            raise _checkpoint_protocol_error(
                checkpoint, f"trainer state disagrees on {key}"
            )
    if state.get("cumulative_audit") != expected_cumulative:
        raise _checkpoint_protocol_error(checkpoint, "trainer cumulative audit mismatch")
    for key in ("trainable_state", "optimizer_state", "rng_state"):
        if not isinstance(state.get(key), Mapping):
            raise _checkpoint_protocol_error(checkpoint, f"trainer state lacks {key}")
    return manifest_episode


def _find_resume_checkpoint(
    output_dir: Path,
    identity: Mapping[str, str],
    *,
    config: PersistentConfig,
    journal_rows: Sequence[Mapping[str, Any]],
    repair_latest: bool = True,
) -> Optional[Path]:
    """Return the newest fully validated checkpoint, including orphaned rolls.

    Publishing a checkpoint directory and publishing ``LATEST.json`` are two
    separate atomic renames.  A kill between them leaves a valid orphaned
    rolling directory, which must be discovered rather than silently rewound.
    Conversely, any published-looking but invalid state is a hard error: an
    older valid checkpoint must never hide corruption in a newer one.
    """
    checkpoints = output_dir / "checkpoints"
    if not checkpoints.exists():
        return None
    if not checkpoints.is_dir():
        raise PersistentProtocolError(f"{checkpoints} is not a directory")

    records: dict[int, Path] = {}
    pointer_path = checkpoints / "LATEST.json"
    for entry in sorted(checkpoints.iterdir(), key=lambda item: item.name):
        if entry.name == pointer_path.name:
            continue
        if _TEMPORARY_CHECKPOINT_RE.match(entry.name):
            # tempfile/mkstemp names are never committed state. A SIGKILL may
            # leave them behind before os.replace; they are safe to ignore.
            continue
        if _PUBLISHED_CHECKPOINT_RE.fullmatch(entry.name) is None:
            raise PersistentProtocolError(
                f"Unexpected entry in checkpoint directory: {entry}"
            )
        episode = _validate_checkpoint_candidate(
            entry,
            identity=identity,
            config=config,
            journal_rows=journal_rows,
        )
        if episode in records:
            raise PersistentProtocolError(
                "Conflicting published checkpoints for episode "
                f"{episode}: {records[episode]} and {entry}"
            )
        records[episode] = entry

    pointer: Optional[dict[str, Any]] = None
    if pointer_path.exists():
        if not pointer_path.is_file():
            raise PersistentProtocolError(f"{pointer_path} is not a file")
        pointer = _load_checkpoint_json(pointer_path)
        pointer_name = str(pointer.get("checkpoint_dir", ""))
        try:
            pointer_episode = int(pointer.get("completed_episodes", -1))
        except (TypeError, ValueError) as exc:
            raise _checkpoint_protocol_error(pointer_path, "invalid episode") from exc
        if pointer.get("run_identity_sha256") != identity["run_identity_sha256"]:
            raise _checkpoint_protocol_error(pointer_path, "run identity mismatch")
        if (
            pointer_episode not in records
            or records[pointer_episode].name != pointer_name
        ):
            raise _checkpoint_protocol_error(
                pointer_path, "does not reference a validated checkpoint"
            )

    if not records:
        if pointer is not None:
            raise _checkpoint_protocol_error(
                pointer_path, "exists without a published checkpoint"
            )
        return None

    episode = max(records)
    checkpoint = records[episode]
    desired_pointer = {
        "checkpoint_dir": checkpoint.name,
        "completed_episodes": episode,
        "run_identity_sha256": identity["run_identity_sha256"],
    }
    if repair_latest and pointer != desired_pointer:
        _atomic_write_json(pointer_path, desired_pointer)
    return checkpoint


def _restore_checkpoint(
    *,
    checkpoint: Path,
    model,
    optimizer: torch.optim.Optimizer,
    identity: Mapping[str, str],
) -> tuple[int, dict[str, Any]]:
    state = _torch_load(checkpoint / "trainer_state.pt")
    for key, expected in identity.items():
        if state.get(key) != expected:
            raise PersistentProtocolError(f"Checkpoint disagrees on {key}")
    _restore_trainable_state(model, state["trainable_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    _restore_rng_state(state["rng_state"])
    cumulative = state.get("cumulative_audit")
    if not isinstance(cumulative, dict):
        raise PersistentProtocolError("Checkpoint lacks cumulative audit")
    return int(state["completed_episodes"]), dict(cumulative)


def _validate_journal(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: PersistentConfig,
    identity: Mapping[str, str],
    queries: Sequence[Mapping[str, str]],
) -> None:
    for index, row in enumerate(rows, 1):
        if row.get("schema_version") != EPISODE_SCHEMA_VERSION:
            raise PersistentProtocolError(f"Journal row {index} has wrong schema")
        if row.get("episode") != index or row.get("stream_index") != index - 1:
            raise PersistentProtocolError(f"Journal is not a strict episode prefix at {index}")
        if (
            row.get("branch") != config.branch
            or row.get("variant") != config.variant
            or row.get("method_id") != config.method_id
            or row.get("run_identity_sha256") != identity["run_identity_sha256"]
        ):
            raise PersistentProtocolError(f"Journal row {index} belongs to another run")
        query = queries[index - 1]
        if (
            row.get("query_id") != query["query_id"]
            or row.get("problem_sha256") != query["problem_sha256"]
        ):
            raise PersistentProtocolError(f"Journal/query mismatch at episode {index}")
        audit = row.get("audit")
        if not isinstance(audit, Mapping):
            raise PersistentProtocolError(f"Journal row {index} lacks raw audit")
        compared = int(audit.get("compared_positions", -1))
        exact = int(audit.get("exact_context_positions", -1))
        exposed = int(audit.get("hindsight_exposed_positions", -1))
        teacher = int(audit.get("teacher_positions", -1))
        if min(compared, exact, exposed, teacher) < 0 or exact > compared or exposed > teacher:
            raise PersistentProtocolError(f"Journal row {index} has impossible audit counts")
        if config.branch == "clean" and (exact != compared or exposed != 0):
            raise PersistentProtocolError(f"Clean journal row {index} violates HER=0/CP=1")
        if config.branch == "privileged" and (exact != 0 or exposed != 0):
            raise PersistentProtocolError(
                f"Privileged journal row {index} violates pre-decision HER=0/CP=0"
            )


def _cumulative_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cumulative = zero_cumulative_audit()
    for row in rows:
        cumulative = accumulate_episode_audit(cumulative, row)
    return cumulative


class SignalController:
    """Defer SIGUSR1 handling to the next episode-safe checkpoint boundary."""

    def __init__(self) -> None:
        self.requested = False
        self._previous: Any = None

    def _handle(self, _signum, _frame) -> None:
        self.requested = True

    def install(self) -> None:
        if hasattr(signal, "SIGUSR1"):
            self._previous = signal.signal(signal.SIGUSR1, self._handle)

    def restore(self) -> None:
        if hasattr(signal, "SIGUSR1") and self._previous is not None:
            signal.signal(signal.SIGUSR1, self._previous)


def _episode_seed(config: PersistentConfig, stream_index: int) -> int:
    return int(config.seed) + stream_index * 100_003


def train_one_episode(
    *,
    model,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    query: Mapping[str, str],
    proposal: Mapping[str, Any],
    stream_index: int,
    config: PersistentConfig,
    run_identity_sha256: str,
) -> dict[str, Any]:
    """Run one label-blind persistent update and return its audit row."""
    started = time.perf_counter()
    output_head = unwrap_causal_lm(model).get_output_embeddings()
    if output_head is None or any(
        parameter.requires_grad for parameter in output_head.parameters()
    ):
        raise PersistentProtocolError(
            "Chunked distillation requires the LM output projection to stay frozen"
        )
    device = input_device(model)
    normal_prompt = problem_prompt(tokenizer, query["problem"])
    privileged_prompt = build_privileged_prompt(tokenizer, query["problem"])
    normal_prompt_ids = _tokenize_prompt(tokenizer, normal_prompt, device)
    privileged_prompt_ids = _tokenize_prompt(tokenizer, privileged_prompt, device)
    max_prompt_tokens = max(
        int(normal_prompt_ids.shape[1]), int(privileged_prompt_ids.shape[1])
    )
    rollout_budget = min(
        config.max_rollout_tokens,
        config.max_sequence_tokens - max_prompt_tokens,
    )
    if rollout_budget <= 0:
        raise PersistentProtocolError(
            f"{query['query_id']} prompts leave no room under the training sequence cap"
        )

    teacher_adapter: Optional[SparseRidgeAdapter] = None
    trust_region_alpha: Optional[float] = None
    trust_region_kl: Optional[float] = None
    if config.branch == "clean":
        if config.teacher_projection_mode == "ridge":
            teacher_adapter, ridge_metrics = _fit_current_student_teacher(
                model, tokenizer, proposal, config
            )
        elif config.teacher_projection_mode == "trust_region":
            ridge_metrics = _ridge_metrics_not_applicable()
    else:
        ridge_metrics = _ridge_metrics_not_applicable()

    seed = _episode_seed(config, stream_index)
    response, prompt_ids, response_ids = generate_response(
        model,
        tokenizer,
        query["problem"],
        adapter=None,
        max_new_tokens=rollout_budget,
        temperature=config.train_temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        seed=seed,
        prompt_override=normal_prompt,
    )
    length = int(response_ids.numel())
    if length <= 0:
        raise PersistentProtocolError(f"{query['query_id']} generated an empty rollout")
    if int(prompt_ids.shape[1]) + length > config.max_sequence_tokens:
        raise PersistentProtocolError("Student sequence exceeded the training cap")

    student_full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    if config.branch == "clean":
        if config.teacher_projection_mode == "ridge":
            teacher_full_ids = student_full_ids.detach().clone()
            teacher_prompt_tokens = int(prompt_ids.shape[1])
        else:
            teacher_full_ids = torch.cat([privileged_prompt_ids, response_ids], dim=1)
            if int(teacher_full_ids.shape[1]) > config.max_sequence_tokens:
                raise PersistentProtocolError(
                    "Privileged-trust-region sequence exceeded the training cap"
                )
            teacher_prompt_tokens = int(privileged_prompt_ids.shape[1])
        teacher_sources = [
            "query",
            "on_policy_prefix",
            "sanitized_skill_card",
            "proposed_candidate_problem",
            "verified_correct_trajectory",
        ]
        if config.variant == "correct_wrong_signed":
            teacher_sources.extend(
                ["verified_wrong_trajectory", "verified_error_frontier"]
            )
        exact_context_positions = length
    else:
        teacher_full_ids = torch.cat([privileged_prompt_ids, response_ids], dim=1)
        if int(teacher_full_ids.shape[1]) > config.max_sequence_tokens:
            raise PersistentProtocolError("Privileged sequence exceeded the training cap")
        if torch.equal(student_full_ids, teacher_full_ids):
            raise PersistentProtocolError("Teacher-only privilege failed to change context")
        exact_context_positions = 0
        teacher_prompt_tokens = int(privileged_prompt_ids.shape[1])
        teacher_sources = [
            "query",
            "on_policy_prefix",
            "predecision_reasoning_method",
        ]

    specialization_no_op = bool(ridge_metrics.get("specialization_no_op", False))
    optimizer_step = not (config.branch == "clean" and specialization_no_op)
    optimizer.zero_grad(set_to_none=True)
    # Transformers activates gradient checkpointing only in training mode.
    # Qwen3 and the frozen LoRA configuration both use zero dropout, so this
    # preserves deterministic same-prefix logits while bounding activations.
    model.train(optimizer_step)
    with torch.set_grad_enabled(optimizer_step):
        student_hidden_all, _ = backbone_forward(
            model,
            input_ids=student_full_ids,
            attention_mask=torch.ones_like(student_full_ids),
            use_cache=False,
        )
    student_start = int(prompt_ids.shape[1]) - 1
    student_hidden = student_hidden_all[:, student_start : student_start + length]
    teacher_hidden_all = None
    teacher_hidden = None
    if config.branch == "clean":
        # Trust-region mode does not use the ridge adapter; teacher logits are
        # computed directly from the privileged-context distillation pass and
        # then exponentially projected toward the student distribution.
        if config.teacher_projection_mode == "trust_region":
            model.eval()
            teacher_start = teacher_prompt_tokens - 1
            with torch.no_grad():
                teacher_hidden_all, _ = backbone_forward(
                    model,
                    input_ids=teacher_full_ids,
                    attention_mask=torch.ones_like(teacher_full_ids),
                    use_cache=False,
                )
                teacher_hidden = teacher_hidden_all[
                    :, teacher_start : teacher_start + length
                ]
                if teacher_hidden.shape[1] != length:
                    raise PersistentProtocolError(
                        "Privileged-trust-region teacher states do not match rollout length"
                    )
            trust_region_alpha, trust_region_kl = _trust_region_alpha(
                model=model,
                student_hidden=student_hidden,
                privileged_hidden=teacher_hidden,
                chunk_size=config.distill_token_chunk_size,
                kl_budget=config.trust_region_kl_budget,
                binary_search_steps=config.trust_region_binary_search_steps,
            )
            model.train(optimizer_step)

            def teacher_for_chunk(
                student_logits: torch.Tensor,
                hidden_chunk: torch.Tensor,
                start: int,
                stop: int,
            ) -> torch.Tensor:
                assert teacher_hidden is not None and trust_region_alpha is not None
                privileged_logits = project_logits(
                    model, teacher_hidden[:, start:stop]
                )
                alpha = float(trust_region_alpha)
                return ((1.0 - alpha) * student_logits) + alpha * privileged_logits
        else:
            assert teacher_adapter is not None

            def teacher_for_chunk(
                student_logits: torch.Tensor,
                hidden_chunk: torch.Tensor,
                _start: int,
                _stop: int,
            ) -> torch.Tensor:
                assert teacher_adapter is not None
                return teacher_adapter.apply_to_logits(student_logits, hidden_chunk)

    else:
        # Privileged teacher activations are fixed targets.  Its no-grad pass
        # remains in evaluation mode, then training mode is restored before
        # the student's streamed backward so checkpointing stays active.
        model.eval()
        with torch.no_grad():
            teacher_hidden_all, _ = backbone_forward(
                model,
                input_ids=teacher_full_ids,
                attention_mask=torch.ones_like(teacher_full_ids),
                use_cache=False,
            )
            teacher_start = teacher_prompt_tokens - 1
            teacher_hidden = teacher_hidden_all[
                :, teacher_start : teacher_start + length
            ]
        model.train(optimizer_step)

        def teacher_for_chunk(
            _student_logits: torch.Tensor,
            _hidden_chunk: torch.Tensor,
            start: int,
            stop: int,
        ) -> torch.Tensor:
            assert teacher_hidden is not None
            return project_logits(model, teacher_hidden[:, start:stop]).detach()

    labels = response_ids.to(student_hidden.device)
    style_task: dict[str, Any] = {
        "partition_version": STYLE_TASK_PARTITION_VERSION,
        "error_definition": STYLE_TASK_ERROR_DEFINITION,
        "style_abs_error_sum": 0.0,
        "style_token_count": 0,
        "task_abs_error_sum": 0.0,
        "task_token_count": 0,
        "other_abs_error_sum": 0.0,
        "other_token_count": 0,
    }

    def observe_chunk(
        _start: int,
        _stop: int,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        _per_token_kl: torch.Tensor,
        chunk_labels: torch.Tensor,
    ) -> None:
        _accumulate_style_task_error(
            style_task,
            style_task_error_accumulators(
                tokenizer, chunk_labels, student_logits, teacher_logits
            ),
        )

    streamed = stream_distillation_chunks(
        student_hidden=student_hidden,
        labels=labels,
        project_student=lambda hidden: project_logits(model, hidden),
        teacher_for_chunk=teacher_for_chunk,
        chunk_size=config.distill_token_chunk_size,
        top_k=config.distill_top_k,
        temperature=config.distill_temperature,
        token_clip=config.distill_token_clip,
        backward=optimizer_step,
        observer=observe_chunk,
    )
    if optimizer_step:
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
        optimizer.step()
    model.train()

    audit = {
        "teacher_positions": length,
        "hindsight_exposed_positions": 0,
        "compared_positions": length,
        "exact_context_positions": exact_context_positions,
        "on_policy_positions": length,
    }
    row = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "branch": config.branch,
        "variant": config.variant,
        "method_id": config.method_id,
        "episode": stream_index + 1,
        "stream_index": stream_index,
        "query_id": query["query_id"],
        "problem_sha256": query["problem_sha256"],
        "source": query["source"],
        "seed": seed,
        "run_identity_sha256": run_identity_sha256,
        "max_sequence_tokens": config.max_sequence_tokens,
        "rollout_token_budget": rollout_budget,
        "student_prompt_tokens": int(prompt_ids.shape[1]),
        "teacher_prompt_tokens": teacher_prompt_tokens,
        "response_tokens": length,
        "optimizer_step": bool(optimizer_step),
        "distill_token_chunk_size": config.distill_token_chunk_size,
        "max_projected_chunk_tokens": streamed.max_chunk_tokens,
        "distillation_loss": streamed.loss,
        "mean_teacher_student_kl": streamed.mean_kl,
        "student_logprob_sum": streamed.student_logprob_sum,
        "student_normalized_logprob": streamed.student_normalized_logprob,
        "teacher_logprob_sum": streamed.teacher_logprob_sum,
        "teacher_normalized_logprob": streamed.teacher_normalized_logprob,
        "teacher_student_normalized_logratio": (
            streamed.teacher_normalized_logprob
            - streamed.student_normalized_logprob
        ),
        "style_task_error": style_task,
        "audit": audit,
        "teacher_context_sources": teacher_sources,
        "student_prefix": response,
        "student_prefix_token_ids": response_ids.detach().cpu()[0].tolist(),
        "student_context_sha256": _token_ids_sha256(student_full_ids),
        "teacher_context_sha256": _token_ids_sha256(teacher_full_ids),
        "privileged_prompt_version": (
            PRIVILEGED_PROMPT_VERSION if config.branch == "privileged" else None
        ),
        "ridge_metrics": ridge_metrics,
        "trust_region": config.teacher_projection_mode == "trust_region",
        "trust_region_alpha": trust_region_alpha,
        "trust_region_kl_budget": config.trust_region_kl_budget,
        "trust_region_achieved_kl": trust_region_kl,
        "episode_seconds": time.perf_counter() - started,
    }

    # Query-local teacher state is explicitly destroyed after its only update.
    del teacher_adapter, streamed
    del student_hidden_all, student_hidden
    if teacher_hidden_all is not None:
        del teacher_hidden_all, teacher_hidden
    gc.collect()
    row["temporary_teacher_destroyed_after_update"] = True
    return row


def run_persistent_training(
    *,
    model,
    tokenizer,
    queries: Sequence[Mapping[str, str]],
    proposals: MutableMapping[str, Mapping[str, Any]],
    config: PersistentConfig,
    output_dir: str | Path,
    input_hashes: Mapping[str, str],
    resume: bool = False,
    runtime_metadata: Optional[Mapping[str, Any]] = None,
    signal_controller: Optional[SignalController] = None,
    proposal_provider: Optional[
        Callable[[Mapping[str, str], int], Mapping[str, Any]]
    ] = None,
    proposal_committer: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Train a persistent branch, optionally proposing support inside the loop."""
    config.validate()
    if len(queries) != config.episodes:
        raise PersistentProtocolError(
            f"Expected {config.episodes} bound queries, received {len(queries)}"
        )
    query_ids = [str(query["query_id"]) for query in queries]
    proposal_ids = list(proposals)
    if config.branch == "clean":
        if proposal_provider is None:
            if set(proposal_ids) != set(query_ids):
                raise PersistentProtocolError("In-memory proposal coverage is not exact")
        elif proposal_ids != query_ids[: len(proposal_ids)]:
            raise PersistentProtocolError(
                "Online proposals must be an exact ordered prefix of the query stream"
            )
        for query in queries[: len(proposal_ids)]:
            _validate_proposal_firewall(proposals[str(query["query_id"])], query)
    elif proposal_ids or proposal_provider is not None or proposal_committer is not None:
        raise PersistentProtocolError(
            "Privileged training must not receive Clean specialization proposals"
        )
    if not hasattr(model, "peft_config"):
        raise PersistentProtocolError("Persistent training requires a PEFT/LoRA model")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise PersistentProtocolError("Persistent LoRA model has no trainable parameters")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    journal_path = destination / "episodes.jsonl"
    complete_path = destination / "COMPLETE.json"
    interrupted_path = destination / "INTERRUPTED.json"
    identity = _checkpoint_identity(config, input_hashes)
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )

    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "branch": config.branch,
        "variant": config.variant,
        "method_id": config.method_id,
        "arguments": config.identity_payload(),
        "runtime": dict(runtime_metadata or {}),
        **identity,
    }
    manifest_path = destination / "run_manifest.json"
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if existing.get("run_identity_sha256") != identity["run_identity_sha256"]:
            raise PersistentProtocolError("Output directory belongs to another run")
    elif resume:
        raise PersistentProtocolError("--resume output is missing run_manifest.json")
    else:
        _atomic_write_json(manifest_path, run_manifest)

    journal_rows = _read_jsonl(journal_path)
    _validate_journal(
        journal_rows,
        config=config,
        identity=identity,
        queries=queries,
    )
    if complete_path.exists():
        if not resume:
            raise PersistentProtocolError("Run is already complete; use --resume to validate")
        complete = _read_json(complete_path)
        if (
            complete.get("run_identity_sha256") != identity["run_identity_sha256"]
            or len(journal_rows) != config.episodes
        ):
            raise PersistentProtocolError("COMPLETE marker disagrees with this run")
        return complete

    checkpoint = _find_resume_checkpoint(
        destination,
        identity,
        config=config,
        journal_rows=journal_rows,
        repair_latest=resume,
    )
    if resume and checkpoint is not None:
        completed, cumulative = _restore_checkpoint(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            identity=identity,
        )
        if len(journal_rows) < completed:
            raise PersistentProtocolError(
                "Journal is shorter than the durable optimizer checkpoint"
            )
        if len(journal_rows) > completed:
            # The optimizer update was not durable unless its checkpoint was
            # published.  Discard only the uncheckpointed suffix and replay it
            # from the restored RNG/optimizer state.
            journal_rows = journal_rows[:completed]
            _atomic_write_jsonl(journal_path, journal_rows)
        recomputed = _cumulative_from_rows(journal_rows)
        if recomputed != cumulative:
            raise PersistentProtocolError(
                "Checkpoint cumulative audit disagrees with its journal prefix"
            )
    else:
        if checkpoint is not None or journal_rows:
            if resume:
                raise PersistentProtocolError(
                    "Resume journal exists without a committed restart checkpoint"
                )
            raise PersistentProtocolError(
                "Nonempty training output requires the explicit --resume flag"
            )
        if interrupted_path.exists():
            raise PersistentProtocolError(
                "Interrupted marker exists without a committed restart checkpoint"
            )
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        completed = 0
        cumulative = zero_cumulative_audit()
        _publish_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            output_dir=destination,
            completed_episodes=0,
            scientific=True,
            cumulative_audit=cumulative,
            journal_rows=journal_rows,
            config=config,
            identity=identity,
        )

    if config.branch == "clean" and proposal_provider is not None and len(proposals) < completed:
        raise PersistentProtocolError(
            "Online proposal prefix is shorter than the restored student checkpoint"
        )

    if interrupted_path.exists():
        interrupted_path.unlink()
    controller = signal_controller or SignalController()
    install_here = signal_controller is None
    if install_here:
        controller.install()
    last_checkpoint: Optional[Path] = None
    try:
        for stream_index in range(completed, config.episodes):
            if controller.requested:
                last_checkpoint = _publish_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    output_dir=destination,
                    completed_episodes=completed,
                    scientific=completed in config.scientific_checkpoints,
                    cumulative_audit=cumulative,
                    journal_rows=journal_rows,
                    config=config,
                    identity=identity,
                )
                break
            query = queries[stream_index]
            query_id = str(query["query_id"])
            proposal: Mapping[str, Any] = {}
            if config.branch == "clean":
                proposal = proposals.get(query_id, {})
                if not proposal:
                    if proposal_provider is None:
                        raise PersistentProtocolError(
                            f"Missing proposal for episode query {query_id}"
                        )
                    proposal = dict(proposal_provider(query, stream_index))
                    _validate_proposal_firewall(proposal, query)
                    if proposal_committer is not None:
                        proposal_committer(proposal)
                    proposals[query_id] = proposal
            row = train_one_episode(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                query=query,
                proposal=proposal,
                stream_index=stream_index,
                config=config,
                run_identity_sha256=identity["run_identity_sha256"],
            )
            journal_rows.append(row)
            cumulative = accumulate_episode_audit(cumulative, row)
            completed = stream_index + 1
            # Commit the auditable row first.  A crash before the next durable
            # optimizer checkpoint makes resume truncate and replay this suffix.
            _atomic_write_jsonl(journal_path, journal_rows)
            scientific = completed in config.scientific_checkpoints
            rolling = completed % config.rolling_checkpoint_interval == 0
            if scientific or rolling or controller.requested:
                last_checkpoint = _publish_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    output_dir=destination,
                    completed_episodes=completed,
                    scientific=scientific,
                    cumulative_audit=cumulative,
                    journal_rows=journal_rows,
                    config=config,
                    identity=identity,
                )
            if controller.requested:
                break
    finally:
        if install_here:
            controller.restore()

    if completed < config.episodes:
        if last_checkpoint is None:
            last_checkpoint = _publish_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                output_dir=destination,
                completed_episodes=completed,
                scientific=completed in config.scientific_checkpoints,
                cumulative_audit=cumulative,
                journal_rows=journal_rows,
                config=config,
                identity=identity,
            )
        result = {
            "status": "interrupted",
            "completed_episodes": completed,
            "restart_checkpoint": str(last_checkpoint),
            "cumulative_audit": _audit_with_rates(cumulative),
            **identity,
        }
        _atomic_write_json(interrupted_path, result)
        return result

    if last_checkpoint is None or completed in config.scientific_checkpoints:
        final_dir = destination / "checkpoints" / _checkpoint_dir_name(
            completed, scientific=True
        )
        if not final_dir.exists():
            last_checkpoint = _publish_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                output_dir=destination,
                completed_episodes=completed,
                scientific=True,
                cumulative_audit=cumulative,
                journal_rows=journal_rows,
                config=config,
                identity=identity,
            )
        else:
            last_checkpoint = final_dir
    result = {
        "status": "complete",
        "completed_episodes": completed,
        "final_checkpoint": str(last_checkpoint),
        "cumulative_audit": _audit_with_rates(cumulative),
        "journal_sha256": file_sha256(journal_path),
        **identity,
    }
    _atomic_write_json(complete_path, result)
    return result
