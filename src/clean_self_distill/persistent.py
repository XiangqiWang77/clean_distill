"""Persistent Locality-Guided Self-Distillation training.

This module implements the long-horizon protocol used by the empirical study.
Unlike the legacy query-local evaluation path, the LoRA student and its AdamW
state are never reset between episodes.  Target answers and reference solutions
are physically absent from both the query stream and this trainer's API.

Independently trained branches share the same query order, rollout budget,
initialization, and optimizer configuration:

* ``clean`` constructs the exponential projection of a pre-decision teacher
  into a student-centered trajectory-level KL trust region (HER=0).
* ``privileged`` gives only the teacher a fixed pre-decision reasoning-method
  instruction (HER=0, CP=0).  It never receives an answer, solution, feedback,
  or future target token.
* ``veto`` applies the published Veto product-of-experts reformulation to that
  same privileged teacher and fits the detached target with forward KL.

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
import resource
import shutil
import signal
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from .heldout import HeldoutProtocolError, load_query_only_manifest
from .io import canonical_json_sha256
from .generation import generate_response, problem_prompt
from .runtime import (
    backbone_forward,
    input_device,
    project_logits,
    render_chat,
    unwrap_causal_lm,
)
from .streaming_distill import stream_distillation_chunks
from .veto import (
    VETO_SCHEDULES,
    VETO_TARGET_VERSION,
    scheduled_veto_beta,
    veto_target_logits,
)


EPISODE_SCHEMA_VERSION = "clean-self-distill-persistent-episode-v1"
CHECKPOINT_SCHEMA_VERSION = "clean-self-distill-persistent-checkpoint-v1"
RUN_SCHEMA_VERSION = "clean-self-distill-persistent-run-v1"
STYLE_TASK_PARTITION_VERSION = "rlcsd-style-task-v1"
STYLE_TASK_ERROR_DEFINITION = (
    "abs_teacher_minus_student_realized_token_logprob_pre_update"
)
PRIVILEGED_PROMPT_VERSION = "predecision-reasoning-method-v1"
REQUEUE_EXIT_CODE = 75

BRANCHES = frozenset({"clean", "privileged", "veto"})
VARIANTS = frozenset({"trust_region", "adaptive_target_reformulation"})
LGSD_DISTILLATION_KL_DIRECTION = "projected_teacher_to_student_forward_kl_v1"
LEGACY_REVERSE_KL_DIRECTION = "student_to_projected_teacher_reverse_kl_v1"
VETO_DISTILLATION_KL_DIRECTION = "adaptive_target_to_student_forward_kl_v1"
PROJECTION_KL_DIRECTION = "projected_teacher_to_pre_update_student_forward_kl_v1"
PROJECTION_SCOPES = frozenset({"trajectory", "token", "fixed"})
PROJECTION_PATHS = frozenset({"exponential", "arithmetic"})
STUDENT_KL_DIRECTIONS = frozenset({"reverse", "forward"})

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
    trust_region_kl_budget: float = 0.004
    trust_region_binary_search_steps: int = 5
    projection_scope: str = "trajectory"
    projection_path: str = "exponential"
    fixed_projection_alpha: float = 0.5595703125
    # LGSD directly fits the detached projected target.  ``reverse`` is kept
    # only to reproduce checkpoints from the legacy TRSD objective.
    student_kl_direction: str = "forward"
    same_prefix_scoring: bool = True
    update_guard: bool = False
    # Veto (Jang et al., Findings of ACL 2026) uses a global product-of-experts
    # coefficient.  Defaults match the reasoning-task setting in the paper and
    # released implementation: beta linearly decays from 0.8 toward zero.
    veto_beta_start: float = 0.8
    veto_beta_end: float = 0.0
    veto_beta_schedule: str = "linear"

    def validate(self) -> None:
        if self.branch not in BRANCHES:
            raise PersistentProtocolError(f"Unknown branch {self.branch!r}")
        if self.variant not in VARIANTS:
            raise PersistentProtocolError(f"Unknown variant {self.variant!r}")
        if self.branch == "veto" and self.variant != "adaptive_target_reformulation":
            raise PersistentProtocolError(
                "Veto requires variant='adaptive_target_reformulation'"
            )
        if self.branch != "veto" and self.variant != "trust_region":
            raise PersistentProtocolError(
                "LGSD/OPSD require variant='trust_region'"
            )
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
        ):
            if int(getattr(self, name)) <= 0:
                raise PersistentProtocolError(f"{name} must be positive")
        for name in (
            "learning_rate",
            "max_grad_norm",
            "distill_temperature",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise PersistentProtocolError(f"{name} must be finite and positive")
        if self.weight_decay < 0 or self.distill_token_clip < 0:
            raise PersistentProtocolError("weight decay and token clip cannot be negative")
        if self.train_temperature < 0 or not 0 < self.top_p <= 1:
            raise PersistentProtocolError("invalid rollout sampling parameters")
        if self.trust_region_kl_budget <= 0:
            raise PersistentProtocolError(
                "trust_region_kl_budget must be positive"
            )
        if self.projection_scope not in PROJECTION_SCOPES:
            raise PersistentProtocolError(
                f"Unknown projection_scope {self.projection_scope!r}"
            )
        if self.projection_path not in PROJECTION_PATHS:
            raise PersistentProtocolError(
                f"Unknown projection_path {self.projection_path!r}"
            )
        if self.student_kl_direction not in STUDENT_KL_DIRECTIONS:
            raise PersistentProtocolError(
                f"Unknown student_kl_direction {self.student_kl_direction!r}"
            )
        if not math.isfinite(float(self.fixed_projection_alpha)) or not (
            0.0 <= float(self.fixed_projection_alpha) <= 1.0
        ):
            raise PersistentProtocolError("fixed_projection_alpha must lie in [0,1]")
        if self.update_guard and self.branch != "clean":
            raise PersistentProtocolError("update_guard is defined only for LGSD")
        if self.branch != "clean" and (
            self.projection_scope != "trajectory"
            or self.projection_path != "exponential"
            or not self.same_prefix_scoring
        ):
            raise PersistentProtocolError(
                "projection geometry ablations are defined only for LGSD"
            )
        if self.veto_beta_schedule not in VETO_SCHEDULES:
            raise PersistentProtocolError(
                f"Unknown Veto beta schedule {self.veto_beta_schedule!r}"
            )
        for name in ("veto_beta_start", "veto_beta_end"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise PersistentProtocolError(
                    f"{name} must be finite and nonnegative"
                )
        if self.branch == "veto" and self.student_kl_direction != "forward":
            raise PersistentProtocolError("Veto is defined with forward KL")
        if self.branch == "veto" and self.distill_temperature != 1.0:
            raise PersistentProtocolError(
                "Formula-faithful Veto requires distill_temperature=1"
            )
        if self.branch != "veto" and (
            self.veto_beta_start != 0.8
            or self.veto_beta_end != 0.0
            or self.veto_beta_schedule != "linear"
        ):
            raise PersistentProtocolError("Veto beta settings are defined only for Veto")
        if not int(self.trust_region_binary_search_steps) > 0:
            raise PersistentProtocolError(
                "trust_region_binary_search_steps must be a positive integer"
            )

    @property
    def method_id(self) -> str:
        if self.branch == "veto":
            return "veto:adaptive_target_reformulation:forward_kl_v1"
        if self.branch == "clean":
            if self.student_kl_direction == "forward":
                path_name = (
                    "geometric_kl_ball"
                    if self.projection_path == "exponential"
                    else "arithmetic_probability_path"
                )
                value = f"lgsd:{path_name}_projection:forward_kl_v1"
            else:
                # Preserve the exact identity of legacy reverse-KL runs so
                # they remain readable without being mislabeled as LGSD-v2.
                value = "trsd:exponential_teacher_projection"
            if self.projection_scope == "token":
                value += ":independent_token_budgets_v1"
            elif self.projection_scope == "fixed":
                value += f":fixed_alpha_{self.fixed_projection_alpha:.8f}"
            if (
                self.projection_path == "arithmetic"
                and self.student_kl_direction == "reverse"
            ):
                value += ":arithmetic_probability_path_v1"
            if not self.same_prefix_scoring:
                value += ":independent_prefix_scoring_v1"
            if self.update_guard:
                value += ":update_guard_v1"
            return value
        if self.student_kl_direction == "forward":
            return "opsd:raw_privileged_teacher:forward_kl_v1"
        return "privileged:predecision_method"

    @property
    def distillation_kl_direction(self) -> str:
        if self.branch == "veto":
            return VETO_DISTILLATION_KL_DIRECTION
        if self.student_kl_direction == "forward":
            return LGSD_DISTILLATION_KL_DIRECTION
        return LEGACY_REVERSE_KL_DIRECTION

    def identity_payload(self) -> dict[str, Any]:
        value = asdict(self)
        # Preserve the byte-identical identity of every pre-guard run.  This
        # lets active restartable jobs safely load code that adds the opt-in
        # guard without invalidating an existing checkpoint.
        if not self.update_guard:
            value.pop("update_guard")
        # Veto-only defaults must not change the identities of already running
        # LGSD/OPSD jobs when this baseline is added to the codebase.
        if self.branch != "veto":
            value.pop("veto_beta_start")
            value.pop("veto_beta_end")
            value.pop("veto_beta_schedule")
        else:
            for name in (
                "trust_region_kl_budget",
                "trust_region_binary_search_steps",
                "projection_scope",
                "projection_path",
                "fixed_projection_alpha",
            ):
                value.pop(name)
        # Keep completed legacy TRSD/Privilege-SD identities byte-for-byte
        # stable.  Forward KL is intentionally retained in every new identity,
        # even though it is now the public CLI default.
        defaults = {
            "projection_scope": "trajectory",
            "projection_path": "exponential",
            "fixed_projection_alpha": 0.5595703125,
            "student_kl_direction": "reverse",
            "same_prefix_scoring": True,
        }
        for name, default in defaults.items():
            if name in value and value[name] == default:
                value.pop(name)
        value["scientific_checkpoints"] = list(self.scientific_checkpoints)
        value.update(
            {
                "run_schema_version": RUN_SCHEMA_VERSION,
                "method_id": self.method_id,
                "privileged_prompt_version": PRIVILEGED_PROMPT_VERSION,
                "style_task_partition_version": STYLE_TASK_PARTITION_VERSION,
                "distillation_kl_direction": self.distillation_kl_direction,
            }
        )
        if self.branch == "veto":
            value["target_reformulation"] = VETO_TARGET_VERSION
        return value


def load_persistent_inputs(
    query_path: str | Path,
    *,
    episodes: int,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Load a physically target-free query stream."""
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
    hashes = {
        "query_manifest_sha256": file_sha256(query_path),
        "teacher_signal_sha256": canonical_json_sha256(
            {"mode": "predecision-exponential-projection-v1"}
        ),
    }
    return queries, hashes


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


def _append_jsonl_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    payload = "".join(_canonical_json(row) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)


def _projected_teacher_logits(
    student_logits: torch.Tensor,
    privileged_logits: torch.Tensor,
    alpha: float | torch.Tensor,
    *,
    path: str,
) -> torch.Tensor:
    """Move from student to teacher on a logit or probability geodesic."""
    if path not in PROJECTION_PATHS:
        raise PersistentProtocolError(f"Unknown projection path {path!r}")
    alpha_tensor = torch.as_tensor(
        alpha, dtype=torch.float32, device=student_logits.device
    )
    while alpha_tensor.ndim < student_logits.ndim:
        alpha_tensor = alpha_tensor.unsqueeze(-1)
    if path == "exponential":
        return (
            (1.0 - alpha_tensor) * student_logits.float()
            + alpha_tensor * privileged_logits.float()
        )
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(privileged_logits.float(), dim=-1)
    return torch.logaddexp(
        student_log_probs + torch.log1p(-alpha_tensor),
        teacher_log_probs + torch.log(alpha_tensor),
    )


def _target_to_student_kl(
    student_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    target_log_probs = F.log_softmax(target_logits.float(), dim=-1)
    target_probs = target_log_probs.exp()
    return torch.sum(target_probs * (target_log_probs - student_log_probs), dim=-1)


def _traced_mean_teacher_kl(
    student_logits: torch.Tensor,
    privileged_logits: torch.Tensor,
    alpha: float,
    *,
    path: str = "exponential",
) -> torch.Tensor:
    """Return per-position KL(projected target || student) for a fixed chunk."""
    if not (0.0 <= alpha <= 1.0):
        raise PersistentProtocolError("trust-region alpha must be in [0,1]")
    projected = _projected_teacher_logits(
        student_logits, privileged_logits, alpha, path=path
    )
    return _target_to_student_kl(student_logits, projected)


@dataclass(frozen=True)
class ProjectionPlan:
    """Memory-bounded projection coefficients and exact KL diagnostics."""

    alpha: float
    achieved_kl: float
    raw_kl: float
    cap_hits: int
    token_alphas: Optional[torch.Tensor] = None


def _trust_region_projection(
    *,
    model,
    student_hidden: torch.Tensor,
    privileged_hidden: torch.Tensor,
    chunk_size: int,
    kl_budget: float,
    binary_search_steps: int,
    scope: str,
    path: str,
    fixed_alpha: float,
) -> ProjectionPlan:
    """Construct trajectory, token-wise, or fixed projection coefficients."""
    if student_hidden.shape != privileged_hidden.shape:
        raise PersistentProtocolError("Student/privileged hidden mismatch")
    if scope not in PROJECTION_SCOPES:
        raise PersistentProtocolError(f"Unknown projection scope {scope!r}")

    def logits_for(start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            project_logits(model, student_hidden[:, start:stop]),
            project_logits(model, privileged_hidden[:, start:stop]),
        )

    def mean_kl(alpha_value: float) -> float:
        total_kl = 0.0
        total_tokens = 0
        with torch.no_grad():
            for start in range(0, student_hidden.shape[1], chunk_size):
                stop = min(start + chunk_size, student_hidden.shape[1])
                student_logits, privileged_logits = logits_for(start, stop)
                projected = _projected_teacher_logits(
                    student_logits, privileged_logits, alpha_value, path=path
                )
                total_kl += float(
                    _target_to_student_kl(student_logits, projected).sum().item()
                )
                total_tokens += stop - start
        return total_kl / float(total_tokens)

    raw_kl = mean_kl(1.0)
    if scope == "fixed":
        achieved = mean_kl(float(fixed_alpha))
        return ProjectionPlan(
            alpha=float(fixed_alpha),
            achieved_kl=achieved,
            raw_kl=raw_kl,
            cap_hits=int(achieved > kl_budget),
        )

    if scope == "trajectory":
        if raw_kl <= kl_budget:
            return ProjectionPlan(1.0, raw_kl, raw_kl, 0)
        low, high = 0.0, 1.0
        for _ in range(int(binary_search_steps)):
            mid = (low + high) / 2.0
            if mean_kl(mid) <= kl_budget:
                low = mid
            else:
                high = mid
        return ProjectionPlan(low, mean_kl(low), raw_kl, 1)

    alpha_chunks: list[torch.Tensor] = []
    achieved_sum = 0.0
    cap_hits = 0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, student_hidden.shape[1], chunk_size):
            stop = min(start + chunk_size, student_hidden.shape[1])
            student_logits, privileged_logits = logits_for(start, stop)
            raw_per_token = _target_to_student_kl(student_logits, privileged_logits)
            active = raw_per_token > float(kl_budget)
            low = torch.zeros_like(raw_per_token)
            high = torch.ones_like(raw_per_token)
            for _ in range(int(binary_search_steps)):
                mid = (low + high) / 2.0
                projected = _projected_teacher_logits(
                    student_logits, privileged_logits, mid, path=path
                )
                value = _target_to_student_kl(student_logits, projected)
                feasible = value <= float(kl_budget)
                low = torch.where(feasible, mid, low)
                high = torch.where(feasible, high, mid)
            alpha_chunk = torch.where(active, low, torch.ones_like(low))
            projected = _projected_teacher_logits(
                student_logits, privileged_logits, alpha_chunk, path=path
            )
            achieved = _target_to_student_kl(student_logits, projected)
            alpha_chunks.append(alpha_chunk.detach())
            achieved_sum += float(achieved.sum().item())
            cap_hits += int(active.sum().item())
            total_tokens += stop - start
    token_alphas = torch.cat(alpha_chunks, dim=1)
    return ProjectionPlan(
        alpha=float(token_alphas.mean().item()),
        achieved_kl=achieved_sum / float(total_tokens),
        raw_kl=raw_kl,
        cap_hits=cap_hits,
        token_alphas=token_alphas,
    )


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
    plan = _trust_region_projection(
        model=model,
        student_hidden=student_hidden,
        privileged_hidden=privileged_hidden,
        chunk_size=chunk_size,
        kl_budget=kl_budget,
        binary_search_steps=binary_search_steps,
        scope="trajectory",
        path="exponential",
        fixed_alpha=1.0,
    )
    return plan.alpha, plan.achieved_kl


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
        "teacher_signal_sha256": hashes["teacher_signal_sha256"],
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
            "distillation_kl_direction": config.distillation_kl_direction,
            "projection_kl_direction": (
                PROJECTION_KL_DIRECTION
                if config.branch == "clean"
                else "not_applicable_unprojected_target"
            ),
            "projection_scope": (
                config.projection_scope if config.branch == "clean" else "raw"
            ),
            "projection_path": (
                config.projection_path if config.branch == "clean" else "raw"
            ),
            "projection_kl_budget": (
                config.trust_region_kl_budget if config.branch == "clean" else None
            ),
            "target_reformulation": (
                VETO_TARGET_VERSION if config.branch == "veto" else None
            ),
            "veto_beta_schedule": (
                {
                    "name": config.veto_beta_schedule,
                    "start": config.veto_beta_start,
                    "end": config.veto_beta_end,
                    "step_denominator": config.episodes,
                }
                if config.branch == "veto"
                else None
            ),
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
    manifest_direction = manifest.get("distillation_kl_direction")
    if manifest_direction is None and config.student_kl_direction == "reverse":
        # Checkpoints published before the direction field was added are
        # unambiguously reverse KL from their legacy method identity.
        pass
    elif manifest_direction != config.distillation_kl_direction:
        raise _checkpoint_protocol_error(
            checkpoint, "manifest disagrees on distillation_kl_direction"
        )
    if config.branch == "veto":
        expected_schedule = {
            "name": config.veto_beta_schedule,
            "start": config.veto_beta_start,
            "end": config.veto_beta_end,
            "step_denominator": config.episodes,
        }
        if manifest.get("target_reformulation") != VETO_TARGET_VERSION:
            raise _checkpoint_protocol_error(
                checkpoint, "manifest disagrees on Veto target reformulation"
            )
        if manifest.get("veto_beta_schedule") != expected_schedule:
            raise _checkpoint_protocol_error(
                checkpoint, "manifest disagrees on Veto beta schedule"
            )
    if config.student_kl_direction == "forward" and config.branch == "clean":
        if manifest.get("projection_kl_direction") != PROJECTION_KL_DIRECTION:
            raise _checkpoint_protocol_error(
                checkpoint, "manifest disagrees on projection_kl_direction"
            )
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
        if config.branch in {"clean", "privileged", "veto"} and (
            exact != 0 or exposed != 0
        ):
            raise PersistentProtocolError(
                f"Pre-decision journal row {index} violates HER=0/CP=0"
            )
        if config.branch == "veto":
            expected_beta = scheduled_veto_beta(
                step=index - 1,
                total_steps=config.episodes,
                beta_start=config.veto_beta_start,
                beta_end=config.veto_beta_end,
                schedule=config.veto_beta_schedule,
            )
            try:
                recorded_beta = float(row["veto_beta"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PersistentProtocolError(
                    f"Veto journal row {index} lacks a valid beta"
                ) from exc
            if not math.isclose(recorded_beta, expected_beta, abs_tol=1e-12):
                raise PersistentProtocolError(
                    f"Veto journal row {index} disagrees with its beta schedule"
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
    stream_index: int,
    config: PersistentConfig,
    run_identity_sha256: str,
) -> dict[str, Any]:
    """Run one label-blind persistent update and return its audit row."""
    cuda_memory_baseline = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        cuda_memory_baseline = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    phase_seconds = {
        "rollout": 0.0,
        "teacher": 0.0,
        "target": 0.0,
        "update": 0.0,
    }

    def sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
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

    trust_region_alpha: Optional[float] = None
    trust_region_kl: Optional[float] = None
    trust_region_raw_kl: Optional[float] = None
    projection_cap_hits = 0
    projection_token_alphas: Optional[torch.Tensor] = None
    veto_beta: Optional[float] = None
    independent_teacher_response_tokens: Optional[int] = None

    seed = _episode_seed(config, stream_index)
    sync_cuda()
    phase_started = time.perf_counter()
    response, prompt_ids, response_ids = generate_response(
        model,
        tokenizer,
        query["problem"],
        max_new_tokens=rollout_budget,
        temperature=config.train_temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        seed=seed,
        prompt_override=normal_prompt,
    )
    sync_cuda()
    phase_seconds["rollout"] += time.perf_counter() - phase_started
    length = int(response_ids.numel())
    if length <= 0:
        raise PersistentProtocolError(f"{query['query_id']} generated an empty rollout")
    if int(prompt_ids.shape[1]) + length > config.max_sequence_tokens:
        raise PersistentProtocolError("Student sequence exceeded the training cap")

    student_full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    teacher_prefix_ids = response_ids
    if config.branch == "clean" and not config.same_prefix_scoring:
        sync_cuda()
        phase_started = time.perf_counter()
        _, _, independent_ids = generate_response(
            model,
            tokenizer,
            query["problem"],
            max_new_tokens=rollout_budget,
            temperature=config.train_temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            seed=seed + 50_000_003,
            prompt_override=privileged_prompt,
        )
        sync_cuda()
        phase_seconds["teacher"] += time.perf_counter() - phase_started
        independent_teacher_response_tokens = int(independent_ids.numel())
        if independent_teacher_response_tokens >= length:
            teacher_prefix_ids = independent_ids[:, :length]
        else:
            filler_id = tokenizer.eos_token_id
            if filler_id is None:
                filler_id = tokenizer.pad_token_id or 0
            padding = torch.full(
                (1, length - independent_teacher_response_tokens),
                int(filler_id),
                dtype=independent_ids.dtype,
                device=independent_ids.device,
            )
            teacher_prefix_ids = torch.cat([independent_ids, padding], dim=1)
    teacher_full_ids = torch.cat([privileged_prompt_ids, teacher_prefix_ids], dim=1)
    if int(teacher_full_ids.shape[1]) > config.max_sequence_tokens:
        raise PersistentProtocolError("Privileged sequence exceeded the training cap")
    if torch.equal(student_full_ids, teacher_full_ids):
        raise PersistentProtocolError("Teacher-only privilege failed to change context")
    exact_context_positions = 0
    teacher_prompt_tokens = int(privileged_prompt_ids.shape[1])
    teacher_sources = ["query", "predecision_reasoning_method"]
    if config.same_prefix_scoring:
        teacher_sources.append("on_policy_prefix")
    else:
        teacher_sources.append("independent_teacher_prefix_position_aligned")
    if config.branch == "clean":
        teacher_sources.append(
            f"student_centered_{config.projection_path}_projection"
        )
    elif config.branch == "veto":
        teacher_sources.extend(
            ["pre_update_student_distribution", VETO_TARGET_VERSION]
        )

    optimizer_step = True
    optimizer.zero_grad(set_to_none=True)
    # Transformers activates gradient checkpointing only in training mode.
    # Qwen3 and the frozen LoRA configuration both use zero dropout, so this
    # preserves deterministic same-prefix logits while bounding activations.
    sync_cuda()
    phase_started = time.perf_counter()
    model.train(optimizer_step)
    with torch.set_grad_enabled(optimizer_step):
        student_hidden_all, _ = backbone_forward(
            model,
            input_ids=student_full_ids,
            attention_mask=torch.ones_like(student_full_ids),
            use_cache=False,
        )
    sync_cuda()
    phase_seconds["update"] += time.perf_counter() - phase_started
    student_start = int(prompt_ids.shape[1]) - 1
    student_hidden = student_hidden_all[:, student_start : student_start + length]
    teacher_hidden_all = None
    teacher_hidden = None
    if config.branch == "clean":
        sync_cuda()
        phase_started = time.perf_counter()
        model.eval()
        teacher_start = teacher_prompt_tokens - 1
        with torch.no_grad():
            teacher_hidden_all, _ = backbone_forward(
                model,
                input_ids=teacher_full_ids,
                attention_mask=torch.ones_like(teacher_full_ids),
                use_cache=False,
            )
            teacher_hidden = teacher_hidden_all[:, teacher_start : teacher_start + length]
            if teacher_hidden.shape[1] != length:
                raise PersistentProtocolError(
                    "Privileged teacher states do not match rollout length"
                )
        sync_cuda()
        phase_seconds["teacher"] += time.perf_counter() - phase_started
        sync_cuda()
        phase_started = time.perf_counter()
        projection_plan = _trust_region_projection(
            model=model,
            student_hidden=student_hidden,
            privileged_hidden=teacher_hidden,
            chunk_size=config.distill_token_chunk_size,
            kl_budget=config.trust_region_kl_budget,
            binary_search_steps=config.trust_region_binary_search_steps,
            scope=config.projection_scope,
            path=config.projection_path,
            fixed_alpha=config.fixed_projection_alpha,
        )
        trust_region_alpha = projection_plan.alpha
        trust_region_kl = projection_plan.achieved_kl
        trust_region_raw_kl = projection_plan.raw_kl
        projection_cap_hits = projection_plan.cap_hits
        projection_token_alphas = projection_plan.token_alphas
        sync_cuda()
        phase_seconds["target"] += time.perf_counter() - phase_started
        model.train(optimizer_step)

        def teacher_for_chunk(
            student_logits: torch.Tensor,
            _hidden_chunk: torch.Tensor,
            start: int,
            stop: int,
        ) -> torch.Tensor:
            assert teacher_hidden is not None and trust_region_alpha is not None
            privileged_logits = project_logits(model, teacher_hidden[:, start:stop])
            alpha: float | torch.Tensor
            if projection_token_alphas is None:
                alpha = float(trust_region_alpha)
            else:
                alpha = projection_token_alphas[:, start:stop]
            return _projected_teacher_logits(
                student_logits,
                privileged_logits,
                alpha,
                path=config.projection_path,
            )

    else:
        # Privileged teacher activations are fixed targets.  Its no-grad pass
        # remains in evaluation mode, then training mode is restored before
        # the student's streamed backward so checkpointing stays active.
        sync_cuda()
        phase_started = time.perf_counter()
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
        sync_cuda()
        phase_seconds["teacher"] += time.perf_counter() - phase_started
        model.train(optimizer_step)

        if config.branch == "veto":
            veto_beta = scheduled_veto_beta(
                step=stream_index,
                total_steps=config.episodes,
                beta_start=config.veto_beta_start,
                beta_end=config.veto_beta_end,
                schedule=config.veto_beta_schedule,
            )

            def teacher_for_chunk(
                student_logits: torch.Tensor,
                _hidden_chunk: torch.Tensor,
                start: int,
                stop: int,
            ) -> torch.Tensor:
                assert teacher_hidden is not None and veto_beta is not None
                privileged_logits = project_logits(
                    model, teacher_hidden[:, start:stop]
                )
                return veto_target_logits(
                    student_logits, privileged_logits, veto_beta
                )

        else:

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

    # Materialize and score the fixed target in a no-backward pass.  Besides
    # giving target construction its own wall-clock measurement, this exposes
    # the realized-trajectory direction needed by the optional update guard.
    sync_cuda()
    phase_started = time.perf_counter()
    target_metrics = stream_distillation_chunks(
        student_hidden=student_hidden,
        labels=labels,
        project_student=lambda hidden: project_logits(model, hidden),
        teacher_for_chunk=teacher_for_chunk,
        chunk_size=config.distill_token_chunk_size,
        top_k=config.distill_top_k,
        temperature=config.distill_temperature,
        token_clip=config.distill_token_clip,
        backward=False,
        observer=observe_chunk,
        kl_direction=config.student_kl_direction,
    )
    sync_cuda()
    phase_seconds["target"] += time.perf_counter() - phase_started

    realized_target_advantage = (
        target_metrics.teacher_normalized_logprob
        - target_metrics.student_normalized_logprob
    )
    guard_rejected = bool(config.update_guard and realized_target_advantage <= 0.0)
    optimizer_step = not guard_rejected
    streamed = target_metrics
    if optimizer_step:
        sync_cuda()
        phase_started = time.perf_counter()
        streamed = stream_distillation_chunks(
            student_hidden=student_hidden,
            labels=labels,
            project_student=lambda hidden: project_logits(model, hidden),
            teacher_for_chunk=teacher_for_chunk,
            chunk_size=config.distill_token_chunk_size,
            top_k=config.distill_top_k,
            temperature=config.distill_temperature,
            token_clip=config.distill_token_clip,
            backward=True,
            observer=None,
            kl_direction=config.student_kl_direction,
        )
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
        optimizer.step()
        sync_cuda()
        phase_seconds["update"] += time.perf_counter() - phase_started
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
        "update_guard": bool(config.update_guard),
        "guard_rejected": guard_rejected,
        "realized_target_logprob_advantage": realized_target_advantage,
        "distill_token_chunk_size": config.distill_token_chunk_size,
        "max_projected_chunk_tokens": streamed.max_chunk_tokens,
        "distillation_loss": streamed.loss,
        "distillation_kl_direction": config.distillation_kl_direction,
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
        "same_prefix_scoring": bool(config.same_prefix_scoring),
        "independent_teacher_response_tokens": independent_teacher_response_tokens,
        "student_prefix": response,
        "student_prefix_token_ids": response_ids.detach().cpu()[0].tolist(),
        "student_context_sha256": _token_ids_sha256(student_full_ids),
        "teacher_context_sha256": _token_ids_sha256(teacher_full_ids),
        "privileged_prompt_version": (
            PRIVILEGED_PROMPT_VERSION
            if config.branch in {"privileged", "veto"}
            else None
        ),
        "target_reformulation": (
            VETO_TARGET_VERSION if config.branch == "veto" else None
        ),
        "veto_beta": veto_beta,
        "veto_beta_start": (
            config.veto_beta_start if config.branch == "veto" else None
        ),
        "veto_beta_end": (
            config.veto_beta_end if config.branch == "veto" else None
        ),
        "veto_beta_schedule": (
            config.veto_beta_schedule if config.branch == "veto" else None
        ),
        "trust_region": config.branch == "clean",
        "trust_region_alpha": trust_region_alpha,
        "trust_region_kl_budget": config.trust_region_kl_budget,
        "trust_region_achieved_kl": trust_region_kl,
        "trust_region_raw_kl": trust_region_raw_kl,
        "projection_scope": config.projection_scope,
        "projection_path": config.projection_path,
        "fixed_projection_alpha": (
            config.fixed_projection_alpha
            if config.projection_scope == "fixed"
            else None
        ),
        "projection_cap_hits": projection_cap_hits,
        "projection_alpha_min": (
            float(projection_token_alphas.min().item())
            if projection_token_alphas is not None
            else trust_region_alpha
        ),
        "projection_alpha_max": (
            float(projection_token_alphas.max().item())
            if projection_token_alphas is not None
            else trust_region_alpha
        ),
        "phase_seconds": phase_seconds,
    }

    # Query-local teacher state is explicitly destroyed after its only update.
    del streamed
    del student_hidden_all, student_hidden
    if teacher_hidden_all is not None:
        del teacher_hidden_all, teacher_hidden
    gc.collect()
    row["temporary_teacher_destroyed_after_update"] = True
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated())
        cuda_peak_reserved = int(torch.cuda.max_memory_reserved())
    else:
        cuda_peak_allocated = 0
        cuda_peak_reserved = 0
    row["episode_seconds"] = time.perf_counter() - started
    row["resource_usage"] = {
        "cuda_memory_baseline_bytes": cuda_memory_baseline,
        "cuda_peak_memory_allocated_bytes": cuda_peak_allocated,
        "cuda_peak_memory_delta_bytes": max(
            cuda_peak_allocated - cuda_memory_baseline, 0
        ),
        "cuda_peak_memory_reserved_bytes": cuda_peak_reserved,
        "process_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }
    return row


def run_persistent_training(
    *,
    model,
    tokenizer,
    queries: Sequence[Mapping[str, str]],
    config: PersistentConfig,
    output_dir: str | Path,
    input_hashes: Mapping[str, str],
    resume: bool = False,
    runtime_metadata: Optional[Mapping[str, Any]] = None,
    signal_controller: Optional[SignalController] = None,
) -> dict[str, Any]:
    """Train persistent LGSD, OPSD, or the matched Veto baseline."""
    config.validate()
    if len(queries) != config.episodes:
        raise PersistentProtocolError(
            f"Expected {config.episodes} bound queries, received {len(queries)}"
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
        "distillation_kl_direction": config.distillation_kl_direction,
        "projection_kl_direction": (
            PROJECTION_KL_DIRECTION
            if config.branch == "clean"
            else "not_applicable_unprojected_target"
        ),
        "target_reformulation": (
            VETO_TARGET_VERSION if config.branch == "veto" else None
        ),
        "veto_beta_schedule": (
            {
                "name": config.veto_beta_schedule,
                "start": config.veto_beta_start,
                "end": config.veto_beta_end,
                "step_denominator": config.episodes,
            }
            if config.branch == "veto"
            else None
        ),
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
            row = train_one_episode(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                query=query,
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
