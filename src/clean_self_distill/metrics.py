"""Metrics for clean-teacher strength, speed, and hindsight auditing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


FORBIDDEN_TEACHER_SOURCES = {
    "target_answer",
    "target_solution",
    "verifier_feedback_on_target",
    "future_target_tokens",
}


@dataclass
class HindsightAudit:
    """Provenance audit for teacher/student comparisons.

    The audit is structural rather than lexical: it records which data fields
    were allowed to construct a teacher and whether both distributions were
    evaluated on byte-identical causal prefixes.
    """

    teacher_events: int = 0
    exposed_events: int = 0
    comparison_events: int = 0
    context_equal_events: int = 0
    compared_token_positions: int = 0
    same_prefix_positions: int = 0
    causal_events: int = 0
    on_policy_events: int = 0
    on_policy_equal_events: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)

    def record_teacher_context(self, sources: Iterable[str], *, causal: bool = True) -> None:
        normalized = {str(source).strip().lower() for source in sources}
        self.teacher_events += 1
        if normalized & FORBIDDEN_TEACHER_SOURCES:
            self.exposed_events += 1
        for source in normalized:
            self.source_counts[source] = self.source_counts.get(source, 0) + 1
        if causal:
            self.causal_events += 1

    def record_same_prefix(
        self,
        student_ids: Any,
        teacher_ids: Any,
        positions: int,
        *,
        on_policy: bool = False,
    ) -> None:
        self.comparison_events += 1
        equal = _sequences_equal(student_ids, teacher_ids)
        if equal:
            self.context_equal_events += 1
            self.same_prefix_positions += int(positions)
        if on_policy:
            self.on_policy_events += 1
            if equal:
                self.on_policy_equal_events += 1
        self.compared_token_positions += int(positions)

    def merge(self, other: "HindsightAudit") -> None:
        for name in (
            "teacher_events",
            "exposed_events",
            "comparison_events",
            "context_equal_events",
            "compared_token_positions",
            "same_prefix_positions",
            "causal_events",
            "on_policy_events",
            "on_policy_equal_events",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for source, count in other.source_counts.items():
            self.source_counts[source] = self.source_counts.get(source, 0) + count

    def compute(self) -> dict[str, float]:
        return {
            "hindsight/teacher_context_events": float(self.teacher_events),
            "hindsight/forbidden_context_events": float(self.exposed_events),
            "hindsight/hindsight_exposure_rate": _ratio(self.exposed_events, self.teacher_events),
            "hindsight/context_parity_rate": _ratio(self.context_equal_events, self.comparison_events),
            "hindsight/same_prefix_fidelity": _ratio(
                self.same_prefix_positions, self.compared_token_positions
            ),
            "hindsight/causal_scoring_rate": _ratio(self.causal_events, self.teacher_events),
            "hindsight/on_policy_same_prefix_rate": _ratio(
                self.on_policy_equal_events, self.on_policy_events
            ),
        }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _sequences_equal(left: Any, right: Any) -> bool:
    try:
        import torch

        if torch.is_tensor(left) and torch.is_tensor(right):
            return bool(left.shape == right.shape and torch.equal(left, right))
    except ImportError:
        pass
    try:
        return list(left) == list(right)
    except TypeError:
        return left == right


def aggregate_teacher_metrics(rows: list[dict[str, Any]], audit: HindsightAudit) -> dict[str, float]:
    """Aggregate query-level results and define the paper-facing metrics.

    HFTG is deliberately zeroed by exposure or context mismatch:

        HFTG = (1 - HER) * CPP * (NLL_student - NLL_clean_teacher)

    FATE divides HFTG by query-local specialization wall time.
    """
    audit_metrics = audit.compute()
    her = audit_metrics["hindsight/hindsight_exposure_rate"]
    cpp = audit_metrics["hindsight/context_parity_rate"]
    nll_gains = [float(row["target_answer_nll_gain"]) for row in rows if "target_answer_nll_gain" in row]
    times = [float(row["specialization_seconds"]) for row in rows if "specialization_seconds" in row]
    feature_times = [
        float(row["feature_extraction_seconds"])
        for row in rows
        if "feature_extraction_seconds" in row
    ]
    solve_times = [
        float(row["closed_form_solve_seconds"])
        for row in rows
        if "closed_form_solve_seconds" in row
    ]
    base_correct = [float(row["base_correct"]) for row in rows if "base_correct" in row]
    teacher_correct = [float(row["teacher_correct"]) for row in rows if "teacher_correct" in row]
    privileged_correct = [
        float(row["privileged_correct"]) for row in rows if "privileged_correct" in row
    ]
    clean_advantage_retention = [
        float(row["clean_advantage_retention"])
        for row in rows
        if "clean_advantage_retention" in row
    ]
    privileged_jsd = [
        float(row["privileged_counterfactual_jsd"])
        for row in rows
        if "privileged_counterfactual_jsd" in row
    ]
    privileged_flip = [
        float(row["privileged_answer_flip_rate"])
        for row in rows
        if "privileged_answer_flip_rate" in row
    ]
    base_pass = [float(row["base_pass_at_n"]) for row in rows if "base_pass_at_n" in row]
    teacher_pass = [float(row["teacher_pass_at_n"]) for row in rows if "teacher_pass_at_n" in row]

    mean_nll_gain = sum(nll_gains) / max(len(nll_gains), 1)
    mean_time = sum(times) / max(len(times), 1)
    mean_feature_time = sum(feature_times) / max(len(feature_times), 1)
    mean_solve_time = sum(solve_times) / max(len(solve_times), 1)
    hftg = (1.0 - her) * cpp * mean_nll_gain
    result = {
        **audit_metrics,
        "teacher/target_answer_nll_gain": mean_nll_gain,
        "teacher/specialization_success_rate": _ratio(
            sum(gain > 0 for gain in nll_gains), len(nll_gains)
        ),
        "speed/mean_specialization_seconds": mean_time,
        "speed/mean_feature_extraction_seconds": mean_feature_time,
        "speed/mean_closed_form_solve_seconds": mean_solve_time,
        "speed/closed_form_solve_time_fraction": mean_solve_time / max(mean_time, 1e-12),
        "hindsight/hindsight_free_transfer_gain": hftg,
        "speed/fast_adaptation_teacher_efficiency": hftg / max(mean_time, 1e-12),
    }
    if base_correct and teacher_correct:
        base_acc = sum(base_correct) / len(base_correct)
        teacher_acc = sum(teacher_correct) / len(teacher_correct)
        result.update(
            {
                "accuracy/base": base_acc,
                "accuracy/temporary_teacher": teacher_acc,
                "teacher/clean_accuracy_gain": teacher_acc - base_acc,
                "hindsight/hindsight_free_accuracy_gain": (1.0 - her)
                * cpp
                * (teacher_acc - base_acc),
                "speed/fast_adaptation_accuracy_efficiency": (1.0 - her)
                * cpp
                * (teacher_acc - base_acc)
                / max(mean_time, 1e-12),
            }
        )
        if privileged_correct:
            privileged_acc = sum(privileged_correct) / len(privileged_correct)
            privileged_gain = privileged_acc - base_acc
            clean_gain = teacher_acc - base_acc
            result.update(
                {
                    "hindsight/privileged_control_accuracy": privileged_acc,
                    "hindsight/hindsight_privilege_gap": privileged_acc - teacher_acc,
                    "hindsight/privileged_accuracy_gain": privileged_gain,
                    "hindsight/clean_gain_fraction_vs_privileged": (
                        clean_gain / privileged_gain if abs(privileged_gain) > 1e-12 else 0.0
                    ),
                    "hindsight/clean_advantage_retention": sum(clean_advantage_retention)
                    / max(len(clean_advantage_retention), 1),
                    "hindsight/clean_counterfactual_jsd": 0.0,
                    "hindsight/privileged_counterfactual_jsd": sum(privileged_jsd)
                    / max(len(privileged_jsd), 1),
                    "hindsight/clean_answer_flip_rate": 0.0,
                    "hindsight/privileged_answer_flip_rate": sum(privileged_flip)
                    / max(len(privileged_flip), 1),
                }
            )
    if base_pass and teacher_pass:
        result["accuracy/base_pass_at_n"] = sum(base_pass) / len(base_pass)
        result["accuracy/temporary_teacher_pass_at_n"] = sum(teacher_pass) / len(teacher_pass)
    return result
