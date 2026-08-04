import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.clean_self_distill.report_horizon_metrics import (
    HorizonReportError,
    build_horizon_report,
    generate_horizon_report,
)


def _windows(prefix_tokens: int) -> list[dict]:
    result = []
    for start, end in ((0, 512), (512, 1024), (1024, 2048), (2048, 4096)):
        count = max(min(prefix_tokens, end) - start, 0)
        result.append(
            {
                "start_token": start,
                "end_token": end,
                "token_count": count,
                "measurement_point": "pre_update",
                "pre_update_mean_teacher_student_kl": 0.25 if count else None,
                "pre_update_teacher_student_top1_agreement": 0.75 if count else None,
                "pre_update_mean_teacher_base_ridge_shift_l2": 1.5 if count else None,
            }
        )
    return result


def _condition(correct: int, tokens: int, *, method: str) -> dict:
    if method == "Base":
        audit = None
    elif method == "Privileged Control":
        audit = {
            "teacher_context_events": 1,
            "forbidden_context_events": 1,
            "comparison_events": 0,
            "compared_token_positions": 0,
            "same_prefix_positions": 0,
        }
    else:
        audit = {
            "teacher_context_events": 4,
            "forbidden_context_events": 0,
            "comparison_events": 3,
            "compared_token_positions": 12288,
            "same_prefix_positions": 12288,
        }
    return {
        "correct": correct,
        "generated_tokens": tokens,
        "truncated": False,
        "audit_counts": audit,
    }


def _row(query_id: str, source: str, *, base_correct: int, csd_correct: int, base_tokens: int) -> dict:
    trace = []
    for step in range(3):
        trace.append(
            {
                "step": step,
                "prefix_tokens": 4096,
                "prefix_truncated": True,
                "trajectory_complete": False,
                "long_horizon_qualified": True,
                "horizon_windows": _windows(4096),
            }
        )
    return {
        "query_id": query_id,
        "source": source,
        "specialization_no_op": False,
        "conditions": {
            "Base": _condition(base_correct, base_tokens, method="Base"),
            "Privileged Control": _condition(1, max(base_tokens - 10, 1), method="Privileged Control"),
            "CSD-T": _condition(csd_correct, base_tokens + 20, method="CSD-T"),
            "CSD-SD": _condition(csd_correct, base_tokens + 30, method="CSD-SD"),
        },
        "task1_artifact": {"target_answer_nll_gain": 0.2},
        "task2_artifact": {
            "distilled_target_answer_nll_gain": 0.1,
            "distillation_config": {
                "steps": 3,
                "prefix_max_new_tokens": 4096,
                "long_horizon_min_tokens": 4096,
            },
            "distillation_steps_completed": 3,
            "distillation_trace": trace,
            "long_horizon_qualified": True,
        },
    }


class HorizonReportTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            _row("amc-1", "amc23", base_correct=1, csd_correct=0, base_tokens=1000),
            _row("aime24-1", "aime24", base_correct=0, csd_correct=1, base_tokens=3000),
        ]
        # Ten held-out late queries make the claim gate deterministic.
        self.rows.extend(
            _row(f"aime25-{index}", "aime25", base_correct=0, csd_correct=1, base_tokens=3000)
            for index in range(9)
        )
        self.summary = {
            "validation": {"status": "passed", "unique_query_count": len(self.rows)},
            "max_tokens": 8192,
        }

    def test_htrain_is_max_contiguous_depth_and_metrics_are_paired(self):
        report = build_horizon_report(
            self.rows,
            self.summary,
            split_tokens=2048,
            long_prefix_threshold=4096,
            late_window_start=2048,
            min_heldout_late_queries=10,
        )
        horizon = report["distillation_horizon"]
        self.assertEqual(horizon["H_train_tokens"], 4096)
        self.assertEqual(
            horizon["sampled_positions_across_independent_rollouts"],
            len(self.rows) * 3 * 4096,
        )
        self.assertTrue(horizon["claim_gate"]["long_horizon_evidence_pass"])
        self.assertEqual(horizon["heldout_late_coverage_query_count"], 10)
        self.assertAlmostEqual(
            horizon["window_metrics"][3]["pre_update_mean_teacher_student_kl"],
            0.25,
        )

        long_aime_sd = next(
            row
            for row in report["performance"]
            if row["dataset_scope"] == "aime"
            and row["target_horizon"] == "long"
            and row["method"] == "CSD-SD"
        )
        self.assertEqual(long_aime_sd["wrong_to_correct"], 10)
        self.assertEqual(long_aime_sd["correct_to_wrong"], 0)
        self.assertGreater(long_aime_sd["gain_vs_base_pp"], 0.0)
        self.assertEqual(long_aime_sd["HER"], 0.0)
        self.assertEqual(long_aime_sd["CPP"], 1.0)
        self.assertEqual(long_aime_sd["HFS"], 1.0)
        self.assertEqual(
            report["temporal_retention"]["aime"]["post_teacher_csd_sd_accuracy"],
            1.0,
        )
        self.assertEqual(
            report["temporal_retention"]["aime"]["teacher_gain_retention"],
            1.0,
        )

    def test_natural_completion_qualifies_but_does_not_fake_late_coverage(self):
        row = copy.deepcopy(self.rows[0])
        row["query_id"] = "amc-natural"
        row["task2_artifact"]["distillation_trace"] = [
            {
                "step": 0,
                "prefix_tokens": 1000,
                "prefix_truncated": False,
                "trajectory_complete": True,
                "long_horizon_qualified": True,
                "horizon_windows": _windows(1000),
            }
        ]
        row["task2_artifact"]["distillation_config"]["steps"] = 1
        row["task2_artifact"]["distillation_steps_completed"] = 1
        summary = {"validation": {"status": "passed", "unique_query_count": 1}, "max_tokens": 8192}
        report = build_horizon_report(
            [row],
            summary,
            split_tokens=2048,
            long_prefix_threshold=4096,
            late_window_start=2048,
            min_heldout_late_queries=1,
        )
        horizon = report["distillation_horizon"]
        self.assertEqual(horizon["qualified_query_count"], 1)
        self.assertEqual(horizon["late_coverage_query_count"], 0)
        self.assertFalse(horizon["claim_gate"]["long_horizon_evidence_pass"])

    def test_old_512_configuration_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["task2_artifact"]["distillation_config"]["prefix_max_new_tokens"] = 512
        with self.assertRaisesRegex(HorizonReportError, "4096-token opportunity"):
            build_horizon_report(
                rows,
                self.summary,
                split_tokens=2048,
                long_prefix_threshold=4096,
                late_window_start=2048,
                min_heldout_late_queries=10,
            )

    def test_generate_writes_all_three_posthoc_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
            (root / "merged_per_query.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in self.rows),
                encoding="utf-8",
            )
            generate_horizon_report(root)
            for name in ("horizon_results.json", "horizon_results.csv", "horizon_results.md"):
                self.assertTrue((root / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
