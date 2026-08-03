"""Synthetic-log contract test for every paper table and Figure 2--9."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.clean_self_distill.analyze_paper_suite import build_tables, load_suite, plot_figures


def _write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


class PaperSuiteAnalysisTest(unittest.TestCase):
    def test_all_tables_and_figures_from_real_log_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposal = {
                "query_id": "q1",
                "specialization_candidates": [
                    {
                        "candidate_id": "c0",
                        "verifier_valid": True,
                        "target_disjoint_audit": {
                            "literal_overlap_rate": 0.01,
                            "fourgram_overlap_rate": 0.0,
                            "literal_overlap_count": 1,
                            "fourgram_overlap_count": 0,
                        },
                    }
                ],
            }
            _write(root / "supports/headline/qwen3_4b/seed_0/proposals.jsonl", proposal)

            task1 = {
                "query_id": "q1",
                "source": "aime24",
                "base_correct": 0.0,
                "teacher_correct": 1.0,
                "base_pass_at_n": 0.0,
                "teacher_pass_at_n": 1.0,
                "base_majority_at_n": 0.0,
                "teacher_majority_at_n": 1.0,
                "target_answer_nll_gain": 0.3,
                "proposal_fit_target_logit_gain": 0.2,
                "specialization_seconds": 0.4,
                "feature_extraction_seconds": 0.35,
                "closed_form_solve_seconds": 0.05,
                "peak_memory_bytes": 100.0,
                "base_generated_tokens": 100,
                "teacher_generated_tokens": 100,
                "support_generated_tokens": 200,
                "support_generation_seconds": 1.0,
                "privileged_correct": 1.0,
                "clean_advantage_retention": 1.0,
                "privileged_counterfactual_jsd": 0.2,
                "privileged_answer_flip_rate": 0.5,
            }
            task2 = {
                **task1,
                "distilled_correct": 1.0,
                "distilled_pass_at_n": 1.0,
                "distilled_majority_at_n": 1.0,
                "teacher_advantage_transfer": 0.8,
            }
            baseline = {
                "query_id": "q1",
                "source": "aime24",
                "base_correct": 0.0,
                "correct": 1.0,
                "target_answer_nll_gain": 0.2,
                "adaptation_seconds": 2.0,
                "peak_memory_bytes": 500.0,
            }
            _write(root / "main/qwen3_4b/seed_0/csd_t/eval_task1_fast_teacher.jsonl", task1)
            _write(root / "main/qwen3_4b/seed_0/csd_sd/eval_task2_clean_distillation.jsonl", task2)
            _write(root / "main/qwen3_4b/seed_0/support_icl/eval_support_icl.jsonl", {"query_id": "q1", "source": "aime24", "correct": 1.0})
            _write(root / "main/qwen3_4b/seed_0/head_sgd/eval_head_sgd.jsonl", baseline)
            _write(root / "main/qwen3_4b/seed_0/support_lora/eval_support_lora.jsonl", baseline)
            _write(root / "main/qwen3_4b/seed_0/self_consistency/eval_task1_fast_teacher.jsonl", task1)
            _write(root / "budget/qwen3_4b/seed_0/samples_1/eval_task1_fast_teacher.jsonl", task1)
            _write(root / "budget/qwen3_4b/seed_0/samples_8/eval_task1_fast_teacher.jsonl", task1)
            _write(root / "hindsight/qwen3_4b/seed_0/eval_task1_fast_teacher.jsonl", task1)
            _write(root / "transfer/qwen3_4b/seed_0/steps_0/eval_task2_clean_distillation.jsonl", task2)
            _write(root / "transfer/qwen3_4b/seed_0/steps_3/eval_task2_clean_distillation.jsonl", task2)
            _write(root / "sensitivity/qwen3_4b/seed_0/ridge_lambda/0p1/eval_task1_fast_teacher.jsonl", task1)
            _write(root / "sensitivity/qwen3_4b/seed_0/support_tokens/256p0/eval_task1_fast_teacher.jsonl", task1)
            _write(root / "sensitivity/qwen3_4b/seed_0/support_count/10p0/eval_task1_fast_teacher.jsonl", task1)

            suite = load_suite(root)
            analysis = root / "analysis"
            build_tables(root, analysis, suite, {"method": {"ridge_lambda": 0.1}})
            figures = plot_figures(root, analysis, suite)

            self.assertTrue((analysis / "tables/table1_main.csv").exists())
            self.assertTrue((analysis / "tables/table9_cost_breakdown.csv").exists())
            names = {Path(path).name for path in figures}
            self.assertTrue(
                {
                    "fig2_support_hygiene.pdf",
                    "fig3_teacher_gain.pdf",
                    "fig4_efficiency_frontier.pdf",
                    "fig5_accuracy_budget.pdf",
                    "fig6_hindsight_audit.pdf",
                    "fig7_transfer_reliability.pdf",
                    "fig8_distillation_transfer.pdf",
                    "fig9_sensitivity.pdf",
                }.issubset(names)
            )


if __name__ == "__main__":
    unittest.main()
