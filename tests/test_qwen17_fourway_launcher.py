from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "scripts/clean_self_distill/slurm/qwen17_fourway_distill_eval.slurm"
REPORTER = ROOT / "scripts/clean_self_distill/14_qwen17_fourway_report.py"
SPEC = importlib.util.spec_from_file_location("qwen17_report", REPORTER)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def make_rows(method: str, episode: int, correct_count: int) -> list[dict[str, object]]:
    sources = ["amc23"] * 83 + ["aime24"] * 30 + ["aime25"] * 30
    return [
        {
            "method": method,
            "checkpoint_episode": episode,
            "query_id": f"q{index:03d}",
            "problem_sha256": f"{index:064x}",
            "source": source,
            "sample_index": 0,
            "correct": index < correct_count,
            "truncated": False,
            "max_new_tokens": 10240,
            "generated_tokens": 100 + index,
            "resource_usage": {"generation_seconds": 1.0},
        }
        for index, source in enumerate(sources)
    ]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_slurm_uses_exactly_four_h100_tasks_and_embeds_base() -> None:
    source = SLURM.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-3%4" in source
    assert "#SBATCH --gres=gpu:h100:1" in source
    assert "#SBATCH --time=11:50:00" in source
    assert "#SBATCH --requeue" in source
    assert source.count("TAG=") == 4
    assert 'if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]' in source
    assert "run_eval base base 0" in source


def test_report_waits_then_builds_five_way_table(tmp_path: Path) -> None:
    assert report.build_if_complete(tmp_path) is False
    settings = {
        "base": ("base", 0, 70),
        "privileged_16": ("privileged_sd", 16, 72),
        "trsd_16": ("trsd", 16, 70),
        "privileged_64": ("privileged_sd", 64, 80),
        "trsd_64": ("trsd", 64, 90),
    }
    for name, (method, episode, correct) in settings.items():
        write_rows(tmp_path / "eval" / name / "scored.jsonl", make_rows(method, episode, correct))
    assert report.build_if_complete(tmp_path) is True
    summary = json.loads((tmp_path / "results/summary.json").read_text(encoding="utf-8"))
    assert summary["comparisons"]["short_term_trsd16_vs_base_pp"] == 0.0
    assert round(summary["comparisons"]["long_term_trsd64_vs_privileged64_pp"], 8) == round(1000 / 143, 8)
    assert (tmp_path / "RUN_COMPLETE").read_text(encoding="utf-8") == "complete\n"
