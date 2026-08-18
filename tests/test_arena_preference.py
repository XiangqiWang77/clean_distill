from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from src.clean_self_distill.arena_preference import (
    PAIR_SCHEMA_VERSION,
    SCORE_SCHEMA_VERSION,
    ArenaPreferenceError,
    align_score_rows,
    sha256_text,
    summarize_score_rows,
    validate_preference_pair,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = _load_script(
    "arena_preference_logprob",
    "scripts/clean_self_distill/60_arena_preference_logprob.py",
)
REPORT = _load_script(
    "arena_preference_report",
    "scripts/clean_self_distill/61_arena_preference_report.py",
)


def _pair(index: int, domain: str = "math") -> dict[str, object]:
    prompt = f"Prompt {index}"
    preferred = f"Preferred response {index}"
    rejected = f"Rejected response {index}"
    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "evaluation_only": True,
        "label_source": "lmarena_human_vote",
        "external_judge_used": False,
        "query_id": f"q{index}",
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "normalized_prompt_sha256": sha256_text(prompt.casefold()),
        "preferred_response": preferred,
        "preferred_response_sha256": sha256_text(preferred),
        "rejected_response": rejected,
        "rejected_response_sha256": sha256_text(rejected),
        "domains": [domain],
        "source": "fixture",
    }


def _score_row(
    pair: dict[str, object],
    index: int,
    *,
    method: str,
    checkpoint: int,
    margin: float,
) -> dict[str, object]:
    rejected = -2.0 - 0.01 * index
    preferred = rejected + margin
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "external_judge_used": False,
        "bradley_terry_used": False,
        "evaluation_identity": {"fixture": True},
        "method": method,
        "checkpoint": checkpoint,
        "query_id": pair["query_id"],
        "global_query_index": index,
        "prompt_sha256": pair["prompt_sha256"],
        "preferred_response_sha256": pair["preferred_response_sha256"],
        "rejected_response_sha256": pair["rejected_response_sha256"],
        "domains": pair["domains"],
        "prompt_token_count": 4,
        "preferred_token_count": 5,
        "rejected_token_count": 5,
        "preferred_logprob_sum": preferred * 5,
        "rejected_logprob_sum": rejected * 5,
        "preferred_mean_logprob": preferred,
        "rejected_mean_logprob": rejected,
        "preference_margin": margin,
        "preference_correct": margin > 0,
        "prompt_truncation": {"applied": False},
        "preferred_truncation": {"applied": False},
        "rejected_truncation": {"applied": False},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_pair_validation_and_score_summary() -> None:
    pair = _pair(0)
    assert validate_preference_pair(pair)["query_id"] == "q0"
    with pytest.raises(ArenaPreferenceError, match="external judge"):
        validate_preference_pair({**pair, "external_judge_used": True})

    rows = [
        _score_row(_pair(index), index, method="Base", checkpoint=0, margin=margin)
        for index, margin in enumerate((0.2, -0.1, 0.4))
    ]
    summary = summarize_score_rows(rows)
    assert summary["preference_margin"] == pytest.approx(1 / 6)
    assert summary["preference_accuracy"] == pytest.approx(2 / 3)
    assert summary["external_judge_used"] is False


def test_alignment_rejects_changed_pair_identity() -> None:
    pairs = [_pair(0), _pair(1)]
    base = [
        _score_row(pair, index, method="Base", checkpoint=0, margin=0.1)
        for index, pair in enumerate(pairs)
    ]
    candidate = [
        _score_row(pair, index, method="OPSD", checkpoint=1, margin=0.2)
        for index, pair in enumerate(pairs)
    ]
    assert len(align_score_rows(base, candidate)) == 2
    candidate[1]["prompt_sha256"] = "0" * 64
    with pytest.raises(ArenaPreferenceError, match="identity differs"):
        align_score_rows(base, candidate)


class TinyTokenizer:
    chat_template = "fixture"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return f"U:{messages[0]['content']}|A:"

    def __call__(self, text, *, add_special_tokens):
        prefix = [99] if add_special_tokens else []
        return {"input_ids": prefix + [ord(char) for char in text]}

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token) for token in token_ids if token != 99)


def test_bounded_prompt_and_response_are_explicit() -> None:
    tokenizer = TinyTokenizer()
    prompt_ids, prompt_audit = SCORER.bounded_prompt(
        tokenizer, "abcdefghij", max_prompt_tokens=10
    )
    assert len(prompt_ids) <= 10
    assert prompt_audit["applied"] is True
    response_ids, response_audit = SCORER.bounded_response(
        tokenizer,
        "0123456789",
        prompt_token_count=len(prompt_ids),
        max_response_tokens=5,
        context_window=14,
    )
    assert len(response_ids) == min(5, 14 - len(prompt_ids))
    assert response_audit["applied"] is True


def test_report_uses_yellow_black_palette_and_preserves_small_digits() -> None:
    assert set(REPORT.METHOD_COLORS.values()) == {
        REPORT.BLACK,
        REPORT.PALE_YELLOW,
        REPORT.YELLOW,
        REPORT.DARK_YELLOW,
    }
    assert REPORT._fmt(0.12349) == "0.123"
    assert REPORT._fmt(0.00012349) == "1.23e-04"
    assert REPORT._fmt(0.0) == "0.000"


def test_complete_report_renders_without_judge_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The production report writes 200-DPI PNG and vector PDF.  Keep this
    # synthetic unit test small while exercising all six plotting paths.
    monkeypatch.setattr(REPORT, "FIGURE_DPI", 72)
    monkeypatch.setattr(REPORT, "FIGURE_FORMATS", ("png",))
    pairs = [_pair(index, "math" if index < 3 else "code") for index in range(6)]
    pair_path = tmp_path / "pairs.jsonl"
    _write_jsonl(pair_path, pairs)

    score_root = tmp_path / "scores"
    base_margins = [0.10, 0.04, -0.02, 0.08, -0.03, 0.02]
    base_rows = [
        _score_row(pair, index, method="Base", checkpoint=0, margin=base_margins[index])
        for index, pair in enumerate(pairs)
    ]
    _write_jsonl(score_root / "base" / "episode_0000.jsonl", base_rows)

    tags = {
        "LGSD-Small": "lgsd_small",
        "LGSD-Medium": "lgsd_medium",
        "LGSD-Large": "lgsd_large",
        "OPSD": "opsd",
    }
    gain_scale = {
        "LGSD-Small": 0.02,
        "LGSD-Medium": 0.07,
        "LGSD-Large": 0.09,
        "OPSD": 0.10,
    }
    for checkpoint in (1, 2):
        for method, tag in tags.items():
            rows = [
                _score_row(
                    pair,
                    index,
                    method=method,
                    checkpoint=checkpoint,
                    margin=base_margins[index]
                    + gain_scale[method] * checkpoint
                    + 0.002 * index,
                )
                for index, pair in enumerate(pairs)
            ]
            _write_jsonl(score_root / tag / f"episode_{checkpoint:04d}.jsonl", rows)

    run_root = tmp_path / "run"
    target_scale = {
        "lgsd_small": 0.01,
        "lgsd_medium": 0.03,
        "lgsd_large": 0.06,
        "opsd": 0.10,
    }
    radius = {
        "lgsd_small": 0.001,
        "lgsd_medium": 0.004,
        "lgsd_large": 0.016,
        "opsd": None,
    }
    for tag in tags.values():
        journal = run_root / "train" / tag / "episodes.jsonl"
        rows = []
        for episode in (1, 2):
            rows.append(
                {
                    "episode": episode,
                    "response_tokens": 10 + episode,
                    "mean_teacher_student_kl": target_scale[tag] * episode,
                    "trust_region_achieved_kl": (
                        target_scale[tag] * episode if tag != "opsd" else None
                    ),
                    "trust_region_raw_kl": (
                        0.1 * episode if tag != "opsd" else None
                    ),
                    "trust_region_alpha": (
                        target_scale[tag] * 5 if tag != "opsd" else None
                    ),
                    "trust_region_kl_budget": radius[tag],
                    "relative_update_norm": 0.001 * episode,
                    "rollout_cap_hit": episode == 2,
                }
            )
        _write_jsonl(journal, rows)

    output = tmp_path / "report"
    args = argparse.Namespace(
        run_root=run_root,
        pairs=pair_path,
        score_root=score_root,
        output_dir=output,
        style_summary=None,
        movement_root=None,
        bootstrap_resamples=50,
        rolling_window=1,
        min_domain_count=1,
        max_domains=5,
    )
    result = REPORT.build(args)
    assert result["external_llm_judge_used"] is False
    assert result["bradley_terry_used"] is False
    assert result["checkpoints"] == [1, 2]
    serialized_result = json.dumps(result, sort_keys=True)
    assert str(tmp_path) not in serialized_result
    assert result["inputs"]["path_basis"] == "run_root_relative_or_basename"
    assert (output / "README.md").is_file()
    assert (output / "arena_preference_main_table.tex").is_file()
    assert b"\r\n" not in (output / "arena_preference_main_table.csv").read_bytes()
    for index in range(1, 7):
        assert (output / f"fig{index}_{['preference_dynamics', 'locality_tradeoff', 'logprob_decomposition', 'domain_heatmap', 'training_diagnostics', 'metric_profile'][index - 1]}.png").is_file()
