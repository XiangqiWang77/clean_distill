import hashlib
import json
import stat
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.clean_self_distill.prepare_empirical_data import (
    DataFirewallError,
    enforce_capacity_guards,
    prepare_empirical_data,
    select_deepmath_records,
)
from src.clean_self_distill.heldout import FORBIDDEN_QUERY_KEYS


def _deepmath_row(
    index: int,
    *,
    problem: str | None = None,
    answer: str | None = None,
    solution: str | None = None,
):
    return {
        "prompt": [{"role": "user", "content": f"wrapped {index}"}],
        "reward_model": {"ground_truth": answer or str(index)},
        "data_source": "deepmath",
        "extra_info": {
            "index": index,
            "problem": problem or f"Deep problem {index}",
            "solution": solution or f"Reference solution {index}",
            "difficulty": 7 + index % 4,
        },
    }


def _heldout_row(source: str, index: int, *, problem: str | None = None):
    return {
        "prompt": [{"role": "user", "content": f"wrapped heldout {index}"}],
        "reward_model": {"ground_truth": str(100 + index)},
        "data_source": source,
        "extra_info": {
            "index": index,
            "problem": problem or f"{source} heldout problem {index}",
            "solution": f"Heldout reference solution {index}",
        },
    }


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).casefold()
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_streamed_hash_split_is_order_independent_and_physically_label_free(tmp_path: Path):
    task_root = tmp_path / "scratch" / "da839" / "clean_distill"
    task_root.mkdir(parents=True)
    rows = [_deepmath_row(index) for index in range(8)]
    deep_a = task_root / "data" / "deep-a.parquet"
    deep_b = task_root / "data" / "deep-b.parquet"
    heldout = task_root / "data" / "heldout.parquet"
    _write_parquet(deep_a, rows)
    _write_parquet(deep_b, list(reversed(rows)))
    _write_parquet(
        heldout,
        [
            _heldout_row("amc23", 0),
            _heldout_row("aime24", 1),
            _heldout_row("amc23", 2),
            _heldout_row("aime25", 3),
        ],
    )
    common = {
        "heldout_path": heldout,
        "task_root": task_root,
        "distill_count": 3,
        "dev_count": 2,
        "heldout_counts": {"amc23": 2, "aime24": 1, "aime25": 1},
        "max_task_bytes": 1_000_000_000,
        "batch_size": 1,
    }
    manifest_a = prepare_empirical_data(
        deepmath_path=deep_a,
        output_dir=task_root / "prepared-a",
        new_download_bytes=deep_a.stat().st_size + heldout.stat().st_size,
        **common,
    )
    manifest_b = prepare_empirical_data(
        deepmath_path=deep_b,
        output_dir=task_root / "prepared-b",
        new_download_bytes=deep_b.stat().st_size + heldout.stat().st_size,
        **common,
    )

    expected = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            row["extra_info"]["problem"].encode("utf-8")
        ).hexdigest(),
    )[:5]
    expected_problems = [row["extra_info"]["problem"] for row in expected]
    distill = _read_jsonl(task_root / "prepared-a" / "distill_queries.jsonl")
    dev = _read_jsonl(task_root / "prepared-a" / "dev_queries.jsonl")
    assert [row["problem"] for row in distill + dev] == expected_problems
    assert manifest_a["counts"] == manifest_b["counts"] == {
        "distill": 3,
        "dev": 2,
        "heldout": 4,
        "heldout_by_source": {"amc23": 2, "aime24": 1, "aime25": 1},
    }
    assert manifest_a["overlap_audit"]["passed"] is True

    for name in ("distill_queries.jsonl", "dev_queries.jsonl", "heldout_queries.jsonl"):
        for row in _read_jsonl(task_root / "prepared-a" / name):
            assert not (set(_all_keys(row)) & FORBIDDEN_QUERY_KEYS)
    labels = _read_jsonl(task_root / "prepared-a" / "distill_labels.sealed.jsonl")
    assert labels and all("answer" in row and "reference_solution" in row for row in labels)
    mode = stat.S_IMODE((task_root / "prepared-a" / "distill_labels.sealed.jsonl").stat().st_mode)
    assert mode == 0o600


def test_conflicting_duplicate_deepmath_group_is_entirely_excluded(tmp_path: Path):
    parquet_a = tmp_path / "conflict-a.parquet"
    parquet_b = tmp_path / "conflict-b.parquet"
    rows = [
        _deepmath_row(0, problem="same", answer="1"),
        _deepmath_row(1, problem="same", answer="2"),
        _deepmath_row(2, problem="safe alpha", answer="3"),
        _deepmath_row(3, problem="safe beta", answer="4"),
    ]
    _write_parquet(parquet_a, rows)
    _write_parquet(parquet_b, list(reversed(rows)))

    selected_a, stats_a = select_deepmath_records(
        parquet_a, count=2, batch_size=1
    )
    selected_b, stats_b = select_deepmath_records(
        parquet_b, count=2, batch_size=1
    )

    assert [row["problem"] for row in selected_a] == [
        row["problem"] for row in selected_b
    ]
    assert {row["problem"] for row in selected_a} == {"safe alpha", "safe beta"}
    for stats in (stats_a, stats_b):
        assert stats["conflicting_duplicate_groups_excluded"] == 1
        assert stats["conflicting_duplicate_rows_excluded"] == 2
        assert stats["eligible_unique_rows"] == 2
        assert stats["selection_passes"] == 2
        assert stats["rows_rescanned_for_selection"] == len(rows)


def test_consistent_normalized_duplicates_choose_canonical_exact_text(tmp_path: Path):
    parquet = tmp_path / "normalized.parquet"
    rows = [
        _deepmath_row(
            0,
            problem="  Equivalent   Problem ",
            answer="7",
            solution="Same verified solution",
        ),
        _deepmath_row(
            1,
            problem="equivalent problem",
            answer="7",
            solution="Same verified solution",
        ),
        _deepmath_row(2, problem="independent problem", answer="8"),
    ]
    _write_parquet(parquet, rows)

    selected, stats = select_deepmath_records(parquet, count=2, batch_size=1)

    variants = [rows[0]["extra_info"]["problem"], rows[1]["extra_info"]["problem"]]
    canonical = min(
        variants,
        key=lambda problem: hashlib.sha256(problem.strip().encode("utf-8")).hexdigest(),
    )
    assert {row["problem"] for row in selected} == {
        canonical.strip(),
        "independent problem",
    }
    assert stats["conflicting_duplicate_groups_excluded"] == 0
    assert stats["consistent_duplicate_rows_deduplicated"] == 1
    assert stats["eligible_unique_rows"] == 2


def test_selection_fails_only_when_unambiguous_groups_are_insufficient(tmp_path: Path):
    parquet = tmp_path / "insufficient.parquet"
    _write_parquet(
        parquet,
        [
            _deepmath_row(0, problem="same", answer="1"),
            _deepmath_row(1, problem=" SAME ", answer="2"),
        ],
    )
    with pytest.raises(DataFirewallError, match="unambiguous normalized-problem groups"):
        select_deepmath_records(parquet, count=1, batch_size=1)


def test_overlap_and_heldout_count_mismatch_fail_closed(tmp_path: Path):
    task_root = tmp_path / "task"
    task_root.mkdir()
    deep = task_root / "deep.parquet"
    heldout = task_root / "heldout.parquet"
    deep_rows = [_deepmath_row(i) for i in range(3)]
    _write_parquet(deep, deep_rows)
    _write_parquet(
        heldout,
        [
            _heldout_row("amc23", 0, problem=deep_rows[0]["extra_info"]["problem"]),
            _heldout_row("aime24", 1),
            _heldout_row("aime25", 2),
        ],
    )
    with pytest.raises(DataFirewallError, match="overlap detected"):
        prepare_empirical_data(
            deepmath_path=deep,
            heldout_path=heldout,
            output_dir=task_root / "prepared",
            task_root=task_root,
            distill_count=2,
            dev_count=1,
            heldout_counts={"amc23": 1, "aime24": 1, "aime25": 1},
            new_download_bytes=deep.stat().st_size + heldout.stat().st_size,
            max_task_bytes=1_000_000_000,
            batch_size=1,
        )
    assert not (task_root / "prepared").exists()


def test_capacity_guards_reject_download_cap_and_paths_outside_scratch(tmp_path: Path):
    root = tmp_path / "task"
    root.mkdir()
    inside = root / "asset"
    inside.write_bytes(b"x")
    outside = tmp_path / "outside"
    outside.write_bytes(b"x")
    with pytest.raises(DataFirewallError, match="escapes"):
        enforce_capacity_guards(
            task_root=root,
            required_paths=(outside,),
            new_download_bytes=1,
            max_new_download_bytes=20,
            max_task_bytes=100,
        )
    with pytest.raises(DataFirewallError, match="downloads exceed"):
        enforce_capacity_guards(
            task_root=root,
            required_paths=(inside,),
            new_download_bytes=21,
            max_new_download_bytes=20,
            max_task_bytes=100,
        )
