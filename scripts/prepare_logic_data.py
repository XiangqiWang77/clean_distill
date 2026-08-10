#!/usr/bin/env python3
"""Materialize SATQuest and LogicSkills in the repository's verl parquet format.

The generated SATQuest train split mirrors DeepMath: it contains a user prompt,
an exact reward target, and a privileged solver-verified solution/certificate.
The SATQuest and LogicSkills validation splits mirror AMC/AIME: they keep the
same top-level contract but deliberately omit ``extra_info.solution``.

This script expects official source snapshots to have been downloaded below
``<scratch-root>/data/sources``.  It requires Python >= 3.10 because SATQuest
uses modern type syntax.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SCRATCH_ROOT = Path("/home/da839/scratch_pi_mg269/da839/clean_distill")
SATQUEST_PROBLEM_TYPES = (
    "SATDP_SAT",
    "SATSP",
    "SATDP_UNSAT",
    "MaxSAT",
    "MCS",
    "MUS",
)
SATQUEST_TRAIN_FORMATS = ("math", "story")
SATQUEST_EVAL_FORMATS = ("math", "story", "dimacs", "dualstory")
SATQUEST_SAT_TYPES = frozenset({"SATDP_SAT", "SATSP"})


PROMPT_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("content", pa.string(), nullable=False),
            pa.field("role", pa.string(), nullable=False),
        ]
    )
)
REWARD_TYPE = pa.struct(
    [
        pa.field("ground_truth", pa.string(), nullable=False),
        pa.field("style", pa.string(), nullable=False),
    ]
)


def _extra_type(fields: Iterable[str]) -> pa.StructType:
    return pa.struct([pa.field(name, pa.string(), nullable=False) for name in fields])


SATQUEST_TRAIN_SCHEMA = pa.schema(
    [
        pa.field("prompt", PROMPT_TYPE, nullable=False),
        pa.field("reward_model", REWARD_TYPE, nullable=False),
        pa.field("data_source", pa.string(), nullable=False),
        pa.field("ability", pa.string(), nullable=False),
        pa.field(
            "extra_info",
            _extra_type(
                (
                    "index",
                    "problem",
                    "solution",
                    "source_id",
                    "problem_type",
                    "question_type",
                    "num_variable",
                    "num_clause",
                    "cnf_dimacs",
                    "verifier_payload",
                )
            ),
            nullable=False,
        ),
    ],
    metadata={b"format": b"verl", b"role": b"self_distill_train"},
)

SATQUEST_EVAL_SCHEMA = pa.schema(
    [
        pa.field("prompt", PROMPT_TYPE, nullable=False),
        pa.field("reward_model", REWARD_TYPE, nullable=False),
        pa.field("data_source", pa.string(), nullable=False),
        pa.field("ability", pa.string(), nullable=False),
        pa.field(
            "extra_info",
            _extra_type(
                (
                    "index",
                    "problem",
                    "source_id",
                    "problem_type",
                    "question_type",
                    "num_variable",
                    "num_clause",
                    "eval_regime",
                    "cnf_dimacs",
                    "verifier_payload",
                )
            ),
            nullable=False,
        ),
    ],
    metadata={b"format": b"verl", b"role": b"self_eval"},
)

LOGICSKILLS_EVAL_SCHEMA = pa.schema(
    [
        pa.field("prompt", PROMPT_TYPE, nullable=False),
        pa.field("reward_model", REWARD_TYPE, nullable=False),
        pa.field("data_source", pa.string(), nullable=False),
        pa.field("ability", pa.string(), nullable=False),
        pa.field(
            "extra_info",
            _extra_type(
                (
                    "index",
                    "problem",
                    "task",
                    "language",
                    "eval_regime",
                    "verifier_payload",
                )
            ),
            nullable=False,
        ),
    ],
    metadata={b"format": b"verl", b"role": b"external_ood_self_eval"},
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def iter_parquet_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def write_parquet(
    path: Path,
    schema: pa.Schema,
    rows: Iterable[dict[str, Any]],
    *,
    batch_size: int,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    buffer: list[dict[str, Any]] = []
    try:
        with pq.ParquetWriter(
            temporary,
            schema,
            compression="zstd",
            use_dictionary=True,
        ) as writer:
            for row in rows:
                buffer.append(row)
                if len(buffer) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    count += len(buffer)
                    buffer.clear()
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                count += len(buffer)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def satquest_eval_regime(num_variable: int, question_type: str) -> str:
    size_ood = num_variable > 4
    format_ood = question_type in {"dimacs", "dualstory"}
    if size_ood and format_ood:
        return "size_format_ood"
    if size_ood:
        return "size_ood"
    if format_ood:
        return "format_ood"
    return "id"


def _satquest_modules(repo: Path, python_deps: Path) -> tuple[Any, Any, Any]:
    for path in (python_deps, repo):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    try:
        from satquest import CNF, create_problem, create_question
    except Exception as exc:
        raise RuntimeError(
            "Could not import the official SATQuest implementation. Run with "
            "Python >= 3.10 and install python-sat==1.8.dev14 into --python-deps."
        ) from exc
    return CNF, create_problem, create_question


def _cnf_for_problem(row: Mapping[str, Any], problem_type: str, CNF: Any) -> Any:
    key = "sat_dimacs" if problem_type in SATQUEST_SAT_TYPES else "unsat_dimacs"
    return CNF(dimacs=str(row[key]))


def _certificate_solution(problem_type: str, answer: str) -> str:
    descriptions = {
        "SATDP_SAT": "SAT decision certificate (1 means satisfiable)",
        "SATDP_UNSAT": "SAT decision certificate (0 means unsatisfiable)",
        "SATSP": "satisfying truth assignment in variable order",
        "MaxSAT": "maximum-satisfaction truth assignment in variable order",
        "MCS": "minimal correction set indicator in clause order",
        "MUS": "minimal unsatisfiable subset indicator in clause order",
    }
    return (
        "The official SATQuest PySAT verifier accepted this solver-generated "
        f"{descriptions[problem_type]}.\n\nAnswer: {answer}"
    )


def satquest_rows(
    source: Path,
    *,
    source_split: str,
    question_types: tuple[str, ...],
    batch_size: int,
    CNF: Any,
    create_problem: Any,
    create_question: Any,
    stats: Counter[str],
) -> Iterator[dict[str, Any]]:
    is_train = source_split == "train"
    for source_row in iter_parquet_rows(source, batch_size):
        source_id = str(source_row["id"])
        num_variable = int(source_row["num_variable"])
        num_clause = int(source_row["num_clause"])
        for problem_type in SATQUEST_PROBLEM_TYPES:
            cnf = _cnf_for_problem(source_row, problem_type, CNF)
            problem = create_problem(problem_type, cnf)
            answer = problem.solution
            if answer is None or not problem.check(answer):
                raise ValueError(
                    f"SATQuest rejected its generated solution for id={source_id} "
                    f"problem_type={problem_type}"
                )
            answer = str(answer)
            source_solver_metadata = (source_row.get("solver_metadatas") or {}).get(
                problem_type
            )
            verifier_payload = canonical_json(
                {
                    "answer_pattern": problem.ANSWER_PATTERN,
                    "benchmark": "SATQuest",
                    "cnf_dimacs": cnf.dimacs,
                    "num_clause": num_clause,
                    "num_variable": num_variable,
                    "problem_type": problem_type,
                    "source_id": source_id,
                    "source_solver_metadata": source_solver_metadata,
                    "verifier": "SATQuest.Problem.check",
                }
            )
            for question_type in question_types:
                question = create_question(question_type)
                prompt_text = problem.accept(question)
                index = (
                    f"satquest-{source_split}-{source_id}-"
                    f"{problem_type.lower()}-{question_type}"
                )
                extra_info = {
                    "index": index,
                    "problem": prompt_text,
                    "source_id": source_id,
                    "problem_type": problem_type,
                    "question_type": question_type,
                    "num_variable": str(num_variable),
                    "num_clause": str(num_clause),
                    "cnf_dimacs": cnf.dimacs,
                    "verifier_payload": verifier_payload,
                }
                if is_train:
                    extra_info = {
                        "index": index,
                        "problem": prompt_text,
                        "solution": _certificate_solution(problem_type, answer),
                        **{key: value for key, value in extra_info.items() if key not in {"index", "problem"}},
                    }
                else:
                    extra_info["eval_regime"] = satquest_eval_regime(
                        num_variable, question_type
                    )
                stats[f"problem_type:{problem_type}"] += 1
                stats[f"question_type:{question_type}"] += 1
                if not is_train:
                    stats[f"eval_regime:{extra_info['eval_regime']}"] += 1
                yield {
                    "prompt": [{"content": prompt_text, "role": "user"}],
                    "reward_model": {
                        "ground_truth": answer,
                        "style": f"satquest/{problem_type.lower()}",
                    },
                    "data_source": "satquest",
                    "ability": "LOGIC",
                    "extra_info": extra_info,
                }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _original_logicskills_rows(repo: Path, task: str) -> dict[str, dict[str, Any]]:
    if task == "countermodel":
        paths = [repo / "Assessors/countermodel/questions_countermodel.json"]
    else:
        paths = [
            repo / f"Assessors/{task}/questions_{task}_carroll.json",
            repo / f"Assessors/{task}/questions_{task}_english.json",
        ]
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _load_json(path):
            item_id = str(row["id"])
            if item_id in by_id:
                raise ValueError(f"Duplicate LogicSkills {task} id: {item_id}")
            by_id[item_id] = row
    return by_id


def _logicskills_target_and_verifier(
    row: Mapping[str, Any], original: Mapping[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    task = str(row["task"])
    if task == "symbolization":
        target = str(row["answer"])
        style = "logicskills/symbolization_z3"
        payload = {
            "benchmark": "LogicSkills",
            "expected_formula": target,
            "task": task,
            "verifier": "Assessors/symbolization/checker.py",
        }
    elif task == "validity":
        target = canonical_json(row["answer"])
        style = "logicskills/validity_exact"
        payload = {
            "benchmark": "LogicSkills",
            "correct_options": row["answer"],
            "domain_constraint_id": original["domain_constraint_id"],
            "option_to_sentence_id": original["option_to_sentence_id"],
            "premise_ids": original["premise_ids"],
            "task": task,
            "verifier": "exact option-set match (official Z3-verified label)",
        }
    elif task == "countermodel":
        verifier_spec = {
            "argument_ast": original["argument_ast"],
            "domain": [0, 1, 2],
        }
        # Countermodels are non-unique.  The label is therefore a verifier
        # specification, not a fabricated canonical textual countermodel.
        target = canonical_json(verifier_spec)
        style = "logicskills/countermodel_z3"
        payload = {
            "benchmark": "LogicSkills",
            **verifier_spec,
            "task": task,
            "verifier": "Assessors/countermodel/checker.py",
        }
    else:
        raise ValueError(f"Unknown LogicSkills task: {task}")
    return target, style, payload


def logicskills_rows(repo: Path, stats: Counter[str]) -> Iterator[dict[str, Any]]:
    data_dir = repo / "logicskills/data"
    for task in ("symbolization", "validity", "countermodel"):
        normalized_rows = _load_jsonl(data_dir / f"{task}.jsonl")
        original_by_id = _original_logicskills_rows(repo, task)
        if len(normalized_rows) != len(original_by_id):
            raise ValueError(
                f"LogicSkills {task} normalized/original count mismatch: "
                f"{len(normalized_rows)} != {len(original_by_id)}"
            )
        for row in normalized_rows:
            item_id = str(row["id"])
            original = original_by_id[item_id]
            target, style, verifier_payload = _logicskills_target_and_verifier(
                row, original
            )
            prompt_text = str(row["input"]).strip()
            language = str(row["language"])
            index = f"logicskills-{task}-{language}-{item_id}"
            stats[f"task:{task}"] += 1
            stats[f"language:{language}"] += 1
            yield {
                "prompt": [{"content": prompt_text, "role": "user"}],
                "reward_model": {"ground_truth": target, "style": style},
                "data_source": "logicskills",
                "ability": "LOGIC",
                "extra_info": {
                    "index": index,
                    "problem": prompt_text,
                    "task": task,
                    "language": language,
                    "eval_regime": "external_ood",
                    "verifier_payload": canonical_json(verifier_payload),
                },
            }


def _file_record(path: Path, rows: int, stats: Counter[str]) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": str(path),
        "rows": rows,
        "sha256": sha256_file(path),
        "statistics": dict(sorted(stats.items())),
    }


def _validate_output(
    path: Path,
    expected_schema: pa.Schema,
    expected_rows: int,
    *,
    expect_solution: bool,
) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"{path} has {parquet.metadata.num_rows} rows; expected {expected_rows}"
        )
    # Parquet round-tripping renames Arrow's list child field from ``item``
    # to ``element``.  Field equality correctly treats those as equivalent,
    # while schema equality with metadata enabled does not on some PyArrow
    # releases, so validate the logical schema and metadata separately.
    if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
        raise ValueError(f"Unexpected schema for {path}: {parquet.schema_arrow}")
    if parquet.schema_arrow.metadata != expected_schema.metadata:
        raise ValueError(f"Unexpected schema metadata for {path}")
    extra_fields = {field.name for field in expected_schema.field("extra_info").type}
    if ("solution" in extra_fields) != expect_solution:
        raise ValueError(f"Incorrect privileged-solution boundary in {path}")
    first = next(parquet.iter_batches(batch_size=1)).to_pylist()[0]
    if not first["prompt"][0]["content"] or not first["reward_model"]["ground_truth"]:
        raise ValueError(f"Empty prompt or reward target in {path}")
    if expect_solution and not first["extra_info"]["solution"]:
        raise ValueError(f"Empty privileged solution in {path}")


def _source_record(path: Path, *, url: str) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "url": url}


def _write_documentation(
    output_root: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> None:
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# Logic datasets

Generated in the same top-level verl parquet contract as the existing DeepMath
training split and AMC/AIME validation split:

- `satquest/train.parquet`: self-distillation data, including a solver-verified
  privileged certificate in `extra_info.solution`.
- `satquest/val.parquet`: ID, size-OOD, format-OOD, and joint size/format-OOD
  self-evaluation data. No privileged `solution` field is present.
- `logicskills/val.parquet`: fixed external-OOD evaluation data for
  symbolization, validity, and countermodel construction. No privileged
  `solution` field is present.

SATQuest training uses Math and Story at 3--4 variables. Evaluation uses the
official disjoint CNF IDs at 3--16 variables and all Math, Story, DIMACS, and
DualStory formats. LogicSkills countermodels are non-unique, so their
`ground_truth` stores the official AST/domain verifier specification rather
than a fake canonical answer.

Provenance, hashes, exact counts, and distributions are recorded in
`{manifest_path.name}`.
"""
    (output_root / "LOGIC_DATASETS.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sys.version_info < (3, 10):
        raise RuntimeError("SATQuest conversion requires Python >= 3.10")
    # Keep the user-facing /home scratch alias in provenance instead of
    # canonicalizing it to the cluster's /nfs mount path.
    scratch_root = args.scratch_root.expanduser().absolute()
    sources = scratch_root / "data/sources"
    output_root = scratch_root / "data/verl"
    satquest_repo = sources / "SATQuest"
    logicskills_repo = sources / "LogicSkills"
    python_deps = sources / "python-deps"
    satquest_train_source = (
        sources / "SATQuest-RFT-3k/data/train-00000-of-00001.parquet"
    )
    satquest_eval_source = sources / "SATQuest-HF/data/test-00000-of-00001.parquet"
    required = (
        satquest_repo,
        logicskills_repo,
        python_deps,
        satquest_train_source,
        satquest_eval_source,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source artifacts:\n" + "\n".join(missing))

    train_ids = {
        int(row["id"]) for row in iter_parquet_rows(satquest_train_source, args.batch_size)
    }
    eval_ids = {
        int(row["id"]) for row in iter_parquet_rows(satquest_eval_source, args.batch_size)
    }
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(f"SATQuest train/eval source IDs overlap: {sorted(overlap)[:10]}")

    CNF, create_problem, create_question = _satquest_modules(
        satquest_repo, python_deps
    )
    satquest_dir = output_root / "satquest"
    logicskills_dir = output_root / "logicskills"
    satquest_train = satquest_dir / "train.parquet"
    satquest_eval = satquest_dir / "val.parquet"
    logicskills_eval = logicskills_dir / "val.parquet"

    train_stats: Counter[str] = Counter()
    print(f"Writing {satquest_train}", flush=True)
    train_count = write_parquet(
        satquest_train,
        SATQUEST_TRAIN_SCHEMA,
        satquest_rows(
            satquest_train_source,
            source_split="train",
            question_types=SATQUEST_TRAIN_FORMATS,
            batch_size=args.batch_size,
            CNF=CNF,
            create_problem=create_problem,
            create_question=create_question,
            stats=train_stats,
        ),
        batch_size=args.batch_size,
    )

    satquest_eval_stats: Counter[str] = Counter()
    print(f"Writing {satquest_eval}", flush=True)
    satquest_eval_count = write_parquet(
        satquest_eval,
        SATQUEST_EVAL_SCHEMA,
        satquest_rows(
            satquest_eval_source,
            source_split="eval",
            question_types=SATQUEST_EVAL_FORMATS,
            batch_size=args.batch_size,
            CNF=CNF,
            create_problem=create_problem,
            create_question=create_question,
            stats=satquest_eval_stats,
        ),
        batch_size=args.batch_size,
    )

    logicskills_stats: Counter[str] = Counter()
    print(f"Writing {logicskills_eval}", flush=True)
    logicskills_eval_count = write_parquet(
        logicskills_eval,
        LOGICSKILLS_EVAL_SCHEMA,
        logicskills_rows(logicskills_repo, logicskills_stats),
        batch_size=args.batch_size,
    )

    expected_train = len(train_ids) * len(SATQUEST_PROBLEM_TYPES) * len(
        SATQUEST_TRAIN_FORMATS
    )
    expected_satquest_eval = len(eval_ids) * len(SATQUEST_PROBLEM_TYPES) * len(
        SATQUEST_EVAL_FORMATS
    )
    _validate_output(
        satquest_train,
        SATQUEST_TRAIN_SCHEMA,
        expected_train,
        expect_solution=True,
    )
    _validate_output(
        satquest_eval,
        SATQUEST_EVAL_SCHEMA,
        expected_satquest_eval,
        expect_solution=False,
    )
    _validate_output(
        logicskills_eval,
        LOGICSKILLS_EVAL_SCHEMA,
        1500,
        expect_solution=False,
    )
    if train_count != expected_train or satquest_eval_count != expected_satquest_eval:
        raise ValueError("SATQuest output counts do not match the source expansion")
    if logicskills_eval_count != 1500:
        raise ValueError(f"LogicSkills produced {logicskills_eval_count}, expected 1500")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format_contract": {
            "reference_eval": str(
                output_root / "amc23+aime24+aime25/val.parquet"
            ),
            "reference_train": str(
                output_root / "deepmath_filtered_level7_10/train.parquet"
            ),
            "top_level_fields": [
                "prompt",
                "reward_model",
                "data_source",
                "ability",
                "extra_info",
            ],
        },
        "outputs": {
            "logicskills_eval": _file_record(
                logicskills_eval, logicskills_eval_count, logicskills_stats
            ),
            "satquest_eval": _file_record(
                satquest_eval, satquest_eval_count, satquest_eval_stats
            ),
            "satquest_train": _file_record(
                satquest_train, train_count, train_stats
            ),
        },
        "split_contract": {
            "logicskills": "fixed official 1,500-item external-OOD benchmark",
            "satquest_eval_formats": list(SATQUEST_EVAL_FORMATS),
            "satquest_eval_source_ids": [min(eval_ids), max(eval_ids)],
            "satquest_train_formats": list(SATQUEST_TRAIN_FORMATS),
            "satquest_train_source_ids": [min(train_ids), max(train_ids)],
            "satquest_train_eval_id_overlap": 0,
            "satquest_train_size": "3--4 variables",
        },
        "sources": {
            "logicskills": {
                "git_commit": git_revision(logicskills_repo),
                "path": str(logicskills_repo),
                "url": "https://github.com/brianrabern/LogicSkills",
            },
            "satquest_code": {
                "git_commit": git_revision(satquest_repo),
                "path": str(satquest_repo),
                "url": "https://github.com/sdpkjc/SATQuest",
            },
            "satquest_eval": _source_record(
                satquest_eval_source,
                url="https://huggingface.co/datasets/sdpkjc/SATQuest",
            ),
            "satquest_train": _source_record(
                satquest_train_source,
                url="https://huggingface.co/datasets/sdpkjc/SATQuest-RFT-3k",
            ),
        },
    }
    manifest_path = output_root / "LOGIC_DATA_MANIFEST.json"
    _write_documentation(output_root, manifest, manifest_path=manifest_path)
    print(
        canonical_json(
            {
                "logicskills_eval": logicskills_eval_count,
                "manifest": str(manifest_path),
                "satquest_eval": satquest_eval_count,
                "satquest_train": train_count,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
