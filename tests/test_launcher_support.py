"""Focused regression tests for the Slurm shard validator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.clean_self_distill.slurm.launcher_support import cmd_validate_shard
from src.clean_self_distill.io import load_query_records


class LauncherSupportTest(unittest.TestCase):
    def test_task2_accepts_the_task_marker_written_by_train_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "problem": "Find the value.",
                        "answer": "1",
                        "source": "amc23",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            record = load_query_records(dataset, include_targets=True)[0]
            artifact = root / "task2.jsonl"
            artifact.write_text(
                json.dumps(
                    {
                        "query_id": record["query_id"],
                        "problem_sha256": record["problem_sha256"],
                        "source": record["source"],
                        "model": "model",
                        "model_revision": "revision",
                        "task": "task2_clean_distillation",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cmd_validate_shard(
                SimpleNamespace(
                    dataset=str(dataset),
                    max_samples=None,
                    num_shards=1,
                    shard_index=0,
                    kind="task2",
                    artifact=str(artifact),
                    model="model",
                    revision="revision",
                )
            )


if __name__ == "__main__":
    unittest.main()
