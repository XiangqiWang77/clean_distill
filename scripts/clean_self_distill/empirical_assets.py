#!/usr/bin/env python3
"""Fail-closed asset and scratch-budget gate for the adopted 8B experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODEL_FILES = {
    ".gitattributes": 1570,
    "LICENSE": 11343,
    "README.md": 16660,
    "config.json": 728,
    "generation_config.json": 239,
    "merges.txt": 1671853,
    "model-00001-of-00005.safetensors": 3996250744,
    "model-00002-of-00005.safetensors": 3993160032,
    "model-00003-of-00005.safetensors": 3959604768,
    "model-00004-of-00005.safetensors": 3187841392,
    "model-00005-of-00005.safetensors": 1244659840,
    "model.safetensors.index.json": 32878,
    "tokenizer.json": 11422654,
    "tokenizer_config.json": 9732,
    "vocab.json": 2776833,
}
MODEL_SHA256 = {
    ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "README.md": "0f36caaff9c2516411a7738db384606263ba653c1e63e61d72f511606164d5a6",
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
    "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
MODEL_BYTES = sum(MODEL_FILES.values())
DEEPMATH_REVISION = "33d7de919af5b03257ff92c30303fddf9afdda4a"
DEEPMATH_BYTES = 587568701
DEEPMATH_SHA256 = "611d3030a2a74eaea9514ab732fc33aa6a35d668c7f804c383772939b159f2a0"


class AssetError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_regular_bytes(root: Path) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += stat.st_size
    return total


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def verify_assets(
    *,
    model_dir: Path,
    deepmath: Path,
    task_root: Path,
    max_new_download_bytes: int,
    max_task_bytes: int,
    full_hash: bool,
) -> dict[str, Any]:
    for path in (model_dir, deepmath, task_root):
        if not path.resolve().is_relative_to(task_root.resolve()):
            raise AssetError(f"Asset escapes task scratch root: {path}")
    mismatches = {
        name: (expected, (model_dir / name).stat().st_size if (model_dir / name).is_file() else None)
        for name, expected in MODEL_FILES.items()
        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size != expected
    }
    if mismatches:
        raise AssetError(f"Qwen3-8B local snapshot size mismatch: {mismatches}")
    if not deepmath.is_file() or deepmath.stat().st_size != DEEPMATH_BYTES:
        actual = deepmath.stat().st_size if deepmath.exists() else None
        raise AssetError(
            f"DeepMath size mismatch expected={DEEPMATH_BYTES} actual={actual}"
        )
    new_download_bytes = MODEL_BYTES + DEEPMATH_BYTES
    if new_download_bytes > max_new_download_bytes:
        raise AssetError(
            f"Pinned downloads use {new_download_bytes}>{max_new_download_bytes} bytes"
        )
    task_bytes = tree_regular_bytes(task_root)
    if task_bytes > max_task_bytes:
        raise AssetError(f"Task scratch uses {task_bytes}>{max_task_bytes} bytes")

    model_hashes: dict[str, str] | None = None
    dataset_hash = DEEPMATH_SHA256
    if full_hash:
        dataset_hash = sha256_file(deepmath)
        if dataset_hash != DEEPMATH_SHA256:
            raise AssetError(
                f"DeepMath SHA mismatch expected={DEEPMATH_SHA256} actual={dataset_hash}"
            )
        model_hashes = {name: sha256_file(model_dir / name) for name in MODEL_FILES}
        hash_mismatches = {
            name: (MODEL_SHA256[name], actual)
            for name, actual in model_hashes.items()
            if actual != MODEL_SHA256[name]
        }
        if hash_mismatches:
            raise AssetError(
                f"Qwen3-8B local snapshot SHA-256 mismatch: {hash_mismatches}"
            )
    return {
        "schema_version": "clean-self-distill-empirical-assets-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dir": str(model_dir.resolve()),
        "model_bytes": MODEL_BYTES,
        "model_files": MODEL_FILES,
        "model_sha256": model_hashes,
        "deepmath_revision": DEEPMATH_REVISION,
        "deepmath_path": str(deepmath.resolve()),
        "deepmath_bytes": DEEPMATH_BYTES,
        "deepmath_sha256": dataset_hash,
        "new_download_bytes": new_download_bytes,
        "max_new_download_bytes": max_new_download_bytes,
        "task_scratch_bytes": task_bytes,
        "max_task_scratch_bytes": max_task_bytes,
        "full_hash_verified": full_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--deepmath", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--max-new-download-bytes", type=int, required=True)
    parser.add_argument("--max-task-bytes", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    manifest = verify_assets(
        model_dir=Path(args.model_dir),
        deepmath=Path(args.deepmath),
        task_root=Path(args.task_root),
        max_new_download_bytes=args.max_new_download_bytes,
        max_task_bytes=args.max_task_bytes,
        full_hash=args.full_hash,
    )
    _atomic_json(Path(args.manifest), manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
