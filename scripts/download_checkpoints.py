#!/usr/bin/env python3
"""Download legacy reverse-KL Qwen3-8B adapters for reproduction."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


TAG = "qwen3-8b-checkpoints-v1"
FILES = ("adapter_model.safetensors", "adapter_config.json", "checkpoint_manifest.json")
LEGACY_ASSET_METHOD = {"legacy-trsd": "trsd", "legacy-opsd": "opsd"}


def infer_repository() -> str:
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    match = re.search(r"github\.com[/:]([^/]+/[^/]+?)(?:\.git)?$", remote)
    if match is None:
        raise RuntimeError("Cannot infer GitHub repository; pass --repository OWNER/REPO")
    return match.group(1)


def download(method: str, repository: str, output_root: Path) -> Path:
    asset_method = LEGACY_ASSET_METHOD[method]
    prefix = f"qwen3-8b-{asset_method}-ep64"
    destination = output_root / f"qwen3-8b-{method}-reverse-kl-ep64"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{prefix}.", dir=output_root) as temporary:
        temporary_path = Path(temporary)
        for filename in FILES:
            asset = f"{prefix}-{filename}"
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    TAG,
                    "--repo",
                    repository,
                    "--pattern",
                    asset,
                    "--dir",
                    str(temporary_path),
                ],
                check=True,
            )
            shutil.move(str(temporary_path / asset), destination / filename)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("legacy-trsd", "legacy-opsd", "all"),
        required=True,
        help="These v1 assets use reverse KL and are not forward-KL LGSD",
    )
    parser.add_argument("--repository", help="GitHub OWNER/REPOSITORY; inferred by default")
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()
    repository = args.repository or infer_repository()
    methods = tuple(LEGACY_ASSET_METHOD) if args.method == "all" else (args.method,)
    for method in methods:
        print(download(method, repository, Path(args.output_dir)))


if __name__ == "__main__":
    main()
