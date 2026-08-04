from pathlib import Path

import pytest

from scripts.clean_self_distill import empirical_assets as assets


def materialize(tmp_path: Path, monkeypatch):
    sizes = {"config.json": 3, "model.safetensors": 5}
    monkeypatch.setattr(assets, "MODEL_FILES", sizes)
    monkeypatch.setattr(assets, "MODEL_BYTES", 8)
    monkeypatch.setattr(assets, "DEEPMATH_BYTES", 4)
    model = tmp_path / "model"
    model.mkdir()
    for name, size in sizes.items():
        (model / name).write_bytes(b"x" * size)
    data = tmp_path / "data.parquet"
    data.write_bytes(b"data")
    return model, data


def test_asset_gate_enforces_both_budgets(tmp_path: Path, monkeypatch):
    model, data = materialize(tmp_path, monkeypatch)
    manifest = assets.verify_assets(
        model_dir=model,
        deepmath=data,
        task_root=tmp_path,
        max_new_download_bytes=20,
        max_task_bytes=100,
        full_hash=False,
    )
    assert manifest["new_download_bytes"] == 12
    with pytest.raises(assets.AssetError, match="Pinned downloads"):
        assets.verify_assets(
            model_dir=model,
            deepmath=data,
            task_root=tmp_path,
            max_new_download_bytes=11,
            max_task_bytes=100,
            full_hash=False,
        )
    with pytest.raises(assets.AssetError, match="Task scratch"):
        assets.verify_assets(
            model_dir=model,
            deepmath=data,
            task_root=tmp_path,
            max_new_download_bytes=20,
            max_task_bytes=11,
            full_hash=False,
        )


def test_asset_gate_rejects_escape_and_size_mismatch(tmp_path: Path, monkeypatch):
    model, data = materialize(tmp_path, monkeypatch)
    (model / "config.json").write_bytes(b"bad!")
    with pytest.raises(assets.AssetError, match="size mismatch"):
        assets.verify_assets(
            model_dir=model,
            deepmath=data,
            task_root=tmp_path,
            max_new_download_bytes=20,
            max_task_bytes=100,
            full_hash=False,
        )
    outside = tmp_path.parent / "outside.parquet"
    outside.write_bytes(b"data")
    with pytest.raises(assets.AssetError, match="escapes"):
        assets.verify_assets(
            model_dir=model,
            deepmath=outside,
            task_root=tmp_path,
            max_new_download_bytes=20,
            max_task_bytes=100,
            full_hash=False,
        )
