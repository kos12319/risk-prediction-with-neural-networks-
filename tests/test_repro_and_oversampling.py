from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.training.backends.pytorch import train_from_config as pytorch_train
from src.training.config import load_config_with_extends, validate_and_normalize_config


def _mk_tmp_cfg(base_cfg_path: Path, tmp_path: Path, *, epochs: int = 2, oversampling_enabled: bool | None = None) -> Path:
    base = load_config_with_extends(base_cfg_path)
    base = validate_and_normalize_config(base, cfg_path=base_cfg_path)

    # Disable tracking and redirect outputs under tmp_path (single-run mode via runs_root)
    base.setdefault("tracking", {})["backend"] = "none"
    base["tracking"].setdefault("wandb", {})
    base["tracking"]["wandb"].update({"enabled": False, "mode": "disabled"})

    # Ensure a tiny model/run
    base.setdefault("model", {})
    base["model"]["epochs"] = int(epochs)
    base["model"]["batch_size"] = 64

    # Keep time split and a small sample CSV to keep runtime low
    base.setdefault("data", {})
    base["data"]["csv_path"] = "data/raw/samples/thesis_data_sample_1k.csv"

    # Output in single-run directory mode for easy artifact discovery
    base["output"] = {"runs_root": (tmp_path / "runs").as_posix()}

    if oversampling_enabled is not None:
        base.setdefault("oversampling", {})["enabled"] = bool(oversampling_enabled)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return cfg_path


def _latest_two_run_dirs(runs_root: Path) -> list[Path]:
    all_runs = sorted([p for p in runs_root.rglob("run_*") if p.is_dir()])
    return all_runs[-2:]


def test_determinism_same_seed_same_metrics(tmp_path: Path):
    # Build a tiny single-run config
    cfg_path = _mk_tmp_cfg(Path("configs/pytorch/base.yaml").resolve(), tmp_path, epochs=2, oversampling_enabled=False)

    # Run twice with the same seed and config
    pytorch_train(cfg_path)
    pytorch_train(cfg_path)

    run_dirs = _latest_two_run_dirs(tmp_path / "runs")
    assert len(run_dirs) == 2, "Expected two runs to compare for determinism"

    m1 = json.loads((run_dirs[0] / "metrics.json").read_text(encoding="utf-8"))
    m2 = json.loads((run_dirs[1] / "metrics.json").read_text(encoding="utf-8"))

    # Exact match is expected on CPU with fixed seeds
    assert m1 == m2, f"Metrics differ across identical seeds: {m1} vs {m2}"


def test_oversampling_train_only_manifest(tmp_path: Path):
    # Enable oversampling and ensure manifest reports before/after for train only
    cfg_path = _mk_tmp_cfg(Path("configs/pytorch/base.yaml").resolve(), tmp_path, epochs=1, oversampling_enabled=True)

    pytorch_train(cfg_path)

    run_dirs = [p for p in (tmp_path / "runs").rglob("run_*") if p.is_dir()]
    assert run_dirs, "No run directory found"
    run_dir = sorted(run_dirs)[-1]

    manifest_path = run_dir / "data_manifest.json"
    assert manifest_path.exists(), "data_manifest.json not found"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_counts = manifest.get("train_class_counts", {})
    test_counts = manifest.get("test_class_counts", {})
    val_counts = manifest.get("val_class_counts", {})
    res = manifest.get("resampling", {})
    assert train_counts and test_counts, "split class counts missing from manifest"
    assert set(res.keys()) >= {"method", "before_counts", "after_counts"}, "resampling details missing"

    # Train counts in manifest reflect post-resampling labels
    assert res["after_counts"] == train_counts, "Train counts must match after resampling"
    # Before resampling counts should differ when imbalance exists (best-effort; allow equality fallback)
    if res["before_counts"] != test_counts:  # only check when there was imbalance to begin with
        assert res["before_counts"] != train_counts, "Before counts should differ from the resampled train counts when oversampled"
    # Validation/test splits must not be affected by resampling
    if val_counts:
        assert val_counts != res["after_counts"], "Validation counts must not equal resampled train after_counts"
    assert test_counts != res["after_counts"], "Test counts must not equal resampled train after_counts"
