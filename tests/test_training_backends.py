from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.training.pipeline import train_from_config


def test_train_from_config_rejects_unknown_backend(tmp_path):
    base_cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    base_cfg["data"]["csv_path"] = "data/raw/samples/thesis_data_sample_100.csv"
    base_cfg["output"] = {"runs_root": (tmp_path / "runs").as_posix()}
    base_cfg.setdefault("tracking", {})["backend"] = "none"
    base_cfg["tracking"].setdefault("wandb", {})["enabled"] = False
    base_cfg["tracking"]["wandb"]["mode"] = "disabled"
    base_cfg["model"]["backend"] = "unknown"
    base_cfg["model"]["epochs"] = 1
    cfg_path = tmp_path / "invalid_backend.yaml"
    cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported model backend"):
        train_from_config(cfg_path)
