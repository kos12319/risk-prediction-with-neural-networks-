from __future__ import annotations

import json
from pathlib import Path

from src.training.backends.pytorch.pipeline import train_from_config as pytorch_train


def test_cv_writes_cv_metrics_and_structure(tmp_path: Path):
    # Use the PyTorch CV smoke config and a transient output to keep runs isolated/fast
    cfg_path = Path("configs/pytorch/cv_smoke.yaml").resolve()
    assert cfg_path.exists(), "cv_smoke.yaml must exist for this test"

    # Build a temporary config with tracking disabled and outputs redirected under tmp_path
    from src.training.config import load_config_with_extends, validate_and_normalize_config
    import yaml

    base_cfg = load_config_with_extends(cfg_path)
    base_cfg = validate_and_normalize_config(base_cfg, cfg_path=cfg_path)

    out_cfg = dict(base_cfg.get("output", {}))
    out_cfg.update(
        {
            "models_dir": (tmp_path / "models").as_posix(),
            "reports_dir": (tmp_path / "reports").as_posix(),
            "figures_dir": (tmp_path / "reports" / "figures").as_posix(),
            "runs_root": (tmp_path / "runs").as_posix(),
        }
    )
    base_cfg["output"] = out_cfg

    tracking = dict(base_cfg.get("tracking", {}))
    wandb_cfg = dict(tracking.get("wandb", {}))
    tracking["backend"] = "none"
    wandb_cfg.update({"enabled": False, "mode": "disabled"})
    tracking["wandb"] = wandb_cfg
    base_cfg["tracking"] = tracking

    dry_cfg = tmp_path / "config_dry.yaml"
    dry_cfg.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    # Run training; with CV enabled the function returns a CV summary
    result = pytorch_train(dry_cfg)
    assert isinstance(result, dict) and result.get("cv_metrics_path"), "Expect cv_metrics_path in result"

    cv_path = Path(result["cv_metrics_path"]).resolve()
    assert cv_path.exists(), f"cv_metrics.json not found at {cv_path}"

    # Minimal schema sanity: has aggregate and folds with required keys
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    assert "aggregate" in cv and "folds" in cv and isinstance(cv["folds"], list)
    assert all(
        set(["fold_id", "run_id", "metrics", "confusion", "threshold"]).issubset(set(rec.keys()))
        for rec in cv["folds"]
    )

