from __future__ import annotations

import argparse
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import yaml

from src.cli._bootstrap import apply_safe_env
from src.utils.logging import setup_logging


@contextmanager
def _prepare_transient_config(base_cfg_path: Path):
    """Create a temp config with tracking disabled and isolated outputs."""
    with tempfile.TemporaryDirectory(prefix="dryrun_h2o_") as tmp:
        tmpdir = Path(tmp)
        from src.training.config import load_config_with_extends, validate_and_normalize_config  # type: ignore

        base_cfg: Dict[str, Any] = load_config_with_extends(base_cfg_path)
        base_cfg = validate_and_normalize_config(base_cfg, cfg_path=base_cfg_path)
        out_cfg = dict(base_cfg.get("output", {}))
        out_cfg.update(
            {
                "models_dir": (tmpdir / "models").as_posix(),
                "reports_dir": (tmpdir / "reports").as_posix(),
                "figures_dir": (tmpdir / "reports" / "figures").as_posix(),
                "runs_root": (tmpdir / "runs").as_posix(),
            }
        )
        base_cfg["output"] = out_cfg

        tracking = dict(base_cfg.get("tracking", {}))
        tracking.update({"backend": "none"})
        wandb_cfg = dict(tracking.get("wandb", {}))
        wandb_cfg.update({"enabled": False, "mode": "disabled"})
        tracking["wandb"] = wandb_cfg
        base_cfg["tracking"] = tracking

        dry_cfg_path = tmpdir / "config_dry.yaml"
        dry_cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

        yield dry_cfg_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an H2O AutoML dry training experiment using temp output dirs (no artifacts persisted)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h2o_default.yaml",
        help="Path to YAML config (H2O backend only)",
    )
    args = parser.parse_args()

    apply_safe_env()
    setup_logging()

    base_cfg_path = Path(args.config).resolve()
    if not base_cfg_path.exists():
        raise SystemExit(f"Config not found: {base_cfg_path}")

    from src.training.config import load_config_with_extends, validate_and_normalize_config  # type: ignore
    cfg = load_config_with_extends(base_cfg_path)
    cfg = validate_and_normalize_config(cfg, cfg_path=base_cfg_path)
    backend = str(cfg.get("model", {}).get("backend", "h2o")).lower()
    if backend not in {"", "h2o"}:
        raise SystemExit("H2O dryrun requires model.backend to be 'h2o' (or omitted)")

    from src.training.backends.h2o import train_from_config as h2o_train  # type: ignore

    with _prepare_transient_config(base_cfg_path) as dry_cfg_path:
        results = h2o_train(dry_cfg_path)
        print(json.dumps({"dry_run": True, **results}, indent=2))


if __name__ == "__main__":
    main()
