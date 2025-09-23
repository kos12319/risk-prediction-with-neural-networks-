from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from src.cli._bootstrap import apply_safe_env
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a dry training experiment using temp output dirs (no artifacts persisted)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (will be extended by a temporary override)",
    )
    args = parser.parse_args()

    # Apply safe env before importing heavy libs (NumPy/Torch)
    apply_safe_env()
    setup_logging()

    # Import after env is set
    from src.training.pipeline import load_config_with_extends, train_from_config  # type: ignore

    base_cfg_path = Path(args.config).resolve()
    if not base_cfg_path.exists():
        raise SystemExit(f"Config not found: {base_cfg_path}")

    with tempfile.TemporaryDirectory(prefix="dryrun_") as tmp:
        tmpdir = Path(tmp)
        # Resolve the config (handles extends/merges) before applying dry-run overrides
        base_cfg: Dict[str, Any] = load_config_with_extends(base_cfg_path)
        out = dict(base_cfg.get("output", {}))
        out.update(
            {
                "models_dir": (tmpdir / "models").as_posix(),
                "reports_dir": (tmpdir / "reports").as_posix(),
                "figures_dir": (tmpdir / "reports" / "figures").as_posix(),
            }
        )
        base_cfg["output"] = out

        # Ensure no experiment logging for dry runs
        tracking = dict(base_cfg.get("tracking", {}))
        tracking.update({"backend": "none"})
        wb = dict(tracking.get("wandb", {}))
        wb.update({"enabled": False, "mode": "disabled"})
        tracking["wandb"] = wb
        base_cfg["tracking"] = tracking

        dry_cfg_path = tmpdir / "config_dry.yaml"
        dry_cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

        # Run training with the temporary config; artifacts land under tmpdir and are removed after exit
        results = train_from_config(dry_cfg_path)
        print(json.dumps({"dry_run": True, **results}, indent=2))


if __name__ == "__main__":
    main()
