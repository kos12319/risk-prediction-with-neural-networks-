from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from src.training.train_nn import train_from_config


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

    base_cfg_path = Path(args.config).resolve()
    if not base_cfg_path.exists():
        raise SystemExit(f"Config not found: {base_cfg_path}")

    with tempfile.TemporaryDirectory(prefix="dryrun_") as tmp:
        tmpdir = Path(tmp)
        # Load base config and merge output overrides (no extends to keep it simple)
        base_cfg: Dict[str, Any] = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}
        out = dict(base_cfg.get("output", {}))
        out.update(
            {
                "models_dir": (tmpdir / "models").as_posix(),
                "reports_dir": (tmpdir / "reports").as_posix(),
                "figures_dir": (tmpdir / "reports" / "figures").as_posix(),
            }
        )
        base_cfg["output"] = out

        dry_cfg_path = tmpdir / "config_dry.yaml"
        dry_cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

        # Run training with the temporary config; artifacts land under tmpdir and are removed after exit
        results = train_from_config(dry_cfg_path)
        print(json.dumps({"dry_run": True, **results}, indent=2))


if __name__ == "__main__":
    main()
