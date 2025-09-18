from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.training.wandb_sync import login_from_env, download_run
from src.training.train_nn import _load_config_with_extends  # reuse config loader


def main():
    parser = argparse.ArgumentParser(description="Download W&B run files into ./wandb/<run_id> using the W&B Python API")
    parser.add_argument("--run", required=True, help="Run selector: entity/project/run_id | project/run_id | run_id")
    parser.add_argument("--config", default="configs/default.yaml", help="Config to resolve defaults (entity/project)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing contents in the target folder (default: overwrite)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing files instead of overwriting")
    args = parser.parse_args()

    # Ensure logged in (no-op if API key not set)
    login_from_env()

    # Load config to resolve defaults
    cfg = _load_config_with_extends(Path(args.config))
    tracking_cfg = cfg.get("tracking", {})
    wb_cfg = tracking_cfg.get("wandb", {})
    default_entity = os.environ.get("WANDB_ENTITY") or os.environ.get("WB_ENTITY") or wb_cfg.get("entity")
    default_project = os.environ.get("WANDB_PROJECT") or wb_cfg.get("project")

    run_sel = (args.run or "").strip("/")
    parts = [p for p in run_sel.split("/") if p]
    if not parts:
        raise SystemExit("--run must include a run id")

    if len(parts) == 3:
        entity, project, run_id = parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        if not default_entity:
            raise SystemExit("Set WANDB_ENTITY or tracking.wandb.entity for project/run_id form")
        entity, project, run_id = default_entity, parts[0], parts[1]
    elif len(parts) == 1:
        if not (default_entity and default_project):
            raise SystemExit("Set WANDB_ENTITY and WANDB_PROJECT (or tracking.wandb in config) for bare run_id form")
        entity, project, run_id = default_entity, default_project, parts[0]
    else:
        raise SystemExit("Invalid --run format")

    # Target: ./wandb-history/<run_id>
    target = Path("wandb-history") / run_id
    target.mkdir(parents=True, exist_ok=True)

    # Use the W&B Python API to download files and artifacts
    res = download_run(
        f"{entity}/{project}/{run_id}",
        target,
        entity=entity,
        project=project,
        skip_existing=(args.skip_existing and not args.force),
    )
    result = {"run_path": res.get("run_path"), "target": str(target)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
