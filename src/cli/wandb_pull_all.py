from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.training.wandb_sync import login_from_env, download_run
from src.training.train_nn import _load_config_with_extends  # reuse config loader


def main():
    parser = argparse.ArgumentParser(description="Download all W&B runs for an entity/project")
    parser.add_argument("--entity", default=None, help="W&B entity/org (falls back to WANDB_ENTITY)")
    parser.add_argument("--project", default=None, help="W&B project (falls back to WANDB_PROJECT or config tracking.wandb.project)")
    parser.add_argument("--target-root", default=None, help="Root directory for downloads (defaults to output.runs_root from config)")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path to resolve defaults")
    args = parser.parse_args()

    # Ensure logged in (no-op if API key not set)
    login_from_env()

    # Resolve defaults from env and config
    cfg = _load_config_with_extends(Path(args.config))
    out_cfg = cfg.get("output", {})
    tracking_cfg = cfg.get("tracking", {})
    wb_cfg = tracking_cfg.get("wandb", {})

    entity = args.entity or os.environ.get("WANDB_ENTITY") or os.environ.get("WB_ENTITY") or wb_cfg.get("entity")
    project = args.project or os.environ.get("WANDB_PROJECT") or wb_cfg.get("project")
    target_root = Path(args.target_root or out_cfg.get("runs_root", "local_runs"))

    if not entity:
        raise SystemExit("W&B entity is required. Set --entity or WANDB_ENTITY")
    if not project:
        raise SystemExit("W&B project is required. Set --project, WANDB_PROJECT, or tracking.wandb.project in config")

    # List runs via public API and download each
    try:
        from wandb.apis.public import Api  # type: ignore
    except Exception as e:
        raise SystemExit(f"Failed to import wandb API: {e}")

    api = Api()
    runs = list(api.runs(f"{entity}/{project}"))

    summary = {"entity": entity, "project": project, "count": len(runs), "downloads": []}
    for run in runs:
        run_id = getattr(run, "id", None) or getattr(run, "name", None)
        if not run_id:
            # Fallback: use path tail
            run_id = str(getattr(run, "path", ["unknown"]))[-1]
        target = target_root / str(run_id) / "wandb"
        try:
            res = download_run(f"{entity}/{project}/{run_id}", target, entity=entity, project=project)
            summary["downloads"].append({"run_id": run_id, **res})
        except Exception as e:
            summary["downloads"].append({"run_id": run_id, "error": str(e)})

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

