from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from src.training.wandb_sync import login_from_env
from src.training.train_nn import _load_config_with_extends  # reuse config loader


def main():
    parser = argparse.ArgumentParser(description="Delete W&B runs and their logged artifacts for a project")
    parser.add_argument("--entity", default=None, help="W&B entity/org (falls back to env/config)")
    parser.add_argument("--project", default=None, help="W&B project (falls back to env/config)")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path to resolve defaults")
    parser.add_argument("--runs-only", action="store_true", help="Delete runs only, keep artifacts")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion without prompt")
    args = parser.parse_args()

    # Ensure logged in
    login_from_env()

    # Resolve defaults
    cfg = _load_config_with_extends(Path(args.config))
    tracking_cfg = cfg.get("tracking", {})
    wb_cfg = tracking_cfg.get("wandb", {})
    entity = args.entity or os.environ.get("WANDB_ENTITY") or os.environ.get("WB_ENTITY") or wb_cfg.get("entity")
    project = args.project or os.environ.get("WANDB_PROJECT") or wb_cfg.get("project")
    if not entity or not project:
        raise SystemExit("Set --entity/--project or provide WANDB_ENTITY/WANDB_PROJECT or tracking.wandb in config")

    if not args.yes:
        raise SystemExit("Refusing to delete without --yes. Re-run with --yes to confirm.")

    try:
        from wandb.apis.public import Api  # type: ignore
    except Exception as e:
        raise SystemExit(f"Failed to import wandb API: {e}")

    api = Api()

    # List runs
    runs = list(api.runs(f"{entity}/{project}"))
    summary = {"entity": entity, "project": project, "run_count": len(runs), "deleted": []}

    for run in runs:
        rid = getattr(run, "id", None) or getattr(run, "name", None)
        rpath = f"{entity}/{project}/{rid}" if rid else None
        deleted_arts: List[str] = []
        # Optionally delete logged artifacts for this run
        if not args.runs_only:
            try:
                for art in run.logged_artifacts():
                    try:
                        name = getattr(art, "name", None) or "unknown"
                        art.delete()
                        deleted_arts.append(str(name))
                    except Exception:
                        continue
            except Exception:
                pass
        # Delete the run
        try:
            run.delete()
            summary["deleted"].append({
                "run_id": rid,
                "run_path": rpath,
                "artifacts_deleted": deleted_arts,
            })
        except Exception as e:
            summary["deleted"].append({
                "run_id": rid,
                "run_path": rpath,
                "error": str(e),
                "artifacts_deleted": deleted_arts,
            })

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
