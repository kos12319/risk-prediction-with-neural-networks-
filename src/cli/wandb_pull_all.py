from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.training.wandb_sync import login_from_env, download_run
import yaml


def _load_config_with_extends_light(cfg_path: Path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    extends = cfg.get("extends")
    if extends:
        base_candidate = cfg_path.parent / f"{extends}.yaml"
        base_path = base_candidate if base_candidate.exists() else Path(extends)
        base_cfg = _load_config_with_extends_light(base_path)
        # Shallow-merge dicts with child overriding base (deep enough for our usage here)
        def _deep_merge(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                out = dict(a)
                for k, v in b.items():
                    out[k] = _deep_merge(a.get(k), v)
                return out
            return b if b is not None else a

        merged = _deep_merge(base_cfg, {k: v for k, v in cfg.items() if k != "extends"})
        return merged
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Download all W&B runs for an entity/project")
    parser.add_argument("--entity", default=None, help="W&B entity/org (falls back to WANDB_ENTITY)")
    parser.add_argument("--project", default=None, help="W&B project (falls back to WANDB_PROJECT or config tracking.wandb.project)")
    parser.add_argument("--target-root", default=None, help="Root directory for downloads (defaults to output.runs_root from config)")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path to resolve defaults")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip runs whose target exists; also skip existing files/artifacts within runs (default)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Do not skip existing targets; re-download/overwrite files",
    )
    args = parser.parse_args()

    # Ensure logged in (no-op if API key not set)
    login_from_env()

    # Resolve defaults from env and config
    cfg = _load_config_with_extends_light(Path(args.config))
    tracking_cfg = cfg.get("tracking", {})
    wb_cfg = tracking_cfg.get("wandb", {})

    entity = args.entity or os.environ.get("WANDB_ENTITY") or os.environ.get("WB_ENTITY") or wb_cfg.get("entity")
    project = args.project or os.environ.get("WANDB_PROJECT") or wb_cfg.get("project")
    # Fixed target root: use ./wandb-history to avoid mixing with local SDK runs
    target_root = Path(args.target_root or "wandb-history")

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
        target = target_root / str(run_id)
        # Skip existing targets unless forced
        if not args.force and args.skip_existing and target.exists() and any(target.iterdir()):
            summary["downloads"].append({
                "run_id": run_id,
                "run_path": f"{entity}/{project}/{run_id}",
                "skipped": True,
                "reason": "target exists",
                "target": str(target),
            })
            continue
        try:
            target.mkdir(parents=True, exist_ok=True)
            res = download_run(
                f"{entity}/{project}/{run_id}",
                target,
                entity=entity,
                project=project,
                skip_existing=(args.skip_existing and not args.force),
            )
            summary["downloads"].append({"run_id": run_id, **res, "target": str(target)})
        except Exception as e:
            summary["downloads"].append({"run_id": run_id, "error": str(e)})

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
