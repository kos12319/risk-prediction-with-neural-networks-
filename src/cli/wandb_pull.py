from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.wandb_sync import login_from_env, download_run
from src.training.train_nn import _load_config_with_extends  # reuse config loader


def main():
    parser = argparse.ArgumentParser(description="Download W&B run files and artifacts to local folder")
    parser.add_argument("--run", required=True, help="Run path or id: entity/project/run_id | project/run_id | run_id")
    parser.add_argument("--target", default=None, help="Target directory to download into (defaults to output.runs_root/<run_id>/wandb)")
    parser.add_argument("--config", default="configs/default.yaml", help="Config to resolve output.runs_root when --target is omitted")
    args = parser.parse_args()

    # Ensure logged in (no-op if API key not set)
    login_from_env()

    # Resolve default target
    target = args.target
    if target is None:
        cfg = _load_config_with_extends(Path(args.config))
        out_cfg = cfg.get("output", {})
        runs_root = out_cfg.get("runs_root", "local_runs")
        run_id = str(args.run).strip("/").split("/")[-1]
        target = Path(runs_root) / run_id / "wandb"

    res = download_run(args.run, target)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

