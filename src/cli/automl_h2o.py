from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cli._bootstrap import apply_safe_env
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk model using the H2O AutoML backend")
    parser.add_argument("--config", type=str, default="configs/default_automl.yaml", help="Path to YAML config")
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Free-text notes describing what changed in this run (included in W&B and README)",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="After training, download the W&B run's files/artifacts into the local run folder (requires WANDB_API_KEY)",
    )
    args = parser.parse_args()

    apply_safe_env()
    setup_logging()

    from src.training.pipeline import load_config_with_extends, train_from_config  # type: ignore
    from src.training.wandb_sync import download_run, login_from_env  # type: ignore

    cfg = load_config_with_extends(Path(args.config))
    backend = str(cfg.get("model", {}).get("backend", "")).lower()
    if backend != "h2o":
        raise ValueError("H2O CLI requires model.backend to be 'h2o'")

    login_from_env()

    results = train_from_config(args.config, notes=args.notes)
    print(json.dumps(results, indent=2))

    if args.pull and results.get("wandb_run_path") and results.get("run_dir"):
        try:
            target = Path(results["run_dir"]) / "wandb"
            target.mkdir(parents=True, exist_ok=True)
            download_run(results["wandb_run_path"], target)
        except Exception:
            pass


if __name__ == "__main__":
    main()
