from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cli._bootstrap import apply_safe_env
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train credit risk NN from config")
    parser.add_argument("--config", type=str, default="configs/pytorch_default.yaml", help="Path to YAML config")
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
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (ignore CUDA/MPS)",
    )
    args = parser.parse_args()

    # Apply safe env before importing heavy libs (NumPy/Torch)
    apply_safe_env()
    setup_logging()

    # Import heavy modules after environment is set
    from src.training.config import load_config_with_extends, validate_and_normalize_config  # type: ignore
    from src.training.backends.pytorch import train_from_config as pytorch_train  # type: ignore
    from src.training.wandb_sync import login_from_env, download_run  # type: ignore

    # Proactively login to W&B via env if available (no-op if not set)
    login_from_env()

    # Optionally force CPU via env for training loop
    if args.cpu:
        import os as _os
        _os.environ["FORCE_CPU"] = "1"

    # Ensure the config targets the PyTorch backend; H2O runs use a dedicated CLI
    cfg = load_config_with_extends(Path(args.config))
    cfg = validate_and_normalize_config(cfg, cfg_path=Path(args.config))
    backend = str(cfg.get("model", {}).get("backend", "pytorch")).lower()
    if backend != "pytorch":
        raise ValueError(
            "PyTorch training CLI requires model.backend to be 'pytorch'. "
            "Use `make automl-h2o` for H2O AutoML runs."
        )

    results = pytorch_train(args.config, notes=args.notes)
    print(json.dumps(results, indent=2))

    # Optional: pull W&B run data into the local run folder
    if args.pull and results.get("wandb_run_path") and results.get("run_dir"):
        try:
            target = Path(results["run_dir"]) / "wandb"
            target.mkdir(parents=True, exist_ok=True)
            download_run(results["wandb_run_path"], target)
        except Exception:
            pass


if __name__ == "__main__":
    main()
