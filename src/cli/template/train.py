from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cli._bootstrap import apply_safe_env
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train Template (sklearn) backend from config")
    parser.add_argument("--config", type=str, default="configs/template_default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Free-text notes describing what changed in this run (included in W&B and README)",
    )
    args = parser.parse_args()

    # Apply safe env before importing heavy libs
    apply_safe_env()
    setup_logging()

    from src.training.config import load_config_with_extends  # type: ignore
    from src.training.backends.template import train_from_config as tpl_train  # type: ignore

    cfg = load_config_with_extends(Path(args.config))
    backend = str(cfg.get("model", {}).get("backend", "template")).lower()
    if backend not in {"", "template"} and backend != "template":
        raise ValueError(
            "Template training CLI requires model.backend to be 'template' or omitted."
        )

    results = tpl_train(args.config, notes=args.notes)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

