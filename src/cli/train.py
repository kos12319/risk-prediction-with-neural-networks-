from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.train_nn import train_from_config


def main():
    parser = argparse.ArgumentParser(description="Train credit risk NN from config")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    results = train_from_config(args.config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

