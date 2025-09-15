from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.data.load import LoadConfig, load_and_prepare
from src.selection.mi_selection import run_mi_selection
from src.selection.l1_selection import run_l1_selection


def load_base_config(cfg_path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run feature selection and save ranked subset")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--method", type=str, default="mi", choices=["mi", "l1"], help="Selection method")
    parser.add_argument("--target_coverage", type=float, default=0.99, help="Fraction of full AUC to achieve")
    parser.add_argument("--missingness_threshold", type=float, default=0.5, help="Drop columns with missing rate above this")
    parser.add_argument("--max_features", type=int, default=0, help="Cap on number of features (0 = no cap)")
    parser.add_argument("--outdir", type=str, default="reports/selection", help="Output directory for artifacts")
    args = parser.parse_args()

    cfg = load_base_config(args.config)
    data_cfg = cfg["data"]
    split_cfg = cfg.get("split", {})

    load_config = LoadConfig(
        csv_path=data_cfg["csv_path"],
        target_col=data_cfg["target_col"],
        target_mapping=data_cfg["target_mapping"],
        parse_dates=data_cfg.get("parse_dates", []),
        drop_leakage=data_cfg.get("drop_leakage", True),
        leakage_cols=data_cfg.get("leakage_cols", []),
        features=data_cfg.get("features", []),
    )
    df = load_and_prepare(load_config)

    features = [
        c
        for c in data_cfg.get("features", [])
        if c != data_cfg["target_col"] and c not in data_cfg.get("parse_dates", []) and c in df.columns
    ]

    outdir = Path(args.outdir) / args.method
    outdir.mkdir(parents=True, exist_ok=True)

    max_features = args.max_features if args.max_features and args.max_features > 0 else None

    common_kwargs = dict(
        df=df,
        features=features,
        target_col=data_cfg["target_col"],
        split_method=split_cfg.get("method", "time"),
        time_col=split_cfg.get("time_col", "issue_d"),
        test_size=split_cfg.get("test_size", 0.2),
        random_state=split_cfg.get("random_state", 42),
        missingness_threshold=args.missingness_threshold,
        target_coverage=args.target_coverage,
        max_features=max_features,
        outdir=outdir,
    )

    if args.method == "mi":
        results = run_mi_selection(**common_kwargs)
    else:
        # L1 selection (with default C from function)
        results = run_l1_selection(**common_kwargs)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

