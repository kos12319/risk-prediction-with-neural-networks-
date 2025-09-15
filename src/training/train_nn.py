from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf

from sklearn.utils import check_random_state

from src.data.load import LoadConfig, load_and_prepare
from src.data.split import random_split, time_based_split
from src.eval.metrics import compute_metrics, plot_learning_curves, save_metrics
from src.features.preprocess import build_preprocessor, identify_feature_types
from src.models.nn import build_mlp


def _ensure_dirs(paths: List[Path]):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _to_dense(X):
    if hasattr(X, "todense"):
        X = X.todense()
    return np.asarray(X)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config_with_extends(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    extends = cfg.get("extends")
    if extends:
        # Resolve base path relative to this config file
        base_candidate = cfg_path.parent / f"{extends}.yaml"
        base_path = base_candidate if base_candidate.exists() else Path(extends)
        base_cfg = _load_config_with_extends(base_path)
        # Child overrides base
        merged = _deep_merge(base_cfg, {k: v for k, v in cfg.items() if k != "extends"})
        return merged
    return cfg


def train_from_config(cfg_path: str | Path):
    cfg_path = Path(cfg_path)
    cfg = _load_config_with_extends(cfg_path)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    os_cfg = cfg.get("oversampling", {"enabled": True})
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]

    models_dir = Path(out_cfg["models_dir"]).resolve()
    reports_dir = Path(out_cfg["reports_dir"]).resolve()
    figures_dir = Path(out_cfg["figures_dir"]).resolve()
    _ensure_dirs([models_dir, reports_dir, figures_dir])

    # Load
    load_config = LoadConfig(
        csv_path=data_cfg["csv_path"],
        target_col=data_cfg["target_col"],
        target_mapping=data_cfg["target_mapping"],
        parse_dates=data_cfg.get("parse_dates", []),
        drop_leakage=data_cfg.get("drop_leakage", True),
        leakage_cols=data_cfg.get("leakage_cols", []),
        features=data_cfg.get("features", []),
    )

    t0 = time.time()
    df = load_and_prepare(load_config)

    # Select features for modeling; keep time cols only for time-split, not as predictors
    features = list(data_cfg.get("features", []))
    time_cols = set(data_cfg.get("parse_dates", []))

    # Always add engineered features if present
    for eng in ["credit_history_length", "income_to_loan_ratio", "fico_avg", "fico_spread"]:
        if eng in df.columns and eng not in features:
            features.append(eng)

    # Remove target and time columns from feature inputs
    feature_inputs = [
        c
        for c in features
        if c != data_cfg["target_col"] and c not in time_cols and c in df.columns
    ]

    # Split
    method = split_cfg.get("method", "random")
    if method == "time":
        time_col = split_cfg.get("time_col", "issue_d")
        train_df, test_df = time_based_split(df, time_col=time_col, test_size=split_cfg.get("test_size", 0.2))
        X_train = train_df[feature_inputs]
        y_train = train_df[data_cfg["target_col"]]
        X_test = test_df[feature_inputs]
        y_test = test_df[data_cfg["target_col"]]
    else:
        X = df[feature_inputs]
        y = df[data_cfg["target_col"]]
        X_train, X_test, y_train, y_test = random_split(
            X,
            y,
            test_size=split_cfg.get("test_size", 0.2),
            random_state=split_cfg.get("random_state", 42),
            stratify=True,
        )

    # Build and fit preprocessor on training data only
    num_cols, cat_cols = identify_feature_types(X_train)
    preproc = build_preprocessor(num_cols, cat_cols)

    X_train_proc = preproc.fit_transform(X_train)
    X_test_proc = preproc.transform(X_test)

    # Oversample only on training set to avoid leakage
    if os_cfg.get("enabled", True):
        ros = RandomOverSampler(random_state=split_cfg.get("random_state", 42))
        X_train_proc, y_train = ros.fit_resample(X_train_proc, y_train)

    # Convert to dense for Keras
    X_train_np = _to_dense(X_train_proc)
    X_test_np = _to_dense(X_test_proc)
    y_train_np = np.asarray(y_train)
    y_test_np = np.asarray(y_test)

    # Build model
    model = build_mlp(
        input_dim=X_train_np.shape[1],
        layers=model_cfg.get("layers", [256, 128, 64, 32]),
        dropout=model_cfg.get("dropout", [0.4, 0.3, 0.2, 0.2]),
        batchnorm=model_cfg.get("batchnorm", True),
        loss=("focal" if model_cfg.get("focal", {}).get("enabled", False) else model_cfg.get("loss", "binary_crossentropy")),
        optimizer=model_cfg.get("optimizer", "adam"),
        focal_alpha=model_cfg.get("focal", {}).get("alpha", 0.25),
        focal_gamma=model_cfg.get("focal", {}).get("gamma", 2.0),
    )

    # Train
    early_patience = model_cfg.get("early_stopping_patience", 3)
    callbacks = []
    if early_patience and early_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_patience, restore_best_weights=True)
        )

    history = model.fit(
        X_train_np,
        y_train_np,
        validation_split=model_cfg.get("val_split", 0.2),
        epochs=model_cfg.get("epochs", 30),
        batch_size=model_cfg.get("batch_size", 128),
        verbose=1,
        callbacks=callbacks,
    )

    # Evaluate
    y_prob = model.predict(X_test_np, verbose=0).flatten()
    metrics = compute_metrics(y_test_np, y_prob)

    # Save artifacts
    model_path = models_dir / out_cfg.get("model_filename", "loan_default_model.h5")
    model.save(model_path.as_posix())

    save_metrics(metrics, reports_dir / "metrics.json")
    plot_learning_curves(history, figures_dir / "learning_curves.png")

    elapsed = time.time() - t0
    return {
        "model_path": model_path.as_posix(),
        "metrics_path": (reports_dir / "metrics.json").as_posix(),
        "figures_path": (figures_dir / "learning_curves.png").as_posix(),
        "roc_auc": metrics.get("roc_auc"),
        "elapsed_sec": elapsed,
        "n_train": int(len(y_train_np)),
        "n_test": int(len(y_test_np)),
        "n_features": int(X_train_np.shape[1]),
    }
