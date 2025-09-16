from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
import json as _json

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import RandomOverSampler

from sklearn.utils import check_random_state
from sklearn.metrics import roc_curve, precision_recall_curve

from src.data.load import LoadConfig, load_and_prepare
from src.data.split import random_split, time_based_split
from src.eval.metrics import (
    compute_metrics_binary,
    plot_learning_curves,
    save_metrics,
    plot_roc_curve,
    plot_pr_curve,
    choose_threshold_youden,
    choose_threshold_f1,
    confusion_metrics_at_threshold,
)
from src.features.preprocess import build_preprocessor, identify_feature_types
"""
Note: TensorFlow is optional. Avoid importing it at module import time
to allow running with the PyTorch backend without having TF installed.
We import the Keras builder lazily only when backend=="tensorflow".
"""


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


class _SimpleHistory:
    """Adapter to mimic Keras History for plotting curves."""

    def __init__(self, loss: List[float], val_loss: List[float]):
        self.history = {"loss": loss, "val_loss": val_loss}


def _train_with_pytorch(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_test_np: np.ndarray,
    y_test_np: np.ndarray,
    model_cfg: Dict[str, Any],
    out_model_path: Path,
) -> tuple[Dict[str, Any], _SimpleHistory]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split as torch_random_split
    from src.models.torch_nn import MLP as TorchMLP, focal_binary_loss as torch_focal_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = TorchMLP(
        input_dim=X_train_np.shape[1],
        layers=model_cfg.get("layers", [256, 128, 64, 32]),
        dropout=model_cfg.get("dropout", [0.4, 0.3, 0.2, 0.2]),
        batchnorm=model_cfg.get("batchnorm", True),
    ).to(device)

    loss_name = "focal" if model_cfg.get("focal", {}).get("enabled", False) else model_cfg.get("loss", "binary_crossentropy")
    if loss_name == "focal":
        criterion = torch_focal_loss(
            gamma=float(model_cfg.get("focal", {}).get("gamma", 2.0)),
            alpha=float(model_cfg.get("focal", {}).get("alpha", 0.25)),
        )
        use_logits = True
    else:
        # BCE with logits is numerically stable
        # Use reduction='none' so we can apply class weights if provided
        bce_none = torch.nn.BCEWithLogitsLoss(reduction="none")
        criterion = bce_none
        use_logits = True

    optimizer_name = model_cfg.get("optimizer", "adam").lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = int(model_cfg.get("epochs", 30))
    batch_size = int(model_cfg.get("batch_size", 128))
    val_split = float(model_cfg.get("val_split", 0.2))
    patience = int(model_cfg.get("early_stopping_patience", 3) or 0)

    X_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)

    # Split into train/val as per val_split
    n_total = len(ds)
    n_val = int(max(1, round(n_total * val_split)))
    n_train = n_total - n_val
    train_ds, val_ds = torch_random_split(ds, [n_train, n_val]) if n_val > 0 else (ds, None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None

    best_val = float("inf")
    best_state: Optional[dict[str, Any]] = None
    wait = 0
    tr_losses: List[float] = []
    va_losses: List[float] = []

    # Optional: class weights (for BCE path). Compute auto if requested via model_cfg["_class_weight"] injected by caller.
    class_weight_cfg = model_cfg.get("_class_weight")
    use_weighted_bce = class_weight_cfg is not None and loss_name != "focal"
    if use_weighted_bce:
        # Expect dict {0: w0, 1: w1}
        w0 = float(class_weight_cfg.get(0, 1.0))
        w1 = float(class_weight_cfg.get(1, 1.0))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # Ensure target has shape (N, 1) to match logits
            yb = yb.view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if use_weighted_bce:
                loss_per = criterion(logits, yb)
                sw = yb * w1 + (1.0 - yb) * w0
                loss = (loss_per * sw).mean()
            else:
                loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        tr_losses.append(epoch_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    yb = yb.view(-1, 1)
                    logits = model(xb)
                    if use_weighted_bce:
                        loss_per = criterion(logits, yb)
                        sw = yb * w1 + (1.0 - yb) * w0
                        loss = (loss_per * sw).mean()
                    else:
                        loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
            va_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if patience and wait >= patience:
                    break
        else:
            va_losses.append(float("nan"))

    # Restore best weights if available
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    Xt = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(Xt).cpu().numpy().reshape(-1)
    y_prob = 1 / (1 + np.exp(-logits)) if use_logits else logits
    # Save model
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    import torch as _torch
    _torch.save(model.state_dict(), out_model_path.as_posix())

    history = _SimpleHistory(tr_losses, va_losses)
    return {"y_prob": y_prob}, history


def train_from_config(cfg_path: str | Path):
    cfg_path = Path(cfg_path)
    cfg = _load_config_with_extends(cfg_path)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    os_cfg = cfg.get("oversampling", {"enabled": True})
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]
    eval_cfg = cfg.get("eval", {})
    training_cfg = cfg.get("training", {})

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

    # Backend selection
    backend = str(model_cfg.get("backend", "pytorch")).lower()

    # Prepare output model filename/extension based on backend
    model_filename = out_cfg.get("model_filename", "loan_default_model.h5")
    if backend == "pytorch" and model_filename.endswith(".h5"):
        model_filename = model_filename.rsplit(".", 1)[0] + ".pt"
    elif backend == "tensorflow" and model_filename.endswith(".pt"):
        model_filename = model_filename.rsplit(".", 1)[0] + ".h5"
    model_path = models_dir / model_filename

    # Optional class weights for BCE
    class_weight_cfg = training_cfg.get("class_weight")
    cw_resolved = None
    if class_weight_cfg is not None:
        if isinstance(class_weight_cfg, str) and class_weight_cfg.lower() == "auto":
            # compute from y_train distribution
            n = float(len(y_train_np))
            n1 = float((y_train_np == 1).sum())
            n0 = n - n1
            # Inverse frequency normalized to mean 1
            w0 = n / (2.0 * max(n0, 1.0))
            w1 = n / (2.0 * max(n1, 1.0))
            cw_resolved = {0: w0, 1: w1}
        elif isinstance(class_weight_cfg, dict):
            try:
                cw_resolved = {int(k): float(v) for k, v in class_weight_cfg.items()}
            except Exception:
                cw_resolved = None
        # Inject into model_cfg for the torch path helper
        if cw_resolved is not None:
            model_cfg = dict(model_cfg)
            model_cfg["_class_weight"] = cw_resolved

    if backend == "tensorflow":
        try:
            import tensorflow as tf  # type: ignore
            from src.models.nn import build_mlp as build_mlp_tf  # lazy import
        except Exception as e:  # pragma: no cover
            raise RuntimeError("TensorFlow backend requested but not installed.") from e

        model = build_mlp_tf(
            input_dim=X_train_np.shape[1],
            layers=model_cfg.get("layers", [256, 128, 64, 32]),
            dropout=model_cfg.get("dropout", [0.4, 0.3, 0.2, 0.2]),
            batchnorm=model_cfg.get("batchnorm", True),
            loss=(
                "focal"
                if model_cfg.get("focal", {}).get("enabled", False)
                else model_cfg.get("loss", "binary_crossentropy")
            ),
            optimizer=model_cfg.get("optimizer", "adam"),
            focal_alpha=model_cfg.get("focal", {}).get("alpha", 0.25),
            focal_gamma=model_cfg.get("focal", {}).get("gamma", 2.0),
        )

        early_patience = model_cfg.get("early_stopping_patience", 3)
        callbacks = []
        if early_patience and early_patience > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=early_patience, restore_best_weights=True
                )
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

        y_prob = model.predict(X_test_np, verbose=0).flatten()
        model.save(model_path.as_posix())
        history_obj = history
    else:
        result, history_obj = _train_with_pytorch(
            X_train_np, y_train_np, X_test_np, y_test_np, model_cfg, model_path
        )
        y_prob = result["y_prob"]

    # Evaluation controls
    pos_label_cfg = eval_cfg.get("pos_label", 1)
    if isinstance(pos_label_cfg, str):
        pos_label_cfg = 0 if pos_label_cfg.lower() in {"default", "charged off", "charged_off"} else 1

    if int(pos_label_cfg) == 1:
        y_true_pos = y_test_np.astype(int)
        y_prob_pos = y_prob
        pos_label_name = "positive=1 (Fully Paid)"
    else:
        # Treat defaults as positive (label=0 in data mapping)
        y_true_pos = (1 - y_test_np).astype(int)
        y_prob_pos = 1.0 - y_prob
        pos_label_name = "positive=default (Charged Off)"

    # Threshold strategy
    thr_cfg = eval_cfg.get("threshold", {})
    strategy = str(thr_cfg.get("strategy", "fixed")).lower()
    if strategy == "youden_j":
        threshold = choose_threshold_youden(y_true_pos, y_prob_pos)
    elif strategy in {"f1", "f1_max", "max_f1"}:
        threshold = choose_threshold_f1(y_true_pos, y_prob_pos)
    else:
        threshold = float(thr_cfg.get("value", 0.5))

    # Compute metrics and plots using chosen positive class and threshold
    metrics = compute_metrics_binary(y_true_pos, y_prob_pos, threshold=threshold)

    # Save common artifacts (latest)
    save_metrics(metrics, reports_dir / "metrics.json")
    plot_learning_curves(history_obj, figures_dir / "learning_curves.png")
    cm = confusion_metrics_at_threshold(y_true_pos, y_prob_pos, threshold)
    plot_roc_curve(y_true_pos, y_prob_pos, figures_dir / "roc_curve.png", point=(cm["fpr"], cm["tpr"]))
    plot_pr_curve(y_true_pos, y_prob_pos, figures_dir / "pr_curve.png", point=(cm["precision"], cm["recall"]))

    # Save confusion metrics
    cm_path = reports_dir / "confusion.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        _json.dump(cm, f, indent=2)

    # Compute and save ROC/PR point sweeps as CSV in run folder later

    # Per-run summary file (timestamped)
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = (reports_dir / "runs" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_fig_dir = run_dir / "figures"
    run_fig_dir.mkdir(parents=True, exist_ok=True)

    # Duplicate latest artifacts into run-specific history folder
    try:
        # Figures
        for fname in ["learning_curves.png", "roc_curve.png", "pr_curve.png"]:
            src = figures_dir / fname
            if src.exists():
                shutil.copy2(src, run_fig_dir / fname)
        # Metrics + confusion
        shutil.copy2(reports_dir / "metrics.json", run_dir / "metrics.json")
        shutil.copy2(cm_path, run_dir / "confusion.json")
        # Model
        if model_path.exists():
            shutil.copy2(model_path, run_dir / model_path.name)
    except Exception:
        pass

    # Save per-threshold sweeps for ROC and PR
    try:
        fpr, tpr, thr_roc = roc_curve(y_true_pos, y_prob_pos)
        with open(run_dir / "roc_points.csv", "w", encoding="utf-8") as f:
            f.write("threshold,fpr,tpr\n")
            # First ROC point corresponds to no threshold (0,0); leave threshold blank
            for i in range(len(fpr)):
                th = "" if i == 0 else float(thr_roc[i - 1])
                f.write(f"{th},{float(fpr[i])},{float(tpr[i])}\n")
        prec, rec, thr_pr = precision_recall_curve(y_true_pos, y_prob_pos)
        with open(run_dir / "pr_points.csv", "w", encoding="utf-8") as f:
            f.write("threshold,precision,recall\n")
            # Precision-Recall pairs are length N; thresholds length N-1; align accordingly
            # Write the baseline point first with empty threshold
            if len(prec) > 0:
                f.write(f",{float(prec[0])},{float(rec[0])}\n")
            for i in range(1, len(prec)):
                th = "" if i - 1 >= len(thr_pr) else float(thr_pr[i - 1])
                f.write(f"{th},{float(prec[i])},{float(rec[i])}\n")
    except Exception:
        pass

    # Save resolved config used for this run
    resolved_cfg_path = run_dir / "config_resolved.yaml"
    try:
        with open(resolved_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception:
        pass

    # Save feature lists used by preprocessor
    try:
        features_manifest = {
            "numerical_features": list(num_cols),
            "categorical_features": list(cat_cols),
            "feature_inputs": list(feature_inputs),
        }
        with open(run_dir / "features.json", "w", encoding="utf-8") as f:
            _json.dump(features_manifest, f, indent=2)
    except Exception:
        pass

    # Save history CSV
    try:
        import csv as _csv
        hist_csv = run_dir / "history.csv"
        with open(hist_csv, "w", newline="", encoding="utf-8") as hf:
            writer = _csv.writer(hf)
            writer.writerow(["epoch", "loss", "val_loss"])
            tr = history_obj.history.get("loss", [])
            va = history_obj.history.get("val_loss", [])
            for i in range(max(len(tr), len(va))):
                l = tr[i] if i < len(tr) else ""
                vl = va[i] if i < len(va) else ""
                writer.writerow([i + 1, l, vl])
    except Exception:
        pass
    summary_lines = [
        f"# Training Summary — {run_id}",
        "",
        f"Config: `{cfg_path}`",
        f"Backend: {backend}",
        f"Positive class: {pos_label_name}",
        f"Threshold strategy: {strategy}",
        f"Chosen threshold: {threshold:.6f}",
        "",
        "## Metrics",
        f"- ROC AUC: {metrics.get('roc_auc'):.3f}",
        f"- Average Precision: {metrics.get('average_precision'):.3f}",
        f"- Precision (at threshold): {cm['precision']:.3f}",
        f"- Recall (TPR): {cm['recall']:.3f}",
        f"- Specificity (TNR): {1.0 - cm['fpr']:.3f}",
        f"- Confusion: TP={int(cm['tp'])}, FP={int(cm['fp'])}, TN={int(cm['tn'])}, FN={int(cm['fn'])}",
        f"- n_train: {int(len(y_train_np))}",
        f"- n_test: {int(len(y_test_np))}",
        f"- n_features: {int(X_train_np.shape[1])}",
        "",
        "## Classification Report (at threshold)",
        "```json",
        _json.dumps(metrics.get("classification_report", {}), indent=2),
        "```",
        "",
        "## Artifacts",
        f"- Model: `{(run_dir / model_path.name).as_posix()}`",
        f"- Metrics: `{(run_dir / 'metrics.json').as_posix()}`",
        f"- Confusion: `{(run_dir / 'confusion.json').as_posix()}`",
        f"- History CSV: `{(run_dir / 'history.csv').as_posix()}`",
        f"- ROC points CSV: `{(run_dir / 'roc_points.csv').as_posix()}`",
        f"- PR points CSV: `{(run_dir / 'pr_points.csv').as_posix()}`",
        f"- Learning curves: `{(run_fig_dir / 'learning_curves.png').as_posix()}`",
        f"- ROC curve: `{(run_fig_dir / 'roc_curve.png').as_posix()}`",
        f"- PR curve: `{(run_fig_dir / 'pr_curve.png').as_posix()}`",
        f"- Resolved config: `{(run_dir / 'config_resolved.yaml').as_posix()}`",
        f"- Features manifest: `{(run_dir / 'features.json').as_posix()}`",
        "",
        "## Notes",
        "- Evaluated defaults as the positive class.",
        "- Threshold selected according to configured strategy and annotated on curves.",
    ]
    with open(run_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    elapsed = time.time() - t0
    return {
        "model_path": model_path.as_posix(),
        "metrics_path": (reports_dir / "metrics.json").as_posix(),
        "figures_path": (figures_dir / "learning_curves.png").as_posix(),
        "roc_curve_path": (figures_dir / "roc_curve.png").as_posix(),
        "pr_curve_path": (figures_dir / "pr_curve.png").as_posix(),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "threshold": metrics.get("threshold"),
        "pos_label": pos_label_name,
        "elapsed_sec": elapsed,
        "n_train": int(len(y_train_np)),
        "n_test": int(len(y_test_np)),
        "n_features": int(X_train_np.shape[1]),
        "run_summary_path": (run_dir / "README.md").as_posix(),
    }
