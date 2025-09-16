from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def compute_metrics(y_true, y_prob) -> Dict:
    y_pred = (np.asarray(y_prob) > 0.5).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    auc = float(roc_auc_score(y_true, y_prob))
    return {"classification_report": report, "roc_auc": auc}


def save_metrics(metrics: Dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_learning_curves(history, out_path: str | Path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    if "loss" in history.history:
        plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(
    y_true, y_prob, out_path: str | Path, point: Optional[Tuple[float, float]] = None
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    if point is not None:
        plt.scatter([point[0]], [point[1]], c="red", label="Chosen threshold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr_curve(
    y_true, y_prob, out_path: str | Path, point: Optional[Tuple[float, float]] = None
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    if point is not None:
        plt.scatter([point[1]], [point[0]], c="red", label="Chosen threshold")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_metrics_binary(y_true, y_prob, threshold: float = 0.5) -> Dict:
    """
    Compute classification report at a threshold, plus ROC AUC and Average Precision.
    Expects y_true coded as 0/1 with 1 as the positive class, and y_prob as P(y=1).
    """
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    auc = float(roc_auc_score(y_true, y_prob))
    ap = float(average_precision_score(y_true, y_prob))
    return {"classification_report": report, "roc_auc": auc, "average_precision": ap, "threshold": float(threshold)}


def choose_threshold_youden(y_true, y_prob) -> float:
    """Return threshold that maximizes Youden's J = TPR - FPR."""
    fpr, tpr, thresh = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(j.argmax())
    return float(thresh[idx])


def choose_threshold_f1(y_true, y_prob) -> float:
    """Return threshold that maximizes F1 (computed from the PR curve)."""
    precision, recall, thresh = precision_recall_curve(y_true, y_prob)
    # thresholds align with precision[1:], recall[1:]
    if len(thresh) == 0:
        return 0.5
    p = precision[1:]
    r = recall[1:]
    f1 = (2 * p * r) / (p + r + 1e-12)
    idx = int(f1.argmax())
    return float(thresh[idx])


def confusion_metrics_at_threshold(y_true, y_prob, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tpr
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
    }
