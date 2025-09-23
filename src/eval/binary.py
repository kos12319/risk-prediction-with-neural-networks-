from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.eval.metrics import (
    choose_threshold_f1,
    choose_threshold_youden,
    compute_metrics_binary,
    confusion_metrics_at_threshold,
)


@dataclass
class ThresholdConfig:
    strategy: str = "fixed"
    value: float = 0.5


@dataclass
class BinaryEvaluationResult:
    threshold: float
    threshold_strategy: str
    threshold_source: str
    metrics: Dict[str, object]
    confusion: Dict[str, float]
    roc_points: Tuple[np.ndarray, np.ndarray, np.ndarray]
    pr_points: Tuple[np.ndarray, np.ndarray, np.ndarray]
    y_true: np.ndarray
    y_prob: np.ndarray
    pos_label: int


def evaluate_binary_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold_cfg: Optional[Dict[str, object]] = None,
    y_true_val: Optional[np.ndarray] = None,
    y_prob_val: Optional[np.ndarray] = None,
    pos_label: int = 1,
) -> BinaryEvaluationResult:
    """Compute metrics, confusion, and curve points for binary classification."""

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    strategy_cfg = threshold_cfg or {}
    strategy = str(strategy_cfg.get("strategy", "fixed")).lower()
    threshold_source = "validation" if y_prob_val is not None and y_true_val is not None else "test"

    if y_prob_val is not None and y_true_val is not None:
        y_true_val_arr = np.asarray(y_true_val).astype(int)
        y_prob_val_arr = np.asarray(y_prob_val).astype(float)
        threshold = _select_threshold(strategy, y_true_val_arr, y_prob_val_arr, default=strategy_cfg.get("value", 0.5))
    else:
        threshold = _select_threshold(strategy, y_true, y_prob, default=strategy_cfg.get("value", 0.5))

    metrics = compute_metrics_binary(y_true, y_prob, threshold=threshold)
    confusion = confusion_metrics_at_threshold(y_true, y_prob, threshold)

    roc_pts = roc_curve(y_true, y_prob)
    pr_pts = precision_recall_curve(y_true, y_prob)

    return BinaryEvaluationResult(
        threshold=float(threshold),
        threshold_strategy=strategy,
        threshold_source=threshold_source,
        metrics=metrics,
        confusion=confusion,
        roc_points=roc_pts,
        pr_points=pr_pts,
        y_true=y_true,
        y_prob=y_prob,
        pos_label=int(pos_label),
    )


def _select_threshold(strategy: str, y_true: np.ndarray, y_prob: np.ndarray, default: float = 0.5) -> float:
    if strategy in {"youden_j", "youden", "j"}:
        return float(choose_threshold_youden(y_true, y_prob))
    if strategy in {"f1", "f1_max", "max_f1"}:
        return float(choose_threshold_f1(y_true, y_prob))
    return float(default)
