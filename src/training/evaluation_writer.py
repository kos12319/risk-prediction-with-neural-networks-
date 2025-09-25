from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from src.eval.binary import BinaryEvaluationResult
from src.eval.metrics import (
    plot_learning_curves,
    plot_pr_curve,
    plot_roc_curve,
    save_metrics,
    confusion_metrics_at_threshold,
)


def _write_curves_csv(
    *,
    evaluation: BinaryEvaluationResult,
    out_dir: Path,
) -> None:
    try:
        fpr, tpr, thr_roc = evaluation.roc_points
        with open(out_dir / "roc_points.csv", "w", encoding="utf-8") as f:
            f.write("threshold,fpr,tpr\n")
            for idx in range(len(fpr)):
                threshold_val = "" if idx == 0 else float(thr_roc[idx - 1])
                f.write(f"{threshold_val},{float(fpr[idx])},{float(tpr[idx])}\n")
    except Exception:
        pass

    try:
        precision, recall, thr_pr = evaluation.pr_points
        with open(out_dir / "pr_points.csv", "w", encoding="utf-8") as f:
            f.write("threshold,precision,recall\n")
            if len(precision) > 0:
                f.write(f",{float(precision[0])},{float(recall[0])}\n")
            for idx in range(1, len(precision)):
                threshold_val = "" if idx - 1 >= len(thr_pr) else float(thr_pr[idx - 1])
                f.write(f"{threshold_val},{float(precision[idx])},{float(recall[idx])}\n")
    except Exception:
        pass


def _write_threshold_grid(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
    n: int = 101,
) -> None:
    try:
        thr_grid = np.linspace(0.0, 1.0, int(n))
        with open(out_dir / "threshold_metrics.csv", "w", encoding="utf-8") as f:
            f.write("threshold,precision,recall,tpr,fpr,specificity,f1\n")
            y_true_eval = np.asarray(y_true).astype(int)
            y_prob_eval = np.asarray(y_prob)
            for th in thr_grid:
                metrics_at_th = confusion_metrics_at_threshold(y_true_eval, y_prob_eval, float(th))
                prec_v = float(metrics_at_th.get("precision", 0.0))
                rec_v = float(metrics_at_th.get("recall", 0.0))
                tpr_v = float(metrics_at_th.get("tpr", 0.0))
                fpr_v = float(metrics_at_th.get("fpr", 0.0))
                spec_v = 1.0 - fpr_v
                f1_v = (2 * prec_v * rec_v) / (prec_v + rec_v + 1e-12)
                f.write(f"{th:.4f},{prec_v:.6f},{rec_v:.6f},{tpr_v:.6f},{fpr_v:.6f},{spec_v:.6f},{f1_v:.6f}\n")
    except Exception:
        pass


def write_basic_eval_artifacts(
    *,
    evaluation: BinaryEvaluationResult,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    reports_dir: Path,
    figures_dir: Path,
    history: Any | None = None,
    point: Optional[Tuple[float, float]] = None,
) -> None:
    """Write the minimal set of artifacts used in CV folds.

    Produces:
    - metrics.json, confusion.json
    - ROC/PR plots (no point markers if not provided)
    - learning_curves.png (best-effort)
    """
    save_metrics(evaluation.metrics, reports_dir / "metrics.json")
    with open(reports_dir / "confusion.json", "w", encoding="utf-8") as f:
        _json.dump(evaluation.confusion, f, indent=2)

    try:
        plot_roc_curve(y_true=y_true, y_prob=y_prob, out_path=figures_dir / "roc_curve.png", point=point)
    except Exception:
        pass
    try:
        plot_pr_curve(y_true=y_true, y_prob=y_prob, out_path=figures_dir / "pr_curve.png", point=None if point is None else (evaluation.confusion.get("precision"), evaluation.confusion.get("recall")))
    except Exception:
        pass
    try:
        plot_learning_curves(history, figures_dir / "learning_curves.png")
    except Exception:
        pass


def write_full_eval_artifacts(
    *,
    evaluation: BinaryEvaluationResult,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    reports_dir: Path,
    figures_dir: Path,
    history: Any | None = None,
    point: Optional[Tuple[float, float]] = None,
) -> None:
    """Write the full set of artifacts used in single runs.

    Superset of basic artifacts; also writes ROC/PR CSVs and the threshold grid CSV.
    """
    write_basic_eval_artifacts(
        evaluation=evaluation,
        y_true=y_true,
        y_prob=y_prob,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        history=history,
        point=point,
    )
    _write_curves_csv(evaluation=evaluation, out_dir=reports_dir)
    _write_threshold_grid(y_true=y_true, y_prob=y_prob, out_dir=reports_dir)


__all__ = [
    "write_basic_eval_artifacts",
    "write_full_eval_artifacts",
]

