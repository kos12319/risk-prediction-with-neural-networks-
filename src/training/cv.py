from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.split import TemporalFold, time_based_kfold_splits
from src.eval.binary import evaluate_binary_classification
from src.eval.metrics import save_metrics
from src.training.evaluation_writer import write_basic_eval_artifacts
from src.features.preprocess import preprocess_tabular_data
from src.utils.artifacts import ArtifactManager
from src.training.interfaces import BackendPipeline, DatasetBundle, RunContext, TrainingRunResult
from src.training.probability import align_probabilities
from src.training.resample import apply_resampling
from src.utils.seed import set_seed

import copy
import json as _json
import logging
import time


logger = logging.getLogger(__name__)


def run_cv_fold(
    *,
    fold: TemporalFold,
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    feature_inputs: Sequence[str],
    data_cfg: Dict[str, Any],
    split_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    os_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    tracking_cfg: Dict[str, Any],
    out_cfg: Dict[str, Any],
    artifact_mgr: ArtifactManager,
    random_state: int,
    notes: Optional[str],
    run_id: str,
    backend: BackendPipeline,
    run_context: RunContext,
) -> TrainingRunResult:
    start_time = time.time()
    split_result = fold.split

    winsor_cfg = data_cfg.get("winsorize") if data_cfg.get("winsorize_enabled", True) else None
    preprocessing_cfg = cfg.get("preprocessing", {})
    preproc_result = preprocess_tabular_data(
        split_result,
        winsorize_cfg=winsor_cfg,
        preprocessing_cfg=preprocessing_cfg,
    )
    logger.info(
        "[CV fold %s] Preprocessing complete | train=%s | val=%s | test=%s",
        fold.fold_id,
        preproc_result.X_train.shape,
        None if preproc_result.X_val is None else preproc_result.X_val.shape,
        preproc_result.X_test.shape,
    )

    resample_result = None
    if os_cfg.get("enabled", True):
        resample_result = apply_resampling(
            preproc_result.X_train,
            preproc_result.y_train,
            method=os_cfg.get("method"),
            random_state=random_state,
            params=os_cfg.get("params"),
        )
        X_train_np = resample_result.X_resampled
        y_train_np = resample_result.y_resampled
        logger.info("[CV fold %s] Using resampled training data %s", fold.fold_id, X_train_np.shape)
    else:
        X_train_np = preproc_result.X_train
        y_train_np = preproc_result.y_train
        logger.info("[CV fold %s] Resampling disabled; train shape %s", fold.fold_id, X_train_np.shape)

    X_val_np = preproc_result.X_val
    y_val_np = preproc_result.y_val
    X_test_np = preproc_result.X_test
    y_test_np = preproc_result.y_test

    feature_names = preproc_result.feature_names

    dataset = DatasetBundle(
        X_train=X_train_np,
        y_train=y_train_np,
        X_val=X_val_np,
        y_val=y_val_np,
        X_test=X_test_np,
        y_test=y_test_np,
        feature_names=feature_names,
    )

    model_cfg_local = backend.prepare_model_config(
        model_cfg=copy.deepcopy(model_cfg),
        training_cfg=training_cfg,
        y_train=y_train_np,
    )

    fold_meta = {"fold_id": fold.fold_id, **(fold.metadata or {})}
    model_path = backend.resolve_model_path(
        out_cfg=out_cfg,
        artifact_mgr=artifact_mgr,
        fold_meta=fold_meta,
    )

    fold_seed = int(random_state + fold.fold_id)
    set_seed(fold_seed)
    logger.info("[CV fold %s] Random seed set to %d", fold.fold_id, fold_seed)

    pos_label_cfg = eval_cfg.get("pos_label", 1)
    if isinstance(pos_label_cfg, str):
        pos_label_cfg = 0 if str(pos_label_cfg).lower() in {"default", "charged off", "charged_off"} else 1
    pos_label_int = int(pos_label_cfg)

    train_start = time.time()
    training_result = backend.train(
        dataset=dataset,
        model_cfg=model_cfg_local,
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        run_context=run_context,
        model_path=model_path,
        random_seed=fold_seed,
        pos_label=pos_label_int,
        fold_meta=fold_meta,
        wandb_run=None,
        wandb_enabled=False,
        cfg=cfg,
    )
    train_end = time.time()
    model_path = training_result.model_path
    history_obj = training_result.history
    raw_result = training_result.raw or {}

    # Align to configured positive class (probability and labels)
    y_true_pos_test = (y_test_np.astype(int) == pos_label_int).astype(int)
    y_prob_pos_test = align_probabilities(training_result.y_prob, training_result.prob_label, pos_label_int)

    eval_start = time.time()
    eval_result = evaluate_binary_classification(
        y_true=y_true_pos_test,
        y_prob=y_prob_pos_test,
        threshold_cfg=(eval_cfg.get("threshold", {}) or {}),
        pos_label=pos_label_int,
    )
    eval_end = time.time()

    # Save minimal artifacts under fold dir via centralized writer
    try:
        write_basic_eval_artifacts(
            evaluation=eval_result,
            y_true=y_test_np,
            y_prob=y_prob_aligned,
            reports_dir=artifact_mgr.reports_dir,
            figures_dir=artifact_mgr.figures_dir,
            history=history_obj,
            point=None,
        )
    except Exception:
        # Fallback to direct JSON writes if central writer import fails
        save_metrics(eval_result.metrics, artifact_mgr.reports_dir / "metrics.json")
        try:
            with open(artifact_mgr.reports_dir / "confusion.json", "w", encoding="utf-8") as f:
                _json.dump(eval_result.confusion, f, indent=2)
        except Exception:
            pass
    durations = {
        "preprocess": float(split_result.metadata.get("preprocess_sec", 0.0)) if hasattr(split_result, "metadata") else 0.0,
        "train": float(train_end - train_start),
        "eval": float(eval_end - eval_start),
        "total": float(time.time() - start_time),
    }

    fold_meta_out = {
        **fold_meta,
        "n_test": int(len(y_test_np)),
    }

    return TrainingRunResult(
        run_id=run_id,
        evaluation=eval_result,
        metrics=eval_result.metrics,
        confusion=eval_result.confusion,
        model_path=Path(model_path),
        durations=durations,
        fold_meta=fold_meta_out,
    )


def run_temporal_cv(
    *,
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    feature_inputs: Sequence[str],
    data_cfg: Dict[str, Any],
    split_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    os_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    tracking_cfg: Dict[str, Any],
    out_cfg: Dict[str, Any],
    artifact_mgr: ArtifactManager,
    run_id: str,
    notes: Optional[str],
    backend: BackendPipeline,
):
    cv_cfg = split_cfg.get("cv", {}) or {}
    n_folds = int(cv_cfg.get("n_folds", 0))
    if n_folds < 2:
        raise ValueError("Temporal CV requires 'n_folds' >= 2 when enabled.")

    time_col = split_cfg.get("time_col", "issue_d")
    folds = time_based_kfold_splits(
        df,
        feature_inputs,
        data_cfg["target_col"],
        time_col=time_col,
        n_folds=n_folds,
        initial_train_fraction=float(cv_cfg.get("initial_train_fraction", 0.4)),
        validation_fraction=float(cv_cfg.get("validation_fraction", model_cfg.get("val_split", 0.2))),
        gap=int(cv_cfg.get("gap", 0)),
        mode=str(cv_cfg.get("mode", "expanding")),
        shuffle_within_folds=bool(cv_cfg.get("shuffle_within_folds", False)),
        random_state=int(split_cfg.get("random_state", 42)),
    )

    folds_root = artifact_mgr.run_dir / "folds"
    fold_models_root = artifact_mgr.models_dir / "folds"
    fold_reports_root = artifact_mgr.reports_dir / "folds"
    fold_figures_root = artifact_mgr.figures_dir / "folds"

    results: List[TrainingRunResult] = []
    fold_records: List[Dict[str, Any]] = []
    random_state = int(split_cfg.get("random_state", 42))

    for fold in folds:
        fold_name = f"fold_{fold.fold_id:02d}"
        fold_mgr = ArtifactManager(
            run_dir=folds_root / fold_name,
            models_dir=fold_models_root / fold_name,
            reports_dir=fold_reports_root / fold_name,
            figures_dir=fold_figures_root / fold_name,
            single_run_mode=True,
        )
        fold_run_id = f"{run_id}_{fold_name}"
        fold_context = RunContext(run_id=fold_run_id, run_dir=fold_mgr.run_dir, artifact_mgr=fold_mgr)
        fold_result = run_cv_fold(
            fold=fold,
            cfg=cfg,
            df=df,
            feature_inputs=feature_inputs,
            data_cfg=data_cfg,
            split_cfg=split_cfg,
            model_cfg=model_cfg,
            os_cfg=os_cfg,
            eval_cfg=eval_cfg,
            training_cfg=training_cfg,
            tracking_cfg=tracking_cfg,
            out_cfg=out_cfg,
            artifact_mgr=fold_mgr,
            random_state=random_state,
            notes=notes,
            run_id=fold_run_id,
            backend=backend,
            run_context=fold_context,
        )
        results.append(fold_result)
        fold_record = {
            "fold_id": fold.fold_id,
            "run_id": fold_run_id,
            "metrics": fold_result.metrics,
            "confusion": fold_result.confusion,
            "threshold": fold_result.evaluation.threshold,
            "durations": fold_result.durations,
            "train_range": fold.train_range,
            "val_range": fold.val_range,
            "test_range": fold.test_range,
            "metadata": fold_result.fold_meta,
        }
        fold_records.append(fold_record)

    roc_aucs = [float(r.metrics.get("roc_auc", float("nan"))) for r in results]
    ap_scores = [float(r.metrics.get("average_precision", float("nan"))) for r in results]
    thresholds = [float(r.evaluation.threshold) for r in results]
    confusion_sums = {k: 0.0 for k in ("tp", "tn", "fp", "fn")}
    total_test = 0
    for res in results:
        meta = res.fold_meta or {}
        n_test = int(meta.get("n_test", 0))
        total_test += n_test
        for key in confusion_sums:
            confusion_sums[key] += float(res.confusion.get(key, 0.0))

    aggregate = {
        "n_folds": len(results),
        "roc_auc_mean": float(np.nanmean(roc_aucs)) if roc_aucs else None,
        "roc_auc_std": float(np.nanstd(roc_aucs, ddof=1)) if len(roc_aucs) > 1 else 0.0,
        "average_precision_mean": float(np.nanmean(ap_scores)) if ap_scores else None,
        "average_precision_std": float(np.nanstd(ap_scores, ddof=1)) if len(ap_scores) > 1 else 0.0,
        "threshold_mean": float(np.nanmean(thresholds)) if thresholds else None,
        "threshold_std": float(np.nanstd(thresholds, ddof=1)) if len(thresholds) > 1 else 0.0,
        "total_test_rows": total_test,
        "confusion_sum": {k: float(v) for k, v in confusion_sums.items()},
    }

    cv_report = {
        "folds": fold_records,
        "aggregate": aggregate,
    }

    report_path = artifact_mgr.reports_dir / "cv_metrics.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            _json.dump(cv_report, f, indent=2)
    except Exception:
        logger.exception("Failed to write cv_metrics.json")

    # Write a lightweight summary README under the CV root
    try:
        summary_lines = [
            f"# Temporal Cross-Validation Summary — {run_id}",
            "",
            f"Folds: {aggregate['n_folds']}",
        ]
        if aggregate.get("roc_auc_mean") is not None:
            summary_lines.append(
                f"ROC AUC (mean±std): {aggregate['roc_auc_mean']:.3f} ± {aggregate['roc_auc_std']:.3f}"
            )
        if aggregate.get("average_precision_mean") is not None:
            summary_lines.append(
                f"Average Precision (mean±std): {aggregate['average_precision_mean']:.3f} ± {aggregate['average_precision_std']:.3f}"
            )
        if aggregate.get("threshold_mean") is not None:
            summary_lines.append(
                f"Threshold (mean±std): {aggregate['threshold_mean']:.4f} ± {aggregate['threshold_std']:.4f}"
            )
        summary_lines.extend(["", "## Fold Metrics"])
        for rec in fold_records:
            summary_lines.append(
                f"- Fold {rec['fold_id']:02d}: ROC AUC={rec['metrics'].get('roc_auc'):.3f}, AP={rec['metrics'].get('average_precision'):.3f}, Threshold={rec['threshold']:.4f}"
            )
        if notes:
            summary_lines.extend(["", "## Notes", notes.strip()])
        summary_path = artifact_mgr.run_dir / "README.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
    except Exception:
        summary_path = artifact_mgr.run_dir / "README.md"
        try:
            summary_path.touch(exist_ok=True)
        except Exception:
            pass

    return {
        "results": results,
        "fold_records": fold_records,
        "aggregate": aggregate,
        "report_path": report_path,
        "summary_path": summary_path,
    }


__all__ = ["run_cv_fold", "run_temporal_cv"]
