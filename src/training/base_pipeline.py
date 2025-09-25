from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from abc import ABC, abstractmethod
import sys
import json as _json
import hashlib
import os
import copy

# Apply safe env as early as possible to avoid BLAS/Accelerate crashes on import
try:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_PROC_BIND", "FALSE")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", ".cache")
    os.environ.setdefault("MPLCONFIGDIR", ".mplcache")
    import platform as _platform  # local import to avoid global dependency
    if _platform.system() == "Darwin" and _platform.machine() in {"arm64", "aarch64"}:
        os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
except Exception:
    pass

import numpy as np
import pandas as pd
import yaml

from src.data.load import LoadConfig, load_and_prepare
from src.data.split import SplitResult, TemporalFold, train_val_test_split, time_based_kfold_splits
from src.eval.binary import BinaryEvaluationResult, evaluate_binary_classification
from src.eval.metrics import (
    plot_learning_curves,
    plot_pr_curve,
    plot_roc_curve,
    save_metrics,
)
from src.features.preprocess import (
    PreprocessResult,
    preprocess_tabular_data,
    resolve_feature_inputs,
)
from src.training.config import load_config_with_extends
from src.training.resample import apply_resampling
from src.utils.artifacts import ArtifactManager
from src.utils.seed import set_seed
import platform


logger = logging.getLogger(__name__)


@dataclass
class TrainingRunResult:
    """Bundle summary for a single training/evaluation run (fold or holdout)."""

    run_id: str
    evaluation: BinaryEvaluationResult
    metrics: Dict[str, Any]
    confusion: Dict[str, Any]
    model_path: Path
    durations: Dict[str, float]
    fold_meta: Optional[Dict[str, Any]] = None


@dataclass
class DatasetBundle:
    """Preprocessed arrays ready for backend training."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: Sequence[str]


@dataclass
class BackendTrainingResult:
    """Standardized payload returned by backend pipelines."""

    y_prob: np.ndarray
    prob_label: int
    model_path: Path
    history: Any
    raw: Dict[str, Any]


@dataclass
class RunContext:
    """Lightweight context shared with backend pipelines."""

    run_id: str
    run_dir: Path
    artifact_mgr: ArtifactManager


class BackendPipeline(ABC):
    """Abstract orchestration contract implemented by each backend."""

    name: str = "backend"

    @abstractmethod
    def validate_config(self, cfg: Dict[str, Any]) -> None:
        """Validate backend-specific portions of the config before running."""

    def apply_env_overrides(self, cfg: Dict[str, Any]) -> None:
        """Apply backend-specific environment overrides (optional)."""
        return None

    @abstractmethod
    def resolve_model_path(
        self,
        *,
        out_cfg: Dict[str, Any],
        artifact_mgr: ArtifactManager,
        fold_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Return the destination path for the trained model artifact."""

    def prepare_model_config(
        self,
        *,
        model_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """Optionally adjust the model config before training."""
        return dict(model_cfg)

    @abstractmethod
    def train(
        self,
        *,
        dataset: DatasetBundle,
        model_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
        eval_cfg: Dict[str, Any],
        run_context: RunContext,
        model_path: Path,
        random_seed: int,
        pos_label: int,
        fold_meta: Optional[Dict[str, Any]] = None,
        wandb_run: Any | None = None,
        wandb_enabled: bool = False,
        cfg: Dict[str, Any] | None = None,
    ) -> BackendTrainingResult:
        """Backend-specific training implementation."""

    def log_wandb(
        self,
        *,
        wandb_run: Any,
        dataset: DatasetBundle,
        model_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
        eval_cfg: Dict[str, Any],
        training_result: BackendTrainingResult,
        metrics: Dict[str, Any],
        confusion: Dict[str, Any],
        run_context: RunContext,
        notes: Optional[str],
    ) -> None:
        """Optional extra W&B logging beyond the shared summary."""

    def extra_artifact_lines(
        self,
        *,
        training_result: BackendTrainingResult,
        run_context: RunContext,
        cfg: Dict[str, Any],
    ) -> Iterable[str]:
        """Additional lines to include in the README artifact."""
        return []

    def format_run_name(
        self,
        *,
        base_context: Dict[str, Any],
        training_result: BackendTrainingResult,
        metrics: Dict[str, Any],
        run_context: RunContext,
        cfg: Dict[str, Any],
    ) -> Optional[str]:
        """Optional override for the W&B run name format."""
        return None

    def additional_wandb_tags(
        self,
        *,
        training_result: BackendTrainingResult,
        run_context: RunContext,
        cfg: Dict[str, Any],
    ) -> Iterable[str]:
        """Optional additional W&B tags for this backend."""
        return []

    def run(self, cfg_path: str | Path, *, notes: Optional[str] = None):
        """Execute the shared pipeline using this backend implementation."""
        return _run_backend_pipeline(cfg_path, backend=self, notes=notes)

def _collect_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info["machine"] = platform.machine()
        info["processor"] = platform.processor()
        info["platform"] = platform.platform()
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    # RAM (best-effort)
    try:
        import psutil  # type: ignore
        info["ram_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        pass
    # Torch device info
    try:
        import torch as _torch
        info["has_cuda"] = bool(_torch.cuda.is_available())
        info["cuda_version"] = getattr(_torch.version, "cuda", None)
        if _torch.cuda.is_available():
            try:
                info["gpu_name"] = _torch.cuda.get_device_name(0)
            except Exception:
                pass
        mps_backend = getattr(_torch.backends, "mps", None)
        if mps_backend is not None:
            try:
                info["has_mps"] = bool(mps_backend.is_available())
            except Exception:
                info["has_mps"] = None
            try:
                info["mps_is_built"] = bool(mps_backend.is_built())
            except Exception:
                pass
            ane_fn = getattr(mps_backend, "is_neural_engine_available", None)
            if callable(ane_fn):
                try:
                    info["has_ane"] = bool(ane_fn())
                except Exception:
                    info["has_ane"] = None
        xpu_backend = getattr(_torch, "xpu", None)
        if xpu_backend is not None:
            try:
                info["has_xpu"] = bool(xpu_backend.is_available())
            except Exception:
                info["has_xpu"] = None
    except Exception:
        pass
    # Threads
    for envk in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        v = os.environ.get(envk)
        if v is not None:
            info[envk.lower()] = v
    return info


def _collect_env_metadata() -> Dict[str, Any]:
    """Collect lightweight environment and git metadata for logging."""
    info: Dict[str, Any] = {"env": {}, "git": {}}
    # Python
    try:
        import sys
        info["env"]["python"] = sys.version.split(" ")[0]
    except Exception:
        pass
    # Library versions
    try:
        import numpy as _np
        info["env"]["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import pandas as _pd
        info["env"]["pandas"] = _pd.__version__
    except Exception:
        pass
    try:
        import sklearn as _sk
        info["env"]["scikit_learn"] = _sk.__version__
    except Exception:
        pass
    try:
        import imblearn as _im
        info["env"]["imbalanced_learn"] = _im.__version__
    except Exception:
        pass
    try:
        import matplotlib as _mpl
        info["env"]["matplotlib"] = _mpl.__version__
    except Exception:
        pass
    try:
        import torch as _torch
        info["env"]["torch"] = _torch.__version__
    except Exception:
        pass
    try:
        import yaml as _yaml
        ver = getattr(_yaml, "__version__", None)
        if ver:
            info["env"]["PyYAML"] = ver
    except Exception:
        pass
    try:
        import wandb as _wandb
        info["env"]["wandb"] = _wandb.__version__
    except Exception:
        pass

    # Git metadata (best-effort)
    try:
        import subprocess
        # commit hash (short)
        sha = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        info["git"]["commit"] = sha
        # dirty flag
        dirty = True
        try:
            subprocess.check_call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            subprocess.check_call(["git", "diff", "--quiet", "--cached"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            dirty = False
        except Exception:
            dirty = True
        info["git"]["dirty"] = dirty
        # branch (optional)
        try:
            br = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            info["git"]["branch"] = br
        except Exception:
            pass
        # remote URL (origin)
        try:
            remote = subprocess.check_output(["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL).decode().strip()
            info["git"]["remote"] = remote
            # Normalize a clickable commit URL for GitHub-style remotes
            try:
                commit_url: Optional[str] = None
                if remote.startswith("git@github.com:"):
                    path = remote.split(":", 1)[1]
                    if path.endswith(".git"):
                        path = path[:-4]
                    commit_url = f"https://github.com/{path}/commit/{sha}"
                elif remote.startswith("https://github.com/"):
                    path = remote.split("https://github.com/", 1)[1]
                    if path.endswith(".git"):
                        path = path[:-4]
                    commit_url = f"https://github.com/{path}/commit/{sha}"
                if commit_url:
                    info["git"]["commit_url"] = commit_url
                # Also a normalized HTTPS repo URL if GitHub
                if remote.startswith("git@github.com:"):
                    path = remote.split(":", 1)[1]
                    if path.endswith(".git"):
                        path = path[:-4]
                    info["git"]["repo_url"] = f"https://github.com/{path}"
                elif remote.startswith("https://github.com/"):
                    repo_https = remote[:-4] if remote.endswith(".git") else remote
                    info["git"]["repo_url"] = repo_https
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass
    return info


def _run_cv_fold(
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
    """Train and evaluate a single temporal CV fold."""

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

    y_prob = np.asarray(training_result.y_prob)
    prob_label_raw = int(training_result.prob_label)

    y_true_pos_test = (y_test_np.astype(int) == pos_label_int).astype(int)
    y_prob_pos_test = _align_probabilities(y_prob, prob_label_raw, pos_label_int)

    y_true_pos_val = None
    y_prob_pos_val = None
    if y_val_np is not None and len(y_val_np) > 0:
        y_true_pos_val = (y_val_np.astype(int) == pos_label_int).astype(int)
        val_buf = None
        if isinstance(raw_result, dict):
            val_buf = raw_result.get("y_prob_val")
        if val_buf is not None:
            y_prob_pos_val = _align_probabilities(val_buf, prob_label_raw, pos_label_int)

    evaluation = evaluate_binary_classification(
        y_true=y_true_pos_test,
        y_prob=y_prob_pos_test,
        threshold_cfg=eval_cfg.get("threshold", {}),
        y_true_val=y_true_pos_val,
        y_prob_val=y_prob_pos_val,
        pos_label=pos_label_int,
    )
    metrics = evaluation.metrics
    confusion = evaluation.confusion

    figures_dir = artifact_mgr.figures_dir
    run_dir = artifact_mgr.run_dir
    save_metrics(metrics, artifact_mgr.metrics_path)
    artifact_mgr.save_confusion(confusion)
    plot_learning_curves(history_obj, figures_dir / "learning_curves.png")
    plot_roc_curve(evaluation.y_true, evaluation.y_prob, figures_dir / "roc_curve.png", point=(confusion["fpr"], confusion["tpr"]))
    plot_pr_curve(evaluation.y_true, evaluation.y_prob, figures_dir / "pr_curve.png", point=(confusion["precision"], confusion["recall"]))

    try:
        fpr, tpr, thr_roc = evaluation.roc_points
        with open(run_dir / "roc_points.csv", "w", encoding="utf-8") as f:
            f.write("threshold,fpr,tpr\n")
            for idx in range(len(fpr)):
                threshold_val = "" if idx == 0 else float(thr_roc[idx - 1])
                f.write(f"{threshold_val},{float(fpr[idx])},{float(tpr[idx])}\n")
        precision, recall, thr_pr = evaluation.pr_points
        with open(run_dir / "pr_points.csv", "w", encoding="utf-8") as f:
            f.write("threshold,precision,recall\n")
            if len(precision) > 0:
                f.write(f",{float(precision[0])},{float(recall[0])}\n")
            for idx in range(1, len(precision)):
                threshold_val = "" if idx - 1 >= len(thr_pr) else float(thr_pr[idx - 1])
                f.write(f"{threshold_val},{float(precision[idx])},{float(recall[idx])}\n")
        import numpy as _np
        from src.eval.metrics import confusion_metrics_at_threshold as _cm_thr

        thr_grid = _np.linspace(0.0, 1.0, 101)
        with open(run_dir / "threshold_metrics.csv", "w", encoding="utf-8") as f:
            f.write("threshold,precision,recall,tpr,fpr,specificity,f1\n")
            y_true_eval = _np.asarray(evaluation.y_true).astype(int)
            y_prob_eval = _np.asarray(evaluation.y_prob)
            for th in thr_grid:
                metrics_at_th = _cm_thr(y_true_eval, y_prob_eval, float(th))
                prec_v = float(metrics_at_th.get("precision", 0.0))
                rec_v = float(metrics_at_th.get("recall", 0.0))
                tpr_v = float(metrics_at_th.get("tpr", 0.0))
                fpr_v = float(metrics_at_th.get("fpr", 0.0))
                spec_v = 1.0 - fpr_v
                f1_v = (2 * prec_v * rec_v) / (prec_v + rec_v + 1e-12)
                f.write(f"{th:.4f},{prec_v:.6f},{rec_v:.6f},{tpr_v:.6f},{fpr_v:.6f},{spec_v:.6f},{f1_v:.6f}\n")
    except Exception:
        pass

    artifact_mgr.stage_run_artifacts(
        model_path,
        figure_names=["learning_curves.png", "roc_curve.png", "pr_curve.png"],
    )

    try:
        features_manifest = {
            "numerical_features": list(preproc_result.numerical_features),
            "categorical_features": list(preproc_result.categorical_features),
            "feature_inputs": list(feature_inputs),
            "encoded_feature_names": list(feature_names),
        }
        with open(run_dir / "features.json", "w", encoding="utf-8") as f:
            _json.dump(features_manifest, f, indent=2)
    except Exception:
        pass

    try:
        win = {
            "fold": fold.fold_id,
            "train_range": fold.train_range,
            "val_range": fold.val_range,
            "test_range": fold.test_range,
            "meta": fold.metadata,
        }
        with open(run_dir / "fold_metadata.json", "w", encoding="utf-8") as f:
            _json.dump(win, f, indent=2)
    except Exception:
        pass

    try:
        manifest: Dict[str, Any] = {
            "fold": fold.fold_id,
            "train_rows": int(len(split_result.train_df)),
            "val_rows": int(len(split_result.val_df)) if split_result.val_df is not None else 0,
            "test_rows": int(len(split_result.test_df)),
            "class_counts": {
                "train": split_result.y_train.value_counts().astype(int).to_dict(),
                "test": split_result.y_test.value_counts().astype(int).to_dict(),
            },
        }
        if split_result.val_df is not None and split_result.y_val is not None:
            manifest["class_counts"]["val"] = split_result.y_val.value_counts().astype(int).to_dict()
        time_col = split_cfg.get("time_col", "issue_d")
        if time_col in df.columns:
            def _fmt_range(frame: Optional[pd.DataFrame]):
                if frame is None or frame.empty or time_col not in frame.columns:
                    return {"min": None, "max": None}
                series = pd.to_datetime(frame[time_col], errors="coerce").dropna()
                if series.empty:
                    return {"min": None, "max": None}
                return {"min": str(series.min().date()), "max": str(series.max().date())}

            manifest["date_ranges"] = {
                "train": _fmt_range(split_result.train_df),
                "val": _fmt_range(split_result.val_df) if split_result.val_df is not None else {"min": None, "max": None},
                "test": _fmt_range(split_result.test_df),
            }
        with open(run_dir / "data_manifest.json", "w", encoding="utf-8") as f:
            _json.dump(manifest, f, indent=2)
    except Exception:
        pass

    eval_end = time.time()

    durations = {
        "preprocess": train_start - start_time,
        "train": train_end - train_start,
        "eval": eval_end - train_end,
        "total": eval_end - start_time,
    }

    summary = [
        f"# Temporal CV Fold {fold.fold_id} — {run_id}",
        "",
        f"Backend: {backend.name}",
        f"Model path: {model_path.name}",
        "",
        "## Metrics",
        f"- ROC AUC: {metrics.get('roc_auc'):.3f}",
        f"- Average Precision: {metrics.get('average_precision'):.3f}",
        f"- Threshold: {evaluation.threshold:.4f}",
        f"- Precision: {confusion['precision']:.3f}",
        f"- Recall: {confusion['recall']:.3f}",
        "",
        "## Durations (s)",
        f"- Preprocess: {durations['preprocess']:.2f}",
        f"- Train: {durations['train']:.2f}",
        f"- Eval: {durations['eval']:.2f}",
        f"- Total: {durations['total']:.2f}",
        "",
        "## Notes",
        notes.strip() if notes else "(no notes)",
    ]
    try:
        with open(run_dir / "README.md", "w", encoding="utf-8") as f:
            f.write("\n".join(summary))
    except Exception:
        pass

    return TrainingRunResult(
        run_id=run_id,
        evaluation=evaluation,
        metrics=metrics,
        confusion=confusion,
        model_path=model_path,
        durations=durations,
        fold_meta={"fold_id": fold.fold_id, **(fold.metadata or {})},
    )


def _run_temporal_cv(
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
) -> List[TrainingRunResult]:
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
        fold_result = _run_cv_fold(
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

    try:
        with open(artifact_mgr.reports_dir / "cv_metrics.json", "w", encoding="utf-8") as f:
            _json.dump(cv_report, f, indent=2)
    except Exception:
        pass

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
        with open(artifact_mgr.run_dir / "README.md", "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
    except Exception:
        pass

    return {
        "results": results,
        "fold_records": fold_records,
        "aggregate": aggregate,
        "report_path": (artifact_mgr.reports_dir / "cv_metrics.json"),
        "summary_path": (artifact_mgr.run_dir / "README.md"),
    }
def _resolve_torch_device(force_cpu: bool) -> tuple["torch.device", Dict[str, Any]]:
    """Best-effort accelerator selection with CPU fallback."""
    import torch

    meta: Dict[str, Any] = {"forced_cpu": bool(force_cpu)}

    if force_cpu:
        meta["selected"] = "cpu"
        meta["reason"] = "FORCE_CPU env toggle"
        return torch.device("cpu"), meta

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        meta.update({"selected": "cuda", "cuda_index": int(idx)})
        try:
            meta["cuda_name"] = torch.cuda.get_device_name(idx)
        except Exception:
            pass
        return torch.device("cuda"), meta

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        try:
            if mps_backend.is_available():
                meta["selected"] = "mps"
                try:
                    meta["mps_is_built"] = bool(mps_backend.is_built())
                except Exception:
                    pass
                ane_fn = getattr(mps_backend, "is_neural_engine_available", None)
                if callable(ane_fn):
                    try:
                        meta["mps_neural_engine"] = bool(ane_fn())
                    except Exception:
                        pass
                return torch.device("mps"), meta
        except Exception:
            pass

    xpu_backend = getattr(torch, "xpu", None)
    if xpu_backend is not None:
        try:
            if callable(getattr(xpu_backend, "is_available", None)) and xpu_backend.is_available():
                meta["selected"] = "xpu"
                return torch.device("xpu"), meta
        except Exception:
            pass

    meta["selected"] = "cpu"
    meta.setdefault("reason", "no accelerator detected")
    return torch.device("cpu"), meta

_TRUE_SET = {"1", "true", "yes", "on", "enabled"}
_FALSE_SET = {"0", "false", "no", "off", "disabled"}


def _env_flag(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    val = str(raw).strip().lower()
    if val in _TRUE_SET:
        return True
    if val in _FALSE_SET:
        return False
    logger.warning("Ignoring invalid boolean override %s=%r", name, raw)
    return None


def _apply_common_env_overrides(cfg: Dict[str, Any]) -> None:
    os_cfg = cfg.setdefault("oversampling", {})
    os_enabled = _env_flag("PIPELINE_OVERSAMPLING_ENABLED")
    if os_enabled is not None:
        os_cfg["enabled"] = os_enabled
        logger.info("Overriding oversampling.enabled via PIPELINE_OVERSAMPLING_ENABLED=%s", os_enabled)
    os_method = os.environ.get("PIPELINE_OVERSAMPLING_METHOD")
    if os_method:
        os_cfg["method"] = os_method
        logger.info("Overriding oversampling.method via PIPELINE_OVERSAMPLING_METHOD=%s", os_method)


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _align_probabilities(y_prob: np.ndarray | List[float], prob_label: Any, pos_label: int) -> np.ndarray:
    probs = np.asarray(y_prob, dtype=float)
    try:
        label = int(prob_label)
    except Exception:
        label = 0 if str(prob_label).lower() in {"0", "charged off", "charged_off", "default"} else 1
    if label == pos_label:
        return probs
    return 1.0 - probs


def _run_backend_pipeline(
    cfg_path: str | Path,
    *,
    backend: BackendPipeline,
    notes: Optional[str] = None,
):
    cfg_path = Path(cfg_path)
    cfg = load_config_with_extends(cfg_path)
    _apply_common_env_overrides(cfg)
    backend.apply_env_overrides(cfg)
    backend.validate_config(cfg)
    logger.info("Starting training pipeline | config=%s", cfg_path)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    os_cfg = cfg.get("oversampling", {"enabled": True})
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]
    eval_cfg = cfg.get("eval", {})
    training_cfg = cfg.get("training", {})
    tracking_cfg = cfg.get("tracking", {})

    # Determine single-run output directory strategy
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    single_run_dir_mode = bool(out_cfg.get("runs_root"))

    # Render a backend-aware group name for both local runs and W&B
    csv_base_init = Path(str(data_cfg.get("csv_path", ""))).stem
    split_method_init = split_cfg.get("method", "time")
    pos_label_init = eval_cfg.get("pos_label", 1)
    if isinstance(pos_label_init, str):
        pos_label_init = 0 if str(pos_label_init).lower() in {"default", "charged off", "charged_off"} else 1
    pos_tok_init = "co" if int(pos_label_init) == 0 else "fp"
    backend_tok_init = backend.name
    try:
        sha_init = _collect_env_metadata().get("git", {}).get("commit")
    except Exception:
        sha_init = None
    ctx_init = {
        "dataset": csv_base_init,
        "split": split_method_init,
        "pos": pos_tok_init,
        "backend": backend_tok_init,
        "sha": sha_init or "",
    }
    wb_cfg_local = tracking_cfg.get("wandb", {}) if isinstance(tracking_cfg, dict) else {}
    default_group_local = f"{csv_base_init}|{split_method_init}|{pos_tok_init}|{backend_tok_init}"
    try:
        tmpl_local = wb_cfg_local.get("group_template") if isinstance(wb_cfg_local, dict) else None
        group_name_for_local = (
            str(tmpl_local).format(**ctx_init) if tmpl_local else default_group_local
        )
    except Exception:
        group_name_for_local = default_group_local

    if single_run_dir_mode:
        runs_root = Path(out_cfg["runs_root"]).resolve()
        run_dir = runs_root / group_name_for_local / run_id
        models_dir = run_dir
        reports_dir = run_dir
        figures_dir = run_dir / "figures"
    else:
        models_dir = Path(out_cfg["models_dir"]).resolve()
        reports_dir = Path(out_cfg["reports_dir"]).resolve()
        figures_dir = Path(out_cfg["figures_dir"]).resolve()
        runs_root = Path(out_cfg.get("runs_root", reports_dir / "runs")).resolve()
        run_dir = runs_root / group_name_for_local / run_id

    # Ensure uniqueness if multiple runs start within the same second
    try:
        if run_dir.exists():
            base = run_id
            idx = 1
            while run_dir.exists():
                run_id = f"{base}_{idx:02d}"
                run_dir = runs_root / group_name_for_local / run_id
                idx += 1
    except Exception:
        pass

    logger.info("Resolved run directories | run_id=%s | run_dir=%s", run_id, run_dir)

    artifact_mgr = ArtifactManager(
        run_dir=run_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        single_run_mode=single_run_dir_mode,
    )
    models_dir = artifact_mgr.models_dir
    reports_dir = artifact_mgr.reports_dir
    figures_dir = artifact_mgr.figures_dir
    run_dir = artifact_mgr.run_dir

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
    t_load_start = t0
    df = load_and_prepare(load_config)
    t_load_end = time.time()
    logger.info("Data loading completed in %.2fs", t_load_end - t_load_start)

    feature_inputs = resolve_feature_inputs(
        df,
        data_cfg.get("features", []),
        data_cfg["target_col"],
        time_columns=list(data_cfg.get("parse_dates", [])),
    )
    logger.info("Resolved %d feature columns for modeling", len(feature_inputs))

    cv_cfg = split_cfg.get("cv", {}) or {}
    cv_enabled = bool(cv_cfg.get("enabled")) and int(cv_cfg.get("n_folds", 0)) >= 2
    cv_summary_for_return: Optional[Dict[str, Any]] = None
    if cv_enabled:
        if str(split_cfg.get("method", "time")).lower() != "time":
            raise ValueError("Temporal k-fold requires split.method='time'.")
        logger.info(
            "Temporal cross-validation enabled | folds=%d | mode=%s",
            int(cv_cfg.get("n_folds", 0)),
            cv_cfg.get("mode", "expanding"),
        )
        cv_output = _run_temporal_cv(
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
            artifact_mgr=artifact_mgr,
            run_id=run_id,
            notes=notes,
            backend=backend,
        )
        elapsed = time.time() - t0
        fold_summaries = [
            {
                "fold_id": res.fold_meta.get("fold_id") if res.fold_meta else None,
                "run_id": res.run_id,
                "metrics": res.metrics,
                "confusion": res.confusion,
                "threshold": float(res.evaluation.threshold),
                "durations": res.durations,
            }
            for res in cv_output["results"]
        ]
        cv_summary = {
            "cv_metrics_path": Path(cv_output["report_path"]).as_posix(),
            "run_dir": run_dir.as_posix(),
            "n_folds": len(cv_output["results"]),
            "backend": model_cfg.get("backend", "pytorch"),
            "elapsed_sec": elapsed,
            "folds": fold_summaries,
            "aggregate": cv_output.get("aggregate"),
        }
        if not bool(cv_cfg.get("train_full_after", False)):
            return cv_summary
        cv_summary_for_return = cv_summary
        logger.info("Temporal CV complete; proceeding to full-data training as requested.")

    # Split into train/val/test with consistent objects
    t_split_start = time.time()
    split_result = train_val_test_split(
        df,
        feature_inputs,
        data_cfg["target_col"],
        method=split_cfg.get("method", "random"),
        time_col=split_cfg.get("time_col", "issue_d"),
        test_size=float(split_cfg.get("test_size", 0.2)),
        val_size=float(model_cfg.get("val_split", 0.2)),
        random_state=int(split_cfg.get("random_state", 42)),
        stratify=bool(split_cfg.get("stratify", True)),
    )
    t_split_end = time.time()
    logger.info("Splitting completed in %.2fs", t_split_end - t_split_start)

    winsor_cfg = data_cfg.get("winsorize") if data_cfg.get("winsorize_enabled", True) else None
    preprocessing_cfg = cfg.get("preprocessing", {})
    preproc_result = preprocess_tabular_data(
        split_result,
        winsorize_cfg=winsor_cfg,
        preprocessing_cfg=preprocessing_cfg,
    )
    logger.info(
        "Preprocessing finished | train=%s | val=%s | test=%s",
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
            random_state=int(split_cfg.get("random_state", 42)),
            params=os_cfg.get("params"),
        )
        X_train_np = resample_result.X_resampled
        y_train_np = resample_result.y_resampled
        logger.info("Using resampled training data with shape %s", X_train_np.shape)
    else:
        X_train_np = preproc_result.X_train
        y_train_np = preproc_result.y_train
        logger.info("Resampling disabled; training shape %s", X_train_np.shape)

    X_val_np = preproc_result.X_val
    y_val_np = preproc_result.y_val
    X_test_np = preproc_result.X_test
    y_test_np = preproc_result.y_test
    t_preproc_end = time.time()
    logger.info("Data preparation stages completed in %.2fs", t_preproc_end - t_load_start)

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

    model_cfg_effective = backend.prepare_model_config(
        model_cfg=copy.deepcopy(model_cfg),
        training_cfg=training_cfg,
        y_train=y_train_np,
    )

    run_context = RunContext(run_id=run_id, run_dir=run_dir, artifact_mgr=artifact_mgr)
    model_path = backend.resolve_model_path(
        out_cfg=out_cfg,
        artifact_mgr=artifact_mgr,
        fold_meta=None,
    )
    logger.info("Resolved model output path: %s", model_path)

    # Optional: W&B initialization
    wandb_run = None
    wandb_enabled = False
    try:
        tracking_backend = str(tracking_cfg.get("backend", "none")).lower()
        if tracking_backend == "wandb" or (tracking_cfg.get("wandb", {}).get("enabled")):
            wandb_enabled = True
            import wandb  # type: ignore
            wb_cfg = tracking_cfg.get("wandb", {})
            mode = str(wb_cfg.get("mode", "online"))
            # Optional login via env var (preferred in headless/CI)
            try:
                api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WB_API_KEY")
                if api_key:
                    try:
                        wandb.login(key=api_key)
                    except Exception:
                        pass
            except Exception:
                pass
            if wb_cfg.get("ignore_globs"):
                # Space-separated patterns per W&B convention
                os.environ["WANDB_IGNORE_GLOBS"] = " ".join(wb_cfg.get("ignore_globs", []))
            # Derive default group/job_type for organization
            csv_base_init = Path(str(data_cfg.get("csv_path", ""))).stem
            split_method_init = split_cfg.get("method", "time")
            pos_label_init = eval_cfg.get("pos_label", 1)
            if isinstance(pos_label_init, str):
                pos_label_init = 0 if str(pos_label_init).lower() in {"default", "charged off", "charged_off"} else 1
            pos_tok_init = "co" if int(pos_label_init) == 0 else "fp"
            default_group = f"{csv_base_init}|{split_method_init}|{pos_tok_init}|{backend_tok_init}"
            # Template context available pre-training
            try:
                sha_init = _collect_env_metadata().get("git", {}).get("commit")
            except Exception:
                sha_init = None
            ctx_init = {
                "dataset": csv_base_init,
                "split": split_method_init,
                "pos": pos_tok_init,
                "backend": backend_tok_init,
                "sha": sha_init or "",
            }
            # Render group/job_type from templates if provided
            group_val = wb_cfg.get("group")
            if not group_val:
                tmpl = wb_cfg.get("group_template")
                if tmpl:
                    try:
                        group_val = str(tmpl).format(**ctx_init)
                    except Exception:
                        group_val = default_group
                else:
                    group_val = default_group
            job_type_val = wb_cfg.get("job_type")
            if not job_type_val:
                tmpl = wb_cfg.get("job_type_template")
                if tmpl:
                    try:
                        job_type_val = str(tmpl).format(**ctx_init)
                    except Exception:
                        job_type_val = "train"
                else:
                    job_type_val = "train"
            # Allow project/entity from env when not provided in config
            _entity_env = os.environ.get("WANDB_ENTITY") or os.environ.get("WB_ENTITY")
            _project_env = os.environ.get("WANDB_PROJECT")
            wandb_run = wandb.init(
                project=wb_cfg.get("project") or _project_env or "loan-risk-mlp",
                entity=wb_cfg.get("entity") or _entity_env or None,
                config=cfg,
                mode=mode,
                group=group_val,
                job_type=job_type_val,
                settings=wandb.Settings(code_dir=None),
            )
            try:
                logger.info(
                    "Initialized W&B run id=%s url=%s",
                    getattr(wandb_run, "id", None),
                    getattr(wandb_run, "url", None),
                )
            except Exception:
                pass
            # Define epoch as step and map metrics to it for clean charts
            try:
                wandb.define_metric("epoch")
                for _m in ["loss", "val_loss", "val_auc", "lr", "epoch_time_sec"]:
                    wandb.define_metric(_m, step_metric="epoch")
            except Exception:
                pass
            # Expose basic identifiers for downstream consumers (results/auto-pull)
            try:
                _wb = wandb.run
                wb_id = getattr(_wb, "id", None)
                wb_path = "/".join([p for p in (getattr(_wb, "entity", None), getattr(_wb, "project", None), getattr(_wb, "id", None)) if p])
                wb_url = getattr(_wb, "url", None)
            except Exception:
                wb_id = None
                wb_path = None
                wb_url = None
    except Exception:
        wandb_enabled = False
        wandb_run = None

    t_train_start = time.time()
    # Seed Python/NumPy/Torch for reproducibility
    seed_value = int(split_cfg.get("random_state", 42))
    set_seed(seed_value)
    logger.info("Random seed set to %d", seed_value)

    # Resolve configured positive label early so epoch val_auc aligns to it
    _pos_cfg = eval_cfg.get("pos_label", 1)
    if isinstance(_pos_cfg, str):
        _pos_cfg = 0 if str(_pos_cfg).lower() in {"default", "charged off", "charged_off"} else 1
    pos_label_for_auc = int(_pos_cfg)

    training_result = backend.train(
        dataset=dataset,
        model_cfg=model_cfg_effective,
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        run_context=run_context,
        model_path=model_path,
        random_seed=seed_value,
        pos_label=pos_label_for_auc,
        fold_meta=None,
        wandb_run=wandb_run,
        wandb_enabled=wandb_enabled,
        cfg=cfg,
    )
    model_path = training_result.model_path
    history_obj = training_result.history
    raw_result = training_result.raw or {}
    y_prob = np.asarray(training_result.y_prob)
    param_count = raw_result.get("param_count")
    device_info = raw_result.get("device_info", {})
    device_used = None
    if isinstance(device_info, dict):
        device_used = device_info.get("selected") or device_info.get("repr")
    if device_used is None:
        device_used = raw_result.get("device")
    epochs_ran = raw_result.get("epochs_ran")
    t_train_end = time.time()
    logger.info("Training stage finished in %.2fs", t_train_end - t_train_start)
    logger.info(
        "Model summary | backend=%s | device=%s | epochs=%s | params=%s",
        backend.name,
        device_used,
        epochs_ran,
        param_count,
    )

    # Evaluation controls
    pos_label_cfg = eval_cfg.get("pos_label", 1)
    if isinstance(pos_label_cfg, str):
        pos_label_cfg = 0 if pos_label_cfg.lower() in {"default", "charged off", "charged_off"} else 1
    pos_label_int = int(pos_label_cfg)

    prob_label_raw = int(training_result.prob_label)

    y_true_pos_test = (y_test_np.astype(int) == pos_label_int).astype(int)
    y_prob_pos_test = _align_probabilities(y_prob, prob_label_raw, pos_label_int)

    y_true_pos_val = None
    y_prob_pos_val = None
    if y_val_np is not None:
        y_true_pos_val = (y_val_np.astype(int) == pos_label_int).astype(int)
        val_buf = raw_result.get("y_prob_val") if isinstance(raw_result, dict) else None
        if val_buf is not None:
            y_prob_pos_val = _align_probabilities(val_buf, prob_label_raw, pos_label_int)
        elif y_true_pos_val is not None and y_true_pos_val.size > 0:
            y_prob_pos_val = None

    evaluation = evaluate_binary_classification(
        y_true=y_true_pos_test,
        y_prob=y_prob_pos_test,
        threshold_cfg=eval_cfg.get("threshold", {}),
        y_true_val=y_true_pos_val,
        y_prob_val=y_prob_pos_val,
        pos_label=pos_label_int,
    )
    metrics = evaluation.metrics
    cm = evaluation.confusion
    threshold = evaluation.threshold
    strategy = evaluation.threshold_strategy
    try:
        threshold_value = float(threshold) if threshold is not None else float("nan")
    except (TypeError, ValueError):
        threshold_value = float("nan")
    roc_auc_val = metrics.get("roc_auc")
    pr_auc_val = metrics.get("average_precision")
    logger.info(
        "Evaluation complete | threshold_strategy=%s | threshold=%s | roc_auc=%s | pr_auc=%s",
        strategy,
        "{:.4f}".format(threshold_value) if not np.isnan(threshold_value) else "n/a",
        "{:.4f}".format(float(roc_auc_val)) if roc_auc_val is not None else "n/a",
        "{:.4f}".format(float(pr_auc_val)) if pr_auc_val is not None else "n/a",
    )

    pos_label_name = (
        "positive=default (Charged Off)" if pos_label_int == 0 else "positive=1 (Fully Paid)"
    )

    if wandb_enabled and wandb_run is not None:
        try:
            backend.log_wandb(
                wandb_run=wandb_run,
                dataset=dataset,
                model_cfg=model_cfg_effective,
                training_cfg=training_cfg,
                eval_cfg=eval_cfg,
                training_result=training_result,
                metrics=metrics,
                confusion=cm,
                run_context=run_context,
                notes=notes,
            )
        except Exception:
            logger.exception("Backend-specific W&B logging failed", exc_info=True)

    # Save common artifacts (latest)
    save_metrics(metrics, artifact_mgr.metrics_path)
    plot_learning_curves(history_obj, figures_dir / "learning_curves.png")
    artifact_mgr.save_confusion(cm)
    plot_roc_curve(evaluation.y_true, evaluation.y_prob, figures_dir / "roc_curve.png", point=(cm["fpr"], cm["tpr"]))
    plot_pr_curve(evaluation.y_true, evaluation.y_prob, figures_dir / "pr_curve.png", point=(cm["precision"], cm["recall"]))
    logger.info("Saved metrics and figures to %s", run_dir)

    cm_path = artifact_mgr.confusion_path
    # W&B: log an interactive confusion matrix visualization
    if wandb_enabled and wandb_run is not None:
        try:
            import wandb  # type: ignore
            import numpy as _np  # local alias to avoid shadowing
            y_pred_pos = (_np.asarray(y_prob_pos_test) >= float(threshold)).astype(int)
            # Class names reflecting the configured positive class
            if int(pos_label_cfg) == 0:
                # 0 => Charged Off is the positive class in our mapping above
                class_names = ["fully_paid", "charged_off"]  # index 0,1 align to y_true_pos
            else:
                class_names = ["charged_off", "fully_paid"]
            cm_plot = wandb.plot.confusion_matrix(
                y_true=_np.asarray(y_true_pos_test).astype(int),
                preds=y_pred_pos.astype(int),
                class_names=class_names,
            )
            wandb.log({"confusion_matrix": cm_plot})
        except Exception:
            pass

    # Compute and save ROC/PR point sweeps as CSV in run folder later

    run_fig_dir = artifact_mgr.figures_run_dir
    artifact_mgr.stage_run_artifacts(
        model_path,
        figure_names=["learning_curves.png", "roc_curve.png", "pr_curve.png"],
    )

    # Save per-threshold sweeps for ROC and PR (CSV)
    try:
        fpr, tpr, thr_roc = evaluation.roc_points
        with open(run_dir / "roc_points.csv", "w", encoding="utf-8") as f:
            f.write("threshold,fpr,tpr\n")
            for idx in range(len(fpr)):
                threshold_val = "" if idx == 0 else float(thr_roc[idx - 1])
                f.write(f"{threshold_val},{float(fpr[idx])},{float(tpr[idx])}\n")

        precision, recall, thr_pr = evaluation.pr_points
        with open(run_dir / "pr_points.csv", "w", encoding="utf-8") as f:
            f.write("threshold,precision,recall\n")
            if len(precision) > 0:
                f.write(f",{float(precision[0])},{float(recall[0])}\n")
            for idx in range(1, len(precision)):
                threshold_val = "" if idx - 1 >= len(thr_pr) else float(thr_pr[idx - 1])
                f.write(f"{threshold_val},{float(precision[idx])},{float(recall[idx])}\n")

        try:
            import numpy as _np
            from src.eval.metrics import confusion_metrics_at_threshold as _cm_thr

            thr_grid = _np.linspace(0.0, 1.0, 101)
            with open(run_dir / "threshold_metrics.csv", "w", encoding="utf-8") as f:
                f.write("threshold,precision,recall,tpr,fpr,specificity,f1\n")
                y_true_eval = _np.asarray(evaluation.y_true).astype(int)
                y_prob_eval = _np.asarray(evaluation.y_prob)
                for th in thr_grid:
                    metrics_at_th = _cm_thr(y_true_eval, y_prob_eval, float(th))
                    prec_v = float(metrics_at_th.get("precision", 0.0))
                    rec_v = float(metrics_at_th.get("recall", 0.0))
                    tpr_v = float(metrics_at_th.get("tpr", 0.0))
                    fpr_v = float(metrics_at_th.get("fpr", 0.0))
                    spec_v = 1.0 - fpr_v
                    f1_v = (2 * prec_v * rec_v) / (prec_v + rec_v + 1e-12)
                    f.write(f"{th:.4f},{prec_v:.6f},{rec_v:.6f},{tpr_v:.6f},{fpr_v:.6f},{spec_v:.6f},{f1_v:.6f}\n")
        except Exception:
            pass
    except Exception:
        pass

    # W&B: also log ROC/PR sweeps and threshold metrics as interactive panels
    if wandb_enabled and wandb_run is not None:
        try:
            import wandb  # type: ignore
            import numpy as _np
            # Built-in ROC/PR plots (interactive)
            try:
                wandb.log({
                    "roc_curve": wandb.plot.roc_curve(_np.asarray(evaluation.y_true).astype(int), _np.asarray(evaluation.y_prob)),
                    "pr_curve": wandb.plot.pr_curve(_np.asarray(evaluation.y_true).astype(int), _np.asarray(evaluation.y_prob)),
                })
            except Exception:
                pass
            # Tables for ROC/PR points
            fpr, tpr, thr_roc = evaluation.roc_points
            roc_tbl = wandb.Table(
                data=[[float(_np.nan if i == 0 else thr_roc[i - 1]), float(fpr[i]), float(tpr[i])] for i in range(len(fpr))],
                columns=["threshold", "fpr", "tpr"],
            )
            prec, rec, thr_pr = evaluation.pr_points
            pr_tbl = wandb.Table(
                data=(([[float(_np.nan), float(prec[0]), float(rec[0])]] if len(prec) > 0 else []) +
                      [[float(thr_pr[i - 1]), float(prec[i]), float(rec[i])] for i in range(1, len(prec))]),
                columns=["threshold", "precision", "recall"],
            )
            # Threshold metrics sweep (precision/recall/specificity/f1 vs threshold)
            try:
                from src.eval.metrics import confusion_metrics_at_threshold as _cm_thr
                thr_grid = _np.linspace(0.0, 1.0, 101)
                rows = []
                for th in thr_grid:
                    m = _cm_thr(_np.asarray(y_true_pos_test).astype(int), _np.asarray(y_prob_pos_test), float(th))
                    prec_v = float(m.get("precision", 0.0))
                    rec_v = float(m.get("recall", 0.0))
                    tpr_v = float(m.get("tpr", 0.0))
                    fpr_v = float(m.get("fpr", 0.0))
                    spec_v = 1.0 - fpr_v
                    f1_v = (2 * prec_v * rec_v) / (prec_v + rec_v + 1e-12)
                    rows.append([float(th), prec_v, rec_v, tpr_v, fpr_v, spec_v, f1_v])
                thr_tbl = wandb.Table(data=rows, columns=["threshold", "precision", "recall", "tpr", "fpr", "specificity", "f1"])
                # Line charts for threshold sweeps
                try:
                    wandb.log({
                        "threshold_precision": wandb.plot.line_series(xs=[r[0] for r in rows], ys=[[r[1] for r in rows]], keys=["precision"], title="Precision vs Threshold", xname="threshold"),
                        "threshold_recall": wandb.plot.line_series(xs=[r[0] for r in rows], ys=[[r[2] for r in rows]], keys=["recall"], title="Recall vs Threshold", xname="threshold"),
                        "threshold_f1": wandb.plot.line_series(xs=[r[0] for r in rows], ys=[[r[6] for r in rows]], keys=["f1"], title="F1 vs Threshold", xname="threshold"),
                    })
                except Exception:
                    pass
            except Exception:
                thr_tbl = None
            log_payload = {"roc_table": roc_tbl, "pr_table": pr_tbl}
            if thr_tbl is not None:
                log_payload["threshold_metrics_table"] = thr_tbl
            wandb.log(log_payload)
        except Exception:
            pass

    # Save resolved config used for this run
    resolved_cfg_path = run_dir / "config_resolved.yaml"
    try:
        with open(resolved_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception:
        pass

    # Snapshot current Python environment (pip freeze)
    try:
        import subprocess
        freeze_txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode()
        (run_dir / "requirements.freeze.txt").write_text(freeze_txt, encoding="utf-8")
    except Exception:
        pass

    # Save data manifest with provenance and date ranges
    dataset_info: Dict[str, Any] | None = None
    try:
        csv_path_raw = str(data_cfg.get("csv_path", ""))
        csv_path = Path(csv_path_raw)
        csv_abs = csv_path.resolve()
        manifest: Dict[str, Any] = {
            "csv_path": csv_path_raw,
            "csv_path_abs": csv_abs.as_posix(),
        }
        try:
            st = csv_abs.stat()
            manifest.update(
                {
                    "filesize_bytes": int(st.st_size),
                    "mtime": int(st.st_mtime),
                    "sha256": _file_sha256(csv_abs),
                }
            )
        except Exception:
            pass

        # Dataset-level stats
        try:
            y_series = df[data_cfg["target_col"]]
            counts = y_series.value_counts().to_dict()
            manifest["n_rows"] = int(df.shape[0])
            manifest["n_cols"] = int(df.shape[1])
            manifest["class_counts"] = {int(k): int(v) for k, v in counts.items()}
        except Exception:
            pass

        # Date ranges
        try:
            time_col = split_cfg.get("time_col", "issue_d")
            if time_col in df.columns:
                def _fmt_range(s):
                    s = s.dropna()
                    if s.empty:
                        return {"min": None, "max": None}
                    return {"min": str(s.min().date()), "max": str(s.max().date())}

                manifest["date_ranges"] = {"dataset": _fmt_range(df[time_col])}
                train_frame = split_result.train_df
                val_frame = split_result.val_df
                test_frame = split_result.test_df
                if train_frame is not None and time_col in train_frame.columns:
                    manifest["date_ranges"]["train"] = _fmt_range(train_frame[time_col])
                if val_frame is not None and time_col in val_frame.columns:
                    manifest["date_ranges"]["val"] = _fmt_range(val_frame[time_col])
                if test_frame is not None and time_col in test_frame.columns:
                    manifest["date_ranges"]["test"] = _fmt_range(test_frame[time_col])
        except Exception:
            pass

        # Train/test class counts
        try:
            import numpy as _np
            def _counts(arr):
                unique, cnts = _np.unique(arr.astype(int), return_counts=True)
                return {int(k): int(v) for k, v in zip(unique, cnts)}
            manifest["train_class_counts"] = _counts(y_train_np)
            manifest["test_class_counts"] = _counts(y_test_np)
            if y_val_np is not None:
                manifest["val_class_counts"] = _counts(y_val_np)
        except Exception:
            pass

        if resample_result is not None:
            manifest["resampling"] = {
                "method": resample_result.method,
                "before_counts": resample_result.before_counts,
                "after_counts": resample_result.after_counts,
            }

        with open(run_dir / "data_manifest.json", "w", encoding="utf-8") as f:
            _json.dump(manifest, f, indent=2)
        dataset_info = manifest
    except Exception:
        pass

    # Save feature lists used by preprocessor
    try:
        features_manifest = {
            "numerical_features": list(preproc_result.numerical_features),
            "categorical_features": list(preproc_result.categorical_features),
            "feature_inputs": list(feature_inputs),
            "encoded_feature_names": list(feature_names),
        }
        with open(run_dir / "features.json", "w", encoding="utf-8") as f:
            _json.dump(features_manifest, f, indent=2)
    except Exception:
        pass

    # Omit writing per-epoch history/stats files to the run directory
    # Compute model size for summary table
    try:
        _model_size = int((run_dir / model_path.name).stat().st_size)
    except Exception:
        _model_size = None

    # Human-readable start/end timestamps (UTC)
    t_eval_end = time.time()
    _start_iso = datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(timespec="seconds")
    _end_iso = datetime.fromtimestamp(t_eval_end, tz=timezone.utc).isoformat(timespec="seconds")

    # Compose README content (rich template)
    notes_text = (notes or "").strip()
    summary_lines = [
        f"# Training Summary — {run_id}",
        "",
        f"Config: `{cfg_path}`",
        f"Backend: {backend.name}",
        f"Positive class: {pos_label_name}",
        f"Threshold strategy: {strategy}",
        f"Chosen threshold: {threshold:.6f}",
        "",
        "## Run Summary",
        "",
        "| Key | Value |",
        "| --- | --- |",
        f"| Device | {device_used or 'n/a'} |",
    ]
    if isinstance(device_info, dict) and device_info:
        detail_parts = []
        for key in ("cuda_name", "cuda_index", "mps_is_built", "mps_neural_engine", "reason"):
            val = device_info.get(key)
            if val is None or val == "":
                continue
            detail_parts.append(f"{key}={val}")
        if detail_parts:
            detail_str = "; ".join(detail_parts)
            summary_lines.append(f"| Device details | {detail_str} |")
    artifacts_section = [
        f"| Epochs (ran) | {int(epochs_ran) if epochs_ran is not None else 'n/a'} |",
        f"| Param count | {int(param_count) if param_count is not None else 'n/a'} |",
        f"| Model size | {(_model_size/1024):.1f} KB |" if _model_size is not None else "| Model size | n/a |",
        f"| Start (UTC) | {_start_iso} |",
        f"| End (UTC) | {_end_iso} |",
        f"| Total time | {(t_eval_end - t0):.2f} s |",
        f"| Load | {(t_load_end - t_load_start):.2f} s |",
        f"| Split | {(t_split_end - t_split_start):.2f} s |",
        f"| Preprocess | {(t_preproc_end - t_split_end):.2f} s |",
        f"| Train | {(t_train_end - t_preproc_end):.2f} s |",
        f"| Eval | {(t_eval_end - t_train_end):.2f} s |",
        "",
        "## What Changed",
        notes_text if notes_text else "(no notes provided)",
        "",
        "## Metrics",
        f"- ROC AUC: {metrics.get('roc_auc'):.3f}",
        f"- Average Precision: {metrics.get('average_precision'):.3f}",
        f"- Precision (at threshold): {cm['precision']:.3f}",
        f"- Recall (TPR): {cm['recall']:.3f}",
        f"- Specificity (TNR): {1.0 - cm['fpr']:.3f}",
        f"- Confusion: TP={int(cm['tp'])}, FP={int(cm['fp'])}, TN={int(cm['tn'])}, FN={int(cm['fn'])}",
       f"- n_train: {int(len(y_train_np))}",
        f"- n_val: {int(len(y_val_np)) if y_val_np is not None else 0}",
        f"- n_test: {int(len(y_test_np))}",
       f"- n_features: {int(X_train_np.shape[1])}",
        f"- Resampling: {resample_result.method if resample_result is not None else 'disabled'}",
        "",
        "## Classification Report (at threshold)",
        "```json",
        _json.dumps(metrics.get("classification_report", {}), indent=2),
        "```",
        "",
        "## Artifacts",
        f"- Model: `{model_path.name}`",
        f"- Metrics: `metrics.json`",
        f"- Confusion: `confusion.json`",
        f"- History CSV: `history.csv`",
        f"- ROC points CSV: `roc_points.csv`",
        f"- PR points CSV: `pr_points.csv`",
        f"- Learning curves: `figures/learning_curves.png`",
        f"- ROC curve: `figures/roc_curve.png`",
        f"- PR curve: `figures/pr_curve.png`",
        f"- Resolved config: `config_resolved.yaml`",
        f"- Features manifest: `features.json`",
    ]
    try:
        extras = list(
            backend.extra_artifact_lines(
                training_result=training_result,
                run_context=run_context,
                cfg=cfg,
            )
        )
        artifacts_section.extend(extras)
    except Exception:
        pass
    artifacts_section.extend([
        "",
        "## Notes",
        ("- Evaluated defaults as the positive class." if int(pos_label_cfg) == 0 else "- Evaluated fully paid as the positive class."),
        "- Threshold selected according to configured strategy and annotated on curves.",
    ])
    summary_lines.extend(artifacts_section)
    # Add a simple threshold sanity note for extreme operating points
    try:
        if cm.get("precision", 1.0) < 1e-3 or cm.get("recall", 1.0) < 1e-3:
            summary_lines.append("")
            summary_lines.append(
                "> Note: precision or recall is near 0 at the chosen threshold. Consider revising the threshold strategy or dataset balance."
            )
    except Exception:
        pass

    # Instead of saving README locally, upload as a W&B artifact file
    readme_content = "\n".join(summary_lines)
    t_eval_end = time.time()

    # W&B: log summary metrics and selected artifacts
    if wandb_enabled and wandb_run is not None:
        try:
            import wandb  # type: ignore
            # Set a friendly run name if not provided
            wb_cfg = tracking_cfg.get("wandb", {})
            if wb_cfg.get("run_name"):
                wandb.run.name = str(wb_cfg.get("run_name"))
            else:
                wandb.run.name = run_id
            # Tags for fast filtering
            try:
                tag_set = {
                    f"backend:{backend.name}",
                    f"split:{split_cfg.get('method', 'time')}",
                    f"threshold:{strategy}",
                    f"pos_label:{int(pos_label_cfg)}",
                }
                csv_base = Path(str(data_cfg.get("csv_path", ""))).name
                if csv_base:
                    tag_set.add(f"data:{csv_base}")
                try:
                    tag_set.update(
                        backend.additional_wandb_tags(
                            training_result=training_result,
                            run_context=run_context,
                            cfg=cfg,
                        )
                    )
                except Exception:
                    pass
                wandb.run.tags = list({*list(wandb.run.tags or []), *tag_set})  # type: ignore[attr-defined]
            except Exception:
                pass
            # Log scalar metrics
            log_payload = {
                "roc_auc": float(metrics.get("roc_auc", float("nan")) or float("nan")),
                "average_precision": float(metrics.get("average_precision", float("nan")) or float("nan")),
                "threshold": float(threshold),
                "precision_at_thr": float(cm.get("precision", float("nan")) or float("nan")),
                "recall_at_thr": float(cm.get("recall", float("nan")) or float("nan")),
                "specificity_at_thr": float(1.0 - cm.get("fpr", 0.0)),
                "n_train": int(len(y_train_np)),
                "n_test": int(len(y_test_np)),
                "n_features": int(X_train_np.shape[1]),
            }
            wandb.log(log_payload)

            # Enrich summary and config with metadata
            try:
                env_meta = _collect_env_metadata()
                sys_meta = _collect_system_info()
                # model size on disk
                model_size = None
                try:
                    st = (run_dir / model_path.name).stat()
                    model_size = int(st.st_size)
                except Exception:
                    pass
                wandb.summary.update({
                    "run_id": run_id,
                    "param_count": int(param_count) if param_count is not None else None,
                    "model_size_bytes": model_size,
                    "model_filename": model_path.name,
                    "threshold_strategy": strategy,
                    "threshold_source": evaluation.threshold_source,
                    "pos_label_name": pos_label_name,
                    "device.used": device_used,
                    "epochs_ran": int(epochs_ran) if epochs_ran is not None else None,
                    # Timing breakdown
                    "timing.total_sec": float(t_eval_end - t0),
                    "timing.load_sec": float(t_load_end - t_load_start),
                    "timing.split_sec": float(t_split_end - t_split_start),
                    "timing.preprocess_sec": float(t_preproc_end - t_split_end),
                    "timing.train_sec": float(t_train_end - t_preproc_end),
                    "timing.eval_sec": float(t_eval_end - t_train_end),
                    # Start/end timestamps
                    "time.start_epoch": int(t0),
                    "time.end_epoch": int(t_eval_end),
                    "time.start_iso": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(timespec="seconds"),
                    "time.end_iso": datetime.fromtimestamp(t_eval_end, tz=timezone.utc).isoformat(timespec="seconds"),
                })
                if isinstance(device_info, dict):
                    wandb.summary.update({f"device.{k}": v for k, v in device_info.items()})
                # env versions and git
                for k, v in (env_meta.get("env", {}) or {}).items():
                    wandb.summary.update({f"env.{k}": v})
                for k, v in (env_meta.get("git", {}) or {}).items():
                    wandb.summary.update({f"git.{k}": v})
                # system info
                for k, v in (sys_meta or {}).items():
                    wandb.summary.update({f"system.{k}": v})
                if resample_result is not None:
                    wandb.summary.update({
                        "resample.method": resample_result.method,
                        "resample.before_counts": resample_result.before_counts,
                        "resample.after_counts": resample_result.after_counts,
                    })
                # add commit as tag, if present
                try:
                    sha = env_meta.get("git", {}).get("commit")
                    if sha:
                        wandb.run.tags = list({*list(wandb.run.tags or []), f"commit:{sha}"})  # type: ignore[attr-defined]
                except Exception:
                    pass
                # Backend-specific artifact logging handled by pipeline-specific logic
                if dataset_info:
                    # add a compact subset of dataset manifest
                    ds = dataset_info
                    wandb.summary.update({
                        "data.csv_path": ds.get("csv_path"),
                        "data.sha256": ds.get("sha256"),
                        "data.n_rows": ds.get("n_rows"),
                        "data.n_cols": ds.get("n_cols"),
                        "data.class_counts": ds.get("class_counts"),
                        "data.date_ranges": ds.get("date_ranges"),
                    })
            except Exception:
                pass

            # Log W&B-native plots (no PNG uploads)
            try:
                # Learning curves as a multi-series line chart
                try:
                    tr = history_obj.history.get("loss", [])
                    va = history_obj.history.get("val_loss", [])
                    npts = max(len(tr), len(va))
                    xs = list(range(1, npts + 1))
                    ys = []
                    keys = []
                    if tr:
                        ys.append([float(x) for x in tr])
                        keys.append("loss")
                    if va:
                        ys.append([float(x) for x in va])
                        keys.append("val_loss")
                    if ys:
                        plot = wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title="Learning Curves", xname="epoch")
                        wandb.log({"learning_curves_plot": plot})
                except Exception:
                    pass
                # ROC and PR curves from raw predictions
                try:
                    _fpr, _tpr, _ = evaluation.roc_points
                    roc_table = wandb.Table(columns=["fpr", "tpr"])  # type: ignore[attr-defined]
                    for i in range(len(_fpr)):
                        roc_table.add_data(float(_fpr[i]), float(_tpr[i]))
                    roc_plot = wandb.plot.line(roc_table, "fpr", "tpr", title="ROC Curve")
                    wandb.log({"roc_curve_plot": roc_plot})
                except Exception:
                    pass
                try:
                    _prec, _rec, _ = evaluation.pr_points
                    pr_table = wandb.Table(columns=["recall", "precision"])  # type: ignore[attr-defined]
                    for i in range(len(_prec)):
                        pr_table.add_data(float(_rec[i]), float(_prec[i]))
                    pr_plot = wandb.plot.line(pr_table, "recall", "precision", title="Precision-Recall Curve")
                    wandb.log({"pr_curve_plot": pr_plot})
                except Exception:
                    pass
            except Exception:
                pass

            # Refine run name using template or default descriptive pattern
            try:
                csv_base = Path(str(data_cfg.get("csv_path", ""))).stem
                split_method = split_cfg.get("method", "time")
                pos_tok = "co" if int(pos_label_cfg) == 0 else "fp"
                nf = int(X_train_np.shape[1])
                nc = int(len(feature_inputs))
                auc = float(metrics.get("roc_auc", float("nan")))
                layers_cfg = model_cfg_effective.get("layers")
                if isinstance(layers_cfg, (list, tuple)):
                    layers_str = "-".join(str(x) for x in layers_cfg)
                else:
                    layers_str = str(layers_cfg or "")
                leader_id = raw_result.get("leader_id") if isinstance(raw_result, dict) else None
                leader_algo = raw_result.get("leader_algo") if isinstance(raw_result, dict) else None
                algo_tok = str(leader_algo).replace(" ", "_") if leader_algo else "auto"
                try:
                    sha = _collect_env_metadata().get("git", {}).get("commit")
                except Exception:
                    sha = None
                base_ctx = {
                    "dataset": csv_base,
                    "split": split_method,
                    "pos": pos_tok,
                    "layers": layers_str,
                    "nf": nf,
                    "nc": nc,
                    "auc": auc,
                    "sha": sha or "",
                    "run_id": run_id,
                    "backend": backend.name,
                    "leader_id": leader_id or "",
                    "leader_algo": algo_tok,
                    "leader_algo_raw": leader_algo or "",
                }
                template = wb_cfg.get("run_name_template")
                name = None
                try:
                    name = backend.format_run_name(
                        base_context=dict(base_ctx),
                        training_result=training_result,
                        metrics=metrics,
                        run_context=run_context,
                        cfg=cfg,
                    )
                except Exception:
                    name = None
                if not name and template:
                    try:
                        name = str(template).format(**base_ctx)
                    except Exception:
                        name = None
                if not name:
                    if backend.name == "pytorch" and layers_str:
                        name = f"{csv_base}|{split_method}|{pos_tok}|mlp[{layers_str}]|nf{nf}|auc{auc:.3f}"
                    elif backend.name == "h2o" and leader_id:
                        leader_tok = str(leader_id).replace(" ", "_")
                        name = f"{csv_base}|{split_method}|{pos_tok}|h2o[{algo_tok}]|{leader_tok}|auc{auc:.3f}"
                    else:
                        name = f"{csv_base}|{split_method}|{pos_tok}|{backend.name}|auc{auc:.3f}"
                if len(name) > 120:
                    name = name[:120]
                wandb.run.name = name
                # Also render tag templates and add static tags
                try:
                    tags_cfg = wb_cfg.get("tags", []) or []
                    tag_tmps = wb_cfg.get("tag_templates", []) or []
                    rendered = []
                    for t in tag_tmps:
                        try:
                            rendered.append(str(t).format(**base_ctx))
                        except Exception:
                            continue
                    extra_tags = []
                    try:
                        extra_tags = list(
                            backend.additional_wandb_tags(
                                training_result=training_result,
                                run_context=run_context,
                                cfg=cfg,
                            )
                        )
                    except Exception:
                        extra_tags = []
                    current = list(wandb.run.tags or [])  # type: ignore[attr-defined]
                    wandb.run.tags = list({*current, *tags_cfg, *rendered, *extra_tags})  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception:
                pass

            # Log a lightweight artifact with key files
            if bool(wb_cfg.get("log_artifacts", True)):
                art = wandb.Artifact(name=run_id, type="run")
                if notes_text:
                    art.description = f"Run notes: {notes_text}"
                for rel in [
                    "metrics.json",
                    "confusion.json",
                    "config_resolved.yaml",
                    "requirements.freeze.txt",
                    "features.json",
                    "history.csv",
                    "training.log",
                    # include the model weights saved for this run
                    f"{model_path.name}",
                ]:
                    p = run_dir / rel
                    if p.exists():
                        art.add_file(p.as_posix(), name=rel)
                # Add README.md dynamically from in-memory content via temp file
                try:
                    import tempfile
                    with tempfile.TemporaryDirectory() as _td:
                        _rp = Path(_td) / "README.md"
                        _rp.write_text(readme_content, encoding="utf-8")
                        art.add_file(_rp.as_posix(), name="README.md")
                except Exception:
                    pass
                wandb.log_artifact(art)

            # Also log a versioned "model" artifact with metadata and aliases
            try:
                model_file = run_dir / model_path.name
                if model_file.exists():
                    model_art = wandb.Artifact(
                        name="loan-default",  # stable name; W&B versions it
                        type="model",
                        metadata={
                            "param_count": int(param_count) if param_count is not None else None,
                            "n_features": int(X_train_np.shape[1]),
                            "layers": model_cfg.get("layers"),
                            "dropout": model_cfg.get("dropout"),
                            "batchnorm": bool(model_cfg.get("batchnorm", True)),
                            "pos_label": int(pos_label_cfg),
                            "threshold_strategy": strategy,
                            "run_id": run_id,
                            # attach git and env pointers as lightweight metadata
                            **({f"env.{k}": v for k, v in (_collect_env_metadata().get("env", {}) or {}).items()}),
                            **({f"git.{k}": v for k, v in (_collect_env_metadata().get("git", {}) or {}).items()}),
                            "notes": notes_text if notes_text else None,
                        },
                    )
                    model_art.add_file(model_file.as_posix(), name=model_path.name)
                    wandb.log_artifact(model_art, aliases=[run_id, "latest"])
            except Exception:
                pass
        except Exception:
            pass

    # Persist basic W&B identifiers for downstream automation
    try:
        if wandb_enabled:
            wb_info = {"id": wb_id if 'wb_id' in locals() else None, "path": wb_path if 'wb_path' in locals() else None, "url": wb_url if 'wb_url' in locals() else None}
            with open((run_dir / "wandb.json"), "w", encoding="utf-8") as _wf:
                _json.dump(wb_info, _wf, indent=2)
    except Exception:
        pass

    # Clean up W&B run if open
    if wandb_enabled and wandb_run is not None:
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:
            pass

    elapsed = time.time() - t0
    # Add a simple threshold sanity note for extreme operating points
    try:
        if cm.get("precision", 1.0) < 1e-3 or cm.get("recall", 1.0) < 1e-3:
            summary_lines.append("\n> Note: Precision/Recall is near 0 at the chosen threshold. Consider revising the threshold strategy or dataset balance.")
    except Exception:
        pass

    result_payload = {
        "model_path": (run_dir / model_path.name).as_posix() if not single_run_dir_mode else model_path.as_posix(),
        "metrics_path": (run_dir / "metrics.json").as_posix() if not single_run_dir_mode else (reports_dir / "metrics.json").as_posix(),
        "figures_path": (run_fig_dir / "learning_curves.png").as_posix(),
        "roc_curve_path": (run_fig_dir / "roc_curve.png").as_posix(),
        "pr_curve_path": (run_fig_dir / "pr_curve.png").as_posix(),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "threshold": evaluation.threshold,
        "pos_label": pos_label_name,
        "backend": backend.name,
        "elapsed_sec": elapsed,
        "n_train": int(len(y_train_np)),
        "n_test": int(len(y_test_np)),
        "n_features": int(X_train_np.shape[1]),
        "run_summary_path": (run_dir / "README.md").as_posix(),
        "run_dir": run_dir.as_posix(),
        "wandb_run_path": (wb_path if wandb_enabled else None),
        "wandb_run_url": (wb_url if wandb_enabled else None),
        "leaderboard_path": raw_result.get("leaderboard_path") if isinstance(raw_result, dict) else None,
    }
    if cv_summary_for_return is not None:
        result_payload["cv_summary"] = cv_summary_for_return
    return result_payload
