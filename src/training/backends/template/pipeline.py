from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from src.training.interfaces import BackendPipeline, BackendTrainingResult, DatasetBundle, RunContext


class TemplatePipeline(BackendPipeline):
    """Minimal sklearn-based backend to demonstrate pluggability.

    Uses LogisticRegression to train quickly on CPU. Intended for smoke tests
    and as a starting point for new backends.
    """

    name = "template"

    def validate_config(self, cfg: Dict[str, Any]) -> None:
        # Accept any config that passes shared validation; backend does not
        # impose additional required fields. Users may optionally set
        # model.backend: template for clarity.
        return None

    def resolve_model_path(
        self,
        *,
        out_cfg: Dict[str, Any],
        artifact_mgr,
        fold_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        filename = str(out_cfg.get("model_filename", "model_template.pkl"))
        if not filename.endswith(".pkl"):
            filename = f"{Path(filename).stem}.pkl"
        if fold_meta and "fold_id" in fold_meta:
            fold_id = int(fold_meta["fold_id"])
            base = Path(filename).stem
            filename = f"{base}_fold{fold_id:02d}.pkl"
        return artifact_mgr.models_dir / filename

    def prepare_model_config(
        self,
        *,
        model_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        # Pass-through; allow overriding C and penalty via model_cfg
        return dict(model_cfg)

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
        from sklearn.linear_model import LogisticRegression  # type: ignore
        import joblib  # type: ignore

        C = float(model_cfg.get("C", 1.0))
        penalty = str(model_cfg.get("penalty", "l2"))
        max_iter = int(model_cfg.get("max_iter", 200))

        clf = LogisticRegression(
            random_state=int(random_seed),
            max_iter=max_iter,
            C=C,
            penalty=penalty,
            solver="lbfgs" if penalty == "l2" else "liblinear",
        )
        clf.fit(dataset.X_train, dataset.y_train.astype(int))

        # Predict probabilities for test (and val if available)
        y_prob_test = clf.predict_proba(dataset.X_test)[:, 1]
        y_prob_val = None
        if dataset.X_val is not None and dataset.y_val is not None:
            y_prob_val = clf.predict_proba(dataset.X_val)[:, 1]

        # Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_path)

        raw: Dict[str, Any] = {
            "y_prob": y_prob_test,
            "y_prob_label": 1,  # sklearn proba for column 1 corresponds to label 1
            "model_path": model_path.as_posix(),
        }
        if y_prob_val is not None:
            raw["y_prob_val"] = y_prob_val

        class _History:
            def __init__(self) -> None:
                self.history: Dict[str, Any] = {}
        history = _History()

        return BackendTrainingResult(
            y_prob=np.asarray(y_prob_test),
            prob_label=1,
            model_path=model_path,
            history=history,
            raw=raw,
        )

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
        # No extra logging beyond shared summaries
        return None

    def extra_artifact_lines(
        self,
        *,
        training_result: BackendTrainingResult,
        run_context: RunContext,
        cfg: Dict[str, Any],
    ) -> Iterable[str]:
        return []
