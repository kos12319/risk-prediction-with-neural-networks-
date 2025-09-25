from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from src.training.base_pipeline import (
    BackendPipeline,
    BackendTrainingResult,
    DatasetBundle,
    RunContext,
)
from src.training.train_nn import attach_wandb_hooks, clear_wandb_hooks, train_pytorch


class PyTorchPipeline(BackendPipeline):
    name = "pytorch"

    def validate_config(self, cfg: Dict[str, Any]) -> None:
        # Backend identity and schema validation (decoupled from shared layer)
        from .schema import validate_backend_config

        validate_backend_config(cfg)

    def resolve_model_path(
        self,
        *,
        out_cfg: Dict[str, Any],
        artifact_mgr,
        fold_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        filename = str(out_cfg.get("model_filename", "loan_default_model.pt"))
        if not filename.endswith(".pt"):
            filename = f"{Path(filename).stem}.pt"
        if fold_meta and "fold_id" in fold_meta:
            fold_id = int(fold_meta["fold_id"])
            base = Path(filename).stem
            filename = f"{base}_fold{fold_id:02d}.pt"
        return artifact_mgr.models_dir / filename

    def prepare_model_config(
        self,
        *,
        model_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        cfg = dict(model_cfg)
        class_weight_cfg = training_cfg.get("class_weight")
        if class_weight_cfg is None:
            return cfg
        weights: Optional[Dict[int, float]] = None
        if isinstance(class_weight_cfg, str) and class_weight_cfg.lower() == "auto":
            n = float(len(y_train))
            n1 = float((y_train == 1).sum())
            n0 = n - n1
            w0 = n / (2.0 * max(n0, 1.0))
            w1 = n / (2.0 * max(n1, 1.0))
            weights = {0: w0, 1: w1}
        elif isinstance(class_weight_cfg, dict):
            try:
                weights = {int(k): float(v) for k, v in class_weight_cfg.items()}
            except Exception:
                weights = None
        if weights is not None:
            cfg["_class_weight"] = weights
        return cfg

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
        clear_wandb_hooks()
        if wandb_enabled and wandb_run is not None:
            attach_wandb_hooks(model_cfg, wandb_run)

        result, history = train_pytorch(
            dataset.X_train,
            dataset.y_train,
            dataset.X_val,
            dataset.y_val,
            dataset.X_test,
            dataset.y_test,
            model_cfg,
            model_path,
            random_seed,
            pos_label,
        )

        if isinstance(result, dict):
            raw = dict(result)
            model_path_resolved = Path(raw.get("model_path", model_path.as_posix()))
            y_prob = np.asarray(raw.get("y_prob"))
            prob_label = int(raw.get("y_prob_label", 1))
        else:
            raw = {"y_prob": result}
            model_path_resolved = model_path
            y_prob = np.asarray(result)
            prob_label = 1

        return BackendTrainingResult(
            y_prob=y_prob,
            prob_label=prob_label,
            model_path=model_path_resolved,
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
        # Core pipeline already logs rich W&B assets; no PyTorch-specific extras yet.
        return None

    def extra_artifact_lines(
        self,
        *,
        training_result: BackendTrainingResult,
        run_context: RunContext,
        cfg: Dict[str, Any],
    ) -> Iterable[str]:
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
        # Prefer a descriptive MLP-style naming if layers are available
        try:
            ds = str(base_context.get("dataset", ""))
            split = str(base_context.get("split", ""))
            pos = str(base_context.get("pos", ""))
            layers = str(base_context.get("layers", "") or "")
            nf = base_context.get("nf")
            auc = float(base_context.get("auc", float("nan")))
            if layers:
                if nf is not None:
                    return f"{ds}|{split}|{pos}|mlp[{layers}]|nf{int(nf)}|auc{auc:.3f}"
                return f"{ds}|{split}|{pos}|mlp[{layers}]|auc{auc:.3f}"
        except Exception:
            pass
        return None

    def additional_wandb_tags(
        self,
        *,
        training_result: BackendTrainingResult,
        run_context: RunContext,
        cfg: Dict[str, Any],
    ) -> Iterable[str]:
        return []


def train_from_config(cfg_path: str | Path, notes: Optional[str] = None):
    pipeline = PyTorchPipeline()
    return pipeline.run(cfg_path, notes=notes)


__all__ = ["train_from_config", "PyTorchPipeline"]
