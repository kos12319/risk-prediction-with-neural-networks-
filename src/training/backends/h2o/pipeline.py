from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.training.base_pipeline import (
    BackendPipeline,
    BackendTrainingResult,
    DatasetBundle,
    RunContext,
)
from src.training.train_h2o import train_h2o

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
    return None


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _env_float_list(name: str) -> Optional[list[float]]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        parts = [seg.strip() for seg in str(raw).replace(";", ",").split(",") if seg.strip()]
        if not parts:
            return []
        return [float(seg) for seg in parts]
    except ValueError:
        return None


class H2OPipeline(BackendPipeline):
    name = "h2o"

    def validate_config(self, cfg: Dict[str, Any]) -> None:
        backend = str(cfg.get("model", {}).get("backend", "h2o")).lower()
        if backend != "h2o":
            raise ValueError("H2O AutoML pipeline requires model.backend to be 'h2o'")

    def apply_env_overrides(self, cfg: Dict[str, Any]) -> None:
        automl_cfg = cfg.setdefault("automl", {})
        balance_flag = _env_flag("H2O_BALANCE_CLASSES")
        if balance_flag is not None:
            automl_cfg["balance_classes"] = balance_flag
        max_after = _env_float("H2O_MAX_AFTER_BALANCE_SIZE")
        if max_after is not None:
            automl_cfg["max_after_balance_size"] = max_after
        class_factors = _env_float_list("H2O_CLASS_SAMPLING_FACTORS")
        if class_factors is not None:
            automl_cfg["class_sampling_factors"] = class_factors

    def resolve_model_path(
        self,
        *,
        out_cfg: Dict[str, Any],
        artifact_mgr,
        fold_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        filename = str(out_cfg.get("model_filename", "loan_default_model.zip"))
        if not filename.endswith(".zip"):
            filename = f"{Path(filename).stem}.zip"
        if fold_meta and "fold_id" in fold_meta:
            fold_id = int(fold_meta["fold_id"])
            base = Path(filename).stem
            filename = f"{base}_fold{fold_id:02d}.zip"
        return artifact_mgr.models_dir / filename

    def prepare_model_config(
        self,
        *,
        model_cfg: Dict[str, Any],
        training_cfg: Dict[str, Any],
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
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
        full_cfg: Dict[str, Any] = cfg or {}
        automl_cfg = full_cfg.get("automl", {})
        data_cfg = full_cfg.get("data", {})

        result, history = train_h2o(
            dataset.X_train,
            dataset.y_train,
            dataset.X_val,
            dataset.y_val,
            dataset.X_test,
            dataset.y_test,
            list(dataset.feature_names),
            data_cfg.get("target_col", "loan_status"),
            automl_cfg,
            model_path,
            run_context.run_dir,
            run_context.run_id,
            pos_label,
        )

        raw = dict(result)
        y_prob = np.asarray(raw.get("y_prob"))
        prob_label = int(raw.get("y_prob_label", 1))
        model_path_resolved = Path(raw.get("model_path", model_path.as_posix()))

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
        raw = training_result.raw
        leader_id = raw.get("leader_id") if isinstance(raw, dict) else None
        figures_dir = run_context.artifact_mgr.figures_dir
        run_dir = run_context.run_dir

        try:
            import wandb  # type: ignore

            lb_path = raw.get("leaderboard_path") if isinstance(raw, dict) else None
            if lb_path:
                lb_path_obj = Path(lb_path)
                if lb_path_obj.exists():
                    lb_df = pd.read_csv(lb_path_obj)
                    if not lb_df.empty:
                        wandb.log({"h2o_leaderboard": wandb.Table(dataframe=lb_df)})

            for metric in ["auc", "logloss", "rmse"]:
                fig_path = figures_dir / f"h2o_leaderboard_{metric}.png"
                if fig_path.exists():
                    wandb.log({f"h2o_leaderboard_{metric}": wandb.Image(str(fig_path))})

            comparison_dir = figures_dir / "comparison"
            for name in [
                "h2o_leaderboard_roc",
                "h2o_leaderboard_pr",
                "h2o_model_correlation",
                "h2o_varimp_heatmap",
                "h2o_pareto_front",
            ]:
                img_path = comparison_dir / f"{name}.png"
                if img_path.exists():
                    wandb.log({name: wandb.Image(str(img_path))})

            if wandb.run is not None:
                artifact = wandb.Artifact(name=f"h2o-analysis-{run_context.run_id}", type="analysis")
                if lb_path and Path(lb_path).exists():
                    artifact.add_file(lb_path, name="leaderboard/h2o_leaderboard.csv")
                for png in figures_dir.rglob("*.png"):
                    artifact.add_file(png.as_posix(), name=f"figures/{png.relative_to(run_dir)}")
                wandb.log_artifact(artifact)

            if leader_id:
                wandb.run.tags = list({*list(wandb.run.tags or []), f"leader:{leader_id}"})  # type: ignore[attr-defined]
        except Exception:
            pass

    def extra_artifact_lines(
        self,
        *,
        training_result: BackendTrainingResult,
        run_context: RunContext,
        cfg: Dict[str, Any],
    ) -> Iterable[str]:
        return [
            "- H2O leaderboard: `h2o_leaderboard.csv`",
            "- Per-family feature importance: `figures/comparison/per_family_varimp/`",
            "- Partial dependence plots: `figures/explanations/partial_dependence/`",
        ]

    def format_run_name(
        self,
        *,
        base_context: Dict[str, Any],
        training_result: BackendTrainingResult,
        metrics: Dict[str, Any],
        run_context: RunContext,
        cfg: Dict[str, Any],
    ) -> Optional[str]:
        raw = training_result.raw if isinstance(training_result.raw, dict) else {}
        leader_id = raw.get("leader_id")
        leader_algo = raw.get("leader_algo")
        if leader_id and leader_algo:
            return (
                f"{base_context['dataset']}|{base_context['split']}|{base_context['pos']}|"
                f"h2o[{str(leader_algo).replace(' ', '_')}]|{str(leader_id).replace(' ', '_')}|auc{base_context['auc']:.3f}"
            )
        return None

    def additional_wandb_tags(
        self,
        *,
        training_result: BackendTrainingResult,
        run_context: RunContext,
        cfg: Dict[str, Any],
    ) -> Iterable[str]:
        tags: list[str] = []
        raw = training_result.raw if isinstance(training_result.raw, dict) else {}
        leader_id = raw.get("leader_id")
        leader_algo = raw.get("leader_algo")
        if leader_id:
            tags.append(f"leader:{str(leader_id).replace(' ', '_')}")
        if leader_algo:
            tags.append(f"leader_algo:{str(leader_algo).replace(' ', '_')}")
        return tags


def train_from_config(cfg_path: str | Path, notes: Optional[str] = None):
    pipeline = H2OPipeline()
    return pipeline.run(cfg_path, notes=notes)


__all__ = ["train_from_config", "H2OPipeline"]
