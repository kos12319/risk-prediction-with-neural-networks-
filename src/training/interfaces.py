from __future__ import annotations

"""
Interfaces and shared datatypes for backend pipelines.

Backends must import `BackendPipeline`, `DatasetBundle`, `BackendTrainingResult`,
`RunContext`, and (optionally) `TrainingRunResult` from this module.
This file contains the canonical definitions so backends never depend on
implementation details in other modules.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence
from abc import ABC, abstractmethod

import numpy as np

from src.utils.artifacts import ArtifactManager


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


@dataclass
class TrainingRunResult:
    """Bundle summary for a single training/evaluation run (fold or holdout)."""

    run_id: str
    evaluation: Any
    metrics: Dict[str, Any]
    confusion: Dict[str, Any]
    model_path: Path
    durations: Dict[str, float]
    fold_meta: Optional[Dict[str, Any]] = None


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

    def format_group_name(
        self,
        *,
        base_context: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Optional[str]:
        """Optional override for group name used in local runs and W&B."""
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

    # Default runner wiring (lazy import to avoid circulars)
    def run(self, cfg_path: str | Path, *, notes: Optional[str] = None):
        from src.training.base_pipeline import _run_backend_pipeline as _impl  # type: ignore

        return _impl(cfg_path, backend=self, notes=notes)


__all__ = [
    "BackendPipeline",
    "BackendTrainingResult",
    "DatasetBundle",
    "RunContext",
    "TrainingRunResult",
]
