from __future__ import annotations

"""
Backend-owned train wrapper for H2O AutoML.

This module exists to decouple the H2O backend from the legacy
top-level implementation location. It re-exports the current
implementation so callers import from the backend namespace:

    from src.training.backends.h2o.train import train_h2o

Future refactors can migrate the implementation here without
changing pipeline imports.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

from src.training.train_h2o import train_h2o as _train_h2o_impl
from src.training.history import SimpleHistory


def train_h2o(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_val_np: Optional[np.ndarray],
    y_val_np: Optional[np.ndarray],
    X_test_np: np.ndarray,
    y_test_np: np.ndarray,
    feature_names: List[str],
    target_name: str,
    automl_cfg: Dict[str, Any],
    model_path: Path,
    run_dir: Path,
    run_id: str,
    pos_label: int,
) -> Tuple[Dict[str, Any], SimpleHistory]:
    return _train_h2o_impl(
        X_train_np,
        y_train_np,
        X_val_np,
        y_val_np,
        X_test_np,
        y_test_np,
        feature_names,
        target_name,
        automl_cfg,
        model_path,
        run_dir,
        run_id,
        pos_label,
    )


__all__ = ["train_h2o"]

