from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Mapping, Optional

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


logger = logging.getLogger(__name__)


@dataclass
class ResampleResult:
    X_resampled: np.ndarray
    y_resampled: np.ndarray
    method: str
    params: Dict[str, Any]
    before_counts: Dict[int, int]
    after_counts: Dict[int, int]


_DEFAULT_METHOD = "random_over_sampler"


def apply_resampling(
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: Optional[str] = None,
    random_state: int = 42,
    params: Optional[Mapping[str, Any]] = None,
) -> ResampleResult:
    """Apply the configured resampling strategy to the training arrays."""

    method_key = (method or _DEFAULT_METHOD).lower()
    strategy = _resolve_strategy(method_key, random_state, params or {})

    before_counts = _class_counts(y)
    X_res, y_res = strategy.fit_resample(X, y)
    after_counts = _class_counts(y_res)
    logger.info("Applied resampling '%s' -> before: %s, after: %s", method_key, before_counts, after_counts)

    return ResampleResult(
        X_resampled=np.asarray(X_res),
        y_resampled=np.asarray(y_res),
        method=method_key,
        params=dict(params or {}),
        before_counts=before_counts,
        after_counts=after_counts,
    )


def available_resamplers() -> Dict[str, str]:
    """Return human-readable names for supported resamplers."""

    return {
        "none": "No resampling",
        "random_over_sampler": "RandomOverSampler",
        "random_under_sampler": "RandomUnderSampler",
        "smote": "Synthetic Minority Over-sampling Technique",
        "smote_tomek": "SMOTE + Tomek Links",
        "smote_enn": "SMOTEENN",
    }


def _resolve_strategy(method: str, random_state: int, params: Mapping[str, Any]):
    normalized = method.replace("-", "_")
    if normalized in {"none", "off", "disabled"}:
        return _PassthroughResampler()
    if normalized == "random_over_sampler":
        return RandomOverSampler(random_state=random_state, **params)
    if normalized == "random_under_sampler":
        return RandomUnderSampler(random_state=random_state, **params)
    if normalized == "smote":
        return SMOTE(random_state=random_state, **params)
    if normalized in {"smote_tomek", "smotetomek"}:
        return SMOTETomek(random_state=random_state, **params)
    if normalized in {"smote_enn", "smoteenn"}:
        return SMOTEENN(random_state=random_state, **params)
    raise ValueError(f"Unsupported resampling method '{method}'.")


def _class_counts(y: np.ndarray) -> Dict[int, int]:
    labels, counts = np.unique(y.astype(int), return_counts=True)
    return {int(k): int(v) for k, v in zip(labels, counts)}


class _PassthroughResampler:
    """Adapter matching the ``fit_resample`` API but leaving data unchanged."""

    def fit_resample(self, X, y):  # noqa: D401 - external API match
        return X, y
