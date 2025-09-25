from __future__ import annotations

from typing import Any, List

import numpy as np


def align_probabilities(y_prob: np.ndarray | List[float], prob_label: Any, pos_label: int) -> np.ndarray:
    """Return probabilities aligned to the configured positive label.

    Many backends emit the probability of class 1 by convention. Others allow
    choosing which class the probability refers to. This helper normalizes the
    emitted probabilities so that they always correspond to ``pos_label``.
    """
    probs = np.asarray(y_prob, dtype=float)
    try:
        label = int(prob_label)
    except Exception:
        label = 0 if str(prob_label).lower() in {"0", "charged off", "charged_off", "default"} else 1
    if label == int(pos_label):
        return probs
    return 1.0 - probs


__all__ = ["align_probabilities"]

