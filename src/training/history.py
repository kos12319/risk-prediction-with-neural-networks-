from __future__ import annotations

from typing import List


class SimpleHistory:
    """Lightweight history container for plotting curves."""

    def __init__(self, loss: List[float], val_loss: List[float]):
        self.history = {"loss": loss, "val_loss": val_loss}
