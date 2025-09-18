from __future__ import annotations

import os
import random
from typing import Callable, Optional

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # no-op if CUDA not available
        # For deterministic behavior where feasible
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    except Exception:
        pass


def make_torch_generator(seed: int):
    """Return a seeded torch.Generator if torch is available, else None."""
    try:
        import torch  # type: ignore

        g = torch.Generator()
        g.manual_seed(int(seed))
        return g
    except Exception:
        return None


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Return a DataLoader worker_init_fn that seeds NumPy/Python per worker."""

    def _init_fn(worker_id: int) -> None:
        seed = int(base_seed) + int(worker_id or 0)
        random.seed(seed)
        np.random.seed(seed)

    return _init_fn

