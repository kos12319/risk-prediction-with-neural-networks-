from __future__ import annotations

# Public entrypoint is routed via the backend-owned runner
from .runner import train_from_config

__all__ = ["train_from_config"]

