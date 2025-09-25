from __future__ import annotations

from pathlib import Path
from typing import Optional

from .pipeline import H2OPipeline


def train_from_config(cfg_path: str | Path, notes: Optional[str] = None):
    """
    Backend-owned orchestration entrypoint for H2O AutoML.

    This thin wrapper delegates to the current pipeline implementation.
    It exists to decouple external callers from the shared base pipeline
    so that future refactors can migrate orchestration into backend-owned
    modules without changing CLIs or call sites.
    """
    pipeline = H2OPipeline()
    return pipeline.run(cfg_path, notes=notes)


__all__ = ["train_from_config"]

