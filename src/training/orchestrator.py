from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base_pipeline import BackendPipeline


def run_backend_pipeline(
    cfg_path: str | Path,
    *,
    backend: BackendPipeline,
    notes: Optional[str] = None,
):
    """Shim orchestrator that delegates to the legacy implementation.

    This indirection allows callers to import a stable entrypoint while we
    progressively move the underlying implementation out of base_pipeline
    without changing behavior.
    """
    # Lazy import to avoid circulars during refactor steps
    from .base_pipeline import _run_backend_pipeline as _impl  # type: ignore

    return _impl(Path(cfg_path), backend=backend, notes=notes)


__all__ = ["run_backend_pipeline"]

