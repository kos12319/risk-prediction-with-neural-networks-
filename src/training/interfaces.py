from __future__ import annotations

"""
Lightweight interfaces for backend pipelines.

Backends should import `BackendPipeline`, `DatasetBundle`, `BackendTrainingResult`,
and `RunContext` from this module instead of `src.training.base_pipeline`.
This indirection decouples backend modules from the shared pipeline
implementation so future refactors can move internals without touching
backend code again.
"""

# Re-export the current implementations from the shared pipeline. This keeps
# imports stable now and allows us to migrate definitions transparently later.
from src.training.base_pipeline import (  # noqa: F401
    BackendPipeline,
    BackendTrainingResult,
    DatasetBundle,
    RunContext,
)

__all__ = [
    "BackendPipeline",
    "BackendTrainingResult",
    "DatasetBundle",
    "RunContext",
]

