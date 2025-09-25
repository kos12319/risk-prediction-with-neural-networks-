from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.training.orchestrator import run_backend_pipeline
from .pipeline import TemplatePipeline


def train_from_config(cfg_path: str | Path, notes: Optional[str] = None):
    pipeline = TemplatePipeline()
    return run_backend_pipeline(cfg_path, backend=pipeline, notes=notes)


__all__ = ["train_from_config"]

