from __future__ import annotations

from pathlib import Path
from typing import Optional

from .pipeline import TemplatePipeline


def train_from_config(cfg_path: str | Path, notes: Optional[str] = None):
    pipeline = TemplatePipeline()
    return pipeline.run(cfg_path, notes=notes)


__all__ = ["train_from_config"]
