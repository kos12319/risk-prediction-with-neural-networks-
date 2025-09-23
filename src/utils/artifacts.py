from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable


@dataclass
class ArtifactPaths:
    run_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    figures_run_dir: Path
    metrics_path: Path
    confusion_path: Path


class ArtifactManager:
    def __init__(
        self,
        *,
        run_dir: Path,
        models_dir: Path,
        reports_dir: Path,
        figures_dir: Path,
        single_run_mode: bool,
    ) -> None:
        self.single_run_mode = single_run_mode
        self.paths = self._prepare_paths(run_dir, models_dir, reports_dir, figures_dir)

    def _prepare_paths(
        self,
        run_dir: Path,
        models_dir: Path,
        reports_dir: Path,
        figures_dir: Path,
    ) -> ArtifactPaths:
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        run_dir.mkdir(parents=True, exist_ok=True)
        if self.single_run_mode:
            figures_run_dir = figures_dir
        else:
            figures_run_dir = run_dir / "figures"
            figures_run_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = reports_dir / "metrics.json"
        confusion_path = reports_dir / "confusion.json"

        return ArtifactPaths(
            run_dir=run_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            figures_run_dir=figures_run_dir,
            metrics_path=metrics_path,
            confusion_path=confusion_path,
        )

    # Paths -----------------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        return self.paths.run_dir

    @property
    def figures_dir(self) -> Path:
        return self.paths.figures_dir

    @property
    def figures_run_dir(self) -> Path:
        return self.paths.figures_run_dir

    @property
    def reports_dir(self) -> Path:
        return self.paths.reports_dir

    @property
    def models_dir(self) -> Path:
        return self.paths.models_dir

    @property
    def metrics_path(self) -> Path:
        return self.paths.metrics_path

    @property
    def confusion_path(self) -> Path:
        return self.paths.confusion_path

    # Writers ----------------------------------------------------------------

    def save_confusion(self, confusion: Dict[str, Any]) -> None:
        self._write_json(self.confusion_path, confusion)

    # Utilities ---------------------------------------------------------------

    def stage_run_artifacts(self, model_path: Path, figure_names: Iterable[str]) -> None:
        if self.single_run_mode:
            return
        for name in figure_names:
            src = self.figures_dir / name
            dst = self.figures_run_dir / name
            if src.exists():
                shutil.copy2(src, dst)
        if self.metrics_path.exists():
            shutil.copy2(self.metrics_path, self.run_dir / self.metrics_path.name)
        if self.confusion_path.exists():
            shutil.copy2(self.confusion_path, self.run_dir / self.confusion_path.name)
        if model_path.exists():
            shutil.copy2(model_path, self.run_dir / model_path.name)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
