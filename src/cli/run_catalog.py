from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunEntry:
    run_id: str
    path: str
    backend: Optional[str] = None
    group: Optional[str] = None
    created: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    confusion: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    model_files: Optional[List[Dict[str, Any]]] = None
    figures: Optional[List[str]] = None
    cv_report: Optional[str] = None


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _file_info(p: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"name": p.name, "path": p.as_posix()}
    try:
        st = p.stat()
        info["bytes"] = int(st.st_size)
        info["mtime"] = int(st.st_mtime)
    except Exception:
        pass
    return info


def _infer_backend(run_dir: Path, metrics: Optional[Dict[str, Any]]) -> Optional[str]:
    # Try metrics.json -> training.log -> folder name token
    if metrics and isinstance(metrics, dict):
        b = metrics.get("backend") or metrics.get("model_backend")
        if isinstance(b, str) and b:
            return b
    try:
        # Common group structure: local_runs/<dataset>|<split>|<pos>|<backend>/run_...
        group = run_dir.parent.name
        if group:
            parts = group.split("|")
            if parts:
                candidate = parts[-1]
                if candidate in {"pytorch", "h2o"}:
                    return candidate
    except Exception:
        pass
    return None


def _collect_run_entry(run_dir: Path) -> Optional[RunEntry]:
    if not run_dir.is_dir():
        return None
    run_id = run_dir.name
    metrics = _safe_load_json(run_dir / "metrics.json")
    confusion = _safe_load_json(run_dir / "confusion.json")
    data_manifest = _safe_load_json(run_dir / "data_manifest.json")
    cv_path = None
    # CV runs place cv_metrics.json under the run root in single-run mode
    if (run_dir / "cv_metrics.json").exists():
        cv_path = (run_dir / "cv_metrics.json").as_posix()
    # Model files (a few common names)
    model_files: List[Dict[str, Any]] = []
    for pat in ("*.pt", "*.zip", "*.model", "*.bin"):
        for p in run_dir.glob(pat):
            model_files.append(_file_info(p))
    # Figures
    figures: List[str] = []
    fig_dir = run_dir / "figures"
    if fig_dir.is_dir():
        for p in sorted(fig_dir.glob("*.png")):
            figures.append(p.name)

    created = None
    try:
        created = os.path.getmtime(run_dir)
        # Encode timestamp as ISO-like float seconds string for simplicity
        created = str(created)
    except Exception:
        created = None

    entry = RunEntry(
        run_id=run_id,
        path=run_dir.as_posix(),
        backend=_infer_backend(run_dir, metrics),
        group=run_dir.parent.name if run_dir.parent != run_dir else None,
        created=created,
        metrics=metrics,
        confusion=confusion,
        data=data_manifest,
        model_files=model_files or None,
        figures=figures or None,
        cv_report=cv_path,
    )
    return entry


def build_catalog(runs_root: Path) -> Dict[str, Any]:
    entries: List[RunEntry] = []
    # Discover run folders: local_runs/**/run_*
    for run_dir in sorted(runs_root.glob("**/run_*")):
        entry = _collect_run_entry(run_dir)
        if entry is not None:
            entries.append(entry)
    catalog = {
        "runs_root": runs_root.as_posix(),
        "count": len(entries),
        "runs": [asdict(e) for e in entries],
    }
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local_runs and emit a JSON run catalog")
    parser.add_argument("--runs-root", type=str, default="local_runs", help="Root folder for runs")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (defaults to <runs-root>/_catalog.json)",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.exists():
        raise SystemExit(f"Runs root not found: {runs_root}")

    catalog = build_catalog(runs_root)
    out_path = Path(args.out) if args.out else runs_root / "_catalog.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(json.dumps({"wrote": out_path.as_posix(), "count": catalog.get("count", 0)}, indent=2))


if __name__ == "__main__":
    main()

