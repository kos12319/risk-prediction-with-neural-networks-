from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def login_from_env() -> bool:
    """Login to W&B using WANDB_API_KEY from environment. Returns True on success."""
    try:
        import wandb  # type: ignore
        key = os.environ.get("WANDB_API_KEY") or os.environ.get("WB_API_KEY")
        if not key:
            return False
        try:
            wandb.login(key=key, relogin=True)
            return True
        except Exception:
            return False
    except Exception:
        return False


def _resolve_run_path(path_or_id: str, default_entity: Optional[str], default_project: Optional[str]) -> str:
    # Accept forms: entity/project/id, project/id (with default entity), or raw id (with defaults)
    parts = [p for p in path_or_id.strip("/").split("/") if p]
    if len(parts) == 3:
        return "/".join(parts)
    if len(parts) == 2:
        if default_entity:
            return f"{default_entity}/{parts[0]}/{parts[1]}"
        raise ValueError("Run path missing entity. Provide entity via WANDB_ENTITY or use entity/project/run_id")
    if len(parts) == 1 and default_entity and default_project:
        return f"{default_entity}/{default_project}/{parts[0]}"
    raise ValueError("Cannot resolve run path. Expected entity/project/run_id or set WANDB_ENTITY and WANDB_PROJECT")


def download_run(
    path_or_id: str,
    target_dir: str | Path,
    *,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    skip_existing: bool = False,
) -> dict:
    """Download all files and logged/used artifacts for a W&B run into target_dir.

    Returns a summary dict with keys: run_path, files_downloaded, artifacts_downloaded.
    """
    from wandb.apis.public import Api  # type: ignore

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    entity_env = entity or os.environ.get("WANDB_ENTITY") or os.environ.get("WB_ENTITY")
    project_env = project or os.environ.get("WANDB_PROJECT")
    run_path = _resolve_run_path(path_or_id, entity_env, project_env)

    api = Api()
    run = api.run(run_path)

    files_downloaded = []
    for f in run.files():
        try:
            # If skip_existing is True, avoid replacing existing files to speed up pulls
            f.download(root=str(target), replace=not bool(skip_existing))
            files_downloaded.append(f.name)
        except Exception:
            continue

    artifacts_downloaded = []
    # Logged artifacts
    try:
        for art in run.logged_artifacts():
            try:
                art_dir = target / "artifacts" / (art.name.replace("/", "_"))
                # Skip downloading artifact contents if target exists and skip_existing requested
                if skip_existing and art_dir.exists() and any(art_dir.iterdir()):
                    artifacts_downloaded.append(art.name)
                    continue
                art.download(root=str(art_dir))
                artifacts_downloaded.append(art.name)
            except Exception:
                continue
    except Exception:
        pass

    # Used artifacts (e.g., datasets)
    try:
        for art in run.used_artifacts():
            try:
                art_dir = target / "artifacts_used" / (art.name.replace("/", "_"))
                if skip_existing and art_dir.exists() and any(art_dir.iterdir()):
                    artifacts_downloaded.append(art.name)
                    continue
                art.download(root=str(art_dir))
                artifacts_downloaded.append(art.name)
            except Exception:
                continue
    except Exception:
        pass

    return {"run_path": run_path, "files_downloaded": files_downloaded, "artifacts_downloaded": artifacts_downloaded}
