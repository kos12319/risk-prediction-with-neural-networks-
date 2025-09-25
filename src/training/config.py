from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating inputs."""

    result = dict(base)
    for key, value in (override or {}).items():
        base_val = result.get(key)
        if isinstance(base_val, dict) and isinstance(value, dict):
            result[key] = _deep_merge(base_val, value)
        else:
            result[key] = value
    return result


def load_config_with_extends(cfg_path: Path) -> Dict[str, Any]:
    """Load a YAML config resolving ``extends`` chains relative to the file."""

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg: Dict[str, Any] = yaml.safe_load(handle) or {}

    extends = cfg.get("extends")
    if extends:
        candidate = cfg_path.parent / f"{extends}.yaml"
        base_path = candidate if candidate.exists() else Path(extends)
        base_cfg = load_config_with_extends(base_path)
        merged = _deep_merge(base_cfg, {k: v for k, v in cfg.items() if k != "extends"})
        return merged
    return cfg


__all__ = ["load_config_with_extends"]

