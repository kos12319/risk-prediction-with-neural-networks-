from __future__ import annotations

import os
from typing import Optional, Dict, Any


_TRUE_SET = {"1", "true", "yes", "on", "enabled"}
_FALSE_SET = {"0", "false", "no", "off", "disabled"}


def env_flag(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    val = str(raw).strip().lower()
    if val in _TRUE_SET:
        return True
    if val in _FALSE_SET:
        return False
    return None


def apply_common_env_overrides(cfg: Dict[str, Any]) -> None:
    """Apply shared environment overrides to the config.

    Currently supports toggling oversampling via environment without touching
    backend-specific behavior.
    """
    os_cfg = cfg.setdefault("oversampling", {})
    os_enabled = env_flag("PIPELINE_OVERSAMPLING_ENABLED")
    if os_enabled is not None:
        os_cfg["enabled"] = os_enabled
    os_method = os.environ.get("PIPELINE_OVERSAMPLING_METHOD")
    if os_method:
        os_cfg["method"] = os_method


__all__ = ["env_flag", "apply_common_env_overrides"]

