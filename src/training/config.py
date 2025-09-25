from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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


def _bool(v: Any, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def validate_and_normalize_config(cfg: Dict[str, Any], *, cfg_path: Optional[Path] = None) -> Dict[str, Any]:
    """Validate backend-agnostic invariants and normalize minor fields.

    Backend-specific checks belong to each backend's pipeline validator.
    This layer ensures data/eval/split blocks are coherent and that file
    references resolve. It raises ``ValueError`` on inconsistencies that
    would lead to incorrect evaluation or crashes.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping (YAML -> dict)")

    cfg_norm = dict(cfg)

    # Model block (backend-specific validation is deferred to pipelines)
    model = cfg_norm.setdefault("model", {}) or {}
    backend = str(model.get("backend", "")).lower()

    # Data block
    data = cfg_norm.setdefault("data", {}) or {}
    csv_path_raw = str(data.get("csv_path", "")).strip()
    if not csv_path_raw:
        raise ValueError("data.csv_path is required and cannot be empty")
    csv_path = Path(csv_path_raw)
    if not csv_path.exists():
        # Try to resolve relative to config file if provided
        if cfg_path is not None and not csv_path.is_absolute():
            candidate = (cfg_path.parent / csv_path).resolve()
            if candidate.exists():
                data["csv_path"] = candidate.as_posix()
            else:
                raise ValueError(f"CSV path not found: {csv_path_raw} (resolved: {candidate})")
        else:
            raise ValueError(f"CSV path not found: {csv_path_raw}")

    # Target mapping must cover exactly two classes mapped to 0/1
    target_col = str(data.get("target_col", "")).strip()
    if not target_col:
        raise ValueError("data.target_col is required")
    tmap = data.get("target_mapping", {}) or {}
    if not isinstance(tmap, dict) or not tmap:
        raise ValueError("data.target_mapping must be a mapping of label->0/1")
    mapped_vals = {int(v) for v in tmap.values()}
    if mapped_vals - {0, 1}:
        raise ValueError("data.target_mapping values must be 0/1 only")
    if 0 not in mapped_vals or 1 not in mapped_vals:
        raise ValueError("data.target_mapping must map to both classes 0 and 1")

    # Evaluation: positive label is either 0 or 1
    eval_cfg = cfg_norm.setdefault("eval", {}) or {}
    pos_label = eval_cfg.get("pos_label", 1)
    if isinstance(pos_label, str):
        pos_label = 0 if pos_label.lower() in {"charged off", "charged_off", "default"} else 1
        eval_cfg["pos_label"] = pos_label
    if int(pos_label) not in {0, 1}:
        raise ValueError("eval.pos_label must be 0 or 1")

    # Threshold strategy
    thr = (eval_cfg.get("threshold") or {})
    strategy = str(thr.get("strategy", "youden_j")).lower()
    if strategy not in {"fixed", "youden_j", "f1"}:
        raise ValueError("eval.threshold.strategy must be one of: fixed|youden_j|f1")
    if strategy == "fixed":
        try:
            _ = float(thr.get("value", 0.5))
        except Exception:
            raise ValueError("eval.threshold.value must be a float when strategy=fixed")

    # Split
    split = cfg_norm.setdefault("split", {}) or {}
    method = str(split.get("method", "time")).lower()
    if method not in {"time", "random"}:
        raise ValueError("split.method must be 'time' or 'random'")
    if method == "time":
        time_col = str(split.get("time_col", "issue_d")).strip()
        if not time_col:
            raise ValueError("split.time_col is required when split.method=time")
        # Encourage parse_dates to include time_col for robust parsing
        parse_dates = data.setdefault("parse_dates", []) or []
        if time_col not in parse_dates:
            parse_dates.append(time_col)
            data["parse_dates"] = parse_dates
    # Temporal CV guardrails (backend-agnostic)
    cv = split.get("cv", {}) or {}
    if isinstance(cv, dict) and cv.get("enabled"):
        n_folds = int(cv.get("n_folds", 0))
        if n_folds < 2:
            raise ValueError("split.cv.enabled requires split.cv.n_folds >= 2")
        init_frac = float(cv.get("initial_train_fraction", 0.0))
        val_frac = float(cv.get("validation_fraction", 0.0))
        if not (0.0 < init_frac < 1.0):
            raise ValueError("split.cv.initial_train_fraction must be in (0, 1)")
        if not (0.0 < val_frac < 1.0):
            raise ValueError("split.cv.validation_fraction must be in (0, 1)")
        mode = str(cv.get("mode", "expanding")).lower()
        if mode not in {"expanding"}:
            raise ValueError("split.cv.mode must be 'expanding' (only supported mode)")

    # Oversampling policy: enabled applies to training subset only (handled in pipeline)
    cfg_norm.setdefault("oversampling", {}) or {}

    # Leakage guardrail: if requested, leakage_cols must be present (list)
    if bool(data.get("drop_leakage", False)):
        leakage_cols = data.get("leakage_cols")
        if not isinstance(leakage_cols, list) or not leakage_cols:
            raise ValueError("data.drop_leakage=true requires non-empty data.leakage_cols list")

    # Do not introduce backend-specific defaults here; backends own their extras.

    return cfg_norm


__all__ = ["load_config_with_extends", "validate_and_normalize_config"]
