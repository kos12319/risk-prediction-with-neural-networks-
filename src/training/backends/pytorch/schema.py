from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, PositiveInt, ValidationError, field_validator


class TrainingSchema(BaseModel):
    epochs: Optional[PositiveInt] = Field(default=None)
    batch_size: Optional[PositiveInt] = Field(default=None)
    lr: Optional[float] = Field(default=None, gt=0)
    weight_decay: Optional[float] = Field(default=None, ge=0)
    class_weight: Optional[Union[str, Dict[int, float]]] = Field(default=None)

    @field_validator("class_weight")
    @classmethod
    def _validate_class_weight(cls, v):  # type: ignore[override]
        if v is None:
            return v
        if isinstance(v, str):
            if v.lower() != "auto":
                raise ValueError("class_weight must be 'auto' or a mapping of {0: w0, 1: w1}")
            return v
        if isinstance(v, dict):
            casted: Dict[int, float] = {}
            for k, val in v.items():
                try:
                    ki = int(k)
                except Exception as e:  # pragma: no cover - defensive
                    raise ValueError("class_weight keys must be 0/1") from e
                if ki not in (0, 1):
                    raise ValueError("class_weight keys must be 0 or 1")
                try:
                    fv = float(val)
                except Exception as e:  # pragma: no cover - defensive
                    raise ValueError("class_weight values must be numeric") from e
                if fv <= 0:
                    raise ValueError("class_weight values must be > 0")
                casted[ki] = fv
            return casted
        raise ValueError("class_weight must be 'auto' or a mapping")


class ModelSchema(BaseModel):
    backend: str = Field(default="pytorch")
    # Minimal architecture hints for naming; fields are optional to stay permissive
    layers: Optional[List[int]] = Field(default=None)
    n_features: Optional[int] = Field(default=None, alias="n_features")


class PyTorchConfigSchema(BaseModel):
    model: ModelSchema
    training: Optional[TrainingSchema] = None

    @classmethod
    def validate_backend(cls, data: dict) -> None:
        backend = str(((data or {}).get("model") or {}).get("backend", "pytorch")).lower()
        if backend not in {"", "pytorch"}:
            # Keep message aligned with tests expecting 'pipeline' wording
            raise ValueError("PyTorch pipeline requires model.backend to be 'pytorch' (or omitted)")


def validate_backend_config(cfg: dict) -> None:
    """Validate PyTorch-specific config using Pydantic (non-intrusive).

    This enforces only backend-specific concerns and stays permissive to
    avoid breaking existing presets. Raises ValueError on validation failure.
    """
    try:
        PyTorchConfigSchema.validate_backend(cfg)
        PyTorchConfigSchema.model_validate(cfg)
    except ValidationError as ve:  # pragma: no cover - thin wrapper
        raise ValueError(f"Invalid PyTorch config: {ve}") from ve


__all__ = ["validate_backend_config"]
