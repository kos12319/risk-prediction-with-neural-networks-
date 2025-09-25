from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError


class AutoMLSchema(BaseModel):
    balance_classes: Optional[bool] = Field(default=None)
    max_after_balance_size: Optional[float] = Field(default=None, gt=0)
    class_sampling_factors: Optional[List[float]] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    max_runtime_secs: Optional[int] = Field(default=None, gt=0)
    max_models: Optional[int] = Field(default=None, gt=0)
    include_algos: Optional[List[str]] = None
    exclude_algos: Optional[List[str]] = None
    nthreads: Optional[int] = None
    log_level: Optional[str] = None
    progress: Optional[bool] = None


class ModelSchema(BaseModel):
    # Optional to allow CLI-implicit backend; validated separately
    backend: Optional[str] = Field(default=None)


class H2OConfigSchema(BaseModel):
    model: ModelSchema
    automl: Optional[AutoMLSchema] = None

    @classmethod
    def validate_backend(cls, data: dict) -> None:
        # Allow omission when using the H2O-specific CLI; treat missing as 'h2o'
        backend = str(((data or {}).get("model") or {}).get("backend", "h2o")).lower()
        if backend not in {"", "h2o"}:
            raise ValueError("H2O backend requires model.backend to be 'h2o' (or omitted)")


def validate_backend_config(cfg: dict) -> None:
    """Validate H2O-specific config using Pydantic.

    Keeps validation scoped to H2O AutoML options and does not alter
    shared data/eval/split invariants. Raises ValueError on failure.
    """
    try:
        H2OConfigSchema.validate_backend(cfg)
        H2OConfigSchema.model_validate(cfg)
    except ValidationError as ve:  # pragma: no cover - thin wrapper
        raise ValueError(f"Invalid H2O config: {ve}") from ve


__all__ = ["validate_backend_config"]
