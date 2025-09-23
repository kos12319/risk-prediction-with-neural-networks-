from __future__ import annotations

from typing import Any, Callable, Dict

import torch.nn as nn

from src.models.torch_nn import MLP


TorchBuilder = Callable[[int, Dict[str, Any]], nn.Module]


class TorchModelRegistry:
    def __init__(self) -> None:
        self._builders: Dict[str, TorchBuilder] = {}

    def register(self, name: str, builder: TorchBuilder) -> None:
        key = name.lower()
        if key in self._builders:
            raise ValueError(f"Torch model '{name}' already registered.")
        self._builders[key] = builder

    def get(self, name: str) -> TorchBuilder:
        key = name.lower()
        if key not in self._builders:
            raise KeyError(f"Torch model '{name}' is not registered.")
        return self._builders[key]

    def available(self) -> Dict[str, TorchBuilder]:
        return dict(self._builders)


_registry = TorchModelRegistry()


def register_torch_model(name: str) -> Callable[[TorchBuilder], TorchBuilder]:
    def decorator(fn: TorchBuilder) -> TorchBuilder:
        _registry.register(name, fn)
        return fn

    return decorator


@register_torch_model("mlp")
def _build_mlp(input_dim: int, cfg: Dict[str, Any]) -> nn.Module:
    layers = cfg.get("layers", [256, 128, 64, 32])
    dropout = cfg.get("dropout", [0.4, 0.3, 0.2, 0.2])
    batchnorm = cfg.get("batchnorm", True)
    return MLP(
        input_dim=input_dim,
        layers=list(layers) if isinstance(layers, (list, tuple)) else [int(layers)],
        dropout=list(dropout) if isinstance(dropout, (list, tuple)) else [float(dropout)],
        batchnorm=bool(batchnorm),
    )


def build_torch_model(input_dim: int, model_cfg: Dict[str, Any]) -> nn.Module:
    architecture = str(model_cfg.get("architecture", "mlp"))
    builder = _registry.get(architecture)
    return builder(input_dim, model_cfg)


def available_torch_models() -> Dict[str, TorchBuilder]:
    return _registry.available()
