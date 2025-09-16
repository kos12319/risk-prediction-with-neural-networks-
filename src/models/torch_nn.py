from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers: List[int],
        dropout: Optional[List[float]] = None,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        modules: List[nn.Module] = []
        prev = input_dim
        dropout = dropout or [0.0] * len(layers)
        for i, units in enumerate(layers):
            modules.append(nn.Linear(prev, units))
            modules.append(nn.ReLU(inplace=True))
            if batchnorm:
                modules.append(nn.BatchNorm1d(units))
            dr = dropout[i] if i < len(dropout) else 0.0
            if dr and dr > 0:
                modules.append(nn.Dropout(p=dr))
            prev = units
        modules.append(nn.Linear(prev, 1))  # logits
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def focal_binary_loss(
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> nn.Module:
    # Focal loss with logits; reduction='mean'
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: shape (N,)
        targets = targets.view(-1, 1)
        bce_loss = bce(logits, targets)
        prob = torch.sigmoid(logits)
        pt = targets * prob + (1 - targets) * (1 - prob)
        alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
        modulating = torch.pow(1.0 - pt, gamma)
        loss = alpha_factor * modulating * bce_loss
        return loss.mean()

    return loss_fn  # type: ignore[return-value]

