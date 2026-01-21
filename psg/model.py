from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int = 2
    hidden_dims: List[int] = None  # type: ignore[assignment]
    dropout: float = 0.0

    def __post_init__(self):
        if self.hidden_dims is None:
            object.__setattr__(self, "hidden_dims", [64, 64])


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig, num_classes: int = 2):
        super().__init__()
        dims = [cfg.input_dim] + list(cfg.hidden_dims) + [num_classes]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(p=cfg.dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
