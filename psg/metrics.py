from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch


def global_l2_norm(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    sq = None
    for t in tensors:
        if t is None:
            continue
        v = t.detach()
        if sq is None:
            sq = v.pow(2).sum()
        else:
            sq = sq + v.pow(2).sum()
    if sq is None:
        return torch.tensor(0.0)
    return sq.sqrt()


def global_l2_norm_sq(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    sq = None
    for t in tensors:
        if t is None:
            continue
        v = t.detach()
        if sq is None:
            sq = v.pow(2).sum()
        else:
            sq = sq + v.pow(2).sum()
    if sq is None:
        return torch.tensor(0.0)
    return sq


def displacement_and_sq(prev_params: list[torch.Tensor], params: Iterable[torch.nn.Parameter]) -> Tuple[torch.Tensor, torch.Tensor]:
    deltas = []
    deltas_sq = None
    for p_prev, p in zip(prev_params, params):
        d = (p.detach() - p_prev)
        deltas.append(d)
        s = d.pow(2).sum()
        deltas_sq = s if deltas_sq is None else (deltas_sq + s)
    if deltas_sq is None:
        z = torch.tensor(0.0)
        return z, z
    return deltas_sq.sqrt(), deltas_sq


@dataclass
class PSGPathStats:
    pl: float = 0.0
    ge: float = 0.0
    steps: int = 0

    def update(self, step_displacement_l2: float, lr: float, grad_norm_l2: float) -> None:
        self.pl += float(step_displacement_l2)
        self.ge += float((lr**2) * (grad_norm_l2**2))
        self.steps += 1
