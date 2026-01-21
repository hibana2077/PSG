from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .metrics import PSGPathStats, displacement_and_sq, global_l2_norm


OptimName = Literal["sgd", "adam"]
SchedName = Literal["constant", "cosine", "step"]


@dataclass(frozen=True)
class TrainConfig:
    optimizer: OptimName = "sgd"
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 50
    grad_clip: float = 0.0
    schedule: SchedName = "constant"
    step_gamma: float = 0.2
    step_milestones: Tuple[float, float] = (0.6, 0.85)  # as fractions of total steps

    # PSG intervention
    path_shrink_lambda: float = 0.0  # shrink step displacement by 1/(1+2*lambda)

    # data noise during training
    input_noise_std: float = 0.0


def make_optimizer(cfg: TrainConfig, model: nn.Module) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def make_scheduler(cfg: TrainConfig, optimizer: Optimizer, total_steps: int):
    if cfg.schedule == "constant":
        return None

    if cfg.schedule == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda s: 0.5 * (1.0 + np.cos(np.pi * min(s, total_steps) / total_steps)),
        )

    if cfg.schedule == "step":
        m1 = int(cfg.step_milestones[0] * total_steps)
        m2 = int(cfg.step_milestones[1] * total_steps)

        def lr_lambda(s: int) -> float:
            if s < m1:
                return 1.0
            if s < m2:
                return cfg.step_gamma
            return cfg.step_gamma * cfg.step_gamma

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unknown schedule: {cfg.schedule}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        total_loss += float(ce(logits, y).item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


@torch.no_grad()
def _apply_path_shrink(model: nn.Module, prev_params: list[torch.Tensor], lam: float) -> None:
    if lam <= 0:
        return
    scale = 1.0 / (1.0 + 2.0 * lam)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    for p_prev, p in zip(prev_params, trainable_params):
        p.copy_(p_prev + (p - p_prev) * scale)


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: str,
    log_every: int = 0,
) -> Dict[str, Any]:
    model.to(device)
    model.train()

    optimizer = make_optimizer(cfg, model)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = make_scheduler(cfg, optimizer, total_steps=total_steps)

    loss_fn = nn.CrossEntropyLoss()
    path = PSGPathStats()

    step = 0
    for _epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            if cfg.input_noise_std and cfg.input_noise_std > 0:
                x = x + cfg.input_noise_std * torch.randn_like(x)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            prev_params = [p.detach().clone() for p in trainable_params]

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            grad_norm = float(global_l2_norm(p.grad for p in trainable_params).item())

            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=cfg.grad_clip
                )

            optimizer.step()

            # apply PSG intervention (post-step shrink on displacement)
            _apply_path_shrink(model, prev_params, cfg.path_shrink_lambda)

            # measure displacement after (possibly shrunk) step
            step_disp, _ = displacement_and_sq(prev_params, trainable_params)
            lr_now = float(optimizer.param_groups[0]["lr"])
            path.update(step_displacement_l2=float(step_disp.item()), lr=lr_now, grad_norm_l2=grad_norm)

            if scheduler is not None:
                scheduler.step()

            step += 1
            if log_every and (step % log_every == 0):
                pass

    train_metrics = evaluate(model, train_loader, device=device)
    test_metrics = evaluate(model, test_loader, device=device)

    return {
        "train": train_metrics,
        "test": test_metrics,
        "gap_loss": float(test_metrics["loss"] - train_metrics["loss"]),
        "gap_acc": float(train_metrics["acc"] - test_metrics["acc"]),
        "pl": float(path.pl),
        "ge": float(path.ge),
        "steps": int(path.steps),
    }
