from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class MoonsConfig:
    n_train: int = 1024
    n_test: int = 1024
    noise: float = 0.2
    label_noise: float = 0.0  # fraction to flip
    standardize: bool = True
    seed: int = 0
    input_noise_std: float = 0.0  # added during training only


def _flip_labels(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac <= 0:
        return y
    rng = np.random.default_rng(seed)
    y = y.copy()
    n = len(y)
    k = int(round(frac * n))
    idx = rng.choice(n, size=k, replace=False)
    y[idx] = 1 - y[idx]
    return y


def make_moons_datasets(cfg: MoonsConfig) -> Tuple[TensorDataset, TensorDataset, StandardScaler | None]:
    n_total = int(cfg.n_train + cfg.n_test)
    x, y = make_moons(n_samples=n_total, noise=cfg.noise, random_state=cfg.seed)
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.n_test, train_size=cfg.n_train, random_state=cfg.seed, stratify=y
    )

    y_train = _flip_labels(y_train, cfg.label_noise, seed=cfg.seed + 13)

    scaler: StandardScaler | None = None
    if cfg.standardize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return train_ds, test_ds, scaler
