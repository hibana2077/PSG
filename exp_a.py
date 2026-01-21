from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from psg.analysis import corr_table
from psg.data import MoonsConfig, make_moons_datasets
from psg.model import MLP, MLPConfig
from psg.plots import scatter
from psg.train import TrainConfig, train_one_run
from psg.utils import ensure_dir, env_info, pick_device, save_json, seed_everything, timestamp


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = pick_device(args.device)

    out_dir = ensure_dir(Path(args.out_dir) / f"exp_a_{timestamp()}")
    save_json(out_dir / "meta.json", {"args": vars(args), "env": env_info()})

    data_cfg = MoonsConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        noise=args.data_noise,
        label_noise=0.0,
        seed=args.seed,
        standardize=True,
        input_noise_std=0.0,
    )
    train_ds, test_ds, _ = make_moons_datasets(data_cfg)

    results: List[Dict[str, Any]] = []

    optimizers = ["sgd", "adam"] if args.optimizers == "all" else [args.optimizers]
    schedules = ["constant", "cosine", "step"] if args.schedules == "all" else [args.schedules]

    grid = list(
        product(
            optimizers,
            [float(x) for x in args.lrs],
            schedules,
            [float(x) for x in args.weight_decays],
            [int(x) for x in args.batch_sizes],
            [float(x) for x in args.grad_clips],
        )
    )

    for opt_name, lr, sched, wd, bs, clip in tqdm(grid, desc="sweep"):
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

        model = MLP(MLPConfig(hidden_dims=[args.hidden, args.hidden], dropout=args.dropout))

        cfg = TrainConfig(
            optimizer=opt_name,  # type: ignore[arg-type]
            lr=lr,
            momentum=args.momentum,
            weight_decay=wd,
            batch_size=bs,
            epochs=args.epochs,
            grad_clip=clip,
            schedule=sched,  # type: ignore[arg-type]
            path_shrink_lambda=0.0,
            input_noise_std=0.0,
        )

        out = train_one_run(model, train_loader, test_loader, cfg=cfg, device=device)

        row: Dict[str, Any] = {
            "optimizer": opt_name,
            "lr": lr,
            "schedule": sched,
            "weight_decay": wd,
            "batch_size": bs,
            "grad_clip": clip,
            "epochs": args.epochs,
            "train_loss": out["train"]["loss"],
            "train_acc": out["train"]["acc"],
            "test_loss": out["test"]["loss"],
            "test_acc": out["test"]["acc"],
            "gap_loss": out["gap_loss"],
            "gap_acc": out["gap_acc"],
            "pl": out["pl"],
            "ge": out["ge"],
            "steps": out["steps"],
        }
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "results.csv", index=False)

    corr = corr_table(df, cols=["pl", "ge", "lr", "weight_decay", "batch_size", "grad_clip"], target="gap_loss")
    corr.to_csv(out_dir / "corr_gap_loss.csv", index=False)

    scatter(df, x="pl", y="gap_loss", hue="optimizer", out_path=out_dir / "scatter_pl_gap_loss.png")
    scatter(df, x="ge", y="gap_loss", hue="optimizer", out_path=out_dir / "scatter_ge_gap_loss.png")
    scatter(df, x="pl", y="test_loss", hue="optimizer", out_path=out_dir / "scatter_pl_test_loss.png")

    print(f"Wrote: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PSG Exp-A: correlation sweep on two-moons")
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n-train", type=int, default=1024)
    p.add_argument("--n-test", type=int, default=1024)
    p.add_argument("--data-noise", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)

    p.add_argument("--optimizers", type=str, default="all", choices=["all", "sgd", "adam"])
    p.add_argument("--schedules", type=str, default="all", choices=["all", "constant", "cosine", "step"])
    p.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    p.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-4, 1e-3])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128])
    p.add_argument("--grad-clips", type=float, nargs="+", default=[0.0, 1.0])

    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
