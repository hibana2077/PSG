from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
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

    out_dir = ensure_dir(Path(args.out_dir) / f"exp_c_{timestamp()}")
    save_json(out_dir / "meta.json", {"args": vars(args), "env": env_info()})

    results: List[Dict[str, Any]] = []

    grid = list(product(args.n_trains, args.label_noises, args.input_noises, args.lambdas))

    for n_train, label_noise, input_noise, lam in tqdm(grid, desc="stress grid"):
        data_cfg = MoonsConfig(
            n_train=int(n_train),
            n_test=int(args.n_test),
            noise=float(args.data_noise),
            label_noise=float(label_noise),
            seed=args.seed,
            standardize=True,
            input_noise_std=float(input_noise),
        )
        train_ds, test_ds, _ = make_moons_datasets(data_cfg)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = MLP(MLPConfig(hidden_dims=[args.hidden, args.hidden], dropout=args.dropout))

        cfg = TrainConfig(
            optimizer=args.optimizer,  # type: ignore[arg-type]
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            grad_clip=args.grad_clip,
            schedule=args.schedule,  # type: ignore[arg-type]
            path_shrink_lambda=float(lam),
            input_noise_std=float(input_noise),
        )

        out = train_one_run(model, train_loader, test_loader, cfg=cfg, device=device)

        results.append(
            {
                "n_train": int(n_train),
                "label_noise": float(label_noise),
                "input_noise_std": float(input_noise),
                "lambda": float(lam),
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
        )

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "results.csv", index=False)

    corr = corr_table(df, cols=["pl", "ge", "label_noise", "n_train", "input_noise_std", "lambda"], target="gap_loss")
    corr.to_csv(out_dir / "corr_gap_loss.csv", index=False)

    scatter(df, x="pl", y="gap_loss", hue="lambda", out_path=out_dir / "scatter_pl_gap_loss_by_lambda.png")
    scatter(df, x="label_noise", y="gap_loss", hue="lambda", out_path=out_dir / "scatter_labelnoise_gap_loss.png")
    scatter(df, x="n_train", y="gap_loss", hue="lambda", out_path=out_dir / "scatter_ntrain_gap_loss.png")

    print(f"Wrote: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PSG Exp-C: stress tests (noise, small-n, input perturbations)")
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n-test", type=int, default=1024)
    p.add_argument("--data-noise", type=float, default=0.2)

    p.add_argument("--n-trains", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--label-noises", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4])
    p.add_argument("--input-noises", type=float, nargs="+", default=[0.0, 0.2])
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.2])

    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    p.add_argument("--schedule", type=str, default="cosine", choices=["constant", "cosine", "step"])
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--grad-clip", type=float, default=0.0)

    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
