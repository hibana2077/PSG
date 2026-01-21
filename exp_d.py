from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from psg.data import MoonsConfig, make_moons_datasets
from psg.model import MLP, MLPConfig
from psg.train import TrainConfig, train_one_run
from psg.utils import ensure_dir, env_info, pick_device, save_json, seed_everything, timestamp


def _flatten_params(model: torch.nn.Module) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        parts.append(p.detach().reshape(-1).to(dtype=torch.float32, device="cpu"))
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts, dim=0)


def _make_loaders(train_ds, test_ds, batch_size: int, shuffle_seed: int) -> Tuple[DataLoader, DataLoader]:
    g = torch.Generator()
    g.manual_seed(int(shuffle_seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _run_one_condition(
    *,
    condition: str,
    init_state: Dict[str, torch.Tensor],
    model_cfg: MLPConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: str,
    seed: int,
    save_dir: Path,
    save_weights: bool,
) -> Dict[str, Any]:
    seed_everything(seed)

    model = MLP(model_cfg)
    model.load_state_dict(init_state)

    out = train_one_run(
        model,
        train_loader,
        test_loader,
        cfg=cfg,
        device=device,
        return_path_profile=True,
    )

    w = _flatten_params(model)
    w_norm = float(torch.linalg.vector_norm(w).item()) if w.numel() else 0.0

    if save_weights:
        torch.save({"w": w}, save_dir / f"weights_{condition}_seed{seed}.pt")

    prof = out.get("path_profile", None)
    if prof is not None:
        pd.DataFrame(
            {
                "step": list(range(1, len(prof["step_disp_l2"]) + 1)),
                "step_disp_l2": prof["step_disp_l2"],
                "lr": prof["lr"],
                "grad_norm_l2": prof["grad_norm_l2"],
            }
        ).to_csv(save_dir / f"path_profile_{condition}_seed{seed}.csv", index=False)

    return {
        "condition": condition,
        "train_loss": out["train"]["loss"],
        "train_acc": out["train"]["acc"],
        "test_loss": out["test"]["loss"],
        "test_acc": out["test"]["acc"],
        "gap_loss": out["gap_loss"],
        "gap_acc": out["gap_acc"],
        "pl": out["pl"],
        "ge": out["ge"],
        "steps": out["steps"],
        "avg_lr": out.get("avg_lr", float("nan")),
        "avg_grad_norm": out.get("avg_grad_norm", float("nan")),
        "w_norm": w_norm,
        "w": w,
    }


def _pick_aligned(
    candidates: List[Dict[str, Any]],
    target_train_loss: float,
    target_w: torch.Tensor,
    loss_tol: float,
) -> Dict[str, Any]:
    for c in candidates:
        c["train_loss_abs_err"] = abs(float(c["train_loss"]) - float(target_train_loss))
        w = c["w"]
        c["endpoint_l2"] = float(torch.linalg.vector_norm(w - target_w).item()) if w.numel() else 0.0

    within = [c for c in candidates if c["train_loss_abs_err"] <= loss_tol]
    pool = within if within else candidates

    # primary: minimize endpoint distance, secondary: minimize train-loss mismatch
    pool = sorted(pool, key=lambda c: (c["endpoint_l2"], c["train_loss_abs_err"]))
    return pool[0]


def run(args: argparse.Namespace) -> None:
    device = pick_device(args.device)

    out_dir = ensure_dir(Path(args.out_dir) / f"exp_d_{timestamp()}")
    save_json(out_dir / "meta.json", {"args": vars(args), "env": env_info()})

    rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for seed in tqdm(list(args.seeds), desc="seeds"):
        seed = int(seed)
        seed_everything(seed)

        data_cfg = MoonsConfig(
            n_train=args.n_train,
            n_test=args.n_test,
            noise=args.data_noise,
            label_noise=args.label_noise,
            seed=seed,
            standardize=True,
            input_noise_std=0.0,
        )
        train_ds, test_ds, _ = make_moons_datasets(data_cfg)

        # lock data order across conditions
        train_loader, test_loader = _make_loaders(
            train_ds,
            test_ds,
            batch_size=args.batch_size,
            shuffle_seed=seed + 777,
        )

        model_cfg = MLPConfig(hidden_dims=[args.hidden, args.hidden], dropout=args.dropout)
        init_model = MLP(model_cfg)
        init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}

        base_cfg = TrainConfig(
            optimizer=args.optimizer,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            grad_clip=args.grad_clip,
            schedule=args.schedule,
            path_shrink_lambda=0.0,
            grad_noise_std=0.0,
            input_noise_std=0.0,
        )

        # A: baseline
        out_a = _run_one_condition(
            condition="A_baseline",
            init_state=init_state,
            model_cfg=model_cfg,
            train_loader=train_loader,
            test_loader=test_loader,
            cfg=base_cfg,
            device=device,
            seed=seed,
            save_dir=out_dir,
            save_weights=args.save_weights,
        )

        target_train_loss = float(out_a["train_loss"])
        w_a: torch.Tensor = out_a["w"]

        # B: path-regularized (path_shrink_lambda)
        sweep_b: List[Dict[str, Any]] = []
        for lam in args.lambdas:
            cfg_b = TrainConfig(**{**asdict(base_cfg), "path_shrink_lambda": float(lam), "grad_noise_std": 0.0})
            out_b = _run_one_condition(
                condition=f"B_pathreg_lam{float(lam):g}",
                init_state=init_state,
                model_cfg=model_cfg,
                train_loader=train_loader,
                test_loader=test_loader,
                cfg=cfg_b,
                device=device,
                seed=seed,
                save_dir=out_dir,
                save_weights=args.save_weights,
            )
            out_b["lambda"] = float(lam)
            sweep_b.append(out_b)

        picked_b = _pick_aligned(sweep_b, target_train_loss=target_train_loss, target_w=w_a, loss_tol=args.loss_tol)
        pd.DataFrame(
            [
                {
                    "seed": seed,
                    "lambda": float(r.get("lambda", float("nan"))),
                    "train_loss": float(r["train_loss"]),
                    "train_loss_abs_err": float(r["train_loss_abs_err"]),
                    "pl": float(r["pl"]),
                    "gap_loss": float(r["gap_loss"]),
                    "endpoint_l2": float(r["endpoint_l2"]),
                }
                for r in sweep_b
            ]
        ).to_csv(out_dir / f"sweep_B_seed{seed}.csv", index=False)

        # C: noise path (gradient noise)
        sweep_c: List[Dict[str, Any]] = []
        for gn in args.grad_noises:
            cfg_c = TrainConfig(**{**asdict(base_cfg), "path_shrink_lambda": 0.0, "grad_noise_std": float(gn)})
            out_c = _run_one_condition(
                condition=f"C_gradnoise{float(gn):g}",
                init_state=init_state,
                model_cfg=model_cfg,
                train_loader=train_loader,
                test_loader=test_loader,
                cfg=cfg_c,
                device=device,
                seed=seed + 123,
                save_dir=out_dir,
                save_weights=args.save_weights,
            )
            out_c["grad_noise_std"] = float(gn)
            sweep_c.append(out_c)

        picked_c = _pick_aligned(sweep_c, target_train_loss=target_train_loss, target_w=w_a, loss_tol=args.loss_tol)
        pd.DataFrame(
            [
                {
                    "seed": seed,
                    "grad_noise_std": float(r.get("grad_noise_std", float("nan"))),
                    "train_loss": float(r["train_loss"]),
                    "train_loss_abs_err": float(r["train_loss_abs_err"]),
                    "pl": float(r["pl"]),
                    "gap_loss": float(r["gap_loss"]),
                    "endpoint_l2": float(r["endpoint_l2"]),
                }
                for r in sweep_c
            ]
        ).to_csv(out_dir / f"sweep_C_seed{seed}.csv", index=False)

        # endpoint distances
        w_b = picked_b["w"]
        w_c = picked_c["w"]
        dist_ab = float(torch.linalg.vector_norm(w_a - w_b).item()) if w_a.numel() else 0.0
        dist_ac = float(torch.linalg.vector_norm(w_a - w_c).item()) if w_a.numel() else 0.0

        pair_rows.append(
            {
                "seed": seed,
                "train_loss_A": float(out_a["train_loss"]),
                "train_loss_B": float(picked_b["train_loss"]),
                "train_loss_C": float(picked_c["train_loss"]),
                "pl_A": float(out_a["pl"]),
                "pl_B": float(picked_b["pl"]),
                "pl_C": float(picked_c["pl"]),
                "gap_loss_A": float(out_a["gap_loss"]),
                "gap_loss_B": float(picked_b["gap_loss"]),
                "gap_loss_C": float(picked_c["gap_loss"]),
                "endpoint_l2_AB": dist_ab,
                "endpoint_l2_AC": dist_ac,
                "endpoint_l2_AB_over_normA": dist_ab / max(float(out_a["w_norm"]), 1e-12),
                "endpoint_l2_AC_over_normA": dist_ac / max(float(out_a["w_norm"]), 1e-12),
                "picked_lambda": float(picked_b.get("lambda", float("nan"))),
                "picked_grad_noise_std": float(picked_c.get("grad_noise_std", float("nan"))),
                "train_loss_tol": float(args.loss_tol),
            }
        )

        def pack_common(r: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "seed": seed,
                "condition": r["condition"],
                "train_loss": float(r["train_loss"]),
                "train_acc": float(r["train_acc"]),
                "test_loss": float(r["test_loss"]),
                "test_acc": float(r["test_acc"]),
                "gap_loss": float(r["gap_loss"]),
                "gap_acc": float(r["gap_acc"]),
                "pl": float(r["pl"]),
                "ge": float(r["ge"]),
                "steps": int(r["steps"]),
                "avg_lr": float(r.get("avg_lr", float("nan"))),
                "avg_grad_norm": float(r.get("avg_grad_norm", float("nan"))),
                "w_norm": float(r["w_norm"]),
            }

        ra = pack_common(out_a)
        ra.update({"lambda": 0.0, "grad_noise_std": 0.0, "endpoint_l2_to_A": 0.0, "train_loss_abs_err_to_A": 0.0, "pl_ratio_to_A": 1.0})

        rb = pack_common(picked_b)
        rb.update(
            {
                "lambda": float(picked_b.get("lambda", float("nan"))),
                "grad_noise_std": 0.0,
                "endpoint_l2_to_A": float(picked_b["endpoint_l2"]),
                "train_loss_abs_err_to_A": float(picked_b["train_loss_abs_err"]),
                "pl_ratio_to_A": float(picked_b["pl"]) / max(float(out_a["pl"]), 1e-12),
            }
        )

        rc = pack_common(picked_c)
        rc.update(
            {
                "lambda": 0.0,
                "grad_noise_std": float(picked_c.get("grad_noise_std", float("nan"))),
                "endpoint_l2_to_A": float(picked_c["endpoint_l2"]),
                "train_loss_abs_err_to_A": float(picked_c["train_loss_abs_err"]),
                "pl_ratio_to_A": float(picked_c["pl"]) / max(float(out_a["pl"]), 1e-12),
            }
        )

        rows.extend([ra, rb, rc])

    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out_dir / "endpoint_pairs.csv", index=False)

    print(f"Wrote: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PSG Exp-D: one-shot experiment (same train loss, similar endpoint, different path => different gap)"
    )
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-test", type=int, default=1024)
    p.add_argument("--data-noise", type=float, default=0.2)
    p.add_argument("--label-noise", type=float, default=0.1)

    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    p.add_argument("--schedule", type=str, default="cosine", choices=["constant", "cosine", "step"])
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--grad-clip", type=float, default=0.0)

    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    p.add_argument("--grad-noises", type=float, nargs="+", default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3])

    p.add_argument(
        "--loss-tol",
        type=float,
        default=0.005,
        help="Absolute tolerance for matching train loss to baseline A before prioritizing endpoint matching.",
    )

    p.add_argument("--save-weights", action="store_true", help="Save flattened endpoint vectors as .pt files")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
