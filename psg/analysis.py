from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def corr_table(df: pd.DataFrame, cols: Sequence[str], target: str) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c == target or c not in df.columns:
            continue
        x = df[c].to_numpy(dtype=float)
        y = df[target].to_numpy(dtype=float)
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            pearson = float(pd.Series(x).corr(pd.Series(y), method="pearson"))
            spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        else:
            pearson = float("nan")
            spearman = float("nan")
        rows.append({"feature": c, "pearson": pearson, "spearman": spearman})
    out = pd.DataFrame(rows).sort_values(by="spearman", key=lambda s: s.abs(), ascending=False)
    return out


def add_train_loss_bins(df: pd.DataFrame, n_bins: int = 5, col: str = "train_loss") -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df
    df["train_loss_bin"] = pd.qcut(df[col], q=n_bins, duplicates="drop")
    return df


def summarize_by_bins(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    gb = df.groupby(group_cols, dropna=False)
    agg = gb[metric_cols].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c).strip() for c in agg.columns.to_flat_index()]
    return agg.reset_index()
