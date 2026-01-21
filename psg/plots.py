from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def scatter(df: pd.DataFrame, x: str, y: str, out_path: Path, hue: Optional[str] = None, title: str = "") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _ensure_dir(out_path.parent)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.8)
    plt.title(title or f"{y} vs {x}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
