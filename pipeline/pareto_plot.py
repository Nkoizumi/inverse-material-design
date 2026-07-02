"""Pareto plot for dual-target BO/MOBO output.

Visualizes:
    - Training data targets (gray scatter)
    - Training-data Pareto front (darker)
    - BO/MOBO candidate predictions with σ error bars (red diamonds)
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def make_pareto_plot(
    training_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    target_cols: list[str],
    out_path: Path,
    maximize: tuple[bool, bool] = (True, True),
) -> Path | None:
    if len(target_cols) != 2:
        log.info("Pareto plot needs exactly 2 targets, got %d. Skipping.", len(target_cols))
        return None
    if not all(c in training_df.columns for c in target_cols):
        log.warning("Training data missing target columns; skipping Pareto plot.")
        return None

    t1, t2 = target_cols
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    train_values = training_df[[t1, t2]].dropna().values
    ax.scatter(train_values[:, 0], train_values[:, 1],
               c="lightgray", s=40, label="Training data", alpha=0.7, zorder=1)

    pf_idx = _pareto_front_indices(train_values, maximize)
    if len(pf_idx) > 0:
        pf_points = train_values[pf_idx]
        # Order along axis 0 for line plotting.
        order = pf_points[:, 0].argsort()
        pf_sorted = pf_points[order]
        ax.plot(pf_sorted[:, 0], pf_sorted[:, 1],
                "k--", alpha=0.4, linewidth=1, zorder=2)
        ax.scatter(pf_points[:, 0], pf_points[:, 1],
                   c="black", s=70, marker="o", facecolors="none",
                   edgecolors="black", linewidths=1.5,
                   label="Training Pareto front", zorder=3)

    # Candidates with error bars.
    cand_pred_t1 = candidates_df[f"pred_{t1}"].values
    cand_pred_t2 = candidates_df[f"pred_{t2}"].values
    cand_sd_t1 = candidates_df.get(f"pred_{t1}_sd", pd.Series(np.zeros(len(candidates_df)))).values
    cand_sd_t2 = candidates_df.get(f"pred_{t2}_sd", pd.Series(np.zeros(len(candidates_df)))).values

    ax.errorbar(
        cand_pred_t1, cand_pred_t2,
        xerr=cand_sd_t1, yerr=cand_sd_t2,
        fmt="D", color="crimson", markersize=10,
        ecolor="crimson", elinewidth=1.5, alpha=0.85,
        markerfacecolor="crimson", markeredgecolor="darkred",
        label="BO candidates (μ ± σ)", zorder=5,
    )

    for i, (x, y) in enumerate(zip(cand_pred_t1, cand_pred_t2)):
        ax.annotate(
            f"#{i+1}", (x, y), xytext=(7, 7), textcoords="offset points",
            fontsize=10, color="darkred", weight="bold",
        )

    ax.set_xlabel(t1)
    ax.set_ylabel(t2)
    ax.set_title(f"Pareto frontier: {t1} vs {t2}")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved Pareto plot to %s.", out_path)
    return out_path


def _pareto_front_indices(values: np.ndarray, maximize: tuple[bool, bool]) -> np.ndarray:
    sign = np.array([1 if m else -1 for m in maximize])
    v = values * sign
    n = v.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominated = (v >= v[i]).all(axis=1) & (v > v[i]).any(axis=1)
        if dominated.any():
            is_pareto[i] = False
    return np.where(is_pareto)[0]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    training = pd.read_parquet(config.DATA_DIR / "raw.parquet")
    candidates = pd.read_parquet(config.DATA_DIR / "candidates.parquet")
    out = config.REPORTS_DIR / "pareto_test.png"
    make_pareto_plot(training, candidates, config.TARGET_COLS, out)
    print(f"Saved {out}")
