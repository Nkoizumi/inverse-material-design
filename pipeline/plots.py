"""Plot generation for the report.

Four figures are produced (dual-target catalyst mode):

  1. Pareto frontier — training data + BO candidate predictions with σ.
  2. Feature importance bars — top XGBoost-gain features per target.
  3. GP vs BNN agreement scatter — predictions of both surrogates on each
     candidate, with diagonal reference line.
  4. Candidate property heatmap — rows = top candidates, cols = key
     catalyst features, color = candidate's value normalised across the
     candidate set (column min-max → 0..1).

Each function returns the path on success or None if the figure can't be drawn.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pareto plot
# ─────────────────────────────────────────────────────────────────────────────
def make_pareto_plot(
    training_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    target_cols: list[str],
    out_path: Path,
    maximize: tuple[bool, bool] = (True, True),
) -> Path | None:
    if len(target_cols) != 2:
        return None
    if not all(c in training_df.columns for c in target_cols):
        return None

    t1, t2 = target_cols
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    train_values = training_df[[t1, t2]].dropna().values
    ax.scatter(train_values[:, 0], train_values[:, 1],
               c="lightgray", s=40, label="Training data", alpha=0.7, zorder=1)

    pf_idx = _pareto_front_indices(train_values, maximize)
    if len(pf_idx) > 0:
        pf = train_values[pf_idx]
        order = pf[:, 0].argsort()
        ax.plot(pf[order, 0], pf[order, 1], "k--", alpha=0.4, linewidth=1, zorder=2)
        ax.scatter(pf[:, 0], pf[:, 1],
                   c="black", s=70, marker="o", facecolors="none",
                   edgecolors="black", linewidths=1.5,
                   label="Training Pareto front", zorder=3)

    cand_t1 = candidates_df[f"pred_{t1}"].values
    cand_t2 = candidates_df[f"pred_{t2}"].values
    sd1 = candidates_df.get(f"pred_{t1}_sd", pd.Series(np.zeros(len(candidates_df)))).values
    sd2 = candidates_df.get(f"pred_{t2}_sd", pd.Series(np.zeros(len(candidates_df)))).values

    ax.errorbar(cand_t1, cand_t2, xerr=sd1, yerr=sd2,
                fmt="D", color="crimson", markersize=10,
                ecolor="crimson", elinewidth=1.5, alpha=0.85,
                markerfacecolor="crimson", markeredgecolor="darkred",
                label="BO candidates (μ ± σ)", zorder=5)
    for i, (x, y) in enumerate(zip(cand_t1, cand_t2)):
        ax.annotate(f"#{i+1}", (x, y), xytext=(7, 7),
                    textcoords="offset points", fontsize=10,
                    color="darkred", weight="bold")

    ax.set_xlabel(t1)
    ax.set_ylabel(t2)
    ax.set_title(f"Pareto frontier: {t1} vs {t2}")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _pareto_front_indices(values: np.ndarray, maximize: tuple[bool, bool]) -> np.ndarray:
    sign = np.array([1 if m else -1 for m in maximize])
    v = values * sign
    n = v.shape[0]
    pf = np.ones(n, dtype=bool)
    for i in range(n):
        if not pf[i]:
            continue
        dominated = (v >= v[i]).all(axis=1) & (v > v[i]).any(axis=1)
        if dominated.any():
            pf[i] = False
    return np.where(pf)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature importance bars
# ─────────────────────────────────────────────────────────────────────────────
def make_feature_importance_plot(
    eda_summary_str: str | None,
    out_path: Path,
    top_k: int = 10,
) -> Path | None:
    if not eda_summary_str:
        return None
    try:
        data = json.loads(eda_summary_str)
    except Exception:
        return None
    top = data.get("top_features_per_target", {})
    if not top:
        return None

    targets = list(top.keys())
    # Bigger figure: bars get room to breathe and labels stay readable when the
    # report is embedded in a viewer that auto-fits images.
    fig, axes = plt.subplots(
        1, len(targets),
        figsize=(9.5 * len(targets), 8),
        constrained_layout=True,
    )
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        entries = top[target][:top_k]
        names = [_short_name(e["feature"], 42) for e in entries][::-1]
        imps = [e["importance"] for e in entries][::-1]
        bars = ax.barh(names, imps, color="steelblue", edgecolor="navy")
        max_imp = max(imps) if imps else 1.0
        for bar, v in zip(bars, imps):
            ax.text(v + max_imp * 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=11)
        ax.set_xlabel("Feature importance (XGBoost gain)", fontsize=12)
        ax.set_title(f"Top {top_k} features — {target}", fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        ax.set_xlim(0, max_imp * 1.18)   # leave room for the value annotation
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _short_name(name: str, max_len: int) -> str:
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


# ─────────────────────────────────────────────────────────────────────────────
# 3. GP vs BNN agreement scatter
# ─────────────────────────────────────────────────────────────────────────────
def make_gp_vs_bnn_scatter(
    candidates_df: pd.DataFrame,
    target_cols: list[str],
    out_path: Path,
) -> Path | None:
    if not all(f"pred_{t}_bnn" in candidates_df.columns for t in target_cols):
        return None

    fig, axes = plt.subplots(1, len(target_cols), figsize=(6 * len(target_cols), 5.5))
    if len(target_cols) == 1:
        axes = [axes]

    for ax, t in zip(axes, target_cols):
        gp = candidates_df[f"pred_{t}"].values
        bnn = candidates_df[f"pred_{t}_bnn"].values
        gp_sd = candidates_df[f"pred_{t}_sd"].values
        bnn_sd = candidates_df[f"pred_{t}_bnn_sd"].values

        ax.errorbar(gp, bnn, xerr=gp_sd, yerr=bnn_sd,
                    fmt="D", color="crimson", markersize=10,
                    ecolor="gray", elinewidth=1, alpha=0.8,
                    markerfacecolor="crimson", markeredgecolor="darkred")

        lo = float(min(gp.min() - gp_sd.max(), bnn.min() - bnn_sd.max()))
        hi = float(max(gp.max() + gp_sd.max(), bnn.max() + bnn_sd.max()))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="GP = BNN")

        for i, (x, y) in enumerate(zip(gp, bnn)):
            ax.annotate(f"#{i+1}", (x, y), xytext=(7, 7),
                        textcoords="offset points", fontsize=10,
                        color="darkred", weight="bold")

        ax.set_xlabel(f"GP predicted {t}")
        ax.set_ylabel(f"BNN predicted {t}")
        ax.set_title(f"GP vs BNN — {t}")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Parity plot — predicted vs actual on the training set
# ─────────────────────────────────────────────────────────────────────────────
def make_parity_plot(
    training_pred_df: pd.DataFrame | None,
    target_cols: list[str],
    out_path: Path,
) -> Path | None:
    if training_pred_df is None or training_pred_df.empty:
        return None
    needed_true = [f"true_{t}" for t in target_cols]
    if not all(c in training_pred_df.columns for c in needed_true):
        return None

    # One row per surrogate, one column per target. Per-panel axis scaling so
    # that a broken BNN can't squash a well-fit GP into a single pixel.
    surrogates = [
        (k, c, m) for k, c, m in
        (("gp", "crimson", "D"),
         ("svgp", "darkgreen", "s"),
         ("bnn", "steelblue", "o"))
        if f"{k}_pred_{target_cols[0]}" in training_pred_df.columns
    ]
    if not surrogates:
        return None

    fig, axes = plt.subplots(
        len(surrogates), len(target_cols),
        figsize=(5.5 * len(target_cols), 4.8 * len(surrogates)),
        squeeze=False,
    )

    for r, (kind, color, marker) in enumerate(surrogates):
        for c, t in enumerate(target_cols):
            ax = axes[r, c]
            y_true = training_pred_df[f"true_{t}"].values
            y_pred = training_pred_df[f"{kind}_pred_{t}"].values
            y_sd = training_pred_df.get(
                f"{kind}_sd_{t}", pd.Series(np.zeros(len(y_pred)))
            ).values

            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            yt, yp, ys = y_true[mask], y_pred[mask], y_sd[mask]
            r2 = 1.0 - np.sum((yt - yp) ** 2) / max(
                np.sum((yt - np.mean(yt)) ** 2), 1e-12)
            mae = float(np.mean(np.abs(yt - yp)))

            # Honest range, but bound the visible window to the TRUE-value
            # span plus the GP's own predictions, ignoring catastrophic BNN
            # outliers. Points outside still plot — they just fall off-axis
            # and we annotate the count.
            true_span = float(np.nanmax(yt) - np.nanmin(yt))
            pad = 0.15 * max(true_span, 1e-6)
            lo = float(np.nanmin(yt)) - pad
            hi = float(np.nanmax(yt)) + pad
            # If THIS surrogate's predictions are reasonable (within 5×true_span
            # of the true range), let the axis follow them; otherwise clamp.
            allowed = 5 * true_span
            pred_in_range = yp[(yp > lo - allowed) & (yp < hi + allowed)]
            if len(pred_in_range) > 0:
                lo = min(lo, float(np.nanmin(pred_in_range)) - pad)
                hi = max(hi, float(np.nanmax(pred_in_range)) + pad)
            off_axis = int(np.sum((yp < lo) | (yp > hi)))

            ax.errorbar(
                yt, yp, yerr=ys,
                fmt=marker, color=color, markersize=6, alpha=0.75,
                ecolor=color, elinewidth=0.6,
                markeredgecolor="black", markeredgewidth=0.4,
                label=f"{kind.upper()}  R²={_fmt_r2(r2)}  MAE={_fmt_num(mae)}",
            )
            ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1,
                    label="y = x")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            if off_axis:
                ax.annotate(
                    f"{off_axis}/{len(yp)} predictions off-axis "
                    f"(|err| > {5}×true range)",
                    xy=(0.5, 0.02), xycoords="axes fraction",
                    ha="center", va="bottom",
                    color="darkred", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="mistyrose", edgecolor="darkred",
                              alpha=0.9),
                )

            ax.set_xlabel(f"True {t}")
            ax.set_ylabel(f"{kind.upper()} predicted {t}")
            ax.set_title(f"{kind.upper()} — {t}")
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
            ax.grid(alpha=0.3, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _fmt_r2(r2: float) -> str:
    if r2 < -1.0:
        return f"{r2:.2e}" if r2 < -100 else f"{r2:.3f}"
    return f"{r2:.3f}"


def _fmt_num(v: float) -> str:
    if abs(v) >= 1000 or (0 < abs(v) < 0.001):
        return f"{v:.2e}"
    return f"{v:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Candidate property heatmap
# ─────────────────────────────────────────────────────────────────────────────
KEY_PROPERTIES = [
    # (display name, properties-dict key, dict-key)
    ("Support work_function (eV)",  "support_properties", "work_function_eV"),
    ("Support acidity NH₃-TPD",     "support_properties", "acidity_NH3_TPD_mmol_g"),
    ("Support basicity CO₂-TPD",    "support_properties", "basicity_CO2_TPD_mmol_g"),
    ("Support lattice a (Å)",       "support_properties", "lattice_a_A"),
    ("Support Sanderson χ",         "support_properties", "sanderson_chi"),
    ("Metal d-band center (eV)",    "metal_properties",   "d_band_center_eV"),
    ("Metal work_function (eV)",    "metal_properties",   "work_function_eV"),
    ("Metal Pauling χ",             "metal_properties",   "pauling_chi"),
]


def make_candidate_heatmap(
    enriched_candidates: list[dict],
    out_path: Path,
) -> Path | None:
    if not enriched_candidates:
        return None

    rows, row_labels = [], []
    for i, cand in enumerate(enriched_candidates):
        label = cand.get("_label", f"Candidate {i+1}")
        row = []
        for _, dict_key, prop_key in KEY_PROPERTIES:
            val = cand.get(dict_key, {}).get(prop_key, np.nan)
            row.append(val)
        rows.append(row)
        row_labels.append(f"#{i+1}: {label}")

    matrix = np.array(rows, dtype=float)
    # Column min-max normalisation for the colour scale.
    col_min = np.nanmin(matrix, axis=0)
    col_max = np.nanmax(matrix, axis=0)
    span = np.where(col_max - col_min > 1e-9, col_max - col_min, 1.0)
    norm_matrix = (matrix - col_min) / span

    fig, ax = plt.subplots(figsize=(0.9 * len(KEY_PROPERTIES) + 6, 0.6 * len(rows) + 2))
    cmap = matplotlib.colormaps["viridis"]
    im = ax.imshow(norm_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(KEY_PROPERTIES)))
    ax.set_xticklabels([p[0] for p in KEY_PROPERTIES],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate cells with the raw values (not normalised).
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if norm_matrix[i, j] < 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=8)

    ax.set_title("Candidate properties (color = normalised across the candidate set)")
    fig.colorbar(im, ax=ax, label="min-max within column")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
