"""Interactive Plotly versions of the report figures.

Each function mirrors a matplotlib plot in `plots.py` but returns a
`plotly.graph_objects.Figure` instead of saving a PNG, so the webui can render
the figure as an interactive widget (`gr.Plot` accepts Plotly figures
directly).

Figure IDs used downstream:
    "pareto"             — training Pareto + BO candidates
    "feature_importance" — held-out permutation-importance bars per target
    "gp_vs_bnn"          — GP vs Bayesian-NN scatter
    "candidate_heatmap"  — top-candidate property heatmap
    "composition"        — metal-phase atomic fractions (fraction schema only)
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)


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


def _short_name(name: str, max_len: int) -> str:
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def make_pareto_figure(
    training_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    target_cols: list[str],
    maximize: tuple[bool, bool] = (True, True),
) -> go.Figure | None:
    if len(target_cols) != 2:
        return None
    if not all(c in training_df.columns for c in target_cols):
        return None

    t1, t2 = target_cols
    fig = go.Figure()

    train_values = training_df[[t1, t2]].dropna().values
    fig.add_trace(go.Scatter(
        x=train_values[:, 0], y=train_values[:, 1],
        mode="markers",
        marker=dict(color="lightgray", size=8, opacity=0.7,
                    line=dict(width=0.5, color="gray")),
        name="Training data",
        hovertemplate=f"{t1}: %{{x:.3f}}<br>{t2}: %{{y:.3f}}<extra></extra>",
    ))

    pf_idx = _pareto_front_indices(train_values, maximize)
    if len(pf_idx) > 0:
        pf = train_values[pf_idx]
        order = pf[:, 0].argsort()
        fig.add_trace(go.Scatter(
            x=pf[order, 0], y=pf[order, 1],
            mode="lines+markers",
            line=dict(color="black", dash="dash", width=1),
            marker=dict(color="rgba(0,0,0,0)", size=12,
                        line=dict(color="black", width=1.5)),
            name="Training Pareto front",
            hovertemplate=f"{t1}: %{{x:.3f}}<br>{t2}: %{{y:.3f}}<extra></extra>",
        ))

    cand_t1 = candidates_df[f"pred_{t1}"].values
    cand_t2 = candidates_df[f"pred_{t2}"].values
    sd1 = candidates_df.get(f"pred_{t1}_sd", pd.Series(np.zeros(len(candidates_df)))).values
    sd2 = candidates_df.get(f"pred_{t2}_sd", pd.Series(np.zeros(len(candidates_df)))).values
    labels = [f"#{i+1}" for i in range(len(candidates_df))]

    hover = []
    for i, (x, y, dx, dy) in enumerate(zip(cand_t1, cand_t2, sd1, sd2)):
        hover.append(
            f"Candidate #{i+1}<br>"
            f"{t1}: {x:.3f} ± {dx:.3f}<br>"
            f"{t2}: {y:.3f} ± {dy:.3f}"
        )
    fig.add_trace(go.Scatter(
        x=cand_t1, y=cand_t2,
        mode="markers+text",
        marker=dict(symbol="diamond", color="crimson", size=12,
                    line=dict(color="darkred", width=1.5)),
        error_x=dict(type="data", array=sd1, color="crimson", thickness=1.5),
        error_y=dict(type="data", array=sd2, color="crimson", thickness=1.5),
        text=labels,
        textposition="top right",
        textfont=dict(color="darkred", size=11),
        name="BO candidates (μ ± σ)",
        hovertext=hover,
        hoverinfo="text",
    ))

    fig.update_layout(
        title=f"Pareto frontier: {t1} vs {t2}",
        xaxis_title=t1,
        yaxis_title=t2,
        template="plotly_white",
        height=560,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


def make_feature_importance_figure(
    eda_summary_str: str | None,
    top_k: int = 10,
) -> go.Figure | None:
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
    fig = make_subplots(
        rows=1, cols=len(targets),
        subplot_titles=[f"Top {top_k} — {t}" for t in targets],
        horizontal_spacing=0.18,
    )

    for col_idx, target in enumerate(targets, start=1):
        entries = top[target][:top_k]
        names = [_short_name(e["feature"], 50) for e in entries][::-1]
        imps = [e["importance"] for e in entries][::-1]
        full_names = [e["feature"] for e in entries][::-1]
        # Surface the fold stability in the tooltip: a feature that topped one
        # fold out of five reads identically to one that topped all five
        # unless the hover says so.
        stab = [(f"<br>stable in {e['folds_in_top_k']}/{e['n_folds']} CV folds"
                 if e.get("n_folds") else "")
                for e in entries][::-1]
        hover = [f"{n}<br>importance: {v:.4f}{k}"
                 for n, v, k in zip(full_names, imps, stab)]

        fig.add_trace(
            go.Bar(
                x=imps, y=names,
                orientation="h",
                marker=dict(color="steelblue", line=dict(color="navy", width=0.5)),
                text=[f"{v:.3f}" for v in imps],
                textposition="outside",
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            ),
            row=1, col=col_idx,
        )
        fig.update_xaxes(title_text="XGBoost gain", row=1, col=col_idx)

    fig.update_layout(
        template="plotly_white",
        height=max(420, 35 * top_k + 120),
        margin=dict(l=10, r=10, t=70, b=40),
    )
    return fig


def make_gp_vs_bnn_figure(
    candidates_df: pd.DataFrame,
    target_cols: list[str],
) -> go.Figure | None:
    if not all(f"pred_{t}_bnn" in candidates_df.columns for t in target_cols):
        return None

    fig = make_subplots(
        rows=1, cols=len(target_cols),
        subplot_titles=[f"GP vs BNN — {t}" for t in target_cols],
        horizontal_spacing=0.12,
    )

    for col_idx, t in enumerate(target_cols, start=1):
        gp = candidates_df[f"pred_{t}"].values
        bnn = candidates_df[f"pred_{t}_bnn"].values
        gp_sd = candidates_df[f"pred_{t}_sd"].values
        bnn_sd = candidates_df[f"pred_{t}_bnn_sd"].values
        labels = [f"#{i+1}" for i in range(len(gp))]

        hover = [
            f"Candidate {labels[i]}<br>"
            f"GP {t}: {gp[i]:.3f} ± {gp_sd[i]:.3f}<br>"
            f"BNN {t}: {bnn[i]:.3f} ± {bnn_sd[i]:.3f}<br>"
            f"|Δ|: {abs(bnn[i] - gp[i]):.3f}"
            for i in range(len(gp))
        ]

        fig.add_trace(
            go.Scatter(
                x=gp, y=bnn,
                mode="markers+text",
                marker=dict(symbol="diamond", color="crimson", size=12,
                            line=dict(color="darkred", width=1.5)),
                error_x=dict(type="data", array=gp_sd, color="gray", thickness=1),
                error_y=dict(type="data", array=bnn_sd, color="gray", thickness=1),
                text=labels,
                textposition="top right",
                textfont=dict(color="darkred", size=11),
                hovertext=hover,
                hoverinfo="text",
                name=t,
                showlegend=False,
            ),
            row=1, col=col_idx,
        )

        lo = float(min((gp - gp_sd).min(), (bnn - bnn_sd).min()))
        hi = float(max((gp + gp_sd).max(), (bnn + bnn_sd).max()))
        fig.add_trace(
            go.Scatter(
                x=[lo, hi], y=[lo, hi],
                mode="lines",
                line=dict(color="black", dash="dash", width=1),
                name="GP = BNN",
                hoverinfo="skip",
                showlegend=(col_idx == 1),
            ),
            row=1, col=col_idx,
        )
        fig.update_xaxes(title_text=f"GP predicted {t}", row=1, col=col_idx)
        fig.update_yaxes(title_text=f"BNN predicted {t}", row=1, col=col_idx)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=10, r=10, t=70, b=40),
    )
    return fig


KEY_PROPERTIES = [
    ("Support work_function (eV)",  "support_properties", "work_function_eV"),
    ("Support acidity NH₃-TPD",     "support_properties", "acidity_NH3_TPD_mmol_g"),
    ("Support basicity CO₂-TPD",    "support_properties", "basicity_CO2_TPD_mmol_g"),
    ("Support lattice a (Å)",       "support_properties", "lattice_a_A"),
    ("Support Sanderson χ",         "support_properties", "sanderson_chi"),
    ("Metal d-band center (eV)",    "metal_properties",   "d_band_center_eV"),
    ("Metal work_function (eV)",    "metal_properties",   "work_function_eV"),
    ("Metal Pauling χ",             "metal_properties",   "pauling_chi"),
]


def make_parity_figure(
    training_pred_df: pd.DataFrame,
    target_cols: list[str],
) -> go.Figure | None:
    """Predicted-vs-actual scatter, one row per surrogate (GP/BNN), one column
    per target. Each panel uses its OWN axis range so a broken surrogate can't
    squash a well-fit one to one pixel."""
    if training_pred_df is None or training_pred_df.empty:
        return None
    needed_true = [f"true_{t}" for t in target_cols]
    if not all(c in training_pred_df.columns for c in needed_true):
        return None

    surrogates = [
        (k, c) for k, c in (("gp", "crimson"),
                            ("svgp", "darkgreen"),
                            ("bnn", "steelblue"))
        if f"{k}_pred_{target_cols[0]}" in training_pred_df.columns
    ]
    if not surrogates:
        return None

    subplot_titles = [
        f"{kind.upper()} — {t}"
        for kind, _ in surrogates
        for t in target_cols
    ]
    fig = make_subplots(
        rows=len(surrogates), cols=len(target_cols),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
    )

    for r, (kind, color) in enumerate(surrogates, start=1):
        for c, t in enumerate(target_cols, start=1):
            y_true = training_pred_df[f"true_{t}"].values
            y_pred = training_pred_df[f"{kind}_pred_{t}"].values
            y_sd = training_pred_df.get(
                f"{kind}_sd_{t}", pd.Series(np.zeros(len(y_pred)))
            ).values

            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            yt, yp, ys = y_true[mask], y_pred[mask], y_sd[mask]
            r2 = _r2(yt, yp)
            mae = float(np.mean(np.abs(yt - yp)))

            true_span = float(np.nanmax(yt) - np.nanmin(yt))
            pad = 0.15 * max(true_span, 1e-6)
            lo = float(np.nanmin(yt)) - pad
            hi = float(np.nanmax(yt)) + pad
            allowed = 5 * true_span
            in_range = yp[(yp > lo - allowed) & (yp < hi + allowed)]
            if len(in_range) > 0:
                lo = min(lo, float(np.nanmin(in_range)) - pad)
                hi = max(hi, float(np.nanmax(in_range)) + pad)
            off_axis = int(np.sum((yp < lo) | (yp > hi)))

            hover = [
                f"sample {i}<br>true: {a:.3f}<br>{kind.upper()} pred: "
                f"{b:.3f} ± {s:.3f}"
                for i, (a, b, s) in enumerate(zip(yt, yp, ys))
            ]
            fig.add_trace(
                go.Scatter(
                    x=yt, y=yp,
                    mode="markers",
                    marker=dict(symbol={"gp": "diamond", "svgp": "square"}.get(kind, "circle"),
                                color=color, size=8, opacity=0.8,
                                line=dict(color="black", width=0.5)),
                    error_y=dict(type="data", array=ys, color=color, thickness=0.8),
                    hovertext=hover, hoverinfo="text",
                    name=(f"{kind.upper()} {t}  R²={_fmt_r2(r2)}  "
                          f"MAE={_fmt_num(mae)}"),
                    showlegend=True,
                    legendgroup=kind,
                ),
                row=r, col=c,
            )
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi], y=[lo, hi],
                    mode="lines",
                    line=dict(color="gray", dash="dash", width=1),
                    name="y = x",
                    hoverinfo="skip",
                    showlegend=(r == 1 and c == 1),
                ),
                row=r, col=c,
            )
            fig.update_xaxes(title_text=f"True {t}", range=[lo, hi], row=r, col=c)
            fig.update_yaxes(
                title_text=f"{kind.upper()} predicted {t}",
                range=[lo, hi], row=r, col=c,
                scaleanchor=f"x{(r-1) * len(target_cols) + c}", scaleratio=1,
            )
            if off_axis:
                fig.add_annotation(
                    text=f"{off_axis}/{len(yp)} predictions off-axis",
                    xref=f"x{(r-1) * len(target_cols) + c} domain",
                    yref=f"y{(r-1) * len(target_cols) + c} domain",
                    x=0.5, y=0.04, showarrow=False,
                    font=dict(color="darkred", size=10),
                    bgcolor="rgba(255,228,225,0.9)",
                    bordercolor="darkred", borderwidth=1, borderpad=3,
                )

    fig.update_layout(
        template="plotly_white",
        height=420 * len(surrogates),
        margin=dict(l=10, r=10, t=70, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _fmt_r2(r2: float) -> str:
    if r2 < -100:
        return f"{r2:.2e}"
    return f"{r2:.3f}"


def _fmt_num(v: float) -> str:
    if abs(v) >= 1000 or (0 < abs(v) < 0.001):
        return f"{v:.2e}"
    return f"{v:.3f}"


def make_candidate_heatmap_figure(enriched_candidates: list[dict]) -> go.Figure | None:
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
    col_min = np.nanmin(matrix, axis=0)
    col_max = np.nanmax(matrix, axis=0)
    span = np.where(col_max - col_min > 1e-9, col_max - col_min, 1.0)
    norm_matrix = (matrix - col_min) / span

    col_labels = [p[0] for p in KEY_PROPERTIES]
    text = [[("" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.2f}")
             for j in range(matrix.shape[1])]
            for i in range(matrix.shape[0])]
    hover = [[f"{row_labels[i]}<br>{col_labels[j]}<br>raw: "
              f"{'NaN' if np.isnan(matrix[i, j]) else f'{matrix[i, j]:.3f}'}"
              f"<br>col min-max: "
              f"{'NaN' if np.isnan(norm_matrix[i, j]) else f'{norm_matrix[i, j]:.2f}'}"
              for j in range(matrix.shape[1])]
             for i in range(matrix.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=norm_matrix,
        x=col_labels,
        y=row_labels,
        colorscale="Viridis",
        zmin=0, zmax=1,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertext=hover,
        hoverinfo="text",
        colorbar=dict(title="col min-max"),
    ))

    fig.update_layout(
        title="Candidate properties (color = column min-max across candidates)",
        template="plotly_white",
        height=max(360, 55 * len(row_labels) + 220),
        xaxis=dict(tickangle=-35, side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=70, b=120),
    )
    return fig


def make_fraction_composition_figure(
    enriched_candidates: list[dict],
    element_cols: list[str],
    support_cations: list[str],
) -> go.Figure | None:
    """Plotly twin of plots.make_fraction_composition_heatmap.

    Column selection and the shared-scale decision live in
    `plots.build_fraction_composition_matrix` so the saved PNG and the webui
    widget always show the same elements.
    """
    from plots import build_fraction_composition_matrix

    built = build_fraction_composition_matrix(
        enriched_candidates, element_cols, support_cations,
    )
    if built is None:
        return None
    row_labels, col_labels, matrix = built
    vmax = float(np.nanmax(matrix)) or 1.0

    text = [[("" if matrix[i, j] <= 0 else f"{matrix[i, j]:.3g}")
             for j in range(matrix.shape[1])]
            for i in range(matrix.shape[0])]
    hover = [[f"{row_labels[i]}<br>{col_labels[j]}<br>atomic fraction: "
              f"{matrix[i, j]:.4f}"
              for j in range(matrix.shape[1])]
             for i in range(matrix.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=col_labels,
        y=row_labels,
        colorscale="Magma",
        zmin=0.0, zmax=vmax,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertext=hover,
        hoverinfo="text",
        colorbar=dict(title="atomic<br>fraction"),
    ))
    fig.update_layout(
        title="Metal-phase atomic fractions (support excluded — see row label)",
        template="plotly_white",
        height=max(360, 55 * len(row_labels) + 200),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=70, b=80),
    )
    return fig
