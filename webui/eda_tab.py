"""Data Exploration (EDA) tab for the inverse-design webui.

Ports the EDA + LLM-decisioning pieces of the original Streamlit auto_eda app
(/home/nao/auto_eda/app.py, tabs 0–6) into Gradio:

  Overview / Missing Values / Distributions / Outliers / Correlations
  LLM Decisions      — Phi-4 + Mistral pick preprocessing methods
  Transformed Data   — apply the LLM-decided sklearn pipeline

Pure Plotly + pandas for the descriptive tabs; the last two use the
`AutoEDAPipeline` class from /home/nao/auto_eda directly (no training, just
the LLM ensemble + sklearn ColumnTransformer it builds). Operates on the RAW
uploaded CSV before matminer featurization.
"""
from __future__ import annotations

import logging
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)

# Make /home/nao/auto_eda's `pipeline` package importable. Step3 does the same;
# both projects ship a `pipeline/` package but our local one is only imported
# via flat modules (`import step1_load`), so the `pipeline.*` namespace stays
# free for auto_eda's orchestrator and llm engine.
_AUTO_EDA_PATH = "/home/nao/auto_eda"
if _AUTO_EDA_PATH not in sys.path:
    sys.path.insert(0, _AUTO_EDA_PATH)


# ─── theme: match the rest of the app (plotly_white, light) ──────────────────
THEME = dict(
    template="plotly_white",
    margin=dict(l=10, r=10, t=60, b=40),
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def row_alignment_error(n_before: int, n_after: int) -> str | None:
    """Return an error message if targets can no longer be paired to rows.

    Tab 2 re-attaches the untransformed target columns to the transformed
    feature frame BY POSITION, which is only valid while the transformer
    preserves both row count and row order.

    It does today: auto_eda's OutlierHandler clips (iqr/zscore) or imputes to
    the column median (isolation_forest) rather than dropping rows, and no
    other step in that pipeline touches the row axis. But this used to be
    written as `df[t].values[:len(tdf)]`, which silently TRUNCATED — so a
    transformer that ever did drop rows would have paired each target value
    with the wrong catalyst and produced a quietly mistrained surrogate, with
    nothing raised and nothing logged.

    Refusing is the right failure mode here: a visible error is recoverable,
    a silent mispairing is not. Returns None when the frames line up.
    """
    if n_before == n_after:
        return None
    return (
        f"Transform changed the row count ({n_before} → {n_after}), so the "
        f"target columns can no longer be matched to their rows by position. "
        f"Refusing to re-attach them rather than risk pairing a target value "
        f"with the wrong catalyst — this needs an index-preserving transform "
        f"before Tab 2 can be used here."
    )


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-tab 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
def render_overview(df: pd.DataFrame) -> None:
    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)
    total_missing = int(df.isna().sum().sum())
    pct_missing = 100 * total_missing / max(df.size, 1)

    gr.Markdown(
        f"**Rows** {len(df):,}  |  **Columns** {len(df.columns)}  |  "
        f"**Numeric** {len(num_cols)}  |  **Categorical** {len(cat_cols)}  |  "
        f"**Missing** {total_missing:,} ({pct_missing:.1f}%)"
    )

    gr.Markdown("#### Data preview (first 20 rows)")
    gr.Dataframe(value=df.head(20), interactive=False, wrap=True)

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Descriptive statistics")
            stats = df.describe(include="all").transpose().round(3)
            stats.insert(0, "column", stats.index)
            gr.Dataframe(value=stats.reset_index(drop=True), interactive=False)
        with gr.Column():
            gr.Markdown("#### Data types & nulls")
            dtype_df = pd.DataFrame({
                "Column":   df.columns,
                "DType":    df.dtypes.astype(str).values,
                "Non-Null": df.notna().sum().values,
                "Null":     df.isna().sum().values,
                "Unique":   df.nunique().values,
            })
            gr.Dataframe(value=dtype_df, interactive=False)

    if num_cols:
        skew_vals = df[num_cols].skew().sort_values()
        fig = px.bar(
            x=skew_vals.values, y=skew_vals.index,
            orientation="h",
            title="Skewness per numeric feature (|skew| > 0.5 hints at non-normality)",
            labels={"x": "Skewness", "y": "Feature"},
            color=skew_vals.values, color_continuous_scale="RdBu_r",
            range_color=[-max(abs(skew_vals.min()), abs(skew_vals.max()), 1.0),
                          max(abs(skew_vals.min()), abs(skew_vals.max()), 1.0)],
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="#d23")
        fig.add_vline(x=-0.5, line_dash="dash", line_color="#d23")
        fig.update_layout(**THEME, height=max(280, 22 * len(num_cols) + 120))
        gr.Plot(value=fig, show_label=False)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-tab 2 — Missing Values
# ─────────────────────────────────────────────────────────────────────────────
def render_missing(df: pd.DataFrame) -> None:
    miss = pd.DataFrame({
        "Column":        df.columns,
        "Missing Count": df.isna().sum().values,
        "Missing %":     (100 * df.isna().mean()).round(2).values,
        "DType":         df.dtypes.astype(str).values,
    })
    miss = miss[miss["Missing Count"] > 0].sort_values(
        "Missing %", ascending=False,
    )

    if miss.empty:
        gr.Markdown("✅ **No missing values in any column.**")
        return

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Per-column missingness")
            gr.Dataframe(value=miss.reset_index(drop=True), interactive=False)
        with gr.Column(scale=2):
            fig_bar = px.bar(
                miss, x="Column", y="Missing %",
                title="Missing-value % per column",
                color="Missing %", color_continuous_scale="Reds",
                text="Missing %",
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(**THEME, height=420)
            gr.Plot(value=fig_bar, show_label=False)

    gr.Markdown("#### Missingness pattern (red = missing)")
    miss_matrix = df.isna().astype(int)
    fig_heat = px.imshow(
        miss_matrix.T, aspect="auto",
        color_continuous_scale=[(0, "#eef"), (1, "#d23")],
        title="Missingness map (rows = features, columns = data points)",
        labels={"x": "Row index", "y": "Feature", "color": "missing"},
    )
    fig_heat.update_layout(**THEME, height=420)
    gr.Plot(value=fig_heat, show_label=False)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-tab 3 — Distributions
# ─────────────────────────────────────────────────────────────────────────────
def render_distributions(df: pd.DataFrame) -> None:
    num_cols = _numeric_cols(df)
    if not num_cols:
        gr.Markdown("*No numeric columns to plot.*")
        return

    feature = gr.Dropdown(
        choices=num_cols, value=num_cols[0],
        label="Feature to inspect",
        interactive=True,
    )

    def _on_feature(feat: str):
        series = df[feat].dropna()
        fig_h = px.histogram(
            df, x=feat, nbins=40, marginal="box",
            title=f"Distribution of {feat}",
            color_discrete_sequence=["#58a6ff"],
        )
        fig_h.update_layout(**THEME, height=420)

        try:
            from scipy import stats
            (osm, osr), (slope, intercept, _) = stats.probplot(series, dist="norm")
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(
                x=osm, y=osr, mode="markers",
                marker=dict(color="#58a6ff", size=5), name="Data"))
            fig_q.add_trace(go.Scatter(
                x=osm, y=slope * np.array(osm) + intercept,
                mode="lines", line=dict(color="#d23", width=2),
                name="Normal",
            ))
            fig_q.update_layout(
                title=f"Q-Q plot of {feat}",
                xaxis_title="Theoretical quantiles",
                yaxis_title="Sample quantiles",
                **THEME, height=420,
            )
        except Exception:
            fig_q = go.Figure()
            fig_q.update_layout(title="Q-Q plot unavailable (scipy missing)", **THEME)

        return fig_h, fig_q

    init_h, init_q = _on_feature(num_cols[0])
    with gr.Row():
        hist_plot = gr.Plot(value=init_h, label="Histogram + box marginal")
        qq_plot = gr.Plot(value=init_q, label="Q-Q plot vs normal")
    feature.change(_on_feature, inputs=[feature], outputs=[hist_plot, qq_plot])
    gr.Markdown("#### Numeric feature statistics")
    stats_df = pd.DataFrame({
        "Feature":  num_cols,
        "Mean":     [df[c].mean() for c in num_cols],
        "Std":      [df[c].std() for c in num_cols],
        "Skewness": [round(df[c].skew(), 3) for c in num_cols],
        "Kurtosis": [round(df[c].kurtosis(), 3) for c in num_cols],
        # NOTE ON THE KURTOSIS THRESHOLD: pandas' .kurtosis() is FISHER'S
        # (excess) kurtosis — a normal distribution scores 0, not 3. The `> 3`
        # here therefore means "excess kurtosis above 3", i.e. tails heavier
        # than roughly a t(5); it is a deliberately conservative flag, not the
        # Pearson-convention "normal = 3" test it resembles. Do not "correct"
        # it to 0 without deciding that: the skew term already catches
        # asymmetric features, and dropping to 0 would mark almost every real
        # column as needing a transform.
        "Needs transform?": [
            "Yes" if abs(df[c].skew()) > 0.5 or abs(df[c].kurtosis()) > 3
            else "No"
            for c in num_cols
        ],
    }).round(4)
    gr.Dataframe(value=stats_df, interactive=False)

    if len(num_cols) >= 2:
        gr.Markdown("#### Violin plots (up to first 8 numeric features)")
        fig_v = go.Figure()
        for c in num_cols[:8]:
            fig_v.add_trace(go.Violin(
                y=df[c].dropna(), name=c,
                box_visible=True, meanline_visible=True,
            ))
        fig_v.update_layout(**THEME, height=450,
                            title="Feature distributions (violin)")
        gr.Plot(value=fig_v, show_label=False)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-tab 4 — Outliers
# ─────────────────────────────────────────────────────────────────────────────
def render_outliers(df: pd.DataFrame) -> None:
    num_cols = _numeric_cols(df)
    if not num_cols:
        gr.Markdown("*No numeric columns to inspect.*")
        return

    with gr.Row():
        method = gr.Radio(
            choices=["IQR", "Z-Score"], value="IQR",
            label="Outlier rule",
            interactive=True,
        )
        mult = gr.Slider(
            minimum=1.0, maximum=3.0, step=0.1, value=1.5,
            label="Multiplier (k)",
            interactive=True,
        )
        feat = gr.Dropdown(
            choices=num_cols, value=num_cols[0],
            label="Feature for box plot",
            interactive=True,
        )
    gr.Markdown(
        "_IQR: outside Q1 − k·IQR or Q3 + k·IQR.   "
        "Z-Score: outside mean ± k·std._"
    )

    def _compute(rule: str, k: float, selected: str):
        rows = []
        for col in num_cols:
            series = df[col].dropna()
            if rule == "IQR":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                mask = (series < q1 - k * iqr) | (series > q3 + k * iqr)
            else:
                mean, std = series.mean(), series.std(ddof=0)
                if std == 0 or pd.isna(std):
                    mask = pd.Series(False, index=series.index)
                else:
                    mask = (series < mean - k * std) | (series > mean + k * std)
            rows.append((col, int(mask.sum()),
                         round(100 * mask.sum() / max(len(series), 1), 2)))

        out_df = pd.DataFrame(rows, columns=["Feature", "Outlier Count", "Outlier %"]) \
                    .sort_values("Outlier Count", ascending=False).reset_index(drop=True)

        positives = out_df[out_df["Outlier Count"] > 0]
        if positives.empty:
            fig_bar = go.Figure()
            fig_bar.update_layout(title=f"No outliers detected ({rule}, k={k})",
                                   **THEME, height=400)
        else:
            fig_bar = px.bar(
                positives, x="Feature", y="Outlier %",
                color="Outlier %", color_continuous_scale="Oranges",
                text="Outlier Count",
                title=f"Outliers detected ({rule}, k={k})",
            )
            fig_bar.update_layout(**THEME, height=400)
            fig_bar.update_traces(textposition="outside")

        fig_box = px.box(
            df, y=selected,
            title=f"Box plot — {selected}",
            color_discrete_sequence=["#58a6ff"],
            points="outliers",
        )
        fig_box.update_layout(**THEME, height=380)

        return out_df, fig_bar, fig_box

    init_df, init_bar, init_box = _compute("IQR", 1.5, num_cols[0])
    counts_table = gr.Dataframe(
        value=init_df, headers=list(init_df.columns),
        label="Outlier counts per feature", interactive=False,
    )
    counts_plot = gr.Plot(value=init_bar, label="Outlier % per feature")
    box_plot = gr.Plot(value=init_box, label="Box plot of selected feature")

    gr.on(
        triggers=[method.change, mult.change, feat.change],
        fn=_compute,
        inputs=[method, mult, feat],
        outputs=[counts_table, counts_plot, box_plot],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-tab 5 — Correlations
# ─────────────────────────────────────────────────────────────────────────────
CORRELATION_COLUMN_CAP = 30


def _correlation_columns(df: pd.DataFrame,
                         num_cols: list[str]) -> tuple[list[str], str]:
    """Choose which numeric columns the correlation heatmap shows.

    A cap is needed — a wide frame produces an unreadable matrix — but the cap
    used to take `num_cols[:30]`, i.e. whichever columns happened to come first
    in the CSV. That is a criterion nobody chose, and it matters because a user
    can read "no high-correlation pairs" off the resulting table and conclude
    something about the dataset. Column order carries no information about
    which correlations are worth seeing.

    Rank by variance instead: a constant or near-constant column cannot
    correlate with anything (its correlation is NaN), so spending a slot on one
    is strictly wasteful, and the highest-variance columns are where real
    structure lives. Returns the columns in their ORIGINAL order so the heatmap
    axes stay in a layout the user recognises from their file.

    Returns (columns, note) where note is "" when nothing was dropped.
    """
    if len(num_cols) <= CORRELATION_COLUMN_CAP:
        return num_cols, ""

    variances = df[num_cols].var(numeric_only=True)
    keep = set(variances.sort_values(ascending=False)
               .head(CORRELATION_COLUMN_CAP).index)
    cols = [c for c in num_cols if c in keep]      # original order
    dropped = [c for c in num_cols if c not in keep]
    note = (
        f"*Showing the {len(cols)} highest-variance numeric columns of "
        f"{len(num_cols)} — a full matrix would be unreadable. "
        f"**{len(dropped)} column(s) are not shown**, so absence of a "
        f"high-correlation pair below is not evidence there is none: "
        f"{', '.join(dropped[:8])}{' …' if len(dropped) > 8 else ''}.*"
    )
    return cols, note


def render_correlations(df: pd.DataFrame) -> None:
    num_cols = _numeric_cols(df)
    if len(num_cols) < 2:
        gr.Markdown("*Need ≥ 2 numeric columns for a correlation matrix.*")
        return

    cols, cap_note = _correlation_columns(df, num_cols)
    if cap_note:
        gr.Markdown(cap_note)

    with gr.Row():
        method = gr.Radio(
            choices=["pearson", "spearman", "kendall"], value="pearson",
            label="Correlation method",
            interactive=True,
        )
        threshold = gr.Slider(
            minimum=0.5, maximum=1.0, step=0.01, value=0.80,
            label="High-correlation threshold (|r|)",
            interactive=True,
        )

    def _compute(method_name: str, thr: float):
        corr = df[cols].corr(method=method_name)
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
            zmin=-1, zmax=1,
            title=f"{method_name.capitalize()} correlation heatmap",
        )
        fig.update_traces(textfont_size=9)
        fig.update_layout(**THEME, height=max(420, 22 * len(cols) + 180))

        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high = []
        for c1 in upper.columns:
            for c2 in upper.index:
                v = upper.loc[c2, c1]
                if pd.notna(v) and abs(v) >= thr:
                    high.append({"Feature 1": c1, "Feature 2": c2,
                                 "Correlation": round(float(v), 4)})
        pairs_df = (pd.DataFrame(high)
                    .sort_values("Correlation", key=lambda s: s.abs(),
                                 ascending=False)
                    .reset_index(drop=True)
                    if high else
                    pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation"]))
        return fig, pairs_df

    init_fig, init_pairs = _compute("pearson", 0.80)
    heat_plot = gr.Plot(value=init_fig, label="Correlation heatmap")
    pairs_table = gr.Dataframe(
        value=init_pairs, headers=list(init_pairs.columns),
        label="High-correlation pairs (above threshold)",
        interactive=False,
    )

    gr.on(
        triggers=[method.change, threshold.change],
        fn=_compute,
        inputs=[method, threshold],
        outputs=[heat_plot, pairs_table],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-tab 6 — LLM Decisions (Phi-4 + Mistral ensemble)
# ─────────────────────────────────────────────────────────────────────────────
_DECISION_FIELDS = [
    ("Imputation",         "imputation_strategy"),
    ("Power Transform",    "power_transform"),
    ("Outlier Method",     "outlier_method"),
    ("Outlier Threshold",  "outlier_threshold"),
    ("Correlation Threshold", "correlation_threshold"),
    ("Scaler",             "scaler"),
]


def _decision_row(d) -> list:
    """Pull the comparison-table values out of an LLMDecision dataclass."""
    return [getattr(d, key) for _, key in _DECISION_FIELDS]


def _build_agreement_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={"text": "Model agreement"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#58a6ff"},
            "steps": [
                {"range": [0,  50], "color": "#fbecec"},
                {"range": [50, 75], "color": "#fff7d6"},
                {"range": [75, 100], "color": "#e9f8ee"},
            ],
            "threshold": {
                "line": {"color": "#198754", "width": 3},
                "thickness": 0.8, "value": 75,
            },
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(template="plotly_white", height=240,
                       margin=dict(l=20, r=20, t=40, b=10))
    return fig


def render_llm_decisions(
    df: pd.DataFrame,
    default_target: str | None,
    initial_targets: list[str],
    targets_state: gr.State,
    ensemble_state: gr.State,
    pipeline_state: gr.State,
    transformed_state: gr.State,
) -> None:
    """LLM-ensemble sub-tab — run Phi-4 + Mistral on the raw df and show
    side-by-side decisions, agreement gauge, and reasoning."""

    gr.Markdown(
        "**Phi-4 + Mistral run in parallel** to pick preprocessing methods "
        "(imputation, power transform, outliers, scaler, encoding). Gemma2 "
        "tiebreaks per-field disagreements. The agreement gauge tells you "
        "how often the two models agreed before any tiebreak."
    )

    cols = df.columns.tolist()
    # Seed the multi-checkbox with the user's tab-1 selection. Fall back to
    # the single-default when tab 1 hasn't been touched yet.
    seed = [t for t in (initial_targets or []) if t in cols]
    if not seed and default_target in cols:
        seed = [default_target]
    with gr.Row():
        target_cb = gr.CheckboxGroup(
            choices=cols, value=seed,
            label="Target columns (excluded from features for the LLM analysis)",
            info="The LLM ensemble profiles the features only. Tick every "
                 "column that should be treated as an output. The first ticked "
                 "column is named as `AutoEDAPipeline.target_col`; all ticked "
                 "columns are dropped from the feature matrix and re-attached "
                 "as-is in the Transformed Data tab.",
            interactive=True,
        )
    with gr.Row():
        run_btn = gr.Button("Run Local LLM Ensemble Analysis", variant="primary")
    status_md = gr.Markdown()

    with gr.Row():
        gauge_plot = gr.Plot(label="Agreement gauge", show_label=False)
        confidence_md = gr.Markdown()

    gr.Markdown("#### Model decision comparison (highlighted rows differ)")
    comparison_table = gr.Dataframe(
        headers=["Decision", "Phi-4", "Mistral", "Ensemble Final"],
        interactive=False, wrap=True,
    )
    gr.Markdown("#### Per-field conflicts (empty when models agreed)")
    conflicts_table = gr.Dataframe(interactive=False, wrap=True)

    gr.Markdown("#### Model reasoning")
    with gr.Tabs():
        with gr.Tab("Phi-4"):
            phi4_md = gr.Markdown()
            phi4_raw = gr.Code(language="json", label="Raw Phi-4 response")
        with gr.Tab("Mistral"):
            mistral_md = gr.Markdown()
            mistral_raw = gr.Code(language="json", label="Raw Mistral response")
        with gr.Tab("Final JSON"):
            final_json = gr.JSON(label="Ensemble Final decision")

    def _run_llm(selected_targets: list | None, tab1_targets: list | None,
                  progress=gr.Progress(track_tqdm=False)):
        # Combine the in-tab checkbox selection with the tab-1 target_state so
        # the user can't accidentally let a tab-1 target leak through.
        all_targets = list(
            {t for t in (selected_targets or []) if t in df.columns}
            | {t for t in (tab1_targets or []) if t in df.columns}
        )
        if not all_targets:
            return ("Tick at least one target column.", None, "", pd.DataFrame(),
                    pd.DataFrame(), "", "", "", "", None, None, None)
        # The first ticked column is named as AutoEDAPipeline.target_col purely
        # to satisfy its API; the LLM ensemble sees only features regardless.
        primary = (selected_targets[0] if selected_targets else all_targets[0])
        try:
            progress(0.1, desc="Importing AutoEDAPipeline")
            try:
                from pipeline.orchestrator import AutoEDAPipeline   # noqa: WPS433
            except Exception as e:
                return (f"Could not import AutoEDAPipeline: {e}",
                        None, "", pd.DataFrame(), pd.DataFrame(),
                        "", "", "", "", None, None, None)

            # Ollama host — read from config if available, else default.
            ollama_host = "http://localhost:11434"
            try:
                import config as _cfg                                # noqa: WPS433
                ollama_host = getattr(_cfg, "OLLAMA_HOST", ollama_host)
            except Exception:
                pass

            progress(0.2, desc="Querying Phi-4 + Mistral via Ollama (this can take ~30-60s)")
            eda = AutoEDAPipeline(
                target_col=primary, task="regression",
                ollama_host=ollama_host, use_local_llm=True,
            )
            # Drop targets AND every known target "twin" (raw/log
            # counterparts) so the auto_eda transformer never sees them.
            # Otherwise the twins survive with a `num__` prefix that
            # sidesteps step4's name-based blocklist and the Pearson-|r|
            # feature ranker picks them as the strongest predictor (it's
            # effectively predicting the target with itself). See
            # config.TARGET_TWINS for the canonical list.
            try:
                import config as _cfg                                # noqa: WPS433
                twin_drop = [c for c in getattr(_cfg, "TARGET_TWINS", ())
                             if c in df.columns and c not in all_targets]
            except Exception:
                twin_drop = []
            X_only = df.drop(columns=list(all_targets) + twin_drop,
                             errors="ignore")
            eda.build_pipeline(X_only)
            # Stash the full target list on the pipeline object so the
            # Transformed Data sub-tab can re-attach every target column.
            eda._all_target_cols = all_targets
            # Stash the same twin-drop list so _do_transform (the next
            # sub-tab) uses an identical X to what build_pipeline saw.
            eda._twin_drop_cols = twin_drop
            ens = eda.ensemble_result_
            if ens is None:
                return ("LLM ensemble returned no decision.", None, "",
                        pd.DataFrame(), pd.DataFrame(), "", "", "", "",
                        None, None, None)
        except Exception as e:
            return (f"LLM ensemble failed: {e}", None, "",
                    pd.DataFrame(), pd.DataFrame(), "", "", "", "",
                    None, None, None)

        progress(0.9, desc="Rendering")
        gauge = _build_agreement_gauge(ens.agreement_score)

        # Report the twin drops too. The guard silently removed columns the
        # user can see in their own CSV (e.g. propane_conversion when
        # optimizing propylene_yield), which looks like data loss unless it is
        # named. See config.TARGET_TWINS for why they cannot stay.
        twin_note = (
            f"  \n**Target twins also excluded** (raw/log/algebraic siblings "
            f"that would leak the target into the features): "
            f"{', '.join(sorted(twin_drop))}"
            if twin_drop else ""
        )
        conf_md = (
            f"**Targets excluded from features**: "
            f"{', '.join(all_targets)} (primary = `{primary}`)"
            f"{twin_note}  \n"
            f"**Phi-4 confidence** {ens.phi4_decision.confidence:.0%}  ·  "
            f"latency {ens.phi4_decision.latency_ms:.0f} ms  \n"
            f"**Mistral confidence** {ens.mistral_decision.confidence:.0%}  ·  "
            f"latency {ens.mistral_decision.latency_ms:.0f} ms  \n"
            f"**Tiebreaker used** {'yes' if ens.tiebreak_used else 'no'}"
        )

        cmp_rows = []
        for label, key in _DECISION_FIELDS:
            cmp_rows.append([
                label,
                str(getattr(ens.phi4_decision, key)),
                str(getattr(ens.mistral_decision, key)),
                str(getattr(ens.final, key)),
            ])
        cmp_df = pd.DataFrame(
            cmp_rows,
            columns=["Decision", "Phi-4", "Mistral", "Ensemble Final"],
        )

        confl_df = (pd.DataFrame(ens.conflicts) if ens.conflicts
                    else pd.DataFrame(columns=["(no conflicts)"]))

        phi4_reasoning = (
            ens.phi4_decision.reasoning_summary or "_(no reasoning returned)_"
        )
        mistral_reasoning = (
            ens.mistral_decision.reasoning_summary or "_(no reasoning returned)_"
        )
        try:
            final_dict = asdict(ens.final)
        except Exception:
            final_dict = {"error": "could not serialize"}

        # Clear any previously-cached transformed_df so the Transformed Data
        # sub-tab is forced to re-run on the new pipeline.
        return (
            (f"✅ Ensemble decision ready. Agreement = "
             f"{ens.agreement_score:.0%}."),
            gauge, conf_md, cmp_df, confl_df,
            phi4_reasoning, ens.phi4_decision.raw_response or "",
            mistral_reasoning, ens.mistral_decision.raw_response or "",
            final_dict, ens, eda,
        )

    run_btn.click(
        _run_llm, inputs=[target_cb, targets_state],
        outputs=[
            status_md,
            gauge_plot, confidence_md,
            comparison_table, conflicts_table,
            phi4_md, phi4_raw,
            mistral_md, mistral_raw,
            final_json,
            ensemble_state, pipeline_state,
        ],
    ).then(
        lambda: None, outputs=[transformed_state],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-tab 7 — Transformed Data
# ─────────────────────────────────────────────────────────────────────────────
def render_transformed_data(
    df: pd.DataFrame,
    targets_state: gr.State,
    pipeline_state: gr.State,
    transformed_state: gr.State,
) -> None:
    """Apply the LLM-decided sklearn pipeline to the raw df and show the
    resulting feature matrix, downloads, and before/after distributions."""

    gr.Markdown(
        "Apply the **LLM-decided preprocessing pipeline** from the previous "
        "sub-tab and inspect the resulting feature matrix. The pipeline runs "
        "imputation → optional power transform → scaler on numeric cols, "
        "plus one-hot / ordinal encoding on categoricals as the ensemble "
        "decided."
    )
    with gr.Row():
        transform_btn = gr.Button("Transform Dataset", variant="primary")
        status_md = gr.Markdown()

    metrics_md = gr.Markdown()
    preview = gr.Dataframe(
        label="First 20 rows of the transformed matrix",
        interactive=False, wrap=True,
    )
    download = gr.File(label="Download transformed CSV")

    gr.Markdown("#### Before vs after — distribution comparison")
    with gr.Row():
        feature_dd = gr.Dropdown(
            label="Feature to compare", choices=[], value=None,
            allow_custom_value=True,
            interactive=True,
        )
    compare_plot = gr.Plot(show_label=False)

    def _do_transform(eda_obj, all_targets):
        if eda_obj is None:
            return (
                "Run the LLM ensemble first (the previous sub-tab).",
                "", None, None, gr.update(choices=[], value=None), None, None,
            )

        # Drop EVERY selected target before transforming. Prefer the list
        # stashed on the pipeline object (which matches what build_pipeline
        # saw); fall back to the live targets_state.
        drop_cols = list(getattr(eda_obj, "_all_target_cols", None)
                         or [c for c in (all_targets or [])
                             if c in df.columns])
        if eda_obj.target_col in df.columns and eda_obj.target_col not in drop_cols:
            drop_cols.append(eda_obj.target_col)
        # Also drop the target twins (raw/log counterparts) so the
        # transformer fits on the SAME column set build_pipeline saw.
        # Otherwise sklearn's ColumnTransformer either errors on unknown
        # columns or produces a different feature matrix from what build
        # was based on.
        twin_drop = list(getattr(eda_obj, "_twin_drop_cols", None) or [])
        if not twin_drop:
            try:
                import config as _cfg                                # noqa: WPS433
                twin_drop = [c for c in getattr(_cfg, "TARGET_TWINS", ())
                             if c in df.columns and c not in drop_cols]
            except Exception:
                twin_drop = []
        features_df = df.drop(columns=drop_cols + twin_drop, errors="ignore")

        try:
            tdf = eda_obj.get_transformed_df(features_df)
        except Exception as e:
            return (
                f"Transform failed: {e}",
                "", None, None, gr.update(choices=[], value=None), None, None,
            )

        # Re-attach the original (un-transformed) target values so multi-target
        # users see ALL their targets next to the transformed features.
        problem = row_alignment_error(len(df), len(tdf))
        if problem:
            return (
                problem,
                "", None, None, gr.update(choices=[], value=None), None, None,
            )

        tdf = tdf.reset_index(drop=True).copy()
        target_block = []
        for t in drop_cols:
            if t in df.columns:
                tdf[t] = df[t].reset_index(drop=True).values
                target_block.append(t)

        before_cols = len(df.columns)
        after_cols = len(tdf.columns)
        metrics = (
            f"**Features before** {before_cols}  ·  "
            f"**after** {after_cols}  ·  "
            f"**Δ** {after_cols - before_cols:+d}  ·  "
            f"**Rows** {len(tdf)}  ·  "
            f"**Targets re-attached** {', '.join(target_block) or '_(none)_'}"
        )

        out_path = Path(tempfile.gettempdir()) / "transformed_features.csv"
        tdf.to_csv(out_path, index=False)

        # Compare on columns that survive both frames (numeric only). After
        # transform, names may be prefixed (e.g. 'num__metal_loading_wt');
        # match by suffix.
        num_before = [c for c in df.select_dtypes(include=np.number).columns
                      if c not in drop_cols]
        match_map = {}
        for raw_name in num_before:
            for tname in tdf.columns:
                if tname == raw_name or tname.endswith(f"__{raw_name}"):
                    match_map[raw_name] = tname
                    break
        choices = list(match_map.keys())
        chosen = choices[0] if choices else None

        return (
            f"✅ Transformed to {after_cols} columns "
            f"(features + {len(target_block)} target column(s) re-attached).",
            metrics, tdf.head(20).round(4),
            str(out_path),
            gr.update(choices=choices, value=chosen),
            tdf, match_map,
        )

    match_state = gr.State(None)

    transform_btn.click(
        _do_transform, inputs=[pipeline_state, targets_state],
        outputs=[status_md, metrics_md, preview, download,
                 feature_dd, transformed_state, match_state],
    )

    def _on_feature(feature_name: str, tdf, match_map):
        if not feature_name or tdf is None or match_map is None:
            return None
        return _make_compare_fig(df, tdf, match_map, feature_name)

    feature_dd.change(
        _on_feature, inputs=[feature_dd, transformed_state, match_state],
        outputs=[compare_plot],
    )


def _make_compare_fig(df: pd.DataFrame, tdf: pd.DataFrame,
                       match_map: dict, feature_name: str):
    """Side-by-side before/after histograms for a single numeric feature."""
    raw_col = feature_name
    tcol = match_map.get(raw_col)
    if tcol is None or raw_col not in df.columns or tcol not in tdf.columns:
        return None
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Before — {raw_col}", f"After — {tcol}"],
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Histogram(x=df[raw_col].dropna(), marker_color="#58a6ff",
                      nbinsx=30, name="Before"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=tdf[tcol].dropna(), marker_color="#56d364",
                      nbinsx=30, name="After"),
        row=1, col=2,
    )
    fig.update_layout(template="plotly_white", height=360,
                       margin=dict(l=10, r=10, t=60, b=40),
                       showlegend=False,
                       title_text=f"Distribution of {raw_col}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Master entry point
# ─────────────────────────────────────────────────────────────────────────────
def render_eda_tab(
    df_state: gr.State,
    default_target_state: gr.State,
    targets_state: gr.State,
    ensemble_state: gr.State,
    pipeline_state: gr.State,
    transformed_state: gr.State,
) -> None:
    """Build the EDA tab's content. Components are re-built whenever `df_state`
    changes (e.g. a new CSV is loaded), via `@gr.render`. The persistent
    states (ensemble/pipeline/transformed) live OUTSIDE the render block so
    they survive rebuilds; they're cleared when df_state changes.

    Call this INSIDE a `with gr.Tab("..."):` block in the parent layout.
    """
    gr.Markdown(
        "Descriptive exploration + LLM-driven preprocessing decisions on the "
        "**raw** uploaded CSV (pre-matminer-featurization). The first five "
        "sub-tabs let you gut-check the dataset; the last two show how "
        "Phi-4 + Mistral pick the transformation pipeline and let you "
        "inspect / download the resulting feature matrix."
    )

    # New CSV → wipe stale LLM state.
    df_state.change(
        lambda: (None, None, None),
        outputs=[ensemble_state, pipeline_state, transformed_state],
    )

    @gr.render(inputs=[df_state, default_target_state, targets_state])
    def _render(df, default_target, targets):
        if df is None or len(df) == 0:
            gr.Markdown("*Load a CSV in tab 1 first.*")
            return

        with gr.Tabs():
            with gr.Tab("Overview"):
                render_overview(df)
            with gr.Tab("Missing Values"):
                render_missing(df)
            with gr.Tab("Distributions"):
                render_distributions(df)
            with gr.Tab("Outliers"):
                render_outliers(df)
            with gr.Tab("Correlations"):
                render_correlations(df)
            with gr.Tab("LLM Decisions"):
                render_llm_decisions(
                    df, default_target, targets or [], targets_state,
                    ensemble_state, pipeline_state, transformed_state,
                )
            with gr.Tab("Transformed Data"):
                render_transformed_data(
                    df, targets_state, pipeline_state, transformed_state,
                )
