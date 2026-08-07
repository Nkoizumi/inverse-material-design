"""Step 3: auto-EDA via the existing /home/nao/auto_eda Orchestrator.

Saves the results dict to disk so step 6 (report) can pull insights without
re-running the LLM ensemble.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

import config

log = logging.getLogger(__name__)


def run_eda(df: pd.DataFrame) -> dict:
    if not config.AUTO_EDA_AVAILABLE:
        log.info("AUTO_EDA_AVAILABLE=False; skipping step 3.")
        return {}

    # The Orchestrator lives in a separate project (see config.AUTO_EDA_PATH
    # for the resolution order). It used to be a hardcoded absolute path into
    # the author's home directory, so step 3 could never run on anyone else's
    # machine. Absence is not an error — the pipeline is fully usable without
    # auto-EDA, which is what `--skip eda` does explicitly.
    auto_eda_path: Path | None = getattr(config, "AUTO_EDA_PATH", None)
    if auto_eda_path is None:
        log.warning(
            "auto-EDA project not found; skipping step 3. Set "
            "INVERSE_DESIGN_AUTO_EDA_PATH to a checkout of "
            "github.com/Nkoizumi/llm-eda-mobo, place one alongside this "
            "repository, or run with `--skip eda`."
        )
        return {}

    if str(auto_eda_path) not in sys.path:
        sys.path.insert(0, str(auto_eda_path))

    try:
        from pipeline.orchestrator import Orchestrator  # noqa: E402
    except Exception as e:
        log.warning("Could not import auto_eda Orchestrator from %s (%s); "
                    "skipping EDA.", auto_eda_path, e)
        return {}

    # Auto-EDA takes a single target. For dual-target runs we EDA the primary one
    # and rely on the surrogate / MOBO step for joint analysis.
    target = config.TARGET_COLS[0]

    # Auto-EDA's sklearn pipeline expects all-numeric input. Keep the targets
    # plus any numeric features; drop role columns (active_metal etc.), any
    # Composition objects, AND any TARGET_TWINS (raw/log siblings of targets
    # that would target-leak through Orchestrator and the importance ranker).
    # All TARGET_COLS are retained so _feature_importances_per_target can fit
    # one model per target downstream.
    import numpy as np
    twin_cols = set(getattr(config, "TARGET_TWINS", set())) - set(config.TARGET_COLS)
    keep = [c for c in df.columns
            if c not in twin_cols
            and (c in config.TARGET_COLS or pd.api.types.is_numeric_dtype(df[c]))]
    work = df[keep].copy()
    # Identify feature columns up front — target NaNs must survive cleanup so
    # the importance ranker can drop missing-target rows per fit, and so we
    # can drop them from work_orch before handing to Orchestrator.
    feature_cols = [c for c in work.columns if c not in config.TARGET_COLS]
    # Clean inf/NaN that matminer can produce — features only.
    work[feature_cols] = (
        work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # Drop near-constant columns — auto_eda's SkewnessKurtosisCorrector applies
    # PowerTransformer(yeo-johnson) to highly-skewed features, which blows up to
    # inf on columns dominated by a single value (e.g. promoter_2 features when
    # most rows have no second promoter).
    variances = work[feature_cols].var()
    low_var = variances[variances < 1e-6].index.tolist()
    if low_var:
        log.info("Dropping %d near-constant feature columns before EDA.", len(low_var))
        work = work.drop(columns=low_var)
        feature_cols = [c for c in feature_cols if c not in low_var]
    # Cast bool columns to int (matminer + pd.get_dummies produce bool which
    # breaks numpy quantile interpolation).
    bool_cols = [c for c in feature_cols if work[c].dtype == bool]
    for c in bool_cols:
        work[c] = work[c].astype(int)
    # Percentile-clip remaining float features to tame extreme values that the
    # PowerTransformer can map to inf.
    float_cols = [c for c in feature_cols
                  if c not in bool_cols and work[c].dtype.kind == "f"]
    for c in float_cols:
        q_low, q_high = work[c].quantile([0.001, 0.999])
        if q_high > q_low:
            work[c] = work[c].clip(q_low, q_high)

    log.info("Running auto-EDA on target=%s (df %s)", target, work.shape)
    orch = Orchestrator(
        task=config.TASK,
        ollama_host=config.OLLAMA_HOST,
    )
    # Orchestrator only strips its `target_col` from features — drop the
    # OTHER TARGET_COLS so they don't leak into its feature matrix.
    other_targets = [t for t in config.TARGET_COLS if t != target and t in work.columns]
    work_orch = work.drop(columns=other_targets) if other_targets else work
    # Orchestrator's XGBoost rejects NaN in y — drop rows where the primary
    # target is missing (real PDH literature has ~2 rows missing TOF / ~15
    # missing deactivation per step4_surrogate docstring).
    n_before = len(work_orch)
    work_orch = work_orch.dropna(subset=[target]).reset_index(drop=True)
    if len(work_orch) < n_before:
        log.info("Dropped %d rows with NaN in primary target %s (kept %d).",
                 n_before - len(work_orch), target, len(work_orch))
    # pre_transformed=True skips auto_eda's internal Transformer pipeline
    # (yeo-johnson + outlier + correlation + polynomial + encoder). The
    # PowerTransformer step produces inf on our near-constant feature columns
    # despite upstream cleaning, so we hand it pre-cleaned data instead.
    results = orch.run(work_orch, target_col=target, pre_transformed=True)

    # Feature importances per target (auto_eda's XGBoost only returns CV metrics).
    # Re-fit a lightweight XGBoost per target purely for importance ranking.
    feature_cols_for_imp = [c for c in work.columns if c not in config.TARGET_COLS]
    importances = _feature_importances_per_target(
        work, feature_cols_for_imp, config.TARGET_COLS, top_k=15,
    )
    results["top_features_per_target"] = importances

    out_pkl = config.DATA_DIR / "eda_results.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)

    # Also dump a JSON-friendly summary for quick inspection.
    summary = _summarize(results)
    summary["top_features_per_target"] = importances
    (config.DATA_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    log.info("Saved EDA results to %s.", out_pkl)
    return results


def _feature_importances_per_target(work: pd.DataFrame, feature_cols: list[str],
                                    targets: list[str], top_k: int = 15) -> dict:
    """Fit one XGBoost regressor per target, return top-K feature names + scores.

    Uses gain-based importance (XGBoost default).
    """
    import xgboost as xgb
    out = {}
    for t in targets:
        if t not in work.columns:
            continue
        # XGBoost rejects NaN in y — drop missing-target rows per fit.
        sub = work[[t, *feature_cols]].dropna(subset=[t])
        if len(sub) < 5:
            log.warning("Target %s has %d non-NaN rows; skipping importance fit.",
                        t, len(sub))
            continue
        X = sub[feature_cols].values
        y = sub[t].values
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.7,
            random_state=config.RANDOM_STATE, n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X, y)
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        top = imp[imp > 0].sort_values(ascending=False).head(top_k)
        out[t] = [{"feature": name, "importance": float(v)} for name, v in top.items()]
        log.info("Top features for %s (n=%d): %s", t, len(sub),
                 ", ".join(f"{r['feature']}({r['importance']:.3f})" for r in out[t][:5]))
    return out


def _summarize(results: dict, *, _depth: int = 0, _max_depth: int = 2) -> dict:
    """JSON-friendly snapshot. Recurses up to _max_depth=2 into nested dicts so
    the orchestrator's per-model results and llm_summary survive in
    eda_summary.json — step6 reads that JSON (not the pickle) and feeds it to
    its LLM prompt. DataFrames are flattened via .to_dict() when cheap."""
    if not isinstance(results, dict):
        return {"type": str(type(results))}
    summary = {}
    for k, v in results.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            summary[k] = v
        elif isinstance(v, (list, tuple)) and len(v) < 20:
            summary[k] = list(v)
        elif isinstance(v, dict) and _depth < _max_depth:
            summary[k] = _summarize(v, _depth=_depth + 1, _max_depth=_max_depth)
        elif hasattr(v, "to_dict") and callable(v.to_dict):
            try:
                summary[k] = v.to_dict()
            except Exception:
                summary[k] = f"<{type(v).__name__}>"
        else:
            summary[k] = f"<{type(v).__name__}>"
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from step1_load import load_dataset
    from step2_featurize import featurize
    df = load_dataset()
    df = featurize(df)
    res = run_eda(df)
    print(_summarize(res))
