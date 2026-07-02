"""Step 1: load a dataset (matminer benchmark or user CSV).

Returns a DataFrame with at least the formula column and target column(s).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

# Targets whose physical distribution typically spans 2+ orders of magnitude
# and should be modeled in log space (heteroscedastic, right-skewed). Each
# entry maps a raw column → derived log10(x + epsilon) column. The user can
# then select the _log target in the UI and pair it with direction="min" /
# "max" as appropriate.
_LOG_DERIVED_TARGETS = {
    "deactivation_rate_h": ("deactivation_rate_log", 1e-4),
}


def load_dataset() -> pd.DataFrame:
    if config.DATASET_SOURCE == "matminer":
        df = _load_matminer(config.MATMINER_DATASET)
    elif config.DATASET_SOURCE == "csv":
        if config.CSV_PATH is None:
            raise ValueError("DATASET_SOURCE='csv' but CSV_PATH is None.")
        df = pd.read_csv(config.CSV_PATH)
    else:
        raise ValueError(f"Unknown DATASET_SOURCE: {config.DATASET_SOURCE}")

    if getattr(config, "CATALYST_FRACTION_MODE", False):
        df = _normalize_atomic_fraction_csv(df)

    df = _add_log_derived_targets(df)

    missing = [c for c in config.TARGET_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Targets {missing} not in columns: {df.columns.tolist()}")

    if config.SUBSAMPLE_N and len(df) > config.SUBSAMPLE_N:
        df = df.sample(n=config.SUBSAMPLE_N, random_state=config.RANDOM_STATE).reset_index(drop=True)
        log.info("Subsampled to %d rows.", len(df))

    cache = config.DATA_DIR / "raw.parquet"
    df.to_parquet(cache)
    log.info("Saved raw dataset to %s (%d rows, %d cols).", cache, len(df), df.shape[1])
    return df


def _add_log_derived_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-emit log10(x+ε) versions of known wide-range targets.

    Catalyst targets like deactivation rate span 3+ orders of magnitude
    (industrial Pt-Sn ~1e-3 /h, Cr Catofin ~1 /h). Modeling them on the raw
    scale lets bad-catalyst outliers dominate y_std, so the GP wastes
    capacity discriminating "is it the bad one" instead of resolving
    improvements among the good ones. log10 + small ε absorbs reported
    zeros and makes the BO loss landscape uniform across scales.
    """
    for raw, (derived, eps) in _LOG_DERIVED_TARGETS.items():
        if raw not in df.columns or derived in df.columns:
            continue
        col = pd.to_numeric(df[raw], errors="coerce")
        if col.notna().sum() == 0:
            continue
        df[derived] = np.log10(col.clip(lower=0) + eps)
        n_finite = int(df[derived].notna().sum())
        log.info("Derived %s = log10(%s + %.0e) (%d non-NaN values).",
                 derived, raw, eps, n_finite)
    return df


def _normalize_atomic_fraction_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Apply CATALYST_FRACTION_RENAME and drop dataset-internal IDs.

    Headers with units in brackets (e.g. `propylene yield`, `deactivation rate
    constant [h-1]`) become awkward to handle downstream — sklearn pipelines
    and pandas all-numeric checks both choke on column names with spaces and
    brackets. We rename to snake_case at load time and drop dataset-internal
    identifier columns (index, rxn_id, cat_id) so they don't accidentally
    leak as numeric features.
    """
    rename = {k: v for k, v in getattr(config, "CATALYST_FRACTION_RENAME", {}).items()
              if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
        log.info("Renamed %d atomic-fraction columns to snake_case: %s",
                 len(rename), list(rename.values()))
    drop = [c for c in ("index", "rxn_id", "cat_id") if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
        log.info("Dropped dataset-internal identifier columns: %s", drop)

    # The ACS Materials Letters dataset includes a `score` column — a
    # paper-internal scalarization of (yield, deactivation_rate). Empirically
    # r ≈ +0.74 against (1 - d/d.max()) × yield, i.e. it's a derived target.
    # Letting it through as a feature lets the Pearson-|r| feature ranker
    # pick it (it consistently shows up in the top-5 kept) and the GP fits
    # the targets with their own scalarization → silently inflated CV R².
    if "score" in df.columns:
        df = df.drop(columns=["score"])
        log.info("Dropped 'score' column (derived from targets; would target-leak).")
    return df


def _load_matminer(name: str) -> pd.DataFrame:
    from matminer.datasets import load_dataset

    df = load_dataset(name)
    # Matbench composition datasets have a "composition" column of pymatgen Composition objects.
    # Step 2 expects the same column name; we keep it.
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = load_dataset()
    print(df.head())
    print(df.dtypes)
