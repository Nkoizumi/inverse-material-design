"""Discrete library generator for atomic-fraction catalyst BO.

Mirrors the role of `catalyst_library.build_library` (role-based path) and
`steels_library.build_steels_library` (single-formula path), but produces
rows in the atomic-fraction schema consumed by
`pipeline.catalyst_fraction_features.featurize_fractions`.

Sampling strategy: empirical (matches training data).
  * Support cation chosen with training-frequency weights (from the dominant
    cation per row in the training set).
  * Number of non-support metals per row sampled from the training
    distribution (typically 1-3).
  * Which metals: sampled without replacement, weighted by per-element
    training presence rate.
  * Per-metal atomic fractions: drawn from the empirical distribution of
    that element's values in training (when nonzero).
  * Total non-support metal capped at ``max_total_metal`` (default 0.05).
    Support fraction = 1 - total_metal.
  * Reaction conditions fixed at training median.

Stays close to in-distribution where the surrogate is honest. Won't
discover wholly new chemistry families — for that we'd add a uniform /
LHS-sampled tail (see deferred work).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_fraction_library(df_train: pd.DataFrame, *,
                            n: int = 10_000,
                            seed: int = 42,
                            element_cols: Sequence[str],
                            support_cations: Sequence[str],
                            condition_cols: Sequence[str],
                            max_total_metal: float = 0.05,
                            ) -> pd.DataFrame:
    """Build a DataFrame of `n` candidate catalysts in atomic-fraction schema.

    Output columns mirror the training CSV (after step1 normalization):
    every element in ``element_cols`` + every condition in ``condition_cols``.
    No target columns. Each row's element fractions sum to 1.0 within
    floating-point precision.
    """
    rng = np.random.default_rng(seed)

    metal_cols = [e for e in element_cols if e not in set(support_cations)]
    metal_cols_in_train = [e for e in metal_cols if e in df_train.columns]
    cations_in_train = [c for c in support_cations if c in df_train.columns]

    # ── 1. Support cation distribution ──────────────────────────────────
    cat_block = df_train[cations_in_train].fillna(0.0)
    dominant = cat_block.idxmax(axis=1)
    dominant = dominant.where(cat_block.max(axis=1) > 0.30, other=pd.NA)
    sup_freq = dominant.value_counts(normalize=True, dropna=True).to_dict()
    log.info("  Empirical support frequencies: %s",
             {k: f"{v:.2f}" for k, v in sup_freq.items()})
    sup_choices = list(sup_freq.keys())
    sup_weights = np.array([sup_freq[c] for c in sup_choices], dtype=float)
    if sup_weights.sum() == 0:
        raise ValueError("No support cation has any row in training above 30%.")
    sup_weights /= sup_weights.sum()

    # ── 2. Per-element presence rate and value distribution ─────────────
    presence_rate: dict[str, float] = {}
    value_pools: dict[str, np.ndarray] = {}
    for e in metal_cols_in_train:
        col = pd.to_numeric(df_train[e], errors="coerce").fillna(0.0).values
        nonzero = col[col > 0]
        if len(nonzero) > 0:
            presence_rate[e] = float(len(nonzero) / len(col))
            value_pools[e] = nonzero
    if not presence_rate:
        raise ValueError("No non-support element has any nonzero training value.")
    log.info("  Top-5 metal presence rates: %s",
             dict(sorted(presence_rate.items(),
                         key=lambda kv: -kv[1])[:5]))

    # ── 3. Number-of-metals distribution ────────────────────────────────
    n_metals_per_row = (df_train[metal_cols_in_train].fillna(0.0) > 0).sum(axis=1)
    # Cap at 4 — beyond that the BO loses physical interpretability quickly.
    n_metals_dist = n_metals_per_row.clip(upper=4).value_counts(normalize=True)
    # Force at least 1 metal per row (a row of pure support has nothing to optimize).
    n_metals_dist = n_metals_dist.drop(0, errors="ignore")
    n_metals_dist = n_metals_dist / n_metals_dist.sum()
    n_choices = n_metals_dist.index.to_numpy(dtype=int)
    n_weights = n_metals_dist.values
    log.info("  Number-of-metals distribution: %s",
             dict(zip(n_choices.tolist(), [f"{w:.2f}" for w in n_weights])))

    # ── 4. Reaction conditions at training median ───────────────────────
    cond_medians = {}
    for c in condition_cols:
        if c in df_train.columns:
            cond_medians[c] = float(
                pd.to_numeric(df_train[c], errors="coerce").median()
            )
    log.info("  Reaction conditions fixed at training median: %s", cond_medians)

    # ── 5. Sample n candidate rows ──────────────────────────────────────
    metal_choice_arr = np.array(list(presence_rate.keys()))
    metal_weight_arr = np.array([presence_rate[e] for e in metal_choice_arr])
    metal_weight_arr = metal_weight_arr / metal_weight_arr.sum()

    rows = []
    for _ in range(n):
        row = {e: 0.0 for e in element_cols}
        row.update(cond_medians)

        # Pick support
        sup = rng.choice(sup_choices, p=sup_weights)
        # Pick number of metals, then which ones (no replacement, weighted)
        k = int(rng.choice(n_choices, p=n_weights))
        k = min(k, len(metal_choice_arr))
        chosen_metals = rng.choice(metal_choice_arr, size=k, replace=False,
                                   p=metal_weight_arr)

        # Sample per-metal atomic fractions from the empirical pool of each
        fractions = np.array([
            float(rng.choice(value_pools[e])) for e in chosen_metals
        ])
        # Cap total at max_total_metal. Scale proportionally if exceeded.
        total = float(fractions.sum())
        if total > max_total_metal:
            fractions = fractions * (max_total_metal / total)
            total = max_total_metal

        for e, f in zip(chosen_metals, fractions):
            row[e] = float(f)
        row[sup] = 1.0 - total

        rows.append(row)

    out = pd.DataFrame(rows, columns=list(element_cols) + list(cond_medians.keys()))
    # Sanity: every row sums to 1 across the element columns.
    sums = out[list(element_cols)].sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-9):
        bad = int((np.abs(sums - 1.0) > 1e-9).sum())
        log.warning("  %d/%d rows have element fractions not summing to 1 "
                    "(max deviation %.2e). Investigate.",
                    bad, len(out), float((sums - 1.0).abs().max()))

    # Round to 6 decimal places. Floating-point drift after the cap-scaling
    # makes raw-fraction columns look messy in human inspection without
    # changing the chemistry meaningfully.
    for e in element_cols:
        out[e] = out[e].round(6)

    log.info("Built atomic-fraction library: %d rows, %d columns.",
             len(out), out.shape[1])
    return out
