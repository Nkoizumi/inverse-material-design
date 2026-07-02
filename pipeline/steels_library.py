"""Discrete search library for the matbench_steels regression target.

Random samples a chemically-plausible Fe-balanced alloy space and returns a
DataFrame with a `composition` column (pymatgen Composition objects) and a
`composition_str` reduced-formula column. Used by step5_inverse._run_steels_discrete
to give optimize_acqf_discrete an explicit library, avoiding the mode-collapse
of the continuous BO path.

Bounds are picked from the matbench_steels training distribution (slightly
extended for exploration). The sampling is uniform-then-reject rather than
LHS because the library is small enough that rejection sampling is fine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

# Per-element atomic-fraction sampling bounds. Extracted from matbench_steels
# training set (described percentiles) with ~10% headroom for exploration.
# Nb is essentially constant in the dataset; we hold it at the training median.
ELEMENT_BOUNDS = {
    "C":  (0.0,    0.020),
    "Mn": (0.0,    0.005),
    "Si": (0.0,    0.012),
    "Cr": (0.0,    0.200),
    "Ni": (0.0,    0.210),
    "Mo": (0.0,    0.060),
    "V":  (0.0,    0.010),
    "Co": (0.0,    0.200),
    "Al": (0.0,    0.040),
    "Ti": (0.0,    0.030),
    "Nb": (1e-4,   1e-4),
}

FE_MIN = 0.55
FE_MAX = 0.90


def build_steels_library(n_target: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Random Fe-balanced steel alloy compositions.

    Strategy:
      1. Sample 3 × n_target attempts uniformly per element from ELEMENT_BOUNDS.
      2. Fe = 1 − sum(others); reject when Fe ∉ [FE_MIN, FE_MAX].
      3. Take the first n_target accepted samples and wrap as pymatgen
         Composition objects.
    """
    rng = np.random.default_rng(seed)
    elements = list(ELEMENT_BOUNDS.keys())

    n_attempts = max(int(n_target * 3), n_target + 100)
    sampled = {
        el: rng.uniform(lo, hi, size=n_attempts)
        for el, (lo, hi) in ELEMENT_BOUNDS.items()
    }
    sums = np.zeros(n_attempts)
    for el in elements:
        sums += sampled[el]
    fe = 1.0 - sums
    accept_mask = (fe >= FE_MIN) & (fe <= FE_MAX)
    accept_idx = np.where(accept_mask)[0][:n_target]

    if len(accept_idx) < n_target:
        # Very loose bounds → almost everything passes; if not, just take what we have.
        pass

    compositions: list[Composition] = []
    for i in accept_idx:
        amounts = {"Fe": float(fe[i])}
        for el in elements:
            v = float(sampled[el][i])
            if v > 0:
                amounts[el] = v
        compositions.append(Composition(amounts))

    return pd.DataFrame({
        "composition": compositions,
        "composition_str": [c.reduced_formula for c in compositions],
    })
