"""Generate a synthetic heterogeneous-catalyst CSV for end-to-end testing.

Schema (one row = one catalyst):
    active_metal              : str          (formula, may be an alloy like Pt0.9Sn0.1)
    promoter_1, promoter_2    : str          (single element, optional empty string)
    support                   : str          (lookup key or mixture syntax)
    metal_loading_wt          : float        (wt%)
    promoter_1_loading_wt     : float        (wt%, 0 if no promoter_1)
    promoter_2_loading_wt     : float        (wt%, 0 if no promoter_2)
    propane_TOF_log           : float        (synthetic target #1 — log10 TOF analogue)
    propane_selectivity       : float        (synthetic target #2 — propene selectivity, 0–1ish)

Distribution:
    - 60 rows with 0/1/2 promoters drawn at random
    -  6 rows with alloy active metals
    -  6 rows with mixed-oxide supports
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
LOOKUPS = ROOT / "data" / "lookups"
OUT_CSV = ROOT / "data" / "synthetic_catalysts.csv"

N_RANDOM = 60
N_ALLOY = 6
N_MIXED = 6

ACTIVE_METALS = ["Pt", "Pd", "Rh", "Ir", "Ru", "Ni"]
MAIN_PROMOTERS = ["Sn", "Re", "Ga", "In"]
DOPANTS = ["K", "Cs", "Ag", "Cu"]
ALLOY_TEMPLATES = [
    "Pt0.9Sn0.1", "Pt0.8Sn0.2", "Pt0.7Sn0.3",
    "Pd0.9Cu0.1", "Pt0.5Pd0.5", "Pt0.95Re0.05",
]
MIXED_SUPPORTS = [
    "CeO2+ZrO2",
    "CeO2:0.7+ZrO2:0.3",
    "MgO:0.3+gamma-Al2O3:0.7",
    "SiO2:0.5+gamma-Al2O3:0.5",
    "TiO2-anatase:0.5+ZrO2:0.5",
    "Y2O3:0.2+ZrO2:0.8",
]

METAL_LOADINGS = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
PROMO_LOADINGS = [0.1, 0.3, 0.5, 1.0, 2.0]


def main(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    metals = pd.read_csv(LOOKUPS / "metal_properties.csv")
    supports = pd.read_csv(LOOKUPS / "support_properties.csv")
    pure_supports = supports["support"].tolist()

    rows = []

    for _ in range(N_RANDOM):
        metal = rng.choice(ACTIVE_METALS)
        promoter_1, promoter_2 = _sample_promoters(rng)
        support = rng.choice(pure_supports)
        rows.append(_row(metal, promoter_1, promoter_2, support, rng, metals, supports))

    for _ in range(N_ALLOY):
        alloy = rng.choice(ALLOY_TEMPLATES)
        support = rng.choice(pure_supports)
        # Alloys often carry no extra promoter, sometimes a single dopant.
        promoter_1 = rng.choice(["", *DOPANTS])
        promoter_2 = ""
        rows.append(_row(alloy, promoter_1, promoter_2, support, rng, metals, supports))

    for _ in range(N_MIXED):
        metal = rng.choice(ACTIVE_METALS)
        promoter_1, promoter_2 = _sample_promoters(rng)
        support = rng.choice(MIXED_SUPPORTS)
        rows.append(_row(metal, promoter_1, promoter_2, support, rng, metals, supports))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows → {OUT_CSV}")
    print(df.head(10))
    print(df["propane_TOF_log"].describe())


def _sample_promoters(rng) -> tuple[str, str]:
    """Draw 0, 1, or 2 promoters with biased mix."""
    n = rng.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
    if n == 0:
        return "", ""
    p1 = rng.choice(MAIN_PROMOTERS)
    if n == 1:
        return p1, ""
    p2 = rng.choice(DOPANTS)
    return p1, p2


def _row(metal: str, promoter_1: str, promoter_2: str, support: str,
         rng, metals_df: pd.DataFrame, supports_df: pd.DataFrame) -> dict:
    m_load = float(rng.choice(METAL_LOADINGS))
    p1_load = 0.0 if promoter_1 == "" else float(rng.choice(PROMO_LOADINGS))
    p2_load = 0.0 if promoter_2 == "" else float(rng.choice(PROMO_LOADINGS))

    activity = _synthetic_target(metal, promoter_1, promoter_2, support,
                                 m_load, p1_load, p2_load, metals_df, supports_df, rng)
    selectivity = _synthetic_selectivity(metal, promoter_1, promoter_2, support,
                                         m_load, p1_load, p2_load, metals_df, supports_df, rng)
    return {
        "active_metal": metal,
        "promoter_1": promoter_1,
        "promoter_2": promoter_2,
        "support": support,
        "metal_loading_wt": m_load,
        "promoter_1_loading_wt": p1_load,
        "promoter_2_loading_wt": p2_load,
        "propane_TOF_log": activity,
        "propane_selectivity": selectivity,
    }


def _synthetic_target(metal, p1, p2, support, m_load, p1_load, p2_load,
                      metals_df, supports_df, rng) -> float:
    metal_d = _weighted_avg(metal, metals_df, "element", "d_band_center_eV", -2.0)
    metal_a = _weighted_avg(metal, metals_df, "element", "lattice_a_A", 3.9)
    sup_a   = _weighted_avg(support, supports_df, "support", "lattice_a_A", 5.0)
    acid    = _weighted_avg(support, supports_df, "support", "acidity_NH3_TPD_mmol_g", 0.3)
    base    = _weighted_avg(support, supports_df, "support", "basicity_CO2_TPD_mmol_g", 0.1)

    metal_score = -metal_d * 0.5
    p1_score = _promoter_score(p1, p1_load)
    p2_score = _promoter_score(p2, p2_load) * 0.6   # second promoter has smaller marginal effect
    acidity_score  = -((acid - 0.30) ** 2) / 0.05
    basicity_score = -base * 0.4
    mismatch = abs(metal_a - sup_a) / max(sup_a, 1e-6)
    mismatch_score = -mismatch * 2.0
    loading_score = -((np.log10(m_load + 1e-3) - 0.0) ** 2)
    noise = float(rng.normal(0.0, 0.15))

    return (metal_score + p1_score + p2_score + acidity_score
            + basicity_score + mismatch_score + loading_score + noise)


def _synthetic_selectivity(metal, p1, p2, support, m_load, p1_load, p2_load,
                           metals_df, supports_df, rng) -> float:
    """Propene selectivity (0–1ish). Trade-off vs activity:
    activity wants moderate acidity, selectivity wants LOW acidity (no cracking)."""
    acid = _weighted_avg(support, supports_df, "support", "acidity_NH3_TPD_mmol_g", 0.3)
    base = _weighted_avg(support, supports_df, "support", "basicity_CO2_TPD_mmol_g", 0.1)
    metal_a = _weighted_avg(metal, metals_df, "element", "lattice_a_A", 3.9)
    sup_a   = _weighted_avg(support, supports_df, "support", "lattice_a_A", 5.0)

    # Strong penalty from acid sites → cracking products
    acid_pen = -acid * 1.5
    # Basicity boosts selectivity (suppresses acid sites)
    base_score = base * 0.6
    # Sn promoter is the textbook selectivity booster
    p1_sel = (
        0.80 if p1 == "Sn"
        else 0.30 if p1 in ("Re", "Ga", "In")
        else 0.0
    ) * (1.0 - np.exp(-p1_load / 0.5))
    # Alkali dopants K/Cs further help selectivity by neutralizing acid sites
    p2_sel = (
        0.50 if p2 in ("K", "Cs")
        else 0.10 if p2 in ("Ag", "Cu")
        else 0.0
    ) * (1.0 - np.exp(-p2_load / 0.5))
    # Lower metal loading → fewer cracking sites
    load_sel = -0.30 * np.log10(m_load + 0.1)
    # Mild strain penalty (sintering → coke)
    mismatch = abs(metal_a - sup_a) / max(sup_a, 1e-6)
    mismatch_pen = -mismatch * 1.0
    noise = float(rng.normal(0.0, 0.08))

    raw = acid_pen + base_score + p1_sel + p2_sel + load_sel + mismatch_pen + noise
    # Map into ~[0, 1] for interpretability via sigmoid; pure numeric for surrogate.
    return float(1.0 / (1.0 + np.exp(-raw)))


def _promoter_score(name: str, load: float) -> float:
    if not name:
        return 0.0
    base = (
        0.80 if name == "Sn"
        else 0.45 if name in ("Re", "Ga", "In")
        else 0.30 if name in ("K", "Cs")
        else 0.20 if name in ("Ag", "Cu")
        else 0.10
    )
    return base * (1.0 - np.exp(-load / 0.5))


def _weighted_avg(cell: str, df: pd.DataFrame, key_col: str, value_col: str,
                  default: float) -> float:
    components = _inline_parse(cell, set(df[key_col].astype(str)))
    if not components:
        return default
    val, w_total = 0.0, 0.0
    for k, w in components.items():
        row = df[df[key_col].astype(str) == k]
        if len(row) == 1 and not pd.isna(row[value_col].iloc[0]):
            val += w * float(row[value_col].iloc[0])
            w_total += w
    return val / w_total if w_total > 0 else default


def _inline_parse(cell: str, lookup_keys: set[str]) -> dict[str, float]:
    s = str(cell).strip()
    if not s:
        return {}
    if s in lookup_keys:
        return {s: 1.0}
    if "+" in s:
        comps = {}
        for part in s.split("+"):
            if ":" in part:
                k, w = part.rsplit(":", 1)
                try:
                    comps[k.strip()] = float(w)
                except ValueError:
                    continue
            else:
                comps[part.strip()] = 1.0
        total = sum(comps.values())
        if total > 0:
            return {k: v / total for k, v in comps.items()}
        return {}
    try:
        from pymatgen.core.composition import Composition
        return {str(el): float(f) for el, f in Composition(s).fractional_composition.items()}
    except Exception:
        return {}


if __name__ == "__main__":
    main()
