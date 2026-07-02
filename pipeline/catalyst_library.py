"""Enumerate plausible heterogeneous catalyst formulations.

The library is the discrete search space for BO/MOBO in catalyst mode. Each row
is a (active_metal, promoter_1, promoter_2, support, loadings) tuple matching
the training CSV schema (without the target columns).

Filters:
    - promoter_2 may be non-empty only if promoter_1 is non-empty
    - promoter loading is 0 when the corresponding promoter is empty
    - promoter_1 and promoter_2 cannot be the same element (when both present)
"""
from __future__ import annotations

import logging
from itertools import product

import pandas as pd

log = logging.getLogger(__name__)


# Defaults — override via config or kwargs.
DEFAULT_ACTIVE_METALS  = ["Pt", "Pd", "Rh", "Ir", "Ru", "Ni"]
DEFAULT_PROMOTERS_1    = ["", "Sn", "Re", "Ga", "In"]
DEFAULT_PROMOTERS_2    = ["", "K", "Cs", "Ag"]
DEFAULT_METAL_LOADINGS = [0.3, 1.0, 2.0, 5.0]
DEFAULT_PROMO_LOADINGS = [0.3, 1.0]


def build_library(
    supports: list[str],
    role_cols: dict[str, str],
    loading_cols: dict[str, str],
    active_metals: list[str] | None = None,
    promoters_1: list[str] | None = None,
    promoters_2: list[str] | None = None,
    metal_loadings: list[float] | None = None,
    promo_loadings: list[float] | None = None,
) -> pd.DataFrame:
    """Cartesian product of plausible catalyst formulations.

    Returns a DataFrame with columns matching the training CSV schema.
    """
    active_metals  = active_metals  or DEFAULT_ACTIVE_METALS
    promoters_1    = promoters_1    or DEFAULT_PROMOTERS_1
    promoters_2    = promoters_2    or DEFAULT_PROMOTERS_2
    metal_loadings = metal_loadings or DEFAULT_METAL_LOADINGS
    promo_loadings = promo_loadings or DEFAULT_PROMO_LOADINGS

    metal_col = role_cols.get("active_metal", "active_metal")
    p1_col    = role_cols.get("promoter_1",  "promoter_1")
    p2_col    = role_cols.get("promoter_2",  "promoter_2")
    sup_col   = role_cols.get("support",     "support")
    m_load_c  = loading_cols.get("active_metal", "metal_loading_wt")
    p1_load_c = loading_cols.get("promoter_1",  "promoter_1_loading_wt")
    p2_load_c = loading_cols.get("promoter_2",  "promoter_2_loading_wt")

    rows = []
    for metal, p1, p2, sup in product(active_metals, promoters_1, promoters_2, supports):
        if p2 != "" and p1 == "":
            continue
        if p1 != "" and p1 == p2:
            continue

        p1_loads = [0.0] if p1 == "" else promo_loadings
        p2_loads = [0.0] if p2 == "" else promo_loadings

        for m_load, p1_load, p2_load in product(metal_loadings, p1_loads, p2_loads):
            rows.append({
                metal_col: metal,
                p1_col:    p1,
                p2_col:    p2,
                sup_col:   sup,
                m_load_c:  m_load,
                p1_load_c: p1_load,
                p2_load_c: p2_load,
            })

    df = pd.DataFrame(rows)
    log.info(
        "Built catalyst library: %d candidates from "
        "%d metals × %d p1 × %d p2 × %d supports × %d×%d×%d loadings.",
        len(df), len(active_metals), len(promoters_1), len(promoters_2),
        len(supports), len(metal_loadings), len(promo_loadings), len(promo_loadings),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config
    sup_lookup = pd.read_csv(config.SUPPORT_LOOKUP_PATH)
    lib = build_library(
        supports=sup_lookup["support"].tolist(),
        role_cols=config.CATALYST_ROLES,
        loading_cols=config.CATALYST_LOADINGS,
    )
    print(lib.head())
    print("Total candidates:", len(lib))
