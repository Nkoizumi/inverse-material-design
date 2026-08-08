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


def build_scoring_library(train_df: "pd.DataFrame"):
    """The library exactly as step 5 will score it: built, featurized, and with
    reaction conditions filled from `train_df` medians.

    Step 4 needs this to know which features can actually discriminate
    candidates (see `dead_library_columns`), and step 5 needs it to score them.
    Both call this so the two views cannot drift — the alternative, each
    rebuilding its own, is how the `interfacial_`/`intf_` and duplicated-R²
    bugs happened.

    Returns ``(library_raw, library_feat)``.
    """
    import config
    from catalyst_features import CatalystFeaturizer

    sup_lookup = pd.read_csv(config.SUPPORT_LOOKUP_PATH)
    library_raw = build_library(
        supports=sup_lookup["support"].tolist(),
        role_cols=config.CATALYST_ROLES,
        loading_cols=config.CATALYST_LOADINGS,
        active_metals=getattr(config, "LIBRARY_ACTIVE_METALS", None),
        promoters_1=getattr(config, "LIBRARY_PROMOTERS_1", None),
        promoters_2=getattr(config, "LIBRARY_PROMOTERS_2", None),
        metal_loadings=getattr(config, "LIBRARY_METAL_LOADINGS", None),
        promo_loadings=getattr(config, "LIBRARY_PROMO_LOADINGS", None),
    )
    library_feat = CatalystFeaturizer(
        roles=config.CATALYST_ROLES,
        loadings=config.CATALYST_LOADINGS,
        support_lookup_path=config.SUPPORT_LOOKUP_PATH,
        metal_lookup_path=config.METAL_LOOKUP_PATH,
        optional_numeric_features=getattr(config, "OPTIONAL_NUMERIC_FEATURES", []),
    ).fit_transform(library_raw)

    # Same training-median fill step 5 applies (its block 2a). Kept here so the
    # variance check sees the values that will actually be scored: a
    # median-filled condition column is CONSTANT across the library, which is
    # precisely what makes it useless for ranking candidates.
    for col in list(getattr(config, "OPTIONAL_NUMERIC_FEATURES", [])):
        if col in train_df.columns and train_df[col].notna().any():
            library_feat[col] = float(pd.to_numeric(train_df[col],
                                                    errors="coerce").median())
            library_feat[f"{col}_present"] = 1

    return library_raw, library_feat


def dead_library_columns(library_feat: "pd.DataFrame") -> set[str]:
    """Feature columns that take the SAME value for every library candidate.

    Such a column cannot change the relative ranking of two candidates — it
    shifts every prediction by the same amount — so spending part of the
    `MAX_GP_FEATURES` budget on it is strictly wasted.

    This is the mirror of the constant-in-TRAINING drop in `prepare_xy`, and it
    matters just as much. Measured 2026-08-08 on pdh_literature: the Pearson-|r|
    ranker spent 26 of its 30 slots on `range` / `avg_dev` / `p-norm` statistics
    of `active_metal`, which are identically zero for any single-element
    composition. They varied in training only because that column also holds
    oxide phases (Cr2O3, Ga2O3, In2O3, Ga8Al2O15), so the ranker had really
    selected an "is the active phase an oxide?" detector — a proxy for data
    provenance. The library contains only pure metals, so all 26 were constant
    across it, 101,816 candidates collapsed to 10 distinct feature vectors, and
    promoters, support and loadings contributed nothing at all.
    """
    numeric = library_feat.select_dtypes(include="number")
    nunique = numeric.nunique(dropna=False)
    return set(nunique[nunique <= 1].index)


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
