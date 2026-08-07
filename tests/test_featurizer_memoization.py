"""The featurizers resolve each DISTINCT cell value once and broadcast the
result. These tests pin the semantics that optimization had to preserve.

Context: a BO library is a Cartesian product, so a 101,816-row role-based
library contains only 11 distinct active metals, 5 promoter-1 values, 6
promoter-2 values and 26 supports. Every feature block used to be recomputed
per row — 11.3 min to featurize the default library, which is why the shipped
presets all trim it. It is 5.6 s now. The risk of that change is a
broadcast/alignment bug silently pairing one catalyst's features with another
catalyst's row, so these tests check the mapping, not just the speed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from catalyst_features import (
    CatalystFeaturizer, LookupIndex, parse_components, weighted_lookup,
)


# ── LookupIndex semantics ────────────────────────────────────────────────────
@pytest.fixture
def lookup():
    return pd.DataFrame({
        "element": ["Pt", "Sn", "Ga", "Dup", "Dup"],
        "chi": [2.28, 1.96, np.nan, 1.0, 1.0],
        "phi": [5.65, 4.42, 4.20, 2.0, 2.0],
        "group": ["noble", "post-tm", "post-tm", "x", "x"],
    })


NUM = ["chi", "phi"]
CAT = ["group"]


def test_duplicate_key_is_treated_as_absent(lookup):
    """The original scan required `len(row) == 1`, so an ambiguous key
    contributed nothing. Preserved."""
    index = LookupIndex.build(lookup, "element", NUM, CAT)
    assert "Dup" not in index.numeric
    assert "Dup" not in index.categorical
    feats = weighted_lookup({"Dup": 1.0}, lookup, "element", NUM, CAT, index=index)
    assert np.isnan(feats["chi"]) and np.isnan(feats["phi"])
    assert pd.isna(feats["group"])


def test_nan_numeric_is_omitted_not_zero(lookup):
    """Ga has no chi. It must drop out of the weighted mean entirely rather
    than contribute a 0 — otherwise every Ga-containing alloy's chi is pulled
    toward zero."""
    index = LookupIndex.build(lookup, "element", NUM, CAT)
    assert "chi" not in index.numeric["Ga"]
    feats = weighted_lookup({"Pt": 0.5, "Ga": 0.5}, lookup, "element", NUM, CAT,
                            index=index)
    assert feats["chi"] == pytest.approx(2.28)          # Pt alone, reweighted
    assert feats["phi"] == pytest.approx(0.5 * 5.65 + 0.5 * 4.20)


def test_all_nan_numeric_yields_nan(lookup):
    feats = weighted_lookup({"Ga": 1.0}, lookup, "element", NUM, CAT)
    assert np.isnan(feats["chi"])


def test_categorical_uses_major_component(lookup):
    feats = weighted_lookup({"Pt": 0.7, "Sn": 0.3}, lookup, "element", NUM, CAT)
    assert feats["group"] == "noble"
    feats = weighted_lookup({"Pt": 0.3, "Sn": 0.7}, lookup, "element", NUM, CAT)
    assert feats["group"] == "post-tm"


def test_categorical_tie_breaks_on_insertion_order(lookup):
    """`max` returns the first maximal key. The optimized path must keep
    iterating the same components dict so ties resolve the same way."""
    assert weighted_lookup({"Pt": 0.5, "Sn": 0.5}, lookup, "element", NUM, CAT)["group"] == "noble"
    assert weighted_lookup({"Sn": 0.5, "Pt": 0.5}, lookup, "element", NUM, CAT)["group"] == "post-tm"


def test_empty_components_returns_empty_dict(lookup):
    assert weighted_lookup({}, lookup, "element", NUM, CAT) == {}


def test_prebuilt_index_matches_on_the_fly(lookup):
    """The whole optimization rests on these two paths agreeing."""
    index = LookupIndex.build(lookup, "element", NUM, CAT)
    for comps in ({"Pt": 1.0}, {"Pt": 0.6, "Sn": 0.4}, {"Ga": 1.0},
                  {"Pt": 0.3, "Ga": 0.3, "Sn": 0.4}, {"Unknown": 1.0}):
        a = weighted_lookup(comps, lookup, "element", NUM, CAT)
        b = weighted_lookup(comps, lookup, "element", NUM, CAT, index=index)
        assert a.keys() == b.keys()
        for k in a:
            if isinstance(a[k], float):
                assert (np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k]
            else:
                assert a[k] == b[k] or (pd.isna(a[k]) and pd.isna(b[k]))


# ── Broadcast correctness on the real featurizer ─────────────────────────────
@pytest.fixture
def featurizer():
    return CatalystFeaturizer(
        roles=config.CATALYST_ROLES,
        loadings=config.CATALYST_LOADINGS,
        support_lookup_path=config.SUPPORT_LOOKUP_PATH,
        metal_lookup_path=config.METAL_LOOKUP_PATH,
        optional_numeric_features=[],
    )


def _catalyst_rows() -> pd.DataFrame:
    """Deliberately repeats values across rows so the distinct-value cache is
    exercised, and interleaves them so an off-by-one broadcast shows up."""
    return pd.DataFrame({
        "active_metal": ["Pt", "Pd", "Pt", "Rh", "Pd", "Pt"],
        "promoter_1":   ["Sn", "",   "Ga", "Sn", "",   "Sn"],
        "promoter_2":   ["K",  "K",  "",   "",   "Cs", "K"],
        "support":      ["gamma-Al2O3", "SiO2", "gamma-Al2O3",
                         "ZrO2", "SiO2", "gamma-Al2O3"],
        "metal_loading_wt":       [1.0, 2.0, 1.0, 0.5, 2.0, 1.0],
        "promoter_1_loading_wt":  [0.3, 0.0, 0.3, 0.3, 0.0, 0.3],
        "promoter_2_loading_wt":  [0.3, 0.3, 0.0, 0.0, 1.0, 0.3],
    })


def test_identical_rows_get_identical_features(featurizer):
    """Rows 0, 2 and 5 differ only in promoter_1 (Sn/Ga/Sn); 0 and 5 are
    identical and must featurize identically."""
    out = featurizer.fit_transform(_catalyst_rows())
    numeric = out.select_dtypes(include=[np.number])
    pd.testing.assert_series_equal(
        numeric.iloc[0], numeric.iloc[5], check_names=False,
    )


def test_distinct_rows_get_distinct_features(featurizer):
    """Guards against the cache collapsing genuinely different catalysts —
    the failure mode that would make BO rank identical predictions."""
    out = featurizer.fit_transform(_catalyst_rows())
    numeric = out.select_dtypes(include=[np.number])
    assert not numeric.iloc[0].equals(numeric.iloc[1])
    assert not numeric.iloc[0].equals(numeric.iloc[2])
    assert not numeric.iloc[1].equals(numeric.iloc[3])


def test_row_order_is_preserved_under_broadcast(featurizer):
    """Featurizing a shuffled frame must give each row the same features it
    got in the original order — the alignment property a reindex bug breaks."""
    df = _catalyst_rows()
    order = [4, 1, 5, 0, 3, 2]
    a = featurizer.fit_transform(df).select_dtypes(include=[np.number])
    b = featurizer.fit_transform(
        df.iloc[order].reset_index(drop=True)
    ).select_dtypes(include=[np.number])
    for new_pos, orig_pos in enumerate(order):
        pd.testing.assert_series_equal(
            a.iloc[orig_pos], b.iloc[new_pos], check_names=False,
        )


def test_unparseable_cell_yields_absent_role(featurizer):
    df = _catalyst_rows()
    df.loc[1, "promoter_1"] = "!!not-an-element!!"
    out = featurizer.fit_transform(df)
    assert out.loc[1, "promoter_1_present"] == 0
    assert out.loc[0, "promoter_1_present"] == 1
    magpie = [c for c in out.columns if c.startswith("promoter_1_MagpieData")]
    assert magpie, "expected promoter_1 Magpie columns"
    assert out.loc[1, magpie].isna().all()


def test_empty_promoter_is_absent_not_zero(featurizer):
    out = featurizer.fit_transform(_catalyst_rows())
    assert out.loc[1, "promoter_1_present"] == 0
    assert out.loc[0, "promoter_1_present"] == 1


def test_alloy_binding_energies_depend_on_loadings(featurizer):
    """The alloy block caches on (element, loading) pairs, so changing only a
    loading must still move the composition-weighted binding energies.

    Uses Pt+Re: both have E_ads values in metal_properties.csv. Note that
    Pt+Sn would NOT work here — Sn's E_ads_* are all NaN, so it drops out of
    the weighted mean entirely and the result is Pt's value at any loading.
    That is intended behaviour (see test_nan_numeric_is_omitted_not_zero),
    not a caching artifact.
    """
    df = _catalyst_rows()
    df.loc[:, "promoter_1"] = ["Re", "", "Re", "Re", "", "Re"]
    df2 = df.copy()
    df2.loc[0, "promoter_1_loading_wt"] = 5.0

    a = featurizer.fit_transform(df)
    b = featurizer.fit_transform(df2)
    ads = [c for c in a.columns if c.startswith("alloy_E_ads_")]
    assert ads, "expected alloy_E_ads_* columns"
    assert not np.allclose(a.loc[0, ads].astype(float).values,
                           b.loc[0, ads].astype(float).values, equal_nan=True)
    # Rows sharing the ORIGINAL key must be untouched — this is what a cache
    # keyed too coarsely (e.g. on element identity alone) would break.
    np.testing.assert_allclose(a.loc[5, ads].astype(float).values,
                               b.loc[5, ads].astype(float).values)
    np.testing.assert_allclose(a.loc[3, ads].astype(float).values,
                               b.loc[3, ads].astype(float).values)


def test_alloy_dblock_fraction_tracks_promoter_loading(featurizer):
    """alloy_dblock_frac is the mole fraction carried by elements that HAVE
    binding data. Raising Sn's loading dilutes Pt, so it must fall — the one
    alloy column that responds to a data-less promoter."""
    df = _catalyst_rows()
    df2 = df.copy()
    df2.loc[0, "promoter_1_loading_wt"] = 5.0        # row 0 is Pt + Sn
    a = featurizer.fit_transform(df).loc[0, "alloy_dblock_frac"]
    b = featurizer.fit_transform(df2).loc[0, "alloy_dblock_frac"]
    assert b < a, f"expected dblock fraction to fall, got {a} -> {b}"


def test_single_row_matches_batch(featurizer):
    """A one-row frame and the same row inside a batch must agree — the cache
    must not depend on what else is in the frame."""
    df = _catalyst_rows()
    batch = featurizer.fit_transform(df).select_dtypes(include=[np.number])
    single = featurizer.fit_transform(
        df.iloc[[3]].reset_index(drop=True)
    ).select_dtypes(include=[np.number])
    common = [c for c in single.columns if c in batch.columns]
    pd.testing.assert_series_equal(
        batch.iloc[3][common], single.iloc[0][common], check_names=False,
    )
