"""The COMPOSITION_ONLY_FEATURES filter must actually match the column names
the featurizers emit.

Its exclusion list tested `startswith("interfacial_")`, but both featurizers
emit `intf_*` (`intf_lattice_mismatch`, `intf_delta_work_function_eV`,
`intf_delta_chi`). The clause matched nothing; those columns were dropped by
the trailing catch-all instead. Harmless, but it read as though the interfacial
block were handled explicitly, and the next person to add a keep-rule above the
catch-all would have inherited a silent bug.
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
from catalyst_features import CatalystFeaturizer
from step4_surrogate import prepare_xy


@pytest.fixture(scope="module")
def featurized() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / "synthetic_catalysts.csv")
    cf = CatalystFeaturizer(
        roles=config.CATALYST_ROLES,
        loadings=config.CATALYST_LOADINGS,
        support_lookup_path=config.SUPPORT_LOOKUP_PATH,
        metal_lookup_path=config.METAL_LOOKUP_PATH,
        optional_numeric_features=[],
    )
    return cf.fit_transform(df)


@pytest.fixture
def composition_only(monkeypatch):
    monkeypatch.setattr(config, "TARGET_COLS", ["propane_TOF_log"])
    monkeypatch.setattr(config, "COMPOSITION_ONLY_FEATURES", True)
    monkeypatch.setattr(config, "MAX_GP_FEATURES", None)
    monkeypatch.setattr(config, "PER_ROLE_FEATURE_CAP", False)
    monkeypatch.setattr(config, "OPTIONAL_NUMERIC_FEATURES", [])


def test_featurizer_emits_intf_not_interfacial(featurized):
    """Pins the prefix the filter has to match. If a featurizer is ever renamed
    to `interfacial_*`, this fails and points at the filter."""
    assert [c for c in featurized.columns if c.startswith("intf_")]
    assert not [c for c in featurized.columns if c.startswith("interfacial_")]


def test_interfacial_columns_are_excluded(featurized, composition_only):
    data = prepare_xy(featurized)
    assert not [c for c in data.feature_cols if c.startswith("intf_")]


def test_physical_and_binding_columns_are_excluded(featurized, composition_only):
    data = prepare_xy(featurized)
    assert not [c for c in data.feature_cols if "_phys_" in c]
    assert not [c for c in data.feature_cols if "E_ads_" in c]
    assert "is_mixed_oxide" not in data.feature_cols


def test_composition_columns_are_kept(featurized, composition_only):
    """The filter is a keep-list, so confirm it does not drop everything."""
    data = prepare_xy(featurized)
    assert [c for c in data.feature_cols if "MagpieData" in c]
    assert [c for c in data.feature_cols if c.endswith("_present")]


def test_filter_is_a_strict_subset(featurized, composition_only, monkeypatch):
    """Composition-only must keep strictly fewer features than the full set."""
    full = prepare_xy(featurized).feature_cols
    monkeypatch.setattr(config, "COMPOSITION_ONLY_FEATURES", False)
    unfiltered = prepare_xy(featurized).feature_cols
    assert set(full) < set(unfiltered)
