"""Functional tests for step4_surrogate.prepare_xy.

The TARGET_TWINS blocklist and the constant-column drop are the two guards
that stand between this pipeline and a silently-inflated CV R². Until now only
the *spelling* of the blocklist was tested (tests/test_config_consistency.py);
these tests check that prepare_xy actually honours it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from step4_surrogate import prepare_xy


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """20 rows: one real feature, one perfectly-leaking twin, one constant."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=20)
    return pd.DataFrame({
        "propane_TOF_log": y,                       # the target
        "propane_TOF_s": 10.0 ** y,                 # algebraic twin — must be dropped
        "real_feature": y * 0.5 + rng.normal(scale=0.5, size=20),
        "noise_feature": rng.normal(size=20),
        "constant_feature": np.full(20, 3.0),       # zero variance — must be dropped
        "active_metal": ["Pt"] * 20,                # non-numeric — must be skipped
    })


@pytest.fixture
def single_target(monkeypatch):
    monkeypatch.setattr(config, "TARGET_COLS", ["propane_TOF_log"])
    monkeypatch.setattr(config, "MAX_GP_FEATURES", None)
    monkeypatch.setattr(config, "COMPOSITION_ONLY_FEATURES", False)
    monkeypatch.setattr(config, "PER_ROLE_FEATURE_CAP", False)
    monkeypatch.setattr(config, "OPTIONAL_NUMERIC_FEATURES", [])


def test_target_twin_is_excluded_from_features(toy_df, single_target):
    """The whole point of the blocklist: an algebraic twin of the target must
    never reach the feature matrix, even though it is numeric and is by far
    the highest-|Pearson r| column available."""
    data = prepare_xy(toy_df)
    assert "propane_TOF_s" not in data.feature_cols
    assert "propane_TOF_log" not in data.feature_cols
    assert "real_feature" in data.feature_cols


def test_twin_survives_the_feature_cap(toy_df, single_target, monkeypatch):
    """With MAX_GP_FEATURES=1 the ranker keeps exactly the top-|r| column. If
    the twin were still in the pool it would win outright — this asserts the
    drop happens BEFORE ranking, not after."""
    monkeypatch.setattr(config, "MAX_GP_FEATURES", 1)
    data = prepare_xy(toy_df)
    assert data.feature_cols == ["real_feature"]


def test_constant_column_is_dropped(toy_df, single_target):
    """A constant-in-training column carries no information but, if the BO
    library varies along it, division by the clamped x_std=1e-8 sends library
    rows ~1e9 sigma away and the GP returns the prior mean for everything."""
    data = prepare_xy(toy_df)
    assert "constant_feature" not in data.feature_cols
    assert (data.X.std(dim=0) > 0).all(), "a zero-variance column survived"


def test_non_numeric_column_is_skipped(toy_df, single_target):
    data = prepare_xy(toy_df)
    assert "active_metal" not in data.feature_cols


def test_rows_with_nan_targets_are_dropped(toy_df, single_target):
    df = toy_df.copy()
    df.loc[[2, 5, 9], "propane_TOF_log"] = np.nan
    data = prepare_xy(df)
    assert data.X.shape[0] == len(toy_df) - 3
    assert data.Y.shape[0] == len(toy_df) - 3
    assert not np.isnan(data.Y.cpu().numpy()).any()


def test_feature_cap_is_respected(toy_df, single_target, monkeypatch):
    monkeypatch.setattr(config, "MAX_GP_FEATURES", 2)
    data = prepare_xy(toy_df)
    assert len(data.feature_cols) <= 2
    assert data.X.shape[1] == len(data.feature_cols)


def test_standardization_round_trips(toy_df, single_target):
    """unstd_y must invert the standardization applied to Y."""
    data = prepare_xy(toy_df)
    recovered = data.unstd_y(data.Y).cpu().numpy().ravel()
    expected = toy_df["propane_TOF_log"].values
    np.testing.assert_allclose(recovered, expected, rtol=1e-9, atol=1e-9)


def test_optional_features_use_training_median_not_zero(toy_df, single_target,
                                                        monkeypatch):
    """Zero-filling a reaction condition (WHSV=0 → infinite residence time) is
    physically wrong and pushes those rows into the standardized tail. They
    must be median-imputed instead, and the median recorded for step 5 to
    reuse on the BO library."""
    monkeypatch.setattr(config, "OPTIONAL_NUMERIC_FEATURES", ["WHSV_h"])
    df = toy_df.copy()
    df["WHSV_h"] = [4.0] * 10 + [6.0] * 9 + [np.nan]
    data = prepare_xy(df)
    assert data.optional_medians["WHSV_h"] == pytest.approx(4.0)
    imputed = data.X[-1, data.feature_cols.index("WHSV_h")].item()
    unstd = imputed * data.x_std[data.feature_cols.index("WHSV_h")].item() \
        + data.x_mean[data.feature_cols.index("WHSV_h")].item()
    assert unstd == pytest.approx(4.0), "NaN was zero-filled instead of median-imputed"
