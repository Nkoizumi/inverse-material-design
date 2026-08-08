"""Features constant across the BO library are dropped before ranking.

`prepare_xy` already drops features constant in TRAINING — they carry no signal
to fit on. This is the mirror: a feature constant across the LIBRARY shifts every
candidate's prediction by the same amount, so it cannot rank them, and any slot
it occupies in the MAX_GP_FEATURES budget is wasted.

Both drops are needed, and missing this one was expensive. Measured 2026-08-08 on
pdh_literature: the Pearson-|r| ranker spent all 30 slots on `active_metal`
features, 26 of them `range` / `avg_dev` / p-norm statistics that are identically
zero for a single-element composition. They varied in training only because that
column also holds oxide phases (Cr2O3, Ga2O3, In2O3, Ga8Al2O15), so the ranker
had really selected an "is the active phase an oxide?" detector — a proxy for
which of two data provenances a row came from.

The library holds only pure metals, so all 26 were constant across it. 101,816
candidates collapsed to 10 distinct feature vectors, promoters/support/loadings
contributed nothing, and BO returned 9 catalysts for a batch size of 20. After
the fix: 66 distinct vectors, 4 promoter and 2 support features in the budget,
and a full batch of 20.
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
from catalyst_library import dead_library_columns
from step4_surrogate import prepare_xy


def test_a_column_with_one_value_across_the_library_is_dead():
    lib = pd.DataFrame({"varies": [1.0, 2.0, 3.0], "constant": [7.0, 7.0, 7.0]})

    assert dead_library_columns(lib) == {"constant"}


def test_the_range_statistic_of_single_element_candidates_is_dead():
    """The real shape of the bug: Magpie `range`/`avg_dev` are identically zero
    for a one-element composition, which is every role-based library candidate.
    """
    lib = pd.DataFrame({
        "active_metal_MagpieData range Number": [0.0, 0.0, 0.0],
        "active_metal_MagpieData mean Number": [78.0, 46.0, 45.0],
    })

    assert dead_library_columns(lib) == {"active_metal_MagpieData range Number"}


def test_non_numeric_columns_are_not_reported():
    lib = pd.DataFrame({"support": ["SiO2", "SiO2"], "x": [1.0, 2.0]})

    assert dead_library_columns(lib) == set()


def test_an_all_nan_column_counts_as_dead():
    """nunique(dropna=False) treats it as one value — correctly, since it cannot
    discriminate candidates either.
    """
    lib = pd.DataFrame({"never_measured": [float("nan")] * 3, "x": [1.0, 2.0, 3.0]})

    assert dead_library_columns(lib) == {"never_measured"}


# ── prepare_xy honours the set ───────────────────────────────────────────────
@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(config, "TARGET_COLS", ["y"])
    monkeypatch.setattr(config, "TARGET_TWINS", set())
    monkeypatch.setattr(config, "MAX_GP_FEATURES", None)
    monkeypatch.setattr(config, "COMPOSITION_ONLY_FEATURES", False)
    monkeypatch.setattr(config, "OPTIONAL_NUMERIC_FEATURES", [])
    return pd.DataFrame({
        "useful": [1.0, 2.0, 3.0, 4.0],
        "dead_in_library": [1.0, 5.0, 2.0, 9.0],   # varies in TRAINING
        "y": [1.0, 2.0, 3.0, 4.0],
    })


def test_a_library_dead_column_is_dropped_even_though_training_varies(toy):
    """The whole point: varying in training is not sufficient. If the library is
    constant on it, it cannot rank candidates.
    """
    data = prepare_xy(toy, {"dead_in_library"})

    assert "dead_in_library" not in data.feature_cols
    assert "useful" in data.feature_cols


def test_without_the_argument_nothing_extra_is_dropped(toy):
    """Non-catalyst paths and existing callers must be unaffected."""
    data = prepare_xy(toy)

    assert set(data.feature_cols) == {"useful", "dead_in_library"}


def test_an_empty_set_is_the_same_as_not_passing_one(toy):
    assert set(prepare_xy(toy, set()).feature_cols) == set(prepare_xy(toy).feature_cols)


def test_dropping_everything_keeps_the_training_features(toy):
    """Degenerate guard: if every feature is constant across the library we
    still have to be able to fit a surrogate, so the drop is skipped rather than
    leaving zero columns.
    """
    data = prepare_xy(toy, {"useful", "dead_in_library"})

    assert len(data.feature_cols) > 0
