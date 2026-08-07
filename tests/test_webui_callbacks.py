"""Tests for the webui callbacks that had no coverage at all.

webui/smoke_test.py exercises three endpoints against a RUNNING server, so it
never ran in CI and never touched Tab 5 or any error path. These call the
callbacks directly.

Importing webui/app.py builds the Gradio Blocks at module scope, so gradio must
be installed; the module skips cleanly when it isn't, keeping the rest of the
suite runnable on a pipeline-only install.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Gradio phones home on import unless this is set; do it before the import.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

pytest.importorskip("gradio", reason="webui tests need gradio installed")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "webui") not in sys.path:
    sys.path.insert(0, str(ROOT / "webui"))

import app  # noqa: E402
import config  # noqa: E402


# ── W1: load_csv output arity ────────────────────────────────────────────────
# The click handler declares six outputs. A branch returning any other number
# raises inside Gradio instead of showing its message.
N_OUTPUTS = 6


def test_missing_synthetic_csv_returns_full_arity(tmp_path, monkeypatch):
    """The bug: this path returned five values, so the 'not found' message the
    branch exists to display could never reach the user."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)      # no synthetic CSV here
    monkeypatch.setattr(app.config, "DATA_DIR", tmp_path)

    out = app.load_csv(None, use_synthetic=True)

    assert len(out) == N_OUTPUTS
    assert "not found" in out[2].lower()


def test_unreadable_upload_returns_full_arity(tmp_path):
    bad = tmp_path / "not_a.csv"
    bad.write_bytes(b"\x00\x01\x02 not a csv \xff\xfe")

    class _F:
        name = str(bad)

    out = app.load_csv(_F(), use_synthetic=False)
    assert len(out) == N_OUTPUTS
    assert isinstance(out[2], str) and out[2]


def test_empty_csv_returns_full_arity(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b\n")            # header only

    class _F:
        name = str(empty)

    out = app.load_csv(_F(), use_synthetic=False)
    assert len(out) == N_OUTPUTS
    assert "empty" in out[2].lower()


def test_successful_load_returns_full_arity():
    out = app.load_csv(None, use_synthetic=True)
    assert len(out) == N_OUTPUTS
    preview, _choices, msg, df, default_target, initial_targets = out
    assert "Loaded" in msg
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    assert default_target is None or default_target in df.columns
    assert all(t in df.columns for t in initial_targets)


def test_all_load_csv_paths_agree_on_arity(tmp_path, monkeypatch):
    """Any future branch must match too — this is the invariant, not the
    individual cases above."""
    monkeypatch.setattr(app.config, "DATA_DIR", tmp_path)
    assert len(app._load_csv_failure("x")) == N_OUTPUTS
    assert len(app.load_csv(None, use_synthetic=True)) == N_OUTPUTS


# ── W2: direction-aware sorting ──────────────────────────────────────────────
@pytest.fixture
def acs_directions(monkeypatch):
    monkeypatch.setattr(config, "TARGET_COLS",
                        ["propylene_yield", "deactivation_rate_log"])
    monkeypatch.setattr(config, "OPTIMIZATION_DIRECTIONS", ["max", "min"])


def test_minimize_target_sorts_best_first(acs_directions):
    """The bug: every column sorted descending, so a minimize target put the
    fastest-deactivating catalysts at the top of the handoff list."""
    assert app._sort_ascending_for("pred_deactivation_rate_log") is True


def test_maximize_target_sorts_descending(acs_directions):
    assert app._sort_ascending_for("pred_propylene_yield") is False


def test_uncertainty_columns_sort_tightest_first(acs_directions):
    assert app._sort_ascending_for("pred_propylene_yield_sd") is True
    assert app._sort_ascending_for("pred_deactivation_rate_log_sd") is True


def test_bnn_crosscheck_columns_follow_their_target(acs_directions):
    assert app._sort_ascending_for("pred_deactivation_rate_log_bnn") is True
    assert app._sort_ascending_for("pred_propylene_yield_bnn") is False


def test_raw_target_column_follows_its_direction(acs_directions):
    assert app._sort_ascending_for("deactivation_rate_log") is True
    assert app._sort_ascending_for("propylene_yield") is False


def test_unknown_column_defaults_to_descending(acs_directions):
    assert app._sort_ascending_for("some_other_column") is False


def test_filter_candidates_puts_best_minimize_row_first(acs_directions):
    df = pd.DataFrame({
        "active_metal": ["Pt", "Pd", "Rh"],
        "support": ["gamma-Al2O3"] * 3,
        "pred_deactivation_rate_log": [-1.0, -3.0, -2.0],   # -3 is best (min)
        "pred_propylene_yield": [0.1, 0.2, 0.3],
    })
    out = app.filter_candidates(df, None, None, "pred_deactivation_rate_log")
    assert out.iloc[0]["pred_deactivation_rate_log"] == pytest.approx(-3.0)
    assert out.iloc[0]["active_metal"] == "Pd"

    out = app.filter_candidates(df, None, None, "pred_propylene_yield")
    assert out.iloc[0]["pred_propylene_yield"] == pytest.approx(0.3)


def test_filter_candidates_filters_still_work(acs_directions):
    df = pd.DataFrame({
        "active_metal": ["Pt", "Pd", "Rh"],
        "support": ["SiO2", "gamma-Al2O3", "SiO2"],
        "pred_propylene_yield": [0.1, 0.2, 0.3],
    })
    assert set(app.filter_candidates(df, ["Pt", "Rh"], None, None)["active_metal"]) == {"Pt", "Rh"}
    assert set(app.filter_candidates(df, None, ["SiO2"], None)["support"]) == {"SiO2"}
    assert len(app.filter_candidates(df, ["Pt"], ["SiO2"], None)) == 1


def test_filter_candidates_handles_empty_input():
    assert app.filter_candidates(None, None, None, None).empty
    assert app.filter_candidates(pd.DataFrame(), None, None, None).empty


def test_filter_candidates_ignores_unknown_sort_column(acs_directions):
    df = pd.DataFrame({"a": [1, 2]})
    pd.testing.assert_frame_equal(app.filter_candidates(df, None, None, "nope"), df)


# ── W6: the composition line that never rendered ─────────────────────────────
def test_candidate_detail_shows_composition():
    """The bug: this line read row['_label'], which step6 adds to its report
    bundle and which is never present in candidates.parquet — so the
    composition never appeared in the detail panel."""
    df = pd.DataFrame({
        "active_metal": ["Pt"],
        "promoter_1": ["Sn"],
        "promoter_2": [""],
        "support": ["gamma-Al2O3"],
        "metal_loading_wt": [1.0],
        "promoter_1_loading_wt": [0.3],
        "promoter_2_loading_wt": [0.0],
        "pred_propane_TOF_log": [1.234],
    })
    md = app.candidate_detail(df, 0)
    assert "**Composition**" in md
    assert "Pt" in md and "Sn" in md and "gamma-Al2O3" in md
    assert "_label" not in md


def test_candidate_detail_bounds_and_empty_cases():
    df = pd.DataFrame({"active_metal": ["Pt"], "support": ["SiO2"]})
    assert "Run the pipeline first" in app.candidate_detail(None, 0)
    assert "Run the pipeline first" in app.candidate_detail(pd.DataFrame(), 0)
    assert "Pick a row" in app.candidate_detail(df, -1)
    assert "Pick a row" in app.candidate_detail(df, 99)
    assert "Pick a row" in app.candidate_detail(df, None)


def test_candidate_detail_fraction_schema_still_works(monkeypatch):
    monkeypatch.setattr(config, "CATALYST_FRACTION_ELEMENTS", ["Al", "Ga", "Pt"])
    monkeypatch.setattr(config, "CATALYST_FRACTION_SUPPORT_CATIONS", ["Al"])
    df = pd.DataFrame({"Al": [0.95], "Ga": [0.03], "Pt": [0.02],
                       "pred_propylene_yield": [0.42]})
    md = app.candidate_detail(df, 0)
    assert "Composition" in md and "Al(sup)" in md
