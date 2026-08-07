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


# ── W3: schema detection drives the featurizer ───────────────────────────────
# The webui read CATALYST_MODE / CATALYST_FRACTION_MODE but never set them, so
# whichever values config.py held at import decided which featurizer ran.
def test_detects_role_schema():
    cols = ["active_metal", "promoter_1", "support", "metal_loading_wt", "y"]
    assert app.detect_schema(cols) == "role"


def test_detects_fraction_schema():
    elements = list(getattr(config, "CATALYST_FRACTION_ELEMENTS", []))[:20]
    assert app.detect_schema(elements + ["propylene yield"]) == "fraction"


def test_detects_single_formula_schema():
    assert app.detect_schema(["composition", "gap expt"]) == "single_formula"


def test_role_schema_wins_over_incidental_element_columns():
    """A role-based CSV that happens to carry element-named columns must not be
    mistaken for a fraction table."""
    cols = ["active_metal", "support", "Al", "Ga", "Mo", "Pt", "Sn", "Zr", "y"]
    assert app.detect_schema(cols) == "role"


def test_a_few_element_columns_do_not_trigger_fraction_mode():
    assert app.detect_schema(["composition", "Al", "Ga", "y"]) == "single_formula"


@pytest.mark.parametrize("filename,expected", [
    ("synthetic_catalysts.csv", "role"),
    ("pdh_literature.csv", "role"),
    ("pdh_ACSMaterialsLetters.csv", "fraction"),
])
def test_detects_the_real_bundled_datasets(filename, expected):
    path = ROOT / "data" / filename
    if not path.exists():
        pytest.skip(f"{filename} not present (not bundled / not downloaded)")
    assert app.detect_schema(pd.read_csv(path, nrows=0).columns) == expected


@pytest.mark.parametrize("schema,catalyst,fraction", [
    ("role", True, False),
    ("fraction", False, True),
    ("single_formula", False, False),
])
def test_apply_schema_mode_sets_both_flags(schema, catalyst, fraction, monkeypatch):
    monkeypatch.setattr(config, "CATALYST_MODE", None)
    monkeypatch.setattr(config, "CATALYST_FRACTION_MODE", None)
    app._apply_schema_mode(schema)
    assert config.CATALYST_MODE is catalyst
    assert config.CATALYST_FRACTION_MODE is fraction


def test_fraction_upload_is_flagged_to_the_user(tmp_path):
    """Fraction mode is only partially supported in the UI; the user must learn
    that at upload, not by reading a degraded report afterwards."""
    elements = list(getattr(config, "CATALYST_FRACTION_ELEMENTS", []))[:20]
    csv = tmp_path / "frac.csv"
    pd.DataFrame({c: [0.05] for c in elements} | {"y": [1.0]}).to_csv(csv, index=False)

    class _F:
        name = str(csv)

    msg = app.load_csv(_F(), use_synthetic=False)[2]
    assert "atomic-fraction" in msg
    assert "minimal report" in msg


def test_role_upload_is_not_flagged():
    msg = app.load_csv(None, use_synthetic=True)[2]
    assert "role-based" in msg
    assert "minimal report" not in msg


# ── W9: only numeric columns may be offered as targets ───────────────────────
def test_target_choices_exclude_non_numeric_columns():
    """A text column picked as a target failed minutes later inside prepare_xy
    rather than at the point of the mistake."""
    _preview, choices, _msg, df, *_ = app.load_csv(None, use_synthetic=True)
    offered = set(choices["choices"] if isinstance(choices, dict)
                  else choices.constructor_args["choices"])
    numeric = set(df.select_dtypes(include="number").columns)
    non_numeric = set(df.columns) - numeric
    assert non_numeric, "fixture has no non-numeric columns; test is vacuous"
    assert offered <= numeric
    assert not (offered & non_numeric)


# ── W10: minimize picks reset when the target set changes ────────────────────
def test_sync_targets_clears_stale_minimize_selection():
    """A stale minimize tick against a target no longer selected would drop
    silently out of OPTIMIZATION_DIRECTIONS."""
    picked, update = app._sync_targets(["a", "b"])
    assert picked == ["a", "b"]
    args = update["choices"] if isinstance(update, dict) else update.constructor_args["choices"]
    val = update["value"] if isinstance(update, dict) else update.constructor_args["value"]
    assert list(args) == ["a", "b"]
    assert val == []


def test_sync_targets_handles_none():
    picked, _ = app._sync_targets(None)
    assert picked == []


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
