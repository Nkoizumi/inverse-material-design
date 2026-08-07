"""The config namespace is validated rather than merely assumed.

`config` is a module whose globals are mutated by presets and by the web UI.
Nothing declared what may be set or checked that a combination made sense, and
that produced real bugs:

  * the web UI read CATALYST_MODE / CATALYST_FRACTION_MODE but never set them,
    so an atomic-fraction upload ran through the role-based featurizer — and
    bypassed the `score` target-leak guard along with it;
  * five settings were read via `getattr(config, ..., default)` and declared
    nowhere, so a typo in a reader or a preset fell back to the default in
    silence;
  * five more were declared and read nowhere, including
    `CHEM_VIABILITY_FILTER = True  # SMACT charge-balance`, which advertised a
    chemical-viability screen that does not exist.

These tests pin the rules that catch that class of thing.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from config_schema import SETTINGS, check, validate
from run_pipeline import PRESETS


def _cfg(**over):
    """A minimal valid config, overridable per-case."""
    base = dict(
        DATASET_SOURCE="csv", CSV_PATH=Path("x.csv"),
        TARGET_COLS=["a"], OPTIMIZATION_DIRECTIONS=["max"],
        TARGET_TWINS={"a"}, CATALYST_MODE=False, CATALYST_FRACTION_MODE=False,
        BO_BATCH_SIZE=20, CV_FOLDS=5,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── the shipped config and every preset must be consistent ───────────────────
def test_shipped_config_is_valid():
    assert validate(config) == []


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_is_valid(name, monkeypatch):
    """A preset that leaves config inconsistent would fail minutes into a run,
    or not at all."""
    for k, v in PRESETS[name].items():
        monkeypatch.setattr(config, k, v, raising=False)
    assert validate(config) == []


# ── the schema-mode rule (the W3 class) ──────────────────────────────────────
def test_both_schema_modes_on_is_rejected():
    problems = validate(_cfg(CATALYST_MODE=True, CATALYST_FRACTION_MODE=True,
                             CATALYST_ROLES={"active_metal": "active_metal"},
                             CATALYST_FRACTION_ELEMENTS=["Al"],
                             CATALYST_FRACTION_SUPPORT_CATIONS=[],
                             TARGET_TWINS={"a"}))
    assert any("both True" in p for p in problems)


def test_exactly_one_schema_mode_is_fine():
    assert validate(_cfg(CATALYST_MODE=True,
                         CATALYST_ROLES={"active_metal": "active_metal"},
                         CATALYST_LOADINGS={})) == []


def test_fraction_support_cation_outside_the_element_panel_is_rejected():
    problems = validate(_cfg(
        CATALYST_FRACTION_MODE=True,
        CATALYST_FRACTION_ELEMENTS=["Al", "Ga"],
        CATALYST_FRACTION_SUPPORT_CATIONS=["Al", "Zr"],   # Zr not in panel
        CATALYST_FRACTION_SUPPORT_OXIDE_MAP={"Al": "gamma-Al2O3", "Zr": "ZrO2"},
    ))
    assert any("never be detected" in p for p in problems)


def test_fraction_support_cation_without_an_oxide_is_rejected():
    problems = validate(_cfg(
        CATALYST_FRACTION_MODE=True,
        CATALYST_FRACTION_ELEMENTS=["Al", "Zr"],
        CATALYST_FRACTION_SUPPORT_CATIONS=["Al", "Zr"],
        CATALYST_FRACTION_SUPPORT_OXIDE_MAP={"Al": "gamma-Al2O3"},
    ))
    assert any("no entry in" in p for p in problems)


def test_loading_for_an_undeclared_role_is_rejected():
    problems = validate(_cfg(
        CATALYST_MODE=True,
        CATALYST_ROLES={"active_metal": "active_metal"},
        CATALYST_LOADINGS={"promoter_9": "promoter_9_loading_wt"},
    ))
    assert any("never read" in p for p in problems)


# ── raw-vs-renamed targets ───────────────────────────────────────────────────
def test_raw_header_target_is_named_precisely_not_as_a_leak():
    """A target given in its pre-rename form used to be reported as a
    TARGET_TWINS leak, which sent a user looking in entirely the wrong place.
    The real fault is that the column will not exist after step 1."""
    problems = validate(_cfg(
        CATALYST_FRACTION_MODE=True,
        CATALYST_FRACTION_ELEMENTS=["Al"],
        CATALYST_FRACTION_SUPPORT_CATIONS=[],
        CATALYST_FRACTION_RENAME={"propylene yield": "propylene_yield"},
        TARGET_COLS=["propylene yield"], OPTIMIZATION_DIRECTIONS=["max"],
        TARGET_TWINS={"propylene_yield"},
    ))
    assert any("RAW CSV header names" in p for p in problems)
    assert any("propylene_yield" in p for p in problems)
    # and it must NOT also be reported as a leak — that was the misleading part
    assert not any("TARGET_TWINS" in p for p in problems)


def test_rate_h_target_points_at_the_log_form():
    problems = validate(_cfg(
        CATALYST_FRACTION_MODE=True,
        CATALYST_FRACTION_ELEMENTS=["Al"],
        CATALYST_FRACTION_SUPPORT_CATIONS=[],
        CATALYST_FRACTION_RENAME={"deactivation rate constant [h-1]": "deactivation_rate_h"},
        TARGET_COLS=["deactivation rate constant [h-1]"],
        OPTIMIZATION_DIRECTIONS=["min"],
        TARGET_TWINS={"deactivation_rate_h"},
    ))
    assert any("deactivation_rate_log" in p for p in problems)


def test_renamed_targets_validate_clean():
    assert validate(_cfg(
        CATALYST_FRACTION_MODE=True,
        CATALYST_FRACTION_ELEMENTS=["Al"],
        CATALYST_FRACTION_SUPPORT_CATIONS=[],
        CATALYST_FRACTION_RENAME={"propylene yield": "propylene_yield"},
        TARGET_COLS=["propylene_yield"], OPTIMIZATION_DIRECTIONS=["max"],
        TARGET_TWINS={"propylene_yield"},
    )) == []


# ── the target-leak guard, enforced at runtime rather than only in tests ─────
def test_target_missing_from_twins_is_rejected():
    problems = validate(_cfg(CATALYST_MODE=True,
                             CATALYST_ROLES={"active_metal": "active_metal"},
                             TARGET_COLS=["a", "brand_new"],
                             OPTIMIZATION_DIRECTIONS=["max", "max"],
                             TARGET_TWINS={"a"}))
    assert any("brand_new" in p and "TARGET_TWINS" in p for p in problems)


# ── targets and directions ───────────────────────────────────────────────────
def test_direction_count_must_match_targets():
    problems = validate(_cfg(TARGET_COLS=["a", "b"], OPTIMIZATION_DIRECTIONS=["max"]))
    assert any("one-to-one" in p for p in problems)


def test_direction_values_must_be_max_or_min():
    problems = validate(_cfg(OPTIMIZATION_DIRECTIONS=["maximise"]))
    assert any("'max' or 'min'" in p for p in problems)


def test_empty_targets_rejected():
    problems = validate(_cfg(TARGET_COLS=[], OPTIMIZATION_DIRECTIONS=[]))
    assert any("nothing to optimize" in p for p in problems)


# ── typos ────────────────────────────────────────────────────────────────────
def test_unknown_setting_is_flagged():
    """The failure mode: a reader does getattr(config, 'MAX_GP_FEATURES', None),
    the user sets MAX_GP_FEATURE, and the cap silently never applies."""
    problems = validate(_cfg(MAX_GP_FEATURE=30))
    assert any("MAX_GP_FEATURE" in p and "typo" in p for p in problems)


def test_wrong_type_is_flagged():
    problems = validate(_cfg(BO_BATCH_SIZE="twenty"))
    assert any("BO_BATCH_SIZE" in p for p in problems)


# ── numeric ranges ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("field,value", [
    ("BO_BATCH_SIZE", 0), ("BO_UNIQUE_OVERSAMPLE", 0),
    ("REPORT_TOP_K", 0), ("BNN_PREDICT_SAMPLES", 0),
])
def test_positive_int_fields(field, value):
    assert validate(_cfg(**{field: value}))


def test_cv_folds_of_one_is_rejected():
    problems = validate(_cfg(CV_FOLDS=1))
    assert any("held-out split" in p for p in problems)


@pytest.mark.parametrize("folds", [0, 2, 5])
def test_valid_cv_folds_accepted(folds):
    assert validate(_cfg(CV_FOLDS=folds)) == []


def test_csv_source_requires_a_path():
    problems = validate(_cfg(CSV_PATH=None))
    assert any("CSV_PATH is None" in p for p in problems)


def test_unknown_dataset_source_rejected():
    problems = validate(_cfg(DATASET_SOURCE="parquet"))
    assert any("DATASET_SOURCE" in p for p in problems)


# ── check() ──────────────────────────────────────────────────────────────────
def test_check_raises_with_every_problem_listed():
    with pytest.raises(ValueError) as e:
        check(_cfg(TARGET_COLS=[], OPTIMIZATION_DIRECTIONS=["max"], CV_FOLDS=1))
    msg = str(e.value)
    assert "nothing to optimize" in msg and "held-out split" in msg


def test_check_passes_on_the_shipped_config():
    check(config)          # must not raise


# ── the settings that used to be undeclared ──────────────────────────────────
@pytest.mark.parametrize("name", [
    "STEELS_LIBRARY", "STEELS_LIBRARY_SIZE", "BO_ACQ_BATCH_SIZE",
    "CATALYST_FRACTION_LIBRARY_SIZE", "CATALYST_FRACTION_MAX_TOTAL_METAL",
])
def test_previously_undeclared_settings_are_now_declared(name):
    """These were read via getattr with a hard-coded default and existed
    nowhere else, so nothing could catch a typo in either place."""
    assert name in SETTINGS
    assert hasattr(config, name)


@pytest.mark.parametrize("name", [
    "FORMULA_COL", "GP_TRAINING_ITERS", "BO_N_INITIAL", "BO_N_ITERATIONS",
    "CHEM_VIABILITY_FILTER",
])
def test_dead_knobs_are_gone(name):
    """Each was set in config.py and read nowhere. CHEM_VIABILITY_FILTER was
    the harmful one: it announced a SMACT charge-balance screen on generated
    candidates that has never existed."""
    assert not hasattr(config, name), (
        f"{name} is back in config.py — if it now does something, add it to "
        f"config_schema.SETTINGS; if not, it misleads whoever sets it."
    )
