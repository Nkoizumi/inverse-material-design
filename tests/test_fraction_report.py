"""The atomic-fraction schema gets a real report, and it describes the catalyst
the surrogate actually saw.

Fraction-mode candidates carry no `active_metal` / `promoter_*` / `support`
columns — only a flat atomic-fraction vector plus reaction conditions. Step 6
used to bail out to a minimal table for them. Now it reconstructs the same
support / metal-phase split the featurizer built, so every property number in
the report is the number the GP was fit on. These tests pin that equivalence,
plus the reporting faults that showed up once the narrative existed:

  * a `|` in the candidate label silently splitting markdown table rows;
  * a 0.005 display cutoff hiding the noble metal — Pt sits near 1e-3 atomic
    fraction in a PDH catalyst, so the cutoff dropped the element the
    catalyst is named for;
  * conditions omitted entirely, leaving two candidates that differ only in
    calcination temperature indistinguishable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
import plots
import step6_report
from catalyst_fraction_features import MIN_SUPPORT_FRACTION, featurize_fractions

ELEMENTS = ["Al", "Ga", "Mg", "Mo", "Pt", "Zr"]
SUPPORT_CATIONS = ["Al", "Zr"]
OXIDE_MAP = {"Al": "gamma-Al2O3", "Zr": "ZrO2"}
CONDITIONS = ["reaction_temp_K", "pretreatment"]

# Al-supported, Ga-Mo promoted, with Pt at 8.4e-4 — the trace level a real PDH
# catalyst carries its noble metal at.
ROW = {"Al": 0.95, "Ga": 0.029, "Mg": 0.0, "Mo": 0.009, "Pt": 0.00084, "Zr": 0.0,
       "reaction_temp_K": 873.0, "pretreatment": 1.0}


@pytest.fixture
def fraction_config(monkeypatch):
    monkeypatch.setattr(config, "CATALYST_MODE", False)
    monkeypatch.setattr(config, "CATALYST_FRACTION_MODE", True)
    monkeypatch.setattr(config, "CATALYST_FRACTION_ELEMENTS", ELEMENTS)
    monkeypatch.setattr(config, "CATALYST_FRACTION_SUPPORT_CATIONS", SUPPORT_CATIONS)
    monkeypatch.setattr(config, "CATALYST_FRACTION_SUPPORT_OXIDE_MAP", OXIDE_MAP)
    monkeypatch.setattr(config, "CATALYST_FRACTION_CONDITIONS", CONDITIONS)
    return config


# ── support reconstruction matches the featurizer ────────────────────────────
def test_support_is_the_dominant_cation_mapped_to_its_oxide(fraction_config):
    assert step6_report._fraction_support(ROW) == ("Al", "gamma-Al2O3", 0.95)


def test_no_support_below_the_featurizer_cutoff(fraction_config):
    """Naming an oxide the featurizer refused to assign would attach
    support_phys_* numbers to a candidate scored without them."""
    row = dict(ROW, Al=MIN_SUPPORT_FRACTION - 0.01)
    cation, oxide, _frac = step6_report._fraction_support(row)
    assert (cation, oxide) == (None, None)


def test_metal_weights_exclude_supports_and_renormalize(fraction_config):
    weights = step6_report._fraction_metal_weights(ROW)
    assert set(weights) == {"Ga", "Mo", "Pt"}
    assert weights["Ga"] == pytest.approx(0.029 / 0.03884, rel=1e-9)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_metal_properties_equal_the_features_the_surrogate_saw(fraction_config, tmp_path):
    """The report's metal_properties must BE the metal_phys_* block, not a
    second, independently-derived estimate of it."""
    featurized = featurize_fractions(
        pd.DataFrame([ROW]),
        element_cols=ELEMENTS,
        support_cations=SUPPORT_CATIONS,
        support_oxide_map=OXIDE_MAP,
        condition_cols=CONDITIONS,
        support_lookup_path=config.SUPPORT_LOOKUP_PATH,
        metal_lookup_path=config.METAL_LOOKUP_PATH,
    )
    enriched = step6_report._enrich_fraction_candidate(
        ROW,
        pd.read_csv(config.SUPPORT_LOOKUP_PATH),
        pd.read_csv(config.METAL_LOOKUP_PATH),
    )
    for prop in ("work_function_eV", "pauling_chi", "d_band_center_eV"):
        assert enriched["metal_properties"][prop] == pytest.approx(
            float(featurized[f"metal_phys_{prop}"].iloc[0]), abs=1e-3,
        ), f"report and featurizer disagree on {prop}"


def test_support_properties_equal_the_features_the_surrogate_saw(fraction_config):
    featurized = featurize_fractions(
        pd.DataFrame([ROW]),
        element_cols=ELEMENTS,
        support_cations=SUPPORT_CATIONS,
        support_oxide_map=OXIDE_MAP,
        condition_cols=CONDITIONS,
        support_lookup_path=config.SUPPORT_LOOKUP_PATH,
        metal_lookup_path=config.METAL_LOOKUP_PATH,
    )
    enriched = step6_report._enrich_fraction_candidate(
        ROW,
        pd.read_csv(config.SUPPORT_LOOKUP_PATH),
        pd.read_csv(config.METAL_LOOKUP_PATH),
    )
    for prop in ("work_function_eV", "acidity_NH3_TPD_mmol_g", "lattice_a_A"):
        assert enriched["support_properties"][prop] == pytest.approx(
            float(featurized[f"support_phys_{prop}"].iloc[0]), abs=1e-3,
        )


def test_no_role_labels_are_invented(fraction_config):
    """This schema has no active metal. Asserting one would put a fact in the
    report — and in the LLM prompt — that the dataset does not contain."""
    enriched = step6_report._enrich_fraction_candidate(ROW, None, None)
    assert "active_metal" not in enriched
    assert "promoter_1" not in enriched
    assert enriched["_schema"] == "fraction"


# ── the label ────────────────────────────────────────────────────────────────
def test_label_never_contains_a_pipe(fraction_config):
    """A `|` in a label splits every markdown table row it lands in."""
    assert "|" not in step6_report._fraction_composition_string(ROW)


def test_label_keeps_the_trace_noble_metal(fraction_config):
    label = step6_report._fraction_composition_string(ROW)
    assert "Pt=0.00084" in label, f"trace element dropped from {label!r}"
    assert "gamma-Al2O3" in label and "Al 0.95" in label


def test_label_accepts_an_explicit_panel(fraction_config, monkeypatch):
    """The webui labels candidates with the panel captured at run time, not
    whatever the ambient config holds now."""
    monkeypatch.setattr(config, "CATALYST_FRACTION_ELEMENTS", [])
    label = step6_report._fraction_composition_string(
        ROW, ELEMENTS, SUPPORT_CATIONS, OXIDE_MAP,
    )
    assert "Ga=0.029" in label


def test_label_without_a_support(fraction_config):
    row = dict(ROW, Al=0.0, Ga=0.7, Mo=0.3, Pt=0.0)
    assert step6_report._fraction_composition_string(row) == "Ga=0.7 Mo=0.3"


# ── conditions ───────────────────────────────────────────────────────────────
def test_conditions_section_lists_every_candidate(fraction_config):
    records = [dict(ROW, _label="a"), dict(ROW, _label="b", reaction_temp_K=923.0)]
    md = step6_report._render_conditions_section(records)
    assert "873" in md and "923" in md
    assert len(md.splitlines()) == 2 + len(records)   # header + separator + rows


def test_pretreatment_code_is_decoded(fraction_config):
    md = step6_report._render_conditions_section([dict(ROW, _label="a")])
    assert "oxidation" in md
    md = step6_report._render_conditions_section(
        [dict(ROW, _label="a", pretreatment=-1.0)])
    assert "reduction" in md


def test_conditions_section_is_empty_without_condition_columns(fraction_config, monkeypatch):
    monkeypatch.setattr(config, "CATALYST_FRACTION_CONDITIONS", [])
    assert step6_report._render_conditions_section([dict(ROW, _label="a")]) == ""


# ── composition figure ───────────────────────────────────────────────────────
def test_composition_matrix_drops_the_support_and_keeps_traces():
    built = plots.build_fraction_composition_matrix(
        [dict(ROW, _label="x", support="gamma-Al2O3",
              support_cation="Al", support_fraction=0.95)],
        ELEMENTS, SUPPORT_CATIONS,
    )
    row_labels, col_labels, matrix = built
    assert "Al" not in col_labels and "Zr" not in col_labels
    assert col_labels == ["Ga", "Mo", "Pt"]          # Mg is zero everywhere
    assert matrix[0][col_labels.index("Pt")] == pytest.approx(0.00084)
    assert row_labels == ["#1: gamma-Al2O3 (Al 0.95)"]


def test_composition_matrix_is_none_for_a_pure_support_batch():
    assert plots.build_fraction_composition_matrix(
        [{"Al": 1.0, "_label": "x"}], ELEMENTS, SUPPORT_CATIONS,
    ) is None


def test_png_and_plotly_show_the_same_elements(tmp_path):
    """Two renderers, one column-selection rule — otherwise the saved report
    and the webui widget disagree about what the batch contains."""
    from interactive_plots import make_fraction_composition_figure

    cands = [dict(ROW, _label="x"), dict(ROW, _label="y", Ga=0.0, Mg=0.04)]
    png = plots.make_fraction_composition_heatmap(
        cands, ELEMENTS, SUPPORT_CATIONS, tmp_path / "c.png")
    fig = make_fraction_composition_figure(cands, ELEMENTS, SUPPORT_CATIONS)
    _rows, cols, _m = plots.build_fraction_composition_matrix(
        cands, ELEMENTS, SUPPORT_CATIONS)
    assert png.exists()
    assert list(fig.data[0].x) == cols


# ── end to end ───────────────────────────────────────────────────────────────
@pytest.fixture
def fraction_bundle(fraction_config, tmp_path, monkeypatch):
    """A full fraction-mode report with the LLM stubbed out."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(config, "TARGET_COLS", ["propylene_yield"])
    monkeypatch.setattr(config, "OPTIMIZATION_DIRECTIONS", ["max"])
    monkeypatch.setattr(config, "REPORT_TOP_K", 2)
    monkeypatch.setattr(config, "DATASET_SOURCE", "csv")

    monkeypatch.setattr(
        step6_report, "_ask_ollama",
        lambda *a, **k: ("narrative ![c](x.png)", {}),
    )

    candidates = pd.DataFrame([
        dict(ROW, pred_propylene_yield=0.42, pred_propylene_yield_sd=0.05),
        dict(ROW, Ga=0.0, Mg=0.04, pred_propylene_yield=0.31,
             pred_propylene_yield_sd=0.06),
    ])
    return step6_report.generate_report_bundle(candidates)


def test_bundle_has_a_narrative_and_a_composition_figure(fraction_bundle):
    assert fraction_bundle.narrative_body.startswith("narrative")
    assert "composition" in {f.figure_id for f in fraction_bundle.figures}


def test_report_body_carries_the_schema_caveat_and_conditions(fraction_bundle):
    body = fraction_bundle.path.read_text()
    assert "Schema: atomic fractions" in body
    assert "Process conditions selected for each candidate" in body
    assert "Metal phase (" in body        # not "Active metal ("
    assert "Active metal (" not in body


def test_every_table_row_has_its_header_column_count(fraction_bundle):
    """The pipe-in-label bug was invisible in the source and obvious only in
    the rendered table, so count the cells of every table in the report."""
    blocks, current = [], []
    for line in fraction_bundle.path.read_text().splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    assert blocks, "no markdown tables rendered"
    for block in blocks:
        n_cols = block[0].count("|")
        for row in block[1:]:
            assert row.count("|") == n_cols, (
                f"row has a stray pipe:\n  header: {block[0]}\n  row:    {row}")


def test_role_mode_is_untouched(monkeypatch, tmp_path):
    """The fraction branch must not have changed what a role-based run gets."""
    monkeypatch.setattr(config, "CATALYST_MODE", True)
    monkeypatch.setattr(config, "CATALYST_FRACTION_MODE", False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(config, "TARGET_COLS", ["y"])
    monkeypatch.setattr(config, "OPTIMIZATION_DIRECTIONS", ["max"])
    monkeypatch.setattr(config, "REPORT_TOP_K", 1)
    monkeypatch.setattr(config, "DATASET_SOURCE", "csv")
    monkeypatch.setattr(step6_report, "_ask_ollama", lambda *a, **k: ("n", {}))

    candidates = pd.DataFrame([{
        "active_metal": "Pt", "promoter_1": "Sn", "promoter_2": "",
        "support": "gamma-Al2O3", "metal_loading_wt": 1.0,
        "promoter_1_loading_wt": 0.5, "promoter_2_loading_wt": np.nan,
        "pred_y": 1.0, "pred_y_sd": 0.1,
    }])
    body = step6_report.generate_report_bundle(candidates).path.read_text()
    assert "Active metal (Pt)" in body
    assert "Schema: atomic fractions" not in body
    assert "Process conditions selected" not in body


def test_missing_lookups_still_yield_a_supported_family(fraction_config, tmp_path,
                                                        monkeypatch):
    """A missing property CSV costs the property NUMBERS, not the chemistry:
    the support/metal split comes from the composition alone."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(config, "TARGET_COLS", ["y"])
    monkeypatch.setattr(config, "OPTIMIZATION_DIRECTIONS", ["max"])
    monkeypatch.setattr(config, "REPORT_TOP_K", 1)
    monkeypatch.setattr(config, "DATASET_SOURCE", "csv")
    monkeypatch.setattr(config, "SUPPORT_LOOKUP_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(step6_report, "_ask_ollama", lambda *a, **k: ("n", {}))

    bundle = step6_report.generate_report_bundle(
        pd.DataFrame([dict(ROW, pred_y=1.0, pred_y_sd=0.1)]))
    record = bundle.candidates_enriched[0]
    assert record["support"] == "gamma-Al2O3"
    assert "unsupported" not in record["_family"]
    assert "support_properties" not in record       # numbers really are gone
