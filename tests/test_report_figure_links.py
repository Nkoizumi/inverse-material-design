"""Every image link in a saved report points at a figure that exists.

The prompt hands the model the exact PNG filenames and tells it to use them
verbatim. phi-4 paraphrases anyway. On the acs_pdh run of 2026-08-07 it wrote
`parity_20260807_202446.png` for a figure saved as
`parity_cv_20260807_202446.png` — it dropped the `_cv`, which is precisely the
part distinguishing held-out CV parity from in-sample parity. The report on disk
had a dead image.

`_repair_figure_links` rewrites near-misses and DROPS anything it cannot resolve
uniquely. The asymmetry is deliberate: a missing figure is a gap, a caption
pointing at the wrong figure is a false statement about the data.
"""
from __future__ import annotations

import pytest

import step6_report
from step6_report import FigureEntry, _repair_figure_links

TS = "20260807_202446"


def _figs(*ids: str) -> list[FigureEntry]:
    files = {
        "parity": f"parity_cv_{TS}.png",
        "pareto": f"pareto_{TS}.png",
        "feature_importance": f"feature_importance_{TS}.png",
        "gp_vs_bnn": f"gp_vs_bnn_{TS}.png",
        "candidate_heatmap": f"candidate_heatmap_{TS}.png",
        "composition": f"composition_{TS}.png",
    }
    return [FigureEntry(figure_id=i, title=i, alt=i, default_description="",
                        png_filename=files[i]) for i in ids]


def test_exact_filenames_are_left_alone():
    md = f"text ![Pareto]({'pareto_' + TS}.png) more"
    assert _repair_figure_links(md, _figs("pareto")) == md


def test_the_observed_parity_miss_is_repaired():
    """The actual failure: `_cv` dropped from the filename."""
    md = f"![Parity plot](parity_{TS}.png)"
    out = _repair_figure_links(md, _figs("parity", "pareto"))
    assert out == f"![Parity plot](parity_cv_{TS}.png)"


def test_alt_text_survives_the_repair():
    md = f"![Predicted vs actual](parity_{TS}.png)"
    assert "![Predicted vs actual](" in _repair_figure_links(md, _figs("parity"))


def test_a_figure_id_with_no_timestamp_resolves():
    """Models also cite the bare figure id."""
    md = "![c](composition.png)"
    assert _repair_figure_links(md, _figs("composition")) == \
        f"![c](composition_{TS}.png)"


def test_a_path_prefix_is_tolerated():
    md = f"![p](reports/pareto_{TS}.png)"
    assert _repair_figure_links(md, _figs("pareto")) == f"![p](pareto_{TS}.png)"


def test_an_invented_figure_is_dropped_not_guessed():
    md = f"before ![x](tsne_projection_{TS}.png) after"
    assert _repair_figure_links(md, _figs("pareto", "parity")) == "before  after"


def test_an_exact_figure_id_wins_over_a_prefix_search():
    """`parity` is the id of the parity figure, so it names it unambiguously
    even though the file is `parity_cv_…`."""
    assert _repair_figure_links("![p](parity.png)", _figs("parity", "pareto")) \
        == f"![p](parity_cv_{TS}.png)"


def test_an_ambiguous_prefix_is_dropped_not_guessed():
    """When `parity` prefixes two figures and is the id of neither, refuse
    rather than pick — citing the wrong one would present in-sample numbers as
    held-out."""
    figs = [
        FigureEntry(figure_id="parity_cv", title="t", alt="a",
                    default_description="", png_filename=f"parity_cv_{TS}.png"),
        FigureEntry(figure_id="parity_insample", title="t", alt="a",
                    default_description="",
                    png_filename=f"parity_insample_{TS}.png"),
    ]
    assert "png" not in _repair_figure_links(f"![p](parity_{TS}.png)", figs)


def test_figures_without_a_png_are_not_link_targets():
    """A webui-only Plotly figure has no file; nothing may resolve to it."""
    figs = [FigureEntry(figure_id="pareto", title="t", alt="a",
                        default_description="", png_filename=None)]
    assert _repair_figure_links(f"![p](pareto_{TS}.png)", figs) == ""


def test_empty_narrative_is_untouched():
    assert _repair_figure_links("", _figs("pareto")) == ""
    assert _repair_figure_links("no images here", _figs("pareto")) == \
        "no images here"


# ── end to end: the saved report has no dead links ───────────────────────────
def test_saved_report_has_no_dead_image_links(tmp_path, monkeypatch):
    import config
    import pandas as pd

    monkeypatch.setattr(config, "CATALYST_MODE", True)
    monkeypatch.setattr(config, "CATALYST_FRACTION_MODE", False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(config, "TARGET_COLS", ["y"])
    monkeypatch.setattr(config, "OPTIMIZATION_DIRECTIONS", ["max"])
    monkeypatch.setattr(config, "REPORT_TOP_K", 1)
    monkeypatch.setattr(config, "DATASET_SOURCE", "csv")

    # A narrative citing one good link, one near-miss, one invention.
    def _fake_ollama(records, eda, targets, figures, agreement=None, **kw):
        names = [f.png_filename for f in figures if f.png_filename]
        near_miss = _mangle(names[0]) if names else "nope.png"
        return (f"a ![one]({near_miss}) b ![two](invented_thing.png)", {})

    def _mangle(name: str) -> str:
        return name.replace("parity_cv_", "parity_").replace(
            "candidate_heatmap_", "candidate_heatmap_")

    monkeypatch.setattr(step6_report, "_ask_ollama", _fake_ollama)

    bundle = step6_report.generate_report_bundle(pd.DataFrame([{
        "active_metal": "Pt", "promoter_1": "", "promoter_2": "",
        "support": "gamma-Al2O3", "pred_y": 1.0, "pred_y_sd": 0.1,
    }]))

    import re
    body = bundle.path.read_text()
    cited = set(re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", body))
    missing = [c for c in cited if not (tmp_path / c).exists()]
    assert not missing, f"report links to files that do not exist: {missing}"
    assert "invented_thing.png" not in body
