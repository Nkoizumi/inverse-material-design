"""Guards against a report claiming held-out CV numbers that the run never
computed.

step6 chooses the parity figure by checking whether
`data/cv_parity_predictions.parquet` exists, and labels it "held-out k-fold
CV". Nothing used to delete that file, so:

  * run once with CV_FOLDS=5, then again with CV_FOLDS=0 → the second run's
    report presents the FIRST run's CV numbers as its own;
  * a crash between step 4 and step 6 leaves the previous run's file in place;
  * switching targets leaves a file whose columns describe other targets.
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
import step4_surrogate
import step6_report


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(step4_surrogate.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(step6_report.config, "DATA_DIR", tmp_path)
    return tmp_path


def _write_parity(path, targets, with_fold=True):
    cols = {}
    for t in targets:
        cols[f"true_{t}"] = [1.0, 2.0, 3.0]
        cols[f"gp_pred_{t}"] = [1.1, 1.9, 3.2]
        cols[f"gp_sd_{t}"] = [0.1, 0.1, 0.1]
    if with_fold:
        cols["fold"] = [1, 2, 3]
    pd.DataFrame(cols).to_parquet(path)


def test_invalidate_removes_both_artifacts(data_dir):
    cv = data_dir / "cv_parity_predictions.parquet"
    ins = data_dir / "training_predictions.parquet"
    _write_parity(cv, ["y"])
    _write_parity(ins, ["y"], with_fold=False)

    step4_surrogate._invalidate_parity_artifacts()

    assert not cv.exists(), "stale CV parity survived a refit"
    assert not ins.exists(), "stale in-sample parity survived a refit"


def test_invalidate_is_a_noop_when_nothing_to_remove(data_dir):
    step4_surrogate._invalidate_parity_artifacts()  # must not raise


def test_cv_artifact_is_preferred_when_it_covers_targets(data_dir, monkeypatch):
    monkeypatch.setattr(config, "TARGET_COLS", ["y1", "y2"])
    _write_parity(data_dir / "cv_parity_predictions.parquet", ["y1", "y2"])
    _write_parity(data_dir / "training_predictions.parquet", ["y1", "y2"],
                  with_fold=False)

    path, is_cv = step6_report._select_parity_artifact()
    assert is_cv is True
    assert path.name == "cv_parity_predictions.parquet"


def test_stale_cv_artifact_for_other_targets_is_rejected(data_dir, monkeypatch):
    """The case that would silently mislabel: a leftover CV file describing a
    different target set. Fall through to in-sample rather than present its
    numbers as this run's held-out CV."""
    monkeypatch.setattr(config, "TARGET_COLS", ["propylene_yield"])
    _write_parity(data_dir / "cv_parity_predictions.parquet", ["propane_TOF_log"])
    _write_parity(data_dir / "training_predictions.parquet", ["propylene_yield"],
                  with_fold=False)

    path, is_cv = step6_report._select_parity_artifact()
    assert is_cv is False, "a CV file for other targets was accepted"
    assert path.name == "training_predictions.parquet"


def test_no_artifact_yields_no_parity_figure(data_dir, monkeypatch):
    monkeypatch.setattr(config, "TARGET_COLS", ["y"])
    path, is_cv = step6_report._select_parity_artifact()
    assert path is None and is_cv is False


def test_caption_falls_back_when_no_artifact(data_dir, monkeypatch):
    """_deterministic_parity_caption must not invent stats out of nothing."""
    monkeypatch.setattr(config, "TARGET_COLS", ["y"])
    assert step6_report._deterministic_parity_caption("llm text") == "llm text"


def test_cv_disabled_leaves_no_cv_artifact(data_dir, monkeypatch):
    """End-to-end on the two functions that matter: a stale CV file present,
    CV then disabled — the CV artifact must not survive into the report."""
    _write_parity(data_dir / "cv_parity_predictions.parquet", ["y"])
    monkeypatch.setattr(config, "CV_FOLDS", 0)
    monkeypatch.setattr(config, "TARGET_COLS", ["y"])

    step4_surrogate._invalidate_parity_artifacts()
    step4_surrogate._save_cv_parity_predictions(pd.DataFrame({"y": [1.0, 2.0]}))

    assert not (data_dir / "cv_parity_predictions.parquet").exists()
    path, is_cv = step6_report._select_parity_artifact()
    assert is_cv is False
