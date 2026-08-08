"""The report states how many samples the surrogate was actually fit on.

It used to print `config.SUBSAMPLE_N` — the subsample-for-fast-iteration knob,
which is None whenever you are using the whole dataset. So every report read
"Samples used: None", including the ones written for real runs. A scientific
report that cannot state its own sample size is worse than one that omits it.

The distinction that makes this more than cosmetic: step 1 loads the CSV, step 4
drops rows whose target is NaN, and those two numbers differ. The PDH literature
CSV loads 85 rows and fits 65. Reporting 85 would overstate the evidence behind
every candidate in the report.
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
import step6_report


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def _write(path, n):
    pd.DataFrame({"true_y": range(n)}).to_parquet(path)


def test_the_count_is_the_rows_the_surrogate_was_fit_on(data_dir):
    """raw = what was loaded, training_predictions = what survived the NaN drop.
    The honest answer is the second one.
    """
    _write(data_dir / "raw.parquet", 85)
    _write(data_dir / "training_predictions.parquet", 65)

    assert step6_report._training_sample_count() == 65


def test_raw_is_the_fallback_when_step_4_left_no_artifact(data_dir):
    _write(data_dir / "raw.parquet", 85)

    assert step6_report._training_sample_count() == 85


def test_no_artifacts_reports_nothing_rather_than_guessing(data_dir):
    assert step6_report._training_sample_count() is None


def test_an_unreadable_artifact_falls_through_instead_of_raising(data_dir):
    """Step 6 is the last step of a long run; a corrupt parquet must not throw
    away a report that is otherwise complete.
    """
    (data_dir / "training_predictions.parquet").write_text("not a parquet")
    _write(data_dir / "raw.parquet", 85)

    assert step6_report._training_sample_count() == 85


def test_the_count_does_not_come_from_the_subsample_knob(data_dir, monkeypatch):
    """The original bug, pinned: SUBSAMPLE_N is None in normal use, and setting
    it must not change what the report claims was used.
    """
    _write(data_dir / "training_predictions.parquet", 65)
    monkeypatch.setattr(config, "SUBSAMPLE_N", None)

    assert step6_report._training_sample_count() == 65
