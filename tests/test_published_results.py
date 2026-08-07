"""Verify the CV numbers the README publishes.

This was the last claim in the repository checked by hand rather than by test.
It got re-verified after every change this session — the scipy pin bump, the
CPU/CUDA switch, the predictive-sigma change — precisely because nothing else
would have caught a regression in it.

Two design choices worth stating:

  * The expected values are PARSED FROM THE README, not hard-coded here.
    Hard-coding them would create a second source of truth for the same
    numbers, and the README could then drift from the test silently — which is
    the failure mode this file exists to prevent. Reading the doc means a code
    change that moves the numbers fails, AND a README edit that is not backed
    by a re-measurement fails.

  * It skips when `data/pdh_ACSMaterialsLetters.csv` is absent. That dataset is
    not redistributable (ACS Materials Letters 6(11) 5138-5145), so it is
    gitignored and CI never has it. The test therefore protects the author's
    working copy, where the file does exist, rather than CI.

Costs ~3.7 s — five GP fits over 210 rows and 30 features. Cheap enough to
leave in the default run.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config

ROOT = Path(__file__).resolve().parent.parent
ACS_CSV = ROOT / "data" / "pdh_ACSMaterialsLetters.csv"

# How far the reproduced value may sit from the published one. These have
# reproduced to three decimals across a scipy bump, a CPU/CUDA switch and a
# BLAS change, so the tolerance is not absorbing known drift — it is only there
# so a last-digit difference on someone else's BLAS is not a failure. Tight
# enough that a re-introduced target leak (which inflates R²) or a broken
# featurizer (which collapses it) cannot hide.
R2_TOL = 0.01
MAE_TOL = 0.005


def _parse_readme_results() -> dict[str, tuple[float, float]]:
    """Pull {target: (R², MAE)} out of the README's results table."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\|\s*([A-Za-z0-9_]+)\s*\|\s*([+-]?\d+\.\d+)\s*\|\s*([+-]?\d+\.\d+)\s*\|"
        r"\s*(?:max|min)\s*\|",
        text, flags=re.MULTILINE,
    )
    return {t: (float(r2), float(mae)) for t, r2, mae in rows}


def _r2_mae(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, int]:
    mask = ~(np.isnan(true) | np.isnan(pred))
    yt, yp = true[mask], pred[mask]
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    ss_res = float(np.sum((yt - yp) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, float(np.mean(np.abs(yt - yp))), int(mask.sum())


# ── the README table itself ──────────────────────────────────────────────────
def test_readme_publishes_a_parseable_results_table():
    """If the table is reformatted so the parser misses it, the reproduction
    test below would silently assert nothing."""
    published = _parse_readme_results()
    assert published, "no results table found in README.md"
    assert set(published) == {"propylene_yield", "deactivation_rate_log"}


# ── the reproduction ─────────────────────────────────────────────────────────
@pytest.mark.skipif(not ACS_CSV.exists(),
                    reason="ACS PDH dataset is not redistributable; see "
                           "scripts/download_acs_pdh.py")
def test_readme_cv_numbers_reproduce(tmp_path, monkeypatch):
    from run_pipeline import PRESETS

    published = _parse_readme_results()

    for k, v in PRESETS["acs_pdh"].items():
        monkeypatch.setattr(config, k, v, raising=False)
    monkeypatch.setattr(config, "SURROGATE_KIND", "gp")
    monkeypatch.setattr(config, "CV_FOLDS", 5)
    monkeypatch.setattr(config, "MAX_GP_FEATURES", 30)
    monkeypatch.setattr(config, "CSV_PATH", ACS_CSV)
    # Write the run's artifacts to a temp dir; a test must not clobber the
    # parquets in the working copy's data/.
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    from step1_load import load_dataset
    from step2_featurize import featurize
    from step4_surrogate import fit_surrogates

    df = featurize(load_dataset())
    assert len(df) == 210, f"expected the published n=210, got {len(df)}"
    fit_surrogates(df)

    parity = pd.read_parquet(tmp_path / "cv_parity_predictions.parquet")

    failures = []
    for target, (want_r2, want_mae) in published.items():
        got_r2, got_mae, n = _r2_mae(parity[f"true_{target}"].values,
                                     parity[f"gp_pred_{target}"].values)
        if abs(got_r2 - want_r2) > R2_TOL:
            failures.append(
                f"{target}: README says R²={want_r2:+.3f}, reproduced "
                f"{got_r2:+.3f} (n={n})"
            )
        if abs(got_mae - want_mae) > MAE_TOL:
            failures.append(
                f"{target}: README says MAE={want_mae:.3f}, reproduced "
                f"{got_mae:.3f} (n={n})"
            )

    assert not failures, (
        "The README's published CV numbers no longer reproduce:\n"
        + "\n".join(f"  • {f}" for f in failures)
        + "\n\nEither the pipeline changed (investigate before shipping — an "
          "R² that went UP may mean a target leak was re-introduced) or the "
          "README needs re-measuring against the current code."
    )
