"""Feature importances reach the step-6 LLM as claims about which property
families drive the chemistry, so they have to be measured out of sample.

They used to be XGBoost gain from a single fit on the FULL dataset. Gain
rewards a feature for every split it was used in, including splits that only
fit noise — and with held-out CV R² ~0.33-0.54 on n=210 there is plenty of
noise to fit. Nothing in the report said the ranking was in-sample.

step3 now scores by K-fold permutation importance on the held-out fold and
reports how many folds independently ranked each feature top-K.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from step3_eda import _feature_importances_per_target
from step6_report import _extract_feature_rankings

import json


@pytest.fixture(scope="module")
def signal_and_noise() -> tuple[pd.DataFrame, list[str]]:
    """Two genuinely predictive columns, forty pure-noise columns."""
    rng = np.random.default_rng(0)
    n = 200
    sig1, sig2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"sig1": sig1, "sig2": sig2})
    for i in range(40):
        df[f"noise{i}"] = rng.normal(size=n)
    df["y"] = 3 * sig1 - 2 * sig2 + rng.normal(scale=0.5, size=n)
    return df, [c for c in df.columns if c != "y"]


@pytest.fixture(scope="module")
def ranked(signal_and_noise):
    df, feats = signal_and_noise
    return _feature_importances_per_target(df, feats, ["y"], top_k=5)["y"]


def test_real_signals_rank_above_noise(ranked):
    top_two = {e["feature"] for e in ranked[:2]}
    assert top_two == {"sig1", "sig2"}


def test_real_signals_are_stable_across_folds(ranked):
    for e in ranked:
        if e["feature"] in ("sig1", "sig2"):
            assert e["folds_in_top_k"] == e["n_folds"], (
                f"{e['feature']} should top every fold; got "
                f"{e['folds_in_top_k']}/{e['n_folds']}"
            )


def test_noise_is_marked_unstable(ranked):
    """The number that matters when reading these: a feature topping one fold
    out of five is noise, and must be distinguishable from one topping all
    five."""
    noise = [e for e in ranked if e["feature"].startswith("noise")]
    assert noise, "fixture should leave some noise in the top-5"
    for e in noise:
        assert e["folds_in_top_k"] < e["n_folds"]
        assert e["importance"] < 0.05


def test_entries_carry_the_full_schema(ranked):
    for e in ranked:
        assert set(e) >= {"feature", "importance", "importance_std",
                          "folds_in_top_k", "n_folds", "method"}
        assert e["method"] == "permutation_cv"
        assert e["n_folds"] >= 2


def test_ranked_descending_by_importance(ranked):
    imps = [e["importance"] for e in ranked]
    assert imps == sorted(imps, reverse=True)


def test_falls_back_to_in_sample_gain_when_too_few_rows():
    """Below ~2 rows per held-out fold a permutation R² is meaningless. Fall
    back, but SAY it fell back."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=8), "b": rng.normal(size=8)})
    df["y"] = df["a"] * 2
    out = _feature_importances_per_target(df, ["a", "b"], ["y"], top_k=2)["y"]
    assert out
    assert all(e["method"] == "gain_in_sample" for e in out)


def test_missing_target_is_skipped(signal_and_noise):
    df, feats = signal_and_noise
    assert _feature_importances_per_target(df, feats, ["not_a_column"]) == {}


def test_importances_are_reproducible(signal_and_noise):
    """Same seed, same ranking — these feed a report someone may re-run."""
    df, feats = signal_and_noise
    a = _feature_importances_per_target(df, feats, ["y"], top_k=5)["y"]
    b = _feature_importances_per_target(df, feats, ["y"], top_k=5)["y"]
    assert [e["feature"] for e in a] == [e["feature"] for e in b]
    np.testing.assert_allclose([e["importance"] for e in a],
                               [e["importance"] for e in b])


# ── what step 6 hands the LLM ────────────────────────────────────────────────
def test_extract_rankings_carries_stability_to_the_prompt(ranked):
    summary = json.dumps({"top_features_per_target": {"y": ranked}})
    out = _extract_feature_rankings(summary)
    assert out["y"]["method"] == "permutation_cv"
    first = out["y"]["features"][0]
    assert first["name"] in ("sig1", "sig2")
    assert "folds" in first["stability"], (
        "the LLM must see how stable a ranking is, not just its name"
    )


def test_extract_rankings_reports_an_empty_ranking_explicitly():
    """No feature with positive held-out importance is a real outcome on
    data-limited targets. The prompt must be told, so the narrative says the
    data does not support a claim instead of inventing one."""
    out = _extract_feature_rankings(json.dumps({"top_features_per_target": {"y": []}}))
    assert out["y"]["features"] == []
    assert "no feature" in out["y"]["note"]


def test_extract_rankings_tolerates_legacy_summaries():
    """eda_summary.json files written before this change have no n_folds."""
    legacy = json.dumps({"top_features_per_target":
                         {"y": [{"feature": "a", "importance": 0.5}]}})
    out = _extract_feature_rankings(legacy)
    assert out["y"]["features"] == [{"name": "a"}]


def test_extract_rankings_handles_missing_or_bad_input():
    assert _extract_feature_rankings(None) == {}
    assert _extract_feature_rankings("not json") == {}
    assert _extract_feature_rankings(json.dumps({})) == {}
