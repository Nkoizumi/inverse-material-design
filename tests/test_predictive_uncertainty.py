"""Every surrogate must report the PREDICTIVE standard deviation.

The report prints GP and BNN sigma side by side and buckets their average into
a high/medium/low confidence column. That only means something if the two
numbers measure the same thing. They did not: the GP returned its latent
posterior sd (epistemic only) while the BNN sampled its `obs` site and so
included observation noise — about a 1.8x definitional gap on the bundled
synthetic set, presented as if comparable.

Predictive is the right side of that choice here: the sigma answers "how much
should I trust this before spending lab time on it", which is a question about
a measurement not yet taken. Understating it is the worse error when handing
candidates to an experimentalist.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import config
from step4_surrogate import GPSurrogate, XYData


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def fitted() -> tuple[GPSurrogate, XYData]:
    torch.manual_seed(0)
    n, d = 40, 3
    X = torch.rand(n, d, dtype=torch.double, device=DEV)
    Y = (X.sum(dim=1, keepdim=True)
         + 0.3 * torch.randn(n, 1, dtype=torch.double, device=DEV))
    data = XYData(
        X=X, Y=Y, feature_cols=[f"f{i}" for i in range(d)], target_cols=["y"],
        x_mean=torch.zeros(d, dtype=torch.double, device=DEV),
        x_std=torch.ones(d, dtype=torch.double, device=DEV),
        y_mean=torch.zeros(1, dtype=torch.double, device=DEV),
        y_std=torch.ones(1, dtype=torch.double, device=DEV),
    )
    gp = GPSurrogate()
    gp.fit(data)
    return gp, data


def test_gp_reports_predictive_not_latent_sd(fitted):
    """The regression this module exists for. Predictive sd = latent sd plus
    the likelihood's noise, so it must be strictly larger."""
    gp, data = fitted
    gp.model.eval()
    with torch.no_grad():
        latent = gp.model.posterior(data.X).variance.clamp_min(1e-12).sqrt()
    _, reported = gp.predict(data.X)
    assert (reported > latent).all(), (
        "predict() returned the latent posterior sd; it must include "
        "observation noise so it is comparable with the BNN's"
    )


def test_predictive_sd_matches_botorch_observation_noise(fitted):
    gp, data = fitted
    gp.model.eval()
    with torch.no_grad():
        expected = (gp.model.posterior(data.X, observation_noise=True)
                    .variance.clamp_min(1e-12).sqrt())
    _, reported = gp.predict(data.X)
    torch.testing.assert_close(reported, expected)


def test_predictive_sd_is_finite_and_positive(fitted):
    gp, data = fitted
    _, sd = gp.predict(data.X)
    assert torch.isfinite(sd).all()
    assert (sd > 0).all()


def test_mean_is_unchanged_by_the_noise_term(fitted):
    """Observation noise widens the interval; it must not move the centre."""
    gp, data = fitted
    gp.model.eval()
    with torch.no_grad():
        latent_mean = gp.model.posterior(data.X).mean
    mean, _ = gp.predict(data.X)
    torch.testing.assert_close(mean, latent_mean)


def test_shapes_survive_the_change(fitted):
    gp, data = fitted
    mean, sd = gp.predict(data.X)
    assert mean.shape == sd.shape == (data.X.shape[0], 1)


# ── the consumer that made this matter ───────────────────────────────────────
def test_confidence_buckets_use_the_reported_sd():
    """_build_confidence normalizes sigma by the training-Y std. With a
    predictive sigma the comparison is meaningful — 'tighter than the data's
    own scatter' — which is the bar the buckets are supposed to express."""
    from step6_report import _build_confidence

    y_stds = {"a": 1.0}
    assert _build_confidence({"pred_a_sd": 0.2}, ["a"], y_stds) == "high"
    assert _build_confidence({"pred_a_sd": 0.7}, ["a"], y_stds) == "medium"
    assert _build_confidence({"pred_a_sd": 1.5}, ["a"], y_stds) == "low"


def test_confidence_is_dash_when_sigma_missing():
    from step6_report import _build_confidence
    assert _build_confidence({}, ["a"], {"a": 1.0}) == "—"
    assert _build_confidence({"pred_a_sd": 0.2}, ["a"], {}) == "—"


def test_widening_sigma_can_only_lower_confidence():
    """Sanity check on the direction: switching from latent to predictive sd
    must never make a candidate look MORE confident."""
    from step6_report import _build_confidence

    order = {"high": 2, "medium": 1, "low": 0, "—": -1}
    y_stds = {"a": 1.0}
    latent = _build_confidence({"pred_a_sd": 0.3}, ["a"], y_stds)
    predictive = _build_confidence({"pred_a_sd": 0.3 * 1.8}, ["a"], y_stds)
    assert order[predictive] <= order[latent]
