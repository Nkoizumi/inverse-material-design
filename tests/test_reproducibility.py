"""Guards against the pipeline silently becoming non-reproducible again.

Background: config.RANDOM_STATE was only ever threaded into scikit-learn's
KFold, DataFrame.sample and the library builders' numpy RNGs. BoTorch
acquisition functions build their own unseeded SobolQMCNormalSampler when
`sampler=` is omitted, so two identical runs scored the discrete library
differently and selected different catalysts. Measured on the acs_pdh preset
before the fix: only 9 of 20 selected catalysts were shared between two
back-to-back runs.

These tests are deliberately small and CPU-only (a few seconds) so they can
run in CI. The end-to-end check lives in the docstring above rather than in
code because a full preset run takes minutes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from seeding import make_sampler, seed_everything

pytestmark = pytest.mark.filterwarnings("ignore")


def test_seed_everything_pins_torch_and_numpy():
    seed_everything(123, quiet=True)
    a_torch, a_np = torch.rand(5), np.random.rand(5)
    seed_everything(123, quiet=True)
    b_torch, b_np = torch.rand(5), np.random.rand(5)
    assert torch.equal(a_torch, b_torch)
    np.testing.assert_array_equal(a_np, b_np)


def test_seed_everything_defaults_to_config_random_state():
    import config
    assert seed_everything(quiet=True) == config.RANDOM_STATE


def test_make_sampler_is_seeded_and_stable():
    """Two samplers built independently must draw identical base samples."""
    from botorch.posteriors import GPyTorchPosterior
    from gpytorch.distributions import MultivariateNormal

    mean = torch.zeros(4, dtype=torch.double)
    cov = torch.eye(4, dtype=torch.double)
    posterior = GPyTorchPosterior(MultivariateNormal(mean, cov))

    s1, s2 = make_sampler(multi_objective=False), make_sampler(multi_objective=False)
    assert torch.equal(s1(posterior), s2(posterior))


def test_make_sampler_matches_botorch_default_sample_counts():
    """We pass an explicit sampler purely for determinism — the MC sample
    count must stay at BoTorch's own default so the acquisition statistics are
    unchanged. If a BoTorch upgrade changes these, this test says so."""
    from botorch.acquisition.acquisition import MCSamplerMixin
    from botorch.acquisition.multi_objective.base import (
        MultiObjectiveMCAcquisitionFunction,
    )
    assert make_sampler(multi_objective=False).sample_shape == \
        MCSamplerMixin._default_sample_shape
    assert make_sampler(multi_objective=True).sample_shape == \
        MultiObjectiveMCAcquisitionFunction._default_sample_shape


def _discrete_bo_once(seed: int = 7) -> list[int]:
    """Fit a tiny GP and rank a fixed discrete choice set, exactly the way
    step5's discrete paths do. Returns the selected choice indices."""
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    seed_everything(seed, quiet=True)
    X = torch.rand(25, 3, dtype=torch.double)
    Y = (X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(25, 1, dtype=torch.double))
    gp = SingleTaskGP(X, Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

    choices = torch.rand(60, 3, dtype=torch.double)
    acq = qLogNoisyExpectedImprovement(
        model=gp, X_baseline=X, sampler=make_sampler(multi_objective=False, seed=seed),
    )
    with torch.no_grad():
        vals = acq(choices.unsqueeze(1))
    return vals.argsort(descending=True)[:10].tolist()


def test_discrete_bo_ranking_is_reproducible():
    """The regression this whole module exists for: an unseeded acquisition
    function reshuffles the top of the candidate ranking between runs."""
    assert _discrete_bo_once() == _discrete_bo_once()


def test_unseeded_acquisition_would_drift():
    """Sanity check on the test itself — confirms the ranking IS sensitive to
    the sampler, so test_discrete_bo_ranking_is_reproducible is not passing
    vacuously. If BoTorch ever makes its default sampler deterministic this
    will fail and the guard above can be relaxed."""
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    seed_everything(7, quiet=True)
    X = torch.rand(25, 3, dtype=torch.double)
    Y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(25, 1, dtype=torch.double)
    gp = SingleTaskGP(X, Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    choices = torch.rand(60, 3, dtype=torch.double)

    rankings = []
    for _ in range(2):
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=X)  # no sampler
        with torch.no_grad():
            rankings.append(acq(choices.unsqueeze(1)).argsort(descending=True)[:10].tolist())
    assert rankings[0] != rankings[1], (
        "BoTorch's default sampler appears deterministic now — re-check "
        "whether the explicit seeded sampler is still required."
    )
