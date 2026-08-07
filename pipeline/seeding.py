"""Deterministic seeding for the pipeline.

Without this, the pipeline is NOT reproducible run-to-run even though
``config.RANDOM_STATE`` is set. ``RANDOM_STATE`` was only ever threaded into
scikit-learn's KFold, ``DataFrame.sample`` and the numpy RNGs in the library
builders. Three unseeded sources remained:

  1. BoTorch acquisition functions build their own ``SobolQMCNormalSampler``
     at construction time when no ``sampler=`` is passed. The QMC base samples
     are drawn from torch's global RNG, so two identical runs score the
     library differently and select different catalysts.
  2. ``fit_gpytorch_mll`` / Adam initialise from the global torch RNG.
  3. The Pyro BNN's variational guide initialises from the global torch RNG.

Measured before this module existed: two back-to-back runs of
``--preset acs_pdh`` (gp only, 3k-row library) shared only 9 of 20 selected
catalysts. After it: 20 of 20. See ``tests/test_reproducibility.py``.

Determinism also holds ACROSS DEVICES, which is not automatic — CUDA and CPU
kernels reduce in different orders, and the acquisition ranking could in
principle diverge on the resulting float differences. Checked on the same
``acs_pdh`` configuration:

    CPU  (torch 2.12.1+cpu)    ┐
                              ├─ identical batch, 20 of 20 catalysts
    CUDA (torch 2.12.1+cu130) ┘

That matters because development here happens on a GPU workstation while CI
runs CPU-only on a hosted runner (see ``.github/workflows/tests.yml``, which
installs the ``+cpu`` wheel deliberately): a candidate list generated locally
is the one CI would generate. Re-check this if the acquisition function, the
sampler, or the BoTorch pin changes — it is an empirical result about the
current stack, not a guarantee BoTorch makes. There is no automated test for
it, since it needs two hardware configurations.

``seed_everything`` is called at the top of ``step4_surrogate.fit_surrogates``
and ``step5_inverse.run_inverse`` so both the CLI driver and the web UI get
determinism without having to remember to ask for it.
"""
from __future__ import annotations

import logging
import os
import random

log = logging.getLogger(__name__)


def resolve_seed(default: int = 42) -> int:
    """Read the seed from config, falling back to `default`.

    Imported lazily so this module stays usable from scripts that never touch
    the config module.
    """
    try:
        import config
    except ImportError:
        return default
    seed = getattr(config, "RANDOM_STATE", default)
    return default if seed is None else int(seed)


def seed_everything(seed: int | None = None, *, quiet: bool = False) -> int:
    """Seed python, numpy and torch (CPU + CUDA) global RNGs. Returns the seed.

    Call this at the start of any stage whose output must be reproducible.
    Deliberately does NOT set ``torch.use_deterministic_algorithms`` — that
    raises on several cuSOLVER paths GPyTorch relies on, and the GP fits here
    are CPU-bound in practice.
    """
    if seed is None:
        seed = resolve_seed()
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    if not quiet:
        log.info("Seeded python/numpy/torch RNGs with %d.", seed)
    return seed


def make_sampler(multi_objective: bool, seed: int | None = None):
    """Build the QMC sampler for a BoTorch acquisition function.

    BoTorch constructs an *unseeded* ``SobolQMCNormalSampler`` when ``sampler``
    is omitted, which is the single largest source of run-to-run drift in the
    selected BO batch. Passing an explicitly seeded sampler pins it.

    ``sample_shape`` mirrors BoTorch 0.18's own defaults —
    ``AcquisitionFunction._default_sample_shape`` (512) for single-objective
    and ``MultiObjectiveMCAcquisitionFunction._default_sample_shape`` (128) —
    so this changes reproducibility only, not the acquisition's statistics.
    """
    import torch
    from botorch.sampling.normal import SobolQMCNormalSampler

    if seed is None:
        seed = resolve_seed()
    num_samples = 128 if multi_objective else 512
    return SobolQMCNormalSampler(
        sample_shape=torch.Size([num_samples]), seed=int(seed),
    )
