"""The GP-vs-BNN cross-check must compare models, not inputs.

step6 tells the LLM to cite the GP-vs-BNN delta as a surrogate-confidence
signal, and `_build_agreement_summary` names the smallest-delta candidate as
"the safest GP-BNN-aligned pick". Two things had to be true for that number to
mean anything, and neither was:

  1. Both models must score the SAME points. `_attach_bnn_predictions` used to
     re-featurize the candidate table and zero-fill whatever `_align_to_training`
     couldn't find. The candidate table carries only composition, so every
     reaction condition was zero-filled for the BNN while the GP had seen it
     median-filled. Dormant under the shipped MAX_GP_FEATURES=30 (no condition
     column survives the |Pearson r| cap) but severe without it: measured up to
     15.7 standard deviations of disagreement on `reaction_temp_C`.

  2. The BNN number must be stable. It is a Monte-Carlo estimate; at the old
     50-sample default two independent draws differed by as much as the delta
     being reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import config
import step5_inverse as s5
from step4_surrogate import XYData


class _StubBNN:
    """Records the tensor it was asked to score."""

    def __init__(self, n_targets: int = 2):
        self.seen: torch.Tensor | None = None
        self.n_targets = n_targets

    def predict(self, X, n_samples=None):
        self.seen = X.clone()
        n = X.shape[0]
        # Deterministic, input-dependent so a wrong tensor gives wrong numbers.
        mean = X[:, :1].expand(n, self.n_targets).clone()
        return mean, torch.full((n, self.n_targets), 0.5, dtype=X.dtype)


def _xy(feature_cols, target_cols) -> XYData:
    d = len(feature_cols)
    return XYData(
        X=torch.zeros(4, d, dtype=torch.double),
        Y=torch.zeros(4, len(target_cols), dtype=torch.double),
        feature_cols=list(feature_cols),
        target_cols=list(target_cols),
        x_mean=torch.zeros(d, dtype=torch.double),
        x_std=torch.ones(d, dtype=torch.double),
        y_mean=torch.zeros(len(target_cols), dtype=torch.double),
        y_std=torch.ones(len(target_cols), dtype=torch.double),
    )


TARGETS = ["propane_conversion", "propane_selectivity"]


def test_bnn_scores_the_drivers_own_tensor():
    """The invariant the fix establishes: whatever tensor the driver scored is
    exactly what the cross-check scores."""
    data = _xy(["f0", "f1", "reaction_temp_C"], TARGETS)
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.double)
    cands = pd.DataFrame({"active_metal": ["Pt", "Pd"]})
    bnn = _StubBNN()

    out = s5._attach_bnn_predictions(cands, data, bnn, X)

    assert torch.equal(bnn.seen, X), "the BNN was fed a re-derived tensor"
    np.testing.assert_allclose(out["pred_propane_conversion_bnn"].values, [1.0, 4.0])


def test_condition_columns_are_not_zero_filled_for_the_bnn():
    """Regression for the specific artifact: a condition column carrying a
    median-filled value must reach the BNN at that value, not at 0."""
    data = _xy(["reaction_temp_C"], TARGETS)
    X = torch.tensor([[823.0], [873.0]], dtype=torch.double)
    bnn = _StubBNN()

    s5._attach_bnn_predictions(pd.DataFrame({"active_metal": ["Pt", "Pd"]}),
                               data, bnn, X)

    assert bnn.seen is not None
    assert not torch.any(bnn.seen == 0.0), "a condition column was zero-filled"
    np.testing.assert_allclose(bnn.seen.numpy().ravel(), [823.0, 873.0])


def test_row_count_mismatch_is_refused_not_broadcast():
    """Silently pairing k feature rows with n candidates would attach the wrong
    predictions to real catalysts. Refuse instead."""
    data = _xy(["f0"], TARGETS)
    cands = pd.DataFrame({"active_metal": ["Pt", "Pd", "Rh"]})
    bnn = _StubBNN()

    out = s5._attach_bnn_predictions(
        cands, data, bnn, torch.zeros(2, 1, dtype=torch.double),
    )

    assert bnn.seen is None, "scored a mismatched tensor"
    assert "pred_propane_conversion_bnn" not in out.columns
    assert len(out) == 3


def test_missing_tensor_skips_cleanly():
    data = _xy(["f0"], TARGETS)
    cands = pd.DataFrame({"active_metal": ["Pt"]})
    out = s5._attach_bnn_predictions(cands, data, _StubBNN(), None)
    assert "pred_propane_conversion_bnn" not in out.columns
    assert len(out) == 1


def test_predictions_are_unstandardized_with_training_moments():
    data = _xy(["f0"], TARGETS)
    data.y_mean = torch.tensor([10.0, 20.0], dtype=torch.double)
    data.y_std = torch.tensor([2.0, 4.0], dtype=torch.double)
    X = torch.tensor([[3.0]], dtype=torch.double)

    out = s5._attach_bnn_predictions(pd.DataFrame({"active_metal": ["Pt"]}),
                                     data, _StubBNN(), X)

    # stub mean = X[:, 0] = 3 -> 3*2+10 = 16 and 3*4+20 = 32; sd 0.5 -> 1.0, 2.0
    assert out["pred_propane_conversion_bnn"].iloc[0] == pytest.approx(16.0)
    assert out["pred_propane_selectivity_bnn"].iloc[0] == pytest.approx(32.0)
    assert out["pred_propane_conversion_bnn_sd"].iloc[0] == pytest.approx(1.0)
    assert out["pred_propane_selectivity_bnn_sd"].iloc[0] == pytest.approx(2.0)


def test_selection_keeps_batch_and_tensor_in_lockstep():
    """Selection is the mechanism that makes the invariant hold; if a future
    edit returns a bare DataFrame again this fails at import/attribute level."""
    cands = pd.DataFrame({"active_metal": ["Pt", "Pd"]})
    X = torch.zeros(2, 3, dtype=torch.double)
    sel = s5.Selection(cands, X)
    assert sel.candidates is cands and sel.X_std is X
    assert len(sel.candidates) == sel.X_std.shape[0]


def test_bnn_sample_count_default_is_not_noise_dominated():
    """Guards the noise floor. At the old default of 50 the MC error in the BNN
    mean was as large as the GP-vs-BNN delta the report cites, so the 'closest
    agreement' candidate was chosen out of noise."""
    assert getattr(config, "BNN_PREDICT_SAMPLES", 0) >= 256, (
        "BNN_PREDICT_SAMPLES below ~256 makes the reported GP-vs-BNN delta "
        "noise-dominated; see config.py for the measurement."
    )
