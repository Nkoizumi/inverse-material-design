"""Tests for the shared step-5 machinery extracted in the Phase 2 consolidation.

The four `_run_*` paths used to carry copy-pasted acquisition, row-recovery and
prediction-attachment blocks. Two alignment bugs lived in exactly one copy each
(the steels library row misalignment and _nearest_known's index mismatch),
which is the argument for having one implementation and testing it here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import config
import step5_inverse as s5
from step4_surrogate import XYData

# Production tensors all come from prepare_xy on a single device; build the
# fixtures on the same one so the helpers are exercised the way they run.
DEV = s5.DEVICE


def _t(data, **kw):
    return torch.tensor(data, dtype=torch.double, device=DEV, **kw)


def _xy(n=5, d=3, t=1, target_cols=None) -> XYData:
    target_cols = target_cols or [f"y{i}" for i in range(t)]
    return XYData(
        X=torch.arange(n * d, dtype=torch.double, device=DEV).reshape(n, d),
        Y=torch.zeros(n, len(target_cols), dtype=torch.double, device=DEV),
        feature_cols=[f"f{i}" for i in range(d)],
        target_cols=list(target_cols),
        x_mean=torch.zeros(d, dtype=torch.double, device=DEV),
        x_std=torch.ones(d, dtype=torch.double, device=DEV),
        y_mean=torch.zeros(len(target_cols), dtype=torch.double, device=DEV),
        y_std=torch.ones(len(target_cols), dtype=torch.double, device=DEV),
    )


# ── _recover_library_rows ────────────────────────────────────────────────────
def test_recover_library_rows_finds_exact_rows():
    lib = _t([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    picked = lib[[2, 0, 3]]
    assert s5._recover_library_rows(lib, picked) == [2, 0, 3]


def test_recover_library_rows_preserves_acquisition_order():
    """Downstream code (the family cap, the dedup) consumes these in
    acquisition order, so the mapping must not sort or dedupe."""
    lib = _t([[float(i)] for i in range(10)])
    order = [7, 1, 9, 0]
    assert s5._recover_library_rows(lib, lib[order]) == order


def test_recover_library_rows_handles_near_misses():
    """optimize_acqf_discrete returns values, not indices; tiny float drift
    must still resolve to the intended row."""
    lib = _t([[0.0], [10.0], [20.0]])
    picked = _t([[19.9999999], [0.0000001]])
    assert s5._recover_library_rows(lib, picked) == [2, 0]


def test_recover_library_rows_on_duplicate_rows_picks_first():
    """Duplicate library tensors are real (loading variants can be numerically
    indistinguishable). argmin takes the first; the caller's identity dedup is
    what collapses them."""
    lib = _t([[1.0], [1.0], [2.0]])
    assert s5._recover_library_rows(lib, _t([[1.0]])) == [0]


# ── _nearest_known (A6) ──────────────────────────────────────────────────────
def test_nearest_known_realigns_after_nan_target_rows_are_dropped():
    """prepare_xy drops rows with a NaN in any target, so data.X indexes the
    SURVIVORS. Indexing the full df with a data.X row index reported the wrong
    material — silently, since both are valid formulas."""
    df = pd.DataFrame({
        "composition_str": ["AAA", "BBB", "CCC", "DDD"],
        "y0": [1.0, np.nan, 3.0, 4.0],          # row 1 is dropped by prepare_xy
    })
    data = _xy(n=3, d=1, target_cols=["y0"])
    data.X = _t([[0.0], [10.0], [20.0]])

    got = s5._nearest_known(_t([[10.0]]), data, df)

    # Survivors are AAA, CCC, DDD -> index 1 is CCC. The old code indexed the
    # full frame and would have said BBB, a row the surrogate never saw.
    assert got == ["CCC"]


def test_nearest_known_without_nan_targets_is_unchanged():
    df = pd.DataFrame({"composition_str": ["AAA", "BBB", "CCC"],
                       "y0": [1.0, 2.0, 3.0]})
    data = _xy(n=3, d=1, target_cols=["y0"])
    data.X = _t([[0.0], [10.0], [20.0]])
    got = s5._nearest_known(_t([[20.0], [0.0]]), data, df)
    assert got == ["CCC", "AAA"]


def test_nearest_known_refuses_to_guess_when_rows_cannot_be_aligned():
    """If the row counts still disagree after re-applying the mask, something
    dropped rows this function doesn't model — say so rather than mislabel."""
    df = pd.DataFrame({"composition_str": ["AAA", "BBB"], "y0": [1.0, 2.0]})
    data = _xy(n=5, d=1, target_cols=["y0"])
    data.X = torch.zeros(5, 1, dtype=torch.double, device=DEV)
    assert s5._nearest_known(torch.zeros(2, 1, dtype=torch.double, device=DEV), data, df) == ["?", "?"]


# ── _attach_predictions ──────────────────────────────────────────────────────
class _StubSurrogate:
    def predict(self, X):
        n = X.shape[0]
        return (X[:, :1].expand(n, 2).clone(),
                torch.full((n, 2), 0.25, dtype=X.dtype, device=X.device))


def test_attach_predictions_unstandardizes_with_training_moments():
    data = _xy(n=2, d=1, target_cols=["a", "b"])
    data.y_mean = _t([100.0, 200.0])
    data.y_std = _t([2.0, 4.0])
    sel = pd.DataFrame({"name": ["x", "y"]})

    out = s5._attach_predictions(sel, _t([[1.0], [3.0]]),
                                 _StubSurrogate(), data)

    np.testing.assert_allclose(out["pred_a"].values, [102.0, 106.0])
    np.testing.assert_allclose(out["pred_b"].values, [204.0, 212.0])
    np.testing.assert_allclose(out["pred_a_sd"].values, [0.5, 0.5])
    np.testing.assert_allclose(out["pred_b_sd"].values, [1.0, 1.0])


# ── _apply_transformer ───────────────────────────────────────────────────────
def test_apply_transformer_is_a_noop_when_none():
    lib = pd.DataFrame({"a": [1.0, 2.0]})
    assert s5._apply_transformer(lib, None, "x") is lib


def test_apply_transformer_falls_back_on_failure():
    """A mis-transformed library is worse than an untransformed one, so a
    raising transformer must not take the run down."""
    class _Boom:
        feature_names_in_ = ["a"]
        def transform(self, X):
            raise RuntimeError("nope")

    lib = pd.DataFrame({"a": [1.0, 2.0]})
    out = s5._apply_transformer(lib, _Boom(), "x")
    pd.testing.assert_frame_equal(out, lib)


def test_apply_transformer_fills_missing_fit_columns():
    class _Echo:
        feature_names_in_ = ["a", "b"]
        def transform(self, X):
            return X.values
        def get_feature_names_out(self):
            return ["a", "b"]

    lib = pd.DataFrame({"a": [1.0, 2.0]})          # "b" absent
    out = s5._apply_transformer(lib, _Echo(), "x")
    assert list(out.columns) == ["a", "b"]
    np.testing.assert_allclose(out["b"].values, [0.0, 0.0])


# ── _align_to_training ───────────────────────────────────────────────────────
def test_align_to_training_orders_and_fills():
    lib = pd.DataFrame({"c": [3.0], "a": [1.0]})
    arr = s5._align_to_training(lib, ["a", "b", "c"])
    np.testing.assert_allclose(arr, [[1.0, 0.0, 3.0]])


# ── _build_acquisition ───────────────────────────────────────────────────────
def test_ref_point_is_a_plain_list_not_a_tensor(monkeypatch):
    """BoTorch 0.18 accepts a Tensor, but the list form is the documented one
    and the Tensor form has silently mis-handled MOBO in other versions."""
    monkeypatch.setattr(config, "OPTIMIZATION_DIRECTIONS", ["max", "min"])
    captured = {}

    class _Model:
        pass

    class _Surr:
        model = _Model()

    import botorch.acquisition.multi_objective.logei as mol

    def _fake(**kwargs):
        captured.update(kwargs)
        return "acq"

    monkeypatch.setattr(mol, "qLogNoisyExpectedHypervolumeImprovement", _fake)

    data = _xy(n=4, d=2, target_cols=["a", "b"])
    data.Y = _t([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])

    s5._build_acquisition(_Surr(), data, multi_objective=True)

    ref = captured["ref_point"]
    assert isinstance(ref, list), f"ref_point must be list[float], got {type(ref)}"
    assert all(isinstance(v, float) for v in ref)
    # signs = [+1, -1]; min over rows of (Y*signs) is [1, -8]; minus 1.
    assert ref == pytest.approx([0.0, -9.0])
