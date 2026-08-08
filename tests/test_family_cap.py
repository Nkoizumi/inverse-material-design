"""`BO_MAX_PER_FAMILY` on the role-based path.

The cap existed only for atomic-fraction mode; on the role-based path the
setting was read nowhere, so it silently did nothing. Family here is the
`active_metal` cell — the role-based analogue of fraction mode's dominant
non-support element, so the knob means the same thing on both paths.

What the cap is for, and what it is not: on training data dominated by one metal
(pdh_literature is 88% Pt) it will surface metals the surrogate has no signal
for. That is the point when hedging an experimental batch, and actively
misleading as a measure of the optimizer — see the 2026-07-01 per-role-cap
result, where diversity improved while the surrogate got worse.

`_dedup_and_cap` runs both filters in one pass because they interact: dedup
decides what is eligible, the cap decides what is diverse, and the fallback that
tops up a short batch must not reintroduce duplicates.
"""
from __future__ import annotations

import pandas as pd
import pytest

import step5_inverse as s5

ID_COLS = ["active_metal", "promoter_1"]


def _lib(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ID_COLS)


def test_without_a_cap_it_is_pure_dedup():
    lib = _lib([("Pt", "Sn"), ("Pt", "Sn"), ("Pt", "In"), ("Pd", "Sn")])

    assert s5._dedup_and_cap([0, 1, 2, 3], lib, ID_COLS, 4) == [0, 2, 3]


def test_identity_duplicates_are_dropped_before_the_cap_counts_them():
    """A duplicate must not consume a family slot — otherwise the cap would be
    spent on catalysts that never reach the batch.
    """
    lib = _lib([("Pt", "Sn"), ("Pt", "Sn"), ("Pt", "In"), ("Ni", "Sn")])

    out = s5._dedup_and_cap([0, 1, 2, 3], lib, ID_COLS, 3,
                            family_cap=2, family_col="active_metal")

    assert out == [0, 2, 3]          # both Pt survive; the dup did not count


def test_the_cap_limits_one_metal_and_lets_others_through():
    lib = _lib([("Pt", "A"), ("Pt", "B"), ("Pt", "C"), ("Pd", "A"), ("Ni", "A")])

    out = s5._dedup_and_cap(list(range(5)), lib, ID_COLS, 4,
                            family_cap=2, family_col="active_metal")

    metals = [lib.iloc[i]["active_metal"] for i in out]
    assert metals.count("Pt") == 2
    assert set(metals) == {"Pt", "Pd", "Ni"}


def test_acquisition_order_is_preserved():
    """The cap reorders nothing; it only removes. The first Pt rows offered are
    the ones kept, because those carry the higher acquisition value.
    """
    lib = _lib([("Pt", "A"), ("Pt", "B"), ("Pt", "C"), ("Pd", "A")])

    out = s5._dedup_and_cap([0, 1, 2, 3], lib, ID_COLS, 3,
                            family_cap=2, family_col="active_metal")

    assert out == [0, 1, 3]


def test_a_short_batch_is_topped_up_rather_than_returned_short():
    """Only Pt available, cap 2, but 4 slots asked for. An under-filled batch is
    worse than a less diverse one, so the cap yields.
    """
    lib = _lib([("Pt", "A"), ("Pt", "B"), ("Pt", "C"), ("Pt", "D")])

    out = s5._dedup_and_cap([0, 1, 2, 3], lib, ID_COLS, 4,
                            family_cap=2, family_col="active_metal")

    assert len(out) == 4
    assert sorted(out) == [0, 1, 2, 3]


def test_the_top_up_never_reintroduces_a_duplicate():
    """The regression that matters: filling from the raw acquisition order
    instead of from dedup survivors would undo the dedup.
    """
    lib = _lib([("Pt", "A"), ("Pt", "A"), ("Pt", "B"), ("Pt", "B")])

    out = s5._dedup_and_cap([0, 1, 2, 3], lib, ID_COLS, 4,
                            family_cap=1, family_col="active_metal")

    identities = [tuple(lib.iloc[i]) for i in out]
    assert len(identities) == len(set(identities)), identities


def test_the_fallback_pool_includes_rows_scanned_after_the_batch_filled():
    """`kept` fills early with Pt; the Pd row appears later in acquisition order
    and must still be reachable as a top-up.
    """
    lib = _lib([("Pt", "A"), ("Pt", "B"), ("Pd", "A")])

    out = s5._dedup_and_cap([0, 1, 2], lib, ID_COLS, 3,
                            family_cap=5, family_col="active_metal")

    assert len(out) == 3


def test_a_blank_metal_is_its_own_family_not_a_crash():
    lib = _lib([("", "A"), ("", "B"), ("Pt", "A")])

    out = s5._dedup_and_cap([0, 1, 2], lib, ID_COLS, 3,
                            family_cap=1, family_col="active_metal")

    assert len(out) == 3             # 1 blank + 1 Pt under cap, 1 topped up
