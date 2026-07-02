"""Static checks that config.TARGET_TWINS stays in sync with TARGET_COLS.

The twin guard in step4_surrogate.prepare_xy is a name-based blocklist. If a
new target gets added to TARGET_COLS without its raw/log/unit siblings being
added to TARGET_TWINS, the Pearson-|r| feature ranker will pick the sibling
as the strongest predictor and the surrogate will fit the target with itself.
These tests fail loudly at CI time instead of producing a silently-bad model.
"""
from __future__ import annotations

import re

import pytest

import config

KNOWN_SUFFIXES = ("_s_log", "_h_log", "_pct_log", "_log", "_s", "_h", "_pct")


def _base_of(name: str) -> str | None:
    """Return the base name if `name` ends in a recognized suffix, else None."""
    for suf in KNOWN_SUFFIXES:
        if name.endswith(suf) and len(name) > len(suf):
            return name[: -len(suf)]
    return None


@pytest.mark.parametrize("target", config.TARGET_COLS)
def test_each_target_has_twin_siblings(target: str) -> None:
    base = _base_of(target)
    if base is None:
        pytest.skip(
            f"Target {target!r} has no recognized suffix — assumed single-form, "
            f"no siblings to guard."
        )
    pattern = re.compile(rf"^{re.escape(base)}(_|$)")
    siblings = {t for t in config.TARGET_TWINS if pattern.match(t)} - {target}
    assert siblings, (
        f"Target {target!r} (base {base!r}) has no sibling entry in "
        f"TARGET_TWINS. Add the raw form (e.g. {base!r}) and any unit "
        f"variants (e.g. {base + '_s'!r}) to config.TARGET_TWINS, or the "
        f"Pearson-|r| feature ranker in step4_surrogate.prepare_xy will pick "
        f"them as the top predictor and the surrogate will fit the target "
        f"with itself."
    )


def test_target_twins_is_a_set() -> None:
    """Type guard — `set(getattr(config, 'TARGET_TWINS', set()))` would mask
    a list-with-duplicates bug. Make it explicit."""
    assert isinstance(config.TARGET_TWINS, (set, frozenset)), (
        "TARGET_TWINS must be a set; got "
        f"{type(config.TARGET_TWINS).__name__}."
    )


def test_no_target_twin_collides_with_a_feature_name() -> None:
    """Sanity check: TARGET_TWINS shouldn't accidentally contain a generic
    feature name (e.g. 'temperature') that would silently drop a real
    predictor. All twins must share a base with some target."""
    target_bases = {(_base_of(t) or t) for t in config.TARGET_COLS}
    target_bases |= set(config.TARGET_COLS)
    orphans = [
        t for t in config.TARGET_TWINS
        if not any(t == b or t.startswith(b + "_") for b in target_bases)
    ]
    assert not orphans, (
        f"TARGET_TWINS entries {orphans!r} don't share a base with any "
        f"TARGET_COLS entry. Either add the matching target or remove the "
        f"twin — leaving it in will silently drop a real feature column."
    )
