"""Static checks that config.TARGET_TWINS stays in sync with every target the
project can actually be run against.

The twin guard in step4_surrogate.prepare_xy is a name-based blocklist. If a
new target gets added without its raw/log/unit siblings being added to
TARGET_TWINS, the Pearson-|r| feature ranker will pick the sibling as the
strongest predictor and the surrogate will fit the target with itself.

IMPORTANT — why these tests iterate over presets rather than ambient config:
TARGET_TWINS is deliberately a UNION across every preset, because config.py's
own TARGET_COLS is just whichever preset was last edited in by hand. An
earlier version of this file asserted every twin shared a base with the
*ambient* config.TARGET_COLS, which made the suite red out of the box (7
"orphans" that are all legitimate entries for other presets). The invariant
that actually matters is: every target of every runnable configuration has
its siblings blocked, and no twin is a stray name that would drop a real
feature column.
"""
from __future__ import annotations

import re

import pytest

import config
from run_pipeline import PRESETS

KNOWN_SUFFIXES = ("_s_log", "_h_log", "_pct_log", "_log", "_s", "_h", "_pct")


def _base_of(name: str) -> str | None:
    """Return the base name if `name` ends in a recognized suffix, else None."""
    for suf in KNOWN_SUFFIXES:
        if name.endswith(suf) and len(name) > len(suf):
            return name[: -len(suf)]
    return None


def _all_configured_targets() -> set[str]:
    """Every target column reachable from ambient config or any preset."""
    targets = set(config.TARGET_COLS)
    for preset in PRESETS.values():
        targets |= set(preset.get("TARGET_COLS", []))
    return targets


def _catalyst_targets() -> set[str]:
    """Configured targets excluding the non-catalyst benchmark presets.

    matbench targets ("yield strength", "gap expt") are plain single-form
    property names from an external benchmark. They have no raw/log siblings
    in their source dataset and nothing to guard, so requiring them to appear
    in TARGET_TWINS would be noise.
    """
    benchmark_targets: set[str] = set()
    for preset in PRESETS.values():
        if preset.get("DATASET_SOURCE") == "matminer":
            benchmark_targets |= set(preset.get("TARGET_COLS", []))
    return _all_configured_targets() - benchmark_targets


# ── The core invariant ───────────────────────────────────────────────────────
@pytest.mark.parametrize("target", sorted(_catalyst_targets()))
def test_every_catalyst_target_is_itself_blocked(target: str) -> None:
    """Every catalyst target must appear in TARGET_TWINS.

    prepare_xy drops `TARGET_COLS` separately, so this looks redundant for the
    target you're currently optimizing — but it is NOT redundant across
    presets. `propylene_yield` is a target under `acs_pdh` and a leaking
    outcome column under any preset that optimizes deactivation alone. The
    blocklist has to cover it either way.
    """
    assert target in config.TARGET_TWINS, (
        f"Target {target!r} is optimized by at least one configuration but is "
        f"not in config.TARGET_TWINS. Any OTHER configuration run against the "
        f"same CSV will see it as a feature column and target-leak through the "
        f"Pearson-|r| ranker in step4_surrogate.prepare_xy."
    )


@pytest.mark.parametrize("target", sorted(_catalyst_targets()))
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


def test_no_stray_twin_entries() -> None:
    """No TARGET_TWINS entry may be an unrelated name.

    A typo or an over-broad entry (e.g. 'temperature') silently deletes a real
    predictor column from every run. Legitimacy is judged against the union of
    targets over ambient config AND all presets — not ambient config alone.
    """
    target_bases = {(_base_of(t) or t) for t in _all_configured_targets()}
    target_bases |= _all_configured_targets()
    orphans = sorted(
        t for t in config.TARGET_TWINS
        if not any(t == b or t.startswith(b + "_") for b in target_bases)
    )
    assert not orphans, (
        f"TARGET_TWINS entries {orphans!r} don't share a base with any target "
        f"in config.TARGET_COLS or in run_pipeline.PRESETS. Either add the "
        f"matching target/preset or remove the twin — leaving it in will "
        f"silently drop a real feature column."
    )


# ── Preset sanity ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_directions_match_targets(name: str) -> None:
    """OPTIMIZATION_DIRECTIONS length must match TARGET_COLS.

    step5_inverse._get_signs raises on a mismatch, but only after step 1, 2
    and 4 have already run — on the ACS preset that is several minutes of
    featurization and GP fitting thrown away.
    """
    preset = PRESETS[name]
    targets = preset.get("TARGET_COLS")
    directions = preset.get("OPTIMIZATION_DIRECTIONS")
    if targets is None or directions is None:
        pytest.skip(f"Preset {name!r} does not set both keys.")
    assert len(targets) == len(directions), (
        f"Preset {name!r} has {len(targets)} target(s) but "
        f"{len(directions)} direction(s)."
    )
    bad = [d for d in directions if d not in ("max", "min")]
    assert not bad, f"Preset {name!r} has invalid directions {bad!r}."


def test_ambient_config_directions_match_targets() -> None:
    assert len(config.TARGET_COLS) == len(config.OPTIMIZATION_DIRECTIONS), (
        f"config.TARGET_COLS has {len(config.TARGET_COLS)} entries but "
        f"OPTIMIZATION_DIRECTIONS has {len(config.OPTIMIZATION_DIRECTIONS)}."
    )
