"""Declared schema and validation for `config`.

WHY THIS IS A VALIDATOR AND NOT A DATACLASS
-------------------------------------------
The obvious "PipelineConfig" refactor is a dataclass holding all 54 settings,
threaded through every function. Two reasons that is not what this is:

  * Restating all 54 defaults in a dataclass makes config.py and the dataclass
    two sources for the same values — a drift surface, which is the exact class
    of bug this codebase keeps producing (the `interfacial_`/`intf_` mismatch,
    the two copies of the R²/MAE arithmetic, the target-twin blocklist that has
    to be kept in step with TARGET_COLS by hand).
  * Threading a config object through 233 read sites in 16 modules is a very
    large change to a working scientific pipeline, and the payoff — reentrancy
    — is not what actually bit us.

What DID bite us is that config is an undeclared, unvalidated namespace:

  * `webui/app.py` read CATALYST_MODE / CATALYST_FRACTION_MODE but never set
    them, so an atomic-fraction upload silently ran through the role-based
    featurizer and bypassed the `score` target-leak guard with it. A rule that
    the schema mode must be internally consistent would have caught it.
  * Five settings (STEELS_LIBRARY, BO_ACQ_BATCH_SIZE, …) were read via
    `getattr(config, ..., default)` and declared nowhere, so a typo in either
    the reader or a preset silently fell back to the default.
  * Five more were declared and read nowhere — including
    `CHEM_VIABILITY_FILTER = True  # SMACT charge-balance`, which told a
    materials scientist that generated candidates were screened for chemical
    viability. They were not, and `smact` is imported nowhere in the project.

So this module declares WHAT may be set and checks that a config is
self-consistent, without owning the values. `check()` runs at pipeline entry so
an invalid combination fails immediately rather than several minutes into a
featurization.
"""
from __future__ import annotations

from pathlib import Path

# Names that exist for path plumbing rather than as user knobs.
_INTERNAL = {"ROOT", "DATA_DIR", "REPORTS_DIR"}

# name -> accepted types. `None` in the tuple means the setting is nullable.
SETTINGS: dict[str, tuple] = {
    # dataset
    "DATASET_SOURCE": (str,),
    "MATMINER_DATASET": (str,),
    "CSV_PATH": (Path, str, type(None)),
    "TARGET_COLS": (list,),
    "OPTIMIZATION_DIRECTIONS": (list,),
    "TARGET_TWINS": (set, frozenset),
    "TASK": (str,),
    "SUBSAMPLE_N": (int, type(None)),
    "RANDOM_STATE": (int,),
    # role-based catalyst schema
    "CATALYST_MODE": (bool,),
    "CATALYST_ROLES": (dict,),
    "CATALYST_LOADINGS": (dict,),
    "SUPPORT_LOOKUP_PATH": (Path, str),
    "METAL_LOOKUP_PATH": (Path, str),
    # atomic-fraction catalyst schema
    "CATALYST_FRACTION_MODE": (bool,),
    "CATALYST_FRACTION_ELEMENTS": (list,),
    "CATALYST_FRACTION_SUPPORT_CATIONS": (list,),
    "CATALYST_FRACTION_SUPPORT_OXIDE_MAP": (dict,),
    "CATALYST_FRACTION_CONDITIONS": (list,),
    "CATALYST_FRACTION_RENAME": (dict,),
    "CATALYST_FRACTION_LIBRARY_SIZE": (int,),
    "CATALYST_FRACTION_MAX_TOTAL_METAL": (float, int),
    # optional measured features
    "OPTIONAL_NUMERIC_FEATURES": (list,),
    # auto-EDA
    "OLLAMA_HOST": (str,),
    "AUTO_EDA_AVAILABLE": (bool,),
    "AUTO_EDA_PATH": (Path, str, type(None)),
    # surrogates
    "SURROGATE_KIND": (str,),
    "BNN_TRAINING_ITERS": (int,),
    "BNN_PREDICT_SAMPLES": (int,),
    "SVGP_NUM_INDUCING": (int,),
    "SVGP_TRAINING_ITERS": (int,),
    "MAX_GP_FEATURES": (int, type(None)),
    "PER_ROLE_FEATURE_CAP": (bool,),
    "COMPOSITION_ONLY_FEATURES": (bool,),
    "GP_LENGTHSCALE_PRIOR_CONCENTRATION": (float, int, type(None)),
    "GP_LENGTHSCALE_PRIOR_RATE": (float, int, type(None)),
    "CV_FOLDS": (int, type(None)),
    # inverse design
    "BO_BATCH_SIZE": (int,),
    "BO_UNIQUE_OVERSAMPLE": (int,),
    "BO_MAX_PER_FAMILY": (int, type(None)),
    "BO_FAMILY_OVERSAMPLE": (int,),
    "BO_ACQ_BATCH_SIZE": (int,),
    # role-based BO library
    "LIBRARY_ACTIVE_METALS": (list,),
    "LIBRARY_PROMOTERS_1": (list,),
    "LIBRARY_PROMOTERS_2": (list,),
    "LIBRARY_METAL_LOADINGS": (list,),
    "LIBRARY_PROMO_LOADINGS": (list,),
    # steels benchmark library
    "STEELS_LIBRARY": (bool,),
    "STEELS_LIBRARY_SIZE": (int,),
    # report
    "REPORT_LLM_MODEL": (str,),
    "REPORT_TOP_K": (int,),
    # Ollama `keep_alive`: seconds as int/float, a duration string like "5m",
    # or -1 for "never unload".
    "REPORT_LLM_KEEP_ALIVE": (int, float, str),
}


def _get(cfg, name):
    return getattr(cfg, name, None)


def validate(cfg) -> list[str]:
    """Return a list of problems with `cfg`. Empty means it is consistent."""
    problems: list[str] = []

    # ── 1. Types, and names nobody declared ──────────────────────────────────
    for name, types in SETTINGS.items():
        if not hasattr(cfg, name):
            continue                       # optional; readers supply defaults
        value = getattr(cfg, name)
        if not isinstance(value, types):
            problems.append(
                f"{name} should be {' | '.join(t.__name__ for t in types)}, "
                f"got {type(value).__name__} ({value!r})"
            )

    unknown = sorted(
        n for n in dir(cfg)
        if n.isupper() and not n.startswith("_")
        and n not in SETTINGS and n not in _INTERNAL
    )
    if unknown:
        problems.append(
            f"unknown setting(s) {unknown} — a typo here is silent, because "
            f"readers fall back to a default. Add them to "
            f"pipeline/config_schema.SETTINGS if they are real."
        )

    # ── 2. Targets and directions ────────────────────────────────────────────
    targets = _get(cfg, "TARGET_COLS") or []
    directions = _get(cfg, "OPTIMIZATION_DIRECTIONS") or []
    if not targets:
        problems.append("TARGET_COLS is empty; there is nothing to optimize.")
    if len(targets) != len(directions):
        problems.append(
            f"OPTIMIZATION_DIRECTIONS has {len(directions)} entries for "
            f"{len(targets)} target(s); they must correspond one-to-one."
        )
    bad = [d for d in directions if d not in ("max", "min")]
    if bad:
        problems.append(f"OPTIMIZATION_DIRECTIONS entries must be 'max' or 'min'; got {bad}.")

    # ── 3. Schema mode ───────────────────────────────────────────────────────
    # The bug this rule exists for: the web UI read these flags but never set
    # them, so an atomic-fraction CSV ran through the role-based featurizer.
    role = bool(_get(cfg, "CATALYST_MODE"))
    fraction = bool(_get(cfg, "CATALYST_FRACTION_MODE"))
    if role and fraction:
        problems.append(
            "CATALYST_MODE and CATALYST_FRACTION_MODE are both True. "
            "step2_featurize gives precedence to fraction mode, so the role "
            "settings would be silently ignored — set exactly one."
        )

    if fraction:
        elements = _get(cfg, "CATALYST_FRACTION_ELEMENTS") or []
        cations = _get(cfg, "CATALYST_FRACTION_SUPPORT_CATIONS") or []
        oxides = _get(cfg, "CATALYST_FRACTION_SUPPORT_OXIDE_MAP") or {}
        if not elements:
            problems.append("CATALYST_FRACTION_MODE is on but "
                            "CATALYST_FRACTION_ELEMENTS is empty.")
        stray = [c for c in cations if c not in elements]
        if stray:
            problems.append(
                f"CATALYST_FRACTION_SUPPORT_CATIONS {stray} are not in "
                f"CATALYST_FRACTION_ELEMENTS, so they can never be detected."
            )
        missing = [c for c in cations if c not in oxides]
        if missing:
            problems.append(
                f"support cation(s) {missing} have no entry in "
                f"CATALYST_FRACTION_SUPPORT_OXIDE_MAP, so their support "
                f"lookup silently yields NaN."
            )

    if role:
        roles = _get(cfg, "CATALYST_ROLES") or {}
        loadings = _get(cfg, "CATALYST_LOADINGS") or {}
        if not roles:
            problems.append("CATALYST_MODE is on but CATALYST_ROLES is empty.")
        stray = [r for r in loadings if r not in roles]
        if stray:
            problems.append(
                f"CATALYST_LOADINGS names role(s) {stray} absent from "
                f"CATALYST_ROLES; those loadings are never read."
            )

    # ── 4a. Targets named before step 1 renames them ─────────────────────────
    # step1_load rewrites an atomic-fraction CSV's headers to snake_case before
    # anything else sees the frame. A target given in its RAW form does not
    # exist afterwards, and step 1 dies with a KeyError. That is a different
    # fault from the leak rule below, and reporting it as a leak sent at least
    # one user looking in entirely the wrong place — so name it precisely and
    # exclude those targets from the leak check.
    rename = _get(cfg, "CATALYST_FRACTION_RENAME") or {}
    pre_rename = [t for t in targets if t in rename]
    if pre_rename:
        mapping = ", ".join(f"{t!r} → {rename[t]!r}" for t in pre_rename)
        problems.append(
            f"target(s) {pre_rename} are RAW CSV header names. step 1 renames "
            f"them ({mapping}), so the name you gave does not exist by the time "
            f"the surrogate is fitted. Select the renamed form instead."
            + (" Note 'deactivation_rate_h' is further derived into "
               "'deactivation_rate_log', which is the form the presets optimize."
               if any(rename[t].endswith("_rate_h") for t in pre_rename) else "")
        )

    # ── 4b. The target-leak guard ────────────────────────────────────────────
    # prepare_xy drops TARGET_TWINS by name. A target missing from that set is
    # a feature column for any OTHER configuration run against the same CSV.
    twins = _get(cfg, "TARGET_TWINS") or set()
    if (role or fraction) and twins:
        unguarded = [t for t in targets if t not in twins and t not in rename]
        if unguarded:
            problems.append(
                f"target(s) {unguarded} are not in TARGET_TWINS. Any run that "
                f"optimizes a different target against the same CSV will see "
                f"them as ordinary features and leak the target into the "
                f"surrogate."
            )

    # ── 5. Dataset source ────────────────────────────────────────────────────
    source = _get(cfg, "DATASET_SOURCE")
    if source not in ("csv", "matminer"):
        problems.append(f"DATASET_SOURCE must be 'csv' or 'matminer'; got {source!r}.")
    if source == "csv" and _get(cfg, "CSV_PATH") is None:
        problems.append("DATASET_SOURCE is 'csv' but CSV_PATH is None.")

    # ── 6. Numeric ranges ────────────────────────────────────────────────────
    for name, low in (("BO_BATCH_SIZE", 1), ("BO_UNIQUE_OVERSAMPLE", 1),
                      ("BO_FAMILY_OVERSAMPLE", 1), ("REPORT_TOP_K", 1),
                      ("BNN_PREDICT_SAMPLES", 1)):
        v = _get(cfg, name)
        if isinstance(v, int) and v < low:
            problems.append(f"{name} must be >= {low}; got {v}.")
    folds = _get(cfg, "CV_FOLDS")
    if isinstance(folds, int) and folds == 1:
        problems.append("CV_FOLDS=1 cannot form a held-out split; use 0 to "
                        "disable CV or >= 2 to enable it.")
    cap = _get(cfg, "MAX_GP_FEATURES")
    if isinstance(cap, int) and cap < 0:
        problems.append(f"MAX_GP_FEATURES must be None, 0 (disabled) or > 0; got {cap}.")

    # ── 7. Lookups the featurizers need ──────────────────────────────────────
    for name in ("SUPPORT_LOOKUP_PATH", "METAL_LOOKUP_PATH"):
        p = _get(cfg, name)
        if p is not None and not Path(p).exists():
            problems.append(f"{name} does not exist: {p}")

    return problems


def check(cfg) -> None:
    """Raise ValueError if `cfg` is inconsistent. Call at pipeline entry.

    Failing here costs a second. The alternative is discovering the problem
    several minutes into a featurization, or — worse, and this is what actually
    happened — not discovering it at all, because the run completes and
    produces plausible candidates from the wrong featurizer.
    """
    problems = validate(cfg)
    if problems:
        raise ValueError(
            "Configuration is inconsistent:\n"
            + "\n".join(f"  • {p}" for p in problems)
        )
