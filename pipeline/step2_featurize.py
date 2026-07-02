"""Step 2: featurize material compositions with matminer.

Uses a minimal but real feature set: ElementProperty (Magpie) + Stoichiometry.
Add more featurizers (ValenceOrbital, IonProperty, OxidationStates, etc.) once the
skeleton is validated.
"""
from __future__ import annotations

import logging
import pandas as pd

import config

log = logging.getLogger(__name__)


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    # CATALYST_FRACTION_MODE takes precedence: it's the atomic-fraction path
    # for datasets like the ACS Materials Letters PDH set.
    if getattr(config, "CATALYST_FRACTION_MODE", False):
        log.info("Featurization mode: catalyst_fractions (atomic-fraction path).")
        return _featurize_catalyst_fractions(df)
    if config.CATALYST_MODE:
        log.info("Featurization mode: catalyst (role-based path).")
        return _featurize_catalyst(df)
    log.info("Featurization mode: single_formula (matminer Magpie + Stoichiometry).")
    return _featurize_single_formula(df)


def _save_parquet(out: pd.DataFrame) -> None:
    """Cache the featurized df. Object columns → nullable `string` so that real
    NaNs survive the parquet round-trip (plain `.astype(str)` turns them into
    the literal "nan")."""
    cache = config.DATA_DIR / "featurized.parquet"
    out = out.copy()
    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].astype("string")
    out.to_parquet(cache)
    log.info("Saved featurized dataset to %s.", cache)


def _featurize_catalyst_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Atomic-fraction catalyst featurization (see catalyst_fraction_features)."""
    from catalyst_fraction_features import featurize_fractions

    log.info("Atomic-fraction catalyst mode: metal-phase Magpie + support "
             "lookup + interfacial + raw fractions + reaction conditions.")
    work = featurize_fractions(
        df,
        element_cols=config.CATALYST_FRACTION_ELEMENTS,
        support_cations=config.CATALYST_FRACTION_SUPPORT_CATIONS,
        support_oxide_map=config.CATALYST_FRACTION_SUPPORT_OXIDE_MAP,
        condition_cols=config.CATALYST_FRACTION_CONDITIONS,
        support_lookup_path=config.SUPPORT_LOOKUP_PATH,
        metal_lookup_path=config.METAL_LOOKUP_PATH,
    )
    feature_cols = [c for c in work.columns if c not in df.columns]
    log.info("Atomic-fraction featurization produced %d new feature columns.",
             len(feature_cols))
    # NaN policy: keep rows. featurize_fractions leaves NaN in support_phys_*
    # for metal-rich rows with no dominant support cation; step4.prepare_xy
    # drops all-NaN columns and imputes remaining cells with 0.
    _save_parquet(work)
    return work


def _featurize_catalyst(df: pd.DataFrame) -> pd.DataFrame:
    """Heterogeneous catalyst featurization: active metal + promoter + support."""
    from catalyst_features import CatalystFeaturizer

    log.info("Catalyst mode: per-role matminer + support/metal lookups + interfacial.")
    cf = CatalystFeaturizer(
        roles=config.CATALYST_ROLES,
        loadings=config.CATALYST_LOADINGS,
        support_lookup_path=config.SUPPORT_LOOKUP_PATH,
        metal_lookup_path=config.METAL_LOOKUP_PATH,
        optional_numeric_features=getattr(config, "OPTIONAL_NUMERIC_FEATURES", []),
    )
    work = cf.fit_transform(df)

    # Cache + report.
    feature_cols = [c for c in work.columns if c not in df.columns]
    log.info("Catalyst featurization produced %d new feature columns.", len(feature_cols))
    # NaN policy: keep rows. CatalystFeaturizer leaves NaN for absent roles
    # and missing optional features; step4.prepare_xy drops all-NaN columns,
    # imputes the rest with 0, and uses {col}_present indicators to keep
    # "absent" distinguishable from "measured = 0".
    _save_parquet(work)
    return work


def _featurize_single_formula(df: pd.DataFrame) -> pd.DataFrame:
    from matminer.featurizers.composition import ElementProperty, Stoichiometry
    from matminer.featurizers.conversions import StrToComposition
    from pymatgen.core.composition import Composition

    work = df.copy()

    # Ensure we have pymatgen Composition objects in `composition` column.
    if "composition" not in work.columns:
        # User CSVs may have a "formula" column instead.
        formula_col = next(
            (c for c in ("formula", "Formula", "Composition") if c in work.columns), None
        )
        if formula_col is None:
            raise KeyError(
                "Expected a `composition` or `formula` column for featurization. "
                f"Got: {work.columns.tolist()}"
            )
        work = StrToComposition(target_col_id="composition").featurize_dataframe(
            work, formula_col, ignore_errors=True
        )

    # Some matminer datasets already give Composition objects; some give strings.
    if not isinstance(work["composition"].iloc[0], Composition):
        work["composition"] = work["composition"].apply(Composition)

    featurizers = [
        ElementProperty.from_preset("magpie"),
        Stoichiometry(),
    ]

    for fz in featurizers:
        log.info("Applying %s …", type(fz).__name__)
        work = fz.featurize_dataframe(work, "composition", ignore_errors=True)

    # Drop any rows that failed featurization.
    feature_cols = [c for c in work.columns if c not in df.columns and c != "composition"]
    before = len(work)
    work = work.dropna(subset=feature_cols).reset_index(drop=True)
    log.info("Featurization: %d → %d rows (%d feature columns).",
             before, len(work), len(feature_cols))

    # Composition column isn't parquet-friendly — store its string form alongside.
    out = work.copy()
    out["composition_str"] = out["composition"].apply(lambda c: c.reduced_formula)
    _save_parquet(out.drop(columns=["composition"]))

    return work


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from step1_load import load_dataset
    df = load_dataset()
    out = featurize(df)
    print(out.shape)
    print(out.columns[:10].tolist(), "...")
