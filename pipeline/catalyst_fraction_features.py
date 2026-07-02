"""Featurization for atomic-fraction catalyst CSVs.

Companion to `catalyst_features.py`. That module handles role-based schemas
(active_metal / promoter_1 / promoter_2 / support + loading columns). THIS
module handles datasets where each catalyst is described as a flat atomic-
fraction vector over a fixed element panel (e.g. the ACS Materials Letters
PDH dataset: 19 element columns summing to ~1, plus reaction conditions).

The strategy is deliberately NOT to guess role labels (which element is the
"active metal"? which is the "promoter"?). Instead:

  1. Pick the dominant cation from {Al, Si, Zr} as the support — these are
     the only support-formers in PDH literature, and they're typically >0.95
     of the total atom count when present. Reconstruct the oxide name
     (Al→gamma-Al2O3, Si→SiO2, Zr→ZrO2) and look up its physical properties.
  2. The remaining elements form the "metal phase". Build a pymatgen
     Composition from them (renormalized to sum to 1) and apply Magpie +
     Stoichiometry. Composition-weighted-lookup against metal_properties.csv
     gives effective work function, d-band center, E_ads_*, etc.
  3. Interfacial descriptors (lattice mismatch, Δχ, Δ work function) are
     emitted when both blocks yield a lattice constant.

This gives roughly the same feature blocks as the role-based featurizer
without per-role decomposition. The non-support element columns are kept
as raw features too — the GP doesn't have to derive them back from Magpie.

Inputs (post step1_load normalization):
  * config.CATALYST_FRACTION_ELEMENTS — element column names (atomic fractions)
  * config.CATALYST_FRACTION_SUPPORT_CATIONS — which elements can be supports
  * config.CATALYST_FRACTION_SUPPORT_OXIDE_MAP — element → support lookup key
  * config.CATALYST_FRACTION_CONDITIONS — reaction-condition feature columns
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from catalyst_features import weighted_lookup

log = logging.getLogger(__name__)

_MIN_SUPPORT_FRACTION = 0.30   # below this, we don't assign a support


def featurize_fractions(df: pd.DataFrame, *,
                        element_cols: list[str],
                        support_cations: list[str],
                        support_oxide_map: dict[str, str],
                        condition_cols: list[str],
                        support_lookup_path: Path,
                        metal_lookup_path: Path) -> pd.DataFrame:
    """Add Magpie / lookup / interfacial feature blocks to an atomic-fraction df.

    Returns a NEW DataFrame containing the original columns plus the feature
    blocks. The original element columns are kept as raw stoichiometry
    features.
    """
    out = df.copy().reset_index(drop=True)

    present_elements = [e for e in element_cols if e in out.columns]
    if not present_elements:
        raise KeyError(
            f"None of the configured element columns {element_cols} are "
            f"present in the df. Columns: {out.columns.tolist()[:20]} …"
        )
    log.info("Atomic-fraction featurization: %d rows, element panel = %s",
             len(out), present_elements)

    # ── 1. Determine the support cation for each row ──────────────────────
    sup_cation_col = "_support_cation"
    sup_oxide_col = "support_oxide"
    sup_fraction_col = "support_fraction"
    cation_block = out[[c for c in support_cations if c in out.columns]]
    if cation_block.empty:
        out[sup_cation_col] = pd.NA
        out[sup_fraction_col] = 0.0
    else:
        out[sup_fraction_col] = cation_block.max(axis=1)
        out[sup_cation_col] = cation_block.idxmax(axis=1)
        # Below the cutoff (e.g. metal-rich rows with no oxide), null it out.
        mask = out[sup_fraction_col] < _MIN_SUPPORT_FRACTION
        if mask.any():
            log.info("  %d/%d rows have no dominant support cation "
                     "(max support fraction < %.2f).",
                     int(mask.sum()), len(out), _MIN_SUPPORT_FRACTION)
            out.loc[mask, sup_cation_col] = pd.NA
    out[sup_oxide_col] = out[sup_cation_col].map(
        support_oxide_map, na_action="ignore"
    )
    log.info("  Support distribution: %s",
             out[sup_oxide_col].value_counts(dropna=False).to_dict())

    # ── 2. Metal-phase Composition + Magpie/Stoichiometry ─────────────────
    out = _add_metal_phase_magpie(out, present_elements, support_cations)

    # ── 3. Support lookup (using reconstructed oxide name) ────────────────
    sup_lookup = pd.read_csv(support_lookup_path) if support_lookup_path.exists() else None
    if sup_lookup is not None:
        out = _join_support_lookup(out, sup_oxide_col, sup_lookup, prefix="support_phys")

    # ── 4. Metal lookup (composition-weighted across non-support elements) ─
    metal_lookup = pd.read_csv(metal_lookup_path) if metal_lookup_path.exists() else None
    if metal_lookup is not None:
        out = _join_metal_lookup_weighted(out, present_elements, support_cations,
                                          metal_lookup, prefix="metal_phys")

    # ── 5. Interfacial descriptors ────────────────────────────────────────
    if {"metal_phys_lattice_a_A", "support_phys_lattice_a_A"}.issubset(out.columns):
        out["intf_lattice_mismatch"] = (
            (out["metal_phys_lattice_a_A"] - out["support_phys_lattice_a_A"]).abs()
            / out["support_phys_lattice_a_A"]
        )
        log.info("  Added intf_lattice_mismatch.")
    if {"metal_phys_work_function_eV", "support_phys_work_function_eV"}.issubset(out.columns):
        out["intf_delta_work_function_eV"] = (
            out["metal_phys_work_function_eV"] - out["support_phys_work_function_eV"]
        )
    if {"metal_phys_pauling_chi", "support_phys_sanderson_chi"}.issubset(out.columns):
        out["intf_delta_chi"] = (
            out["metal_phys_pauling_chi"] - out["support_phys_sanderson_chi"]
        )

    # ── 6. Reaction conditions pass through ───────────────────────────────
    present_conditions = [c for c in condition_cols if c in out.columns]
    if present_conditions:
        for c in present_conditions:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        log.info("  Reaction-condition features passed through: %s",
                 present_conditions)

    # ── 7. Cleanup: drop scratch column ───────────────────────────────────
    out = out.drop(columns=[sup_cation_col])
    return out


def _add_metal_phase_magpie(df: pd.DataFrame, element_cols: list[str],
                            support_cations: list[str]) -> pd.DataFrame:
    """Build a metal-phase Composition per row and Magpie/Stoichiometry-featurize."""
    from matminer.featurizers.composition import ElementProperty, Stoichiometry
    from pymatgen.core.composition import Composition

    metal_elements = [e for e in element_cols if e not in support_cations]
    if not metal_elements:
        log.warning("No metal-phase elements after removing support cations.")
        return df

    def _row_to_metal_composition(row) -> Composition | None:
        weights = {}
        for e in metal_elements:
            v = row.get(e)
            if pd.notna(v) and float(v) > 0:
                weights[e] = float(v)
        if not weights:
            return None
        return Composition(weights)

    df["metal_phase_composition"] = df.apply(_row_to_metal_composition, axis=1)

    has_metal = df["metal_phase_composition"].notna()
    n_no_metal = int((~has_metal).sum())
    if n_no_metal:
        log.info("  %d/%d rows have NO metal phase (pure support); their "
                 "Magpie/Stoichiometry features will be NaN.",
                 n_no_metal, len(df))

    sub = df.loc[has_metal, ["metal_phase_composition"]].copy()
    for featurizer in [ElementProperty.from_preset("magpie"), Stoichiometry()]:
        log.info("  Applying %s to metal-phase Composition …",
                 type(featurizer).__name__)
        sub = featurizer.featurize_dataframe(sub, "metal_phase_composition",
                                             ignore_errors=True)
    new_cols = [c for c in sub.columns if c != "metal_phase_composition"]
    sub = sub[new_cols].rename(columns={c: f"metal_phase_{c}" for c in new_cols})

    df = df.join(sub)
    df["metal_phase_present"] = has_metal.astype(float)
    df = df.drop(columns=["metal_phase_composition"])
    log.info("  Added %d metal-phase Magpie/Stoichiometry features.", len(new_cols))
    return df


def _join_support_lookup(df: pd.DataFrame, oxide_col: str,
                          lookup_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    key_col = "support"
    numeric_cols = [c for c in lookup_df.columns
                    if c != key_col and pd.api.types.is_numeric_dtype(lookup_df[c])]
    cat_cols = [c for c in lookup_df.columns
                if c != key_col and not pd.api.types.is_numeric_dtype(lookup_df[c])
                and c != "pymatgen_formula"]

    rows = []
    for oxide in df[oxide_col]:
        if pd.isna(oxide):
            rows.append({})
        else:
            rows.append(weighted_lookup({str(oxide): 1.0}, lookup_df, key_col,
                                         numeric_cols, cat_cols))
    sup_feats = pd.DataFrame(rows).add_prefix(f"{prefix}_")
    df = pd.concat([df.reset_index(drop=True),
                    sup_feats.reset_index(drop=True)], axis=1)
    n = sup_feats.shape[1]
    log.info("  Added %d support_phys lookup features.", n)
    return df


def _join_metal_lookup_weighted(df: pd.DataFrame, element_cols: list[str],
                                 support_cations: list[str],
                                 lookup_df: pd.DataFrame, prefix: str
                                 ) -> pd.DataFrame:
    """Composition-weighted average of metal lookup columns across the
    metal-phase elements (i.e. element_cols minus support_cations).

    Uses weighted_lookup() from catalyst_features so behavior matches the
    role-based path exactly: numeric → atom-fraction-weighted mean across
    components with non-NaN lookup values; categorical → major-component
    vote.
    """
    key_col = "element"
    numeric_cols = [c for c in lookup_df.columns
                    if c != key_col and pd.api.types.is_numeric_dtype(lookup_df[c])]
    cat_cols = [c for c in lookup_df.columns
                if c != key_col and not pd.api.types.is_numeric_dtype(lookup_df[c])]

    metal_elements = [e for e in element_cols if e not in support_cations]

    rows = []
    for _, row in df.iterrows():
        weights = {}
        for e in metal_elements:
            v = row.get(e)
            if pd.notna(v) and float(v) > 0:
                weights[e] = float(v)
        # Renormalize so the lookup gets a probability distribution.
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        rows.append(weighted_lookup(weights, lookup_df, key_col,
                                     numeric_cols, cat_cols))
    metal_feats = pd.DataFrame(rows).add_prefix(f"{prefix}_")
    df = pd.concat([df.reset_index(drop=True),
                    metal_feats.reset_index(drop=True)], axis=1)
    log.info("  Added %d metal_phys composition-weighted features.",
             metal_feats.shape[1])
    return df
