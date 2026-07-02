"""Featurization for heterogeneous catalysts (active metal + promoter + support).

Supports alloys (e.g. ``Pt0.9Sn0.1``) and mixed oxides (e.g. ``CeO2:0.7+ZrO2:0.3``).

Three feature blocks, each guarded by column availability:

  1. Per-role matminer block — ElementProperty (Magpie) + Stoichiometry applied
     once per role column, prefixed (e.g. ``metal_MagpieData mean Electronegativity``).
  2. Support / metal physical-property lookup — surface energy, work function,
     NH3/CO2-TPD acidity/basicity, lattice constant, electronegativity, etc.
     Weighted-averaged across components for alloys and mixed oxides.
  3. Interfacial descriptors — Δχ_Pauling, Δ work function, lattice mismatch
     (only when active_metal AND support are both present).

Mixed-component syntax (any role column):
    ``Pt``                       — single key (matches lookup directly)
    ``Pt0.9Sn0.1``               — alloy parsed by pymatgen
    ``CeO2+ZrO2``                — equal-weight mixed oxide
    ``CeO2:0.7+ZrO2:0.3``        — weighted mixed oxide
    ``gamma-Al2O3:0.5+SiO2:0.5`` — named-key mixture using lookup phase prefixes
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Mixture parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_components(cell: str, lookup_keys: set[str] | None = None) -> dict[str, float]:
    """Parse a cell into ``{lookup_key: normalized_weight}``.

    Resolution order:
      1. Exact match in lookup_keys      → ``{cell: 1.0}``
      2. Mixture syntax with '+'         → split into named components
      3. pymatgen Composition (alloys)   → element-wise fractional composition
      4. Fail                            → ``{}``
    """
    s = str(cell).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return {}

    lookup_keys = lookup_keys or set()

    if s in lookup_keys:
        return {s: 1.0}

    if "+" in s:
        comps = {}
        for part in s.split("+"):
            if ":" in part:
                k, w = part.rsplit(":", 1)
                try:
                    comps[k.strip()] = float(w)
                except ValueError:
                    continue
            else:
                comps[part.strip()] = 1.0
        total = sum(comps.values())
        if total > 0:
            return {k: v / total for k, v in comps.items()}
        return {}

    try:
        from pymatgen.core.composition import Composition
        comp = Composition(s)
        return {str(el): float(frac) for el, frac in comp.fractional_composition.items()}
    except Exception:
        return {}


def to_composition(cell: str, lookup_df: pd.DataFrame | None = None,
                   key_col: str = "support",
                   formula_col: str = "pymatgen_formula"):
    """Convert a cell to a pymatgen Composition, supporting mixed-oxide syntax
    and named lookup keys (e.g. ``gamma-Al2O3``).
    """
    from pymatgen.core.composition import Composition

    s = str(cell).strip()
    if not s:
        return None

    lookup_map = {}
    if lookup_df is not None and formula_col in lookup_df.columns:
        lookup_map = dict(zip(lookup_df[key_col].astype(str), lookup_df[formula_col].astype(str)))

    def _resolve(k: str) -> Composition | None:
        if k in lookup_map:
            try:
                return Composition(lookup_map[k])
            except Exception:
                return None
        try:
            return Composition(k)
        except Exception:
            return None

    if "+" in s:
        combined = Composition({})
        for part in s.split("+"):
            if ":" in part:
                k, w = part.rsplit(":", 1)
                try:
                    weight = float(w)
                except ValueError:
                    continue
            else:
                k, weight = part, 1.0
            comp = _resolve(k.strip())
            if comp is None:
                continue
            for el, amt in comp.items():
                combined += Composition({el: amt * weight})
        return combined if len(combined) > 0 else None

    return _resolve(s)


def weighted_lookup(components: dict[str, float], lookup_df: pd.DataFrame,
                    key_col: str, numeric_cols: list[str],
                    cat_cols: list[str]) -> dict:
    """Weighted average of numeric columns and major-component vote for categoricals."""
    if not components:
        return {}

    feats = {}
    for nc in numeric_cols:
        val, total_w = 0.0, 0.0
        for k, w in components.items():
            row = lookup_df[lookup_df[key_col].astype(str) == k]
            if len(row) == 1 and not pd.isna(row[nc].iloc[0]):
                val += w * float(row[nc].iloc[0])
                total_w += w
        if total_w > 0:
            feats[nc] = val / total_w
        else:
            feats[nc] = np.nan

    for cc in cat_cols:
        major = max(components.items(), key=lambda kv: kv[1])[0]
        row = lookup_df[lookup_df[key_col].astype(str) == major]
        if len(row) == 1:
            feats[cc] = row[cc].iloc[0]
        else:
            feats[cc] = np.nan

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Featurizer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CatalystFeaturizer:
    roles: dict[str, str] = field(default_factory=dict)
    loadings: dict[str, str] = field(default_factory=dict)
    support_lookup_path: Path | None = None
    metal_lookup_path: Path | None = None
    matminer_preset: str = "magpie"
    optional_numeric_features: list[str] = field(default_factory=list)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)

        sup_lookup = pd.read_csv(self.support_lookup_path) if (
            self.support_lookup_path and Path(self.support_lookup_path).exists()
        ) else None
        metal_lookup = pd.read_csv(self.metal_lookup_path) if (
            self.metal_lookup_path and Path(self.metal_lookup_path).exists()
        ) else None

        # ── 1. Matminer per role ────────────────────────────────────────
        for role, col in self.roles.items():
            if col not in out.columns or out[col].isna().all():
                continue
            lookup_for_role = sup_lookup if role == "support" else None
            out = self._featurize_role(out, col, prefix=role, lookup=lookup_for_role)

        # ── 2. Physical-property lookup per role ────────────────────────
        # Support → support_lookup. active_metal / promoter_* / dopant_* → metal_lookup.
        for role, col in self.roles.items():
            if col not in out.columns:
                continue
            kind = self._lookup_kind_for_role(role)
            prefix = self._prefix_for_role(role)
            if kind == "support" and sup_lookup is not None:
                out = self._join_lookup_weighted(
                    out, col, sup_lookup, key_col="support", prefix=prefix,
                    skip_cols={"pymatgen_formula"},
                )
            elif kind == "metal" and metal_lookup is not None:
                out = self._join_lookup_weighted(
                    out, col, metal_lookup, key_col="element", prefix=prefix,
                )

        # ── 3. Alloy-level effective adsorbate binding energies ─────────
        # Atomic-fraction-weighted average of monometallic E_ads_* across
        # active_metal + promoter_1 + promoter_2 (option-1 alloy treatment:
        # linear combination of OC20-style monometallic values).
        if metal_lookup is not None:
            out = self._add_alloy_effective_features(out, metal_lookup)

        # Mixed-oxide indicator from the support cell.
        if "support" in self.roles and self.roles["support"] in out.columns:
            out = self._add_mixed_oxide_indicator(out, self.roles["support"], sup_lookup)

        # ── 4. Interfacial descriptors (active_metal × support) ─────────
        if {"metal_phys_lattice_a_A", "support_phys_lattice_a_A"}.issubset(out.columns):
            out = self._add_interfacial(out)

        # ── 5. Loading columns passed through ───────────────────────────
        for role, col in self.loadings.items():
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

        # ── 6. Optional measured features (dispersion, BET, …) ──────────
        for col in self.optional_numeric_features:
            if col not in out.columns:
                continue
            numeric = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_present"] = numeric.notna().astype(int)
            out[col] = numeric        # NaN preserved; imputed in prepare_xy
            log.info("Optional feature '%s' present in %d/%d rows.",
                     col, int(numeric.notna().sum()), len(out))

        return out

    @staticmethod
    def _lookup_kind_for_role(role: str) -> str | None:
        """Map role name to lookup kind: 'support', 'metal', or None."""
        if role == "support":
            return "support"
        if role == "active_metal" or role.startswith("promoter") or role.startswith("dopant"):
            return "metal"
        return None

    @staticmethod
    def _prefix_for_role(role: str) -> str:
        """Prefix for lookup-derived columns. Keeps active_metal as 'metal_phys' for
        interfacial-descriptor backwards compatibility."""
        if role == "active_metal":
            return "metal_phys"
        return f"{role}_phys"

    # ──────────────────────────────────────────────────────────────────────
    def _featurize_role(self, df: pd.DataFrame, col: str, prefix: str,
                        lookup: pd.DataFrame | None) -> pd.DataFrame:
        from matminer.featurizers.composition import ElementProperty, Stoichiometry

        log.info("Matminer featurizing role '%s' (col='%s') …", prefix, col)

        compositions = df[col].apply(lambda s: to_composition(s, lookup_df=lookup))
        mask = compositions.notna()

        # Presence indicator (0 = role absent for this row, 1 = present).
        df = df.copy()
        df[f"{prefix}_present"] = mask.astype(int).values

        if not mask.any():
            log.warning("No parseable compositions in '%s' — only presence flag added.", col)
            return df

        temp = pd.DataFrame({"composition": compositions[mask].values}, index=df.index[mask])

        featurizers = [
            ElementProperty.from_preset(self.matminer_preset),
            Stoichiometry(),
        ]
        for fz in featurizers:
            temp = fz.featurize_dataframe(temp, "composition", ignore_errors=True)

        new_cols = [c for c in temp.columns if c != "composition"]
        rename = {c: f"{prefix}_{c}" for c in new_cols}
        temp = temp.rename(columns=rename)[list(rename.values())]
        return df.join(temp, how="left")

    def _join_lookup_weighted(self, df: pd.DataFrame, col: str,
                              lookup: pd.DataFrame, key_col: str, prefix: str,
                              skip_cols: set | None = None) -> pd.DataFrame:
        skip_cols = skip_cols or set()
        if key_col not in lookup.columns:
            raise KeyError(f"Lookup missing key column '{key_col}'.")

        numeric_cols = [c for c in lookup.columns
                        if c != key_col and c not in skip_cols
                        and pd.api.types.is_numeric_dtype(lookup[c])]
        cat_cols = [c for c in lookup.columns
                    if c != key_col and c not in skip_cols
                    and not pd.api.types.is_numeric_dtype(lookup[c])]
        lookup_keys = set(lookup[key_col].astype(str))

        feat_rows = []
        for cell in df[col].astype(str):
            components = parse_components(cell, lookup_keys)
            feats = weighted_lookup(components, lookup, key_col, numeric_cols, cat_cols)
            feat_rows.append(feats)

        feat_df = pd.DataFrame(feat_rows, index=df.index)
        feat_df = feat_df.rename(columns={c: f"{prefix}_{c}" for c in feat_df.columns})

        # One-hot the categoricals.
        for cc in cat_cols:
            target = f"{prefix}_{cc}"
            if target in feat_df.columns:
                dummies = pd.get_dummies(feat_df[target], prefix=target, dummy_na=False)
                feat_df = pd.concat([feat_df.drop(columns=[target]), dummies], axis=1)

        log.info("Joined %s lookup (%d numeric + %d categorical features, weighted).",
                 prefix, len(numeric_cols), len(cat_cols))
        return df.join(feat_df, how="left")

    def _add_alloy_effective_features(self, df: pd.DataFrame,
                                       metal_lookup: pd.DataFrame) -> pd.DataFrame:
        from pymatgen.core.periodic_table import Element

        ads_cols = [c for c in metal_lookup.columns if c.startswith("E_ads_")]
        if not ads_cols or "element" not in metal_lookup.columns:
            return df

        lookup: dict[str, dict] = {}
        for _, row in metal_lookup.iterrows():
            el = str(row["element"]).strip()
            if not el:
                continue
            try:
                mass = float(Element(el).atomic_mass)
            except Exception:
                continue
            entry = {"mass": mass}
            for c in ads_cols:
                v = row[c]
                entry[c] = float(v) if not pd.isna(v) else np.nan
            lookup[el] = entry

        role_cols: list[tuple[str, str]] = []
        for role, ecol in self.roles.items():
            if role == "support" or role not in self.loadings:
                continue
            lcol = self.loadings[role]
            if ecol in df.columns and lcol in df.columns:
                role_cols.append((ecol, lcol))

        if not role_cols:
            return df

        out_cols: dict[str, list] = {f"alloy_{c}": [] for c in ads_cols}
        out_cols["alloy_dblock_frac"] = []
        out_cols["is_alloy"] = []

        for _, row in df.iterrows():
            moles: dict[str, float] = {}
            for ecol, lcol in role_cols:
                el = str(row[ecol]).strip()
                if not el or el.lower() in ("nan", "none"):
                    continue
                if el not in lookup:
                    continue
                loading = pd.to_numeric(row[lcol], errors="coerce")
                if pd.isna(loading) or loading <= 0:
                    continue
                moles[el] = float(loading) / lookup[el]["mass"]

            total = sum(moles.values())
            if total == 0:
                for c in ads_cols:
                    out_cols[f"alloy_{c}"].append(np.nan)
                out_cols["alloy_dblock_frac"].append(np.nan)
                out_cols["is_alloy"].append(0)
                continue

            x = {el: m / total for el, m in moles.items()}
            out_cols["is_alloy"].append(int(len(x) > 1))

            for c in ads_cols:
                num = 0.0
                w_total = 0.0
                for el, frac in x.items():
                    v = lookup[el][c]
                    if not np.isnan(v):
                        num += frac * v
                        w_total += frac
                out_cols[f"alloy_{c}"].append(num / w_total if w_total > 0 else np.nan)

            # d-block fraction: use E_ads_C as the canonical "is d-block" probe
            probe = "E_ads_C_eV" if "E_ads_C_eV" in ads_cols else ads_cols[0]
            dblock_x = sum(frac for el, frac in x.items() if not np.isnan(lookup[el][probe]))
            out_cols["alloy_dblock_frac"].append(dblock_x)

        log.info("Added alloy-effective features (%d binding cols + dblock_frac + is_alloy).",
                 len(ads_cols))
        return df.join(pd.DataFrame(out_cols, index=df.index), how="left")

    def _add_mixed_oxide_indicator(self, df: pd.DataFrame, support_col: str,
                                    sup_lookup: pd.DataFrame | None) -> pd.DataFrame:
        lookup_keys = set(sup_lookup["support"].astype(str)) if (
            sup_lookup is not None and "support" in sup_lookup.columns
        ) else set()
        df = df.copy()
        df["is_mixed_oxide"] = df[support_col].apply(
            lambda s: int(len(parse_components(s, lookup_keys)) > 1)
        )
        return df

    def _add_interfacial(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Computing interfacial descriptors …")

        a_m = df["metal_phys_lattice_a_A"]
        a_s = df["support_phys_lattice_a_A"]
        df["intf_lattice_mismatch"] = (a_m - a_s).abs() / a_s

        if {"metal_phys_work_function_eV", "support_phys_work_function_eV"}.issubset(df.columns):
            df["intf_delta_work_function_eV"] = (
                df["metal_phys_work_function_eV"] - df["support_phys_work_function_eV"]
            )

        if {"metal_phys_pauling_chi", "support_phys_sanderson_chi"}.issubset(df.columns):
            df["intf_delta_chi"] = (
                df["metal_phys_pauling_chi"] - df["support_phys_sanderson_chi"]
            )

        return df
