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


@dataclass(frozen=True)
class LookupIndex:
    """Pre-resolved property lookup, keyed by the lookup's key column.

    ``weighted_lookup`` used to re-scan the whole lookup DataFrame with a
    boolean mask for every (row × property column × component) triple. On the
    45-row metal table that is ~1.8 ms per catalyst row, so a 100k-row BO
    library spent minutes doing nothing but repeated equality scans over 45
    rows. Resolving each key once up front turns the inner loop into dict
    lookups.

    Semantics deliberately mirror the original scan:
      * a key appearing more than once in the lookup is treated as ABSENT
        (the old code required ``len(row) == 1``);
      * NaN numeric cells are omitted, so they contribute neither value nor
        weight to the weighted mean;
      * categorical values are taken verbatim, NaN included.
    """
    numeric: dict[str, dict[str, float]]
    categorical: dict[str, dict[str, object]]

    @classmethod
    def build(cls, lookup_df: pd.DataFrame, key_col: str,
              numeric_cols: list[str], cat_cols: list[str]) -> "LookupIndex":
        keys = lookup_df[key_col].astype(str)
        counts = keys.value_counts()
        unambiguous = set(counts[counts == 1].index)

        numeric: dict[str, dict[str, float]] = {}
        categorical: dict[str, dict[str, object]] = {}
        for pos, key in enumerate(keys):
            if key not in unambiguous:
                continue
            row = lookup_df.iloc[pos]
            numeric[key] = {c: float(row[c]) for c in numeric_cols
                            if not pd.isna(row[c])}
            categorical[key] = {c: row[c] for c in cat_cols}
        return cls(numeric=numeric, categorical=categorical)


def weighted_lookup(components: dict[str, float], lookup_df: pd.DataFrame,
                    key_col: str, numeric_cols: list[str],
                    cat_cols: list[str],
                    index: LookupIndex | None = None) -> dict:
    """Weighted average of numeric columns and major-component vote for categoricals.

    Pass a prebuilt ``index`` when calling this in a loop — building one per
    call defeats the point. Callers that only resolve a handful of rows (e.g.
    step6's per-candidate enrichment) can omit it.
    """
    if not components:
        return {}

    if index is None:
        index = LookupIndex.build(lookup_df, key_col, numeric_cols, cat_cols)

    feats: dict = {}
    for nc in numeric_cols:
        val, total_w = 0.0, 0.0
        for k, w in components.items():
            entry = index.numeric.get(k)
            if entry is not None and nc in entry:
                val += w * entry[nc]
                total_w += w
        feats[nc] = val / total_w if total_w > 0 else np.nan

    if cat_cols:
        # `max` on ties picks the first key in insertion order — same as the
        # original, which iterated the same `components` dict.
        major = max(components.items(), key=lambda kv: kv[1])[0]
        major_cats = index.categorical.get(major)
        for cc in cat_cols:
            feats[cc] = major_cats[cc] if major_cats is not None else np.nan

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

        # Featurize DISTINCT cell values only, then broadcast back. A BO
        # library is a Cartesian product, so a 100k-row library still has only
        # ~11 distinct active metals and ~26 distinct supports — Magpie was
        # being recomputed for each of the ~9k rows sharing a value.
        cells = df[col].astype(str)
        distinct = pd.Index(cells.unique())
        log.info("Matminer featurizing role '%s' (col='%s') — %d distinct of "
                 "%d rows.", prefix, col, len(distinct), len(df))

        comp_by_cell = {c: to_composition(c, lookup_df=lookup) for c in distinct}
        mask = cells.map(lambda c: comp_by_cell[c] is not None)

        # Presence indicator (0 = role absent for this row, 1 = present).
        df = df.copy()
        df[f"{prefix}_present"] = mask.astype(int).values

        if not mask.any():
            log.warning("No parseable compositions in '%s' — only presence flag added.", col)
            return df

        parseable = [c for c in distinct if comp_by_cell[c] is not None]
        temp = pd.DataFrame({"composition": [comp_by_cell[c] for c in parseable]},
                            index=pd.Index(parseable, name=None))

        featurizers = [
            ElementProperty.from_preset(self.matminer_preset),
            Stoichiometry(),
        ]
        for fz in featurizers:
            temp = fz.featurize_dataframe(temp, "composition", ignore_errors=True)

        new_cols = [c for c in temp.columns if c != "composition"]
        rename = {c: f"{prefix}_{c}" for c in new_cols}
        temp = temp.rename(columns=rename)[list(rename.values())]

        # Broadcast the per-distinct-value rows back out. Cells whose
        # composition failed to parse are absent from `temp`, so reindex gives
        # them NaN — the same result the old per-row left-join produced.
        expanded = temp.reindex(cells.values)
        expanded.index = df.index
        return df.join(expanded, how="left")

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

        # Resolve the lookup once, then resolve each DISTINCT cell once. The
        # old version re-parsed the cell with pymatgen and re-scanned the
        # lookup DataFrame for every row.
        index = LookupIndex.build(lookup, key_col, numeric_cols, cat_cols)
        cells = df[col].astype(str)
        feats_by_cell = {
            cell: weighted_lookup(parse_components(cell, lookup_keys), lookup,
                                  key_col, numeric_cols, cat_cols, index=index)
            for cell in cells.unique()
        }
        feat_rows = [feats_by_cell[cell] for cell in cells]

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

        # d-block fraction: use E_ads_C as the canonical "is d-block" probe
        probe = "E_ads_C_eV" if "E_ads_C_eV" in ads_cols else ads_cols[0]

        def _compute(cells: tuple[tuple[str, str], ...]) -> dict:
            """Alloy features for one (element, loading) combination."""
            moles: dict[str, float] = {}
            for el_raw, load_raw in cells:
                el = el_raw.strip()
                if not el or el.lower() in ("nan", "none"):
                    continue
                if el not in lookup:
                    continue
                loading = pd.to_numeric(load_raw, errors="coerce")
                if pd.isna(loading) or loading <= 0:
                    continue
                moles[el] = float(loading) / lookup[el]["mass"]

            total = sum(moles.values())
            if total == 0:
                res = {f"alloy_{c}": np.nan for c in ads_cols}
                res["alloy_dblock_frac"] = np.nan
                res["is_alloy"] = 0
                return res

            x = {el: m / total for el, m in moles.items()}
            res = {"is_alloy": int(len(x) > 1)}
            for c in ads_cols:
                num = 0.0
                w_total = 0.0
                for el, frac in x.items():
                    v = lookup[el][c]
                    if not np.isnan(v):
                        num += frac * v
                        w_total += frac
                res[f"alloy_{c}"] = num / w_total if w_total > 0 else np.nan
            res["alloy_dblock_frac"] = sum(
                frac for el, frac in x.items() if not np.isnan(lookup[el][probe])
            )
            return res

        # One computation per DISTINCT (element, loading) combination rather
        # than per row — the loading grids are small, so a 100k-row library has
        # only a few hundred distinct combinations. Keys are strings so that a
        # NaN loading still hashes (NaN != NaN would otherwise never cache).
        col_pairs = [(df[ecol].astype(str), df[lcol].astype(str))
                     for ecol, lcol in role_cols]
        row_keys = list(zip(*[tuple(zip(e, l)) for e, l in col_pairs]))

        cache: dict[tuple, dict] = {}
        rows = []
        for key in row_keys:
            res = cache.get(key)
            if res is None:
                res = cache[key] = _compute(key)
            rows.append(res)

        # Preserve the original column order: binding columns, then dblock, then
        # is_alloy. Building from a list of dicts would order by first-seen key.
        ordered = [f"alloy_{c}" for c in ads_cols] + ["alloy_dblock_frac", "is_alloy"]
        out_cols = {name: [r[name] for r in rows] for name in ordered}

        log.info("Added alloy-effective features (%d binding cols + dblock_frac + "
                 "is_alloy) from %d distinct combinations across %d rows.",
                 len(ads_cols), len(cache), len(df))
        return df.join(pd.DataFrame(out_cols, index=df.index), how="left")

    def _add_mixed_oxide_indicator(self, df: pd.DataFrame, support_col: str,
                                    sup_lookup: pd.DataFrame | None) -> pd.DataFrame:
        lookup_keys = set(sup_lookup["support"].astype(str)) if (
            sup_lookup is not None and "support" in sup_lookup.columns
        ) else set()
        df = df.copy()
        # parse_components goes through pymatgen for anything that isn't a bare
        # lookup key, so resolve each distinct support cell once.
        cells = df[support_col].astype(str)
        by_cell = {c: int(len(parse_components(c, lookup_keys)) > 1)
                   for c in cells.unique()}
        df["is_mixed_oxide"] = cells.map(by_cell).values
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
