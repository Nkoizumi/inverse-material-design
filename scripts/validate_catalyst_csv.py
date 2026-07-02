"""Sanity-check a catalyst CSV against the lookup tables.

Usage:
    python scripts/validate_catalyst_csv.py data/pdh_template.csv

Reports rows whose `active_metal`, `promoter_1`, `promoter_2`, or `support`
names won't match the canonical keys in metal_properties.csv /
support_properties.csv.

Resolution rules (in order):
  1. Exact match in the lookup → OK.
  2. Mixture syntax with '+'    → split on '+' and ':', each component must
                                  resolve via rules 1 or 3 (informational).
  3. Parseable pymatgen Composition with at least one element in the metal
                                  lookup → OK as partial-feature (warning).
  4. Otherwise                   → error.

Blank support is a warning (treated as "bulk active phase" by the featurizer
with support_present=0), not an error. Exit status is non-zero only when
genuine unresolvable names remain.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METAL_LOOKUP = ROOT / "data" / "lookups" / "metal_properties.csv"
SUPPORT_LOOKUP = ROOT / "data" / "lookups" / "support_properties.csv"

METAL_ROLES = ("active_metal", "promoter_1", "promoter_2")
SUPPORT_ROLES = ("support",)


def _split_mixture(s: str) -> list[str] | None:
    if "+" not in s:
        return None
    parts = []
    for part in s.split("+"):
        k = part.rsplit(":", 1)[0] if ":" in part else part
        parts.append(k.strip())
    return parts


def _pymatgen_elements(s: str) -> set[str] | None:
    try:
        from pymatgen.core.composition import Composition
        return {str(el) for el in Composition(s).elements}
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8", encoding_errors="replace")
    metals = set(pd.read_csv(METAL_LOOKUP)["element"].astype(str))
    supports = set(pd.read_csv(SUPPORT_LOOKUP)["support"].astype(str))

    def _blank(v) -> bool:
        s = str(v).strip()
        return s == "" or s.lower() in {"nan", "none"}

    keep = ~(df.get("active_metal", "").apply(_blank)
             & df.get("support", "").apply(_blank))
    n_skipped = int((~keep).sum())
    df = df[keep].reset_index()

    errors: list[str] = []
    warnings: list[str] = []

    def _check(role: str, valid: set[str], kind: str) -> None:
        if role not in df.columns:
            return
        for _, r in df.iterrows():
            s = str(r[role]).strip()
            row_no = int(r["index"]) + 2
            if _blank(s):
                if role == "support":
                    warnings.append(f"row {row_no} support='' (treated as bulk catalyst, presence=0)")
                continue
            if s in valid:
                continue
            mixture = _split_mixture(s)
            if mixture is not None:
                missing = [m for m in mixture if m not in valid]
                if not missing:
                    continue
                # Mixture with some unknown components: fall through to pymatgen for those.
                elems = _pymatgen_elements(s) if kind == "metal" else None
                if elems and any(e in valid for e in elems):
                    warnings.append(f"row {row_no} {role}={s!r} (mixture; partial-feature via pymatgen)")
                    continue
                near_hint = ""
                errors.append(f"row {row_no} {role}={s!r} (unknown components: {missing}){near_hint}")
                continue
            if kind == "metal":
                elems = _pymatgen_elements(s)
                if elems and any(e in valid for e in elems):
                    warnings.append(f"row {row_no} {role}={s!r} (oxide/composition; partial-feature via pymatgen)")
                    continue
            near = [k for k in valid if k.lower() == s.lower()]
            hint = f" (try '{near[0]}'?)" if near else ""
            errors.append(f"row {row_no} {role}={s!r}{hint}")

    for r in METAL_ROLES:
        _check(r, metals, kind="metal")
    for r in SUPPORT_ROLES:
        _check(r, supports, kind="support")

    if warnings:
        print(f"⚠ {args.csv.name}: {len(warnings)} informational warning(s):")
        for line in warnings:
            print(f"  - {line}")
        print()

    if not errors:
        print(f"✓ {args.csv.name}: {len(df)} populated rows, all role names resolvable "
              f"(metals={len(metals)}, supports={len(supports)}; "
              f"skipped {n_skipped} placeholder rows).")
        return 0

    print(f"✗ {args.csv.name}: {len(errors)} unresolved role(s):", file=sys.stderr)
    for line in errors:
        print(f"  - {line}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"Known supports: {sorted(supports)}", file=sys.stderr)
    print(f"Known metals : {sorted(metals)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
