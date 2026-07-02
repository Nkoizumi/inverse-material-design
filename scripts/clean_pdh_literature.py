"""One-shot cleanup of data/pdh_literature.csv.

Renames legacy/free-text role values to canonical lookup keys, fixes the
``MgAl2O5`` typo, renames target/feature columns to what config.py expects,
adds ``propane_TOF_log = log10(propane_TOF_s)``, converts
``propane_selectivity`` from percent to 0-1 fraction, and replaces the ``-|``
mojibake in reference strings. The raw input is preserved as
``data/pdh_literature_raw.csv`` on first run.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "pdh_literature.csv"
BACKUP = ROOT / "data" / "pdh_literature_raw.csv"

SUPPORT_RENAME = {
    "MgAl2O5": "MgAl2O4",
    "Magnesium-aluminum oxide": "MgAl2O4",
    "Al2O3": "gamma-Al2O3",
    "ZSM-5": "H-ZSM5",
    "TiO2-Al2O3": "TiO2-anatase:0.5+gamma-Al2O3:0.5",
}

COLUMN_RENAME = {
    "WHSV h-1": "WHSV_h",
    "specific_activity_s-1": "propane_TOF_s",
    "deactivation_rate_h-1": "deactivation_rate_h",
}


def main() -> None:
    if not BACKUP.exists():
        shutil.copy(SRC, BACKUP)
        print(f"Backed up raw → {BACKUP.name}")

    df = pd.read_csv(BACKUP, encoding="utf-8", encoding_errors="replace")
    n_in = len(df)

    df["support"] = (
        df["support"].astype(str).replace(SUPPORT_RENAME).replace({"nan": "", "None": ""})
    )

    df = df.rename(columns=COLUMN_RENAME)

    tof = pd.to_numeric(df["propane_TOF_s"], errors="coerce")
    df["propane_TOF_s"] = tof
    df["propane_TOF_log"] = np.log10(tof.where(tof > 0))

    sel = pd.to_numeric(df["propane_selectivity"], errors="coerce")
    df["propane_selectivity"] = sel / 100.0

    df["deactivation_rate_h"] = pd.to_numeric(df["deactivation_rate_h"], errors="coerce")

    for col in ("reference_paper", "reference_doi", "notes"):
        if col in df.columns:
            s = df[col].astype(str)
            s = s.str.replace("�|", "-", regex=False)
            s = s.str.replace("?20", ", 20", regex=False)
            df[col] = s.replace({"nan": ""})

    df.to_csv(SRC, index=False)

    n_supports = df["support"].replace("", np.nan).nunique(dropna=True)
    n_targets_present = {
        c: int(df[c].notna().sum())
        for c in ("propane_TOF_log", "propane_selectivity", "deactivation_rate_h")
    }
    print(f"Wrote {SRC.name}: {n_in} rows, {n_supports} unique supports.")
    print(f"Target coverage: {n_targets_present}")


if __name__ == "__main__":
    main()
