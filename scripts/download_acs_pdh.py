"""ACS Materials Letters PDH catalyst dataset — instructions.

This project's `--preset acs_pdh` expects the CSV at
`data/pdh_ACSMaterialsLetters.csv`.

**The source paper is NOT open-access**, so this project cannot legally
re-distribute the dataset. If you have institutional access to the paper,
download the SI table manually and place it at the target path shown below.
If you do not, skip the ACS example — the synthetic-catalyst and
matbench_steels examples in the README work without external data.

Citation (please cite the paper if you use the dataset):

    Oh, J.; Lee, J.; Park, J.; Jeon, N.; Na, G. S.; Chang, H.; Huh, J.;
    Kim, H. W.; Yun, Y.
    Data-Driven Development of Heterogeneous Catalysts for Propane
    Dehydrogenation with Machine Learning and Metaheuristic Optimization.
    ACS Materials Letters 2024, 6 (11), 5138–5145.
    doi: 10.1021/acsmaterialslett.4c01367
    https://doi.org/10.1021/acsmaterialslett.4c01367
"""
from __future__ import annotations

from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "data" / "pdh_ACSMaterialsLetters.csv"

INSTRUCTIONS = """\
ACS Materials Letters PDH dataset — manual download required
=============================================================

Status: {status}
Target: {dest}

The dataset comes from:

    Oh, J.; Lee, J.; Park, J.; Jeon, N.; Na, G. S.; Chang, H.; Huh, J.;
    Kim, H. W.; Yun, Y.
    Data-Driven Development of Heterogeneous Catalysts for Propane
    Dehydrogenation with Machine Learning and Metaheuristic Optimization.
    ACS Materials Letters 2024, 6 (11), 5138–5145.
    doi: 10.1021/acsmaterialslett.4c01367

The paper is NOT open-access; if you have institutional access, download
the supporting-information table (210 catalyst rows, 19-element atomic-
fraction composition + reaction conditions + propylene yield + deactivation
rate constant + score), convert it to CSV if it is an XLSX, and save it as:

    {dest}

After placing the file, verify the download with:

    python -c "import pandas as pd; \\
        print(pd.read_csv('{dest}').shape)"

Expected shape: (210, ~30) columns (varies slightly depending on which
sheet of the SI table you exported).
"""


def main() -> None:
    if DEST.exists():
        status = f"already present ({DEST.stat().st_size} bytes)"
    else:
        status = "MISSING — see instructions below"
    print(INSTRUCTIONS.format(dest=DEST, status=status))


if __name__ == "__main__":
    main()
