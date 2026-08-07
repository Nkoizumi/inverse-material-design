"""End-to-end driver for the inverse material design pipeline.

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip eda report      # run everything except EDA and report
    python run_pipeline.py --preset matbench_steels --skip eda report
    python run_pipeline.py --preset acs_pdh --skip eda report
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make pipeline/ importable as flat modules.
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent))


PRESETS = {
    # Bundled synthetic catalyst example (n=72, role-based schema). Full 6-step
    # pipeline + webui work end-to-end. Dual-target maximize.
    # Library trimmed to a fast-demo size: 4 metals × 4 promoters × 4 promoters
    # × 4 supports × 4×2×2 loadings ≈ 8k catalysts (vs the default 100k), so
    # the whole pipeline runs in ~1 min for the README quick-start.
    #
    # NOTE: featurization is no longer the reason to trim. Since the
    # featurizers resolve distinct cell values once, the full 101,816-row
    # library featurizes in ~6 s (was ~11 min). What now dominates a
    # full-library run is the discrete acquisition — optimize_acqf_discrete
    # over 100k choices with q = BO_BATCH_SIZE × BO_UNIQUE_OVERSAMPLE takes
    # ~4-5 min. Widen the library freely if you can spend that; installing
    # `ninja` lets BoTorch compile its fused qLogEHVI kernel for a further
    # ~3× on the multi-objective path.
    "synthetic_catalyst": {
        "DATASET_SOURCE": "csv",
        "CSV_PATH": Path(__file__).parent / "data" / "synthetic_catalysts.csv",
        "CATALYST_MODE": True,
        "CATALYST_FRACTION_MODE": False,
        "TARGET_COLS": ["propane_TOF_log", "propane_selectivity"],
        "OPTIMIZATION_DIRECTIONS": ["max", "max"],
        "SURROGATE_KIND": "both",
        "LIBRARY_ACTIVE_METALS": ["Pt", "Pd", "Rh", "Ir"],
        "LIBRARY_PROMOTERS_1": ["", "Sn", "Re", "Ga"],
        "LIBRARY_PROMOTERS_2": ["", "K", "Cs", "La"],
    },
    # Matbench small composition-only regression — 312 steel alloys, target =
    # yield strength (MPa). Single-target maximize; GP-sized.
    "matbench_steels": {
        "DATASET_SOURCE": "matminer",
        "MATMINER_DATASET": "matbench_steels",
        "TARGET_COLS": ["yield strength"],
        "OPTIMIZATION_DIRECTIONS": ["max"],
        "CATALYST_MODE": False,
        "SURROGATE_KIND": "gp",
        # Steels discrete library — avoids continuous-BO mode collapse.
        "STEELS_LIBRARY": True,
        "STEELS_LIBRARY_SIZE": 10_000,
    },
    # ACS Materials Letters PDH dataset (n=210, atomic-fraction schema).
    # Targets: maximize propylene yield, minimize deactivation rate (log).
    # Signal expectations (from honest 5-fold CV, 2026-06-24 baseline):
    #   GP yield R²≈+0.53   GP deact_log R²≈+0.25
    # Webui does NOT work in fraction mode yet (Tab 4 KeyError, Step 6 report
    # writer expects role-shape candidates). CLI only. Run with:
    #   python run_pipeline.py --preset acs_pdh --skip eda report
    "acs_pdh": {
        "DATASET_SOURCE": "csv",
        "CSV_PATH": Path(__file__).parent / "data" / "pdh_ACSMaterialsLetters.csv",
        "CATALYST_MODE": False,
        "CATALYST_FRACTION_MODE": True,
        "TARGET_COLS": ["propylene_yield", "deactivation_rate_log"],
        "OPTIMIZATION_DIRECTIONS": ["max", "min"],
        "SURROGATE_KIND": "both",
    },
}


def _apply_preset(name: str) -> None:
    import config
    overrides = PRESETS[name]
    for k, v in overrides.items():
        setattr(config, k, v)
    logging.getLogger("run").info(
        "Applied preset '%s': %s", name,
        ", ".join(f"{k}={v}" for k, v in overrides.items()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["load", "featurize", "eda", "surrogate", "inverse", "report"],
        help="Steps to skip.",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Apply a built-in config preset (overrides config.py).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("run")

    if args.preset:
        _apply_preset(args.preset)

    # Fail on an inconsistent config here, not several minutes into a
    # featurization — and, more to the point, not silently. An
    # invalid schema-mode combination produces a completed run with
    # plausible-looking candidates from the wrong featurizer.
    import config
    from config_schema import check
    check(config)

    skip = set(args.skip)
    t0 = time.time()

    if "load" in skip:
        log.info("Skipping step 1 (load).")
        import pandas as pd
        import config
        df = pd.read_parquet(config.DATA_DIR / "raw.parquet")
    else:
        log.info("STEP 1: load")
        from step1_load import load_dataset
        df = load_dataset()

    if "featurize" in skip:
        log.info("Skipping step 2 (featurize).")
        import pandas as pd
        import config
        df = pd.read_parquet(config.DATA_DIR / "featurized.parquet")
    else:
        log.info("STEP 2: featurize")
        from step2_featurize import featurize
        df = featurize(df)

    if "eda" in skip:
        log.info("Skipping step 3 (eda).")
        eda_results = None
    else:
        log.info("STEP 3: auto-EDA")
        from step3_eda import run_eda
        eda_results = run_eda(df)

    if "surrogate" in skip:
        log.info("Skipping step 4 — cannot run inverse without surrogates. Aborting.")
        return
    log.info("STEP 4: surrogate")
    from step4_surrogate import fit_surrogates
    surrogates = fit_surrogates(df)

    if "inverse" in skip:
        log.info("Skipping step 5 (inverse).")
        import pandas as pd
        import config
        candidates = pd.read_parquet(config.DATA_DIR / "candidates.parquet")
    else:
        log.info("STEP 5: inverse design (BO/MOBO)")
        from step5_inverse import run_inverse
        candidates = run_inverse(df, surrogates)

    if "report" in skip:
        log.info("Skipping step 6 (report).")
    else:
        log.info("STEP 6: scientific report")
        from step6_report import generate_report
        out = generate_report(candidates, eda_results)
        log.info("Report at %s", out)

    log.info("Pipeline finished in %.1fs.", time.time() - t0)


if __name__ == "__main__":
    main()
