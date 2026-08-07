"""Fast end-to-end check that steps 1 → 5 still wire together. Used by CI.

Why this exists instead of just running the README quick-start
--------------------------------------------------------------
`run_pipeline.py --preset synthetic_catalyst` runs MOBO over an ~8k-row
library at q = BO_BATCH_SIZE × BO_UNIQUE_OVERSAMPLE = 80. On a workstation
that is GPU-assisted (`step5_inverse.DEVICE` picks CUDA when it is available)
and finishes in about two minutes. On a CPU-only CI runner the same command
spent over twelve minutes in step 5 alone on a 24-core machine without
finishing — a two-core hosted runner is worse still.

The integration signal we actually want per-PR is "does every stage still hand
off to the next, and does the surrogate discriminate between candidates". That
does not need the full search space. This trims the library and the batch to
the smallest size that still exercises:

  * step 1 load + log-derived targets
  * step 2 role-based featurization (matminer + lookups + interfacial)
  * step 4 GP *and* BNN fits, plus the k-fold CV parity path
  * step 5 discrete MOBO, identity dedup, and the BNN cross-check
    (which scores the driver's own tensor — see step5_inverse)

The full README command still runs in CI, but on the `quickstart` job, which
is manual (`workflow_dispatch`) precisely because of the runtime above.

Run it yourself with:  python scripts/ci_smoke.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("ci_smoke")

    import config
    from run_pipeline import _apply_preset

    # Reuse the preset rather than restating its values, so this cannot drift
    # away from what the README documents.
    _apply_preset("synthetic_catalyst")

    # Trim the search space. Keep BOTH surrogates and CV on: the BNN
    # cross-check and the per-fold feature selection are exactly the kind of
    # cross-stage wiring a smoke test should cover.
    config.LIBRARY_ACTIVE_METALS = ["Pt", "Pd"]
    config.LIBRARY_PROMOTERS_1 = ["", "Sn"]
    config.LIBRARY_PROMOTERS_2 = [""]
    config.LIBRARY_METAL_LOADINGS = [1.0, 5.0]
    config.LIBRARY_PROMO_LOADINGS = [0.3]
    config.BO_BATCH_SIZE = 3
    config.BO_UNIQUE_OVERSAMPLE = 2
    config.CV_FOLDS = 2
    config.BNN_TRAINING_ITERS = 200

    from config_schema import check
    check(config)

    from step1_load import load_dataset
    from step2_featurize import featurize
    from step4_surrogate import fit_surrogates
    from step5_inverse import run_inverse

    t0 = time.time()
    df = load_dataset()
    df = featurize(df)
    surrogates = fit_surrogates(df)
    candidates = run_inverse(df, surrogates)
    elapsed = time.time() - t0

    # ── Assertions ───────────────────────────────────────────────────────────
    if candidates is None or candidates.empty:
        log.error("step 5 returned no candidates.")
        return 1

    preds = [c for c in candidates.columns
             if c.startswith("pred_") and not c.endswith(("_sd", "_bnn", "_bnn_sd"))]
    if not preds:
        log.error("candidates carry no prediction columns: %s",
                  list(candidates.columns))
        return 1

    for col in preds:
        # A collapsed GP posterior returns the prior mean for every library
        # row. It raises nothing — the pipeline reports success and emits N
        # identical predictions — so it has to be asserted explicitly. This has
        # been a real failure mode here (it is why prepare_xy drops
        # constant-in-training columns and why step 5 median-fills the
        # library's reaction conditions).
        if candidates[col].round(9).nunique() == 1:
            log.error("every candidate shares one predicted %s (%s) — the "
                      "surrogate posterior collapsed.",
                      col, candidates[col].iloc[0])
            return 1

    # The BNN cross-check must reach the candidate table in role-based mode.
    bnn_cols = [c for c in candidates.columns if c.endswith("_bnn")]
    if not bnn_cols:
        log.error("no BNN cross-check columns; expected them with "
                  "SURROGATE_KIND=%r.", config.SURROGATE_KIND)
        return 1

    parity = config.DATA_DIR / "cv_parity_predictions.parquet"
    if not parity.exists():
        log.error("CV_FOLDS=%s but %s was not written.",
                  config.CV_FOLDS, parity.name)
        return 1

    log.info("OK — %d candidates in %.1fs; predictions %s; cross-check %s.",
             len(candidates), elapsed, preds, bnn_cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
