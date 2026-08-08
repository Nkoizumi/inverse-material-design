"""Central configuration for the inverse material design pipeline."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
# CHECKPOINTS_DIR was removed along with step 4's write-only surrogate pickle:
# nothing read it, and creating the directory implied a resume capability the
# project does not have. See step4_surrogate.fit_surrogates.

for d in (DATA_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True)


# Dataset selection
# Set DATASET_SOURCE to "matminer" to load a benchmark, or "csv" to load CSV_PATH.
DATASET_SOURCE = "csv"
MATMINER_DATASET = "matbench_expt_gap"             # used when DATASET_SOURCE == "matminer"
CSV_PATH: Path | None = DATA_DIR / "pdh_literature.csv"

# Target columns. List length determines single vs multi-task / BO vs MOBO.
#   Matbench benchmark example:    TARGET_COLS = ["gap expt"]
#   Synthetic catalyst dual-target: TARGET_COLS = ["propane_TOF_log", "propane_selectivity"]
#   PDH conversion (max) only:     TARGET_COLS = ["propane_conversion"]              # fractional, 0–1
#   PDH conversion + selectivity:  TARGET_COLS = ["propane_conversion", "propane_selectivity"]
TARGET_COLS = ["propane_conversion", "propane_selectivity"]
# Direction per target: "max" to maximize, "min" to minimize. Length must match
# TARGET_COLS. Example for "high selectivity AND low deactivation":
#   TARGET_COLS = ["selectivity", "deactivation_rate"]
#   OPTIMIZATION_DIRECTIONS = ["max", "min"]
# propane_conversion is a fractional 0–1 value; "max" is the typical choice.
OPTIMIZATION_DIRECTIONS = ["max", "max"]

# Target "twins" — every column name that is a raw, log, or otherwise derived
# form of one of our targets. These must be EXCLUDED from features along with
# TARGET_COLS itself, otherwise a Pearson-|r|-with-target feature ranker (or
# any other supervised selector) silently picks them as the strongest
# predictor and the surrogate fits the target with itself. The set is applied
# in two places:
#   • webui/eda_tab.py: dropped from X_only BEFORE Tab 2's transformer fit,
#     so the twins never get sklearn-prefixed (`num__propane_TOF_s`) and
#     never appear in the transformed feature matrix.
#   • pipeline/step4_surrogate.prepare_xy: kept as a defense-in-depth drop
#     for the non-Tab-2 path (CLI catalyst flow → step4 directly).
# Extend this set when a new catalyst target is added.
TARGET_TWINS = {
    "propane_TOF_log", "propane_TOF_s", "propane_TOF",
    "deactivation_rate_log", "deactivation_rate_h", "deactivation_rate",
    # Propane conversion (fractional 0–1). yield = conversion × selectivity, so
    # propylene_yield is an algebraic twin and must be blocked alongside.
    "propane_conversion", "propane_conversion_pct", "propylene_yield",
    # Reaction outcome, not a design input — library candidates have no value
    # for it, so it can't legitimately contribute to inverse-design scoring.
    "propane_selectivity",
}

# Task type (regression / classification)
TASK = "regression"

# Subsample for fast iteration. Set to None for full dataset.
SUBSAMPLE_N = None
RANDOM_STATE = 42

# ─── Heterogeneous-catalyst mode ─────────────────────────────────────────────
# When True, expect the CSV to have separate columns for each catalyst role and
# dispatch step 2 to pipeline.catalyst_features.CatalystFeaturizer instead of
# the plain single-formula matminer path.
CATALYST_MODE = True

CATALYST_ROLES = {
    "active_metal": "active_metal",
    "promoter_1":   "promoter_1",
    "promoter_2":   "promoter_2",     # remove if you have at most one promoter
    "support":      "support",
}
CATALYST_LOADINGS = {
    "active_metal": "metal_loading_wt",
    "promoter_1":   "promoter_1_loading_wt",
    "promoter_2":   "promoter_2_loading_wt",
}
SUPPORT_LOOKUP_PATH = DATA_DIR / "lookups" / "support_properties.csv"
METAL_LOOKUP_PATH   = DATA_DIR / "lookups" / "metal_properties.csv"

# ─── Atomic-fraction catalyst mode ──────────────────────────────────────────
# Alternative to the role-based catalyst path. For datasets that report each
# catalyst as a flat atomic-fraction vector over a fixed element panel (e.g.
# ACS Materials Letters PDH dataset: 19 element columns + reaction conditions
# + targets, no explicit active_metal/promoter/support role labels).
#
# When CATALYST_FRACTION_MODE is True, step1_load renames the dataset-specific
# header (units in brackets etc.) to clean snake_case, then step2_featurize
# dispatches to pipeline.catalyst_fraction_features. Takes precedence over
# CATALYST_MODE when both are True.
#
# Featurization: per row, the dominant cation from CATALYST_FRACTION_SUPPORT_
# CATIONS is treated as the support, reconstructed via the OXIDE_MAP, and used
# for the support_phys lookup. The remaining elements form a "metal-phase"
# Composition, Magpie-featurized and joined to metal_phys via composition-
# weighted averaging. Interfacial descriptors are applied if both blocks
# yield a lattice constant.
CATALYST_FRACTION_MODE = False

CATALYST_FRACTION_ELEMENTS = [
    "Al", "B", "Co", "Cr", "Cu", "Fe", "Ga", "La", "Mg", "Mn",
    "Mo", "Nb", "Ni", "P", "Pt", "Si", "Sn", "V", "Zn", "Zr",
]
CATALYST_FRACTION_SUPPORT_CATIONS = ["Al", "Si", "Zr"]
CATALYST_FRACTION_SUPPORT_OXIDE_MAP = {
    "Al": "gamma-Al2O3",  # default to γ-alumina, the standard PDH support
    "Si": "SiO2",
    "Zr": "ZrO2",
}
# Reaction conditions for atomic-fraction CSVs (after column-name normalization).
# These pass through as numeric features and are NOT featurized via Magpie.
CATALYST_FRACTION_CONDITIONS = [
    "calcination_temp_K", "calcination_time_h", "reaction_temp_K",
    "total_pressure_bar", "propane_flow_rate_sccm", "pretreatment",
]
# Column-rename map applied by step1_load when CATALYST_FRACTION_MODE is True.
# Extend if a new atomic-fraction dataset uses different header text.
CATALYST_FRACTION_RENAME = {
    "calcination temperature [K]":      "calcination_temp_K",
    "calcination time [h]":             "calcination_time_h",
    "reaction temperature [K]":         "reaction_temp_K",
    "total pressure [bar]":             "total_pressure_bar",
    "propane flow rate [sccm]":         "propane_flow_rate_sccm",
    "oxidation (1) / reduction (-1)":   "pretreatment",
    "deactivation rate constant [h-1]": "deactivation_rate_h",  # routes to _log via step1
    "propylene yield":                  "propylene_yield",
    # Fractional 0–1; if the source CSV reports % instead, divide before load
    # (or add a "propane conversion [%]" → propane_conversion_pct slot and
    # convert downstream).
    "propane conversion":               "propane_conversion",
    "propane conversion [-]":           "propane_conversion",
}

# Optional measured features — passed through as numeric features IF present in
# the CSV. Each gets a `{col}_present` indicator so the surrogate can tell
# "not measured" apart from "measured = 0". Add or remove freely.
OPTIONAL_NUMERIC_FEATURES = [
    "metal_dispersion_pct",      # e.g. CO chemisorption, %
    "particle_size_nm",          # TEM / XRD
    "BET_surface_area_m2_g",
    "pore_volume_cm3_g",
    "calcination_temp_C",
    "reduction_temp_C",
    # Reaction-condition slots
    "reaction_temp_C",
    # "reaction_pressure_atm",   # 0/85 rows populated in PDH literature CSV (implicit 1 atm); re-enable if a future dataset reports it
    "WHSV_h",                    # weight hourly space velocity, h^-1 (PDH literature)
    # "GHSV_h",                  # gas hourly space velocity, h^-1 (use if your CSV reports GHSV instead)
    "H2_HC_ratio",
]

# Auto-EDA
OLLAMA_HOST = "http://localhost:11434"
AUTO_EDA_AVAILABLE = True   # set False to skip step 3 entirely

# Step 3 imports the Orchestrator from a SEPARATE project (github.com/Nkoizumi/
# llm-eda-mobo, developed locally as ~/auto_eda). It is not a pip dependency
# and not vendored here. Resolution order:
#   1. $INVERSE_DESIGN_AUTO_EDA_PATH
#   2. a sibling `auto_eda` checkout next to this repository
#   3. ~/auto_eda
# If none of those resolve, step 3 logs a warning and is skipped — the rest of
# the pipeline runs fine without it (`--skip eda` makes that explicit).
AUTO_EDA_PATH: Path | None = next(
    (
        p for p in (
            Path(os.environ["INVERSE_DESIGN_AUTO_EDA_PATH"])
            if os.environ.get("INVERSE_DESIGN_AUTO_EDA_PATH") else None,
            ROOT.parent / "auto_eda",
            Path.home() / "auto_eda",
        )
        if p is not None and p.is_dir()
    ),
    None,
)

# Surrogate
# Accepted values:
#   "gp"   — exact SingleTaskGP (best for ≲ ~500 training rows)
#   "svgp" — Sparse / Variational GP (mid-sized: ~500–10k rows)
#   "bnn"  — Pyro variational BNN (large datasets)
#   "both" — gp + bnn (back-compat alias)
#   "all"  — gp + svgp + bnn
#   Or comma-separated, e.g. "gp,svgp" or "svgp,bnn".
SURROGATE_KIND = "both"
BNN_TRAINING_ITERS = 1000
SVGP_NUM_INDUCING = 256      # inducing points; clamped to N if smaller
SVGP_TRAINING_ITERS = 400

# Posterior samples drawn per BNN prediction. This is a Monte-Carlo estimate,
# so the sample count sets the noise floor of every BNN number in the report.
# Measured on the PDH literature set (n=85, q=20 candidates): two independent
# draws of the BNN mean disagreed by 0.030 at 50 samples — as large as the
# GP-vs-BNN delta (~0.027) the report cites as a confidence signal, i.e. the
# "closest agreement" candidate was being picked out of noise. 512 puts the
# estimator noise ~3x below that signal (0.009) and costs milliseconds at
# these batch sizes.
BNN_PREDICT_SAMPLES = 512

# Feature reduction cap fed to the surrogate. With small-n catalyst datasets
# (n_train ~ 60) and matminer + lookup features (~500–600 dims), the GP
# kernel length-scales optimize tight and the posterior collapses to the
# prior for any library point not in-sample. Capping feature count to a
# small multiple of n_train lets the kernel actually generalize. Features
# are ranked by max |Pearson r| against any target.
# Set to None or 0 to disable.
MAX_GP_FEATURES = 30

# When True, the MAX_GP_FEATURES budget is split evenly across catalyst roles
# (active_metal, promoter_1, promoter_2, support), so the Pearson-|r| ranker
# picks top-K within each role rather than globally. Without this, an
# imbalanced training set (e.g. Pt+Sn on γ-Al2O3 dominant) causes the ranker
# to pick almost exclusively active_metal features — the surrogate then only
# discriminates catalysts by metal identity and MOBO can't leave the training
# centroid's support / promoter. Any slots a role can't fill (constant-drop
# or fewer candidates than the per-role slot count) fall back to a global
# pool ranked by |Pearson r|.
PER_ROLE_FEATURE_CAP = False

# When True, restrict the feature set to composition-derived columns only:
#   • Matminer per-role features (Magpie + Stoichiometry n-norms)
#   • Role presence flags (`{role}_present`)
#   • Role loading columns (`metal_loading_wt`, etc.)
# Drops physical-property lookups (`*_phys_*`), alloy adsorbate energies
# (`*E_ads_*`), interfacial cross-terms, mixed-oxide indicator, and reaction-
# condition columns (BET, temperatures, WHSV, H2/HC). Rationale: with
# CV R² ≈ 0 on the current dataset, the Pearson-|r| ranker in prepare_xy is
# picking noise; a smaller, physically-grounded feature set lets the surrogate
# see a more meaningful basis before the MAX_GP_FEATURES cap applies.
COMPOSITION_ONLY_FEATURES = False

# GP kernel length-scale prior (Gamma(concentration, rate)). BoTorch's default
# Gamma(3.0, 6.0) has mode ~0.33, which biases toward short length-scales that
# tightly interpolate training data but collapse to the prior elsewhere — the
# typical n << d failure mode. We bias toward longer scales (mode ~2 by
# default) so the kernel "reaches" library candidates instead of treating
# them as infinitely far away. Setting both to None falls back to the
# BoTorch default.
GP_LENGTHSCALE_PRIOR_CONCENTRATION = 3.0   # Gamma α
GP_LENGTHSCALE_PRIOR_RATE          = 1.0   # Gamma β → mode = (α-1)/β = 2.0

# K-fold CV parity. Set to 0 / None to skip CV (then step6 falls back to the
# cheaper in-sample parity). CV refits every surrogate K times, so for BNN this
# roughly multiplies step-4 wall time by K.
CV_FOLDS = 5

# Inverse design
BO_BATCH_SIZE = 20
# In catalyst mode, over-fetch by this factor before deduplicating the
# batch on displayed catalyst identity (metal, promoters, support, loadings).
# Different library-tensor rows can argmin back to the same library_raw row
# when the featurizer produces near-identical tensors — dedup guarantees
# BO_BATCH_SIZE unique catalysts (or logs a warning if it can't reach that).
BO_UNIQUE_OVERSAMPLE = 4
# Family-diversity cap for the BO batch. When set to a positive integer, the
# candidate selection greedily takes acquisition-ranked catalysts but drops any
# whose "family" already has this many entries in the batch. Family definition
# is mode-dependent:
#   • Atomic-fraction mode: dominant non-support element (fraction > 0.005).
#   • Role-based mode: the active_metal cell — the same idea, since that is the
#     role a chemist names the catalyst by.
# On training data dominated by one metal (pdh_literature is 88% Pt) the
# role-based cap will surface metals the surrogate has no signal for. That is
# deliberate — diversity for hedging, carrying honestly-large sigma — but it is
# the wrong setting for benchmark runs, where BO should exploit unrestricted.
# Rationale: with strong-signal Pareto axes, BO can over-concentrate a single
# family (e.g. 15/20 Mg-based low-deactivation picks on ACS PDH at K=30). A
# cap of 4 on q=20 lets a family lead but keeps 4-6 chemistries represented
# for handoff to experiment.
#
BO_MAX_PER_FAMILY = 4
# Over-fetch multiplier when BO_MAX_PER_FAMILY is set. Must be large enough
# that the acquisition-ranked pool still contains BO_BATCH_SIZE catalysts
# after the family-cap sweep. 5× is comfortable for q=20 with 6-family panels.
BO_FAMILY_OVERSAMPLE = 5
# Chunk size for optimize_acqf_discrete's posterior evaluation. Lower it if a
# large library exhausts GPU memory; it does not affect which candidates are
# selected, only how many are scored at once.
BO_ACQ_BATCH_SIZE = 512

# Atomic-fraction BO library — only used when CATALYST_FRACTION_MODE is True.
# The library is SAMPLED from the training data's composition manifold rather
# than enumerated, so its size is a knob rather than a product of panel sizes.
CATALYST_FRACTION_LIBRARY_SIZE = 10_000
# Cap on total non-support atomic fraction when sampling that library.
CATALYST_FRACTION_MAX_TOTAL_METAL = 0.05

# Steels benchmark library — only used by the matbench_steels preset, which
# sets STEELS_LIBRARY=True to route step 5 through the discrete steels path
# instead of continuous BO.
STEELS_LIBRARY = False
STEELS_LIBRARY_SIZE = 10_000

# Catalyst design library — only used when CATALYST_MODE is True.
# Cartesian product over these lists builds the discrete search space.
LIBRARY_ACTIVE_METALS  = ["Pt", "Pd", "Rh", "Ir", "Ru", "Ni", "Au", "Co", "Fe", "Cu", "Mo"]
LIBRARY_PROMOTERS_1    = ["", "Sn", "Re", "Ga", "In"]
# La and Ce are the two most-studied rare-earth promoters in heterogeneous
# catalysis (PDH, reforming, three-way etc.); other lanthanides (Pr, Nd, Sm,
# Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Sc, Y) live in the lookup CSV and can
# be ticked in the webui's Catalyst Library tab if needed.
LIBRARY_PROMOTERS_2    = ["", "K", "Cs", "Ag", "La", "Ce"]
LIBRARY_METAL_LOADINGS = [0.3, 1.0, 2.0, 5.0]
LIBRARY_PROMO_LOADINGS = [0.3, 1.0]

# Report
REPORT_LLM_MODEL = "phi4:14b-q4_K_M"   # solid non-thinking model; qwen3:32b enters chain-of-thought and stalls
REPORT_TOP_K = 5

# How long Ollama keeps the report model in VRAM after step 6 finishes, in the
# form Ollama's `keep_alive` accepts (0 = unload immediately, "5m" = Ollama's
# default, -1 = keep forever). Ollama and steps 4-5 share one GPU: step 5's
# acquisition peaks at ~3.2 GB allocated, so on a 24 GB card any model leaving
# less than that free makes the *next* run fail with a CUDA OOM. Measured
# 2026-08-08: phi4:14b (10 GB resident) is safe, qwen3:32b (20 GB) is not.
# Set to "5m" if you re-run reports back-to-back and would rather pay VRAM than
# a model reload.
REPORT_LLM_KEEP_ALIVE = 0
