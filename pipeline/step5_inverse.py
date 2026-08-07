"""Step 5: inverse design loop.

Two modes, picked by ``config.CATALYST_MODE``:

  - Continuous (single-formula benchmarks): BoTorch BO/MOBO in standardized
    feature space; report nearest known material by Euclidean distance.

  - Catalyst (heterogeneous catalysts): enumerate a discrete library of plausible
    formulations via ``catalyst_library.build_library``, featurize with
    ``CatalystFeaturizer``, align columns to the training feature set, and run
    ``optimize_acqf_discrete`` on the library tensor. Returns actual catalyst
    formulations the user can synthesize.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

import config
from seeding import make_sampler, seed_everything
from step4_surrogate import XYData

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


class Selection(NamedTuple):
    """A BO batch plus the exact standardized features it was scored on.

    Carrying X_std out of the `_run_*` functions is what lets the BNN
    cross-check evaluate the same points the driver did. Re-deriving the
    features from `candidates` is not equivalent — see
    `_attach_bnn_predictions`.
    """
    candidates: pd.DataFrame
    X_std: torch.Tensor


def _get_signs(n_targets: int) -> torch.Tensor:
    """+1 per "max" target, -1 per "min". Defaults to all-max when unset."""
    directions = getattr(config, "OPTIMIZATION_DIRECTIONS", None) or ["max"] * n_targets
    if len(directions) != n_targets:
        raise ValueError(
            f"OPTIMIZATION_DIRECTIONS has {len(directions)} entries but "
            f"there are {n_targets} target(s). Expected one direction per target."
        )
    bad = [d for d in directions if d not in ("max", "min")]
    if bad:
        raise ValueError(f"OPTIMIZATION_DIRECTIONS entries must be 'max' or 'min'; got {bad}.")
    return torch.tensor(
        [1.0 if d == "max" else -1.0 for d in directions],
        dtype=DTYPE, device=DEVICE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_inverse(df: pd.DataFrame, surrogates: dict,
                transformer=None) -> pd.DataFrame:
    # Re-seed at the top of step 5 rather than relying on whatever RNG state
    # step 4 happened to leave behind: the amount of randomness step 4 consumes
    # depends on CV_FOLDS and SURROGATE_KIND, so without this the BO batch
    # would silently change when you toggle CV on or off.
    seed_everything()

    data: XYData = surrogates["data"]
    # Driver priority: exact GP > SVGP > BNN. BNN cross-check kept for the
    # candidate report when GP is the driver and BNN is also fit (existing
    # behavior). With SVGP added in this pass, it is currently only used as
    # the BO driver itself or for the parity plot — not as a per-candidate
    # cross-check in the report (would need new columns plumbed into
    # plots.make_gp_vs_bnn_scatter / interactive_plots.make_gp_vs_bnn_figure).
    gp = surrogates.get("gp")
    svgp = surrogates.get("svgp")
    bnn = surrogates.get("bnn")
    surrogate = gp or svgp or bnn
    if surrogate is None:
        raise RuntimeError("No surrogate available — fit one in step 4 first.")

    if getattr(config, "CATALYST_FRACTION_MODE", False):
        selection = _run_catalyst_fraction_discrete(df, data, surrogate)
    elif config.CATALYST_MODE:
        selection = _run_catalyst_discrete(df, data, surrogate, transformer=transformer)
    elif getattr(config, "STEELS_LIBRARY", False):
        selection = _run_steels_discrete(df, data, surrogate, transformer=transformer)
    else:
        selection = _run_continuous(df, data, surrogate)

    out = selection.candidates

    # If a BNN is also fit, attach its predictions for the selected candidates
    # so the report can show a GP-vs-BNN comparison. Works in every mode: the
    # cross-check now scores the SAME feature tensor the driver scored rather
    # than re-deriving one from the displayed candidate table.
    if bnn is not None and surrogate is not bnn:
        out = _attach_bnn_predictions(out, data, bnn, selection.X_std)
        out.to_parquet(config.DATA_DIR / "candidates.parquet")

    return out


def _attach_bnn_predictions(candidates_df: pd.DataFrame, data: XYData, bnn,
                            X_std: torch.Tensor) -> pd.DataFrame:
    """Score the BNN on the SAME standardized features the driver scored.

    This used to re-featurize `candidates_df` with a fresh CatalystFeaturizer
    and push the result through `_align_to_training`, which zero-fills any
    column it can't find. That produced a systematically different input than
    the one the GP saw, because the displayed candidate table carries only
    composition — active_metal, promoters, support and loadings. Every
    reaction condition (`reaction_temp_C`, `WHSV_h`, `H2_HC_ratio`, …) was
    therefore zero-filled for the BNN and its `{col}_present` indicator set to
    0, while step 2a had filled exactly those columns with the TRAINING MEDIAN
    for the GP. Any Tab-2 transformer was skipped as well.

    The Δ column in the report — which the LLM is told to cite as a
    surrogate-confidence signal, and which `_build_agreement_summary` uses to
    name the "safest GP-BNN-aligned pick" — was thus a mixture of genuine
    model disagreement and a pure input artifact.

    Using the driver's own tensor removes the artifact, and as a side effect
    drops the CatalystFeaturizer dependency that restricted the cross-check to
    role-based catalyst mode.
    """
    if X_std is None:
        log.info("No feature tensor available for the selected candidates; "
                 "skipping BNN cross-check.")
        return candidates_df
    if X_std.shape[0] != len(candidates_df):
        log.warning(
            "BNN cross-check skipped: %d feature rows for %d candidates. The "
            "selection and its tensor must stay in lockstep.",
            X_std.shape[0], len(candidates_df),
        )
        return candidates_df

    mean_std, sd_std = bnn.predict(X_std)
    pred_real = mean_std * data.y_std + data.y_mean
    pred_sd_real = sd_std * data.y_std

    out = candidates_df.copy()
    for t, name in enumerate(data.target_cols):
        out[f"pred_{name}_bnn"] = pred_real[:, t].detach().cpu().numpy()
        out[f"pred_{name}_bnn_sd"] = pred_sd_real[:, t].detach().cpu().numpy()
    log.info("Attached BNN cross-check predictions to %d candidates "
             "(scored on the driver's own feature tensor).", len(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared machinery
#
# The four `_run_*` paths differ only in how they build and featurize their
# library. Everything after that — acquisition, discrete selection, row
# recovery, prediction attachment — used to be copy-pasted per path, which is
# how two alignment bugs came to exist in one copy each. Fix them here once.
# ─────────────────────────────────────────────────────────────────────────────
def _build_acquisition(surrogate, data: XYData, multi_objective: bool):
    """qLogNEHVI for multi-objective, qLogNEI otherwise.

    `signs` flips minimize-targets so every path can treat the problem as
    maximization; the ref_point therefore lives in objective (post-flip) space.
    """
    signs = _get_signs(data.Y.shape[1])

    if multi_objective:
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.acquisition.multi_objective.objective import (
            WeightedMCMultiOutputObjective,
        )
        # ref_point as a plain list, not a Tensor: BoTorch 0.18 accepts both,
        # but the Tensor form has silently mis-handled MOBO in other versions.
        ref = ((data.Y * signs).min(dim=0).values - 1.0).detach().cpu().tolist()
        return qLogNoisyExpectedHypervolumeImprovement(
            model=surrogate.model, X_baseline=data.X, ref_point=ref,
            objective=WeightedMCMultiOutputObjective(weights=signs),
            sampler=make_sampler(multi_objective=True),
        )

    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.objective import ScalarizedPosteriorTransform
    # signs * y is in standardized space; arg-max-of-(signs*y) ≡ min y when signs=-1.
    return qLogNoisyExpectedImprovement(
        model=surrogate.model, X_baseline=data.X,
        posterior_transform=ScalarizedPosteriorTransform(weights=signs),
        sampler=make_sampler(multi_objective=False),
    )


def _apply_transformer(library_feat: pd.DataFrame, transformer, label: str) -> pd.DataFrame:
    """Apply (never re-fit) a Tab-2 sklearn transformer to a featurized library.

    Missing columns the transformer was fit on are filled with 0; surplus
    columns are dropped. On any failure the raw featurized library is returned
    unchanged — a mis-transformed library is worse than an untransformed one.
    """
    if transformer is None:
        return library_feat
    try:
        fit_cols = list(getattr(transformer, "feature_names_in_", []))
        if not fit_cols:
            return library_feat
        lib_in = library_feat.reindex(columns=fit_cols, fill_value=0.0)
        out = transformer.transform(lib_in)
        # Use the full pipeline's get_feature_names_out — it threads names
        # through every step including HighCorrelationRemover, so out_cols
        # correctly reflects post-correlation-removal columns. Falling back to
        # the preprocessor's names would over-count and zero-fill downstream.
        try:
            out_cols = list(transformer.get_feature_names_out())
        except Exception:
            try:
                out_cols = list(
                    transformer.named_steps["preprocessor"].get_feature_names_out()
                )
            except Exception:
                out_cols = None
        if isinstance(out, pd.DataFrame):
            if out_cols is not None and len(out_cols) == out.shape[1]:
                out.columns = out_cols
            out.index = library_feat.index
            result = out
        else:
            arr = np.asarray(out)
            if out_cols is None or len(out_cols) != arr.shape[1]:
                out_cols = [f"feat_{i}" for i in range(arr.shape[1])]
            result = pd.DataFrame(arr, columns=out_cols, index=library_feat.index)
        log.info("Applied Tab-2 transformer to %s library (%d → %d cols).",
                 label, len(fit_cols), len(out_cols))
        return result
    except Exception as e:
        log.warning("Tab-2 transformer apply on %s library failed (%s); "
                    "falling back to raw featurized library.", label, e)
        return library_feat


def _standardize_library(library_feat: pd.DataFrame, data: XYData) -> torch.Tensor:
    """Align a featurized library to the training columns and standardize it
    with the training moments."""
    X_lib = torch.tensor(_align_to_training(library_feat, data.feature_cols),
                         dtype=DTYPE, device=DEVICE)
    X_lib_std = (X_lib - data.x_mean) / data.x_std
    log.info("Library tensor: %s. Training tensor: %s.",
             tuple(X_lib_std.shape), tuple(data.X.shape))
    return X_lib_std


def _recover_library_rows(X_lib_std: torch.Tensor,
                          candidates_std: torch.Tensor) -> list[int]:
    """Map each selected point back to its library row by nearest neighbour.

    optimize_acqf_discrete returns the chosen rows' VALUES, not their indices,
    so they have to be located again. Ties are possible when the featurizer
    produces numerically indistinguishable rows; argmin picks the first, which
    the caller's identity dedup then collapses.
    """
    # cdist over the whole batch rather than a Python loop over candidates.
    return torch.cdist(candidates_std, X_lib_std).argmin(dim=1).cpu().tolist()


def _attach_predictions(selected: pd.DataFrame, sel_X: torch.Tensor,
                        surrogate, data: XYData) -> pd.DataFrame:
    """Attach the driver's mean / sd for the selected rows, in original Y units."""
    mean_std, sd_std = surrogate.predict(sel_X)
    pred_real = mean_std * data.y_std + data.y_mean
    pred_sd_real = sd_std * data.y_std
    for t, name in enumerate(data.target_cols):
        selected[f"pred_{name}"] = pred_real[:, t].detach().cpu().numpy()
        selected[f"pred_{name}_sd"] = pred_sd_real[:, t].detach().cpu().numpy()
    return selected


def _select_discrete(acq, X_lib_std: torch.Tensor, q: int) -> torch.Tensor:
    """Run optimize_acqf_discrete, clamping q to the library size."""
    from botorch.optim.optimize import optimize_acqf_discrete

    candidates_std, _ = optimize_acqf_discrete(
        acq_function=acq,
        q=min(int(q), X_lib_std.shape[0]),
        choices=X_lib_std,
        unique=True,
        max_batch_size=int(getattr(config, "BO_ACQ_BATCH_SIZE", 512)),
    )
    return candidates_std


def _save_candidates(selected: pd.DataFrame, kind: str, library_size: int) -> None:
    out_path = config.DATA_DIR / "candidates.parquet"
    selected.to_parquet(out_path)
    log.info("Selected %d %s candidates from library of %d.",
             len(selected), kind, library_size)


# ─────────────────────────────────────────────────────────────────────────────
# Continuous (single-formula) path
# ─────────────────────────────────────────────────────────────────────────────
def _run_continuous(df: pd.DataFrame, data: XYData, surrogate) -> Selection:
    multi_objective = data.Y.shape[1] > 1
    log.info("Continuous %s …", "MOBO (qLogNEHVI)" if multi_objective else "BO (qLogNEI)")

    candidates_std = _optimize_continuous(surrogate, data, multi_objective)
    nearest = _nearest_known(candidates_std, data, df)

    out = pd.DataFrame({
        "candidate_idx": range(candidates_std.shape[0]),
        "nearest_material": nearest,
    })
    out = _attach_predictions(out, candidates_std, surrogate, data)

    out_path = config.DATA_DIR / "candidates.parquet"
    out.to_parquet(out_path)
    log.info("Saved %d candidates to %s.", len(out), out_path)
    return Selection(out, candidates_std)


def _optimize_continuous(surrogate, data: XYData, multi_objective: bool) -> torch.Tensor:
    from botorch.optim import optimize_acqf

    d = data.X.shape[1]
    bounds = torch.stack([
        torch.full((d,), -3.0, dtype=DTYPE, device=DEVICE),
        torch.full((d,),  3.0, dtype=DTYPE, device=DEVICE),
    ])

    candidates, _ = optimize_acqf(
        acq_function=_build_acquisition(surrogate, data, multi_objective),
        bounds=bounds, q=config.BO_BATCH_SIZE,
        num_restarts=10, raw_samples=256,
    )
    return candidates.detach()


def _nearest_known(candidates: torch.Tensor, data: XYData, df: pd.DataFrame) -> list[str]:
    """Nearest training material to each candidate, by standardized distance.

    `data.X` indexes the rows that SURVIVED prepare_xy, which drops every row
    with a NaN in any target. Indexing the full `df` with a `data.X` row index
    therefore reported the wrong material whenever the dataset had a missing
    target — silently, since both are valid formulas. Re-apply the same mask
    here so the two line up.
    """
    aligned = df
    targets = [t for t in data.target_cols if t in df.columns]
    if targets:
        aligned = df.loc[df[targets].notna().all(axis=1)].reset_index(drop=True)

    if len(aligned) != data.X.shape[0]:
        log.warning(
            "Cannot align training rows to the surrogate's tensor (%d rows vs "
            "%d) — reporting nearest material as '?'. prepare_xy must have "
            "dropped rows for a reason this function does not model.",
            len(aligned), data.X.shape[0],
        )
        return ["?"] * candidates.shape[0]

    formulas = (
        aligned["composition_str"] if "composition_str" in aligned.columns
        else aligned["composition"].astype(str)
    ).tolist()
    nearest_idx = torch.cdist(candidates, data.X).argmin(dim=1).cpu().tolist()
    return [formulas[i] for i in nearest_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Catalyst discrete path
# ─────────────────────────────────────────────────────────────────────────────
def _run_catalyst_discrete(df: pd.DataFrame, data: XYData, surrogate,
                            transformer=None) -> Selection:
    from catalyst_features import CatalystFeaturizer
    from catalyst_library import build_library

    multi_objective = data.Y.shape[1] > 1
    log.info("Catalyst-mode discrete %s …",
             "MOBO (qLogNEHVI)" if multi_objective else "BO (qLogNEI)")

    # 1. Build library --------------------------------------------------------
    sup_lookup = pd.read_csv(config.SUPPORT_LOOKUP_PATH)
    library_raw = build_library(
        supports=sup_lookup["support"].tolist(),
        role_cols=config.CATALYST_ROLES,
        loading_cols=config.CATALYST_LOADINGS,
        active_metals=getattr(config, "LIBRARY_ACTIVE_METALS", None),
        promoters_1=getattr(config, "LIBRARY_PROMOTERS_1", None),
        promoters_2=getattr(config, "LIBRARY_PROMOTERS_2", None),
        metal_loadings=getattr(config, "LIBRARY_METAL_LOADINGS", None),
        promo_loadings=getattr(config, "LIBRARY_PROMO_LOADINGS", None),
    )

    # 2. Featurize library through the same pipeline --------------------------
    cf = CatalystFeaturizer(
        roles=config.CATALYST_ROLES,
        loadings=config.CATALYST_LOADINGS,
        support_lookup_path=config.SUPPORT_LOOKUP_PATH,
        metal_lookup_path=config.METAL_LOOKUP_PATH,
        optional_numeric_features=getattr(config, "OPTIONAL_NUMERIC_FEATURES", []),
    )
    library_feat = cf.fit_transform(library_raw)

    # 2a. Fill optional measured/reaction-condition features with TRAINING
    # MEDIANS rather than zero. The BO library only specifies composition
    # (active_metal, promoters, support, loadings), so reaction conditions
    # like reaction_temp_C, WHSV_h, H2_HC_ratio are absent and would
    # otherwise be zero-filled by the reindex below. After standardization,
    # zero-filled columns become (0 - μ_train)/σ_train — a large constant
    # negative number — and push every library row into the same out-of-
    # distribution corner of the GP's input space, so the kernel returns
    # the prior mean for every candidate (mode collapse).
    # Imputing with the training median places the library at the same
    # operating point as the training centroid, so the GP can actually
    # discriminate compositions.
    optional_cols = list(getattr(config, "OPTIONAL_NUMERIC_FEATURES", []))
    fill_log = []
    for col in optional_cols:
        if col in df.columns and df[col].notna().any():
            med = float(pd.to_numeric(df[col], errors="coerce").median())
            library_feat[col] = med
            library_feat[f"{col}_present"] = 1
            fill_log.append(f"{col}={med:g}")
    if fill_log:
        log.info("Filled BO library with training-median conditions: %s",
                 ", ".join(fill_log))

    # 2b. Optional Tab-2 transformer — apply (NOT re-fit) so the library's
    # column space matches the training transformed df.
    library_feat = _apply_transformer(library_feat, transformer, "BO")

    # 3. Align columns to training feature set --------------------------------
    X_lib_std = _standardize_library(library_feat, data)

    # 4. Build acquisition function -------------------------------------------
    acq = _build_acquisition(surrogate, data, multi_objective)

    # 5. Discrete acquisition -------------------------------------------------
    # Over-fetch by BO_UNIQUE_OVERSAMPLE so we still hit BO_BATCH_SIZE after
    # deduping on displayed catalyst identity. optimize_acqf_discrete's
    # unique=True prevents choice-index reuse but not identity collapse:
    # different library-tensor rows can argmin back to the same library_raw
    # row when the featurizer produces near-identical tensors across e.g.
    # promoter-loading variants that are numerically indistinguishable.
    q_target = int(config.BO_BATCH_SIZE)
    oversample = int(getattr(config, "BO_UNIQUE_OVERSAMPLE", 4))
    q_ask = min(q_target * oversample, X_lib_std.shape[0])
    candidates_std = _select_discrete(acq, X_lib_std, q_ask)

    # 6. Recover library rows for selected candidates -------------------------
    selected_idx = _recover_library_rows(X_lib_std, candidates_std)

    # 6a. Dedup on displayed catalyst identity, cap at q_target.
    id_cols = [c for c in
               list(config.CATALYST_ROLES.values()) + list(config.CATALYST_LOADINGS.values())
               if c in library_raw.columns]
    seen: set = set()
    unique_idx: list[int] = []
    for i in selected_idx:
        key = tuple(library_raw.iloc[i][k] for k in id_cols)
        if key in seen:
            continue
        seen.add(key)
        unique_idx.append(i)
        if len(unique_idx) >= q_target:
            break
    n_dupes = len(selected_idx) - len(unique_idx)
    if n_dupes:
        log.info("Deduped %d duplicate catalyst identities from BO batch "
                 "(fetched q=%d, kept %d unique).",
                 n_dupes, q_ask, len(unique_idx))
    if len(unique_idx) < q_target:
        log.warning("Only %d unique catalysts survived dedup (target %d). "
                    "Raise BO_UNIQUE_OVERSAMPLE (currently %d) or widen the library.",
                    len(unique_idx), q_target, oversample)
    selected_idx = unique_idx

    selected = library_raw.iloc[selected_idx].reset_index(drop=True)

    # 7. Attach predictions + save -------------------------------------------
    sel_X = X_lib_std[selected_idx]
    selected = _attach_predictions(selected, sel_X, surrogate, data)
    _save_candidates(selected, "catalyst", len(library_raw))
    return Selection(selected, sel_X)


def _run_catalyst_fraction_discrete(df: pd.DataFrame, data: XYData,
                                     surrogate) -> Selection:
    """Discrete BO over an empirical atomic-fraction catalyst library.

    Companion to `_run_catalyst_discrete` (role-based path). Same acquisition
    and discrete-optimization plumbing; differences are the library builder
    (`catalyst_fraction_library.build_fraction_library`) and the featurizer
    (`catalyst_fraction_features.featurize_fractions`). No Tab-2 transformer
    branch — Tab 2 is gated off when CATALYST_MODE is False but in case the
    flag combination is unusual, we skip transforms here unconditionally.
    """
    from catalyst_fraction_features import featurize_fractions
    from catalyst_fraction_library import build_fraction_library

    multi_objective = data.Y.shape[1] > 1
    log.info("Atomic-fraction catalyst discrete %s …",
             "MOBO (qLogNEHVI)" if multi_objective else "BO (qLogNEI)")

    # 1. Build library --------------------------------------------------------
    library_raw = build_fraction_library(
        df,
        n=int(getattr(config, "CATALYST_FRACTION_LIBRARY_SIZE", 10_000)),
        seed=int(getattr(config, "RANDOM_STATE", 42)),
        element_cols=config.CATALYST_FRACTION_ELEMENTS,
        support_cations=config.CATALYST_FRACTION_SUPPORT_CATIONS,
        condition_cols=config.CATALYST_FRACTION_CONDITIONS,
        max_total_metal=float(
            getattr(config, "CATALYST_FRACTION_MAX_TOTAL_METAL", 0.05)
        ),
    )

    # 2. Featurize library through the same atomic-fraction pipeline --------
    library_feat = featurize_fractions(
        library_raw,
        element_cols=config.CATALYST_FRACTION_ELEMENTS,
        support_cations=config.CATALYST_FRACTION_SUPPORT_CATIONS,
        support_oxide_map=config.CATALYST_FRACTION_SUPPORT_OXIDE_MAP,
        condition_cols=config.CATALYST_FRACTION_CONDITIONS,
        support_lookup_path=config.SUPPORT_LOOKUP_PATH,
        metal_lookup_path=config.METAL_LOOKUP_PATH,
    )

    # 3. Align columns to training feature set --------------------------------
    X_lib_std = _standardize_library(library_feat, data)

    # 4. Build acquisition function -------------------------------------------
    acq = _build_acquisition(surrogate, data, multi_objective)

    # 5. Discrete acquisition -------------------------------------------------
    family_cap = getattr(config, "BO_MAX_PER_FAMILY", None) or 0
    q_target = int(config.BO_BATCH_SIZE)
    if family_cap:
        oversample = int(getattr(config, "BO_FAMILY_OVERSAMPLE", 5))
        q_fetch = min(q_target * oversample, X_lib_std.shape[0])
    else:
        q_fetch = q_target
    candidates_std = _select_discrete(acq, X_lib_std, q_fetch)

    # 6. Recover library rows for selected candidates -------------------------
    ordered_idx = _recover_library_rows(X_lib_std, candidates_std)

    # 6a. Family-diversity cap (optional). Family = dominant non-support
    # element (fraction > 0.005). Greedy in acquisition order.
    if family_cap:
        non_support = [e for e in config.CATALYST_FRACTION_ELEMENTS
                       if e not in set(config.CATALYST_FRACTION_SUPPORT_CATIONS)]
        def _family(row):
            fracs = {e: float(row.get(e, 0.0)) for e in non_support}
            fracs = {e: v for e, v in fracs.items() if v > 0.005}
            return max(fracs, key=fracs.get) if fracs else "(no-metal)"

        selected_idx: list[int] = []
        counts: dict[str, int] = {}
        dropped_over_cap = 0
        for i in ordered_idx:
            fam = _family(library_raw.iloc[i])
            if counts.get(fam, 0) >= family_cap:
                dropped_over_cap += 1
                continue
            selected_idx.append(i)
            counts[fam] = counts.get(fam, 0) + 1
            if len(selected_idx) >= q_target:
                break
        if len(selected_idx) < q_target:
            # Under-filled — fall back to top-acquisition catalysts ignoring cap.
            log.warning("Family-diversity cap=%d left %d/%d slots empty after "
                        "%d over-cap drops. Raise BO_FAMILY_OVERSAMPLE "
                        "(currently %d) or lower cap. Filling with best-acq.",
                        family_cap, q_target - len(selected_idx), q_target,
                        dropped_over_cap, oversample)
            chosen = set(selected_idx)
            for i in ordered_idx:
                if i in chosen:
                    continue
                selected_idx.append(i)
                if len(selected_idx) >= q_target:
                    break
        log.info("Family cap=%d applied: %s (dropped %d over-cap).",
                 family_cap, dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                 dropped_over_cap)
    else:
        selected_idx = ordered_idx[:q_target]

    selected = library_raw.iloc[selected_idx].reset_index(drop=True)

    # 7. Attach predictions + save -------------------------------------------
    sel_X = X_lib_std[selected_idx]
    selected = _attach_predictions(selected, sel_X, surrogate, data)
    _save_candidates(selected, "catalyst", len(library_raw))
    return Selection(selected, sel_X)


def _run_steels_discrete(df: pd.DataFrame, data: XYData, surrogate,
                          transformer=None) -> Selection:
    """Discrete BO over a random Fe-balanced steel library. Same acquisition
    plumbing as the catalyst path; differences are the library builder and the
    matminer single-formula featurizer (Magpie + Stoichiometry)."""
    from steels_library import build_steels_library
    from matminer.featurizers.composition import ElementProperty, Stoichiometry

    multi_objective = data.Y.shape[1] > 1
    log.info("Steels-mode discrete %s …",
             "MOBO (qLogNEHVI)" if multi_objective else "BO (qLogNEI)")

    n_target = int(getattr(config, "STEELS_LIBRARY_SIZE", 10_000))
    library_raw = build_steels_library(n_target=n_target)

    # 1. Featurize the library through the same matminer transforms used in
    # step 2's single-formula path. NaN rows (rare; element coverage gaps) are
    # dropped — step2 does the same.
    log.info("Featurizing %d steel candidates …", len(library_raw))
    work = library_raw.copy()
    for fz in (ElementProperty.from_preset("magpie"), Stoichiometry()):
        work = fz.featurize_dataframe(work, "composition", ignore_errors=True)
    feature_cols_lib = [c for c in work.columns
                        if c not in ("composition", "composition_str")]
    before = len(work)
    # Select the surviving rows by POSITION and apply the same mask to both
    # frames. The previous version read `work.index` *after*
    # `reset_index(drop=True)`, which yields 0..n-1 and therefore selected the
    # first n rows of library_raw rather than the rows that actually survived
    # — silently pairing each candidate's composition string with a different
    # candidate's features and predictions.
    kept_pos = np.flatnonzero(work[feature_cols_lib].notna().all(axis=1).values)
    work = work.iloc[kept_pos].reset_index(drop=True)
    if before != len(work):
        log.info("Dropped %d library rows that failed featurization.", before - len(work))
    library_raw = library_raw.iloc[kept_pos].reset_index(drop=True)
    library_feat = work.drop(columns=["composition"])

    # 2. Optional Tab-2 transformer — apply (NOT re-fit) so library matches
    # the surrogate's training column space.
    library_feat = _apply_transformer(library_feat, transformer, "steels")

    # 3. Align to surrogate's training feature columns + standardize.
    X_lib_std = _standardize_library(library_feat, data)

    # 4. Acquisition (signs handle min/max).
    acq = _build_acquisition(surrogate, data, multi_objective)
    candidates_std = _select_discrete(acq, X_lib_std, config.BO_BATCH_SIZE)

    # 5. Recover library rows + attach predictions.
    selected_idx = _recover_library_rows(X_lib_std, candidates_std)
    selected = library_raw.iloc[selected_idx][["composition_str"]].reset_index(drop=True)
    sel_X = X_lib_std[selected_idx]
    selected = _attach_predictions(selected, sel_X, surrogate, data)
    _save_candidates(selected, "steel", len(library_raw))
    return Selection(selected, sel_X)


def _align_to_training(library_feat: pd.DataFrame, training_feature_cols: list[str]) -> np.ndarray:
    """Return library feature matrix in the column order the surrogate expects.

    Missing columns get zero (the GP will treat them as the standardized origin
    after rescaling; combined with the `{role}_present` indicators this is the
    same convention as training-time NaN handling).
    """
    df = library_feat.copy()
    for c in training_feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[training_feature_cols].fillna(0.0).astype(float).values


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from step1_load import load_dataset
    from step2_featurize import featurize
    from step4_surrogate import fit_surrogates
    df = load_dataset()
    df = featurize(df)
    surrogates = fit_surrogates(df)
    cands = run_inverse(df, surrogates)
    print(cands.head(20).to_string())
