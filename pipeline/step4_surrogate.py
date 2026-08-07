"""Step 4: surrogate models.

Three surrogate paths, all exposing the same .fit / .predict interface so step 5
can swap them:

  GPSurrogate    — BoTorch exact SingleTaskGP (small data: ≲ ~500 rows)
  SVGPSurrogate  — Sparse / Variational GP (mid-sized: ~500–10k rows)
  BNNSurrogate   — Pyro variational BNN (large datasets)

Multi-task is selected automatically when len(config.TARGET_COLS) > 1 (one
SingleTask* per target, wrapped in ModelListGP for GP/SVGP; single MLP head
for BNN).
"""
from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from seeding import seed_everything

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double

_KIND_ALIASES = {"both": {"gp", "bnn"}, "all": {"gp", "svgp", "bnn"}}


def _active_kinds() -> set[str]:
    """Parse config.SURROGATE_KIND → set of surrogate names.

    Accepts a single token ("gp"/"svgp"/"bnn"), a comma-list ("gp,svgp"),
    or one of the aliases "both" / "all".
    """
    raw = (getattr(config, "SURROGATE_KIND", "gp") or "gp").lower().strip()
    if raw in _KIND_ALIASES:
        return set(_KIND_ALIASES[raw])
    return {p.strip() for p in raw.split(",") if p.strip()}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class XYData:
    X: torch.Tensor      # (N, D) standardized features
    Y: torch.Tensor      # (N, T) standardized targets
    feature_cols: list
    target_cols: list
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    optional_medians: dict = field(default_factory=dict)  # {col: train_median} for OPTIONAL_NUMERIC_FEATURES

    def unstd_y(self, y_std: torch.Tensor) -> torch.Tensor:
        return y_std * self.y_std + self.y_mean


def prepare_xy(df: pd.DataFrame) -> XYData:
    """Split features and targets, standardize both.

    Catalyst-mode features can have NaN for absent roles (e.g. promoter_2 empty).
    We drop columns that are entirely NaN, then impute remaining NaN with 0 — the
    `{role}_present` indicator column lets the GP distinguish "absent" from "zero".

    Rows with NaN in ANY target are dropped before tensor build — botorch's
    SingleTaskGP rejects NaN Y outright, and silent imputation would bias the
    surrogate. Real PDH literature data has ~2 rows missing TOF and ~15 rows
    missing deactivation, so this is exercised in practice.
    """
    targets = config.TARGET_COLS
    target_mask = df[list(targets)].notna().all(axis=1)
    n_dropped = int((~target_mask).sum())
    if n_dropped:
        log.info("Dropping %d rows with NaN in target(s) %s (kept %d).",
                 n_dropped, list(targets), int(target_mask.sum()))
        df = df.loc[target_mask].reset_index(drop=True)

    drop = set(targets) | {
        "composition", "composition_str",
        # Provenance / row-identifier columns that leak in if numeric.
        "entry_id", "reference_paper", "reference_doi", "notes",
    }
    # Defense-in-depth for the non-Tab-2 path: drop every known target twin
    # (raw/log counterparts) so the Pearson-|r| feature ranker can't pick
    # them. The Tab 2 path drops these earlier (webui/eda_tab.py) so the
    # twins never get sklearn-prefixed and slip past this name-based filter.
    drop |= set(getattr(config, "TARGET_TWINS", set()))
    feature_cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    skipped = [c for c in df.columns if c not in drop and c not in feature_cols]
    if skipped:
        log.info("Skipped %d non-numeric columns: %s%s",
                 len(skipped), skipped[:5], " …" if len(skipped) > 5 else "")

    # Composition-only filter: keep Matminer per-role + presence flags + loadings.
    # Drops physical lookups, alloy binding, interfacial, mixed-oxide, reaction
    # conditions. Applied here so it also propagates through library alignment
    # in step5 (via data.feature_cols) — the featurizer keeps producing the
    # full set for the library, but only these columns are used downstream.
    if getattr(config, "COMPOSITION_ONLY_FEATURES", False):
        loading_cols = set(getattr(config, "CATALYST_LOADINGS", {}).values())
        def _is_composition(c: str) -> bool:
            if "_phys_" in c or "E_ads_" in c or c.startswith("interfacial_"):
                return False
            if c == "is_mixed_oxide":
                return False
            if c.endswith("_present"):
                return True
            if c in loading_cols:
                return True
            if "MagpieData" in c:
                return True
            # Stoichiometry n-norm columns are role_-prefixed with e.g. "0-norm".
            if any(c.endswith(f"_{k}-norm") for k in (0, 2, 3, 5, 7, 10)):
                return True
            return False
        kept = [c for c in feature_cols if _is_composition(c)]
        dropped_cols = [c for c in feature_cols if c not in kept]
        log.info(
            "Composition-only filter: kept %d/%d features "
            "(dropped %d non-composition, e.g. %s%s).",
            len(kept), len(feature_cols), len(dropped_cols),
            dropped_cols[:5], " …" if len(dropped_cols) > 5 else "",
        )
        feature_cols = kept

    work = df[feature_cols].copy()
    all_nan = [c for c in feature_cols if work[c].isna().all()]
    if all_nan:
        log.info("Dropping %d all-NaN feature columns.", len(all_nan))
        work = work.drop(columns=all_nan)
        feature_cols = [c for c in feature_cols if c not in all_nan]
    # Training-median impute OPTIONAL_NUMERIC_FEATURES (reaction conditions,
    # BET, etc.) before the generic zero-fill. Zero-filling H2_HC_ratio=0 or
    # WHSV_h=0 is physically wrong (pure propane / infinite residence time)
    # and pushes those rows to the standardized tail, muting the signal from
    # the {col}_present indicator emitted by the featurizer. Library step-2a
    # uses the same imputation; matching them here lets the surrogate treat
    # train and library as one operating regime. Called per-CV-fold, so the
    # median is fold-local — no leakage.
    optional_cols = [c for c in getattr(config, "OPTIONAL_NUMERIC_FEATURES", [])
                     if c in work.columns]
    optional_medians: dict = {}
    if optional_cols:
        optional_medians = {c: float(pd.to_numeric(work[c], errors="coerce").median())
                            for c in optional_cols}
        # Drop NaN medians (column entirely NaN in this fold — extremely rare
        # after the all-NaN filter above but possible in small CV folds).
        optional_medians = {c: v for c, v in optional_medians.items()
                            if not np.isnan(v)}
        n_med = int(work[list(optional_medians)].isna().sum().sum()) if optional_medians else 0
        if n_med:
            work = work.fillna(optional_medians)
            log.info("Training-median imputed %d NaN cells across %d optional "
                     "numeric columns (%s).", n_med, len(optional_medians),
                     ", ".join(f"{c}={v:g}" for c, v in optional_medians.items()))
    n_nan = int(work.isna().sum().sum())
    if n_nan:
        log.info("Imputing %d NaN feature cells with 0 (absent-role features).", n_nan)
        work = work.fillna(0.0)
    # Force a clean numeric numpy array — pd.get_dummies may produce
    # nullable-bool columns that confuse torch.tensor.
    work = work.astype(float, copy=False)

    # Drop columns that are constant in training. They carry zero information
    # for the GP (kernel along a constant direction is identically 1), but if
    # the BO library has any variation along them, dividing by the clamped
    # x_std=1e-8 sends library points to ~1e9 standard deviations from
    # training. The GP then sees every library row as infinitely far away and
    # returns the prior mean → identical predictions for every candidate.
    raw_std = work.std(axis=0, ddof=0).values
    constant_mask = raw_std < 1e-6
    if constant_mask.any():
        const_cols = [feature_cols[i] for i in range(len(feature_cols)) if constant_mask[i]]
        log.info("Dropping %d constant-in-training feature columns (zero info for GP): %s%s",
                 len(const_cols), const_cols[:5], " …" if len(const_cols) > 5 else "")
        keep_cols = [c for c in feature_cols if c not in set(const_cols)]
        work = work[keep_cols]
        feature_cols = keep_cols

    # Cap feature count by max |Pearson r| against any target. With small n
    # (catalyst data: ~60 rows) and ~500 features, the GP kernel collapses
    # (over-fits in-sample, prior elsewhere). Keeping the K features most
    # correlated with the targets gives the kernel something it can actually
    # exploit. K = config.MAX_GP_FEATURES; set to None / 0 to disable.
    max_feats = getattr(config, "MAX_GP_FEATURES", None)
    if max_feats and len(feature_cols) > max_feats:
        Y_for_corr = df[list(targets)].astype(float).values
        X_for_corr = work.values
        # Pearson r per (feature, target). nan_to_num for any residual zero-
        # variance edge case after the constant-drop above.
        x_centered = X_for_corr - X_for_corr.mean(axis=0)
        y_centered = Y_for_corr - Y_for_corr.mean(axis=0)
        x_norm = np.sqrt((x_centered ** 2).sum(axis=0)).clip(min=1e-12)
        y_norm = np.sqrt((y_centered ** 2).sum(axis=0)).clip(min=1e-12)
        # Shape: (n_features, n_targets)
        corr = (x_centered.T @ y_centered) / (x_norm[:, None] * y_norm[None, :])
        score = np.nan_to_num(np.abs(corr), nan=0.0).max(axis=1)

        per_role = bool(getattr(config, "PER_ROLE_FEATURE_CAP", False))
        if per_role:
            keep_idx = _rank_features_per_role(feature_cols, score, max_feats)
            source = "per-role"
        else:
            keep_idx = np.argsort(score)[::-1][:max_feats].tolist()
            source = "global"
        keep_cols = [feature_cols[i] for i in sorted(keep_idx)]
        dropped = len(feature_cols) - len(keep_cols)
        log.info("Capping GP features at %d / %d (dropped %d low-|r|-with-target, "
                 "ranker=%s). Top 5 kept: %s",
                 max_feats, len(feature_cols), dropped, source, keep_cols[:5])
        work = work[keep_cols]
        feature_cols = keep_cols

    X = torch.tensor(work.values, dtype=DTYPE, device=DEVICE)
    Y = torch.tensor(df[targets].values, dtype=DTYPE, device=DEVICE).reshape(-1, len(targets))

    x_mean, x_std = X.mean(dim=0), X.std(dim=0).clamp_min(1e-8)
    y_mean, y_std = Y.mean(dim=0), Y.std(dim=0).clamp_min(1e-8)

    Xs = (X - x_mean) / x_std
    Ys = (Y - y_mean) / y_std

    return XYData(Xs, Ys, feature_cols, list(targets), x_mean, x_std, y_mean, y_std,
                  optional_medians=optional_medians)


def _rank_features_per_role(feature_cols: list[str], score: np.ndarray,
                             max_feats: int) -> list[int]:
    """Split the MAX_GP_FEATURES budget across catalyst roles, then fall back
    to a global |Pearson r| pool to fill any unspent slots.

    Prevents the ranker from picking almost exclusively active_metal features
    when the training set's support / promoter columns have low variance (the
    Pt+Sn+γ-Al2O3-dominant regime): each role gets ~max_feats // n_roles slots
    from its own top-|r| pool; roles that don't have enough candidates return
    their leftovers to a global pool that fills the remaining slots.
    """
    role_names = ["active_metal", "promoter_1", "promoter_2", "support"]
    role_loading = {
        "metal_loading_wt": "active_metal",
        "promoter_1_loading_wt": "promoter_1",
        "promoter_2_loading_wt": "promoter_2",
    }

    def _role_for(col: str) -> str | None:
        for r in role_names:
            if col.startswith(f"{r}_"):
                return r
        return role_loading.get(col)

    role_idx: dict[str, list[int]] = {r: [] for r in role_names}
    unassigned: list[int] = []
    for i, c in enumerate(feature_cols):
        r = _role_for(c)
        (role_idx[r] if r is not None else unassigned).append(i)

    slots_per_role = max_feats // len(role_names)
    kept: list[int] = []
    leftover_pool: list[int] = list(unassigned)
    for r in role_names:
        r_indices = role_idx[r]
        if not r_indices:
            continue
        r_scores = score[r_indices]
        order = np.argsort(r_scores)[::-1]
        take = min(slots_per_role, len(r_indices))
        kept.extend(r_indices[j] for j in order[:take].tolist())
        leftover_pool.extend(r_indices[j] for j in order[take:].tolist())

    remaining = max_feats - len(kept)
    if remaining > 0 and leftover_pool:
        pool_scores = score[leftover_pool]
        pool_order = np.argsort(pool_scores)[::-1]
        for j in pool_order[:remaining].tolist():
            kept.append(leftover_pool[j])

    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate base class
# ─────────────────────────────────────────────────────────────────────────────
class Surrogate(ABC):
    @abstractmethod
    def fit(self, data: XYData) -> None: ...

    @abstractmethod
    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) in STANDARDIZED Y space."""


# ─────────────────────────────────────────────────────────────────────────────
# GP surrogate (single or multi-task)
# ─────────────────────────────────────────────────────────────────────────────
class GPSurrogate(Surrogate):
    def __init__(self):
        self.model = None
        self.mll = None
        self.multi_task = False

    def fit(self, data: XYData) -> None:
        from botorch.models import SingleTaskGP, ModelListGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

        self.multi_task = data.Y.shape[1] > 1
        d = data.X.shape[1]

        def _build_covar():
            # Build a fresh ScaleKernel(MaternKernel) per target — kernel
            # modules are stateful (raw_lengthscale buffer) so they cannot be
            # shared across SingleTaskGPs.
            from gpytorch.kernels import MaternKernel, ScaleKernel
            from gpytorch.priors import GammaPrior
            conc = getattr(config, "GP_LENGTHSCALE_PRIOR_CONCENTRATION", None)
            rate = getattr(config, "GP_LENGTHSCALE_PRIOR_RATE", None)
            if conc is None or rate is None:
                return None  # fall back to BoTorch default
            ls_prior = GammaPrior(float(conc), float(rate))
            base = MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_prior=ls_prior)
            # Initialise lengthscale at the prior mode (≈ (α-1)/β) so the MLL
            # optimizer starts in the "kernel actually reaches library
            # points" regime instead of the BoTorch default (~0.6) where it
            # is stuck near the short-scale local optimum.
            mode = max((float(conc) - 1.0) / float(rate), 0.1)
            with torch.no_grad():
                base.lengthscale = torch.full_like(base.lengthscale, mode)
            return ScaleKernel(base, outputscale_prior=GammaPrior(2.0, 0.15))

        if self.multi_task:
            # Independent SingleTaskGP per target, wrapped in ModelListGP.
            # KroneckerMultiTaskGP explodes memory on hypervolume sampling for
            # large libraries (Kronecker structure of cov matrix). ModelListGP
            # is the BoTorch idiomatic MOBO surrogate and avoids the blowup.
            gps = []
            for t in range(data.Y.shape[1]):
                covar = _build_covar()
                kw = {"covar_module": covar} if covar is not None else {}
                gps.append(SingleTaskGP(data.X, data.Y[:, [t]], **kw).to(DEVICE, DTYPE))
            self.model = ModelListGP(*gps)
            self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        else:
            covar = _build_covar()
            kw = {"covar_module": covar} if covar is not None else {}
            self.model = SingleTaskGP(data.X, data.Y, **kw).to(DEVICE, DTYPE)
            self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)
        log.info("GP fit complete (multi_task=%s, N=%d, T=%d, d=%d).",
                 self.multi_task, data.X.shape[0], data.Y.shape[1], d)

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X)
            mean = posterior.mean
            std = posterior.variance.clamp_min(1e-12).sqrt()
        # ModelListGP posterior.mean is (N, T); SingleTaskGP is (N, 1). Same shape.
        if mean.ndim == 1:
            mean = mean.unsqueeze(-1)
            std = std.unsqueeze(-1)
        return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# SVGP surrogate — Sparse / Variational GP (BoTorch SingleTaskVariationalGP)
# ─────────────────────────────────────────────────────────────────────────────
class SVGPSurrogate(Surrogate):
    """Bridges exact GP (small) and BNN (large). Same Gaussian posterior contract
    as GPSurrogate so qLogNEI / qLogNEHVI plug in unchanged."""

    def __init__(self):
        self.model = None
        self.multi_task = False
        self.num_inducing = int(getattr(config, "SVGP_NUM_INDUCING", 256))
        self.training_iters = int(getattr(config, "SVGP_TRAINING_ITERS", 400))

    def fit(self, data: XYData) -> None:
        from botorch.models import SingleTaskVariationalGP, ModelListGP

        n, t_dim = data.X.shape[0], data.Y.shape[1]
        m = max(2, min(self.num_inducing, n))
        self.multi_task = t_dim > 1

        if self.multi_task:
            heads = []
            for t in range(t_dim):
                gp = SingleTaskVariationalGP(
                    train_X=data.X,
                    train_Y=data.Y[:, [t]],
                    inducing_points=m,
                    learn_inducing_points=True,
                ).to(DEVICE, DTYPE)
                self._train_one(gp, data.X, data.Y[:, [t]])
                heads.append(gp)
            self.model = ModelListGP(*heads)
        else:
            gp = SingleTaskVariationalGP(
                train_X=data.X,
                train_Y=data.Y,
                inducing_points=m,
                learn_inducing_points=True,
            ).to(DEVICE, DTYPE)
            self._train_one(gp, data.X, data.Y)
            self.model = gp

        log.info("SVGP fit complete (multi_task=%s, N=%d, T=%d, M=%d).",
                 self.multi_task, n, t_dim, m)

    def _train_one(self, gp, X: torch.Tensor, Y: torch.Tensor) -> None:
        from gpytorch.mlls import VariationalELBO

        gp.train()
        mll = VariationalELBO(gp.likelihood, gp.model, num_data=X.shape[0])
        optimizer = torch.optim.Adam(gp.parameters(), lr=0.01)
        y_target = Y.squeeze(-1) if (Y.dim() == 2 and Y.shape[-1] == 1) else Y
        for step in range(self.training_iters):
            optimizer.zero_grad()
            output = gp.model(X)
            loss = -mll(output, y_target)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                log.info("SVGP SVI step %d  ELBO loss=%.3f", step, loss.item())

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X)
            mean = posterior.mean
            std = posterior.variance.clamp_min(1e-12).sqrt()
        if mean.ndim == 1:
            mean = mean.unsqueeze(-1)
            std = std.unsqueeze(-1)
        return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian NN surrogate (Pyro variational)
# ─────────────────────────────────────────────────────────────────────────────
class BNNSurrogate(Surrogate):
    """Minimal MLP with variational weights via Pyro."""

    def __init__(self, hidden=(64, 64)):
        self.hidden = hidden
        self.guide = None
        self.model_fn = None
        self.svi = None
        self.in_dim = None
        self.out_dim = None

    def _model(self, X, Y=None):
        import pyro
        import pyro.distributions as dist
        import torch.nn.functional as F

        a = X
        for i, h in enumerate(self.hidden):
            w = pyro.sample(
                f"w{i}",
                dist.Normal(torch.zeros(a.shape[-1], h, device=DEVICE, dtype=DTYPE), 1.0).to_event(2),
            )
            b = pyro.sample(
                f"b{i}",
                dist.Normal(torch.zeros(h, device=DEVICE, dtype=DTYPE), 1.0).to_event(1),
            )
            a = F.relu(a @ w + b)

        w_out = pyro.sample(
            "w_out",
            dist.Normal(torch.zeros(a.shape[-1], self.out_dim, device=DEVICE, dtype=DTYPE), 1.0).to_event(2),
        )
        b_out = pyro.sample(
            "b_out",
            dist.Normal(torch.zeros(self.out_dim, device=DEVICE, dtype=DTYPE), 1.0).to_event(1),
        )
        mean = a @ w_out + b_out

        sigma = pyro.sample("sigma", dist.HalfNormal(torch.tensor(1.0, device=DEVICE, dtype=DTYPE)))
        with pyro.plate("data", X.shape[0]):
            return pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=Y)

    def fit(self, data: XYData) -> None:
        import pyro
        from pyro.infer import SVI, Trace_ELBO
        from pyro.infer.autoguide import AutoNormal
        from pyro.optim import Adam

        pyro.clear_param_store()
        self.in_dim = data.X.shape[1]
        self.out_dim = data.Y.shape[1]
        self.model_fn = self._model
        self.guide = AutoNormal(self.model_fn)
        self.svi = SVI(self.model_fn, self.guide, Adam({"lr": 1e-3}), loss=Trace_ELBO())

        for step in range(config.BNN_TRAINING_ITERS):
            loss = self.svi.step(data.X, data.Y)
            if step % 200 == 0:
                log.info("BNN SVI step %d  ELBO loss=%.3f", step, loss)
        log.info("BNN fit complete (N=%d, in=%d, out=%d).",
                 data.X.shape[0], self.in_dim, self.out_dim)

    def predict(self, X: torch.Tensor,
                n_samples: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Monte-Carlo posterior predictive. See config.BNN_PREDICT_SAMPLES for
        why the default is not small — the sample count is the noise floor of
        every BNN number that reaches the report."""
        from pyro.infer import Predictive

        if n_samples is None:
            n_samples = int(getattr(config, "BNN_PREDICT_SAMPLES", 512))
        predictive = Predictive(self.model_fn, guide=self.guide, num_samples=n_samples,
                                return_sites=("obs",))
        samples = predictive(X)["obs"]   # (S, N, out_dim)
        return samples.mean(0), samples.std(0)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def fit_surrogates(df: pd.DataFrame) -> dict:
    # Pin the global RNGs before any model is built: fit_gpytorch_mll, the SVGP
    # Adam loop and Pyro's AutoNormal guide all initialise from torch's global
    # RNG, so an unseeded run gives a different surrogate every time.
    seed_everything()
    _invalidate_parity_artifacts()
    data = prepare_xy(df)
    out = {"data": data}
    kinds = _active_kinds()

    if "gp" in kinds:
        gp = GPSurrogate()
        gp.fit(data)
        out["gp"] = gp

    if "svgp" in kinds:
        svgp = SVGPSurrogate()
        svgp.fit(data)
        out["svgp"] = svgp

    if "bnn" in kinds:
        bnn = BNNSurrogate()
        bnn.fit(data)
        out["bnn"] = bnn

    # Cache training-set parity predictions so step6_report can draw the parity
    # plot without holding onto the live torch models. One row per training
    # sample, columns: true_<t>, gp_pred_<t>, gp_sd_<t>, [bnn_pred_<t>, bnn_sd_<t>].
    _save_training_predictions(data, out)
    _save_cv_parity_predictions(df)

    ckpt = config.CHECKPOINTS_DIR / "surrogates.pkl"
    # GP state via state_dict; BNN via Pyro param store. Pickle the whole dict for now
    # (works for GP; BNN reload needs pyro.get_param_store() handling — skeleton-only).
    try:
        with open(ckpt, "wb") as f:
            pickle.dump({k: v for k, v in out.items() if k != "data"}, f)
        log.info("Saved surrogates to %s.", ckpt)
    except Exception as e:
        log.warning("Could not pickle surrogates (%s). Skipping checkpoint.", e)

    return out


PARITY_ARTIFACTS = ("training_predictions.parquet", "cv_parity_predictions.parquet")


def _invalidate_parity_artifacts() -> None:
    """Delete last run's parity parquets before refitting.

    step6 picks which parity figure to draw with a bare
    `cv_parity_predictions.parquet.exists()` check and labels it "held-out
    k-fold CV". Without this, running once with CV_FOLDS=5 and then again with
    CV_FOLDS=0 leaves the old file on disk and the second run's report claims
    held-out generalization numbers that the run never computed. The same
    applies when a run crashes between step 4 and step 6, or when the target
    set changes between runs.

    Each artifact is rewritten below if — and only if — it is actually
    produced, so deleting up front makes "file present" mean "this run made
    it".
    """
    for name in PARITY_ARTIFACTS:
        path = config.DATA_DIR / name
        if path.exists():
            try:
                path.unlink()
                log.info("Invalidated stale parity artifact %s.", name)
            except OSError as e:
                log.warning("Could not remove stale %s (%s). step6 may report "
                            "parity numbers from an earlier run.", name, e)


def _save_training_predictions(data: XYData, surrogates: dict) -> None:
    """Predict each fitted surrogate on its OWN training set, un-standardize
    back to the original Y units, and save as data/training_predictions.parquet.
    This is in-sample (not a held-out fit), so it's a basic sanity diagnostic
    — perfect agreement means the model interpolates the data; large scatter
    means even with full training data, the surrogate can't reproduce y."""
    target_cols = data.target_cols
    y_true = (data.Y * data.y_std + data.y_mean).cpu().numpy()  # (N, T)
    rows = {f"true_{t}": y_true[:, i] for i, t in enumerate(target_cols)}

    for kind in ("gp", "svgp", "bnn"):
        surr = surrogates.get(kind)
        if surr is None:
            continue
        try:
            mean_std, sd_std = surr.predict(data.X)             # standardized
            mean = (mean_std * data.y_std + data.y_mean).cpu().numpy()
            sd = (sd_std * data.y_std).cpu().numpy()
            for i, t in enumerate(target_cols):
                rows[f"{kind}_pred_{t}"] = mean[:, i]
                rows[f"{kind}_sd_{t}"] = sd[:, i]
        except Exception as e:
            log.warning("Training-set prediction for %s failed: %s", kind, e)

    df_pred = pd.DataFrame(rows)
    out_path = config.DATA_DIR / "training_predictions.parquet"
    df_pred.to_parquet(out_path)
    log.info("Saved training parity predictions to %s (shape=%s).",
             out_path, df_pred.shape)


def _project_test(test_df: pd.DataFrame, train_data: XYData) -> tuple[torch.Tensor, np.ndarray]:
    """Map a held-out test slice into the train fold's standardized feature
    space, with no leakage. Returns (X_std, Y_true_raw).

    Uses the train fold's OPTIONAL_NUMERIC_FEATURES medians for imputation so
    held-out rows land in the same operating regime as training. Zero-filling
    condition columns here would push non-Pt held-out rows to the standardized
    tail and understate CV skill.
    """
    work = test_df[train_data.feature_cols].copy()
    if train_data.optional_medians:
        cols_here = [c for c in train_data.optional_medians if c in work.columns]
        if cols_here:
            fill_map = {c: train_data.optional_medians[c] for c in cols_here}
            work = work.fillna(fill_map)
    work = work.fillna(0.0).astype(float, copy=False)
    X = torch.tensor(work.values, dtype=DTYPE, device=DEVICE)
    Xs = (X - train_data.x_mean) / train_data.x_std
    Y_true = test_df[train_data.target_cols].values
    return Xs, Y_true


def _save_cv_parity_predictions(df: pd.DataFrame) -> None:
    """Run k-fold CV (config.CV_FOLDS) refitting each surrogate per fold and
    collect HELD-OUT predictions in original Y units. Saves to
    `data/cv_parity_predictions.parquet`. Skipped when CV_FOLDS is falsy.

    Per-fold standardization (no leakage): each fold builds its own XYData from
    the train slice and projects the test slice through that fold's mean/std.
    """
    k = getattr(config, "CV_FOLDS", 0) or 0
    if k < 2:
        log.info("CV parity skipped (CV_FOLDS=%s).", k)
        return

    try:
        from sklearn.model_selection import KFold
    except ImportError:
        log.warning("scikit-learn missing — skipping CV parity.")
        return

    kf = KFold(n_splits=k, shuffle=True, random_state=config.RANDOM_STATE)
    n = len(df)
    target_cols = config.TARGET_COLS

    cols = {f"true_{t}": np.full(n, np.nan) for t in target_cols}
    kinds = _active_kinds()
    for kind in ("gp", "svgp", "bnn"):
        if kind in kinds:
            for t in target_cols:
                cols[f"{kind}_pred_{t}"] = np.full(n, np.nan)
                cols[f"{kind}_sd_{t}"] = np.full(n, np.nan)
    cols["fold"] = np.full(n, -1, dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n)), start=1):
        log.info("CV fold %d/%d (train=%d, test=%d).",
                 fold_idx, k, len(train_idx), len(test_idx))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        try:
            train_data = prepare_xy(train_df)
        except Exception as e:
            log.warning("Fold %d prepare_xy failed (%s); skipping fold.", fold_idx, e)
            continue
        try:
            test_X, test_Y = _project_test(test_df, train_data)
        except KeyError as e:
            log.warning("Fold %d test projection failed (%s); skipping fold.", fold_idx, e)
            continue

        for i, t in enumerate(target_cols):
            cols[f"true_{t}"][test_idx] = test_Y[:, i]
        cols["fold"][test_idx] = fold_idx

        # Refit per surrogate kind. Fold models are discarded; only their
        # held-out predictions persist.
        for kind, ctor in (
            ("gp", GPSurrogate),
            ("svgp", SVGPSurrogate),
            ("bnn", BNNSurrogate),
        ):
            if kind not in kinds:
                continue
            try:
                surr = ctor()
                surr.fit(train_data)
                mean_std, sd_std = surr.predict(test_X)
                mean = (mean_std * train_data.y_std + train_data.y_mean).cpu().numpy()
                sd = (sd_std * train_data.y_std).cpu().numpy()
                for i, t in enumerate(target_cols):
                    cols[f"{kind}_pred_{t}"][test_idx] = mean[:, i]
                    cols[f"{kind}_sd_{t}"][test_idx] = sd[:, i]
            except Exception as e:
                log.warning("Fold %d %s failed (%s).", fold_idx, kind.upper(), e)

    df_cv = pd.DataFrame(cols)
    out_path = config.DATA_DIR / "cv_parity_predictions.parquet"
    df_cv.to_parquet(out_path)
    log.info("Saved %d-fold CV parity predictions to %s (shape=%s).",
             k, out_path, df_cv.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from step1_load import load_dataset
    from step2_featurize import featurize
    df = load_dataset()
    df = featurize(df)
    out = fit_surrogates(df)
    data = out["data"]
    for k, surr in out.items():
        if k == "data":
            continue
        mean, std = surr.predict(data.X[:5])
        print(f"{k}: mean={mean.flatten().tolist()}  std={std.flatten().tolist()}")
