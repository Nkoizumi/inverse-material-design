"""Gradio web UI for the inverse material design pipeline.

Four tabs:
  1. Upload & Config  — upload CSV (or use synthetic), set targets and roles.
  2. Catalyst Library — configure the discrete search space for BO.
  3. Run & Report     — fit surrogates, run BO, render the report inline with
                       interactive Plotly figures and per-figure LLM captions.
  4. Explore          — interactive table of candidates with property drill-down.

Reuses the existing pipeline modules. Pipeline functions still read from
`config` module globals; the UI mutates those globals before each invocation.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import gradio as gr
import pandas as pd

# Make the project importable when running from any working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

import config  # noqa: E402
# Import step modules directly (not via `from pipeline import …`) so the name
# `pipeline` stays free for auto_eda — both projects have a `pipeline/` folder
# and the package-style import would cache OUR pipeline in sys.modules,
# breaking step3_eda's `from pipeline.orchestrator import Orchestrator`.
import step1_load                                                   # noqa: E402
import step2_featurize                                              # noqa: E402
import step3_eda                                                    # noqa: E402
import step4_surrogate                                              # noqa: E402
import step5_inverse                                                # noqa: E402
import step6_report                                                 # noqa: E402

from eda_tab import render_eda_tab                                  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Lookup-driven defaults
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTS_DF = pd.read_csv(config.SUPPORT_LOOKUP_PATH)
METALS_DF = pd.read_csv(config.METAL_LOOKUP_PATH)
ALL_SUPPORTS = SUPPORTS_DF["support"].tolist()
ALL_METALS = METALS_DF["element"].tolist()
# Convention: empty string means "no promoter" in this slot.
PROMOTER_NONE = "(none)"


def _list_str_to_floats(s: str) -> list[float]:
    out = []
    for piece in s.replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.append(float(piece))
        except ValueError:
            continue
    return out


def _parse_promoter_selection(selected: list[str]) -> list[str]:
    """UI uses '(none)' for the empty-string slot; convert back."""
    return ["" if x == PROMOTER_NONE else x for x in selected]


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — upload & config
# ─────────────────────────────────────────────────────────────────────────────
def detect_schema(columns) -> str:
    """Classify a CSV's schema: "role", "fraction" or "single_formula".

    The webui used to read `config.CATALYST_MODE` / `CATALYST_FRACTION_MODE`
    but never SET them, so whichever values `config.py` happened to hold at
    import time decided which featurizer ran. Uploading an atomic-fraction CSV
    therefore went through the role-based featurizer, which skips every missing
    role column and yields a near-empty feature set — a silently useless run
    rather than an error. Conversely, editing config.py to run `acs_pdh` from
    the CLI left the webui stuck in fraction mode.

    Detection mirrors what the featurizers actually require:
      * role   — has the configured role columns (active_metal, support, …)
      * fraction — no role columns, but several bare element-symbol columns
        from CATALYST_FRACTION_ELEMENTS (the ACS PDH set has 20)
      * single_formula — neither; a `composition`/`formula` column
    """
    cols = set(columns)
    role_cols = set(getattr(config, "CATALYST_ROLES", {}).values())
    if role_cols and len(role_cols & cols) >= 2:
        return "role"

    elements = set(getattr(config, "CATALYST_FRACTION_ELEMENTS", []))
    # 5 is comfortably above what a role-based CSV could hit by accident (an
    # element-named column like "Al" would have to appear five times over)
    # and far below the 20 the real fraction datasets carry.
    if len(elements & cols) >= 5:
        return "fraction"

    return "single_formula"


_SCHEMA_LABEL = {
    "role": "role-based catalyst (active_metal / promoters / support)",
    "fraction": "atomic-fraction catalyst (element columns)",
    "single_formula": "single formula (composition column)",
}


def _apply_schema_mode(schema: str) -> None:
    """Set the config flags that decide which featurizer step 2 dispatches to."""
    config.CATALYST_MODE = schema == "role"
    config.CATALYST_FRACTION_MODE = schema == "fraction"


def _load_csv_failure(message: str):
    """Failure return for load_csv, in the exact output arity Gradio expects.

    The click handler declares SIX outputs (preview, target_select,
    load_status, df_state, default_target_state, targets_state). The
    missing-synthetic-CSV path used to return five, so Gradio raised on arity
    instead of showing the message the branch was written to display — the
    error path was itself broken. Building every failure return here keeps the
    two arities from drifting again.
    """
    return (None, gr.update(choices=[], value=[]), message, None, None, [])


def load_csv(file_obj, use_synthetic: bool):
    """Read a CSV (uploaded or synthetic). Returns preview + target-column
    choices + status message + the FULL df (for the EDA tab's state) +
    the default target column name (for the LLM-decisions sub-tab) +
    the initial target selection."""
    if use_synthetic or file_obj is None:
        path = config.DATA_DIR / "synthetic_catalysts.csv"
        if not path.exists():
            return _load_csv_failure(
                f"Synthetic CSV not found at `{path}`. Generate it with "
                "`python scripts/generate_synthetic_catalysts.py`, or untick "
                "the box and upload your own CSV."
            )
        source = f"synthetic ({path.name})"
    else:
        path = Path(file_obj.name)
        source = path.name

    # A malformed or unreadable upload must surface as a message, not as an
    # unhandled exception inside the Gradio callback.
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return _load_csv_failure(f"Could not read `{source}` as CSV: {e}")
    if df.empty or df.shape[1] == 0:
        return _load_csv_failure(f"`{source}` parsed as an empty table.")

    # Auto-derive log-scale versions of wide-dynamic-range targets (e.g.
    # deactivation_rate_h → deactivation_rate_log) so they show up in the
    # target-column picker. Same logic step1_load uses on the canonical load
    # path; mirroring it here keeps Tab 1's dropdown in sync.
    try:
        # Webui convention: import step modules directly, not via
        # `pipeline.` package prefix (avoids namespace clash with auto_eda's
        # `pipeline` package).
        from step1_load import _add_log_derived_targets
        df = _add_log_derived_targets(df)
    except Exception as e:
        log.warning("Log-derived target generation skipped (%s).", e)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Prefer one of the pipeline-configured targets if it's present, otherwise
    # fall back to the first numeric column.
    target_default = None
    for c in config.TARGET_COLS:
        if c in df.columns:
            target_default = c
            break
    if target_default is None and numeric_cols:
        target_default = numeric_cols[0]

    schema = detect_schema(df.columns)
    msg = (
        f"Loaded **{source}** — {len(df)} rows × {df.shape[1]} columns.  \n"
        f"Detected schema: **{_SCHEMA_LABEL[schema]}**.  \n"
        f"Numeric columns (candidates for targets): {', '.join(numeric_cols) or '*(none)*'}"
    )
    if schema == "fraction":
        msg += (
            "\n\n⚠️ Atomic-fraction mode is partially supported in the UI: "
            "Tab 3 (Catalyst Library) does not apply — the BO library is "
            "sampled from your data instead — Tab 5's metal/support filters "
            "become no-ops, and step 6 emits a **minimal report** (candidate "
            "table + summary stats, no LLM narrative or figures). The "
            "surrogate, BO and candidate table all work normally."
        )
    elif schema == "single_formula":
        msg += (
            "\n\n⚠️ No catalyst role columns and no element-fraction columns "
            "found. This will run the single-formula (Magpie) path, which has "
            "no BO library in the UI — use the CLI presets for benchmark "
            "datasets."
        )
    initial_targets = numeric_cols[:2]
    return (
        df.head(10),
        gr.update(choices=df.columns.tolist(), value=initial_targets),
        msg,
        df,
        target_default,
        initial_targets,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — catalyst library size estimate
# ─────────────────────────────────────────────────────────────────────────────
def estimate_library_size(metals, promoters_1, promoters_2,
                          metal_loads_str, promo_loads_str):
    """Quick library-size estimate before BO runs."""
    p1 = _parse_promoter_selection(promoters_1)
    p2 = _parse_promoter_selection(promoters_2)
    m_loads = _list_str_to_floats(metal_loads_str)
    p_loads = _list_str_to_floats(promo_loads_str)

    if not (metals and ALL_SUPPORTS and m_loads):
        return "Pick at least one metal, support and metal loading."

    count = 0
    for metal in metals:
        for sup in ALL_SUPPORTS:
            for p1_choice in p1:
                for p2_choice in p2:
                    if p2_choice != "" and p1_choice == "":
                        continue
                    if p1_choice != "" and p1_choice == p2_choice:
                        continue
                    p1_loads = [0.0] if p1_choice == "" else p_loads
                    p2_loads = [0.0] if p2_choice == "" else p_loads
                    count += len(m_loads) * len(p1_loads) * len(p2_loads)
    return f"**Estimated library size: {count:,} catalyst candidates.**"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — run pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    file_obj, use_synthetic, target_cols, minimize_cols,
    metals, promoters_1, promoters_2,
    metal_loads_str, promo_loads_str,
    surrogate_kind, bo_batch_size, report_llm_model,
    transformed_df, eda_obj,
    progress=gr.Progress(track_tqdm=False),
):
    """Execute the full pipeline end-to-end with the UI-supplied overrides.

    Yields tuples for outputs: (log_text, report_bundle, report_file_path,
    candidates_df). `report_bundle` is the ReportBundle from step6 (used by the
    interleaved @gr.render block); `candidates_df` feeds tab 4.
    """
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        return "\n".join(log_lines)

    try:
        # ── Push overrides into the config module ──────────────────────
        if use_synthetic or file_obj is None:
            config.DATASET_SOURCE = "csv"
            config.CSV_PATH = config.DATA_DIR / "synthetic_catalysts.csv"
        else:
            config.DATASET_SOURCE = "csv"
            config.CSV_PATH = Path(file_obj.name)

        if not target_cols:
            yield log("Pick at least one target column."), None, None, pd.DataFrame()
            return

        # Decide which featurizer step 2 will dispatch to, from the CSV itself.
        # This has to happen BEFORE step 1: CATALYST_FRACTION_MODE changes what
        # load_dataset does (it renames the bracketed headers to snake_case and
        # drops dataset-internal id columns), so detecting after the load would
        # be too late. Peek at the header only.
        try:
            header = pd.read_csv(config.CSV_PATH, nrows=0).columns
        except Exception as e:
            yield log(f"Could not read `{config.CSV_PATH}`: {e}"), None, None, pd.DataFrame()
            return
        schema = detect_schema(header)
        _apply_schema_mode(schema)
        log(f"Schema detected: {_SCHEMA_LABEL[schema]} "
            f"(CATALYST_MODE={config.CATALYST_MODE}, "
            f"CATALYST_FRACTION_MODE={config.CATALYST_FRACTION_MODE}).")
        if schema == "fraction":
            log("Fraction mode: Tab 3's library settings are ignored (the BO "
                "library is sampled from your data), and step 6 will emit the "
                "minimal report.")

        config.TARGET_COLS = list(target_cols)
        minimize_set = set(minimize_cols or [])
        config.OPTIMIZATION_DIRECTIONS = [
            "min" if t in minimize_set else "max" for t in target_cols
        ]
        log(f"Directions: " + ", ".join(
            f"{t}→{d}" for t, d in zip(target_cols, config.OPTIMIZATION_DIRECTIONS)
        ))
        config.SURROGATE_KIND = surrogate_kind
        config.BO_BATCH_SIZE = int(bo_batch_size)
        config.REPORT_LLM_MODEL = report_llm_model

        config.LIBRARY_ACTIVE_METALS = list(metals)
        config.LIBRARY_PROMOTERS_1 = _parse_promoter_selection(promoters_1)
        config.LIBRARY_PROMOTERS_2 = _parse_promoter_selection(promoters_2)
        config.LIBRARY_METAL_LOADINGS = _list_str_to_floats(metal_loads_str)
        config.LIBRARY_PROMO_LOADINGS = _list_str_to_floats(promo_loads_str)

        progress(0.05, desc="Step 1 — load CSV")
        df = step1_load.load_dataset()
        yield log(f"Loaded {len(df)} rows."), None, None, pd.DataFrame()

        progress(0.20, desc="Step 2 — featurize")
        df = step2_featurize.featurize(df)
        yield log(f"Featurized → {df.shape[1]} columns."), None, None, pd.DataFrame()

        # If Tab 2 produced a transformed dataframe, swap it in for the surrogate.
        # Sanity-check: targets present + row count match. If anything looks off,
        # log and fall back to the raw featurized df.
        #
        # CATALYST_MODE skip: in catalyst mode the raw featurized df has ~600
        # physically-meaningful columns (Magpie per role + support/metal
        # lookups + interfacial). Tab 2's generic preprocessing reduces this
        # to a small numerical subset that lines up poorly with the BO
        # library (library lives at the same composition manifold as
        # training, but Tab 2's standardization parameters were fitted on
        # training rows that don't cover the library's catalyst space).
        # Empirically that puts library rows out-of-distribution in the
        # transformed feature space, the GP collapses to a constant
        # posterior, and the BO can't discriminate. The MAX_GP_FEATURES
        # cap + Pearson-r ranking in step4 already does dimensionality
        # reduction more appropriately for the catalyst case.
        transformer = None
        # Both catalyst schemas skip Tab 2's transform. For role-based mode the
        # reason is the one below (library rows land out-of-distribution in the
        # transformed space). For atomic-fraction mode the reason is stronger:
        # _run_catalyst_fraction_discrete has NO transformer branch at all, so
        # the surrogate would fit on transformed columns while the BO library
        # was featurized raw — every library column would miss the training
        # names, `_align_to_training` would zero-fill the lot, and the GP would
        # return the prior mean for every candidate. Before schema detection
        # this was unreachable (CATALYST_MODE was always True in the UI);
        # enabling fraction mode makes it reachable, hence the explicit gate.
        if getattr(config, "CATALYST_MODE", False) or getattr(
            config, "CATALYST_FRACTION_MODE", False
        ):
            yield log(
                "Tab 2 transform skipped: catalyst schema "
                f"({'role-based' if config.CATALYST_MODE else 'atomic-fraction'}). "
                "Catalyst BO uses the raw featurized df (step4 caps features "
                f"at MAX_GP_FEATURES={getattr(config, 'MAX_GP_FEATURES', None)})."
            ), None, None, pd.DataFrame()
        elif transformed_df is not None and isinstance(transformed_df, pd.DataFrame):
            missing = [t for t in target_cols if t not in transformed_df.columns]
            if missing:
                yield log(
                    f"Tab 2 transformed df ignored: missing target(s) {missing}."
                ), None, None, pd.DataFrame()
            elif len(transformed_df) != len(df):
                yield log(
                    f"Tab 2 transformed df ignored: row mismatch "
                    f"(tdf={len(transformed_df)}, raw={len(df)})."
                ), None, None, pd.DataFrame()
            else:
                df = transformed_df
                pipeline_ = getattr(eda_obj, "pipeline_", None) if eda_obj else None
                transformer = pipeline_
                yield log(
                    f"Using Tab 2 transformed features ({df.shape[1]} cols, "
                    f"transformer={'live' if transformer is not None else 'missing'})."
                ), None, None, pd.DataFrame()

        progress(0.40, desc="Step 3 — auto-EDA")
        if eda_obj is not None and getattr(eda_obj, "pipeline_", None) is not None:
            yield log("Step 3 skipped: Tab 2 already produced a fitted EDA pipeline."), \
                None, None, pd.DataFrame()
        else:
            try:
                step3_eda.run_eda(df)
                yield log("Auto-EDA complete."), None, None, pd.DataFrame()
            except Exception as e:
                yield log(f"Auto-EDA skipped ({e})."), None, None, pd.DataFrame()

        progress(0.55, desc="Step 4 — fit surrogates")
        surrogates = step4_surrogate.fit_surrogates(df)
        yield log("Surrogate(s) fit."), None, None, pd.DataFrame()

        progress(0.75, desc="Step 5 — BO / MOBO")
        candidates = step5_inverse.run_inverse(df, surrogates, transformer=transformer)
        yield log(f"BO selected {len(candidates)} candidates."), None, None, pd.DataFrame()

        progress(0.90, desc="Step 6 — report (LLM narrative + per-figure captions)")
        bundle = step6_report.generate_report_bundle(candidates)

        progress(1.0, desc="done")
        yield (
            log(f"Done in {len(log_lines)} steps. Report at {bundle.path}."),
            bundle,
            str(bundle.path),
            candidates,
        )

    except Exception as e:
        tb = traceback.format_exc()
        yield log(f"FAILED: {e}\n\n```\n{tb}\n```"), None, None, pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Interleaved narrative ↔ interactive-figure rendering
# ─────────────────────────────────────────────────────────────────────────────
import re                                                              # noqa: E402

_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _split_narrative_around_figures(narrative: str, figures) -> list[tuple[str, object]]:
    """Walk the LLM narrative; whenever an `![alt](filename.png)` matches a known
    figure, split the narrative there and insert the figure. Returns a list of
    ("text", str) or ("figure", FigureEntry) chunks in order.

    A figure is embedded only at its FIRST mention; subsequent mentions of the
    same filename are stripped to avoid duplicates. Figures never mentioned in
    the narrative are appended at the end.
    """
    if not figures:
        return [("text", narrative)] if narrative else []

    by_filename = {f.png_filename: f for f in figures if f.png_filename}
    embedded_ids: set[str] = set()
    chunks: list[tuple[str, object]] = []
    cursor = 0

    for m in _MD_IMG_RE.finditer(narrative):
        fname = m.group(1).strip()
        fig = by_filename.get(fname)
        if fig is None:
            continue  # not one of our figures — leave the image markdown in place
        # text before this image
        before = narrative[cursor:m.start()]
        if before.strip():
            chunks.append(("text", before))
        if fig.figure_id not in embedded_ids:
            chunks.append(("figure", fig))
            embedded_ids.add(fig.figure_id)
        # else: silently drop the duplicate image reference
        cursor = m.end()

    tail = narrative[cursor:]
    if tail.strip():
        chunks.append(("text", tail))

    # Append any figures that were never referenced in the narrative.
    for fig in figures:
        if fig.figure_id not in embedded_ids:
            chunks.append(("figure", fig))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — explore candidates
# ─────────────────────────────────────────────────────────────────────────────
def _is_fraction_schema(df: pd.DataFrame) -> bool:
    """Detect atomic-fraction candidates: no role columns, has element columns."""
    if df is None or len(df) == 0:
        return False
    if "active_metal" in df.columns:
        return False
    try:
        import config as _cfg
        elems = getattr(_cfg, "CATALYST_FRACTION_ELEMENTS", [])
    except Exception:
        elems = []
    return bool(elems) and any(e in df.columns for e in elems)


def candidate_detail(candidates_df: pd.DataFrame, selected_index: int):
    """Show full property tables for the selected candidate row."""
    if candidates_df is None or len(candidates_df) == 0:
        return "Run the pipeline first."
    if selected_index is None or selected_index < 0 or selected_index >= len(candidates_df):
        return "Pick a row from the candidate table above."
    row = candidates_df.iloc[int(selected_index)].to_dict()

    # Atomic-fraction schema: render composition string + reaction conditions
    # + predictions. Skip the role-based lookup enrichment (not applicable).
    if _is_fraction_schema(candidates_df):
        import config as _cfg
        elems = getattr(_cfg, "CATALYST_FRACTION_ELEMENTS", [])
        supports = set(getattr(_cfg, "CATALYST_FRACTION_SUPPORT_CATIONS", []))
        conds = getattr(_cfg, "CATALYST_FRACTION_CONDITIONS", [])
        # Dominant support = argmax over support cations present in row
        sup_frac = {s: float(row.get(s, 0.0)) for s in supports if s in row}
        sup_str = ""
        if sup_frac:
            sup = max(sup_frac, key=sup_frac.get)
            sup_str = f"{sup}(sup)={sup_frac[sup]:.2f}"
        non_sup = sorted(((e, float(row.get(e, 0.0))) for e in elems
                          if e not in supports and float(row.get(e, 0.0)) > 0.005),
                         key=lambda t: -t[1])
        metal_str = " ".join(f"{e}={v:.3f}" for e, v in non_sup)
        parts = [f"### Candidate {int(selected_index) + 1}",
                 f"- **Composition**: {sup_str} | {metal_str}" if sup_str else f"- **Composition**: {metal_str}"]
        if conds:
            parts.append("\n#### Reaction conditions")
            for c in conds:
                if c in row:
                    parts.append(f"- `{c}` = {row[c]}")
        parts.append("\n#### Predictions")
        for k, v in row.items():
            if k.startswith("pred_"):
                parts.append(f"- `{k}` = {v:.4f}" if isinstance(v, float) else f"- `{k}` = {v}")
        return "\n".join(parts)

    # Role-based schema (original path).
    enriched = step6_report._enrich_candidate(row, SUPPORTS_DF, METALS_DF)

    parts = [f"### Candidate {int(selected_index) + 1}"]
    # The composition line used to read row["_label"], but `_label` is added by
    # step6's report bundle and is never present in candidates.parquet — so the
    # line silently never rendered. Build it from the same formatter step6 uses.
    composition = step6_report._format_catalyst_label(row)
    if composition:
        parts.append(f"- **Composition**: {composition}")
    for label, key in [("Active metal", "active_metal"),
                       ("Promoter 1", "promoter_1"),
                       ("Promoter 2", "promoter_2"),
                       ("Support", "support"),
                       ("Metal loading (wt%)", "metal_loading_wt"),
                       ("Promoter 1 loading (wt%)", "promoter_1_loading_wt"),
                       ("Promoter 2 loading (wt%)", "promoter_2_loading_wt")]:
        if key in row:
            parts.append(f"- **{label}**: {row[key]}")

    for slot in ("support_properties", "metal_properties",
                 "promoter_1_properties", "promoter_2_properties"):
        if slot in enriched:
            parts.append(f"\n#### {slot.replace('_', ' ').title()}")
            for k, v in enriched[slot].items():
                parts.append(f"- `{k}` = {v}")

    parts.append("\n#### Predictions")
    for k, v in row.items():
        if k.startswith("pred_"):
            parts.append(f"- `{k}` = {v:.4f}" if isinstance(v, float) else f"- `{k}` = {v}")

    return "\n".join(parts)


def _sort_ascending_for(col: str) -> bool:
    """Should `col` sort ascending to put the BEST candidates first?

    This tab exists to hand candidates to an experimentalist, so "first row"
    has to mean "most promising". Sorting every column descending — as this
    used to — silently put the WORST candidates on top for any minimize
    target: on the ACS setup, sorting by `pred_deactivation_rate_log`
    (direction "min") ranked the fastest-deactivating catalysts first.

    Rules:
      * a prediction column for a "min" target  -> ascending (lower is better)
      * an uncertainty column (`*_sd`)          -> ascending (tighter is better)
      * everything else                         -> descending
    """
    targets = list(getattr(config, "TARGET_COLS", []))
    directions = list(getattr(config, "OPTIMIZATION_DIRECTIONS", []))
    direction_of = dict(zip(targets, directions))

    if col.endswith("_sd"):
        return True

    # `pred_<target>` from the BO output, or the raw `<target>` column.
    name = col[len("pred_"):] if col.startswith("pred_") else col
    # BNN cross-check columns are `pred_<target>_bnn`.
    if name.endswith("_bnn"):
        name = name[: -len("_bnn")]
    return direction_of.get(name, "max") == "min"


def filter_candidates(candidates_df, metal_filter, support_filter, sort_col):
    if candidates_df is None or len(candidates_df) == 0:
        return pd.DataFrame()
    out = candidates_df.copy()
    # Role-based filters — only applied when columns exist. For fraction-mode
    # candidates the filters degrade to no-ops; the sort still works.
    if metal_filter and "active_metal" in out.columns:
        out = out[out["active_metal"].isin(metal_filter)]
    if support_filter and "support" in out.columns:
        out = out[out["support"].isin(support_filter)]
    if sort_col and sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=_sort_ascending_for(sort_col))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
def get_available_ollama_models() -> list[str]:
    try:
        import ollama
        resp = ollama.list()
        return [m.get("name") or m.get("model") for m in resp.get("models", [])] or [
            config.REPORT_LLM_MODEL
        ]
    except Exception:
        return [config.REPORT_LLM_MODEL]


with gr.Blocks(title="Inverse Material Design") as demo:
    gr.Markdown(
        "# Inverse Material Design\n"
        "Upload a catalyst dataset, configure the design space, and let BO "
        "propose promising new catalysts."
    )

    candidates_state = gr.State(pd.DataFrame())
    bundle_state = gr.State(None)   # ReportBundle from step6 (or None)
    df_state = gr.State(None)       # raw loaded df (drives the EDA tab)
    default_target_state = gr.State(None)  # first numeric col, default for LLM tab
    targets_state = gr.State([])    # ALL targets currently checked in tab 1
    # LLM-decisioning state. These persist across @gr.render rebuilds inside
    # the EDA tab; cleared whenever df_state changes.
    ensemble_state = gr.State(None)
    eda_pipeline_state = gr.State(None)
    transformed_state = gr.State(None)

    with gr.Tabs() as tabs:
        # ── Tab 1: Upload & config ──────────────────────────────────────
        with gr.Tab("1. Upload & Config"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload catalyst CSV",
                        file_types=[".csv"],
                    )
                    use_synth = gr.Checkbox(
                        label="Use built-in synthetic catalyst dataset",
                        value=True,
                    )
                    load_btn = gr.Button("Load & inspect", variant="primary")
                with gr.Column(scale=3):
                    load_status = gr.Markdown()
                    preview = gr.Dataframe(label="Preview (first 10 rows)")
                    target_select = gr.CheckboxGroup(
                        label="Target columns (pick 1 for BO; 2 for MOBO)",
                    )
                    minimize_select = gr.CheckboxGroup(
                        label=("Targets to MINIMIZE (others maximized). "
                               "E.g. tick a deactivation-rate column."),
                        choices=[], value=[],
                    )

            load_btn.click(
                load_csv,
                inputs=[file_input, use_synth],
                outputs=[preview, target_select, load_status, df_state,
                         default_target_state, targets_state],
            )

            # Keep targets_state synced with manual checkbox edits, and keep
            # the minimize-picker's choices in lockstep with the targets picked.
            def _sync_targets(picked):
                picked = picked or []
                return picked, gr.update(
                    choices=picked,
                    value=[v for v in (picked or []) if False],  # reset on change
                )

            target_select.change(
                _sync_targets,
                inputs=[target_select],
                outputs=[targets_state, minimize_select],
            )

        # ── Tab 2: Data exploration (EDA) ───────────────────────────────
        with gr.Tab("2. Data Exploration"):
            render_eda_tab(
                df_state, default_target_state, targets_state,
                ensemble_state, eda_pipeline_state, transformed_state,
            )

        # ── Tab 3: Library config ───────────────────────────────────────
        with gr.Tab("3. Catalyst Library"):
            gr.Markdown(
                "Configure the discrete search space for BO. "
                "Empty `(none)` in a promoter slot disables it for that combination."
            )
            with gr.Row():
                metals_in = gr.CheckboxGroup(
                    choices=ALL_METALS,
                    value=config.LIBRARY_ACTIVE_METALS,
                    label="Active metals",
                )
            with gr.Row():
                promo1_in = gr.CheckboxGroup(
                    choices=[PROMOTER_NONE] + ALL_METALS,
                    value=[PROMOTER_NONE if x == "" else x for x in config.LIBRARY_PROMOTERS_1],
                    label="Promoter 1 (main promoter)",
                )
                promo2_in = gr.CheckboxGroup(
                    choices=[PROMOTER_NONE] + ALL_METALS,
                    value=[PROMOTER_NONE if x == "" else x for x in config.LIBRARY_PROMOTERS_2],
                    label="Promoter 2 (dopant)",
                )
            with gr.Row():
                metal_loads_in = gr.Textbox(
                    value=", ".join(str(x) for x in config.LIBRARY_METAL_LOADINGS),
                    label="Metal loading values (wt%, comma-separated)",
                )
                promo_loads_in = gr.Textbox(
                    value=", ".join(str(x) for x in config.LIBRARY_PROMO_LOADINGS),
                    label="Promoter loading values (wt%, comma-separated)",
                )
            gr.Markdown(f"Supports are read from `{config.SUPPORT_LOOKUP_PATH.name}` "
                        f"({len(ALL_SUPPORTS)} entries) and used as-is.")
            lib_size_btn = gr.Button("Estimate library size")
            lib_size_out = gr.Markdown()
            lib_size_btn.click(
                estimate_library_size,
                inputs=[metals_in, promo1_in, promo2_in,
                        metal_loads_in, promo_loads_in],
                outputs=[lib_size_out],
            )

        # ── Tab 4: Run + Report ─────────────────────────────────────────
        with gr.Tab("4. Run & Report"):
            with gr.Row():
                surrogate_in = gr.Dropdown(
                    choices=["gp", "svgp", "bnn", "gp,svgp", "both", "all"],
                    value=config.SURROGATE_KIND,
                    label=("Surrogate kind  "
                           "(gp=small / svgp=mid / bnn=large; "
                           "both=gp+bnn, all=gp+svgp+bnn)"),
                )
                batch_in = gr.Slider(
                    minimum=1, maximum=50,
                    step=1, value=config.BO_BATCH_SIZE,
                    label="BO batch size",
                )
                llm_models_in = gr.Dropdown(
                    choices=get_available_ollama_models(),
                    value=config.REPORT_LLM_MODEL,
                    label="Ollama model for the report narrative",
                )
            run_btn = gr.Button("Run pipeline", variant="primary")
            run_log = gr.Textbox(label="Progress log", lines=10, max_lines=20)
            report_file = gr.File(label="Download the markdown report")

            gr.Markdown("---\n### Report")

            @gr.render(inputs=[bundle_state])
            def _render_report(bundle):
                if bundle is None:
                    gr.Markdown(
                        "*No report yet — configure the pipeline above and "
                        "click **Run pipeline**.*"
                    )
                    return

                # Header
                gr.Markdown(
                    f"**Generated** {bundle.ts}  \n"
                    f"**Dataset**: `{bundle.dataset_source}` "
                    f"({bundle.dataset_name})  \n"
                    f"**Targets**: {bundle.targets}  \n"
                    f"**Samples used**: {bundle.n_samples}"
                )

                chunks = _split_narrative_around_figures(
                    bundle.narrative_body, bundle.figures,
                )
                if not chunks:
                    gr.Markdown("*(empty narrative)*")
                    return

                for kind, content in chunks:
                    if kind == "text":
                        gr.Markdown(content)
                    else:
                        fig = content
                        with gr.Group():
                            gr.Markdown(f"#### {fig.title}")
                            if fig.plotly_fig is not None:
                                gr.Plot(value=fig.plotly_fig, show_label=False)
                            elif fig.png_filename:
                                gr.Image(
                                    value=str(config.REPORTS_DIR / fig.png_filename),
                                    show_label=False,
                                )
                            caption = fig.llm_caption or fig.default_description
                            gr.Markdown(f"*{caption}*")

                per_cand = getattr(bundle, "per_candidate_section", "")
                if per_cand:
                    gr.Markdown(
                        "---\n## Per-candidate notes\n\n"
                        "*Rendered deterministically from the property "
                        "lookups; numbers cited come straight from "
                        "`data/lookups/{support,metal}_properties.csv`.*\n\n"
                        + per_cand
                    )

            run_btn.click(
                run_pipeline,
                inputs=[
                    file_input, use_synth, target_select, minimize_select,
                    metals_in, promo1_in, promo2_in,
                    metal_loads_in, promo_loads_in,
                    surrogate_in, batch_in, llm_models_in,
                    transformed_state, eda_pipeline_state,
                ],
                outputs=[run_log, bundle_state, report_file, candidates_state],
            )

        # ── Tab 5: Explore ──────────────────────────────────────────────
        with gr.Tab("5. Explore candidates"):
            gr.Markdown(
                "Filter and sort the candidates from the last run. "
                "Type a row index below the table to see the full property tables."
            )
            with gr.Row():
                metal_filter = gr.CheckboxGroup(
                    choices=ALL_METALS, label="Filter by active metal",
                )
                support_filter = gr.CheckboxGroup(
                    choices=ALL_SUPPORTS, label="Filter by support",
                )
                sort_in = gr.Dropdown(
                    choices=[],
                    label=("Sort by column (best first — minimize targets and "
                           "σ columns sort ascending)"),
                )
            filter_btn = gr.Button("Apply filters")
            candidate_table = gr.Dataframe(label="Candidates", interactive=False)
            with gr.Row():
                row_idx = gr.Number(label="Row index to inspect (0-based)", value=0)
                detail_btn = gr.Button("Show details")
            candidate_detail_md = gr.Markdown()

            # Refresh the sort dropdown whenever the candidate state changes.
            # Schema-agnostic: pick every prediction column plus the training
            # targets currently configured.
            def _update_filters(df):
                if df is None or len(df) == 0:
                    return gr.update(choices=[])
                target_cols = tuple(getattr(config, "TARGET_COLS", []))
                cols = [c for c in df.columns
                        if c.startswith("pred_") or c in target_cols]
                return gr.update(choices=cols)

            candidates_state.change(
                _update_filters, inputs=[candidates_state], outputs=[sort_in],
            )

            filter_btn.click(
                filter_candidates,
                inputs=[candidates_state, metal_filter, support_filter, sort_in],
                outputs=[candidate_table],
            )

            detail_btn.click(
                candidate_detail,
                inputs=[candidates_state, row_idx],
                outputs=[candidate_detail_md],
            )


if __name__ == "__main__":
    import os
    # Use the first free port in [7860, 7869]; lets you start a second instance
    # without hunting down the first one.
    port_env = os.environ.get("GRADIO_SERVER_PORT")
    if port_env:
        port = int(port_env)
    else:
        import socket
        port = 7860
        for candidate in range(7860, 7870):
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", candidate))
                    port = candidate
                    break
                except OSError:
                    continue
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_error=True,
    )
