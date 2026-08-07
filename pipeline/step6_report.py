"""Step 6: scientific report via a local LLM (Ollama).

Renders a Jinja2 template with the discovered candidates + surrogate predictions
+ optional EDA insights, then asks Ollama for a narrative explaining chemical
viability. Output is a Markdown report under reports/ AND a ReportBundle
returned in-memory for the webui (which renders the narrative interleaved with
interactive Plotly figures and per-figure LLM captions).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import config

log = logging.getLogger(__name__)


@dataclass
class FigureEntry:
    """One report figure with both static + interactive renderings."""
    figure_id: str                          # "pareto" | "feature_importance" | "gp_vs_bnn" | "candidate_heatmap"
    title: str
    alt: str
    default_description: str                # fallback caption if the LLM can't write one
    png_filename: str | None = None         # filename inside REPORTS_DIR (for the saved MD)
    plotly_fig: Any | None = None           # plotly.graph_objects.Figure for the webui
    llm_caption: str = ""                   # LLM-written interpretation (filled after the call)


@dataclass
class ReportBundle:
    """Everything the CLI saves to disk + everything the webui needs to render."""
    path: Path
    ts: str
    dataset_source: str
    dataset_name: str
    targets: list[str]
    n_samples: int | None
    eda_summary_str: str | None
    candidates_enriched: list[dict]
    has_bnn: bool
    narrative_body: str
    per_candidate_section: str = ""  # deterministic per-candidate bullets
    figures: list[FigureEntry] = field(default_factory=list)


_TEMPLATE = """{%- macro fmt(v) -%}
{%- if v is none or v != v -%}—{%- else -%}{{ "%.3f"|format(v) }}{%- endif -%}
{%- endmacro -%}
{%- macro fmt_delta(a, b) -%}
{%- if a is none or b is none or a != a or b != b -%}—{%- else -%}{{ "%.3f"|format(b - a) }}{%- endif -%}
{%- endmacro -%}
# Inverse Material Design Report

*Generated {{ ts }}*

## Dataset
- Source: `{{ dataset_source }}` ({{ dataset_name }})
- Targets: {{ targets }}
- Samples used: {{ n_samples }}
{% if schema_note %}
{{ schema_note }}
{% endif %}
## EDA highlights
{% if eda_summary %}
```json
{{ eda_summary }}
```
{% else %}
*(auto-EDA was not run.)*
{% endif %}

## Top {{ top_k }} candidates (sorted by predicted {{ targets[0] }})

These rows are rendered **directly from the BO output**, not via the LLM. Numbers here are authoritative — any value in the prose below that conflicts with this table is an LLM error.

| # | {{ label_kind }} | Family | {% for t in targets %}{{ t }} (pred ± σ) | {% endfor %}Confidence |
|---|---|---|{% for t in targets %}---|{% endfor %}---|
{% for row in candidates %}| {{ loop.index }} | {{ row['_label'] }} | {{ row['_family'] }} | {% for t in targets %}{{ fmt(row.get('pred_' + t)) }} ± {{ fmt(row.get('pred_' + t + '_sd')) }} | {% endfor %}{{ row['_confidence'] }} |
{% endfor %}
{% if conditions_section %}
### Process conditions selected for each candidate

The BO searched composition **and** reaction conditions jointly, so a candidate is only fully specified with the row below. Rendered directly from the BO output.

{{ conditions_section }}
{% endif %}
{% if has_bnn %}

### GP vs Bayesian-NN cross-check

The GP drives the BO acquisition. The Bayesian NN is an independent surrogate; agreement between the two strengthens confidence in a candidate.

Both σ are **predictive** standard deviations — they include measurement noise, so they answer "how far off would a lab measurement of this candidate be", not "how uncertain is the model about the latent value". The two columns are therefore directly comparable; a difference between them is genuine model disagreement, not a difference in what σ means.

| # | {{ label_kind }} | {% for t in targets %}GP {{ t }} | BNN {{ t }} | Δ | {% endfor %}
|---|---|{% for t in targets %}---|---|---|{% endfor %}
{% for row in candidates %}| {{ loop.index }} | {{ row['_label'] }} | {% for t in targets %}{{ fmt(row.get('pred_' + t)) }} ± {{ fmt(row.get('pred_' + t + '_sd')) }} | {{ fmt(row.get('pred_' + t + '_bnn')) }} ± {{ fmt(row.get('pred_' + t + '_bnn_sd')) }} | {{ fmt_delta(row.get('pred_' + t), row.get('pred_' + t + '_bnn')) }} | {% endfor %}
{% endfor %}
{% endif %}

## Why these materials are promising

{{ llm_narrative }}

## Per-candidate notes

These bullets are rendered **deterministically from the property lookups**, not the LLM. Numbers cited here come straight from `data/lookups/{support,metal}_properties.csv`.

{{ per_candidate_section }}

## Figure interpretations (LLM)

{% for fig in figures %}### {{ fig.title }}

![{{ fig.alt }}]({{ fig.png_filename }})

{{ fig.llm_caption or fig.default_description }}

{% endfor %}
## Candidate properties (verbatim from lookup)

These tables are rendered **directly from the property lookups**, not via the LLM.
Use them to cross-check any numeric claim in the narrative above.

{% for row in candidates %}
### Candidate {{ loop.index }} — {{ row['_label'] }}

{% if row.get('support_properties') %}**Support ({{ row['support'] }})**

| property | value |
|---|---|
{% for k, v in row['support_properties'].items() %}| {{ k }} | {{ v }} |
{% endfor %}
{% endif %}
{% if row.get('metal_properties') %}
**{{ row['_metal_heading'] }}**

| property | value |
|---|---|
{% for k, v in row['metal_properties'].items() %}| {{ k }} | {{ v }} |
{% endfor %}
{% endif %}
{% if row.get('promoter_1_properties') %}
**Promoter 1 ({{ row['promoter_1'] }})**

| property | value |
|---|---|
{% for k, v in row['promoter_1_properties'].items() %}| {{ k }} | {{ v }} |
{% endfor %}
{% endif %}
{% if row.get('promoter_2_properties') %}
**Promoter 2 ({{ row['promoter_2'] }})**

| property | value |
|---|---|
{% for k, v in row['promoter_2_properties'].items() %}| {{ k }} | {{ v }} |
{% endfor %}
{% endif %}
{% endfor %}
"""


def _weighted_props(components: dict[str, float], lookup: pd.DataFrame,
                    key_col: str, skip: set[str]) -> dict | None:
    """Composition-weighted lookup values for already-resolved components,
    rounded and stripped of NaNs. Shared by the role-based and atomic-fraction
    enrichment paths so both quote the property numbers the same way."""
    from catalyst_features import weighted_lookup

    if not components:
        return None
    numeric_cols = [c for c in lookup.columns
                    if c != key_col and c not in skip
                    and pd.api.types.is_numeric_dtype(lookup[c])]
    cat_cols = [c for c in lookup.columns
                if c != key_col and c not in skip
                and not pd.api.types.is_numeric_dtype(lookup[c])]
    feats = weighted_lookup(components, lookup, key_col, numeric_cols, cat_cols)
    props = {
        k: (round(v, 3) if isinstance(v, float) else v)
        for k, v in feats.items() if v is not None and not (isinstance(v, float) and pd.isna(v))
    }
    return props or None


def _enrich_candidate(row: dict, sup_lookup: pd.DataFrame,
                      metal_lookup: pd.DataFrame) -> dict:
    """Attach the actual lookup-derived support/metal/promoter features for one
    candidate. The LLM is instructed to cite ONLY these numbers when quoting
    properties, which prevents hallucinated values like '0.49 mmol/g' for ZnO
    when the lookup says 0.15."""
    from catalyst_features import parse_components

    enriched = dict(row)
    enriched["_schema"] = "role"

    def _props_from_lookup(cell: str | None, lookup: pd.DataFrame,
                            key_col: str, skip: set[str]) -> dict | None:
        if not cell or str(cell).strip() == "":
            return None
        keys = set(lookup[key_col].astype(str))
        components = parse_components(str(cell), keys)
        return _weighted_props(components, lookup, key_col, skip)

    sup_props = _props_from_lookup(
        row.get("support"), sup_lookup, "support", skip={"pymatgen_formula"},
    )
    if sup_props:
        enriched["support_properties"] = sup_props

    metal_props = _props_from_lookup(
        row.get("active_metal"), metal_lookup, "element", skip=set(),
    )
    if metal_props:
        enriched["metal_properties"] = metal_props

    for slot in ("promoter_1", "promoter_2"):
        props = _props_from_lookup(
            row.get(slot), metal_lookup, "element", skip=set(),
        )
        if props:
            enriched[f"{slot}_properties"] = props

    enriched["_metal_heading"] = f"Active metal ({row.get('active_metal')})"
    return enriched


def _build_family(row: dict, sup_lookup: pd.DataFrame | None) -> str:
    """Chemistry-style descriptor used in the deterministic comparison table.

    Example: ``Pt-Sn / amphoteric oxide``. Falls back gracefully if support is
    blank (bulk catalyst) or unknown to the lookup.
    """
    m = str(row.get("active_metal") or "").strip()
    p1 = str(row.get("promoter_1") or "").strip()
    metal_part = f"{m}-{p1}" if (m and p1) else (m or p1)

    support = str(row.get("support") or "").strip()
    if not support or support.lower() in ("nan", "none"):
        return f"{metal_part} (bulk)" if metal_part else "bulk"

    keys = [p.rsplit(":", 1)[0].strip() for p in support.split("+")] if "+" in support else [support]
    classes: list[str] = []
    if sup_lookup is not None and "oxide_class" in sup_lookup.columns:
        for k in keys:
            match = sup_lookup[sup_lookup["support"].astype(str) == k]
            if len(match) == 1:
                cls = str(match["oxide_class"].iloc[0]).strip()
                if cls and cls != "nan":
                    classes.append(cls)
    oxide_cls = "/".join(sorted(set(classes))) if classes else "oxide"
    descriptor = f"{oxide_cls} oxide" if not oxide_cls.endswith("oxide") else oxide_cls
    return f"{metal_part} / {descriptor}" if metal_part else descriptor


def _build_confidence(row: dict, targets: list[str], y_stds: dict[str, float]) -> str:
    """Bucket predictive uncertainty as 'high' / 'medium' / 'low' confidence.

    Normalizes each target's σ by that target's training-Y std, then averages.
    σ̄/σ_train < 0.5 → high confidence; < 1.0 → medium; otherwise low (the
    surrogate is at or beyond its prior variance).

    The σ fed in here is the PREDICTIVE standard deviation from every
    surrogate — it includes observation noise (see Surrogate in
    step4_surrogate). That is what makes the ratio meaningful: σ_train is the
    spread of measured values, so comparing a predictive σ against it asks
    "would this prediction be tighter than the data's own scatter?". The GP
    previously supplied its latent sd here, which understated the ratio by
    ~1.8x on the bundled synthetic set and was not comparable with the BNN's.
    """
    sigmas_norm: list[float] = []
    for t in targets:
        sd = row.get(f"pred_{t}_sd")
        std = y_stds.get(t)
        if sd is None or std is None or std == 0 or pd.isna(sd):
            continue
        sigmas_norm.append(float(sd) / float(std))
    if not sigmas_norm:
        return "—"
    mean_norm = sum(sigmas_norm) / len(sigmas_norm)
    if mean_norm < 0.5:
        return "high"
    if mean_norm < 1.0:
        return "medium"
    return "low"


def _is_blank(v: Any) -> bool:
    """True for None, empty string, the literal string 'nan', or a NaN float."""
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    s = str(v).strip()
    return s == "" or s.lower() == "nan"


def _format_loading(load: Any) -> str | None:
    """Format a wt% loading as '5%' / '0.3%'. None if missing, NaN, or zero."""
    if _is_blank(load):
        return None
    try:
        val = float(load)
    except (TypeError, ValueError):
        return None
    if pd.isna(val) or val == 0:
        return None
    return f"{val:.2g}%"


def _format_catalyst_label(row: dict) -> str:
    if "active_metal" in row:
        parts = []
        for name_key, load_key in (
            ("active_metal", "metal_loading_wt"),
            ("promoter_1", "promoter_1_loading_wt"),
            ("promoter_2", "promoter_2_loading_wt"),
        ):
            name = row.get(name_key)
            if _is_blank(name):
                continue
            load_str = _format_loading(row.get(load_key))
            parts.append(f"{load_str} {name}" if load_str else str(name))
        s = row.get("support")
        prefix = " + ".join(parts)
        if _is_blank(s):
            return prefix
        return f"{prefix} / {s}" if prefix else str(s)
    # Composition-only mode (matbench, steels library): "Base-El1Pct-El2Pct-…"
    # listing the dominant element followed by up to 3 alloying elements with
    # atomic % rounded. Falls back to the raw formula if parsing fails.
    comp_str = str(row.get("composition_str") or row.get("nearest_material") or "")
    if not comp_str:
        return ""
    try:
        from pymatgen.core.composition import Composition
        items = sorted(Composition(comp_str).fractional_composition.as_dict().items(),
                       key=lambda kv: -kv[1])
        base_el, _ = items[0]
        alloying = [(el, f) for el, f in items[1:] if f >= 0.005][:3]
        parts = [base_el]
        for el, f in alloying:
            pct = f * 100
            parts.append(f"{el}{int(round(pct))}" if pct >= 2 else f"{el}{pct:.1f}")
        return "-".join(parts)
    except Exception:
        return comp_str


# ─────────────────────────────────────────────────────────────────────────────
# Atomic-fraction schema
#
# These candidates have no role columns: each is a flat atomic-fraction vector
# over config.CATALYST_FRACTION_ELEMENTS plus reaction conditions. Everything
# below reconstructs the SAME support / metal-phase decomposition that
# catalyst_fraction_features.featurize_fractions fed the surrogate — the report
# must not describe a catalyst the model never saw.
# ─────────────────────────────────────────────────────────────────────────────
_MIN_METAL_DISPLAY = 0.005   # below this an element is noise, not a component

_FRACTION_SCHEMA_NOTE = (
    "> **Schema: atomic fractions.** Each catalyst is a fraction vector over a "
    "fixed element panel plus reaction conditions — the dataset assigns no "
    "`active metal` / `promoter` / `support` roles. This report reconstructs "
    "the support the way the featurizer did (dominant Al/Si/Zr cation → its "
    "oxide) and treats every remaining element as one composition-weighted "
    "*metal phase*. Element-level roles are therefore NOT asserted anywhere "
    "below; the property numbers are the same weighted lookups the surrogate "
    "was fit on."
)


def _fraction_of(row: dict, el: str) -> float:
    """Atomic fraction of `el`, 0.0 when absent, non-numeric, or NaN."""
    v = row.get(el)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(f) else f


def _fraction_support(row: dict, support_cations: list[str] | None = None,
                      oxide_map: dict[str, str] | None = None,
                      ) -> tuple[str | None, str | None, float]:
    """`(support cation, reconstructed oxide key, support atomic fraction)`.

    Deliberately re-uses catalyst_fraction_features.MIN_SUPPORT_FRACTION rather
    than just taking the dominant cation: below that cutoff the featurizer
    assigned NO support and emitted no support_phys_* block, so naming an oxide
    here would attach lookup numbers to a candidate the surrogate scored
    without them.

    `support_cations` / `oxide_map` default to the ambient config; the webui
    passes the panel captured with the run instead, so a candidate table stays
    labelled by the run that produced it.
    """
    from catalyst_fraction_features import MIN_SUPPORT_FRACTION

    cations = (support_cations if support_cations is not None
               else getattr(config, "CATALYST_FRACTION_SUPPORT_CATIONS", []))
    oxide_map = (oxide_map if oxide_map is not None
                 else getattr(config, "CATALYST_FRACTION_SUPPORT_OXIDE_MAP", {}))
    present = {c: _fraction_of(row, c) for c in cations if c in row}
    if not present:
        return None, None, 0.0
    cation = max(present, key=present.get)
    frac = present[cation]
    if frac < MIN_SUPPORT_FRACTION:
        return None, None, frac
    return cation, oxide_map.get(cation), frac


def _fraction_metal_weights(row: dict) -> dict[str, float]:
    """Renormalized `{element: weight}` over the non-support panel elements.

    Mirrors catalyst_fraction_features._join_metal_lookup_weighted (every
    element with fraction > 0, renormalized to sum to 1) so the
    `metal_properties` quoted in the report are numerically the metal_phys_*
    features the surrogate was fit on.
    """
    elems = getattr(config, "CATALYST_FRACTION_ELEMENTS", [])
    supports = set(getattr(config, "CATALYST_FRACTION_SUPPORT_CATIONS", []))
    weights = {e: _fraction_of(row, e) for e in elems
               if e not in supports and e in row}
    weights = {e: v for e, v in weights.items() if v > 0}
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {e: v / total for e, v in weights.items()}


def _fraction_metal_formula(weights: dict[str, float], max_terms: int = 4) -> str:
    """`Mg0.83Fe0.17` — metal-phase formula from renormalized weights."""
    items = sorted(weights.items(), key=lambda kv: -kv[1])[:max_terms]
    return "".join(f"{e}{w:.2f}" for e, w in items)


def _fraction_composition_string(
    row: dict,
    element_cols: list[str] | None = None,
    support_cations: list[str] | None = None,
    oxide_map: dict[str, str] | None = None,
) -> str:
    """`Ga=0.029 Mo=0.009 Pt=0.0008 / gamma-Al2O3 (Al 0.96)` — one-line label.

    Metal-phase elements first, support last, mirroring the role-mode label
    (`5% Pt + 0.3% Sn / gamma-Al2O3`) so both schemas read the same way.

    Two things this must not do. It must not contain a `|`: the label lands in
    markdown table cells, and a pipe silently splits the row into extra
    columns. And it must not apply a minimum-fraction cutoff to the metals —
    the noble metal in a PDH catalyst sits around 1e-3 atomic fraction, so a
    0.005 threshold would drop the very element the catalyst is named for.
    Fractions are printed with `%.3g`, which keeps 0.00084 legible.
    """
    elems = (element_cols if element_cols is not None
             else getattr(config, "CATALYST_FRACTION_ELEMENTS", []))
    # Keep the configured order — `_fraction_support` breaks a tie between two
    # equally-abundant cations by it, and a set round-trip would make that
    # choice vary between processes.
    support_list = list(support_cations if support_cations is not None
                        else getattr(config, "CATALYST_FRACTION_SUPPORT_CATIONS", []))
    supports = set(support_list)
    cation, oxide, sup_frac = _fraction_support(row, support_list, oxide_map)

    non_sup = sorted(((e, _fraction_of(row, e)) for e in elems
                      if e not in supports and _fraction_of(row, e) > 0),
                     key=lambda t: -t[1])
    metal_str = " ".join(f"{e}={v:.3g}" for e, v in non_sup)

    if not cation:
        return metal_str or "(no assigned composition)"
    sup_str = f"{oxide or cation} ({cation} {sup_frac:.2f})"
    return f"{metal_str} / {sup_str}" if metal_str else f"bare {sup_str}"


def _enrich_fraction_candidate(row: dict, sup_lookup: pd.DataFrame | None,
                               metal_lookup: pd.DataFrame | None) -> dict:
    """Atomic-fraction analogue of `_enrich_candidate`.

    Produces the same keys the downstream machinery reads — `support`,
    `support_properties`, `metal_properties`, `_metal_heading` — so the
    template, the property heatmap and the prompt builder need no special
    case beyond wording. What it does NOT produce is `active_metal` /
    `promoter_*`: this schema carries no role labels and inventing one would
    hand the LLM a fact the dataset does not contain.
    """
    enriched = dict(row)
    enriched["_schema"] = "fraction"

    cation, oxide, sup_frac = _fraction_support(row)
    if oxide:
        enriched["support"] = oxide
        enriched["support_cation"] = cation
        enriched["support_fraction"] = round(sup_frac, 4)
        if sup_lookup is not None:
            props = _weighted_props({oxide: 1.0}, sup_lookup, "support",
                                    skip={"pymatgen_formula"})
            if props:
                enriched["support_properties"] = props

    weights = _fraction_metal_weights(row)
    if weights:
        formula = _fraction_metal_formula(weights)
        enriched["metal_phase"] = {e: round(w, 3) for e, w in
                                   sorted(weights.items(), key=lambda kv: -kv[1])}
        enriched["metal_phase_formula"] = formula
        enriched["_metal_heading"] = f"Metal phase ({formula}, composition-weighted)"
        if metal_lookup is not None:
            props = _weighted_props(weights, metal_lookup, "element", skip=set())
            if props:
                enriched["metal_properties"] = props
    else:
        enriched["_metal_heading"] = "Metal phase (none — unpromoted support)"

    return enriched


def _build_fraction_family(row: dict, sup_lookup: pd.DataFrame | None) -> str:
    """Chemistry-style descriptor for a fraction candidate, e.g.
    ``Mg-Fe / amphoteric oxide``. Mirrors `_build_family`'s output shape so the
    comparison table reads the same across both schemas — but the metal part is
    the two largest non-support elements, not a role assignment."""
    elems = getattr(config, "CATALYST_FRACTION_ELEMENTS", [])
    supports = set(getattr(config, "CATALYST_FRACTION_SUPPORT_CATIONS", []))
    top = sorted(((e, _fraction_of(row, e)) for e in elems
                  if e not in supports and _fraction_of(row, e) > _MIN_METAL_DISPLAY),
                 key=lambda t: -t[1])[:2]
    metal_part = "-".join(e for e, _ in top)

    support = row.get("support")
    if not support:
        return f"{metal_part} (unsupported)" if metal_part else "unsupported"

    oxide_cls = ""
    if sup_lookup is not None and "oxide_class" in sup_lookup.columns:
        match = sup_lookup[sup_lookup["support"].astype(str) == str(support)]
        if len(match) == 1:
            cls = str(match["oxide_class"].iloc[0]).strip()
            if cls and cls != "nan":
                oxide_cls = cls
    descriptor = f"{oxide_cls} oxide" if oxide_cls else str(support)
    return f"{metal_part} / {descriptor}" if metal_part else f"bare {descriptor}"


_PRETREATMENT_LABELS = {1.0: "oxidation", -1.0: "reduction"}


def _format_condition(col: str, value: Any) -> str:
    """Render one reaction-condition cell. `pretreatment` is a ±1 code in the
    ACS schema, so print what it means rather than the bare number."""
    if _is_blank(value):
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if col == "pretreatment":
        label = _PRETREATMENT_LABELS.get(v)
        return f"{label} ({v:+.0f})" if label else f"{v:g}"
    return f"{v:g}"


def _render_conditions_section(records: list[dict]) -> str:
    """Markdown table of the reaction conditions the BO picked per candidate.

    Fraction-mode BO searches composition AND process conditions jointly, so
    omitting them (as the old minimal report did) leaves every candidate
    under-specified — two rows with identical compositions can differ only in
    calcination temperature. Empty string when the schema has no condition
    columns, which drops the whole section from the template.
    """
    cols = [c for c in getattr(config, "CATALYST_FRACTION_CONDITIONS", [])
            if any(c in r for r in records)]
    if not cols or not records:
        return ""
    header = "| # | Candidate | " + " | ".join(cols) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for i, r in enumerate(records, start=1):
        cells = [str(i), str(r.get("_label", "?"))]
        cells += [_format_condition(c, r.get(c)) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_report_bundle(
    candidates: pd.DataFrame,
    eda_results: dict | None = None,
) -> ReportBundle:
    """Build figures + ask the LLM + write the MD file. Returns the full bundle
    so the webui can render interactive Plotly figures with per-figure captions."""
    from jinja2 import Template

    # Single wall-clock stamp shared by the report filename, the figure PNG
    # filenames, and the in-body timestamp — otherwise a slow LLM call between
    # _build_figures() and the file write can offset them by ≥1 second.
    now = datetime.now()
    ts_str = now.isoformat(timespec="seconds")
    ts_file = now.strftime("%Y%m%d_%H%M%S")

    # Atomic-fraction candidates carry no role columns, so the label, the
    # family descriptor and the lookup enrichment each need their own
    # reconstruction of the support / metal-phase split. Everything downstream
    # of enrichment (figures, prompt, template) is shared with the role path.
    is_fraction = (getattr(config, "CATALYST_FRACTION_MODE", False)
                   and "active_metal" not in candidates.columns)

    directions = getattr(config, "OPTIMIZATION_DIRECTIONS", ["max"] * len(config.TARGET_COLS))
    primary_dir = directions[0] if directions else "max"
    top = candidates.sort_values(f"pred_{config.TARGET_COLS[0]}",
                                 ascending=(primary_dir == "min")).head(config.REPORT_TOP_K)
    top_records = top.to_dict(orient="records")
    for row in top_records:
        row["_label"] = (_fraction_composition_string(row) if is_fraction
                         else _format_catalyst_label(row))

    enriched_records = top_records
    sup_lookup_df: pd.DataFrame | None = None
    if config.CATALYST_MODE or is_fraction:
        enrich = _enrich_fraction_candidate if is_fraction else _enrich_candidate
        try:
            sup_lookup_df = pd.read_csv(config.SUPPORT_LOOKUP_PATH)
            metal_lookup = pd.read_csv(config.METAL_LOOKUP_PATH)
            enriched_records = [
                enrich(r, sup_lookup_df, metal_lookup) for r in top_records
            ]
        except Exception as e:
            log.warning("Could not enrich candidates with lookup props (%s).", e)
            if is_fraction:
                # The support/metal-phase split does not need the lookups —
                # only the property values do. Without this the family column
                # would read "unsupported" for a supported catalyst just
                # because a CSV was unreadable.
                enriched_records = [_enrich_fraction_candidate(r, None, None)
                                    for r in top_records]

    # Deterministic Family + Confidence columns for the top-K table.
    y_stds: dict[str, float] = {}
    raw_path = config.DATA_DIR / "raw.parquet"
    if raw_path.exists():
        try:
            raw_df = pd.read_parquet(raw_path)
            for t in config.TARGET_COLS:
                if t in raw_df.columns:
                    s = pd.to_numeric(raw_df[t], errors="coerce").dropna()
                    if len(s) > 1:
                        y_stds[t] = float(s.std())
        except Exception as e:
            log.warning("Could not load raw.parquet for Y-std normalization (%s).", e)
    build_family = _build_fraction_family if is_fraction else _build_family
    for row in enriched_records:
        row["_family"] = build_family(row, sup_lookup_df)
        row["_confidence"] = _build_confidence(row, list(config.TARGET_COLS), y_stds)

    agreement_summary = _build_agreement_summary(enriched_records, config.TARGET_COLS)

    eda_summary_str = None
    summary_path = config.DATA_DIR / "eda_summary.json"
    if summary_path.exists():
        eda_summary_str = summary_path.read_text()

    figures = _build_figures(candidates, enriched_records, eda_summary_str,
                             ts_file, is_fraction=is_fraction)

    has_bnn = all(
        f"pred_{t}_bnn" in candidates.columns for t in config.TARGET_COLS
    )

    llm_narrative, captions = _ask_ollama(
        enriched_records, eda_summary_str, config.TARGET_COLS, figures,
        agreement_summary, is_fraction=is_fraction,
    )
    llm_narrative = _normalize_narrative_headings(
        _repair_figure_links(llm_narrative, figures))

    # Attach the captions to the corresponding figures (fall back to the
    # hand-written default_description if the LLM didn't provide one).
    # Parity is special-cased: the LLM tends to produce optimistic "tight
    # clustering / high R²" boilerplate regardless of the actual numbers, so
    # we override it with a deterministic R²-grounded verdict.
    for fig in figures:
        if fig.figure_id == "parity":
            fig.llm_caption = _deterministic_parity_caption(
                captions.get("parity", ""),
            )
        else:
            fig.llm_caption = (
                captions.get(fig.figure_id, "").strip()
                or fig.default_description
            )

    per_candidate_section = _render_per_candidate_section(enriched_records)
    conditions_section = _render_conditions_section(enriched_records) if is_fraction else ""
    if is_fraction:
        label_kind = "Catalyst (atomic fractions)"
    elif getattr(config, "CATALYST_MODE", False):
        label_kind = "Catalyst"
    else:
        label_kind = "Composition"
    tmpl = Template(_TEMPLATE)
    md = tmpl.render(
        ts=ts_str,
        dataset_source=config.DATASET_SOURCE,
        dataset_name=config.MATMINER_DATASET if config.DATASET_SOURCE == "matminer" else str(config.CSV_PATH),
        label_kind=label_kind,
        schema_note=_FRACTION_SCHEMA_NOTE if is_fraction else "",
        targets=config.TARGET_COLS,
        n_samples=config.SUBSAMPLE_N,
        top_k=config.REPORT_TOP_K,
        eda_summary=eda_summary_str,
        candidates=enriched_records,
        llm_narrative=llm_narrative,
        per_candidate_section=per_candidate_section,
        conditions_section=conditions_section,
        figures=figures,
        has_bnn=has_bnn,
    )

    out = config.REPORTS_DIR / f"report_{ts_file}.md"
    out.write_text(md)
    log.info("Wrote report to %s.", out)

    return ReportBundle(
        path=out,
        ts=ts_str,
        dataset_source=config.DATASET_SOURCE,
        dataset_name=(config.MATMINER_DATASET
                      if config.DATASET_SOURCE == "matminer"
                      else str(config.CSV_PATH)),
        targets=list(config.TARGET_COLS),
        n_samples=config.SUBSAMPLE_N,
        eda_summary_str=eda_summary_str,
        candidates_enriched=enriched_records,
        has_bnn=has_bnn,
        narrative_body=llm_narrative,
        per_candidate_section=per_candidate_section,
        figures=figures,
    )


def generate_report(candidates: pd.DataFrame, eda_results: dict | None = None) -> Path:
    """Backward-compatible wrapper used by `run_pipeline.py`."""
    return generate_report_bundle(candidates, eda_results).path


_CAPTIONS_BLOCK_RE = re.compile(
    r"<<<FIGURE_CAPTIONS>>>\s*(.*?)\s*<<<END_FIGURE_CAPTIONS>>>",
    re.DOTALL,
)

_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_TS_SUFFIX_RE = re.compile(r"_\d{8}_\d{6}$")


def _figure_link_key(filename: str) -> str:
    """`parity_cv_20260807_202446.png` → `parity_cv`. Strips the directory,
    extension and the shared wall-clock stamp, leaving the part that identifies
    which figure was meant."""
    return _TS_SUFFIX_RE.sub("", Path(filename).name.rsplit(".", 1)[0])


def _repair_figure_links(narrative: str, figures: list[FigureEntry]) -> str:
    """Point every `![alt](file.png)` in the narrative at a figure that exists.

    The prompt gives the model the exact filenames and tells it to use them
    verbatim; phi-4 still paraphrases. Observed: it embedded
    `parity_20260807_202446.png` when the file written was
    `parity_cv_20260807_202446.png` — the CV/in-sample distinction is in the
    filename, and the model dropped it. That leaves a dead image in the saved
    Markdown. (The webui survives it: `_split_narrative_around_figures`
    ignores unknown filenames and appends the real figure at the end. The `.md`
    on disk has no such fallback, which is why this is fixed at the source
    rather than in the renderer.)

    A near-miss is rewritten only when it resolves to exactly ONE figure —
    either by `figure_id` or by one key being a prefix of the other. An
    ambiguous or unrecognizable link is dropped, not guessed: a missing figure
    is recoverable, a caption pointing at the WRONG figure is a false claim
    about the data.
    """
    if not narrative:
        return narrative

    known = {f.png_filename for f in figures if f.png_filename}
    aliases: dict[str, set[str]] = {}
    for f in figures:
        if not f.png_filename:
            continue
        for key in {_figure_link_key(f.png_filename), f.figure_id}:
            aliases.setdefault(key, set()).add(f.png_filename)

    def _resolve(cited: str) -> str | None:
        name = Path(cited).name
        if name in known:
            return name
        key = _figure_link_key(name)
        if not key:
            return None
        hits = set(aliases.get(key, ()))
        if not hits:
            hits = {fn for k, fns in aliases.items()
                    if k.startswith(key) or key.startswith(k)
                    for fn in fns}
        return next(iter(hits)) if len(hits) == 1 else None

    alt_by_file = {f.png_filename: f.alt for f in figures if f.png_filename}

    def _alt_for(alt: str, filename: str) -> str:
        """Keep a written-out alt; replace a filename masquerading as one.

        The model frequently emits `![pareto_20260807_202446.png](pareto_…png)`.
        That is not alt text — a screen reader would read the timestamp aloud —
        and FigureEntry already carries a description written for the purpose.
        """
        stripped = alt.strip()
        looks_like_filename = (
            not stripped
            or stripped.lower().endswith((".png", ".jpg", ".svg"))
            or _figure_link_key(stripped) == _figure_link_key(filename)
        )
        return alt_by_file.get(filename, stripped) if looks_like_filename else stripped

    def _sub(m: re.Match) -> str:
        alt, cited = m.group(1), m.group(2)
        fixed = _resolve(cited)
        if not fixed:
            log.warning("Dropped unresolvable figure link from narrative: %r.", cited)
            return ""
        if fixed != cited:
            log.info("Repaired figure link in narrative: %r → %r.", cited, fixed)
        new_alt = _alt_for(alt, fixed)
        if new_alt != alt:
            log.info("Replaced filename-shaped alt text for %r.", fixed)
        return f"![{new_alt}]({fixed})"

    return _MD_IMAGE_RE.sub(_sub, narrative)


# Headings the model echoes back from the prompt's own task description
# ("PART 2 — NARRATIVE BODY (markdown)") rather than titling its content with.
_STRAY_HEADING_RE = re.compile(
    r"^(?:part\s*\d+\b.*|narrative(?:\s+body)?(?:\s*\(markdown\))?"
    r"|markdown\s+narrative|narrative\s+section)$",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*$", re.MULTILINE)


def _normalize_narrative_headings(narrative: str) -> str:
    """Drop echoed task labels, and nest the model's headings under the report's.

    The narrative is rendered inside the report's own `## Why these materials
    are promising` section, so a `##` heading from the model becomes a SIBLING
    of `## Per-candidate notes` — the document then reads as though the
    narrative ended where it did not. Relative depth between the model's own
    headings is preserved; only the whole block is pushed down to `###`.
    """
    if not narrative:
        return narrative

    def _drop_stray(m: re.Match) -> str:
        text = m.group(2).strip().rstrip(":.-—– ")
        if _STRAY_HEADING_RE.match(text):
            log.info("Dropped echoed task heading from narrative: %r.", text)
            return ""
        return m.group(0)

    out = _HEADING_RE.sub(_drop_stray, narrative)

    levels = [len(m.group(1)) for m in _HEADING_RE.finditer(out)]
    if levels and min(levels) < 3:
        shift = 3 - min(levels)
        out = _HEADING_RE.sub(
            lambda m: f"{'#' * min(6, len(m.group(1)) + shift)} {m.group(2)}", out,
        )
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _render_per_candidate_section(top_candidates: list[dict]) -> str:
    """Deterministic per-candidate bullets built directly from lookup-derived
    properties. The LLM cannot hallucinate labels or values here because no
    LLM is involved — phi-4 has demonstrated repeated failure to follow
    'use these labels verbatim' instructions for this section."""
    out: list[str] = []
    for i, c in enumerate(top_candidates, start=1):
        label = c.get("_label", "?")
        family = c.get("_family", "")
        conf = c.get("_confidence", "—")
        header = f"**{label}**"
        meta = []
        if family:
            meta.append(family)
        meta.append(f"{conf} confidence")
        header += f" — *{', '.join(meta)}*"

        chunks: list[str] = []
        sup_props = c.get("support_properties") or {}
        if sup_props:
            sup_bits: list[str] = []
            oxide_cls = sup_props.get("oxide_class")
            if oxide_cls:
                sup_bits.append(f"{oxide_cls}")
            for k, name in (
                ("acidity_NH3_TPD_mmol_g", "NH₃ acidity"),
                ("basicity_CO2_TPD_mmol_g", "CO₂ basicity"),
                ("E_Ov_eV", "E_Ov"),
                ("work_function_eV", "Φ"),
                ("surface_energy_J_m2", "γ_surf"),
            ):
                if k in sup_props:
                    sup_bits.append(f"{name}={sup_props[k]}")
            if sup_bits:
                chunks.append(f"Support: {', '.join(sup_bits)}.")

        metal_props = c.get("metal_properties") or {}
        # Role schema names one active metal; fraction schema has a weighted
        # metal phase and no role labels at all — say which one this is rather
        # than printing a mixture under an "Active …" heading.
        if c.get("_schema") == "fraction":
            m_name = c.get("metal_phase_formula")
            m_prefix = "Metal phase"
        else:
            m_name = c.get("active_metal")
            m_prefix = "Active"
        if metal_props and m_name:
            bits: list[str] = []
            for k, name in (
                ("d_band_center_eV", "d-band"),
                ("pauling_chi", "χ"),
                ("work_function_eV", "Φ"),
                ("E_ads_C_eV", "E_ads(C)"),
                ("E_ads_H_eV", "E_ads(H)"),
            ):
                if k in metal_props:
                    bits.append(f"{name}={metal_props[k]}")
            if bits:
                chunks.append(f"{m_prefix} {m_name}: {', '.join(bits)}.")

        for slot_key, slot_name_attr in (
            ("promoter_1_properties", "promoter_1"),
            ("promoter_2_properties", "promoter_2"),
        ):
            props = c.get(slot_key) or {}
            name = c.get(slot_name_attr)
            if props and name:
                bits = []
                for k, label_short in (
                    ("pauling_chi", "χ"),
                    ("work_function_eV", "Φ"),
                    ("d_band_center_eV", "d-band"),
                ):
                    if k in props:
                        bits.append(f"{label_short}={props[k]}")
                if bits:
                    chunks.append(f"{slot_name_attr.replace('_', ' ').capitalize()} {name}: {', '.join(bits)}.")

        body = " ".join(chunks) if chunks else "(no lookup-derived properties)"
        out.append(f"- {header}\n  {body}")
    return "\n\n".join(out)


def _render_candidates_for_prompt(top_candidates: list[dict]) -> str:
    """Render each candidate as a labeled text block with only the fields the
    LLM needs (composition, family, lookup-derived properties). Predicted
    target values are deliberately omitted — those live in the deterministic
    upstream table; including them here invites paraphrase errors."""
    blocks: list[str] = []
    for i, c in enumerate(top_candidates, start=1):
        lines = [f"CANDIDATE {i}: {c.get('_label', '?')}"]
        if c.get("_family"):
            lines.append(f"  family: {c['_family']}")
        is_fraction = c.get("_schema") == "fraction"
        if is_fraction:
            sup, sup_frac = c.get("support"), c.get("support_fraction")
            if sup:
                lines.append(f"  support: {sup} (reconstructed from "
                             f"{c.get('support_cation')} fraction {sup_frac})")
            phase = c.get("metal_phase") or {}
            if phase:
                lines.append("  metal_phase (renormalized over non-support "
                             "elements): " +
                             ", ".join(f"{e}={w}" for e, w in phase.items()))
            conds = {k: c.get(k)
                     for k in getattr(config, "CATALYST_FRACTION_CONDITIONS", [])
                     if k in c}
            if conds:
                lines.append("  reaction_conditions: {" +
                             ", ".join(f"{k}={v}" for k, v in conds.items()) + "}")
        else:
            for slot, key in (("active_metal", "active_metal"),
                              ("promoter_1", "promoter_1"),
                              ("promoter_2", "promoter_2"),
                              ("support", "support")):
                v = c.get(key)
                if v not in (None, "", "nan"):
                    lines.append(f"  {slot}: {v}")
        metal_prop_label = ("metal_phase_properties" if is_fraction
                            else "active_metal_properties")
        for prop_key, prop_label in (
            ("support_properties", "support_properties"),
            ("metal_properties", metal_prop_label),
            ("promoter_1_properties", "promoter_1_properties"),
            ("promoter_2_properties", "promoter_2_properties"),
        ):
            props = c.get(prop_key)
            if props:
                kv = ", ".join(f"{k}={v}" for k, v in props.items())
                lines.append(f"  {prop_label}: {{{kv}}}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


_FRACTION_PROMPT_SCHEMA = (
    "─── (0) SCHEMA: ATOMIC FRACTIONS — READ FIRST ───\n"
    "This dataset describes each catalyst as a vector of ATOMIC FRACTIONS over "
    "a fixed element panel, plus reaction conditions. It contains NO role "
    "labels. Consequences you must respect:\n"
    "  • Never call an element the 'active metal', a 'promoter', or a "
    "'dopant' — those roles are not in the data. Say 'metal-phase "
    "constituent', 'minor component', or name the element and its fraction.\n"
    "  • The support shown per candidate was RECONSTRUCTED: the dominant "
    "Al/Si/Zr cation was mapped to its oxide. It is an inference from the "
    "composition, not a reported field — describe it as such if you mention "
    "it.\n"
    "  • `metal_phase_properties` are COMPOSITION-WEIGHTED averages over the "
    "non-support elements, so they describe the metal phase as a whole, not "
    "any single element. Do not attribute such a value to one element.\n"
    "  • Reaction conditions (calcination/reaction temperature, pressure, "
    "flow, pretreatment) were optimized JOINTLY with composition. A claim "
    "about what makes the batch promising that ignores conditions is "
    "incomplete.\n\n"
)


def _ask_ollama(top_candidates: list[dict], eda_summary: str | None,
                target_cols: list[str], figures: list[FigureEntry],
                agreement_summary: dict | None = None,
                is_fraction: bool = False) -> tuple[str, dict[str, str]]:
    """Ask the local Ollama model for the narrative + per-figure captions.

    Returns (narrative_body, captions_dict). `captions_dict` is keyed by
    `figure_id`. If the captions block is missing or malformed, captions are
    empty and the caller falls back to default_description.
    """
    try:
        import ollama
    except ImportError:
        return "*(ollama package not installed — narrative skipped.)*", {}

    feature_rankings = _extract_feature_rankings(eda_summary)

    figure_lines = []
    for f in figures:
        figure_lines.append(
            f"  • id={f.figure_id} | file={f.png_filename} | {f.default_description}"
        )
    figure_block = "(no figures available)" if not figure_lines else "\n".join(figure_lines)

    n_candidates = len(top_candidates)
    # ensure_ascii=False or the model copies the escape verbatim: the
    # interpretation string contains "|Δ| = 0.014", which json.dumps renders as
    # "|\u0394| = 0.014", and phi-4 reproduced that literally in the narrative.
    agreement_block = (
        json.dumps(agreement_summary, indent=2, ensure_ascii=False)
        if agreement_summary else "(GP-vs-BNN agreement summary unavailable)"
    )

    figure_id_list = ", ".join(f.figure_id for f in figures) or "(none)"

    label_list_block = "\n".join(
        f"  {i+1}. {c.get('_label', '?')!r}"
        for i, c in enumerate(top_candidates)
    )
    candidates_text = _render_candidates_for_prompt(top_candidates)

    prompt = (
        "You are a materials scientist writing the analysis section of a "
        "scientific report.\n"
        f"Goal: maximize {target_cols}.\n"
        + (_FRACTION_PROMPT_SCHEMA if is_fraction else "") +
        f"There are EXACTLY {n_candidates} candidates. They are listed below "
        "with the exact labels you MUST use, in this order:\n\n"
        f"{label_list_block}\n\n"
        "These are the ONLY catalysts you may discuss. Do not invent any "
        "other catalysts (no 'Pd/Al2O3', no 'Pt/Zeolite' — only the labels "
        "above). Do not draw on generic catalysis knowledge for candidates "
        "outside this list. Do not swap supports, metals, or promoters "
        "between candidates.\n\n"
        "─── (A) CANDIDATES + ACTUAL PROPERTY VALUES ───\n"
        "One block per candidate. Cite ONLY values from a candidate's own "
        "block when discussing that candidate. If a property is absent, do "
        "not cite a number for it.\n\n"
        f"{candidates_text}\n\n"
        "─── (B) FEATURE IMPORTANCE RANKINGS (names + stability) ───\n"
        "Use to argue WHICH property families matter most. No numeric scores; "
        "refer to features by name only.\n"
        "`stability` is how many cross-validation folds independently ranked "
        "that feature in their own top-K, measured by HELD-OUT permutation "
        "importance (`method: permutation_cv`). Weight your claims by it: a "
        "feature stable in most folds is a real signal; one appearing in a "
        "single fold is noise and must NOT be presented as an established "
        "driver. If `method` is `gain_in_sample` the ranking had too little "
        "data for a held-out split — say it is indicative only. If a target's "
        "feature list is EMPTY, no feature had positive held-out importance: "
        "state plainly that the data does not support a feature-importance "
        "claim for that target rather than falling back on generic "
        "catalysis intuition.\n\n"
        f"{json.dumps(feature_rankings, indent=2, ensure_ascii=False)}\n\n"
        "─── (C) GP-vs-BNN AGREEMENT SUMMARY ───\n"
        "Use this exact information in your surrogate-confidence paragraph; "
        "do not invert it.\n\n"
        f"{agreement_block}\n\n"
        "─── (D) FIGURES AVAILABLE FOR EMBEDDING ───\n"
        "Embed each at least once with markdown image syntax. Use the exact "
        "filenames; no path prefix; format `![alt text](filename.png)`.\n"
        f"{figure_block}\n\n"
        "─── TASK ───\n"
        "Your output has TWO parts in this exact order:\n\n"
        "PART 1 — FIGURE CAPTIONS (machine-parsed).\n"
        "Output this block first, with NO text before it:\n\n"
        "<<<FIGURE_CAPTIONS>>>\n"
        "{\n"
        f"  // one short interpretation (2-4 sentences) per figure id below: "
        f"{figure_id_list}\n"
        "  \"<figure_id>\": \"<plain-text caption, no markdown image syntax>\",\n"
        "  ...\n"
        "}\n"
        "<<<END_FIGURE_CAPTIONS>>>\n\n"
        "Rules for captions: VALID JSON only (use double quotes, no trailing "
        "commas, no comments in the final output). Each caption explains what "
        "the figure SHOWS and what it MEANS for the design decision. Do not "
        "embed image markdown inside captions. You may cite numeric values "
        "from (A), (C), or the figure's own description in (D) (e.g. R²/MAE "
        "for the parity figure).\n\n"
        "FOR THE PARITY CAPTION SPECIFICALLY: read the per-target R² and MAE "
        "values written in the parity figure's description above; cite the "
        "actual R² values verbatim and judge the surrogate's quality from "
        "those numbers — do NOT default to phrases like 'tight clustering' or "
        "'high R² and low MAE' unless the cited R² is actually ≥ 0.7. If R² "
        "is <0.3 or negative, say so explicitly and note that BO suggestions "
        "should be treated as speculative.\n\n"
        "PART 2 — NARRATIVE BODY (markdown).\n"
        "After the END_FIGURE_CAPTIONS line, produce the narrative with this "
        "structure. DO NOT produce a comparison table and DO NOT write any "
        "per-candidate prose — both are rendered deterministically upstream "
        "from the actual BO output. Stick to general framing only.\n\n"
        "1. **Two short context paragraphs** — Pareto plot (embed it) and "
        "feature importance per target (embed it). Discuss what the figures "
        "reveal at the population level; do not name individual candidates.\n\n"
        + ("1b. **One paragraph on composition** — embed the metal-phase "
           "composition heatmap (figure_id=composition). Describe how much "
           "chemical diversity the batch actually spans: which elements recur "
           "across candidates, whether the batch is concentrated on one or two "
           "of them, and at what fraction level. Talk about the SET, not "
           "individual rows.\n\n" if is_fraction else "") +
        "2. **Closing paragraph on surrogate confidence** — embed BOTH the "
        "GP-vs-BNN scatter AND (if available) the parity figure "
        "(figure_id=parity). Read the parity figure's description in (D) "
        "carefully: it tells you whether parity is k-fold-CV HELD-OUT (a "
        "generalization signal) or IN-SAMPLE (a debug signal only). Comment "
        "accordingly — for CV parity, talk about generalization to unseen "
        "catalysts; for in-sample, only talk about whether the surrogate can "
        "fit its training data, and explicitly note CV would be needed for a "
        "generalization claim. "
        # Without a BNN there is no agreement summary, and ordering the model
        # to name a best-agreement candidate anyway is what makes it invent
        # one. Ask for the opposite statement instead.
        + ("Then name the candidate with the smallest "
           "|delta| from (C). Use the exact label and index from (C)'s "
           "`best_agreement_candidate_label` field; do NOT name any other "
           "candidate as having the closest agreement.\n\n"
           if agreement_summary else
           "(C) is UNAVAILABLE for this run — only one surrogate was fit, so "
           "there is no GP-vs-BNN comparison. Say exactly that: no "
           "cross-surrogate agreement check was available. Do NOT name a "
           "best-agreement candidate, and do not describe any candidate as "
           "aligning closely between models.\n\n")
        +
        "Strict rules:\n"
        "  • Do NOT name individual candidates in the prose; the per-candidate "
        "section is rendered deterministically below your narrative.\n"
        "  • Never cite an importance score as a property value.\n"
        "  • Use figure filenames exactly as given in (D).\n"
        "  • Whole narrative body under ~350 words.\n"
    )

    try:
        resp = ollama.chat(
            model=config.REPORT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        content = resp["message"]["content"]
    except Exception as e:
        log.warning("Ollama call failed (%s); returning placeholder.", e)
        return f"*(LLM call failed: {e})*", {}

    captions, narrative = _split_captions(content)
    return narrative, captions


def _split_captions(content: str) -> tuple[dict[str, str], str]:
    """Pull the <<<FIGURE_CAPTIONS>>>…<<<END_FIGURE_CAPTIONS>>> JSON block out
    of the LLM output. Returns (captions_dict, remaining_narrative).
    Tolerates a missing or malformed block."""
    m = _CAPTIONS_BLOCK_RE.search(content)
    if not m:
        return {}, content.strip()

    raw_json = m.group(1).strip()
    # Strip line comments the LLM might leave in despite instructions.
    cleaned = re.sub(r"//[^\n]*", "", raw_json)
    try:
        captions = json.loads(cleaned)
        if not isinstance(captions, dict):
            captions = {}
    except Exception as e:
        log.warning("Could not parse FIGURE_CAPTIONS JSON: %s", e)
        captions = {}

    narrative = (content[:m.start()] + content[m.end():]).strip()
    return {str(k): str(v) for k, v in captions.items()}, narrative


def _build_figures(
    candidates: pd.DataFrame,
    enriched_records: list[dict],
    eda_summary_str: str | None,
    ts: str,
    is_fraction: bool = False,
) -> list[FigureEntry]:
    """Generate static PNGs (for the saved MD) + Plotly figures (for the webui).
    Returns one FigureEntry per successfully-rendered figure. `ts` is the shared
    wall-clock stamp (YYYYmmdd_HHMMSS) — must match the report filename so the
    Markdown's image links don't drift off by a second."""
    directions = getattr(config, "OPTIMIZATION_DIRECTIONS", ["max"] * len(config.TARGET_COLS))
    from plots import (
        make_pareto_plot, make_feature_importance_plot,
        make_gp_vs_bnn_scatter, make_candidate_heatmap, make_parity_plot,
        make_fraction_composition_heatmap,
    )
    from interactive_plots import (
        make_pareto_figure, make_feature_importance_figure,
        make_gp_vs_bnn_figure, make_candidate_heatmap_figure,
        make_parity_figure, make_fraction_composition_figure,
    )

    out: list[FigureEntry] = []

    # Parity: prefer k-fold CV (held-out) parity, fall back to in-sample.
    # step4 deletes both artifacts before refitting, so presence here means
    # "produced by the run that just finished". The extra covers-our-targets
    # check catches the remaining case where a file survives a crash and the
    # target set has since changed — labelling one target's CV numbers as if
    # they described another's would be worse than drawing no figure at all.
    parity_path, is_cv = _select_parity_artifact()

    if parity_path is not None:
        try:
            parity_df = pd.read_parquet(parity_path)
            n_folds = (int(parity_df["fold"].nunique())
                       if "fold" in parity_df.columns else None)
            fname = f"parity_{'cv' if is_cv else 'insample'}_{ts}.png"
            png = make_parity_plot(parity_df, config.TARGET_COLS,
                                   config.REPORTS_DIR / fname)
            interactive = make_parity_figure(parity_df, config.TARGET_COLS)
            stats_lines = _parity_stats_lines(parity_df, config.TARGET_COLS)
            if png or interactive:
                if is_cv:
                    title = (f"Surrogate parity ({n_folds}-fold CV, held-out)"
                             if n_folds else "Surrogate parity (k-fold CV, held-out)")
                    description = (
                        "Per-target scatter of TRUE values (x) vs HELD-OUT "
                        "k-fold-CV predictions (y)"
                        f"{' (k=' + str(n_folds) + ')' if n_folds else ''}, "
                        "shuffled with fixed seed. Dashed line = y = x. "
                        "Held-out R² and MAE per surrogate per target:\n"
                        + stats_lines +
                        "\nInterpret R² LITERALLY: ≥0.7 = strong "
                        "generalization; 0.3-0.7 = weak/noisy; <0.3 (or "
                        "negative) = the surrogate does NOT generalize and BO "
                        "suggestions should be treated as speculative. Do NOT "
                        "praise the parity if R² is low."
                    )
                else:
                    title = "Surrogate parity (in-sample)"
                    description = (
                        "Per-target scatter of TRUE target values (x) vs the "
                        "surrogate's IN-SAMPLE predictions (y) on the "
                        "training set. Dashed line = y = x. This is a "
                        "DEBUG-ONLY signal — generalization is NOT measured "
                        "(enable CV_FOLDS for that). In-sample R² and MAE:\n"
                        + stats_lines +
                        "\nNote in your caption that these are in-sample, not "
                        "held-out, numbers."
                    )
                out.append(FigureEntry(
                    figure_id="parity",
                    title=title,
                    alt=("Predicted vs actual (k-fold CV, held-out)"
                         if is_cv else "Predicted vs actual (in-sample)"),
                    default_description=description,
                    png_filename=fname if png else None,
                    plotly_fig=interactive,
                ))
        except Exception as e:
            log.warning("Parity plot failed: %s", e)

    raw_path = config.DATA_DIR / "raw.parquet"
    if len(config.TARGET_COLS) == 2 and raw_path.exists():
        try:
            training = pd.read_parquet(raw_path)
            fname = f"pareto_{ts}.png"
            maximize = tuple(d == "max" for d in directions[:2])
            png = make_pareto_plot(training, candidates, config.TARGET_COLS,
                                   config.REPORTS_DIR / fname, maximize=maximize)
            interactive = make_pareto_figure(training, candidates, config.TARGET_COLS,
                                             maximize=maximize)
            if png or interactive:
                out.append(FigureEntry(
                    figure_id="pareto",
                    title="Pareto frontier",
                    alt="Pareto frontier of training data and BO candidates",
                    default_description=(
                        f"Scatter of {config.TARGET_COLS[0]} (x) vs "
                        f"{config.TARGET_COLS[1]} (y). Gray = training catalysts; "
                        "hollow black = training Pareto front; red diamonds = "
                        "BO candidates with σ error bars labelled #1..#N."
                    ),
                    png_filename=fname if png else None,
                    plotly_fig=interactive,
                ))
        except Exception as e:
            log.warning("Pareto plot failed: %s", e)

    try:
        fname = f"feature_importance_{ts}.png"
        png = make_feature_importance_plot(eda_summary_str,
                                           config.REPORTS_DIR / fname)
        interactive = make_feature_importance_figure(eda_summary_str)
        if png or interactive:
            out.append(FigureEntry(
                figure_id="feature_importance",
                title="Feature importance",
                alt="Top XGBoost-gain features per target",
                default_description=(
                    "Two horizontal-bar panels (one per target) showing the top 10 "
                    "predictive features from the EDA XGBoost surrogate. Useful "
                    "for arguing WHICH property families matter most."
                ),
                png_filename=fname if png else None,
                plotly_fig=interactive,
            ))
    except Exception as e:
        log.warning("Feature-importance plot failed: %s", e)

    try:
        fname = f"gp_vs_bnn_{ts}.png"
        png = make_gp_vs_bnn_scatter(candidates, config.TARGET_COLS,
                                     config.REPORTS_DIR / fname)
        interactive = make_gp_vs_bnn_figure(candidates, config.TARGET_COLS)
        if png or interactive:
            out.append(FigureEntry(
                figure_id="gp_vs_bnn",
                title="GP vs Bayesian-NN agreement",
                alt="GP vs Bayesian-NN agreement scatter",
                default_description=(
                    "Per-target scatter of GP-predicted vs BNN-predicted target "
                    "values for the BO candidates. Dashed diagonal = perfect "
                    "agreement. Points far from the diagonal indicate the two "
                    "surrogates disagree, so the candidate prediction should be "
                    "treated with extra caution."
                ),
                png_filename=fname if png else None,
                plotly_fig=interactive,
            ))
    except Exception as e:
        log.warning("GP-vs-BNN scatter failed: %s", e)

    try:
        fname = f"candidate_heatmap_{ts}.png"
        png = make_candidate_heatmap(enriched_records, config.REPORTS_DIR / fname)
        interactive = make_candidate_heatmap_figure(enriched_records)
        if png or interactive:
            out.append(FigureEntry(
                figure_id="candidate_heatmap",
                title="Candidate property heatmap",
                alt="Candidate property heatmap",
                default_description=(
                    "Heatmap with rows = top candidates and columns = key "
                    "support / metal properties (work function, acidity, "
                    "basicity, lattice constant, d-band center, χ). Color is "
                    "column min-max normalised across the candidate set; "
                    "annotated with the raw property values."
                ),
                png_filename=fname if png else None,
                plotly_fig=interactive,
            ))
    except Exception as e:
        log.warning("Candidate heatmap failed: %s", e)

    # Fraction schema only: the candidate's identity IS its composition vector,
    # and the top-K table can only show it as a one-line string. The heatmap is
    # where you see whether the batch actually explored different chemistries
    # or just re-picked the same element at different loadings.
    if is_fraction:
        try:
            element_cols = list(getattr(config, "CATALYST_FRACTION_ELEMENTS", []))
            support_cations = list(getattr(config, "CATALYST_FRACTION_SUPPORT_CATIONS", []))
            fname = f"composition_{ts}.png"
            png = make_fraction_composition_heatmap(
                enriched_records, element_cols, support_cations,
                config.REPORTS_DIR / fname,
            )
            interactive = make_fraction_composition_figure(
                enriched_records, element_cols, support_cations,
            )
            if png or interactive:
                out.append(FigureEntry(
                    figure_id="composition",
                    title="Metal-phase composition",
                    alt="Metal-phase atomic fractions per candidate",
                    default_description=(
                        "Heatmap with rows = top candidates and columns = every "
                        "non-support element any candidate carries. Colour is "
                        "the atomic fraction itself on ONE shared scale "
                        "(fractions share a unit, so no per-column "
                        "normalisation), annotated with the raw value; blank = "
                        "element absent. The support is excluded because it is "
                        "~0.95 in every row — it is named in each row label "
                        "instead. Note the scale: a trace element near 1e-3 "
                        "reads as almost black even though it may be the "
                        "catalytically active one."
                    ),
                    png_filename=fname if png else None,
                    plotly_fig=interactive,
                ))
        except Exception as e:
            log.warning("Fraction composition heatmap failed: %s", e)

    log.info("Generated %d figures for the report.", len(out))
    return out


def _parity_covers_targets(path: Path, target_cols: list[str]) -> bool:
    """True if `path` holds true-value columns for every configured target."""
    try:
        cols = set(pd.read_parquet(path).columns)
    except Exception as e:
        log.warning("Could not read parity artifact %s (%s); ignoring it.", path.name, e)
        return False
    missing = [t for t in target_cols if f"true_{t}" not in cols]
    if missing:
        log.warning(
            "Parity artifact %s has no true-value column for target(s) %s — it "
            "is left over from a run with a different target set. Ignoring it "
            "rather than mislabelling its numbers.", path.name, missing,
        )
        return False
    return True


def _select_parity_artifact() -> tuple[Path | None, bool]:
    """Pick the parity parquet to plot. Returns (path, is_cv)."""
    cv_path = config.DATA_DIR / "cv_parity_predictions.parquet"
    in_sample_path = config.DATA_DIR / "training_predictions.parquet"
    targets = list(config.TARGET_COLS)
    for path, is_cv in ((cv_path, True), (in_sample_path, False)):
        if path.exists() and _parity_covers_targets(path, targets):
            return path, is_cv
    return None, False


def _build_agreement_summary(records: list[dict], target_cols: list[str]) -> dict | None:
    """Compact top-level GP-vs-BNN agreement summary. Keeps it OUT of each
    candidate dict (which bloats per-candidate JSON and confuses small models)
    and gives the LLM a single short block to consult."""
    if not records:
        return None
    primary = target_cols[0]

    per_candidate = []
    for i, r in enumerate(records):
        entry = {
            "index": i + 1,
            "label": r.get("_label", f"candidate {i+1}"),
        }
        for t in target_cols:
            gp_key, bnn_key = f"pred_{t}", f"pred_{t}_bnn"
            if gp_key in r and bnn_key in r:
                d = r[bnn_key] - r[gp_key]
                entry[f"{t}_abs_delta"] = round(abs(d), 3)
        per_candidate.append(entry)

    primary_key = f"{primary}_abs_delta"
    candidates_with_delta = [c for c in per_candidate if primary_key in c]
    if not candidates_with_delta:
        return None
    best = min(candidates_with_delta, key=lambda c: c[primary_key])

    return {
        "primary_target": primary,
        "per_candidate_abs_delta": per_candidate,
        "best_agreement_candidate_index": best["index"],
        "best_agreement_candidate_label": best["label"],
        "best_agreement_abs_delta": best[primary_key],
        "interpretation": (
            f"Candidate #{best['index']} ({best['label']}) has the smallest "
            f"|delta| on {primary} (|Δ| = {best[primary_key]:.3f}). It is the "
            "safest GP-BNN-aligned pick. State this verbatim in your "
            "surrogate-confidence paragraph; do NOT name any other candidate "
            "as having the closest agreement."
        ),
    }


def _compute_parity_stats(
    parity_df: pd.DataFrame, target_cols: list[str],
) -> list[tuple[str, str, float, float, int]]:
    """Return list of (kind, target, r2, mae, n) tuples."""
    import numpy as np
    out: list[tuple[str, str, float, float, int]] = []
    for t in target_cols:
        true_col = f"true_{t}"
        if true_col not in parity_df.columns:
            continue
        y_true = parity_df[true_col].values
        for kind in ("gp", "svgp", "bnn"):
            mean_col = f"{kind}_pred_{t}"
            if mean_col not in parity_df.columns:
                continue
            y_pred = parity_df[mean_col].values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if mask.sum() < 2:
                continue
            yt, yp = y_true[mask], y_pred[mask]
            ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
            ss_res = float(np.sum((yt - yp) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mae = float(np.mean(np.abs(yt - yp)))
            out.append((kind, t, r2, mae, int(mask.sum())))
    return out


def _deterministic_parity_caption(llm_caption_raw: str) -> str:
    """Build a parity caption from the actual stats parquet, with an
    R²-grounded verdict. The LLM's own caption (if produced) is appended for
    flavor but the leading factual statement is guaranteed honest.
    """
    # Must use the SAME selection as _build_figures, otherwise the caption can
    # quote held-out CV numbers under an in-sample figure.
    parity_path, is_cv = _select_parity_artifact()
    if parity_path is None:
        return llm_caption_raw.strip()

    try:
        parity_df = pd.read_parquet(parity_path)
    except Exception:
        return llm_caption_raw.strip()

    stats = _compute_parity_stats(parity_df, config.TARGET_COLS)
    if not stats:
        return llm_caption_raw.strip()

    parts: list[str] = []
    parts.append(
        f"{'Held-out k-fold' if is_cv else 'In-sample'} parity statistics:"
    )
    for kind, target, r2, mae, n in stats:
        parts.append(
            f"  - {kind.upper()} on {target}: R² = {r2:.3f}, MAE = {mae:.3g} (n={n})"
        )

    # Verdict: take the BEST per-target R² across surrogates (the surrogate
    # we'd actually rely on). For each target, take max GP/BNN R².
    by_target: dict[str, float] = {}
    for kind, target, r2, _mae, _n in stats:
        by_target[target] = max(by_target.get(target, -1e30), r2)
    worst = min(by_target.values())
    best = max(by_target.values())
    if best >= 0.7:
        verdict = (
            "Verdict: the surrogate generalizes well on the strongest target; "
            "BO suggestions there can be taken as informed."
        )
    elif best >= 0.3:
        verdict = (
            "Verdict: weak/noisy generalization. Treat BO rankings as "
            "directional only and validate top candidates experimentally "
            "before committing."
        )
    else:
        verdict = (
            "Verdict: the surrogate does NOT generalize on the held-out data "
            f"(best R² across targets is only {best:.3f}). BO suggestions "
            "should be treated as speculative — they reflect data-poor "
            "extrapolation, not learned chemistry. Adding training data or "
            "reducing feature dimensionality is more important than running "
            "more BO rounds."
        )
    if worst < best and best >= 0.3:
        verdict += (
            f" Note the worst-target R² is {worst:.3f}, so confidence is "
            "uneven across the two objectives."
        )

    deterministic = "\n".join(parts) + "\n\n" + verdict
    # If the LLM produced something non-trivial, append it as a brief gloss —
    # but the deterministic verdict comes first so it can't be drowned out.
    llm = llm_caption_raw.strip()
    if llm:
        deterministic += f"\n\nLLM gloss: {llm}"
    return deterministic


def _parity_stats_lines(parity_df: pd.DataFrame, target_cols: list[str]) -> str:
    """Per (surrogate, target) held-out R² and MAE, as bullet lines for the
    LLM prompt. Lets the LLM ground its caption in real numbers instead of
    defaulting to generic 'tight clustering' praise.

    Shares _compute_parity_stats with the deterministic caption: these were
    two independent copies of the same R²/MAE arithmetic, so a fix to one
    would have silently disagreed with the other in the SAME report.
    """
    lines = [
        f"  - {kind.upper()} on {target}: R² = {r2:.3f}, MAE = {mae:.3g} (n = {n})"
        for kind, target, r2, mae, n in _compute_parity_stats(parity_df, target_cols)
    ]
    return "\n".join(lines) if lines else "  (stats unavailable)"


def _extract_feature_rankings(eda_summary: str | None) -> dict:
    """Feature names per target, annotated with how STABLE each ranking is.

    Names alone let the LLM present a one-fold fluke and a five-fold-consistent
    signal in the same confident register. step3 now scores these by held-out
    permutation importance across CV folds, so `folds_in_top_k` says how many
    folds independently ranked a feature top-K. That annotation travels into
    the prompt so the narrative can hedge where the data does.

    Shape: {target: {"method": str, "features": [{"name", "stability"}, ...]}}
    Falls back to bare names for summaries written before this field existed.
    """
    if not eda_summary:
        return {}
    try:
        data = json.loads(eda_summary)
    except Exception:
        return {}

    out: dict = {}
    for target, entries in (data.get("top_features_per_target") or {}).items():
        if not entries:
            out[target] = {"method": "none",
                           "note": "no feature had positive held-out importance",
                           "features": []}
            continue
        method = entries[0].get("method", "unknown")
        features = []
        for e in entries:
            item = {"name": e["feature"]}
            if e.get("n_folds"):
                item["stability"] = f"{e.get('folds_in_top_k', 0)}/{e['n_folds']} folds"
            features.append(item)
        out[target] = {"method": method, "features": features}
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    candidates = pd.read_parquet(config.DATA_DIR / "candidates.parquet")
    out = generate_report(candidates)
    print(f"Report: {out}")
