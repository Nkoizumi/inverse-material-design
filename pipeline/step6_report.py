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
**Active metal ({{ row['active_metal'] }})**

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


def _enrich_candidate(row: dict, sup_lookup: pd.DataFrame,
                      metal_lookup: pd.DataFrame) -> dict:
    """Attach the actual lookup-derived support/metal/promoter features for one
    candidate. The LLM is instructed to cite ONLY these numbers when quoting
    properties, which prevents hallucinated values like '0.49 mmol/g' for ZnO
    when the lookup says 0.15."""
    from catalyst_features import parse_components, weighted_lookup

    enriched = dict(row)

    def _props_from_lookup(cell: str | None, lookup: pd.DataFrame,
                            key_col: str, skip: set[str]) -> dict | None:
        if not cell or str(cell).strip() == "":
            return None
        keys = set(lookup[key_col].astype(str))
        components = parse_components(str(cell), keys)
        if not components:
            return None
        numeric_cols = [c for c in lookup.columns
                        if c != key_col and c not in skip
                        and pd.api.types.is_numeric_dtype(lookup[c])]
        cat_cols = [c for c in lookup.columns
                    if c != key_col and c not in skip
                    and not pd.api.types.is_numeric_dtype(lookup[c])]
        feats = weighted_lookup(components, lookup, key_col, numeric_cols, cat_cols)
        return {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in feats.items() if v is not None and not (isinstance(v, float) and pd.isna(v))
        }

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


def _fraction_composition_string(row: dict) -> str:
    """Compact `Al(sup)=0.95 | Ga=0.029 Mo=0.010` label for a fraction candidate."""
    elems = getattr(config, "CATALYST_FRACTION_ELEMENTS", [])
    supports = set(getattr(config, "CATALYST_FRACTION_SUPPORT_CATIONS", []))
    sup_fracs = {s: float(row.get(s, 0.0)) for s in supports if s in row}
    sup_str = ""
    if sup_fracs:
        sup = max(sup_fracs, key=sup_fracs.get)
        sup_str = f"{sup}(sup)={sup_fracs[sup]:.2f}"
    non_sup = sorted(((e, float(row.get(e, 0.0))) for e in elems
                      if e not in supports and float(row.get(e, 0.0)) > 0.005),
                     key=lambda t: -t[1])
    metal_str = " ".join(f"{e}={v:.3f}" for e, v in non_sup)
    return f"{sup_str} | {metal_str}" if sup_str else metal_str


def _generate_fraction_minimal_report(
    candidates: pd.DataFrame,
    eda_results: dict | None,
    ts_str: str,
    ts_file: str,
) -> ReportBundle:
    """Minimal report for atomic-fraction candidates: composition table + summary
    stats. No LLM narrative, no per-candidate role-based enrichment, no figures.
    Intended as a v0.1 graceful-degrade until fraction-mode narrative lands."""
    log.info("Fraction-mode candidates: emitting minimal report "
             "(no LLM narrative / role-based enrichment).")

    targets = list(config.TARGET_COLS)
    directions = getattr(config, "OPTIMIZATION_DIRECTIONS", ["max"] * len(targets))
    primary = targets[0] if targets else None

    top = candidates
    if primary:
        top = candidates.sort_values(f"pred_{primary}",
                                     ascending=(directions[0] == "min"))
    top = top.head(getattr(config, "REPORT_TOP_K", 10)).reset_index(drop=True)

    # Compact composition + predictions rows.
    lines = [
        f"# Inverse design report (minimal — atomic-fraction mode)",
        f"",
        f"- Generated: `{ts_str}`",
        f"- Candidates: {len(candidates)}  (showing top {len(top)} by "
        f"{'max' if directions and directions[0] == 'max' else 'min'} `{primary}`)",
        f"- Targets: {', '.join(targets)}",
        f"",
        f"> Atomic-fraction schema does not yet have a full LLM narrative or "
        f"per-candidate chemistry section. Enable role-based mode "
        f"(`CATALYST_MODE=True, CATALYST_FRACTION_MODE=False`) with a role-shaped "
        f"CSV for the full report, or wait for the fraction-mode narrative feature.",
        f"",
        f"## Top candidates",
        f"",
    ]
    hdr_cols = ["#", "Composition"] + [
        col for t in targets for col in (f"pred_{t}", f"pred_{t}_sd") if col in top.columns
    ]
    lines.append("| " + " | ".join(hdr_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr_cols)) + "|")
    for i, r in top.iterrows():
        cells = [str(i + 1), _fraction_composition_string(r.to_dict())]
        for t in targets:
            for col in (f"pred_{t}", f"pred_{t}_sd"):
                if col in top.columns:
                    v = r[col]
                    cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")

    # Summary stats of predictions across the full batch.
    lines.extend(["", "## Prediction summary (full batch)", ""])
    for t in targets:
        col = f"pred_{t}"
        if col not in candidates.columns:
            continue
        s = candidates[col].astype(float)
        sd_col = f"pred_{t}_sd"
        sd_mean = candidates[sd_col].astype(float).mean() if sd_col in candidates.columns else float("nan")
        lines.append(f"- **{t}**: pred min/median/max = {s.min():.4f} / "
                     f"{s.median():.4f} / {s.max():.4f}  (σ mean = {sd_mean:.4f})")

    body = "\n".join(lines)
    report_dir = getattr(config, "REPORTS_DIR", config.DATA_DIR / "reports")
    report_dir.mkdir(exist_ok=True)
    out_path = report_dir / f"report_{ts_file}.md"
    out_path.write_text(body, encoding="utf-8")
    log.info("Saved minimal fraction-mode report to %s.", out_path)

    dataset_name = str(getattr(config, "CSV_PATH", "") or getattr(config, "MATMINER_DATASET", "?"))
    return ReportBundle(
        path=out_path,
        ts=ts_str,
        dataset_source=getattr(config, "DATASET_SOURCE", "?"),
        dataset_name=Path(dataset_name).name if dataset_name else "?",
        targets=list(targets),
        n_samples=None,
        eda_summary_str=None,
        candidates_enriched=[],
        has_bnn=False,
        narrative_body=body,
        per_candidate_section="",
        figures=[],
    )


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

    # Atomic-fraction schema: role-based enrichment + per-candidate narrative
    # bullets don't apply. Emit a minimal report (candidates table + summary
    # stats) so the pipeline still produces an artifact without breaking on
    # missing `active_metal` columns. Full fraction-mode narrative + figures
    # are on the roadmap (probably arriving with DCP dataset integration).
    if getattr(config, "CATALYST_FRACTION_MODE", False) and "active_metal" not in candidates.columns:
        return _generate_fraction_minimal_report(candidates, eda_results, ts_str, ts_file)

    directions = getattr(config, "OPTIMIZATION_DIRECTIONS", ["max"] * len(config.TARGET_COLS))
    primary_dir = directions[0] if directions else "max"
    top = candidates.sort_values(f"pred_{config.TARGET_COLS[0]}",
                                 ascending=(primary_dir == "min")).head(config.REPORT_TOP_K)
    top_records = top.to_dict(orient="records")
    for row in top_records:
        row["_label"] = _format_catalyst_label(row)

    enriched_records = top_records
    sup_lookup_df: pd.DataFrame | None = None
    if config.CATALYST_MODE:
        try:
            sup_lookup_df = pd.read_csv(config.SUPPORT_LOOKUP_PATH)
            metal_lookup = pd.read_csv(config.METAL_LOOKUP_PATH)
            enriched_records = [
                _enrich_candidate(r, sup_lookup_df, metal_lookup) for r in top_records
            ]
        except Exception as e:
            log.warning("Could not enrich candidates with lookup props (%s).", e)

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
    for row in enriched_records:
        row["_family"] = _build_family(row, sup_lookup_df)
        row["_confidence"] = _build_confidence(row, list(config.TARGET_COLS), y_stds)

    agreement_summary = _build_agreement_summary(enriched_records, config.TARGET_COLS)

    eda_summary_str = None
    summary_path = config.DATA_DIR / "eda_summary.json"
    if summary_path.exists():
        eda_summary_str = summary_path.read_text()

    figures = _build_figures(candidates, enriched_records, eda_summary_str, ts_file)

    has_bnn = all(
        f"pred_{t}_bnn" in candidates.columns for t in config.TARGET_COLS
    )

    llm_narrative, captions = _ask_ollama(
        enriched_records, eda_summary_str, config.TARGET_COLS, figures,
        agreement_summary,
    )

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
    tmpl = Template(_TEMPLATE)
    md = tmpl.render(
        ts=ts_str,
        dataset_source=config.DATASET_SOURCE,
        dataset_name=config.MATMINER_DATASET if config.DATASET_SOURCE == "matminer" else str(config.CSV_PATH),
        label_kind="Catalyst" if getattr(config, "CATALYST_MODE", False) else "Composition",
        targets=config.TARGET_COLS,
        n_samples=config.SUBSAMPLE_N,
        top_k=config.REPORT_TOP_K,
        eda_summary=eda_summary_str,
        candidates=enriched_records,
        llm_narrative=llm_narrative,
        per_candidate_section=per_candidate_section,
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
        m_name = c.get("active_metal")
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
                chunks.append(f"Active {m_name}: {', '.join(bits)}.")

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
        for slot, key in (("active_metal", "active_metal"),
                          ("promoter_1", "promoter_1"),
                          ("promoter_2", "promoter_2"),
                          ("support", "support")):
            v = c.get(key)
            if v not in (None, "", "nan"):
                lines.append(f"  {slot}: {v}")
        for prop_key, prop_label in (
            ("support_properties", "support_properties"),
            ("metal_properties", "active_metal_properties"),
            ("promoter_1_properties", "promoter_1_properties"),
            ("promoter_2_properties", "promoter_2_properties"),
        ):
            props = c.get(prop_key)
            if props:
                kv = ", ".join(f"{k}={v}" for k, v in props.items())
                lines.append(f"  {prop_label}: {{{kv}}}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _ask_ollama(top_candidates: list[dict], eda_summary: str | None,
                target_cols: list[str], figures: list[FigureEntry],
                agreement_summary: dict | None = None) -> tuple[str, dict[str, str]]:
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
    agreement_block = (
        json.dumps(agreement_summary, indent=2)
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
        f"{json.dumps(feature_rankings, indent=2)}\n\n"
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
        "2. **Closing paragraph on surrogate confidence** — embed BOTH the "
        "GP-vs-BNN scatter AND (if available) the parity figure "
        "(figure_id=parity). Read the parity figure's description in (D) "
        "carefully: it tells you whether parity is k-fold-CV HELD-OUT (a "
        "generalization signal) or IN-SAMPLE (a debug signal only). Comment "
        "accordingly — for CV parity, talk about generalization to unseen "
        "catalysts; for in-sample, only talk about whether the surrogate can "
        "fit its training data, and explicitly note CV would be needed for a "
        "generalization claim. Then name the candidate with the smallest "
        "|delta| from (C). Use the exact label and index from (C)'s "
        "`best_agreement_candidate_label` field; do NOT name any other "
        "candidate as having the closest agreement.\n\n"
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
) -> list[FigureEntry]:
    """Generate static PNGs (for the saved MD) + Plotly figures (for the webui).
    Returns one FigureEntry per successfully-rendered figure. `ts` is the shared
    wall-clock stamp (YYYYmmdd_HHMMSS) — must match the report filename so the
    Markdown's image links don't drift off by a second."""
    directions = getattr(config, "OPTIMIZATION_DIRECTIONS", ["max"] * len(config.TARGET_COLS))
    from plots import (
        make_pareto_plot, make_feature_importance_plot,
        make_gp_vs_bnn_scatter, make_candidate_heatmap, make_parity_plot,
    )
    from interactive_plots import (
        make_pareto_figure, make_feature_importance_figure,
        make_gp_vs_bnn_figure, make_candidate_heatmap_figure,
        make_parity_figure,
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
