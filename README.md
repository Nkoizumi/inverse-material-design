# Inverse Material Design

End-to-end pipeline for inverse catalyst design:
**CSV → Matminer featurization → auto-EDA → GP/BNN surrogate → BO/MOBO → local-LLM scientific report.**

Ships with a Gradio web UI, three worked examples, and honest 5-fold CV
throughout. Validated on real propane-dehydrogenation (PDH) catalyst data,
where the pipeline recovers textbook chemistry (Ga-Mo top-yield,
Mg-modified low-deactivation) on the published ACS Materials Letters dataset.

## Pipeline

| step | module | description |
|---|---|---|
| 1 | `pipeline/step1_load.py` | Load CSV or matminer benchmark dataset. |
| 2 | `pipeline/step2_featurize.py` | Formula → high-dim vectors (Matminer). Dispatches role-based catalyst, atomic-fraction catalyst, or single-formula featurization. |
| 3 | `pipeline/step3_eda.py` | Optional auto-EDA via a local Ollama ensemble (`auto_eda` module). |
| 4 | `pipeline/step4_surrogate.py` | GP (BoTorch, exact SingleTaskGP or MultiTaskGP) + Bayesian NN (Pyro). 5-fold CV parity by default. |
| 5 | `pipeline/step5_inverse.py` | qLogNEI (single-objective) or qLogNEHVI (multi-objective) discrete BO. Family-diversity cap available for balanced experimental handoff. |
| 6 | `pipeline/step6_report.py` | Local LLM (Ollama) writes a scientific report on the top candidates. |

## Install

```bash
conda create -n inverse_material_design python=3.12 -y
conda activate inverse_material_design
pip install -r requirements.txt   # see the file for the full stack

# For the local LLM report step (optional):
# install Ollama and pull a model
ollama pull phi4
```

## Three worked examples

The pipeline ships with three presets so you can benchmark it on data of
increasing chemical realism.

### 1. Synthetic catalyst — Quick start (mechanics demo)

Full 6-step pipeline + web UI end-to-end on 72 synthetic Pt-based PDH
catalysts. Every stage runs on your machine; no external data downloads.

```bash
# CLI (steps 1, 2, 4, 5, 6 — skip auto-EDA)
python run_pipeline.py --preset synthetic_catalyst --skip eda

# Gradio web UI (full 4-tab experience)
python webui/app.py
```

Result: report + top-20 candidate table + interactive Pareto/parity/heatmap
figures. ~7 min end-to-end including the LLM narrative.

### 2. matbench_steels — Reproducible benchmark

312 steel alloys, single-target yield strength (MPa). Composition-only
regression. Reproduces published matbench numbers.

```bash
python run_pipeline.py --preset matbench_steels --skip eda report
```

**Expected result:** GP 5-fold CV R² = 0.690, MAE = 109 MPa (matbench
leaderboard best ~95 MPa). Runtime: ~30 s.

### 3. ACS Materials Letters PDH — Real-data validation *(requires paper access)*

210 published PDH catalysts, atomic-fraction schema. Targets: max propylene
yield, min deactivation-rate log.

> **Note.** The source paper is not open-access, so the CSV is **not
> bundled** with this repository. If you have institutional access to
> [ACS Materials Letters **6**(11), 5138–5145 (2024)](https://doi.org/10.1021/acsmaterialslett.4c01367),
> download the SI table manually and place it at
> `data/pdh_ACSMaterialsLetters.csv` — `python scripts/download_acs_pdh.py`
> prints exact instructions. Users without paper access can skip this
> example; the synthetic and matbench examples above cover the full
> pipeline mechanics without external data.

```bash
python scripts/download_acs_pdh.py            # prints manual-download instructions + citation
# ... place data/pdh_ACSMaterialsLetters.csv ...
python run_pipeline.py --preset acs_pdh --skip eda report
```

**Honest 5-fold CV numbers reproduced in this repository (n=210, MAX_GP_FEATURES=30):**

| target | GP R² | GP MAE | direction |
|---|---|---|---|
| propylene_yield | +0.541 | 0.062 | max |
| deactivation_rate_log | +0.325 | 0.332 | min |

**BO chemistry validation:** on the K=30 GP surrogate with family cap = 4,
the top-20 candidates cover 9 non-support-element families. Top-yield
candidates cluster in the Ga-Mo family (textbook PDH activity); the
lowest-deactivation candidates are Mg-modified γ-Al₂O₃ (textbook
low-acidity, low-coking). Both patterns match the qualitative conclusions
of the source paper — an independent recovery from BO on the same data.

**Cite the paper** when using this dataset (see `CITATION.cff`).

## Web UI

`python webui/app.py` launches a 5-tab Gradio app. The **synthetic
catalyst** dataset works through all tabs end-to-end. For the ACS
atomic-fraction dataset, Tab 5 (Explore) auto-detects the schema and
renders a sortable candidate table (role-based filters degrade to no-ops),
and step 6 emits a **minimal report** — the full role-based narrative +
figures apply to role-based catalysts only. A fraction-mode narrative /
figure path is on the roadmap.

## Configuration

Everything is in `config.py`. Key knobs:

- `TARGET_COLS`, `OPTIMIZATION_DIRECTIONS` — what to optimize, in which direction.
- `CATALYST_MODE` / `CATALYST_FRACTION_MODE` — schema of the input CSV.
- `MAX_GP_FEATURES` — Pearson-|r| cap on features fed to the GP (default 30).
- `BO_MAX_PER_FAMILY` — family-diversity cap for the BO batch (default 4 for fraction mode; disabled for role-based).
- `SURROGATE_KIND` — `gp` (default), `bnn`, or `both`.
- `CV_FOLDS` — 5-fold CV for parity plots (0 to disable).

Presets in `run_pipeline.py` override these in-process — see the `PRESETS`
dict for exact settings.

## Layout

```
inverse_material_design/
├── config.py              # central configuration
├── pipeline/              # six pipeline steps + featurizers + library builders
├── run_pipeline.py        # CLI driver + presets
├── webui/                 # Gradio web UI
├── scripts/               # download scripts, validation utilities
├── data/                  # cached datasets + featurized DataFrames
├── data/lookups/          # curated metal / support property tables
├── reports/               # generated reports (gitignored)
└── checkpoints/           # surrogate model state (gitignored)
```

## Data provenance

- **`data/synthetic_catalysts.csv`** — generated for pipeline validation.
  Freely usable.
- **`data/lookups/metal_properties.csv`, `support_properties.csv`** —
  curated by the author from public references (work functions, Pauling
  electronegativities, d-band centers, adsorbate binding energies from OC20
  / Hammer-Nørskov literature). Cite the underlying sources noted in
  `scripts/extract_oc20_features.py` when using these tables.
- **`data/pdh_literature.csv`, `pdh_template.csv`** — digitized from
  peer-reviewed PDH papers; per-row DOIs are in the `reference_doi` column.
- **`data/pdh_ACSMaterialsLetters.csv`** — **not bundled**. Download from
  the ACS supplementary information via `scripts/download_acs_pdh.py`.
  Cite the original paper when using this dataset (see `CITATION.cff`).

## Known limitations

- Atomic-fraction schema: web-UI Tab 5 (Explore) filters degrade to
  no-ops; step 6 emits a minimal report (no LLM narrative or per-candidate
  chemistry section).
- PDH literature dataset (n=85) is data-limited. Cannot separate
  composition from reaction conditions cleanly at this sample size. Use
  ACS (n=210) or wait for the DCP dataset for anything beyond exploration.
- The auto-EDA step relies on a local Ollama daemon at
  `http://localhost:11434`. Skip with `--skip eda` if you don't have one.

## Citation

If you use this software, please cite via `CITATION.cff`. If you use the
bundled ACS PDH dataset, also cite the original ACS Materials Letters
paper — see `scripts/download_acs_pdh.py`.

## License

Apache-2.0. See `LICENSE`.
