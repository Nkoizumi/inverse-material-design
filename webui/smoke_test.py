"""Drive the Gradio app through gradio_client to verify the full pipeline
end-to-end via the same HTTP path the browser uses."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from gradio_client import Client

import os
BASE_URL = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:7860/")


def main() -> int:
    c = Client(BASE_URL)
    print(f"Connected to {BASE_URL}")

    # ── Tab 1: load synthetic data ──────────────────────────────────────
    print("\n[1] /load_csv with use_synthetic=True")
    t = time.time()
    preview, target_choices, msg = c.predict(
        file_obj=None,
        use_synthetic=True,
        api_name="/load_csv",
    )
    print(f"    {time.time() - t:.1f}s — {msg.splitlines()[0]}")
    assert preview, "preview is empty"
    assert "Loaded" in msg, f"expected 'Loaded' in msg; got: {msg}"
    print(f"    target columns offered: {target_choices['value']}")

    # ── Tab 2: estimate library size ────────────────────────────────────
    print("\n[2] /estimate_library_size with default config")
    t = time.time()
    size_msg = c.predict(
        metals=["Pt", "Pd"],
        promoters_1=["(none)", "Sn"],
        promoters_2=["(none)"],
        metal_loads_str="0.5, 1.0",
        promo_loads_str="0.5, 1.0",
        api_name="/estimate_library_size",
    )
    print(f"    {time.time() - t:.1f}s — {size_msg}")
    assert "Estimated library size" in size_msg, f"unexpected: {size_msg}"

    # ── Tab 3: run pipeline end-to-end ──────────────────────────────────
    print("\n[3] /run_pipeline (this will take ~60-90s with BNN + LLM)")
    t = time.time()
    result = c.predict(
        file_obj=None,
        use_synthetic=True,
        target_cols=["propane_TOF_log", "propane_selectivity"],
        metals=["Pt", "Pd", "Ir", "Ru"],
        promoters_1=["(none)", "Sn", "Re"],
        promoters_2=["(none)", "Ag"],
        metal_loads_str="0.5, 1.0, 2.0",
        promo_loads_str="0.5, 1.0",
        surrogate_kind="both",
        bo_batch_size=4,
        report_llm_model="phi4:14b-q4_K_M",
        api_name="/run_pipeline",
    )
    elapsed = time.time() - t
    print(f"    {elapsed:.1f}s — pipeline complete")

    # The endpoint now returns at minimum (progress_log, report_file).
    # gr.State outputs (bundle, candidates) aren't exposed by the HTTP API.
    if isinstance(result, (tuple, list)):
        if len(result) == 2:
            progress_log, report_file = result
        else:
            progress_log = result[0]
            report_file = result[1] if len(result) > 1 else None
    else:
        progress_log = str(result)
        report_file = None

    print(f"    progress log tail:")
    for line in progress_log.strip().split("\n")[-6:]:
        print(f"      {line}")

    assert report_file, "no report file returned"
    report_path = report_file["url"] if isinstance(report_file, dict) else report_file
    print(f"    report file: {report_path}")

    # Find the most-recently-written MD report on disk and inspect it. The
    # interactive figures live in the webui only; the saved MD is the durable
    # record we can verify here.
    from pathlib import Path as _P
    reports_dir = _P(__file__).resolve().parents[1] / "reports"
    md_files = sorted(reports_dir.glob("report_*.md"))
    assert md_files, "no report MD files written"
    latest = md_files[-1]
    report_md = latest.read_text()
    print(f"    on-disk report: {latest.name} ({len(report_md):,} chars)")

    assert "## Why these materials are promising" in report_md, (
        "report missing narrative section"
    )
    assert "## Figure interpretations (LLM)" in report_md, (
        "report missing per-figure LLM caption section"
    )

    # ── Quick sanity counts in the report ──────────────────────────────
    n_pareto = report_md.count("pareto_")
    n_feat = report_md.count("feature_importance_")
    n_heat = report_md.count("candidate_heatmap_")
    n_gp_bnn = report_md.count("gp_vs_bnn_")
    print(f"    figure mentions: Pareto={n_pareto}  FI={n_feat}  heatmap={n_heat}  GPvsBNN={n_gp_bnn}")
    assert n_pareto >= 1 and n_feat >= 1 and n_heat >= 1 and n_gp_bnn >= 1, (
        "at least one figure missing from report"
    )

    print("\n✓ Smoke test passed end-to-end through Gradio HTTP API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
