"""The report model must not keep the GPU after step 6 is done.

Ollama and steps 4-5 share one device. Step 5's acquisition peaks at ~3.2 GB
*allocated* — small — but Ollama's default is to hold a model in VRAM for five
minutes after it answers, and step 6 is the last thing the pipeline does. So the
model that wrote the report is still resident when the *next* run starts, and
that run is the one that dies.

Measured 2026-08-08 on a 24 GB RTX 4090, `--preset acs_pdh`:

    resident model            free VRAM   result
    none                        23.3 GB   ok
    phi4:14b-q4_K_M  (10 GB)    13.4 GB   ok
    qwen3:32b-q4_K_M (20 GB)     2.7 GB   CUDA OOM in step 5

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which the OOM message itself
suggests, was tried and does not help: it cuts reserved-but-unallocated waste
from 636 MiB to 114 MiB, but 2.7 GB is genuinely below what the acquisition
needs. Unloading the model is the fix, not allocator tuning.

Both assertions below matter, and the first one is the subtle one: `_ask_ollama`
wraps its call in a bare `except Exception` and returns a placeholder narrative.
If `keep_alive` ever stops being a valid kwarg — a library signature change, a
typo — every report silently degrades to "*(LLM call failed: …)*" and the OOM
returns, with the pipeline still exiting 0 and nothing pointing back here.
"""
from __future__ import annotations

import config
import step6_report


def _fake_ollama(captured: dict):
    class _FakeOllama:
        @staticmethod
        def chat(model, messages, options=None, keep_alive=None):
            captured["model"] = model
            captured["keep_alive"] = keep_alive
            return {"message": {"content": "narrative"}}

    return _FakeOllama


def test_the_report_model_is_unloaded_so_the_next_run_gets_the_gpu(monkeypatch):
    captured: dict = {}
    monkeypatch.setitem(
        __import__("sys").modules, "ollama", _fake_ollama(captured)
    )

    narrative, _ = step6_report._ask_ollama([{"_label": "x"}], None, ["y"], [])

    assert "keep_alive" in captured, (
        "ollama.chat was never reached — _ask_ollama swallowed an exception "
        "and returned a placeholder report instead of raising"
    )
    assert "LLM call failed" not in narrative
    assert captured["keep_alive"] == config.REPORT_LLM_KEEP_ALIVE


def test_the_keep_alive_default_actually_frees_the_card(monkeypatch):
    """0 means "unload now". A non-zero default would leave the bug in place
    while looking configured.
    """
    assert config.REPORT_LLM_KEEP_ALIVE == 0


def test_a_duration_string_is_passed_through_untouched(monkeypatch):
    """Ollama accepts "5m"/-1 as well as seconds; the opt-out has to survive
    the trip rather than being coerced to an int.
    """
    captured: dict = {}
    monkeypatch.setattr(config, "REPORT_LLM_KEEP_ALIVE", "5m")
    monkeypatch.setitem(
        __import__("sys").modules, "ollama", _fake_ollama(captured)
    )

    step6_report._ask_ollama([{"_label": "x"}], None, ["y"], [])

    assert captured["keep_alive"] == "5m"
