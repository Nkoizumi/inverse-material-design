"""Importing from the auto-EDA project must not reorder sys.path permanently.

Two defects this covers.

`webui/eda_tab.py` hardcoded `sys.path.insert(0, "/home/nao/auto_eda")` at
MODULE IMPORT time. So Tab 2's LLM feature could not work on any other machine
— the import fell through to THIS project's `pipeline` package, raised, and was
swallowed into a "could not import" message. That is the same defect fixed in
step3_eda earlier; the fix missed this second copy.

And because the insert ran on import and was never removed, merely importing
eda_tab put the other project ahead of ours for the rest of the process. Both
repositories contain an `app.py`, so a later `import app` resolved to theirs.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import config
from auto_eda_bridge import AutoEDAUnavailable, auto_eda_path, load

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "webui") not in sys.path:
    sys.path.insert(0, str(ROOT / "webui"))


def _auto_eda_entries() -> list[str]:
    return [p for p in sys.path if "auto_eda" in p]


# ── no permanent shadowing ───────────────────────────────────────────────────
def test_importing_eda_tab_does_not_touch_sys_path():
    """The regression: `import eda_tab` used to prepend the other project."""
    pytest.importorskip("gradio")
    before = list(sys.path)
    sys.modules.pop("eda_tab", None)
    import eda_tab  # noqa: F401
    assert sys.path == before, (
        "importing eda_tab modified sys.path; it must not shadow this project"
    )


@pytest.mark.parametrize("name", [
    "webui/eda_tab.py", "pipeline/step3_eda.py", "pipeline/auto_eda_bridge.py",
])
def test_no_sys_path_insert_of_a_literal_absolute_path(name):
    """B1 was fixed in step3_eda but a second hardcoded copy survived here.

    Checks the dangerous CONSTRUCT via AST rather than grepping for the string,
    so prose in a docstring explaining the old bug does not trip it — and so a
    different hardcoded path would be caught too.
    """
    tree = ast.parse((ROOT / name).read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "insert"
                and isinstance(f.value, ast.Attribute) and f.value.attr == "path"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    and arg.value.startswith("/"):
                offenders.append(f"line {node.lineno}: {arg.value}")
    assert not offenders, (
        f"{name} inserts a literal absolute path onto sys.path: {offenders}. "
        f"Resolve it through config.AUTO_EDA_PATH via auto_eda_bridge instead."
    )


def test_load_restores_sys_path_on_success():
    if auto_eda_path() is None or not auto_eda_path().is_dir():
        pytest.skip("auto-EDA project not installed")
    before = list(sys.path)
    load("Orchestrator")
    assert sys.path == before
    assert not _auto_eda_entries()


def test_load_restores_sys_path_on_failure(monkeypatch):
    monkeypatch.setattr(config, "AUTO_EDA_PATH", "/definitely/not/here")
    before = list(sys.path)
    with pytest.raises(AutoEDAUnavailable):
        load("Orchestrator")
    assert sys.path == before


# ── clear failures rather than swallowed ones ────────────────────────────────
def test_missing_project_raises_with_actionable_message(monkeypatch):
    monkeypatch.setattr(config, "AUTO_EDA_PATH", None)
    with pytest.raises(AutoEDAUnavailable) as e:
        load("Orchestrator")
    msg = str(e.value)
    assert "INVERSE_DESIGN_AUTO_EDA_PATH" in msg
    assert "--skip eda" in msg


def test_missing_symbol_raises_rather_than_returning_none(monkeypatch):
    if auto_eda_path() is None or not auto_eda_path().is_dir():
        pytest.skip("auto-EDA project not installed")
    with pytest.raises(AutoEDAUnavailable):
        load("NoSuchSymbol")


def test_path_resolution_comes_from_config(monkeypatch, tmp_path):
    target = tmp_path / "somewhere"
    target.mkdir()
    monkeypatch.setattr(config, "AUTO_EDA_PATH", target)
    assert auto_eda_path() == target


# ── step 3 degrades rather than failing ──────────────────────────────────────
def test_step3_skips_cleanly_when_unavailable(monkeypatch, caplog):
    import pandas as pd

    import step3_eda
    monkeypatch.setattr(config, "AUTO_EDA_AVAILABLE", True)
    monkeypatch.setattr(config, "AUTO_EDA_PATH", None)
    df = pd.DataFrame({"propane_TOF_log": [1.0, 2.0, 3.0], "f": [0.1, 0.2, 0.3]})
    with caplog.at_level("WARNING"):
        assert step3_eda.run_eda(df) == {}
    assert "auto-EDA project not found" in caplog.text
