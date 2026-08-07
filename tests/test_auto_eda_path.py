"""Step 3 must degrade gracefully when the separate auto-EDA project is absent.

The path was hardcoded to `/home/nao/auto_eda`, so step 3 could not run on any
machine but the author's — and the README only warned about needing Ollama.
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
import step3_eda


@pytest.fixture
def toy_df():
    return pd.DataFrame({"propane_TOF_log": [1.0, 2.0, 3.0], "feat": [0.1, 0.2, 0.3]})


def test_missing_auto_eda_project_skips_without_raising(toy_df, monkeypatch, caplog):
    monkeypatch.setattr(config, "AUTO_EDA_AVAILABLE", True)
    monkeypatch.setattr(config, "AUTO_EDA_PATH", None)
    with caplog.at_level("WARNING"):
        assert step3_eda.run_eda(toy_df) == {}
    assert "auto-EDA project not found" in caplog.text


def test_disabled_flag_skips_before_touching_the_path(toy_df, monkeypatch):
    monkeypatch.setattr(config, "AUTO_EDA_AVAILABLE", False)
    monkeypatch.delattr(config, "AUTO_EDA_PATH", raising=False)
    assert step3_eda.run_eda(toy_df) == {}


def test_config_exposes_auto_eda_path_attribute():
    """May be None (project not installed) but must always be defined, since
    step3 reads it and the webui surfaces EDA availability."""
    assert hasattr(config, "AUTO_EDA_PATH")
    assert config.AUTO_EDA_PATH is None or config.AUTO_EDA_PATH.is_dir()


def test_env_var_takes_precedence(tmp_path, monkeypatch):
    """Re-import config under a custom env var and confirm it wins."""
    import importlib
    target = tmp_path / "my_auto_eda"
    target.mkdir()
    monkeypatch.setenv("INVERSE_DESIGN_AUTO_EDA_PATH", str(target))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.AUTO_EDA_PATH == target
    finally:
        monkeypatch.delenv("INVERSE_DESIGN_AUTO_EDA_PATH", raising=False)
        importlib.reload(config)
