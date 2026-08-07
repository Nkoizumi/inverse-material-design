"""Single place that knows how to import from the separate auto-EDA project.

Both step 3 (CLI) and the web UI's Tab 2 need `pipeline.orchestrator` from
github.com/Nkoizumi/llm-eda-mobo, which is not a pip dependency and ships its
own `pipeline/` package — the same name as ours.

Two problems this exists to fix.

HARDCODED PATH. `webui/eda_tab.py` carried `_AUTO_EDA_PATH = "/home/nao/auto_eda"`,
so Tab 2's LLM feature could not work on anyone else's machine: the import fell
through to OUR `pipeline` package, raised, and was swallowed into a "could not
import" message. This is the same defect that was fixed in step3_eda earlier
(config.AUTO_EDA_PATH resolves $INVERSE_DESIGN_AUTO_EDA_PATH → a sibling
checkout → ~/auto_eda); that fix simply missed this second copy. Both call
sites now resolve through config.

PERMANENT SHADOWING. eda_tab did `sys.path.insert(0, ...)` at MODULE IMPORT
time and never removed it, so merely importing eda_tab put the other project
ahead of this one for the rest of the process. Both repositories also contain
an `app.py`, so `import app` after that resolves to theirs — which is exactly
what happened while reviewing this file.

The insertion is now scoped to the import itself. Note that `sys.modules` is
deliberately NOT purged afterwards: auto_eda's `pipeline` package stays cached,
which is fine because this project never imports its own `pipeline` as a
package — every internal import is flat (`import step1_load`), a convention
`webui/app.py` documents and relies on. Purging would risk duplicate module
objects for no gain.
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)


class AutoEDAUnavailable(RuntimeError):
    """auto-EDA is not installed, or its orchestrator could not be imported."""


def auto_eda_path() -> Path | None:
    """Resolved location of the auto-EDA checkout, or None."""
    try:
        import config
    except ImportError:
        return None
    path = getattr(config, "AUTO_EDA_PATH", None)
    return Path(path) if path else None


@contextmanager
def _path_prepended(path: Path):
    """Put `path` first on sys.path for the duration of the block only."""
    saved = list(sys.path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = saved


def load(attr: str, module: str = "pipeline.orchestrator"):
    """Import `attr` from auto-EDA's `module`.

    Raises AutoEDAUnavailable with an actionable message when the project is
    not present or the import fails — callers turn that into a skip, because
    auto-EDA is optional and the rest of the pipeline runs without it.
    """
    path = auto_eda_path()
    if path is None or not path.is_dir():
        raise AutoEDAUnavailable(
            "auto-EDA project not found. Set INVERSE_DESIGN_AUTO_EDA_PATH to a "
            "checkout of github.com/Nkoizumi/llm-eda-mobo, place one alongside "
            "this repository, or run with `--skip eda`."
        )
    try:
        with _path_prepended(path):
            mod = __import__(module, fromlist=[attr])
            return getattr(mod, attr)
    except Exception as e:
        raise AutoEDAUnavailable(
            f"Could not import {attr} from {module} at {path}: {e}"
        ) from e
