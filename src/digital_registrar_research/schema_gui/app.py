"""Streamlit entry point for the schema editor.

Run via the console script ``registrar-schema-gui`` (see
``pyproject.toml``), which delegates to :func:`main_cli`. The CLI shells
out to ``streamlit run`` against this file.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    """Streamlit page renderer — called by ``streamlit run app.py``.

    Uses absolute imports because Streamlit execs this file as
    ``__main__``, not as a package member — relative imports would
    raise ``ImportError`` at script-runner load time.
    """
    import streamlit as st

    st.set_page_config(page_title="drr schema editor", layout="wide")

    from digital_registrar_research.schema_gui.views import (
        aliases_view,
        enums_view,
        extraction_view,
        fields_view,
        preview_view,
        sidebar,
    )

    sidebar.render_sidebar()

    state = st.session_state.get("organ_state")
    if state is None:
        st.title("drr schema editor")
        st.markdown(
            "Pick an organ in the sidebar and click **Load** to start editing. "
            "The three-layer schema architecture is documented in "
            "[`docs/architecture/schemas.md`](https://github.com/kblab2024/drr-next/blob/main/docs/architecture/schemas.md)."
        )
        return

    st.title(f"`{state.organ_key}` — `{state.class_prefix}CancerCase`")
    if state.layer1_read_only_reason:
        st.warning("Layer 1 read-only: " + state.layer1_read_only_reason)

    tabs = st.tabs([
        "Enums (L1)",
        "Fields (L1)",
        "Descriptions & groups (L2)",
        "Aliases (L3)",
        "Preview & save",
    ])
    with tabs[0]:
        enums_view.render(state)
    with tabs[1]:
        fields_view.render(state)
    with tabs[2]:
        extraction_view.render(state)
    with tabs[3]:
        aliases_view.render(state)
    with tabs[4]:
        preview_view.render(state)


def main_cli() -> int:
    """Console-script wrapper: launches ``streamlit run`` on this module.

    Mirrors the pattern used by ``registrar-annotate``. Returns the
    streamlit subprocess's exit code.
    """
    parser = argparse.ArgumentParser(
        prog="registrar-schema-gui",
        description="Streamlit GUI for editing the three-layer per-organ schema.",
    )
    parser.add_argument(
        "--port", type=int, default=8501,
        help="port for streamlit to bind (default: 8501)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="do not auto-open a browser",
    )
    args = parser.parse_args()

    import subprocess

    app_path = Path(__file__).resolve()
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
    ]
    if args.no_browser:
        cmd.extend(["--server.headless", "true"])
    env = dict(os.environ)
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    # When ``streamlit run app.py`` invokes this file, the script entry
    # point is ``main()`` not ``main_cli()``.
    main()
