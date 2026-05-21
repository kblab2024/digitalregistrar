"""Preview tab: side-by-side current vs rendered for each layer, plus validation."""
from __future__ import annotations

import streamlit as st

from ..loaders import _schemas_root
from ..state import OrganState
from ..validation import validate
from ..writers import render_layer1, render_layer2, render_layer3


def render(state: OrganState) -> None:
    report = validate(state)
    if report.errors:
        st.error(f"{len(report.errors)} validation error(s) — save blocked:")
        for e in report.errors:
            st.caption(f"• {e}")
    if report.warnings:
        with st.expander(f"⚠ {len(report.warnings)} warning(s)", expanded=False):
            for w in report.warnings:
                st.caption(f"• {w}")

    st.divider()
    root = _schemas_root()
    layer1_path = root / "pydantic" / f"{state.organ_key}.py"
    layer2_path = root / "extraction" / f"{state.organ_key}.py"
    layer3_path = root / "aliases" / f"{state.organ_key}.toml"

    _show_layer_diff(
        "Layer 1 (Pydantic)", layer1_path,
        render_layer1(state) if not state.layer1_read_only_reason else "(read-only)",
    )
    _show_layer_diff("Layer 2 (extraction)", layer2_path, render_layer2(state))
    _show_layer_diff(
        "Layer 3 (aliases TOML)", layer3_path,
        render_layer3(state),
    )


def _show_layer_diff(title: str, path, rendered: str) -> None:
    with st.expander(title, expanded=False):
        on_disk = path.read_text(encoding="utf-8") if path.is_file() else "(no file on disk)"
        cols = st.columns(2)
        cols[0].caption(f"On disk — {path.name}")
        cols[0].code(on_disk, language="python" if str(path).endswith(".py") else "toml")
        cols[1].caption("Rendered (would be saved)")
        cols[1].code(rendered, language="python" if str(path).endswith(".py") else "toml")
