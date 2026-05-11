"""Sidebar: organ picker, new-organ wizard, reload + save controls."""
from __future__ import annotations

import streamlit as st

from ..loaders import load_organ
from ..save import save_organ
from ..validation import validate


def render_sidebar() -> None:
    """Draw the sidebar. Mutates ``st.session_state.organ_state``."""
    from digital_registrar_research.schemas import CASE_MODELS
    organs = sorted(CASE_MODELS.keys())

    st.sidebar.title("Schema editor")
    st.sidebar.caption("Three-layer per-organ schema (Pydantic + extraction + aliases)")

    state = st.session_state.get("organ_state")
    current_key = state.organ_key if state is not None else organs[0]

    picked = st.sidebar.selectbox(
        "Organ", organs,
        index=organs.index(current_key) if current_key in organs else 0,
    )

    cols = st.sidebar.columns(2)
    if cols[0].button("Load", use_container_width=True):
        st.session_state.organ_state = load_organ(picked)
        st.session_state.last_save_report = None
        st.rerun()

    if cols[1].button("Reload", use_container_width=True, disabled=state is None):
        if state is not None:
            st.session_state.organ_state = load_organ(state.organ_key)
            st.session_state.last_save_report = None
            st.rerun()

    st.sidebar.divider()
    _render_new_organ_form()

    st.sidebar.divider()
    _render_save_controls()


def _render_new_organ_form() -> None:
    from ..schema_gui_wrappers import create_new_organ  # local import to allow patching

    with st.sidebar.expander("➕ New organ", expanded=False):
        organ_key = st.text_input(
            "organ_key", key="new_organ_key",
            placeholder="e.g. skeletal",
            help="Lowercase ASCII; pattern ^[a-z][a-z0-9_]*$"
        )
        class_prefix = st.text_input(
            "class_prefix (optional)", key="new_class_prefix",
            placeholder="auto-derived from organ_key",
        )
        if st.button("Create", key="create_organ_btn"):
            if not organ_key.strip():
                st.warning("organ_key is required")
            else:
                try:
                    report = create_new_organ(
                        organ_key.strip(),
                        class_prefix=class_prefix.strip() or None,
                    )
                    st.success(f"Created {report.organ_key!r}. Loading…")
                    st.session_state.organ_state = load_organ(report.organ_key)
                    st.session_state.last_save_report = None
                    st.rerun()
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")


def _render_save_controls() -> None:
    state = st.session_state.get("organ_state")
    if state is None:
        st.sidebar.info("No organ loaded.")
        return

    report = validate(state)
    if report.errors:
        st.sidebar.error(f"⛔ {len(report.errors)} validation error(s) — fix to enable save")
        for e in report.errors[:5]:
            st.sidebar.caption(f"• {e}")
    if report.warnings:
        st.sidebar.warning(f"⚠ {len(report.warnings)} warning(s)")
    if state.layer1_read_only_reason:
        st.sidebar.info("Layer 1 read-only: " + state.layer1_read_only_reason[:80])

    if st.sidebar.button("💾 Save all layers", disabled=not report.ok(), use_container_width=True):
        with st.spinner("Rendering, smoke-importing, swapping, regenerating JSON…"):
            sr = save_organ(state)
        st.session_state.last_save_report = sr
        st.rerun()

    last = st.session_state.get("last_save_report")
    if last:
        if last.success:
            st.sidebar.success(
                f"Saved {len(last.files_written)} file(s)"
                + (" + JSON regenerated" if last.json_regenerated else "")
            )
        else:
            st.sidebar.error(last.error)
            if last.stderr:
                with st.sidebar.expander("stderr"):
                    st.code(last.stderr)
