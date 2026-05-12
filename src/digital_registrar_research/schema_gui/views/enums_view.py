"""Layer 1: enum classes editor.

Each ``LowerStrEnum`` subclass becomes a section: rename the class,
edit its UPPER_SNAKE value list. Adding/removing a value here directly
affects the JSON schema's enum list on next save.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..state import EnumDecl, OrganState


def render(state: OrganState) -> None:
    if state.layer1_read_only_reason:
        st.info(state.layer1_read_only_reason)
        st.caption("You can still browse the enums below, but edits are not persisted.")

    disabled = bool(state.layer1_read_only_reason)
    st.markdown(f"**{len(state.enums)} enum class(es)** in `{state.class_prefix}CancerCase`")

    for idx, enum in enumerate(state.enums):
        with st.expander(f"`{enum.name}` ({len(enum.values)} value(s))", expanded=False):
            new_name = st.text_input(
                "Class name", value=enum.name,
                key=f"enum_name_{idx}", disabled=disabled,
            )
            if new_name != enum.name and not disabled:
                enum.name = new_name

            df = pd.DataFrame({"value": enum.values})
            edited = st.data_editor(
                df, num_rows="dynamic",
                key=f"enum_values_{idx}", disabled=disabled,
                use_container_width=True,
            )
            if not disabled:
                values = [v for v in edited["value"].tolist() if isinstance(v, str) and v.strip()]
                enum.values = values

    if not disabled and st.button("➕ Add enum class"):
        state.enums.append(EnumDecl(name="NewEnum", values=["NOT_STATED"]))
        st.rerun()
