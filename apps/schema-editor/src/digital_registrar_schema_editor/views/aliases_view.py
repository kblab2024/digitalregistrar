"""Layer 3: surface-form aliases editor (TOML round-trip).

Each TOML table becomes an expander. Inside, rows are
``canonical_enum_value → comma-separated surface-form list``. Adding a
new table requires picking a field path from Layer 2's FIELD_META keys.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..state import AliasGroup, OrganState


def render(state: OrganState) -> None:
    st.subheader("Surface-form aliases")
    st.caption(
        "Maps canonical LowerStrEnum values to surface-form strings the "
        "LM might encounter. Edited in pure TOML round-trip mode — no "
        "Python edit, no JSON regen needed for alias-only changes."
    )

    if not state.aliases:
        st.info("No aliases on this organ yet.")

    for i, grp in enumerate(state.aliases):
        with st.expander(f"[{grp.field_path}] — {len(grp.entries)} alias(es)", expanded=False):
            df = pd.DataFrame(
                [{"canonical": k, "surface_forms": ", ".join(v)}
                 for k, v in grp.entries.items()],
                columns=["canonical", "surface_forms"],
            )
            edited = st.data_editor(
                df, num_rows="dynamic", use_container_width=True,
                column_config={
                    "surface_forms": st.column_config.TextColumn(width="large"),
                },
                key=f"aliases_{i}",
            )
            new_entries: dict[str, list[str]] = {}
            for _, row in edited.iterrows():
                canon = (row.get("canonical") or "").strip() if isinstance(row.get("canonical"), str) else ""
                if not canon:
                    continue
                raw = row.get("surface_forms") or ""
                forms = [s.strip() for s in raw.split(",") if s.strip()]
                if forms:
                    new_entries[canon] = forms
            grp.entries = new_entries
            if st.button(f"🗑 Remove table [{grp.field_path}]", key=f"rm_aliases_{i}"):
                state.aliases.pop(i)
                st.rerun()

    st.divider()
    field_paths = [r.path for r in state.field_meta]
    used = {g.field_path for g in state.aliases}
    available = [p for p in field_paths if p not in used]
    if available:
        new_path = st.selectbox(
            "Add alias table for field path:",
            options=[""] + available,
            key="new_alias_path",
        )
        if new_path and st.button("➕ Add", key="add_alias_btn"):
            state.aliases.append(AliasGroup(field_path=new_path, entries={}))
            st.rerun()
