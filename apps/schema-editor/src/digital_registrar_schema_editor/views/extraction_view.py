"""Layer 2: GROUP_INSTRUCTIONS + FIELD_META editor."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..state import FieldMetaRow, GroupInstruction, OrganState


def render(state: OrganState) -> None:
    st.subheader("Group instructions")
    st.caption(
        "Group ordering controls extraction order. Toggle "
        "`add_default_suffix` to append `+ DEFAULT_GROUP_INSTRUCTION` "
        "at write time."
    )

    gi_df = pd.DataFrame(
        [{
            "group_name": g.group_name,
            "instruction": g.instruction,
            "add_default_suffix": g.has_default_suffix,
        } for g in state.group_instructions],
        columns=["group_name", "instruction", "add_default_suffix"],
    )
    gi_edited = st.data_editor(
        gi_df, num_rows="dynamic", use_container_width=True,
        key="group_instructions_table",
    )
    new_groups: list[GroupInstruction] = []
    for _, row in gi_edited.iterrows():
        name = (row.get("group_name") or "").strip() if isinstance(row.get("group_name"), str) else ""
        if not name:
            continue
        new_groups.append(GroupInstruction(
            group_name=name,
            instruction=row.get("instruction") or "",
            has_default_suffix=bool(row.get("add_default_suffix", True)),
        ))
    state.group_instructions = new_groups

    group_names = [g.group_name for g in state.group_instructions]

    st.divider()
    st.subheader("Field metadata")
    st.caption(
        "Dotted paths refer to nested sub-fields (e.g. `margins.distance`). "
        "Top-level fields must have a `group`; sub-fields leave it empty."
    )

    fm_df = pd.DataFrame(
        [{
            "path": r.path, "group": r.group or "", "desc": r.desc,
        } for r in state.field_meta],
        columns=["path", "group", "desc"],
    )
    fm_edited = st.data_editor(
        fm_df, num_rows="dynamic", use_container_width=True,
        column_config={
            "group": st.column_config.SelectboxColumn(
                options=[""] + group_names,
                help="Empty for nested sub-fields; required for top-level",
            ),
            "desc": st.column_config.TextColumn(width="large"),
        },
        key="field_meta_table",
    )
    new_meta: list[FieldMetaRow] = []
    for _, row in fm_edited.iterrows():
        path = (row.get("path") or "").strip() if isinstance(row.get("path"), str) else ""
        if not path:
            continue
        new_meta.append(FieldMetaRow(
            path=path,
            desc=row.get("desc") or "",
            group=(row.get("group") or None) or None,
        ))
    state.field_meta = new_meta
