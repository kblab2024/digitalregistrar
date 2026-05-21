"""Layer 1: field declarations + StagingSpec editor.

The case-model body is a flat list of declarations. The editor surfaces
each as a row: name, type annotation (free-text), kind (bare /
margin_spec / ln_spec / biomarker_spec / passthrough), and a JSON-shaped
options blob for the spec markers' kwargs. Default values for ``bare``
fields are constrained to literal Python expressions.

StagingSpec lives below the fields table as a dedicated form because it
has special structure (pt/pn/stage_groups must always be present).
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ..state import FieldDecl, FieldKind, OrganState, SpecOptions, StagingDecl

_KIND_OPTIONS: list[FieldKind] = [
    "bare", "margin_spec", "ln_spec", "biomarker_spec", "passthrough",
]


def render(state: OrganState) -> None:
    if state.layer1_read_only_reason:
        st.info(state.layer1_read_only_reason)
    disabled = bool(state.layer1_read_only_reason)

    st.subheader("Fields")
    st.caption(
        f"`{state.class_prefix}CancerCase` declares {len(state.fields)} field(s). "
        "`bare` fields are plain Pydantic; `*_spec` fields auto-expand via "
        "`@assemble_case_model`."
    )

    rows = []
    for fd in state.fields:
        rows.append({
            "name": fd.name,
            "type_expr": fd.type_expr,
            "kind": fd.kind,
            "default_or_kwargs": (
                json.dumps(fd.spec_options.kwargs) if fd.spec_options else fd.default_expr
            ),
        })
    df = pd.DataFrame(rows, columns=["name", "type_expr", "kind", "default_or_kwargs"])
    edited = st.data_editor(
        df, num_rows="dynamic", disabled=disabled,
        column_config={
            "kind": st.column_config.SelectboxColumn(options=_KIND_OPTIONS),
        },
        use_container_width=True, key="fields_table",
    )
    if not disabled:
        new_fields: list[FieldDecl] = []
        for _, row in edited.iterrows():
            name = (row.get("name") or "").strip() if isinstance(row.get("name"), str) else ""
            if not name:
                continue
            kind: FieldKind = row.get("kind") or "bare"
            type_expr = (row.get("type_expr") or "").strip()
            payload = row.get("default_or_kwargs") or ""
            if kind in ("margin_spec", "ln_spec", "biomarker_spec"):
                kwargs: dict[str, str] = {}
                if payload:
                    try:
                        loaded = json.loads(payload)
                        if isinstance(loaded, dict):
                            kwargs = {str(k): str(v) for k, v in loaded.items()}
                    except json.JSONDecodeError:
                        st.warning(
                            f"field {name!r}: kwargs JSON did not parse "
                            "(must be a JSON object) — keeping previous"
                        )
                        # Fall back to existing parse for this field if any.
                        for old in state.fields:
                            if old.name == name and old.spec_options:
                                kwargs = old.spec_options.kwargs
                                break
                new_fields.append(FieldDecl(
                    name=name, type_expr=type_expr, kind=kind,
                    spec_options=SpecOptions(kwargs=kwargs),
                ))
            else:
                new_fields.append(FieldDecl(
                    name=name, type_expr=type_expr, kind=kind,
                    default_expr=payload,
                ))
        state.fields = new_fields

    st.divider()
    st.subheader("Staging (`_STAGING: ClassVar = StagingSpec(...)`)")
    if state.staging is None:
        st.info("No staging spec on this organ.")
        if not disabled and st.button("➕ Add staging spec"):
            state.staging = StagingDecl(pt="", pn="", stage_groups={"stage_group": ""})
            st.rerun()
        return

    _render_staging_form(state.staging, disabled)


def _render_staging_form(s: StagingDecl, disabled: bool) -> None:
    cols = st.columns(2)
    s.pt = cols[0].text_input("pt= (enum)", value=s.pt, disabled=disabled, key="stg_pt")
    s.pn = cols[1].text_input("pn= (enum)", value=s.pn, disabled=disabled, key="stg_pn")

    cols = st.columns(2)
    s.pm = cols[0].text_input(
        "pm= (optional; default Literal[mx, m0, m1])",
        value=s.pm or "", disabled=disabled, key="stg_pm",
    ) or None
    s.tnm_descriptor = cols[1].text_input(
        "tnm_descriptor= (optional)",
        value=s.tnm_descriptor or "", disabled=disabled, key="stg_tnm",
    ) or None

    st.caption("stage_groups: rows of `field_name → enum class name`")
    sg_df = pd.DataFrame(
        [{"field_name": k, "enum": v} for k, v in s.stage_groups.items()],
        columns=["field_name", "enum"],
    )
    sg_edited = st.data_editor(
        sg_df, num_rows="dynamic", disabled=disabled,
        key="stg_groups_table",
    )
    if not disabled:
        new_groups: dict[str, str] = {}
        for _, row in sg_edited.iterrows():
            fn = (row.get("field_name") or "").strip() if isinstance(row.get("field_name"), str) else ""
            en = (row.get("enum") or "").strip() if isinstance(row.get("enum"), str) else ""
            if fn:
                new_groups[fn] = en
        s.stage_groups = new_groups

    cols = st.columns(3)
    s.include_tnm_descriptor = cols[0].checkbox(
        "include_tnm_descriptor", value=s.include_tnm_descriptor,
        disabled=disabled, key="stg_inc_tnm",
    )
    s.include_ajcc_version = cols[1].checkbox(
        "include_ajcc_version", value=s.include_ajcc_version,
        disabled=disabled, key="stg_inc_ajcc",
    )
    s.lean = cols[2].checkbox(
        "lean", value=s.lean, disabled=disabled, key="stg_lean",
    )
