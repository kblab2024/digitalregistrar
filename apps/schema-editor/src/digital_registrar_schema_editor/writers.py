"""Render :class:`OrganState` back to Layer 1 / Layer 2 / Layer 3 source.

Layer 1 (Pydantic source) is rebuilt from the structured state plus the
verbatim ``imports_source`` and ``passthrough_blocks`` captured at load
time. Layer 2 (extraction module) is rebuilt entirely from state. Layer
3 (aliases TOML) is rebuilt entirely from state.

All writers produce text only; the save flow in :mod:`.save` is
responsible for atomic file replacement and post-write formatting.
"""
from __future__ import annotations

from io import StringIO

from .state import OrganState

_AUTO_BANNER = (
    "# Edited via registrar-schema-gui. Hand-edit any field freely;\n"
    "# the GUI will round-trip its content on next load.\n"
)


# ---------------------------------------------------------------------------
# Layer 1: schemas/pydantic/<organ>.py
# ---------------------------------------------------------------------------


def render_layer1(state: OrganState) -> str:
    """Re-emit the Pydantic case-model file.

    Layout: module docstring → imports (verbatim from load) → enum
    classes → passthrough blocks (e.g. hand-authored nested BaseModels) →
    ``@assemble_case_model()`` class. Trailing newline preserved so the
    file matches Black's expectation.
    """
    out = StringIO()

    # Module docstring.
    if state.module_docstring:
        out.write(f'"""{state.module_docstring}"""\n')

    # Imports block — verbatim.
    if state.imports_source:
        out.write(state.imports_source)
        if not state.imports_source.endswith("\n"):
            out.write("\n")

    # Enum classes.
    out.write("\n")
    out.write("# --- Per-organ closed vocabularies ----------------------------------------\n")
    for enum in state.enums:
        out.write("\n\n")
        out.write(f"class {enum.name}(LowerStrEnum):\n")
        if not enum.values:
            out.write("    pass\n")
            continue
        for value in enum.values:
            out.write(f"    {value} = auto()\n")

    # Passthrough blocks — hand-authored sections preserved verbatim.
    for block in state.passthrough_blocks:
        out.write("\n\n")
        out.write(block)
        if not block.endswith("\n"):
            out.write("\n")

    # Case-model class.
    out.write("\n\n")
    out.write("# --- Case-model -----------------------------------------------------------\n\n\n")
    out.write("@assemble_case_model()\n")
    out.write(f"class {state.class_prefix}CancerCase(BaseModel):\n")
    if state.case_model_docstring:
        out.write(f'    """{state.case_model_docstring}"""\n')

    body_parts: list[str] = []
    for fd in state.fields:
        body_parts.append(_render_field_decl(fd))
    if state.staging is not None:
        body_parts.append(_render_staging_block(state.staging))

    if not body_parts:
        out.write("    pass\n")
    else:
        out.write("\n")
        out.write("\n".join(body_parts))
        if not body_parts[-1].endswith("\n"):
            out.write("\n")

    return out.getvalue()


def _render_field_decl(fd) -> str:
    """Render one field declaration line."""
    if fd.kind == "bare":
        if fd.default_expr:
            return f"    {fd.name}: {fd.type_expr} = {fd.default_expr}\n"
        return f"    {fd.name}: {fd.type_expr}\n"
    if fd.kind in ("margin_spec", "ln_spec", "biomarker_spec"):
        marker_name = {
            "margin_spec": "MarginSpec",
            "ln_spec": "LNSpec",
            "biomarker_spec": "BiomarkerSpec",
        }[fd.kind]
        kwargs = fd.spec_options.kwargs if fd.spec_options else {}
        if not kwargs:
            return f"    {fd.name}: {fd.type_expr} = {marker_name}()\n"
        # Keep argument order as authored; emit on one line when short,
        # break onto multiple lines when long. Threshold is conservative
        # so Black will leave it alone.
        joined = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        single = f"    {fd.name}: {fd.type_expr} = {marker_name}({joined})\n"
        if len(single) <= 100:
            return single
        # Multiline form.
        lines = [f"    {fd.name}: {fd.type_expr} = {marker_name}("]
        for k, v in kwargs.items():
            lines.append(f"        {k}={v},")
        lines.append("    )")
        return "\n".join(lines) + "\n"
    # passthrough — emit the captured default expression verbatim.
    if fd.default_expr:
        return f"    {fd.name}: {fd.type_expr} = {fd.default_expr}\n"
    return f"    {fd.name}: {fd.type_expr}\n"


def _render_staging_block(s) -> str:
    """Render the ``_STAGING: ClassVar = StagingSpec(...)`` block."""
    parts: list[str] = ["    _STAGING: ClassVar = StagingSpec("]
    parts.append(f"        pt={s.pt},")
    parts.append(f"        pn={s.pn},")
    if s.pm is not None:
        parts.append(f"        pm={s.pm},")
    if s.tnm_descriptor is not None:
        parts.append(f"        tnm_descriptor={s.tnm_descriptor},")
    if s.stage_groups:
        sg_inner = ", ".join(
            f'"{k}": {v}' for k, v in s.stage_groups.items()
        )
        parts.append(f"        stage_groups={{{sg_inner}}},")
    if s.group is not None:
        parts.append(f'        group="{s.group}",')
    if not s.include_ajcc_version:
        parts.append("        include_ajcc_version=False,")
    if not s.include_tnm_descriptor:
        parts.append("        include_tnm_descriptor=False,")
    if s.lean is False:
        parts.append("        lean=False,")
    else:
        parts.append("        lean=True,")
    parts.append("    )")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Layer 2: schemas/extraction/<organ>.py
# ---------------------------------------------------------------------------


def render_layer2(state: OrganState) -> str:
    """Re-emit the extraction-metadata module from state.

    Loses the file's original comments (the ``# --- nonnested ---`` style
    section dividers) but preserves all data. The header docstring is
    canonical — points at the sibling files and warns about
    ``value_hints``.
    """
    out = StringIO()
    out.write(f'"""Extraction metadata for {state.organ_key} cancer (Layer 2 of 3).\n')
    out.write("\n")
    out.write("Edit this file for per-field ``desc`` strings (LM-facing) and\n")
    out.write("``GROUP_INSTRUCTIONS`` (per-decomposition-group framing).\n")
    out.write("\n")
    out.write(f"Surface-form aliases live in ``schemas/aliases/{state.organ_key}.toml``.\n")
    out.write('"""\n')
    out.write("from __future__ import annotations\n\n")
    out.write("from ..pydantic._factory_helpers import DEFAULT_GROUP_INSTRUCTION\n")
    out.write("from . import FieldMeta\n\n\n")

    # GROUP_INSTRUCTIONS
    out.write("GROUP_INSTRUCTIONS: dict[str, str] = {\n")
    for gi in state.group_instructions:
        body = _py_string_literal(gi.instruction)
        if gi.has_default_suffix:
            out.write(f"    {_py_string_literal(gi.group_name)}: {body} + \" \" + DEFAULT_GROUP_INSTRUCTION,\n")
        else:
            out.write(f"    {_py_string_literal(gi.group_name)}: {body},\n")
    out.write("}\n\n\n")

    # FIELD_META
    out.write("FIELD_META: dict[str, FieldMeta] = {\n")
    for row in state.field_meta:
        path_lit = _py_string_literal(row.path)
        desc_lit = _py_string_literal(row.desc)
        if row.group:
            group_lit = _py_string_literal(row.group)
            out.write(f"    {path_lit}: {{\"group\": {group_lit}, \"desc\": {desc_lit}}},\n")
        else:
            out.write(f"    {path_lit}: {{\"desc\": {desc_lit}}},\n")
    out.write("}\n\n\n")

    out.write('__all__ = ["FIELD_META", "GROUP_INSTRUCTIONS"]\n')
    return out.getvalue()


def _py_string_literal(s: str) -> str:
    """Emit a Python string literal that round-trips ``s`` exactly.

    Prefers double-quotes; falls back to single-quotes if the value
    contains a double-quote and no single-quote. Newlines and other
    control chars use ``repr`` to stay safe.
    """
    if "\n" in s or "\\" in s or "\t" in s:
        return repr(s)
    if '"' not in s:
        return f'"{s}"'
    if "'" not in s:
        return f"'{s}'"
    return repr(s)


# ---------------------------------------------------------------------------
# Layer 3: schemas/aliases/<organ>.toml
# ---------------------------------------------------------------------------


def render_layer3(state: OrganState) -> str:
    """Re-emit the aliases TOML.

    Custom emitter (rather than ``tomli_w``) so the output style matches
    the existing TOMLs: one table per field path, one alias per line,
    aligned ``=``. The original file's prose comments are lost — a
    single auto-generated banner replaces them.
    """
    out = StringIO()
    out.write(f"# Surface-form aliases for {state.organ_key} (Layer 3 of 3 — data-only).\n")
    out.write("# Edited via registrar-schema-gui. Each table is a dotted FIELD_META\n")
    out.write("# path; each key inside is a canonical LowerStrEnum value mapped to a\n")
    out.write("# list of surface forms the LM might encounter.\n")
    if not state.aliases:
        return out.getvalue()
    for grp in state.aliases:
        out.write("\n")
        out.write(f'["{grp.field_path}"]\n')
        if not grp.entries:
            continue
        # Right-pad the canonical key column for readability.
        max_key = max(len(k) for k in grp.entries.keys())
        for canonical, surface_forms in grp.entries.items():
            forms = ", ".join(_toml_string_literal(s) for s in surface_forms)
            out.write(f"{canonical.ljust(max_key)} = [{forms}]\n")
    return out.getvalue()


def _toml_string_literal(s: str) -> str:
    """Basic TOML string literal: double-quoted with backslash escaping."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


__all__ = ["render_layer1", "render_layer2", "render_layer3"]
