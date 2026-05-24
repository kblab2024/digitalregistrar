"""Parse on-disk schema files (Layers 1/2/3) into editable :class:`OrganState`.

Layer 1 uses ``ast`` to recover the structured shape. The loader recognises
the canonical v2 layout (``LowerStrEnum`` subclasses + an
``@assemble_case_model()``-decorated ``<Organ>CancerCase`` BaseModel) and
sets ``layer1_read_only_reason`` if the file uses constructs the GUI can't
faithfully round-trip (e.g. direct ``make_ln_type`` calls, hand-authored
inline ``BaseModel`` subclasses other than the case-model).

Layer 2 imports the extraction module and reads its ``GROUP_INSTRUCTIONS``
and ``FIELD_META`` dicts directly — these are pure data so no AST work
needed.

Layer 3 uses ``tomllib`` to load the alias table-of-tables.
"""
from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

from digital_registrar.schemas.pydantic._factory_helpers import DEFAULT_GROUP_INSTRUCTION

from .state import (
    AliasGroup,
    EnumDecl,
    FieldDecl,
    FieldMetaRow,
    GroupInstruction,
    OrganState,
    SpecOptions,
    StagingDecl,
)

_SPEC_MARKER_NAMES = {"MarginSpec", "LNSpec", "BiomarkerSpec"}
_SPEC_KIND_BY_NAME = {
    "MarginSpec": "margin_spec",
    "LNSpec": "ln_spec",
    "BiomarkerSpec": "biomarker_spec",
}


def _schemas_root() -> Path:
    """Resolve the live ``schemas/`` directory on disk.

    Computed from the installed module path so the GUI works in editable
    and non-editable installs. A future feature could let the user point
    at a sandbox tree; today we always edit the live schemas.
    """
    from digital_registrar.schemas import generate as _gen
    return Path(_gen.__file__).resolve().parent


def load_organ(organ_key: str) -> OrganState:
    """Load all three layers for ``organ_key`` into a fresh :class:`OrganState`."""
    root = _schemas_root()
    layer1_path = root / "pydantic" / f"{organ_key}.py"
    layer2_path = root / "extraction" / f"{organ_key}.py"
    layer3_path = root / "aliases" / f"{organ_key}.toml"

    state = _load_layer1(layer1_path, organ_key)
    _load_layer2(state, layer2_path)
    _load_layer3(state, layer3_path)
    return state


# ---------------------------------------------------------------------------
# Layer 1
# ---------------------------------------------------------------------------


def _load_layer1(path: Path, organ_key: str) -> OrganState:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    state = OrganState(organ_key=organ_key, class_prefix="")
    state.layer1_mtime_at_load = path.stat().st_mtime

    # Module docstring.
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        state.module_docstring = tree.body[0].value.value

    # Imports block: collect every top-level Import / ImportFrom node and
    # take the contiguous source segment covering them. Anything between
    # the docstring and the first class def goes into ``imports_source``.
    import_nodes = [
        n for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    if import_nodes:
        start = import_nodes[0].lineno
        end = import_nodes[-1].end_lineno or import_nodes[-1].lineno
        lines = src.splitlines()
        state.imports_source = "\n".join(lines[start - 1:end])

    # Walk top-level ClassDef nodes.
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if _is_lowerstrenum_subclass(node):
            state.enums.append(_parse_enum(node))
            continue
        if _is_case_model(node, organ_key):
            state.class_prefix = node.name.removesuffix("CancerCase")
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                state.case_model_docstring = node.body[0].value.value
            _parse_case_model_body(node, state, src)
            continue
        # Anything else (hand-authored inline BaseModel, etc.) is preserved
        # verbatim and surfaces as read-only.
        segment = ast.get_source_segment(src, node)
        if segment is not None:
            state.passthrough_blocks.append(segment)
        state.layer1_read_only_reason = (
            f"organ has a hand-authored top-level class ({node.name}); "
            "Layer 1 editing is disabled to avoid corrupting it. Edit "
            "the Pydantic file by hand to change shape, then reload."
        )

    if not state.class_prefix:
        # Couldn't find a case-model — guess from filename.
        state.class_prefix = organ_key.replace("_", " ").title().replace(" ", "")
        state.layer1_read_only_reason = (
            "no @assemble_case_model() class found; Layer 1 editing disabled"
        )
    return state


def _is_lowerstrenum_subclass(node: ast.ClassDef) -> bool:
    return any(
        isinstance(b, ast.Name) and b.id == "LowerStrEnum"
        for b in node.bases
    )


def _is_case_model(node: ast.ClassDef, organ_key: str) -> bool:
    """True iff ``node`` is the ``<Organ>CancerCase`` class decorated by
    ``@assemble_case_model()``. We match by decorator name rather than by
    suffix so a hand-authored class with a similar name doesn't fool us.
    """
    for dec in node.decorator_list:
        # Match `@assemble_case_model(...)` and `@assemble_case_model`.
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name) and target.id == "assemble_case_model":
            return True
    return False


def _parse_enum(node: ast.ClassDef) -> EnumDecl:
    values: list[str] = []
    for stmt in node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if (isinstance(target, ast.Name)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id == "auto"):
                values.append(target.id)
    return EnumDecl(name=node.name, values=values)


def _parse_case_model_body(node: ast.ClassDef, state: OrganState, src: str) -> None:
    """Populate ``state.fields`` and ``state.staging`` from the class body."""
    for stmt in node.body:
        if isinstance(stmt, ast.Expr):
            continue  # already captured as case_model_docstring
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            type_expr = _source_or_unparse(src, stmt.annotation)
            value = stmt.value

            # _STAGING ClassVar.
            if name == "_STAGING":
                if value is not None and isinstance(value, ast.Call):
                    state.staging = _parse_staging_spec(value, src)
                continue

            if value is None:
                # Annotation-only — uncommon; treat as bare with no default.
                state.fields.append(FieldDecl(
                    name=name, type_expr=type_expr, kind="bare",
                    default_expr="",
                ))
                continue

            kind, spec_options, default_expr = _classify_field_value(value, src)
            state.fields.append(FieldDecl(
                name=name,
                type_expr=type_expr,
                kind=kind,
                default_expr=default_expr,
                spec_options=spec_options,
            ))
        else:
            # Unknown statement in the body — preserve readability but mark
            # the file as advanced so save doesn't drop it.
            state.layer1_read_only_reason = state.layer1_read_only_reason or (
                "case-model body contains a non-AnnAssign statement; "
                "Layer 1 editing disabled"
            )


def _classify_field_value(value: ast.expr, src: str) -> tuple[str, SpecOptions | None, str]:
    """Return ``(kind, spec_options, default_expr)`` for one AnnAssign value.

    Recognises ``MarginSpec(...)``, ``LNSpec(...)``, ``BiomarkerSpec(...)``,
    and bare ``None``. Anything else (e.g. ``Field(default=...)``,
    ``[]``) round-trips as ``passthrough`` with the captured source.
    """
    if isinstance(value, ast.Constant) and value.value is None:
        return "bare", None, "None"
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        func_name = value.func.id
        if func_name in _SPEC_MARKER_NAMES:
            kwargs: dict[str, str] = {}
            for kw in value.keywords:
                if kw.arg is None:
                    continue
                kwargs[kw.arg] = _source_or_unparse(src, kw.value)
            return _SPEC_KIND_BY_NAME[func_name], SpecOptions(kwargs=kwargs), ""
    return "passthrough", None, _source_or_unparse(src, value)


def _parse_staging_spec(call: ast.Call, src: str) -> StagingDecl:
    """Parse a ``StagingSpec(pt=..., pn=..., stage_groups={...}, ...)`` call."""
    kw = {k.arg: k.value for k in call.keywords if k.arg is not None}

    def name_of(node: ast.expr | None) -> str | None:
        return _source_or_unparse(src, node) if node is not None else None

    pt = name_of(kw.get("pt")) or ""
    pn = name_of(kw.get("pn")) or ""
    pm = name_of(kw.get("pm"))
    tnm = name_of(kw.get("tnm_descriptor"))

    stage_groups: dict[str, str] = {}
    sg_node = kw.get("stage_groups")
    if isinstance(sg_node, ast.Dict):
        for key_node, val_node in zip(sg_node.keys, sg_node.values, strict=True):
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                stage_groups[key_node.value] = _source_or_unparse(src, val_node)

    include_ajcc_version = _const_bool(kw.get("include_ajcc_version"), True)
    include_tnm_descriptor = _const_bool(kw.get("include_tnm_descriptor"), True)
    lean = _const_bool(kw.get("lean"), True)
    group = None
    g_node = kw.get("group")
    if isinstance(g_node, ast.Constant) and isinstance(g_node.value, str):
        group = g_node.value

    return StagingDecl(
        pt=pt, pn=pn, pm=pm, tnm_descriptor=tnm,
        stage_groups=stage_groups,
        include_ajcc_version=include_ajcc_version,
        include_tnm_descriptor=include_tnm_descriptor,
        lean=lean,
        group=group,
    )


def _const_bool(node: ast.expr | None, default: bool) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return default


def _source_or_unparse(src: str, node: ast.AST) -> str:
    """Return the exact source text for ``node`` if locatable, else unparse.

    ``ast.get_source_segment`` works only for nodes Python emits with line
    info. The fallback to :func:`ast.unparse` covers synthesised nodes
    (we don't expect any in this codebase but keep the safety net).
    """
    if hasattr(node, "lineno"):
        seg = ast.get_source_segment(src, node)
        if seg is not None:
            return seg
    return ast.unparse(node)


# ---------------------------------------------------------------------------
# Layer 2
# ---------------------------------------------------------------------------


def _load_layer2(state: OrganState, path: Path) -> None:
    """Import the extraction module and read its ``GROUP_INSTRUCTIONS`` /
    ``FIELD_META`` dicts. Records mtime for race-condition detection.
    """
    state.layer2_mtime_at_load = path.stat().st_mtime

    mod = importlib.import_module(
        f"digital_registrar.schemas.extraction.{state.organ_key}"
    )
    # The author's source may use insertion-order; Python dicts preserve it.
    suffix = " " + DEFAULT_GROUP_INSTRUCTION
    for group_name, instruction in mod.GROUP_INSTRUCTIONS.items():
        has_default_suffix = instruction.endswith(suffix)
        body = (instruction.removesuffix(suffix) if has_default_suffix
                else instruction)
        state.group_instructions.append(GroupInstruction(
            group_name=group_name,
            instruction=body,
            has_default_suffix=has_default_suffix,
        ))

    for path_key, entry in mod.FIELD_META.items():
        state.field_meta.append(FieldMetaRow(
            path=path_key,
            desc=entry.get("desc", ""),
            group=entry.get("group"),
        ))


# ---------------------------------------------------------------------------
# Layer 3
# ---------------------------------------------------------------------------


def _load_layer3(state: OrganState, path: Path) -> None:
    state.has_layer3_file = path.is_file()
    if not state.has_layer3_file:
        state.layer3_mtime_at_load = 0.0
        return
    state.layer3_mtime_at_load = path.stat().st_mtime
    with path.open("rb") as f:
        data = tomllib.load(f)
    for field_path, entries in data.items():
        if not isinstance(entries, dict):
            continue
        state.aliases.append(AliasGroup(
            field_path=field_path,
            entries={
                str(k): [str(s) for s in v]
                for k, v in entries.items()
                if isinstance(v, list)
            },
        ))


__all__ = ["load_organ"]
