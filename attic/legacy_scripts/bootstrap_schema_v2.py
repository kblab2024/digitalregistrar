#!/usr/bin/env python3
"""One-shot generator: legacy DSPy signatures -> v2 flat Pydantic case-models.

For each organ in ``models.modellist.organmodels``, this walks the per-subsection
DSPy signature classes (``BreastCancerNonnested`` etc.), reads their
``OutputField`` declarations via Python AST (for source-level fidelity), and
writes a v2 case-model file at::

    src/digital_registrar/schemas/pydantic/<organ>.py.new

The ``.py.new`` suffix is intentional — the existing thin wrappers are NOT
overwritten. The user reviews each ``.py.new``, hand-edits group assignments
where the legacy class-name-suffix heuristic is wrong, then renames it.

After v2 ships, this script and ``schemas/pydantic/_builder.py`` should both
be deleted (mark them ``# DEPRECATED``).

DEPRECATED: remove after v2 ships.
"""
from __future__ import annotations

import argparse
import ast
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

# Imported only for the canonical organ -> [signature class names] mapping;
# we do not introspect dspy at all here (we read the AST instead).
from digital_registrar.models.modellist import organmodels  # noqa: E402

MODELS_DIR = SRC / "digital_registrar" / "models"
SCHEMAS_DIR = SRC / "digital_registrar" / "schemas" / "pydantic"

# Legacy class-name suffix -> v2 group tag.
SUFFIX_TO_GROUP: dict[str, str] = {
    "Nonnested": "nonnested",
    "Staging": "staging",
    "Margins": "margins",
    "LN": "lymph_nodes",
    "Biomarkers": "biomarkers",
    "Grading": "grading",
    "Extent": "extent",
    "VascularInvasion": "vascular_invasion",
    "Othernested": "othernested",
    # Special: not suffixed with "Cancer*"
    "DCIS": "dcis",
}

# Source-file basename per organ (registry-key drift: "colorectal" -> "colon.py").
ORGAN_TO_SOURCE: dict[str, str] = {
    "breast": "breast.py",
    "cervix": "cervix.py",
    "colorectal": "colon.py",
    "esophagus": "esophagus.py",
    "liver": "liver.py",
    "lung": "lung.py",
    "pancreas": "pancreas.py",
    "prostate": "prostate.py",
    "stomach": "stomach.py",
    "thyroid": "thyroid.py",
}

# Class-name-prefix on the case-model (often differs from organ key).
ORGAN_TO_PASCAL: dict[str, str] = {
    "breast": "Breast", "cervix": "Cervix", "colorectal": "Colorectal",
    "esophagus": "Esophagus", "liver": "Liver", "lung": "Lung",
    "pancreas": "Pancreas", "prostate": "Prostate", "stomach": "Stomach",
    "thyroid": "Thyroid",
}


def _group_for_class(cls_name: str) -> str:
    """Map a legacy DSPy signature class name to a v2 group tag."""
    if cls_name == "DCIS":
        return "dcis"
    # All others end with one of the known suffixes after a "Cancer" anchor.
    # Search longest suffix first (e.g. "VascularInvasion" before "Invasion").
    for suffix in sorted(SUFFIX_TO_GROUP, key=len, reverse=True):
        if cls_name.endswith(suffix):
            return SUFFIX_TO_GROUP[suffix]
    raise ValueError(
        f"Cannot derive group tag from {cls_name!r}. Add a suffix to "
        f"SUFFIX_TO_GROUP in scripts/bootstrap_schema_v2.py."
    )


def _is_output_field_call(node: ast.AST) -> bool:
    """Detect ``dspy.OutputField(...)`` (with or without the dspy prefix)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "OutputField":
        return True
    if isinstance(func, ast.Name) and func.id == "OutputField":
        return True
    return False


def _extract_desc(call: ast.Call) -> str:
    """Pull the ``desc=...`` keyword from an ``OutputField(...)`` call."""
    for kw in call.keywords:
        if kw.arg == "desc" and isinstance(kw.value, ast.Constant):
            v = kw.value.value
            if isinstance(v, str):
                return v
    return ""


def _annotation_source(node: ast.AST, source: str) -> str:
    """Render a type annotation back to source text using the original file.

    Uses ast.get_source_segment so triple-quoted strings, line breaks, and
    custom Literal members are preserved verbatim.
    """
    seg = ast.get_source_segment(source, node)
    return seg if seg is not None else ast.unparse(node)


def _custom_types_in(annotation_node: ast.AST, known: set[str]) -> set[str]:
    """Walk an annotation AST and return any ``Name`` ids that are in *known*.

    Catches names inside subscripts (``list[BreastMargin]``), unions
    (``BreastMargin | None``), and nested forms.
    """
    used: set[str] = set()
    for sub in ast.walk(annotation_node):
        if isinstance(sub, ast.Name) and sub.id in known:
            used.add(sub.id)
    return used


def _walk_signature(class_node: ast.ClassDef, source: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (group_instruction, [(field_name, annotation_src, desc), ...])."""
    docstring = ast.get_docstring(class_node) or ""
    instr = " ".join(docstring.split()).strip()  # collapse whitespace
    fields: list[tuple[str, str, str]] = []
    for stmt in class_node.body:
        if not isinstance(stmt, ast.AnnAssign):
            continue
        if not isinstance(stmt.target, ast.Name):
            continue
        name = stmt.target.id
        if name in {"report", "report_jsonized"}:
            continue
        if stmt.value is None or not _is_output_field_call(stmt.value):
            continue
        ann_src = _annotation_source(stmt.annotation, source)
        desc = _extract_desc(stmt.value)
        fields.append((name, ann_src, desc))
    return instr, fields


# Names from `_common_types` we may need to import.
COMMON_TYPE_NAMES = {
    "BreastMargin", "BreastLN", "BreastBiomarker",
    "CervixMargin", "CervixLN",
    "ColonMargin", "ColonLN", "ColonBiomarker",
    "EsophagusMargin", "EsophagusLN",
    "LiverMargin", "LiverLN",
    "LungMargin", "LungLN", "LungHistologicalPattern", "LungBiomarker",
    "PancreasMargin", "PancreasLN",
    "ProstateLN",
    "StomachMargin", "StomachLN",
    "ThyroidMargin", "ThyroidLN",
}


def _format_desc_literal(desc: str) -> str:
    """Render a description as a Python string literal.

    Multi-line descriptions become triple-quoted; short ones become single-quoted.
    """
    if "\n" in desc:
        body = desc.replace('"""', '\\"\\"\\"')
        return f'"""{body}"""'
    body = desc.replace('"', '\\"')
    return f'"{body}"'


def render_organ_schema(organ: str) -> str:
    """Build the v2 case-model source for a single organ."""
    src_path = MODELS_DIR / ORGAN_TO_SOURCE[organ]
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    legacy_classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }

    organ_class_names = list(organmodels[organ])
    pascal = ORGAN_TO_PASCAL[organ]

    instructions_by_group: dict[str, str] = {}  # ordered dict by insertion
    fields_lines: list[str] = []
    custom_types_used: set[str] = set()

    for cls_name in organ_class_names:
        node = legacy_classes.get(cls_name)
        if node is None:
            print(f"  WARN: {cls_name} not found in {src_path.name}", file=sys.stderr)
            continue
        group = _group_for_class(cls_name)
        instr, fields = _walk_signature(node, source)
        # First docstring per group wins (preserves first-seen ordering).
        instructions_by_group.setdefault(group, instr)
        fields_lines.append(f"\n    # --- {group} ({cls_name}) ---")
        # Walk the AST again to harvest custom types — keyed by field name.
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                custom_types_used |= _custom_types_in(stmt.annotation, COMMON_TYPE_NAMES)
        for name, ann_src, desc in fields:
            desc_lit = _format_desc_literal(desc)
            # Wrap long descriptions onto multiple lines for readability.
            if len(desc) <= 80:
                fields_lines.append(
                    f"    {name}: {ann_src} = GroupedField(group={group!r}, desc={desc_lit})"
                )
            else:
                fields_lines.append(
                    f"    {name}: {ann_src} = GroupedField(\n"
                    f"        group={group!r},\n"
                    f"        desc={desc_lit},\n"
                    f"    )"
                )

    # Render the instruction map in registration order.
    instr_lines = []
    for group, instr in instructions_by_group.items():
        instr_clean = instr or "Extract the listed items."
        # Strip trailing "DO NOT JUST RETURN NULL..." since the default already covers it.
        # Append the default boilerplate via concatenation to keep it editable.
        if "DO NOT JUST RETURN NULL" in instr_clean.upper():
            head, _, _ = instr_clean.partition(" DO NOT JUST RETURN NULL")
            head = head.rstrip(".")
            instr_lines.append(
                f"        {group!r}: {head!r} + ' ' + DEFAULT_GROUP_INSTRUCTION,"
            )
        else:
            instr_lines.append(f"        {group!r}: {instr_clean!r},")

    common_types_import = ""
    if custom_types_used:
        ordered = sorted(custom_types_used)
        common_types_import = (
            f"from ._common_types import {', '.join(ordered)}\n"
        )

    # NB: no `from __future__ import annotations` — keeping annotations
    # eagerly evaluated avoids needing model_rebuild() for nested-BaseModel
    # forward refs. Project floor is Python 3.10 so PEP-604 union syntax works
    # at runtime without the future import.
    header = textwrap.dedent(f'''\
        """Canonical case-model for {organ} cancer (v2, schema-driven).

        SINGLE SOURCE OF TRUTH. DSPy signatures are built dynamically from the
        fields below by ``digital_registrar.signatures.factory``;
        JSON Schema is auto-generated by the ``registrar-schemas`` CLI.

        Field-level group tags (set via :func:`GroupedField`) drive how the
        factory decomposes extraction into one or more LM calls. The
        insertion order of ``_GROUP_INSTRUCTIONS`` controls the extraction
        order AND first-wins semantics on duplicate field names.

        Authored mechanically by ``scripts/bootstrap_schema_v2.py`` from
        the legacy DSPy signatures; review and refine clinically before
        merging.
        """
        from typing import ClassVar, Literal

        from pydantic import BaseModel

        from ._factory_helpers import GroupedField, DEFAULT_GROUP_INSTRUCTION
        ''')

    body = textwrap.dedent(f'''\

        class {pascal}CancerCase(BaseModel):
            """Canonical extracted case record for {organ} cancer."""

            _GROUP_INSTRUCTIONS: ClassVar[dict[str, str]] = {{
        {{INSTRUCTIONS}}
            }}
        {{FIELDS}}
        ''')

    rendered = (
        header
        + common_types_import
        + body.replace("{INSTRUCTIONS}", "\n".join(instr_lines))
              .replace("{FIELDS}", "\n".join(fields_lines))
    )
    return rendered


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--organ",
        choices=sorted(organmodels.keys()),
        help="Limit generation to a single organ. Default: all 10.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=SCHEMAS_DIR,
        help="Where to write *.py.new files (default: schemas/pydantic/).",
    )
    args = ap.parse_args()

    targets = [args.organ] if args.organ else sorted(organmodels.keys())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for organ in targets:
        if organ not in ORGAN_TO_SOURCE:
            print(f"  SKIP: no source mapping for {organ!r}", file=sys.stderr)
            continue
        print(f"  bootstrapping {organ} ...")
        text = render_organ_schema(organ)
        out_path = args.out_dir / f"{organ}.py.new"
        out_path.write_text(text, encoding="utf-8")
        print(f"    wrote {out_path.relative_to(REPO_ROOT)} ({len(text)} bytes)")
    print("\nDone. Review *.py.new files; rename to *.py to activate.")


if __name__ == "__main__":
    main()
