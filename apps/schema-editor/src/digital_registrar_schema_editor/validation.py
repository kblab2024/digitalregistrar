"""Pre-save invariants over :class:`OrganState`.

Mirrors the runtime invariants enforced by ``EXTRACTION_META`` and the
``test_schema_concordance`` suite, but runs entirely against in-memory
state so the GUI can show errors as the user types — long before save.

Categorisation: ``errors`` block the save; ``warnings`` don't. The
caller (the Streamlit save button) disables itself if any error is
present.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .state import OrganState

_PY_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENUM_VALUE_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.errors


def validate(state: OrganState) -> ValidationReport:
    rep = ValidationReport()
    _check_layer1(state, rep)
    _check_layer2(state, rep)
    _check_layer3(state, rep)
    return rep


# ---------------------------------------------------------------------------
# Layer 1
# ---------------------------------------------------------------------------


def _check_layer1(state: OrganState, rep: ValidationReport) -> None:
    if state.layer1_read_only_reason:
        # Read-only mode skips structural checks — we won't be writing.
        return

    seen_enum_names: set[str] = set()
    for enum in state.enums:
        if not _PY_IDENT_RE.match(enum.name):
            rep.errors.append(f"L1: enum class name {enum.name!r} is not a valid identifier")
        if enum.name in seen_enum_names:
            rep.errors.append(f"L1: duplicate enum class {enum.name!r}")
        seen_enum_names.add(enum.name)
        seen_values: set[str] = set()
        for v in enum.values:
            if not _ENUM_VALUE_RE.match(v):
                rep.errors.append(f"L1: enum value {enum.name}.{v} must be UPPER_SNAKE")
            if v in seen_values:
                rep.errors.append(f"L1: duplicate enum value {enum.name}.{v}")
            seen_values.add(v)
        if not enum.values:
            rep.warnings.append(f"L1: enum {enum.name} has no values")

    seen_field_names: set[str] = set()
    for fd in state.fields:
        if not _PY_IDENT_RE.match(fd.name):
            rep.errors.append(f"L1: field name {fd.name!r} is not a valid identifier")
        if fd.name in seen_field_names:
            rep.errors.append(f"L1: duplicate field {fd.name!r}")
        seen_field_names.add(fd.name)

    if state.staging is not None:
        s = state.staging
        if not s.pt:
            rep.errors.append("L1: staging requires pt= enum")
        if not s.pn:
            rep.errors.append("L1: staging requires pn= enum")
        if not s.stage_groups:
            rep.errors.append("L1: staging requires at least one stage_groups entry")
        for sg_name in s.stage_groups:
            if not _PY_IDENT_RE.match(sg_name):
                rep.errors.append(f"L1: stage_group field name {sg_name!r} invalid")


# ---------------------------------------------------------------------------
# Layer 2
# ---------------------------------------------------------------------------


def _check_layer2(state: OrganState, rep: ValidationReport) -> None:
    # Every group used by a field meta entry must exist in GROUP_INSTRUCTIONS.
    group_names = {gi.group_name for gi in state.group_instructions}
    seen_groups: set[str] = set()
    for gi in state.group_instructions:
        if not gi.group_name:
            rep.errors.append("L2: empty group name in GROUP_INSTRUCTIONS")
            continue
        if gi.group_name in seen_groups:
            rep.errors.append(f"L2: duplicate group {gi.group_name!r}")
        seen_groups.add(gi.group_name)
        if not gi.instruction.strip():
            rep.warnings.append(f"L2: group {gi.group_name!r} has empty instruction")

    seen_paths: set[str] = set()
    for row in state.field_meta:
        if not row.path:
            rep.errors.append("L2: FIELD_META has empty key")
            continue
        if row.path in seen_paths:
            rep.errors.append(f"L2: duplicate FIELD_META key {row.path!r}")
        seen_paths.add(row.path)
        if not row.desc.strip():
            rep.warnings.append(f"L2: empty desc for {row.path!r}")
        is_top_level = "." not in row.path
        if is_top_level:
            if not row.group:
                rep.errors.append(f"L2: top-level field {row.path!r} needs a group")
            elif row.group not in group_names:
                rep.errors.append(
                    f"L2: field {row.path!r} references unknown group {row.group!r}"
                )
        else:
            if row.group:
                rep.warnings.append(
                    f"L2: nested sub-field {row.path!r} should not carry a group"
                )

    # Orphan groups (defined but unused) -> warning only.
    used_groups = {r.group for r in state.field_meta if r.group}
    for g in group_names - used_groups:
        rep.warnings.append(f"L2: group {g!r} defined but unused")


# ---------------------------------------------------------------------------
# Layer 3
# ---------------------------------------------------------------------------


def _check_layer3(state: OrganState, rep: ValidationReport) -> None:
    field_meta_paths = {r.path for r in state.field_meta}
    for grp in state.aliases:
        if grp.field_path not in field_meta_paths:
            rep.errors.append(
                f"L3: alias table [{grp.field_path}] references unknown field path "
                f"(not in FIELD_META)"
            )
        if not grp.entries:
            rep.warnings.append(f"L3: alias table [{grp.field_path}] has no entries")
        for canonical, surface_forms in grp.entries.items():
            if not surface_forms:
                rep.errors.append(
                    f"L3: [{grp.field_path}].{canonical} has empty surface-form list"
                )


__all__ = ["validate", "ValidationReport"]
