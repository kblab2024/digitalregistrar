"""In-memory state dataclasses for one organ across all three schema layers.

Loaded by :mod:`.loaders` from disk; mutated by the Streamlit views in
:mod:`.views`; rendered back to disk by :mod:`.writers`. Round-trip
fidelity (identity load → save) is a hard invariant on at least the
ten existing organs.

Design notes
------------
- ``EnumDecl.values`` are uppercase Python identifiers (the LHS of
  ``NAME = auto()``). The lowercase wire-form is derived at runtime by
  ``LowerStrEnum._generate_next_value_``; we don't track it separately.
- ``FieldDecl.kind`` discriminates spec-marker fields from bare ones.
  The string-typed ``type_expr`` captures the annotation as authored
  (e.g. ``"LungProcedure | None"``) — we don't try to re-parse types,
  just round-trip the source text.
- ``passthrough_blocks`` holds verbatim source for things the GUI
  doesn't understand structurally (e.g. an inline ``BaseModel`` like
  ``LungHistologicalPattern``, or a ``make_ln_type(...)`` direct call).
  The writer emits these unchanged between the enum block and the
  case-model class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FieldKind = Literal[
    "bare",          # `name: T | None = None`
    "margin_spec",   # `name: T | None = MarginSpec(...)`
    "ln_spec",       # `name: T | None = LNSpec(...)`
    "biomarker_spec",# `name: list = BiomarkerSpec(...)` (annotation is `list`)
    "passthrough",   # anything else — preserved verbatim
]


@dataclass
class EnumDecl:
    """One ``class XEnum(LowerStrEnum): ...`` block.

    ``values`` are uppercase identifiers — the LHS of ``NAME = auto()``.
    Drop a value to remove it; reorder the list to reorder the emitted
    file (which affects the iteration order of the StrEnum members).
    """

    name: str
    values: list[str] = field(default_factory=list)


@dataclass
class SpecOptions:
    """Keyword args carried on a spec marker (``MarginSpec(...)`` etc.).

    Stored as a dict of ``{kwarg_name: source_text}`` so the writer can
    emit them back literally. ``side=LymphNodeSideRL`` round-trips as
    ``{"side": "LymphNodeSideRL"}``; ``expression_required=True`` as
    ``{"expression_required": "True"}``.
    """

    kwargs: dict[str, str] = field(default_factory=dict)


@dataclass
class FieldDecl:
    """One declaration on the decorated case-model class body."""

    name: str
    type_expr: str          # the annotation as authored, e.g. "LungProcedure | None"
    kind: FieldKind
    default_expr: str = "None"          # for `kind="bare"` — usually "None"
    spec_options: SpecOptions | None = None  # set for spec_marker kinds


@dataclass
class StagingDecl:
    """The ``_STAGING: ClassVar = StagingSpec(...)`` block."""

    pt: str                                 # enum class name, e.g. "LungPT"
    pn: str
    stage_groups: dict[str, str] = field(default_factory=dict)  # field_name -> enum class name
    pm: str | None = None                   # default: omitted from emit
    tnm_descriptor: str | None = None
    group: str | None = None                # "staging" — omit if default
    include_ajcc_version: bool = True
    include_tnm_descriptor: bool = True
    lean: bool = True                       # always True post-v2


@dataclass
class GroupInstruction:
    """One row of the ``GROUP_INSTRUCTIONS`` dict."""

    group_name: str
    instruction: str
    has_default_suffix: bool = True   # whether `+ DEFAULT_GROUP_INSTRUCTION` was appended


@dataclass
class FieldMetaRow:
    """One row of the ``FIELD_META`` dict.

    ``path`` is the dotted key as it appears in the source (e.g.
    ``"biomarkers.percentage"``). Top-level entries (no dot) carry a
    ``group``; sub-field entries (dotted) have ``group=None``.
    """

    path: str
    desc: str
    group: str | None = None


@dataclass
class AliasGroup:
    """One TOML table — ``[field.path]`` → ``{enum_value: [surface_forms]}``."""

    field_path: str
    entries: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class OrganState:
    """Complete editable state for one organ.

    ``organ_key`` is the registry key (e.g. ``"lung"``); ``class_prefix``
    is the PascalCase prefix used by the case-model class and per-organ
    enums (e.g. ``"Lung"`` for ``LungCancerCase`` / ``LungProcedure``).
    Both are immutable in the GUI — renaming an organ requires sweeping
    every consumer and is out of scope.
    """

    organ_key: str
    class_prefix: str

    # --- Layer 1 ---
    module_docstring: str = ""
    imports_source: str = ""               # verbatim import block (top of file)
    enums: list[EnumDecl] = field(default_factory=list)
    passthrough_blocks: list[str] = field(default_factory=list)  # verbatim source between enums and case-model
    fields: list[FieldDecl] = field(default_factory=list)
    staging: StagingDecl | None = None
    case_model_docstring: str = ""

    # --- Layer 2 ---
    group_instructions: list[GroupInstruction] = field(default_factory=list)
    field_meta: list[FieldMetaRow] = field(default_factory=list)

    # --- Layer 3 ---
    aliases: list[AliasGroup] = field(default_factory=list)

    # Bookkeeping for race-condition detection
    layer1_mtime_at_load: float = 0.0
    layer2_mtime_at_load: float = 0.0
    layer3_mtime_at_load: float = 0.0
    has_layer3_file: bool = True           # False if the TOML didn't exist at load time

    # Read-only banner for organs the GUI can't fully round-trip
    layer1_read_only_reason: str = ""


__all__ = [
    "EnumDecl",
    "FieldDecl",
    "FieldKind",
    "FieldMetaRow",
    "GroupInstruction",
    "AliasGroup",
    "OrganState",
    "SpecOptions",
    "StagingDecl",
]
