"""Lean-schema builder: spec markers + ``@assemble_case_model`` decorator.

Per-organ canonical schemas declare only what differs (the lymph-node /
margin / TNM / stage-group vocabularies, plus per-organ overrides) and
import everything they need from this module. The decorator walks the
declared class, expands every spec marker into its verbose Pydantic
form, and returns the fully-formed ``BaseModel`` whose
``model_json_schema()`` is consumed by the JSON-schema CLI and the
signature factory.

The clean schemas declared by every per-organ module carry no
LM-facing description text — that lives in the sibling
``schemas/extraction/<organ>.py`` metadata module. Spec markers are
shape-only.
"""
from __future__ import annotations

import typing as _t
from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from ._common_factories import make_ln_type, make_margin_type

__all__ = [
    "MarginSpec",
    "LNSpec",
    "BiomarkerSpec",
    "StagingSpec",
    "assemble_case_model",
]


# --- Spec markers (FieldInfo subclasses) ----------------------------------
# Pydantic accepts FieldInfo subclasses as field defaults. The decorator
# detects these subclasses on the lean class and rebuilds the field with
# the synthesised nested type.


class _SpecMarker(FieldInfo):
    """Common base for field-level spec markers."""

    __slots__ = ("_marker_kind", "_marker_data")


class MarginSpec(_SpecMarker):
    """Marks a margin-list field for expansion to ``list[<Organ>Margin]``.

    The annotation on the field must be ``StrEnum | None`` (or
    ``Literal[<sites...>] | None``) — the decorator strips the
    optionality, synthesises the nested model via
    :func:`make_margin_type`, and rewrites the annotation to
    ``list[<Organ>Margin] | None``.

    No descriptions are accepted here; they live in the sibling
    ``schemas/extraction/<organ>.py`` metadata file under the dotted
    keys ``"<field>"``, ``"<field>.margin_category"``,
    ``"<field>.distance"``.
    """

    def __init__(
        self,
        *,
        group: str = "margins",
        type_name: str | None = None,
    ):
        super().__init__(default=None)
        self._marker_kind = "margin"
        self._marker_data = {
            "group": group,
            "type_name": type_name,
        }


class LNSpec(_SpecMarker):
    """Marks a lymph-node-list field for expansion to ``list[<Organ>LN]``.

    For most organs the field annotation is ``StrEnum | None``. For
    organs that have no per-station category (only ``LiverLN`` today)
    pass ``category=None`` and leave the annotation as ``None``.
    """

    def __init__(
        self,
        *,
        group: str = "lymph_nodes",
        side: Any | None = None,
        category: Any | None = ...,
        type_name: str | None = None,
    ):
        super().__init__(default=None)
        self._marker_kind = "ln"
        self._marker_data = {
            "group": group,
            "side": side,
            # ``category=...`` (Ellipsis) means "use the field's annotation";
            # ``category=None`` means "no category field on the nested type".
            "category": category,
            "type_name": type_name,
        }


class BiomarkerSpec(_SpecMarker):
    """Marks a biomarker-list field for expansion to ``list[<Organ>Biomarker>]``.

    The synthesised ``<Organ>Biomarker`` carries: ``biomarker_category``
    (StrEnum / Literal of the supplied categories), optional
    ``expression`` (or required, for lung), optional ``percentage``,
    optional ``score`` (omitted entirely when ``has_score=False``), and
    optional ``biomarker_name``. ``percentage_null_word`` is preserved
    as a hint for the metadata author and ignored at shape-emission time.
    """

    def __init__(
        self,
        *,
        categories: Any,
        group: str = "biomarkers",
        expression_required: bool = False,
        has_score: bool = True,
        percentage_null_word: str = "null",
        type_name: str | None = None,
    ):
        super().__init__(default=None)
        self._marker_kind = "biomarker"
        self._marker_data = {
            "group": group,
            "categories": categories,
            "expression_required": expression_required,
            "has_score": has_score,
            "percentage_null_word": percentage_null_word,
            "type_name": type_name,
        }


# --- Staging spec ----------------------------------------------------------
# Carried as a ClassVar on the lean schema; the decorator reads it and
# generates 6+ fields (tnm_descriptor, pt, pn, pm, stage groups, ajcc).


@dataclass(frozen=True)
class StagingSpec:
    """Declarative spec for the ``staging`` group's six (or seven) fields.

    Most organs need only ``pt``, ``pn``, ``stage_groups``. ``pm`` and
    ``tnm_descriptor`` default to the values used by 9/10 organs.

    ``stage_groups`` keys are field names (``"stage_group"``,
    ``"pathologic_stage_group"``, ``"anatomic_stage_group"``,
    ``"overall_stage"``). The value is the StrEnum / Literal of permitted
    stage codes for that field. Descriptions live in the sibling
    extraction-metadata module keyed by these field names.
    """

    pt: Any
    pn: Any
    stage_groups: dict[str, Any]
    pm: Any = Literal["mx", "m0", "m1"]
    tnm_descriptor: Any = Literal["y", "r", "m"]
    group: str = "staging"
    include_ajcc_version: bool = True
    include_tnm_descriptor: bool = True
    # ``lean`` is a no-op flag retained for source-level compatibility
    # with Phase-1/2 schema files; every staging block is now lean.
    lean: bool = True

    def iter_field_names(self) -> list[str]:
        """Return the staging field names in emit order.

        Useful when authoring an extraction-metadata file: pass these as
        the keys of ``FIELD_META`` so the metadata file doesn't drift
        from the schema's actual field names. Order matches
        :meth:`emit_fields`.
        """
        names: list[str] = []
        if self.include_tnm_descriptor:
            names.append("tnm_descriptor")
        names.extend(["pt_category", "pn_category", "pm_category"])
        names.extend(self.stage_groups.keys())
        if self.include_ajcc_version:
            names.append("ajcc_version")
        return names

    def emit_fields(self) -> list[tuple[str, Any, FieldInfo]]:
        """Return the staging fields in declaration order.

        Order matches the legacy hand-authored layout: tnm_descriptor →
        pt → pn → pm → <stage groups, in dict-insertion order> → ajcc_version.
        """
        out: list[tuple[str, Any, FieldInfo]] = []
        f = lambda: Field(default=None)  # noqa: E731

        if self.include_tnm_descriptor:
            out.append(("tnm_descriptor", Optional[self.tnm_descriptor], f()))

        out.append(("pt_category", Optional[self.pt], f()))
        out.append(("pn_category", Optional[self.pn], f()))
        out.append(("pm_category", Optional[self.pm], f()))

        for fname, spec in self.stage_groups.items():
            # Spec is the type itself (StrEnum / Literal); legacy tuple
            # form (type, custom_desc) is no longer accepted — descriptions
            # belong in the metadata module.
            out.append((fname, Optional[spec], f()))

        if self.include_ajcc_version:
            out.append(("ajcc_version", Optional[int], f()))

        return out


# --- Biomarker factory -----------------------------------------------------


def make_biomarker_type(
    name: str,
    *,
    categories: Any,
    expression_required: bool = False,
    has_score: bool = True,
) -> type[BaseModel]:
    """Build a clean ``<Organ>Biomarker`` ``BaseModel`` (no descriptions)."""
    fields: dict[str, Any] = {
        "biomarker_category": (Optional[categories], Field(None)),
    }
    fields["expression"] = (
        (bool, ...) if expression_required else (Optional[bool], Field(None))
    )
    fields["percentage"] = (Optional[int], Field(None))
    if has_score:
        fields["score"] = (Optional[Literal[0, 1, 2, 3]], Field(None))
    fields["biomarker_name"] = (Optional[str], Field(None))
    return create_model(name, __base__=BaseModel, **fields)


# --- Type-prefix overrides -------------------------------------------------
# A handful of organs have a registry key that differs from the legacy
# nested-type name (colorectal -> Colon*) or organ-text (cervix -> cervical).

_TYPE_PREFIX_OVERRIDES: dict[str, tuple[str, str]] = {
    "ColorectalCancerCase": ("colon", "Colon"),
    "CervixCancerCase": ("cervical", "Cervix"),
    "EsophagusCancerCase": ("esophageal", "Esophagus"),
}


def _organ_and_prefix(case_cls_name: str) -> tuple[str, str]:
    """``"BreastCancerCase"`` → ``("breast", "Breast")`` (with overrides)."""
    if case_cls_name in _TYPE_PREFIX_OVERRIDES:
        return _TYPE_PREFIX_OVERRIDES[case_cls_name]
    base = case_cls_name.removesuffix("CancerCase").removesuffix("Case")
    return base.lower(), base


# --- Helpers ---------------------------------------------------------------


def _strip_optional(ann: Any) -> Any:
    """Return ``X`` from ``X | None`` / ``Optional[X]``; no-op otherwise."""
    import types as _pytypes

    origin = _t.get_origin(ann)
    args = _t.get_args(ann)
    if origin in (_t.Union, _pytypes.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return ann


# --- The decorator ---------------------------------------------------------


def assemble_case_model(*, organ: str | None = None, type_prefix: str | None = None):
    """Class decorator: expand spec markers + ``_STAGING`` into a Pydantic model.

    Walks the decorated case-model's ``model_fields``, expands every
    :class:`MarginSpec` / :class:`LNSpec` / :class:`BiomarkerSpec` into
    its corresponding ``list[<NestedType>] | None`` shape, and inserts
    the staging fields produced by ``cls._STAGING.emit_fields()`` if a
    ``StagingSpec`` ClassVar is present.

    ``organ`` and ``type_prefix`` override the auto-derivation from the
    case-model class name.
    """

    def _decorate(cls: type) -> type:
        _, derived_prefix = _organ_and_prefix(cls.__name__)
        prefix = type_prefix or derived_prefix

        new_specs: dict[str, tuple[Any, FieldInfo]] = {}

        for fname, fi in cls.model_fields.items():
            ann = fi.annotation
            if isinstance(fi, MarginSpec):
                cat_type = _strip_optional(ann)
                md = fi._marker_data
                type_name = md["type_name"] or f"{prefix}Margin"
                margin_type = make_margin_type(type_name, category_type=cat_type)
                new_specs[fname] = (list[margin_type] | None, Field(default=None))
            elif isinstance(fi, LNSpec):
                md = fi._marker_data
                # category=... (Ellipsis) means "read from annotation"
                if md["category"] is ...:
                    cat_type = _strip_optional(ann)
                else:
                    cat_type = md["category"]
                type_name = md["type_name"] or f"{prefix}LN"
                ln_type = make_ln_type(
                    type_name,
                    category_type=cat_type,
                    side_type=md["side"],
                )
                new_specs[fname] = (list[ln_type] | None, Field(default=None))
            elif isinstance(fi, BiomarkerSpec):
                md = fi._marker_data
                type_name = md["type_name"] or f"{prefix}Biomarker"
                bio_type = make_biomarker_type(
                    type_name,
                    categories=md["categories"],
                    expression_required=md["expression_required"],
                    has_score=md["has_score"],
                )
                new_specs[fname] = (list[bio_type] | None, Field(default=None))
            else:
                new_specs[fname] = (ann, fi)

        # Inject staging fields if the class declares a _STAGING spec.
        staging = getattr(cls, "_STAGING", None)
        if isinstance(staging, StagingSpec):
            for fname, ann, fi in staging.emit_fields():
                new_specs[fname] = (ann, fi)

        new_cls = create_model(
            cls.__name__,
            __base__=BaseModel,
            __doc__=cls.__doc__,
            **new_specs,
        )
        new_cls.__module__ = cls.__module__
        new_cls.__qualname__ = cls.__qualname__

        # Preserve any other class vars the schema author defined.
        for key, value in cls.__dict__.items():
            if key.startswith("__") and key.endswith("__"):
                continue
            if key in new_specs:
                continue
            try:
                setattr(new_cls, key, value)
            except (AttributeError, TypeError):
                pass

        return new_cls

    return _decorate
