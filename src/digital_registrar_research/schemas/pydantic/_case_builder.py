"""Lean-schema builder: spec markers + ``@assemble_case_model`` decorator.

Per-organ canonical schemas declare only what differs (the lymph-node /
margin / TNM / stage-group vocabularies, plus per-organ overrides) and
import everything they need from this module. The decorator walks the
declared class, expands every spec marker into its verbose Pydantic
form, and returns the fully-formed ``BaseModel`` whose ``model_json_schema()``
is consumed by the JSON-schema CLI and the signature factory.

The factory helpers in :mod:`._factory_helpers` (which the signature
factory imports) are not the concern of canonical-schema authors and
remain a separate module so that downstream consumers don't pull in this
builder when they only need ``GroupedField`` / ``iter_custom_types``.
"""
from __future__ import annotations

import typing as _t
from dataclasses import dataclass, field as _dc_field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from ._common_factories import (
    LymphNodeSide,
    LymphNodeSideRL,
    make_ln_type,
    make_margin_type,
)
from ._factory_helpers import DEFAULT_GROUP_INSTRUCTION, GroupedField

__all__ = [
    "DEFAULT_GROUP_INSTRUCTION",
    "GroupedField",
    "LymphNodeSide",
    "LymphNodeSideRL",
    "MarginSpec",
    "LNSpec",
    "BiomarkerSpec",
    "StagingSpec",
    "assemble_case_model",
]


# --- Description templates -------------------------------------------------
# Staging block — every organ today uses these exact strings, so templating
# is byte-safe. Per-field overrides exist on StagingSpec for the few outliers
# (colorectal's tnm_descriptor wording, breast's pathologic-stage-group
# caveat, cervix's "FIGO" wording).

_TNM_DESCRIPTOR_DESC = (
    'identify the tnm descriptor of the tumor. e.g., "y" (post-therapy), "r", etc.'
)
_PT_DESC = "identify the pt category of the tumor"
_PN_DESC = "identify the pn category of the tumor"
_PM_DESC = (
    "identify the pm category of the tumor. if you see cM0 or cM1, etc., "
    "code as mx, since pathological M category is not available"
)
_AJCC_VERSION_DESC = "identify the ajcc version of the pathological staging"


def _humanize_stage_group_field(name: str) -> str:
    """``"pathologic_stage_group"`` -> ``"pathologic stage group"`` etc."""
    return name.replace("_", " ")


def _default_stage_group_desc(field_name: str) -> str:
    return f"identify the {_humanize_stage_group_field(field_name)} of the tumor"


# Biomarker block — breast/colon use one set of wording, lung diverges in two
# spots ("null" -> "None" in the percentage description, expression required,
# no score field). All three are templatable from a small param set.

_BIOMARKER_CATEGORY_DESC_TPL = (
    "acceptable value for biomarker categories in {organ} cancer. "
    "If not included in these standard categories, should be classified as others."
)
_BIOMARKER_EXPRESSION_DESC = (
    "specify whether or not the biomarker is expressed here. "
    "For Her-2 please refer to the score field, and don't fill in this field."
)
_BIOMARKER_PERCENTAGE_DESC_TPL = (
    "the percentage of tumor cells showing positive expression of the "
    "biomarker, rounded to integer. if not specified, return {null_word}"
)
_BIOMARKER_SCORE_DESC = (
    "specify the Her-2 expression score, negative: score 0 or 1, "
    "equivocal: score 2, positive: score 3 of the biomarker here, if applicable."
)
_BIOMARKER_NAME_DESC = "specify the name of the biomarker here."


# --- Spec markers (FieldInfo subclasses) ----------------------------------
# Pydantic accepts FieldInfo subclasses as field defaults. The decorator
# detects these subclasses on the lean class and rebuilds the field with
# the synthesised nested type.


class _SpecMarker(FieldInfo):
    """Common base for field-level spec markers."""

    __slots__ = ("_marker_kind", "_marker_data")


class MarginSpec(_SpecMarker):
    """Marks a margin-list field for expansion to ``list[<Organ>Margin]``.

    The annotation on the field must be ``Literal[<sites...>] | None`` —
    the decorator strips the ``Literal``, synthesises the nested model
    via :func:`make_margin_type`, and rewrites the annotation.
    """

    def __init__(
        self,
        *,
        desc: str,
        group: str = "margins",
        category_desc: str | None = None,
        distance_desc: str | None = None,
        type_name: str | None = None,
    ):
        super().__init__(default=None, description=desc)
        self._marker_kind = "margin"
        self._marker_data = {
            "desc": desc,
            "group": group,
            "category_desc": category_desc,
            "distance_desc": distance_desc,
            "type_name": type_name,
        }


class LNSpec(_SpecMarker):
    """Marks a lymph-node-list field for expansion to ``list[<Organ>LN]``.

    For most organs the field annotation is ``Literal[<categories...>] | None``.
    For organs that have no per-station category (only ``LiverLN`` today)
    pass ``category=None`` and leave the annotation as ``None | list``.
    """

    def __init__(
        self,
        *,
        desc: str,
        group: str = "lymph_nodes",
        side: Any | None = None,
        category: Any | None = ...,
        category_desc: str | None = None,
        category_group_word: str = "station",
        side_desc: str | None = None,
        station_name_desc: str | None = None,
        type_name: str | None = None,
    ):
        super().__init__(default=None, description=desc)
        self._marker_kind = "ln"
        self._marker_data = {
            "desc": desc,
            "group": group,
            "side": side,
            # ``category=...`` (Ellipsis) means "use the field's Literal annotation";
            # ``category=None`` means "no category field on the nested type".
            "category": category,
            "category_desc": category_desc,
            "category_group_word": category_group_word,
            "side_desc": side_desc,
            "station_name_desc": station_name_desc,
            "type_name": type_name,
        }


class BiomarkerSpec(_SpecMarker):
    """Marks a biomarker-list field for expansion to ``list[<Organ>Biomarker>]``.

    The synthesised ``<Organ>Biomarker`` carries: ``biomarker_category``
    (Literal of the supplied categories), optional ``expression`` (or
    required, for lung), optional ``percentage``, optional ``score``
    (omitted entirely when ``has_score=False``), and optional
    ``biomarker_name``.
    """

    def __init__(
        self,
        *,
        desc: str,
        categories: Any,
        group: str = "biomarkers",
        expression_required: bool = False,
        has_score: bool = True,
        percentage_null_word: str = "null",
        type_name: str | None = None,
    ):
        super().__init__(default=None, description=desc)
        self._marker_kind = "biomarker"
        self._marker_data = {
            "desc": desc,
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
    ``"overall_stage"``). The value is either the ``Literal[...]`` of
    permitted stage codes, or a ``(Literal[...], custom_desc)`` tuple
    when the description deviates from the default
    ``"identify the <field name humanised> of the tumor"``.
    """

    pt: Any
    pn: Any
    stage_groups: dict[str, Any]
    pm: Any = Literal["mx", "m0", "m1"]
    tnm_descriptor: Any = Literal["y", "r", "m"]
    group: str = "staging"
    pt_desc: str = _PT_DESC
    pn_desc: str = _PN_DESC
    pm_desc: str = _PM_DESC
    tnm_descriptor_desc: str = _TNM_DESCRIPTOR_DESC
    ajcc_version_desc: str = _AJCC_VERSION_DESC
    include_ajcc_version: bool = True
    include_tnm_descriptor: bool = True

    def emit_fields(self) -> list[tuple[str, Any, FieldInfo]]:
        """Return the staging fields in declaration order.

        Order matches the legacy hand-authored layout: tnm_descriptor →
        pt → pn → pm → <stage groups, in dict-insertion order> → ajcc_version.
        ``iter_fields_by_group`` preserves this within the ``staging`` group.
        """
        out: list[tuple[str, Any, FieldInfo]] = []

        if self.include_tnm_descriptor:
            out.append((
                "tnm_descriptor",
                Optional[self.tnm_descriptor],
                GroupedField(group=self.group, desc=self.tnm_descriptor_desc),
            ))

        out.append((
            "pt_category",
            Optional[self.pt],
            GroupedField(group=self.group, desc=self.pt_desc),
        ))
        out.append((
            "pn_category",
            Optional[self.pn],
            GroupedField(group=self.group, desc=self.pn_desc),
        ))
        out.append((
            "pm_category",
            Optional[self.pm],
            GroupedField(group=self.group, desc=self.pm_desc),
        ))

        for fname, spec in self.stage_groups.items():
            if isinstance(spec, tuple):
                lit, desc = spec
            else:
                lit = spec
                desc = _default_stage_group_desc(fname)
            out.append((
                fname,
                Optional[lit],
                GroupedField(group=self.group, desc=desc),
            ))

        if self.include_ajcc_version:
            out.append((
                "ajcc_version",
                Optional[int],
                GroupedField(group=self.group, desc=self.ajcc_version_desc),
            ))

        return out


# --- Biomarker factory -----------------------------------------------------


def make_biomarker_type(
    name: str,
    *,
    organ: str,
    categories: Any,
    expression_required: bool = False,
    has_score: bool = True,
    percentage_null_word: str = "null",
) -> type[BaseModel]:
    """Build a ``<Organ>Biomarker`` ``BaseModel`` with templated descriptions."""
    cat_desc = _BIOMARKER_CATEGORY_DESC_TPL.format(organ=organ)
    pct_desc = _BIOMARKER_PERCENTAGE_DESC_TPL.format(null_word=percentage_null_word)

    fields: dict[str, Any] = {
        "biomarker_category": (
            Optional[categories],
            Field(None, description=cat_desc),
        ),
    }
    if expression_required:
        fields["expression"] = (bool, ...)
    else:
        fields["expression"] = (
            Optional[bool],
            Field(None, description=_BIOMARKER_EXPRESSION_DESC),
        )
    fields["percentage"] = (
        Optional[int],
        Field(None, description=pct_desc),
    )
    if has_score:
        fields["score"] = (
            Optional[Literal[0, 1, 2, 3]],
            Field(None, description=_BIOMARKER_SCORE_DESC),
        )
    fields["biomarker_name"] = (
        Optional[str],
        Field(None, description=_BIOMARKER_NAME_DESC),
    )
    return create_model(name, __base__=BaseModel, **fields)


# --- Type-prefix overrides -------------------------------------------------
# A handful of organs have a registry key that differs from the legacy
# nested-type name (colorectal -> Colon*) or organ-text (cervix -> cervical).
# Surfacing both axes here lets the canonical schemas stay declarative.

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


def _expanded_grouped_field(*, desc: str, group: str) -> FieldInfo:
    """Build the post-expansion ``Field`` for margin/LN/biomarker list fields.

    Mirrors what :func:`GroupedField` produces, but kept inline here so the
    decorator can construct it without importing the public helper twice.
    """
    return Field(
        default=None,
        description=desc,
        json_schema_extra={"group": group, "dspy_desc": desc},
    )


# --- The decorator ---------------------------------------------------------


def assemble_case_model(*, organ: str | None = None, type_prefix: str | None = None):
    """Class decorator: expand spec markers + ``_STAGING`` into a Pydantic model.

    Walks the decorated case-model's ``model_fields``, expands every
    :class:`MarginSpec` / :class:`LNSpec` / :class:`BiomarkerSpec` into
    its corresponding ``list[<NestedType>] | None`` shape, and inserts the
    six (or seven) staging fields produced by ``cls._STAGING.emit_fields()``
    if a ``StagingSpec`` ClassVar is present.

    ``organ`` and ``type_prefix`` override the auto-derivation from the
    case-model class name. Default behaviour: ``"BreastCancerCase"`` →
    ``("breast", "Breast")``; consult ``_TYPE_PREFIX_OVERRIDES`` for the
    handful of organs whose legacy type prefix differs from the registry key.
    """

    def _decorate(cls: type) -> type:
        derived_organ, derived_prefix = _organ_and_prefix(cls.__name__)
        organ_key = organ or derived_organ
        prefix = type_prefix or derived_prefix

        new_specs: dict[str, tuple[Any, FieldInfo]] = {}

        for fname, fi in cls.model_fields.items():
            ann = fi.annotation
            if isinstance(fi, MarginSpec):
                cat_type = _strip_optional(ann)
                md = fi._marker_data
                type_name = md["type_name"] or f"{prefix}Margin"
                margin_type = make_margin_type(
                    type_name,
                    organ=organ_key,
                    category_type=cat_type,
                    category_desc=md["category_desc"],
                    **(
                        {"distance_desc": md["distance_desc"]}
                        if md["distance_desc"]
                        else {}
                    ),
                )
                new_specs[fname] = (
                    list[margin_type] | None,
                    _expanded_grouped_field(desc=md["desc"], group=md["group"]),
                )
            elif isinstance(fi, LNSpec):
                md = fi._marker_data
                # category=... (Ellipsis sentinel) means "read from annotation"
                if md["category"] is ...:
                    cat_type = _strip_optional(ann)
                else:
                    cat_type = md["category"]
                type_name = md["type_name"] or f"{prefix}LN"
                ln_kwargs = {
                    "organ": organ_key,
                    "category_type": cat_type,
                    "category_desc": md["category_desc"],
                    "category_group_word": md["category_group_word"],
                    "side_type": md["side"],
                    "side_desc": md["side_desc"],
                }
                if md["station_name_desc"]:
                    ln_kwargs["station_name_desc"] = md["station_name_desc"]
                ln_type = make_ln_type(type_name, **ln_kwargs)
                new_specs[fname] = (
                    list[ln_type] | None,
                    _expanded_grouped_field(desc=md["desc"], group=md["group"]),
                )
            elif isinstance(fi, BiomarkerSpec):
                md = fi._marker_data
                type_name = md["type_name"] or f"{prefix}Biomarker"
                bio_type = make_biomarker_type(
                    type_name,
                    organ=organ_key,
                    categories=md["categories"],
                    expression_required=md["expression_required"],
                    has_score=md["has_score"],
                    percentage_null_word=md["percentage_null_word"],
                )
                new_specs[fname] = (
                    list[bio_type] | None,
                    _expanded_grouped_field(desc=md["desc"], group=md["group"]),
                )
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

        # Preserve class vars (most importantly _GROUP_INSTRUCTIONS).
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
