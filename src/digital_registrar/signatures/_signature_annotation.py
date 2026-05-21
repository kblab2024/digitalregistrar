"""Type-shape converter: clean Pydantic annotation -> DSPy-friendly annotation.

DSPy receives ``Literal[...]`` for closed vocabularies because that's what
its pydantic-driven JSON-schema generator emits to the LM. Our clean
Pydantic layer uses ``StrEnum`` subclasses (one per closed vocabulary,
each with an explicit ``NOT_STATED`` member). This module bridges the two
at signature-build time:

- ``enum_to_literal`` turns a ``StrEnum`` into ``Literal[<member values>]``
- ``signature_annotation`` walks an annotation and rewrites every nested
  StrEnum / nested-BaseModel-with-enums into the DSPy form
- ``rebuild_basemodel_for_dspy`` produces a parallel Pydantic class for a
  nested type, with both its annotations and field descriptions baked in
  from the per-organ ``FIELD_META`` (using dotted keys for sub-fields)

Round-trip guarantee: ``StrEnum.value`` -> Literal string emitted by the LM
-> ``MyCase.model_validate({...})`` re-coerces the bare string back into
the enum member. Pydantic v2 handles this natively.
"""
from __future__ import annotations

import types as _pytypes
import typing as _t
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from ..schemas.extraction import FieldMeta


def enum_to_literal(enum_cls: type[Enum]) -> Any:
    """Build ``Literal[<value1>, <value2>, ...]`` from an Enum subclass.

    All members are included — ``NOT_STATED`` is exposed to the LM as a
    valid output rather than filtered. This is a strict superset of the
    legacy ``Literal[...] | None`` expressivity and gives small models a
    cleaner signal than emitting null.
    """
    values = tuple(member.value for member in enum_cls)
    return Literal[values]  # type: ignore[valid-type]


def _is_strenum(tp: Any) -> bool:
    """A StrEnum subclass (or any Enum, in case schemas use IntEnum later)."""
    return isinstance(tp, type) and issubclass(tp, Enum)


def signature_annotation(
    annotation: Any,
    *,
    field_meta: dict[str, FieldMeta] | None = None,
    prefix: str | None = None,
) -> Any:
    """Rewrite ``annotation`` for DSPy emission.

    Handles, recursively:
      - ``StrEnum`` subclass            -> ``Literal[<values>]``
      - ``list[T]`` / ``List[T]``       -> ``list[signature_annotation(T)]``
      - ``X | None`` / ``Optional[X]``  -> ``signature_annotation(X) | None``
      - ``Union[A, B, ...]``            -> ``Union[signature_annotation(A), ...]``
      - ``BaseModel`` subclass          -> rebuilt via :func:`rebuild_basemodel_for_dspy`
                                           when ``field_meta`` and ``prefix`` are
                                           provided
      - everything else                 -> identity

    ``prefix`` carries the parent field name through generic containers so
    a nested-type encountered inside ``list[X]`` / ``X | None`` is
    rebuilt under the right dotted-key namespace.
    """
    if _is_strenum(annotation):
        return enum_to_literal(annotation)

    origin = _t.get_origin(annotation)
    args = _t.get_args(annotation)

    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if field_meta is not None and prefix is not None:
                return rebuild_basemodel_for_dspy(annotation, field_meta, prefix)
            return annotation
        return annotation

    if origin is list:
        if args:
            return list[
                signature_annotation(args[0], field_meta=field_meta, prefix=prefix)
            ]
        return annotation

    if origin in (_t.Union, _pytypes.UnionType):
        new_args = tuple(
            signature_annotation(a, field_meta=field_meta, prefix=prefix)
            for a in args
        )
        # Reconstruct the union; ``X | None`` pattern preserved naturally.
        result = new_args[0]
        for arg in new_args[1:]:
            result = result | arg
        return result

    # Other generic origins (tuple, dict, ...): walk args defensively.
    new_args = tuple(
        signature_annotation(a, field_meta=field_meta, prefix=prefix) for a in args
    )
    try:
        return origin[new_args] if len(new_args) > 1 else origin[new_args[0]]
    except TypeError:
        return annotation


def _model_has_enums(cls: type[BaseModel]) -> bool:
    """True if any field annotation contains an Enum subclass anywhere in its tree."""

    def visit(tp: Any) -> bool:
        if _is_strenum(tp):
            return True
        for arg in _t.get_args(tp):
            if visit(arg):
                return True
        return False

    return any(visit(fi.annotation) for fi in cls.model_fields.values())


_REBUILD_CACHE: dict[tuple[int, str], type[BaseModel]] = {}


def rebuild_basemodel_for_dspy(
    cls: type[BaseModel],
    field_meta: dict[str, FieldMeta],
    prefix: str,
) -> type[BaseModel]:
    """Produce a DSPy-flavored copy of a nested ``BaseModel``.

    For each sub-field, the rebuilt class carries:
      - the annotation passed through :func:`signature_annotation`
        (StrEnum members -> Literal values)
      - a ``Field(description=...)`` injected from
        ``field_meta[f"{prefix}.{sub_field_name}"]["desc"]``

    When the source class has no Enum members anywhere AND no description
    overrides apply, the original class is returned unchanged (a no-op
    fast path that keeps DSPy's ``custom_types`` cache stable).

    Memoized by ``(id(cls), prefix)``. ``field_meta`` is treated as a
    per-organ constant — caller is responsible for not feeding the cache
    inconsistent dicts under the same prefix.
    """
    cache_key = (id(cls), prefix)
    cached = _REBUILD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    has_enums = _model_has_enums(cls)
    has_desc_overrides = any(
        key.startswith(f"{prefix}.") for key in field_meta
    )
    if not has_enums and not has_desc_overrides:
        _REBUILD_CACHE[cache_key] = cls
        return cls

    new_fields: dict[str, tuple[Any, FieldInfo]] = {}
    for sub_name, sub_info in cls.model_fields.items():
        new_ann = signature_annotation(sub_info.annotation, field_meta=field_meta)
        meta_entry = field_meta.get(f"{prefix}.{sub_name}")
        if meta_entry is not None:
            desc = meta_entry.get("desc", "")
            default = sub_info.default if sub_info.default is not None else None
            new_fi = Field(default=default, description=desc)
        else:
            # Preserve the original field's defaults/required-ness.
            new_fi = sub_info
        new_fields[sub_name] = (new_ann, new_fi)

    rebuilt = create_model(
        cls.__name__,
        __base__=BaseModel,
        **new_fields,  # type: ignore[arg-type]
    )
    rebuilt.__module__ = cls.__module__
    rebuilt.__doc__ = cls.__doc__

    _REBUILD_CACHE[cache_key] = rebuilt
    return rebuilt


__all__ = [
    "enum_to_literal",
    "signature_annotation",
    "rebuild_basemodel_for_dspy",
]
