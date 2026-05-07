"""DSPy signatures with ``str`` outputs — the B2 ablation.

The default monolithic signature uses ``Literal[...]`` enums and typed
numerics so DSPy's adapters can enforce schema conformance at generation
time. B2 strips every **scalar** output annotation down to ``str | None``
to measure how much accuracy is lost when type discipline is removed.

Crucially, list-of-Pydantic fields (e.g. ``list[BreastBiomarker] | None``
on breast, ``list[LungCancerOthernested] | None`` on lung) are left
untouched — coercing them to ``str | None`` would force the model to
serialise structured data into prose and discard the very organ-specific
shape we want to evaluate against. The ablation tests "what if scalar
enums become free-text strings?" — not "what if structured nested data
disappears?".
"""
from __future__ import annotations

from functools import cache
from typing import Union, get_args, get_origin

import dspy
from pydantic import BaseModel

from .monolithic import (
    INPUT_FIELD_NAMES,
    MONOLITHIC_DOCSTRING,
    get_monolithic_signature,
    list_monolithic_fields,
)

STR_DOCSTRING_SUFFIX = (
    "\n\nFor each scalar field below, return a free-text answer using "
    "wording from the report — do not pick from a fixed set; we will "
    "normalise your answers afterwards. For nested-list fields (those "
    "whose type is a list of objects), keep returning the structured "
    "list shape, not free text."
)


def _is_scalar_leaf(annotation: object) -> bool:
    """Return True for ``Literal[...]``, ``int``, ``float``, ``bool``,
    ``str``, or any ``X | None`` whose ``X`` is one of those leaves.

    Returns False for ``list[<PydanticModel>]``, ``list[<PydanticModel>] |
    None``, dict-typed fields, and any container holding a Pydantic
    BaseModel — those should NOT be coerced to string.
    """
    origin = get_origin(annotation)
    if origin is Union:
        # Strip the NoneType arm and recurse on what's left.
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) != 1:
            return False
        return _is_scalar_leaf(non_none[0])
    if origin is None:
        # Bare type. Scalars: int, float, bool, str.
        return annotation in (int, float, bool, str)
    # Container types — list, dict, etc.
    if origin in (list, tuple, set, dict):
        return False
    # Literal[...] has origin = typing.Literal.
    try:
        from typing import Literal
        if origin is Literal or repr(origin).endswith("Literal"):
            return True
    except Exception:
        pass
    # Defensive: anything we can't classify as scalar, leave alone.
    return False


def _contains_pydantic_model(annotation: object) -> bool:
    """Return True if ``annotation`` (or any of its type args) is a
    Pydantic BaseModel subclass."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    for arg in get_args(annotation):
        if _contains_pydantic_model(arg):
            return True
    return False


@cache
def get_str_signature(organ: str) -> type[dspy.Signature]:
    """Build a DSPy signature for ``organ`` with every **scalar** output
    typed as ``str | None``. Nested list-of-Pydantic fields are
    preserved.
    """
    base = get_monolithic_signature(organ)

    merged_annotations: dict[str, object] = {}
    merged_attrs: dict[str, object] = {}

    # Inputs first — keep their original typing and descriptor.
    for name, finfo in base.input_fields.items():
        merged_annotations[name] = finfo.annotation
        extras = dict(finfo.json_schema_extra or {})
        merged_attrs[name] = dspy.InputField(desc=extras.get("desc", ""))

    # Outputs — coerce only scalars.
    for name, finfo in base.output_fields.items():
        if name in INPUT_FIELD_NAMES:
            continue
        annotation = finfo.annotation
        extras = dict(finfo.json_schema_extra or {})
        if _contains_pydantic_model(annotation):
            # Preserve nested-list structure.
            merged_annotations[name] = annotation
        elif _is_scalar_leaf(annotation):
            merged_annotations[name] = str | None
        else:
            # Unknown / complex container without Pydantic models — be
            # conservative, leave alone rather than risk type erasure.
            merged_annotations[name] = annotation
        merged_attrs[name] = dspy.OutputField(desc=extras.get("desc", ""))

    merged_attrs["__annotations__"] = merged_annotations
    field_names = [n for n in merged_annotations if n not in INPUT_FIELD_NAMES]
    field_list = "\n".join(f"  - {n}" for n in field_names)
    merged_attrs["__doc__"] = (
        MONOLITHIC_DOCSTRING.format(organ=organ, field_list=field_list)
        + STR_DOCSTRING_SUFFIX
    )

    cls_name = f"{organ.title()}CancerStrOutputs"
    sig_cls = type(cls_name, (dspy.Signature,), merged_attrs)
    return sig_cls


def list_str_fields(organ: str) -> list[str]:
    """Field names for the str-outputs signature — same as the monolithic
    one because we don't add or drop fields, just retype scalars."""
    return list_monolithic_fields(organ)


__all__ = ["get_str_signature", "list_str_fields"]
