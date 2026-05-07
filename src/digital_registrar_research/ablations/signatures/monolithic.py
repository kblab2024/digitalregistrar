"""
Monolithic DSPy signatures — one per organ, collapsed from the 5–7
per-subsection signatures that ship with the parent project.

Rather than hand-copy every field, we introspect the existing organ
signatures (BreastCancerNonnested, BreastCancerStaging, ...) and merge
their output fields into a single new `dspy.Signature` subclass per
organ. This keeps the monolithic baseline automatically in sync with
the modular baseline as the parent project evolves.

Usage:
    from signatures.monolithic import get_monolithic_signature
    sig = get_monolithic_signature("breast")
    predictor = dspy.Predict(sig)
    result = predictor(report=paragraphs, report_jsonized={})
"""
from __future__ import annotations

from functools import cache

import dspy

from ...models.breast import *  # noqa: F401, F403
from ...models.cervix import *  # noqa: F401, F403
from ...models.colon import *  # noqa: F401, F403

# Wildcard import every organ module so all signature classes are in
# this namespace — mirrors how pipeline.py resolves them.
from ...models.common import *  # noqa: F401, F403
from ...models.esophagus import *  # noqa: F401, F403
from ...models.liver import *  # noqa: F401, F403
from ...models.lung import *  # noqa: F401, F403
from ...models.modellist import organmodels
from ...models.pancreas import *  # noqa: F401, F403
from ...models.prostate import *  # noqa: F401, F403
from ...models.stomach import *  # noqa: F401, F403
from ...models.thyroid import *  # noqa: F401, F403

MONOLITHIC_DOCSTRING = (
    "You are a cancer registrar. Extract ALL structured fields listed "
    "below from the given {organ} cancer excision report in a single "
    "pass. DO NOT JUST RETURN NULL. If an item is not present in the "
    "report, return null for that item, but try your best to fill in "
    "the others. Return every field in your response even if it is "
    "null — do not omit fields.\n\n"
    "REQUIRED FIELDS (you MUST include every name below as a key in "
    "your output, with the value or null):\n{field_list}"
)

INPUT_FIELD_NAMES = {"report", "report_jsonized"}


def _iter_output_fields(cls: type) -> list[tuple[str, object, object]]:
    """Yield ``(name, type_hint, dspy.OutputField descriptor)`` for each
    output field declared on a DSPy signature class.

    DSPy stores fields on ``cls.output_fields`` (a dict of pydantic
    ``FieldInfo``) — NOT on ``cls.__dict__``. The previous implementation
    inspected ``__dict__`` and silently produced an empty list, which
    caused ``get_monolithic_signature`` to build degenerate signatures
    with no output fields at all (the root cause of the
    ``dspy_monolithic`` / ``str_outputs`` "no organ-specific output" bug).
    """
    out = []
    output_fields = getattr(cls, "output_fields", {}) or {}
    for name, finfo in output_fields.items():
        if name in INPUT_FIELD_NAMES:
            continue
        type_hint = finfo.annotation
        # Re-build a fresh OutputField descriptor from the original
        # `desc` so the merged signature renders the same prompt the
        # parent pipeline uses. (We can't reuse the FieldInfo object
        # directly because dspy.Signature's metaclass will overwrite
        # json_schema_extra during class construction.)
        extras = dict(finfo.json_schema_extra or {})
        descriptor = dspy.OutputField(desc=extras.get("desc", ""))
        out.append((name, type_hint, descriptor))
    return out


@cache
def get_monolithic_signature(organ: str) -> type[dspy.Signature]:
    """Return a dynamically-built dspy.Signature class that contains
    every output field declared by the per-subsection signatures for
    the given organ.

    Field conflicts (same name declared by multiple subsection
    signatures) are resolved by first-wins — which matches the parent
    pipeline's `output_report["cancer_data"].update(organ_data)`
    ordering in `pipeline.CancerPipeline.forward`.
    """
    if organ not in organmodels:
        raise ValueError(f"Unknown organ '{organ}'. "
                         f"Known: {sorted(organmodels)}")

    merged_annotations: dict[str, object] = {}
    merged_attrs: dict[str, object] = {}
    seen_fields: set[str] = set()

    for sig_name in organmodels[organ]:
        cls = globals().get(sig_name)
        if cls is None:
            raise RuntimeError(
                f"Per-subsection signature {sig_name!r} not importable — "
                f"check the wildcard imports at the top of this module.")
        for name, type_hint, descriptor in _iter_output_fields(cls):
            if name in seen_fields:
                continue
            seen_fields.add(name)
            merged_annotations[name] = type_hint
            merged_attrs[name] = descriptor

    # Two standard input fields, identical to the parent modular design.
    merged_annotations = {
        "report": list,
        "report_jsonized": dict,
        **merged_annotations,
    }
    merged_attrs["report"] = dspy.InputField(
        desc="The pathological report for this cancer excision, "
             "separated into paragraphs.")
    merged_attrs["report_jsonized"] = dspy.InputField(
        desc="A roughly structured JSON summary of the report, produced "
             "by an upstream signature. May be an empty dict if that step "
             "is skipped.")
    merged_attrs["__annotations__"] = merged_annotations
    field_names = [n for n in merged_annotations if n not in INPUT_FIELD_NAMES]
    field_list = "\n".join(f"  - {n}" for n in field_names)
    merged_attrs["__doc__"] = MONOLITHIC_DOCSTRING.format(
        organ=organ, field_list=field_list)

    cls_name = f"{organ.title()}CancerMonolithic"
    sig_cls = type(cls_name, (dspy.Signature,), merged_attrs)
    return sig_cls


def list_monolithic_fields(organ: str) -> list[str]:
    """Return the ordered list of output field names that the monolithic
    signature for this organ will produce. Useful for validation and
    for driving prompt construction in the raw-JSON runner."""
    sig = get_monolithic_signature(organ)
    return [
        name for name in sig.__annotations__
        if name not in INPUT_FIELD_NAMES
    ]


if __name__ == "__main__":
    # Smoke test: list field counts per organ.
    for organ in sorted(organmodels):
        fields = list_monolithic_fields(organ)
        print(f"{organ:12s}  {len(fields):3d} fields")
