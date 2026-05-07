"""Build a canonical Pydantic case-model per organ.

The source material is already the DSPy signatures in `...models/<organ>.py`.
Rather than hand-duplicate ~100 field definitions across ten organs, we
introspect each per-subsection DSPy signature (see `organmodels` in
`models/modellist.py`) and compose one flat `<Organ>CancerCase(BaseModel)`
whose fields mirror the union of every signature's `OutputField`.

The resulting Pydantic class IS the canonical representation that tools
consume — it round-trips gold annotations, serialises to JSON, and is the
shape `schemas/data/<organ>.json` is generated from. Because construction
is derived from the DSPy signatures, the two are concordant by
construction; the concordance test in `tests/test_schema_concordance.py`
pins this contract so any future drift surfaces loudly.
"""
from __future__ import annotations

from functools import cache
from typing import Any

from pydantic import BaseModel, Field, create_model

from ...models.breast import *  # noqa: F401, F403
from ...models.cervix import *  # noqa: F401, F403
from ...models.colon import *  # noqa: F401, F403

# Wildcard-import every per-organ DSPy signature module so signature
# classes are discoverable by name — mirrors how `pipeline.py` resolves
# them through `globals().get(name)`.
from ...models.common import *  # noqa: F401, F403
from ...models.esophagus import *  # noqa: F401, F403
from ...models.liver import *  # noqa: F401, F403
from ...models.lung import *  # noqa: F401, F403
from ...models.modellist import organmodels
from ...models.pancreas import *  # noqa: F401, F403
from ...models.prostate import *  # noqa: F401, F403
from ...models.stomach import *  # noqa: F401, F403
from ...models.thyroid import *  # noqa: F401, F403

INPUT_FIELD_NAMES = {"report", "report_jsonized"}


def _iter_signature_output_fields(cls: type) -> list[tuple[str, Any, Any]]:
    """Yield (name, type_hint, pydantic_field_info) for each OutputField on a DSPy signature class.

    DSPy signatures are themselves `pydantic.BaseModel` subclasses, so we read fields
    from `cls.model_fields`. OutputFields are tagged via
    `field_info.json_schema_extra["__dspy_field_type"] == "output"`.
    """
    out: list[tuple[str, Any, Any]] = []
    for name, field_info in getattr(cls, "model_fields", {}).items():
        if name in INPUT_FIELD_NAMES:
            continue
        extra = getattr(field_info, "json_schema_extra", None) or {}
        if isinstance(extra, dict) and extra.get("__dspy_field_type") != "output":
            continue
        desc = ""
        if isinstance(extra, dict):
            desc = extra.get("desc") or ""
        if not desc:
            desc = field_info.description or ""
        # `field_info.annotation` is the resolved type (already wraps Optional[...] for `X|None`).
        type_hint = field_info.annotation
        out.append((name, type_hint, Field(default=None, description=desc)))
    return out


def _merge_fields_for_organ(organ: str) -> dict[str, tuple[Any, Any]]:
    """Return `{field_name: (type_hint, Field)}` by first-wins union of the per-subsection signatures.

    First-wins matches `CancerPipeline.forward`'s `cancer_data.update(organ_data)`
    ordering: when two subsections define the same field name, the first one
    listed in `organmodels[organ]` wins.
    """
    import sys
    module_globals = sys.modules[__name__].__dict__

    if organ not in organmodels:
        raise ValueError(f"Unknown organ {organ!r}; known: {sorted(organmodels)}")

    merged: dict[str, tuple[Any, Any]] = {}
    for sig_name in organmodels[organ]:
        cls = module_globals.get(sig_name)
        if cls is None:
            raise RuntimeError(
                f"Signature class {sig_name!r} not importable in {__name__}. "
                f"Check the wildcard imports at the top of this module."
            )
        for name, type_hint, field_info in _iter_signature_output_fields(cls):
            if name in merged:
                continue
            # type_hint is already Optional via `X | None` in the source signatures;
            # don't double-wrap.
            merged[name] = (type_hint, field_info)
    return merged


@cache
def build_case_model(organ: str) -> type[BaseModel]:
    """Return (and cache) the canonical Pydantic case-model for `organ`.

    The model name is `<Organ>CancerCase` where `<Organ>` is the capitalised,
    public-facing organ key (e.g. `ColorectalCancerCase` even though the
    underlying DSPy signatures are named `ColonCancer*` — see the
    naming-drift note in docs/schemas.md).
    """
    fields = _merge_fields_for_organ(organ)
    class_name = f"{organ.capitalize()}CancerCase"
    model = create_model(class_name, __base__=BaseModel, **fields)
    model.__doc__ = (
        f"Canonical extracted case record for {organ} cancer. "
        f"Built dynamically from the DSPy signatures listed in "
        f"models.modellist.organmodels[{organ!r}]."
    )
    return model


__all__ = ["build_case_model", "INPUT_FIELD_NAMES"]
