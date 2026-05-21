"""Per-organ extraction metadata — Layer 2 of the three-layer schema.

Layer 1 (``schemas/pydantic/<organ>.py``) owns shape & validation: types,
StrEnum membership, defaults, JSON Schema generation. Those modules carry
no LM-facing prompt content.

Layer 2 (this package, one file per organ) owns extraction semantics:
per-field LM-facing description, the group tag that drives per-group
decomposition, and the per-group instruction strings. Both the DSPy
factory and the JSON-schema CLI consume this layer; the Pydantic layer
stays clean.

Layer 3 (``schemas/aliases/<organ>.toml`` — data-only) carries
surface-form aliases for closed-vocabulary enum fields. Loaded at import
time and merged into ``EXTRACTION_META[organ]['fields'][path]['value_hints']``
by :func:`_build_organ_meta` below. Edit the TOML files directly to add
or change aliases — no Python edit, no JSON regen.

Conventions
-----------
``FIELD_META`` is keyed by field name. Nested sub-fields use a dotted
path: ``"biomarkers.percentage"`` overrides the description on the
``percentage`` field of the synthesised ``<Organ>Biomarker`` nested type.
This single dotted-key convention covers every nested type we synthesise
today (``<Organ>Margin``, ``<Organ>LN``, ``<Organ>Biomarker``).

``GROUP_INSTRUCTIONS`` is insertion-ordered; the iteration order doubles
as the canonical extraction order (mirrors the legacy
``_GROUP_INSTRUCTIONS`` ClassVar's role).

Each per-organ module exports ``FIELD_META`` and ``GROUP_INSTRUCTIONS``.
Registration is **automatic** — drop a new ``<organ>.py`` into this
directory and the next process import picks it up via
:mod:`._registry`. The Layer-1 / Layer-2 organ keys must match exactly;
a mismatch raises :class:`RuntimeError` at import time.
"""
from __future__ import annotations

import warnings
from typing import NotRequired, TypedDict, cast


class FieldMeta(TypedDict):
    """Per-field extraction metadata.

    ``desc`` is required. ``group`` is required for top-level fields and
    omitted for nested sub-field keys (those reached via the dotted
    convention, e.g. ``"biomarkers.percentage"``). ``examples`` is
    reserved for future use; the factory ignores it today.

    ``value_hints`` is **not authored in this module's per-organ files**.
    It is merged in at registry-build time from
    ``schemas/aliases/<organ>.toml`` (the Layer-3 data files). Any
    in-source ``value_hints`` will be detected by :func:`_build_organ_meta`
    and discarded with a warning — aliases live in the TOML data files
    only, so domain experts can edit them without touching Python.
    """

    desc: str
    group: NotRequired[str]
    examples: NotRequired[list[object]]
    value_hints: NotRequired[dict[str, list[str]]]


class OrganExtractionMeta(TypedDict):
    """The metadata bundle a single organ contributes to :data:`EXTRACTION_META`."""

    fields: dict[str, FieldMeta]
    groups: dict[str, str]


from ..aliases import load_aliases  # noqa: E402
from ._registry import discover_extraction_modules  # noqa: E402


def _build_organ_meta(organ: str, module) -> OrganExtractionMeta:
    """Construct one organ's ``OrganExtractionMeta`` from Layer 2 + Layer 3.

    Reads ``module.FIELD_META`` (Layer 2) and merges in
    ``load_aliases(organ)`` (Layer 3). Two invariants are enforced:

    - In-source ``value_hints`` keys on ``FIELD_META`` entries are
      discarded with a ``UserWarning``. Aliases are TOML-only.
    - A TOML field path that does not exist in ``FIELD_META`` raises
      ``ValueError`` at import time — a silently-dead alias is the worst
      failure mode (the LM keeps missing the variant with no signal why).
    """
    fields: dict[str, FieldMeta] = {}
    for path, entry in module.FIELD_META.items():
        copied = cast(FieldMeta, dict(entry))
        if "value_hints" in copied:
            warnings.warn(
                f"{module.__name__}: field {path!r} has in-source "
                f"`value_hints`. Aliases must live in "
                f"schemas/aliases/{organ}.toml only — in-source "
                "value_hints will be discarded.",
                UserWarning,
                stacklevel=2,
            )
            del copied["value_hints"]
        fields[path] = copied

    for path, hints in load_aliases(organ).items():
        if path not in fields:
            raise ValueError(
                f"aliases/{organ}.toml references unknown field path "
                f"{path!r} (not in {module.__name__}.FIELD_META)"
            )
        fields[path]["value_hints"] = hints
    return {"fields": fields, "groups": module.GROUP_INSTRUCTIONS}


def _build_extraction_meta() -> dict[str, OrganExtractionMeta]:
    """Discover Layer-2 modules, verify Layer-1 parity, build the registry.

    Performs the cross-layer parity check inline so a missing extraction
    module (or stray extraction file with no matching pydantic case-model)
    fails loudly at import time rather than silently dropping the organ.
    """
    # Imported here to avoid a top-level cycle while Layer 1's __init__
    # is still in flight.
    from ..pydantic import CASE_MODELS

    pydantic_keys = set(CASE_MODELS.keys())
    extraction_modules = discover_extraction_modules()
    extraction_keys = set(extraction_modules.keys())

    if pydantic_keys != extraction_keys:
        missing_l2 = pydantic_keys - extraction_keys
        orphan_l2 = extraction_keys - pydantic_keys
        details = []
        if missing_l2:
            details.append(
                f"organs in pydantic/ but missing extraction module: {sorted(missing_l2)}"
            )
        if orphan_l2:
            details.append(
                f"organs in extraction/ but missing pydantic case-model: {sorted(orphan_l2)}"
            )
        raise RuntimeError(
            "Layer 1 / Layer 2 organ-set mismatch — " + "; ".join(details)
        )

    return {
        organ: _build_organ_meta(organ, extraction_modules[organ])
        for organ in sorted(pydantic_keys)
    }


EXTRACTION_META: dict[str, OrganExtractionMeta] = _build_extraction_meta()


__all__ = ["FieldMeta", "OrganExtractionMeta", "EXTRACTION_META"]
