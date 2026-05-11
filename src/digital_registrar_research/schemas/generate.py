"""Regenerate `data/<organ>.json` from the canonical Pydantic case-models.

Usage:
    python -m digital_registrar_research.schemas.generate          # write
    python -m digital_registrar_research.schemas.generate --check  # verify only; exit 1 on drift

CI runs `--check` on every PR so a change to a DSPy signature that would
change the generated schema can't silently diverge from the checked-in
JSON artifact that downstream tools (annotation UI, ablations raw-JSON
runner) consume.
"""
from __future__ import annotations

import argparse
import json
import sys
import typing as _t
from pathlib import Path

from pydantic import BaseModel

from .extraction import EXTRACTION_META, FieldMeta
from .pydantic import CASE_MODELS, IsCancerCase

SCHEMAS_DATA_DIR = Path(__file__).resolve().parent / "data"

# Per-organ models + the top-level routing case-model; everything but `bladder`,
# which remains a hand-curated annotation-UI schema (no DSPy signature pipeline yet).
ALL_MODELS: dict[str, type] = {**CASE_MODELS, "common": IsCancerCase}


def _nested_type_prefix_map(model: type[BaseModel]) -> dict[str, str]:
    """Build ``{NestedTypeName: parent_field_name}`` for one-level nested BaseModels.

    Used by :func:`inject_descriptions_from_meta` to map ``$defs`` entries
    back to their dotted-key prefix in the metadata dict. Today every
    nested type appears under exactly one parent field
    (``BreastBiomarker`` -> ``biomarkers``, ``BreastMargin`` -> ``margins``,
    etc.), so first-match is unambiguous.
    """
    out: dict[str, str] = {}

    def find_basemodel(annotation: object) -> type[BaseModel] | None:
        if (
            isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
            and annotation is not BaseModel
        ):
            return annotation
        for arg in _t.get_args(annotation):
            found = find_basemodel(arg)
            if found is not None:
                return found
        return None

    for fname, fi in model.model_fields.items():
        nested = find_basemodel(fi.annotation)
        if nested is not None:
            out.setdefault(nested.__name__, fname)
    return out


def inject_descriptions_from_meta(
    schema: dict,
    field_meta: dict[str, FieldMeta],
    model: type[BaseModel],
) -> None:
    """Mutate ``schema`` in place: assign ``description`` from metadata.

    Top-level fields look up ``field_meta[name]["desc"]``. Nested-type
    sub-fields (``$defs[<NestedType>].properties[<sub>]``) look up
    ``field_meta[f"{parent_field}.{sub}"]["desc"]`` using the
    nested-type-to-parent-field map derived from ``model``.

    A metadata key without a matching schema property is a no-op (the
    schema is the source of structural truth). A schema property without
    a metadata entry is left with whatever description Pydantic emitted
    (if any). The Phase-0/1 invariant is that the metadata descriptions
    must exactly match the legacy ``GroupedField(desc=...)`` strings, so
    the rendered JSON stays byte-identical to the on-disk snapshot.
    """
    properties = schema.get("properties", {}) or {}
    for fname, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        entry = field_meta.get(fname)
        if not entry:
            continue
        desc = entry.get("desc") or ""
        if desc:
            # ``description`` is the canonical JSON-Schema slot;
            # ``dspy_desc`` mirrors it for the legacy on-disk shape so
            # consumers that already key off ``dspy_desc`` (older eval
            # pipelines, the annotation UI) keep working.
            prop["description"] = desc
            prop["dspy_desc"] = desc
        group = entry.get("group")
        if group:
            prop["group"] = group
        # Pydantic auto-derives ``title`` for primitive-typed fields but
        # not for ``$ref``-containing ``anyOf`` (e.g. ``StrEnum | None``).
        # Backfill so the on-disk JSON shape stays consistent across
        # lean-mode (StrEnum) and legacy (inline Literal) authoring.
        if "title" not in prop:
            prop["title"] = _autotitle(fname)

    defs = schema.get("$defs", {}) or {}
    prefix_map = _nested_type_prefix_map(model)
    for def_name, def_schema in defs.items():
        if not isinstance(def_schema, dict):
            continue
        prefix = prefix_map.get(def_name)
        if prefix is None:
            continue
        sub_props = def_schema.get("properties", {}) or {}
        for sub_name, sub_prop in sub_props.items():
            if not isinstance(sub_prop, dict):
                continue
            entry = field_meta.get(f"{prefix}.{sub_name}")
            if entry and entry.get("desc"):
                sub_prop["description"] = entry["desc"]
            if "title" not in sub_prop:
                sub_prop["title"] = _autotitle(sub_name)


def _autotitle(field_name: str) -> str:
    """Derive a JSON-Schema ``title`` from a field name the way Pydantic does
    for primitive fields: replace underscores with spaces, then title-case.

    ``"biomarker_category"`` -> ``"Biomarker Category"``,
    ``"tnm_descriptor"`` -> ``"Tnm Descriptor"``.
    """
    return field_name.replace("_", " ").title()


def _generated_banner(name: str) -> str:
    """Standard 'do-not-edit-by-hand' banner injected into every rendered JSON."""
    return (
        f"AUTO-GENERATED — DO NOT EDIT. Source of truth: "
        f"src/digital_registrar_research/schemas/pydantic/{name}.py "
        f"+ src/digital_registrar_research/schemas/extraction/{name}.py "
        f"+ src/digital_registrar_research/schemas/aliases/{name}.toml (if present). "
        f"Regenerate via `registrar-schemas`. CI fails on drift."
    )


def _render_model(name: str, model: type) -> str:
    """Return JSON-schema text for a Pydantic model (sorted keys + 2-space indent).

    When ``name`` is registered in :data:`EXTRACTION_META`, descriptions
    are injected from the metadata dict before serialization. When it
    isn't (Phase-0 organs, or the ``common`` IsCancer model), the schema
    is passed through unchanged. A ``$comment`` banner is always
    prepended so a hand-edit reads as obviously wrong even before CI's
    ``--check`` catches the drift.
    """
    schema = model.model_json_schema()
    organ_meta = EXTRACTION_META.get(name)
    if organ_meta is not None:
        inject_descriptions_from_meta(schema, organ_meta["fields"], model)
    schema["$comment"] = _generated_banner(name)
    return json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _write_all() -> list[Path]:
    SCHEMAS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in sorted(ALL_MODELS.items()):
        out_path = SCHEMAS_DATA_DIR / f"{name}.json"
        out_path.write_text(_render_model(name, model), encoding="utf-8")
        written.append(out_path)
    return written


def _check_all() -> list[str]:
    """Return a list of names whose on-disk JSON doesn't match the current Pydantic models."""
    drifted: list[str] = []
    for name, model in sorted(ALL_MODELS.items()):
        on_disk_path = SCHEMAS_DATA_DIR / f"{name}.json"
        if not on_disk_path.exists():
            drifted.append(f"{name} (missing file)")
            continue
        on_disk = on_disk_path.read_text(encoding="utf-8")
        fresh = _render_model(name, model)
        if on_disk.strip() != fresh.strip():
            drifted.append(name)
    return drifted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="Verify only; fail on drift.")
    args = ap.parse_args()

    if args.check:
        drifted = _check_all()
        if drifted:
            print(f"[schemas] DRIFT detected in: {', '.join(drifted)}", file=sys.stderr)
            print("[schemas] Run `python -m digital_registrar_research.schemas.generate` to regenerate.", file=sys.stderr)
            sys.exit(1)
        print(f"[schemas] OK — all {len(ALL_MODELS)} schemas match their Pydantic models.")
        return

    written = _write_all()
    print(f"[schemas] Wrote {len(written)} schemas to {SCHEMAS_DATA_DIR}:")
    for p in written:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
