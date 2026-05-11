# Canonical schema architecture

> Last updated: 2026-05-11 · Reflects: `d11d072`

The cancer schema is structured as **three layers of source-of-truth + one
generated artifact**. All four representations describe the same fields;
concordance is enforced in CI.

## The four layers

```
┌──────────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — Pydantic shape                                                 │
│   src/digital_registrar_research/schemas/pydantic/<organ>.py             │
│   ✱ canonical enum values + types + groups                               │
│   ✱ consumed at inference time by pipeline_factory.py                    │
└──────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│ LAYER 2 — Extraction metadata                                            │
│   src/digital_registrar_research/schemas/extraction/<organ>.py           │
│   ✱ per-field `desc` + `group`, plus GROUP_INSTRUCTIONS                  │
│   ✱ loaded once, cached in EXTRACTION_META                               │
└──────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│ LAYER 3 — Aliases (TOML, domain-expert-editable)                         │
│   src/digital_registrar_research/schemas/aliases/<organ>.toml            │
│   ✱ canonical enum value → list of surface-form strings                  │
│   ✱ merged into value_hints by _build_organ_meta()                       │
│   ✱ NO Python edit required to add a new surface form                    │
└──────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│ GENERATED — JSON schema                                                  │
│   src/digital_registrar_research/schemas/data/<organ>.json               │
│   ✱ written by `registrar-schemas`; AUTO-GENERATED, DO NOT EDIT          │
│   ✱ consumed by the annotation UI and the raw-JSON ablation runner       │
│   ✱ NOT used by the main inference pipeline                              │
└──────────────────────────────────────────────────────────────────────────┘
```

## Edit table

| Want to change… | Edit | Then |
|---|---|---|
| A new canonical enum value | `schemas/pydantic/<organ>.py` | `registrar-schemas`, commit JSON diff |
| A surface form / synonym / clone name | `schemas/aliases/<organ>.toml` | Nothing — loaded on next import |
| A per-field description | `schemas/extraction/<organ>.py` | Nothing — loaded on next import |
| The group ordering or instruction | `schemas/extraction/<organ>.py` | Nothing |
| Anything in `schemas/data/*.json` | **Don't** — every file's `$comment` says so |

## Invariants enforced

- `_build_organ_meta` **raises** if `aliases/<organ>.toml` references an
  unknown field path.
- `_build_organ_meta` **warns + discards** if anyone hand-adds
  `value_hints` to a Layer 2 `FIELD_META` entry (TOML-only).
- [`tests/schemas/test_schema_concordance.py::test_alias_canonicals_exist_in_enum`](../../tests/schemas/test_schema_concordance.py)
  raises if a TOML canonical key isn't a real enum value.
- `registrar-schemas --check` exits 1 if the on-disk JSON drifts from
  `model.model_json_schema()`.

## Authoring a case-model

Every Pydantic field carries a `group=` tag that drives factory
decomposition. The class-level `_GROUP_INSTRUCTIONS: ClassVar[dict]`
maps each tag to its per-group LM instruction. Insertion order is
load-bearing — it controls extraction order AND first-wins semantics on
duplicate field names.

```python
from typing import ClassVar, Literal
from pydantic import BaseModel
from ._factory_helpers import GroupedField, DEFAULT_GROUP_INSTRUCTION
from ._common_types import BreastMargin, BreastLN, BreastBiomarker

class BreastCancerCase(BaseModel):
    """Canonical extracted case record for breast cancer."""

    _GROUP_INSTRUCTIONS: ClassVar[dict[str, str]] = {
        "nonnested":   "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "dcis":        "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "grading":     "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "staging":     "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "margins":     "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "lymph_nodes": "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "biomarkers":  "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
    }

    procedure: Literal['partial_mastectomy', ...] | None = GroupedField(
        group="nonnested", desc="identify which surgery procedure was used.")
    pT: Literal["t1mi", "t1a", ...] | None = GroupedField(
        group="staging", desc="identify the pT category.")
    margins: list[BreastMargin] | None = GroupedField(
        group="margins", desc="return all involved margins...")
```

### Group tags in current use

`nonnested` · `staging` · `margins` · `lymph_nodes` · `biomarkers` ·
`dcis` (breast) · `grading` (breast) · `extent` (liver) ·
`vascular_invasion` (liver) · `othernested` (lung).

A new organ can use whichever group names make clinical sense — the
factory requires only that every field has a `group=` and every used
group has an entry in `_GROUP_INSTRUCTIONS`.

### Why descriptions are duplicated

`GroupedField` writes the description to BOTH `Pydantic.Field(description=)`
and `json_schema_extra["dspy_desc"]`. The first feeds JSON schema
generation; the second feeds the factory. `field_desc()` reads
`dspy_desc` first, so the factory wins on drift. Keep them aligned.

### Recurring nested types: `MarginField` / `LNField`

Surgical margins and lymph nodes share boilerplate across all ten
organs — only the *category vocabulary* (and, for LN, whether a side
field exists) varies. The shared fields live in
[`schemas/pydantic/_common_factories.py`](../../src/digital_registrar_research/schemas/pydantic/_common_factories.py).

The schema author writes ONLY the organ-specific category Literal. The
`@auto_expand_recurring` class decorator detects fields tagged with
`MarginField` / `LNField` and rewrites them at class-creation time:

```python
@auto_expand_recurring
class BladderCancerCase(BaseModel):
    ...
    margins: Literal["proximal", "distal", "perivesical", "others"] | None = MarginField(
        desc="return all involved margins...")

    regional_lymph_node: Literal[
        "pelvic", "internal_iliac", "external_iliac", "others",
    ] | None = LNField(
        desc="return all involved regional lymph nodes...",
        side_type=LymphNodeSide,
    )
```

After decoration, `BladderCancerCase.model_fields["margins"].annotation`
is `list[BladderMargin] | None` and the synthesised `BladderMargin`
nested type appears in the JSON schema's `$defs`. `json_schema_extra`
(group / dspy_desc tags) is preserved.

**Naming.** The synthesised type is named `<Organ>Margin` / `<Organ>LN`
where `<Organ>` is derived from the case-model class name. Three
legacy naming-drift overrides are pinned in `_TYPE_PREFIX_OVERRIDES` so
colorectal synthesises `Colon{Margin,LN}`, cervix uses `"cervical"`,
and esophagus uses `"esophageal"`.

**When to skip the markers.** Liver's LN is the only outlier (no
category, no side); it uses `make_ln_type` directly with
`category_type=None` and a regular `GroupedField`. Prostate's "margins"
group is a set of flat fields — `MarginField` doesn't apply there.

## Public API

```python
from digital_registrar_research.schemas import (
    list_organs,            # ['breast', 'cervix', 'colorectal', 'esophagus',
                            #  'liver', 'lung', 'pancreas', 'prostate', 'stomach', 'thyroid']
    load_pydantic_model,    # organ -> type[BaseModel]
    load_json_schema,       # organ -> dict (read from packaged data/)
    CASE_MODELS,            # the registry itself
)

LungCase = load_pydantic_model("lung")
case = LungCase.model_validate(some_extracted_dict)
print(case.model_dump_json(exclude_none=True))
```

The 10 `<Organ>CancerCase` classes import directly too:

```python
from digital_registrar_research.schemas.pydantic import (
    LungCancerCase, BreastCancerCase,
)
```

Schema-authoring helpers are re-exported from the same module:

```python
from digital_registrar_research.schemas.pydantic import (
    GroupedField, DEFAULT_GROUP_INSTRUCTION,
    iter_fields_by_group, group_order, group_instructions,
)
```

## Regenerating JSON schemas

```bash
registrar-schemas              # rewrites src/.../schemas/data/*.json
registrar-schemas --check      # CI invocation; exit 1 on drift
```

CI fails any PR that changes a Pydantic schema without regenerating the
JSON.

## Concordance

[`tests/schemas/test_schema_concordance.py`](../../tests/schemas/test_schema_concordance.py)
pins:

- **`test_pydantic_to_json_parity[organ]`** —
  `model.model_json_schema()` matches the on-disk `data/<organ>.json`
  byte-for-byte.
- **`test_every_field_has_a_group[organ]`** — every Pydantic field
  carries a non-empty `group=` tag.
- **`test_every_group_has_an_instruction[organ]`** — every group used
  by a field has an entry in `_GROUP_INSTRUCTIONS`.
- **`test_router_literal_matches_registry`** — the factory-built
  router's `cancer_category` Literal == `set(CASE_MODELS) | {"others"}`.
- **`test_jsonize_literal_excludes_others`** — preserves the legacy
  asymmetry (the jsonize Literal does NOT include `"others"`).
- **`test_alias_canonicals_exist_in_enum`** — every TOML alias canonical
  key is a real enum value in the Pydantic model.

Run with `pytest tests/schemas/ tests/pipeline/test_v2_parity.py`.

## Naming-drift note: colorectal vs colon

The registry key is `"colorectal"` everywhere — `cancer_category`
Literal in the router, the JSON schema filename, the annotation UI
dropdown. The nested types (`ColonMargin`, `ColonLN`, `ColonBiomarker`)
live in `_common_types.py` under the `Colon*` prefix because that's how
the legacy code authored them; the type name doesn't have to match the
registry key. Concordance test `test_colorectal_uses_colon_common_types`
pins this so the inconsistency can't propagate further.

## Adding a new cancer type

1. Author `schemas/pydantic/<organ>.py` (BaseModel with
   `_GROUP_INSTRUCTIONS` and `GroupedField` declarations; copy an
   existing organ; if new nested types are needed, add them to
   `_common_types.py` first).
2. Register the new class in
   [`schemas/pydantic/__init__.py`](../../src/digital_registrar_research/schemas/pydantic/__init__.py)'s
   `CASE_MODELS` dict.
3. Run `registrar-schemas` to generate `schemas/data/<organ>.json`.
4. Run `pytest tests/schemas/ tests/pipeline/test_v2_parity.py`.

No DSPy edits, no router edits. The factory rebuilds DSPy signatures
from the registry on the next pipeline construction. The router's
`cancer_category` Literal auto-updates because it's derived from
`sorted(CASE_MODELS)`.

## Bootstrap & builder — archived

The one-shot generator `bootstrap_schema_v2.py` and the legacy
`schemas/pydantic/_builder.py` are archived at
[`scripts/_legacy/`](../../scripts/_legacy/). v2 shipped at commit
`259cf27`; the generator is no longer needed and is preserved only for
historical reference.

## Bladder note

`models/bladder.py` exists with legacy DSPy signatures, and
`schemas/data/bladder.json` is hand-curated, but bladder is **not** in
`CASE_MODELS` — i.e. neither pipeline engine routes to it. The
annotation UI knows about bladder via its own `CANCER_TO_FILE` map, so
doctors can hand-annotate bladder cases. To promote bladder, follow
"Adding a new cancer type" above with `<organ>="bladder"`.
