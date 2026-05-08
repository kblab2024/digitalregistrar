# Canonical schema layer

In v2 the relationship between Pydantic, DSPy, and JSON Schema is
inverted from the legacy layout. **Pydantic is the source of truth.**

Two representations describe the same fields, and concordance is
enforced in CI:

1. **Pydantic case-models** in
   [`schemas/pydantic/<organ>.py`](../src/digital_registrar_research/schemas/pydantic)
   — hand-authored, the SINGLE SOURCE OF TRUTH. DSPy signatures are
   built dynamically from these by the
   [signature factory](../src/digital_registrar_research/signatures/factory.py)
   at pipeline construction time.
2. **JSON Schemas** in
   [`schemas/data/<organ>.json`](../src/digital_registrar_research/schemas/data)
   — auto-generated from `model.model_json_schema()` by the
   `registrar-schemas` CLI. Consumed by the annotation UI and the
   raw-JSON ablation runners.

The legacy DSPy signatures in `models/<organ>.py` still exist (the
legacy pipeline uses them), but they are no longer canonical. They will
be retired in a follow-up PR after v2 validates.

Concordance is enforced by
[`tests/test_schema_concordance.py`](../tests/test_schema_concordance.py)
and by `registrar-schemas --check` in CI.

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
        # Order = extraction order = first-wins order.
        "nonnested":   "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "dcis":        "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "grading":     "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "staging":     "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "margins":     "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "lymph_nodes": "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
        "biomarkers":  "...lead-in... " + DEFAULT_GROUP_INSTRUCTION,
    }

    # --- nonnested ---
    procedure: Literal['partial_mastectomy', ...] | None = GroupedField(
        group="nonnested",
        desc="identify which surgery procedure was used.",
    )
    # --- staging ---
    pT: Literal["t1mi", "t1a", ...] | None = GroupedField(
        group="staging", desc="identify the pT category.")
    # --- margins (nested) ---
    margins: list[BreastMargin] | None = GroupedField(
        group="margins", desc="return all involved margins...")
```

### Group-tag conventions

The current 10 organ schemas use these group tags (mirroring the
legacy decomposition):

`nonnested` · `staging` · `margins` · `lymph_nodes` · `biomarkers` ·
`dcis` (breast only) · `grading` (breast only) · `extent` (liver only) ·
`vascular_invasion` (liver only) · `othernested` (lung only)

A new organ can use whichever group names make clinical sense — the
factory only requires that every field has a `group=` and every used
group has an entry in `_GROUP_INSTRUCTIONS`.

### Why descriptions are duplicated

`GroupedField` writes the description to BOTH `Pydantic.Field(description=)`
and `json_schema_extra["dspy_desc"]`. The first feeds JSON schema
generation; the second feeds the factory. Updating one without the other
will drift — `field_desc()` always reads `dspy_desc` first, so the
factory wins. Keep them aligned, ideally by editing through the future
[GUI tool](schema_gui_blueprint.md).

### Recurring nested types: auto-expansion via `MarginField` / `LNField`

Surgical margins and lymph nodes follow a recurring pattern across all
ten organs — only the *category vocabulary* (and, for LN, whether a side
field exists) varies. The shared boilerplate (`margin_involved: bool`,
`distance: int | None`, `involved: int`, `examined: int`,
`station_name: str | None`, plus the long distance description) lives in
[`schemas/pydantic/_common_factories.py`](../src/digital_registrar_research/schemas/pydantic/_common_factories.py).

The schema author writes ONLY the organ-specific category Literal. The
`@auto_expand_recurring` class decorator detects fields tagged with
:func:`MarginField` / :func:`LNField` and rewrites them at class-creation
time:

```python
from typing import ClassVar, Literal
from pydantic import BaseModel
from ._factory_helpers import GroupedField, DEFAULT_GROUP_INSTRUCTION
from ._common_factories import (
    LNField, LymphNodeSide, MarginField, auto_expand_recurring,
)

@auto_expand_recurring
class BladderCancerCase(BaseModel):
    """Canonical extracted case record for bladder cancer."""

    _GROUP_INSTRUCTIONS: ClassVar[dict[str, str]] = {
        "nonnested":   "..." + DEFAULT_GROUP_INSTRUCTION,
        "margins":     "..." + DEFAULT_GROUP_INSTRUCTION,
        "lymph_nodes": "..." + DEFAULT_GROUP_INSTRUCTION,
    }

    procedure: Literal[...] | None = GroupedField(group="nonnested", desc="...")

    # The schema declares ONLY the category Literal; the decorator rewrites
    # this to ``list[BladderMargin] | None`` and synthesises BladderMargin.
    margins: Literal["proximal", "distal", "perivesical", "others"] | None = MarginField(
        desc="return all involved margins...",
    )

    regional_lymph_node: Literal[
        "pelvic", "internal_iliac", "external_iliac", "others",
    ] | None = LNField(
        desc="return all involved regional lymph nodes...",
        side_type=LymphNodeSide,   # most organs; lung uses LymphNodeSideRL; some omit (None)
    )
```

After decoration, `BladderCancerCase.model_fields["margins"].annotation`
is `list[BladderMargin] | None` and the synthesised `BladderMargin`
nested type appears in the JSON schema's `$defs`. The factory's
``json_schema_extra`` (group / dspy_desc tags) is preserved on the
rewritten field so the v2 signature factory keeps working unchanged.

**Naming.** The synthesised type is named `<Organ>Margin` /
`<Organ>LN` where `<Organ>` is derived from the case-model class name.
Three legacy naming-drift overrides are pinned in
``_TYPE_PREFIX_OVERRIDES`` so colorectal synthesises ``Colon{Margin,LN}``,
cervix uses ``"cervical"`` in descriptions, and esophagus uses
``"esophageal"``. Pass ``type_name=`` on `MarginField`/`LNField` if you
need a different override on a per-call basis.

**When to skip the markers.** Liver's LN is the only outlier (no category,
no side); it uses :func:`make_ln_type` directly with `category_type=None`
and a regular `GroupedField`. Prostate's "margins" group is a set of
flat fields, not a list — `MarginField` doesn't apply there.

**Lower-level escape hatches.** :func:`make_margin_type` and
:func:`make_ln_type` are still available if you need to build a nested
type explicitly (for tests, GUI tooling, or non-standard organs). Both
also accept ``Enum`` in place of ``Literal`` — JSON Schema output is
identical for the two.

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

The 10 `<Organ>CancerCase` classes are also importable directly:

```python
from digital_registrar_research.schemas.pydantic import (
    LungCancerCase, BreastCancerCase,
)
```

The schema-authoring helpers are re-exported from the same module for
convenience:

```python
from digital_registrar_research.schemas.pydantic import (
    GroupedField, DEFAULT_GROUP_INSTRUCTION,
    iter_fields_by_group, group_order, group_instructions,
)
```

## Regenerating JSON schemas

When you edit a Pydantic case-model:

```bash
registrar-schemas              # rewrites src/.../schemas/data/*.json
registrar-schemas --check      # CI invocation; exit 1 if anything drifted
```

CI fails any PR that changes a Pydantic schema without regenerating the
JSON.

## Concordance

[`tests/test_schema_concordance.py`](../tests/test_schema_concordance.py) pins:

- **`test_pydantic_to_json_parity[organ]`** —
  `model.model_json_schema()` matches the on-disk
  `data/<organ>.json` byte-for-byte.
- **`test_every_field_has_a_group[organ]`** — every Pydantic field
  carries a non-empty `group=` tag.
- **`test_every_group_has_an_instruction[organ]`** — every group used
  by a field has an entry in `_GROUP_INSTRUCTIONS`.
- **`test_router_literal_matches_registry`** — the factory-built
  router's `cancer_category` Literal == `set(CASE_MODELS) | {"others"}`.
- **`test_jsonize_literal_excludes_others`** — preserves the legacy
  asymmetry (the jsonize Literal does NOT include `"others"`).

Run with: `pytest tests/test_schema_concordance.py
tests/pipeline/test_v2_parity.py`.

## Bootstrap script (one-shot, throwaway)

[`scripts/bootstrap_schema_v2.py`](../scripts/bootstrap_schema_v2.py)
generates the v2 case-model files from the legacy DSPy signatures. It
walks each legacy signature's `OutputField`s, derives a group tag from
the class-name suffix, pulls the docstring as the group instruction, and
writes `<organ>.py.new` next to the existing wrappers.

After all 10 schemas are committed via this two-step seed-and-refine
flow, the bootstrap script and
[`schemas/pydantic/_builder.py`](../src/digital_registrar_research/schemas/pydantic/_builder.py)
should both be deleted in a follow-up PR (they're marked DEPRECATED).

## Naming-drift note: colorectal vs colon

The registry key is `"colorectal"` everywhere — the `cancer_category`
Literal in the router, the JSON schema filename, the annotation UI
dropdown. The nested types it references (`ColonMargin`, `ColonLN`,
`ColonBiomarker`) live in `_common_types.py` under the `Colon*` prefix
because that's how the legacy code authored them, and the type name
doesn't have to match the registry key. The concordance test
`test_colorectal_uses_colon_common_types` pins this mapping so the
inconsistency can't propagate further.

## Adding a new cancer type

1. Author `schemas/pydantic/<organ>.py` (a `BaseModel` with
   `_GROUP_INSTRUCTIONS` and `GroupedField` declarations — copy an
   existing organ as a template; if the schema needs new nested types,
   add them to `_common_types.py` first).
2. Register the new class in
   [`schemas/pydantic/__init__.py`](../src/digital_registrar_research/schemas/pydantic/__init__.py)'s
   `CASE_MODELS` dict.
3. Run `registrar-schemas` to generate `schemas/data/<organ>.json`.
4. Run `pytest tests/test_schema_concordance.py
   tests/pipeline/test_v2_parity.py` to validate.

That's it — no DSPy edits, no router edits. The factory rebuilds DSPy
signatures from the registry on the next pipeline construction. The
router's `cancer_category` Literal auto-updates because it's derived
from `sorted(CASE_MODELS)`.

## Bladder note

`models/bladder.py` exists with legacy DSPy signatures, and
`schemas/data/bladder.json` is hand-curated, but bladder is **not** in
`CASE_MODELS` — i.e. neither pipeline engine routes to it. The
annotation UI knows about bladder via its own `CANCER_TO_FILE` map, so
doctors can hand-annotate bladder cases. To promote bladder, follow the
"Adding a new cancer type" steps above with `<organ>="bladder"`.
