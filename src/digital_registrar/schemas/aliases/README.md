# Surface-form aliases (Layer 3)

Each TOML file in this directory maps **canonical enum value → list of
surface forms** the LM might see in pathology reports. The values are
loaded at Python import time and merged into
`EXTRACTION_META[organ]['fields'][path]['value_hints']`, which the LM
sees as part of the field description in the prompt.

This is the **right place to add or remove a surface-form variant**
(antibody clone name, all-caps biomarker, dialect spelling, legacy
data with a different canonical form, …). It does not require any
Python edit, JSON regen, or rebuild.

## How to add an alias

1. Open the relevant organ TOML (`lung.toml`, `breast.toml`, …). Create
   the file if it doesn't exist yet.
2. Under the field path (matching a key in
   `schemas/extraction/<organ>.py::FIELD_META`), add a string to the
   canonical value's list:
   ```toml
   ["biomarkers.biomarker_category"]
   alk = ["alk", "ALK", "ALK D5F3", "Alk-lung", "the new variant"]
   ```
3. Save. Done. The next pipeline process picks it up.

## What goes here vs. elsewhere

| Want to change… | Edit |
| --- | --- |
| **Surface form / synonym / clone name** | `schemas/aliases/<organ>.toml` ← **here** |
| Per-field description (the LM-facing explanation) | `schemas/extraction/<organ>.py` |
| Add / remove / rename an enum value | `schemas/pydantic/<organ>.py`, then run `registrar-schemas` |
| `schemas/data/<organ>.json` | **NEVER directly** — it is auto-generated |

## File format

Top-level TOML tables are **dotted field paths**, matching
`FIELD_META` keys. Inside each table, keys are the **canonical enum
value** (lowercase, matching the `LowerStrEnum` value), and values are
**lists of surface-form strings**:

```toml
# Comments are allowed (TOML supports them, unlike JSON).
# Convention: include the canonical form itself in the list so the
# rendered prompt is self-contained.

["biomarkers.biomarker_category"]
alk  = ["alk", "ALK", "ALK D5F3"]   # D5F3 is Roche's antibody clone
ros1 = ["ros1", "ROS1", "ROS-1"]
```

## Runtime semantics

| Edit | When does inference see it? |
| --- | --- |
| Add / change a value in a TOML file | **Next Python process import.** No JSON regen, no rebuild. |
| Same edit inside a long-running Jupyter kernel / web server | Stale until kernel restart (standard Python import caching). |

## Invariants enforced by CI

The loader in `schemas/extraction/__init__.py::_build_organ_meta`
raises if a TOML field path is not in `FIELD_META` (silently-dead
aliases are the worst failure mode). The test
`tests/test_schema_concordance.py::test_alias_canonicals_exist_in_enum`
raises if a canonical key in any TOML is not a real value on the
corresponding pydantic enum.

A typo'd field path or canonical value will fail loudly at import
time / CI, not silently no-op.
