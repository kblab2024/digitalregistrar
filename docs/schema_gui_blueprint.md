# Schema editor GUI — design blueprint

> Status: blueprint only. Not implemented in this PR. Tracked for a follow-up.

## Why

The v2 redesign makes Pydantic case-models the single source of truth.
Each `schemas/pydantic/<organ>.py` is now hand-authored: a `BaseModel`
with `_GROUP_INSTRUCTIONS: ClassVar[dict]` and one `GroupedField(group=,
desc=)` per output field. The clinician owns the schema; the engineer
owns the factory. Adding a new cancer type = write one Pydantic file.

Hand-editing Pydantic source works but it puts a non-trivial Python tax
on a clinician who, by their own admission, "speaks poor Pydantic." The
GUI removes that tax: a clinician can edit field definitions, group
assignments, enum values, and per-group LM instructions in a tabular
interface, then write the file back.

## Scope

In scope (this blueprint):
- Round-trip a single organ schema (load → edit → save).
- Author all field types currently used: `bool`, `int`, `str`, `Literal`,
  `list[X]`, `Optional[...]`, references to nested `BaseModel`s already
  defined in `schemas/pydantic/_common_types.py`.
- Manage `_GROUP_INSTRUCTIONS` (add/remove/reorder rows; insertion order
  is the extraction order).
- Author per-field group tag, description.
- Trigger JSON regeneration via the existing `registrar-schemas` CLI.
- Validate before save: every field has a `group=`; every used group has
  a `_GROUP_INSTRUCTIONS` entry; no orphan groups.

Out of scope (future):
- Authoring brand-new nested types in `_common_types.py` (the GUI uses
  what's already declared there; new nested types require an engineer).
- Bootstrapping a brand-new organ from scratch — first PR uses
  `scripts/bootstrap_schema_v2.py` to seed the file; the GUI takes over
  for refinement.
- Editing the legacy `models/<organ>.py` files (those are deprecated and
  will be deleted post-v2-validation).

## Stack

- **Streamlit** — already in the `[annotation]` extra; no new dependency.
  `st.data_editor` gives a tabular field editor with inline type/enum
  edits. The annotation app already uses Streamlit so the project doesn't
  pick up a second framework.
- **`ast`** for round-tripping the schema source. Read existing files via
  `ast.parse(...)` → walk `ClassDef` → `AnnAssign` to recover field
  declarations. Write back via a Jinja-style template (one
  `GroupedField(...)` per field, plus the `_GROUP_INSTRUCTIONS` dict).
- **`subprocess`** to invoke `python -m digital_registrar_research.schemas.generate`
  on save, refreshing `schemas/data/<organ>.json` from the new Pydantic.
- **`black` + `ruff format`** post-write so the generated file lands
  exactly the way `pre-commit` would format it (no spurious diffs).

## UX flow

1. Pick an organ from the `CASE_MODELS` registry dropdown (or "+ new").
   "+ new" requires the user to type the organ key (must be a valid
   Python identifier and not already in `CASE_MODELS`).

2. **Groups panel** (top half).
   `st.data_editor` row per group: `group_name`, `instruction_text`.
   Reorderable. Adding a new group inserts a row; deleting a group is
   blocked if any field still references it (validation at save).

3. **Fields panel** (bottom half).
   `st.data_editor` row per field with columns:
   - `field_name` (Python identifier)
   - `group` (dropdown of groups defined above)
   - `type_kind` (one of: bool, int, str, literal, list_of_literal,
     list_of_nested, nested) — drives the next two columns
   - `enum_values` (comma-separated, only when `type_kind` is literal /
     list_of_literal)
   - `nested_type` (dropdown of names in `_common_types.py`, only when
     `type_kind` is nested / list_of_nested)
   - `optional` (checkbox; default ON, mirrors current convention)
   - `description` (string)

4. **Preview panel**.
   Live-renders the Pydantic source the GUI would write. User can spot
   bad output before committing.

5. **Save** button:
   1. Run client-side validation (see below).
   2. Render the source file to a temporary path.
   3. Run `black` + `ruff` on the temp path.
   4. Move the temp file over `schemas/pydantic/<organ>.py` atomically.
   5. Run `python -m digital_registrar_research.schemas.generate` to
      refresh `schemas/data/<organ>.json`.
   6. Re-run `pytest tests/test_schema_concordance.py
      tests/pipeline/test_v2_parity.py` and surface failures inline.

## Validation rules (enforced before save)

- Every field has a non-empty `group` tag.
- Every used `group` has an entry in `_GROUP_INSTRUCTIONS`.
- No two fields share a name.
- Field names are valid Python identifiers; group names too.
- For `Literal` fields, no duplicate enum values.
- Editing an existing organ keeps the registry key stable (renaming an
  organ key is out of scope — would require sweeping `CASE_MODELS` and
  every file under `schemas/data/`).

## Round-trip details

**Loading:**
```python
import ast
src = open("schemas/pydantic/breast.py").read()
tree = ast.parse(src)
class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
# walk class_node.body for AnnAssign nodes -> field name, annotation, default call
```
For each `AnnAssign` whose `value` is `Call(func=Name('GroupedField'), ...)`,
extract `group=`, `desc=`, default value, and the annotation source. The
`_GROUP_INSTRUCTIONS` dict literal is a sibling assignment in the class body.

**Writing:**
- Re-emit the file from a Jinja template. Header (docstring + imports)
  is rendered from a fixed template; `_GROUP_INSTRUCTIONS` and field
  declarations are rendered from the in-memory state.
- Field annotations are rendered from `(type_kind, enum_values,
  nested_type, optional)` — no AST manipulation needed since we
  control the schema's authoring surface.

## Hand-off to the bootstrap script

`scripts/bootstrap_schema_v2.py` already produces a `<organ>.py.new`
file from the legacy DSPy signatures. The GUI picks up where the
bootstrap leaves off:

1. Engineer runs `python scripts/bootstrap_schema_v2.py --organ liver`.
2. Engineer reviews `liver.py.new` and renames it to `liver.py`.
3. Clinician opens the GUI, picks `liver`, and refines the
   group assignments / wording.

After all 10 organs are committed via this two-step flow, the bootstrap
script can be deleted.

## Anti-goals

- The GUI is **not** a replacement for code review on schema changes —
  every save still goes through git as a normal PR.
- The GUI does **not** edit the factory or pipeline logic. Those stay in
  Python.
- The GUI does **not** know about DSPy. It only edits Pydantic schemas;
  the factory rebuilds DSPy signatures from Pydantic at pipeline
  construction time.

## Estimate

Single-engineer week to first useful version (load + edit + save +
validate). Add another week if the GUI should also drive `pytest` from
inside Streamlit and surface failure traces inline.
