# registrar-schema-gui

Streamlit editor for the three-layer per-organ schema. Round-trips
`schemas/pydantic/<organ>.py` (Layer 1 — shape & enums),
`schemas/extraction/<organ>.py` (Layer 2 — LM-facing descriptions &
group instructions), and `schemas/aliases/<organ>.toml` (Layer 3 —
surface-form synonyms). Save also calls `registrar-schemas` to refresh
the generated JSON snapshot.

## Quick start

```bash
pip install -e .[schema-gui]
registrar-schema-gui
```

The app binds to `localhost:8501` by default. Pick an organ from the
sidebar dropdown, click **Load**, edit across the 5 tabs, then **Save
all layers**.

## What it edits

| Tab | Layer | Storage |
|---|---|---|
| Enums | 1 | `LowerStrEnum` value lists in `pydantic/<organ>.py` |
| Fields | 1 | Bare field decls + `MarginSpec` / `LNSpec` / `BiomarkerSpec` markers + `_STAGING: StagingSpec(...)` |
| Descriptions & groups | 2 | `FIELD_META` + `GROUP_INSTRUCTIONS` in `extraction/<organ>.py` |
| Aliases | 3 | TOML tables in `aliases/<organ>.toml` |
| Preview & save | — | Side-by-side diff of on-disk vs rendered; save controls |

## Layer 1 read-only mode

Organs with hand-authored top-level classes (e.g. `lung` carries an
inline `LungHistologicalPattern: BaseModel` for the `othernested`
group) surface as **Layer 1 read-only** — the GUI can't faithfully
round-trip the hand-authored block, so to avoid corrupting it,
Layer 1 saves are blocked for those organs. Layers 2 + 3 remain
editable.

To make Layer 1 editable for an organ in read-only mode, hand-edit
the Pydantic source until the inline `BaseModel` is removed or the
organ shape uses only spec markers + bare fields, then reload.

## New-organ wizard

The sidebar's **➕ New organ** expander wraps
`schemas.create_organ.create_organ()`. It scaffolds the three layer
files from the templates in `schemas/_templates/`, smoke-imports them
to verify registration in `CASE_MODELS`, regenerates the JSON
snapshot, and drops you into the editor on the newly-created organ.

The organ key must match `^[a-z][a-z0-9_]*$` and must not already be
registered.

## Save flow

1. **Validate** state (block save on errors; warnings don't block).
2. **Race-check** on-disk mtimes against load-time mtimes; abort if
   any file was modified externally.
3. **Render** Layer 1/2/3 to sibling temp files.
4. **Format** Layer 1/2 with `ruff format` (best-effort; silent
   if ruff isn't installed).
5. **Smoke-import** the rendered files in a subprocess so a syntax
   error or unregistrable schema can't corrupt the live tree.
6. **Atomic replace** each layer file via `os.replace`.
7. **Regenerate** `data/<organ>.json` via `python -m
   digital_registrar.schemas.generate`.

If any step fails, the on-disk tree is left untouched.

## Identity round-trip guarantee

For all 9 currently-editable organs, load → save without edits
produces byte-identical `model.model_json_schema()` output (verified
in development). Lung is the only organ currently flagged read-only
because of its `LungHistologicalPattern` inline `BaseModel`.

## Limitations

- **Layer 3 comments lost**: the TOML round-trip drops prose comments
  (the `# Molecular biomarkers ...` / `# Lymph node stations ...` style
  headers). The table-of-tables structure and per-line alignment is
  preserved.
- **Single user assumption**: two browser tabs editing the same organ
  will race. The save flow detects this via mtime drift and aborts.
- **AppTest harness recursion**: `streamlit.testing.v1.AppTest` hits a
  recursion limit when rendering all 5 tabs (a known Streamlit
  reflection quirk with wide `st.data_editor` payloads). The app
  itself runs cleanly under `streamlit run`; only the headless test
  framework is affected. Smoke-test by opening the URL in a browser.
