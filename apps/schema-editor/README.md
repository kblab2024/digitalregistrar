# digital-registrar-schema-editor

Streamlit editor for the Digital Registrar three-layer per-organ schema (Pydantic + extraction metadata + aliases).

## Install

```bash
pip install digital-registrar-schema-editor
registrar-schema-gui
```

Default port: 8501.

## What it does

Edits the schema files inside the installed `digital_registrar.schemas` package:

- **Layer 1 (Pydantic)**: field definitions and enums.
- **Layer 2 (Extraction)**: field descriptions and group assignments for decomposed extraction.
- **Layer 3 (Aliases)**: TOML alias maps for terminology.

Generates JSON schemas under `digital_registrar/schemas/data/` on save. Use `registrar-schemas --check` to verify schema concordance.
