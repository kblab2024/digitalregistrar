# `schemas/` — three-layer cancer-schema architecture

The per-organ cancer schema is split into **three layers of source-of-truth** plus
one generated artifact. Each layer has a single purpose; combined, they let
domain experts edit surface-form aliases without touching Python, and let the
canonical Pydantic models stay clean of LM-prompt verbiage.

```
schemas/
├── pydantic/          Layer 1 — shape (types, LowerStrEnums, defaults).
│   └── <organ>.py     The SINGLE SOURCE OF TRUTH for the organ's shape.
├── extraction/        Layer 2 — extraction metadata (LM-facing descriptions).
│   └── <organ>.py     FIELD_META + GROUP_INSTRUCTIONS for one organ.
├── aliases/           Layer 3 — surface-form aliases (data-only TOML).
│   └── <organ>.toml   Canonical → [surface forms]. Edit without touching Python.
├── data/              Generated JSON snapshots — DO NOT EDIT BY HAND.
│   └── <organ>.json   Produced by `registrar-schemas`; CI fails on drift.
├── _templates/        Templates used by `create_organ` (see below).
├── create_organ.py    Scaffolds a new organ across all three layers in one shot.
└── generate.py        Regenerates `data/*.json` from Layers 1 + 2.
```

## Cheat-sheet — where do I edit?

| Want to change…                                            | Run / Edit                                                                                              |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Add a brand-new organ**                                  | `python -m digital_registrar.schemas.create_organ <organ_key>`                                  |
| Add / remove / rename a field                              | `pydantic/<organ>.py`  +  `extraction/<organ>.py` (matching FIELD_META entry), then `registrar-schemas` |
| Add / remove / rename an enum value                        | `pydantic/<organ>.py`, then `registrar-schemas`                                                          |
| Customize an LM-facing description                         | `extraction/<organ>.py`                                                                                  |
| Add a surface form / synonym / antibody clone name         | `aliases/<organ>.toml` — no Python edit, no JSON regen                                                   |
| Add a per-group instruction                                | `extraction/<organ>.py::GROUP_INSTRUCTIONS`                                                              |
| Regenerate `data/<organ>.json`                             | `python -m digital_registrar.schemas.generate` (alias: `registrar-schemas`)                     |
| **NEVER edit**                                             | `data/*.json` (auto-generated; CI fails on drift)                                                        |

## Adding a new organ

The scaffolder writes the three per-organ files with a sensible default
(nonnested + staging + margins + lymph nodes; no biomarkers — those are
organ-specific, add per organ when needed). After scaffolding, edit the
generated files to match the real clinical schema.

```bash
python -m digital_registrar.schemas.create_organ skeletal
# Writes:
#   src/digital_registrar/schemas/pydantic/skeletal.py
#   src/digital_registrar/schemas/extraction/skeletal.py
#   src/digital_registrar/schemas/aliases/skeletal.toml
# Then regenerates data/skeletal.json via registrar-schemas.
```

Auto-discovery means **you don't edit any index files**: dropping
`pydantic/skeletal.py` into the directory is enough for the next process
import to find it. The cross-layer parity check in
`extraction/__init__.py` will fail loudly if you create a pydantic
case-model without a matching extraction module (or vice versa).

Flags:
- `--overwrite` — clobber existing per-organ files (default refuses).
- `--no-regen-json` — skip the `registrar-schemas` subprocess (faster).
- `--class-prefix=Cutaneous` — override the default `<Organ>CancerCase`
  class name (useful when the organ key doesn't match the pathology
  jargon: `skin` organ key but `CutaneousCancerCase` class).

## Programmatic use (future GUI)

```python
from digital_registrar.schemas.create_organ import create_organ
report = create_organ("skeletal", base_dir=Path("/tmp/sandbox"), regen_json=False)
print(report.files_written)
```

`base_dir` lets a GUI sandbox writes anywhere on disk. The internal
``_render_template`` step uses placeholder substitution (`__ORGAN_KEY__`,
`__CLASS_PREFIX__`); a future GUI can either replace the template files
directly or build a higher-level renderer.

## Auto-discovery contract

Both Layer 1 (`pydantic/`) and Layer 2 (`extraction/`) auto-discover
per-organ files at import time:

- Files starting with `_` are skipped (treated as private helpers).
- Every Layer-1 file must export `<Organ>CancerCase` (PascalCase of the
  snake_case file stem; e.g. `head_neck.py` → `HeadNeckCancerCase`).
- Every Layer-2 file must export `FIELD_META` and `GROUP_INSTRUCTIONS`.
- Layer-1 and Layer-2 organ keys must match exactly — a missing or stray
  file raises `RuntimeError` at import time.

Layer 3 (`aliases/`) is opt-in per organ: missing TOML files load as
empty dicts.

## See also

- `aliases/README.md` — detailed TOML format reference and runtime semantics.
- Generated banners — every `data/*.json` carries an `$comment` field
  pointing back at the three source layers, so a hand-edit is obviously
  wrong even before CI's `--check` mode catches the drift.
