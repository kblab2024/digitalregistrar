# `_legacy/` — preserved-but-inactive modules

Files here are kept for historical reference. They are **not** on the active
import path used by `runner.py` or any console script, **not** exercised in CI,
and **not** to be edited as part of normal development.

## What lives here

| File | Replaced by | Last active |
|---|---|---|
| `pipeline_dspy_strict.py` | `pipeline_factory.py` | pre-v2 schema refactor (commit `259cf27`) |
| `pipeline_structured.py`  | `pipeline_factory.py` | pre-v2 schema refactor (commit `259cf27`) |

`pipeline_dspy_strict.py` imports `load_decoding_kwargs` from `pipeline_structured.py`.
The pair is self-contained — nothing outside `_legacy/` imports either of them.

## When to look here

Only when reproducing a result that explicitly references the old pipeline
variants in a paper or experiment log. For anything else, the current
extraction pipeline is `pipeline.py` + `pipeline_factory.py`, both reached
through `runner.py`. See `docs/architecture/pipeline.md`.

## When to delete

Drop the whole directory once no external citation or experiment branch
depends on it. `git log -- src/digital_registrar/_legacy/` will
recover anything needed.
