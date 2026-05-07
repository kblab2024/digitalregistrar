# Canonical schema layer

Three representations describe the same fields:

1. **DSPy signatures** in `models/<organ>.py` — drive LM calls; carry the prompts.
2. **Canonical Pydantic case-models** in `schemas/pydantic/<organ>.py` — the source of truth for tooling.
3. **JSON schemas** in `schemas/data/<organ>.json` — generated artifacts consumed by the annotation UI and the raw-JSON ablation runner.

Concordance is enforced by [`tests/test_schema_concordance.py`](../tests/test_schema_concordance.py) and by `registrar-schemas --check` in CI.

## Public API

```python
from digital_registrar_research.schemas import (
    list_organs,            # ['breast', 'cervix', 'colorectal', 'esophagus', 'liver',
                            #  'lung', 'pancreas', 'prostate', 'stomach', 'thyroid']
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
from digital_registrar_research.schemas.pydantic import LungCancerCase, BreastCancerCase
```

## How the case-models are built

Each `<organ>.py` is a one-liner:

```python
from ._builder import build_case_model

LungCancerCase = build_case_model("lung")
```

`build_case_model(organ)` introspects the per-subsection DSPy signatures listed in `models.modellist.organmodels[organ]` (which are themselves Pydantic models) and composes one flat `<Organ>CancerCase` whose fields are the union of every signature's `OutputField`. First-wins on duplicate names — matches `CancerPipeline.forward`'s `cancer_data.update(...)` ordering.

This means **the Pydantic case-model is concordant-by-construction** with the DSPy signatures. The concordance test still pins the contract so any future change to either side surfaces loudly.

## Regenerating JSON schemas

When you change a DSPy signature (add a field, change a Literal vocabulary):

```bash
registrar-schemas              # rewrites src/.../schemas/data/*.json
registrar-schemas --check      # CI invocation; exit 1 if anything drifted
```

CI fails any PR that changes a signature without regenerating the JSON.

## Naming-drift note: colorectal vs colon

`models/colon.py` defines `ColonCancer*` classes for historical reasons, but the public-facing organ name is `"colorectal"` everywhere else (the `cancer_category` Literal in `is_cancer`, the `modellist.organmodels` key, the JSON schema filename, the annotation UI dropdown). The canonical Pydantic class is `ColorectalCancerCase`; it imports the underlying `Colon*` types internally. The concordance test pins this mapping so the inconsistency can't propagate further.

## Bladder note

`models/bladder.py` exists with DSPy signatures, and `schemas/data/bladder.json` ships in the package, but bladder is **not** in `organmodels` — i.e. the runtime extraction pipeline doesn't route to it. The annotation UI knows about bladder via its own `CANCER_TO_FILE` map, so doctors can hand-annotate bladder cases. To promote bladder to a fully-supported organ:

1. Add `"bladder": [...subsection class names...]` to `models/modellist.py`.
2. Add `"bladder"` to the `cancer_category` Literal in `models/common.py`.
3. Create `schemas/pydantic/bladder.py` (one-liner) and add to the registry in `schemas/pydantic/__init__.py`.
4. Run `registrar-schemas` and `pytest`.
