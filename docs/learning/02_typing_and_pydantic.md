# 02 — Type hints, type subscription, and pydantic Field

> Goal: stop guessing what `BreastProcedure | None`, `Literal[...]`, `Field(None, ...)`, and `ClassVar` mean. By the end you should be able to read [src/digital_registrar/schemas/pydantic/cervix.py](../../src/digital_registrar/schemas/pydantic/cervix.py) line by line.

This doc has two halves:

1. **Python type hints** — annotations, generics, `Literal`, `Optional`, `Annotated`, `ClassVar`, `TypedDict`.
2. **Pydantic in depth** — `BaseModel`, `Field`, validators, `model_validate` / `model_dump` / `model_json_schema`, and the StrEnum↔Literal round-trip that ties Layer 1 to the DSPy emission.

If you haven't read [01 — class machinery](01_class_machinery.md), skim §0–§4 of it first; it explains classes-are-values which underlies pydantic's whole design.

---

## 0. ELI5: type hints are labels

In plain Python:

```python
def add(a, b):
    return a + b
```

Python doesn't care what `a` and `b` are. You can call `add(1, 2)`, `add("hi", "there")`, or `add([1], [2])`. The function runs until it can't.

Type hints add labels that say what you *intend*:

```python
def add(a: int, b: int) -> int:
    return a + b
```

Python **still doesn't enforce them**. Calling `add("hi", "there")` works the same as before. The hints are read by:
- Type checkers (mypy, pyright) at development time.
- IDEs for autocomplete.
- Libraries like **pydantic** at runtime to perform actual validation and coercion.

That's the key: type hints alone are just decoration. Pydantic *runs* them.

---

## 1. Type subscription — `list[X]`, `dict[K, V]`, etc.

When a type takes parameters, you "subscribe" it with square brackets:

```python
names: list[str]                 # list whose elements are str
scores: dict[str, int]           # dict from str keys to int values
pair: tuple[int, str]            # tuple of exactly (int, str)
```

These are also called **parametrized generics**. `list` is a generic; `list[str]` is its parametrized form.

Real examples from `drr-next`:

[src/digital_registrar/schemas/extraction/__init__.py:43-62](../../src/digital_registrar/schemas/extraction/__init__.py#L43-L62):
```python
class FieldMeta(TypedDict):
    desc: str
    group: NotRequired[str]
    examples: NotRequired[list[object]]
    value_hints: NotRequired[dict[str, list[str]]]
```

And [src/digital_registrar/chunking/protocols.py:34-36](../../src/digital_registrar/chunking/protocols.py#L34-L36):
```python
span: tuple[int, int]
labels: frozenset[str] = field(default_factory=frozenset)
meta: dict[str, Any] = field(default_factory=dict)
```

`frozenset[str]` works the same way — it's a `frozenset` whose elements are `str`.

**Python version note**: `list[str]`, `dict[str, int]`, etc. work natively from Python 3.9+. Older code uses `typing.List`, `typing.Dict` (capitalized). drr-next uses the modern lowercase forms throughout.

---

## 2. `from __future__ import annotations` — type hints become strings

You'll see this line at the top of nearly every file in `drr-next`:

```python
from __future__ import annotations
```

It's a "future feature" import — it opts the file into a Python behavior change defined by **PEP 563**. Specifically: **all type annotations in this file become strings at runtime, not evaluated types.**

### Without the import

```python
class Chunk:
    text: str
    span: tuple[int, int]
```

When Python builds this class, it actually evaluates `tuple[int, int]` and stores the resulting generic alias on the class. Two costs:
- It happens at import time (small but real).
- Every name referenced inside the annotation must already exist when the annotation is read.

### With the import

```python
from __future__ import annotations

class Chunk:
    text: str
    span: tuple[int, int]
```

Python stores the annotations as **literal strings**: `"str"`, `"tuple[int, int]"`. They are not evaluated until something — pydantic, mypy, your own introspection code — asks for them. The standard way to evaluate them on demand is `typing.get_type_hints(cls)`.

### Three concrete benefits in drr-next

**1. Self-referencing methods work without quotes.**

[src/digital_registrar/chunking/protocols.py:38-43](../../src/digital_registrar/chunking/protocols.py#L38-L43):

```python
@dataclass(frozen=True)
class Chunk:
    ...
    def with_labels(self, labels: frozenset[str]) -> Chunk:
        return Chunk(...)
```

Without the `__future__` import, that `-> Chunk` would fail at class-definition time because `Chunk` doesn't exist yet when the method is being built. With the import, the annotation is the literal string `"Chunk"` until somebody resolves it — and by then `Chunk` does exist.

**2. Forward references in general.**

If two classes refer to each other:

```python
from __future__ import annotations

class A:
    sibling: B   # B isn't defined yet — no problem, this is a string

class B:
    sibling: A   # A is already defined, but the string form is consistent
```

Without the import, you'd have to quote one of the names manually: `sibling: "B"`. With the import, *everything* is a string, so the quoting is unnecessary.

**3. Cheaper imports.**

Building a generic like `dict[str, FieldMeta]` at import time costs a tiny amount of work. Across a project with hundreds of annotated classes, that adds up. Deferring evaluation keeps startup snappy.

### `TYPE_CHECKING` — imports that exist only for the type checker

A related pattern. drr-next uses it sparingly, but you'll see it in many Python codebases:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heavy_module import HeavyClass

def process(x: HeavyClass) -> None:
    ...
```

`TYPE_CHECKING` is a constant that is **`True` when a type checker (mypy, pyright) is analyzing the file** and **`False` at actual runtime**. The import inside `if TYPE_CHECKING:` only runs during type checking. At runtime, `HeavyClass` is never imported.

This combines with `from __future__ import annotations`: the annotation `x: HeavyClass` is just a string at runtime, so Python never tries to resolve `HeavyClass`. Mypy still sees the type because the conditional import is visible during type checking.

When to reach for it: **breaking import cycles** caused by type annotations, or **avoiding heavy runtime imports** for type-only references.

### How pydantic still gets real types

Pydantic depends on knowing the actual types of fields to validate. So if annotations are strings, how does pydantic make it work?

When you define a `BaseModel`, pydantic calls `typing.get_type_hints(cls)` internally. That function evaluates each annotation string against the module's namespace and returns the resolved types. As long as every name referenced in an annotation is importable at runtime (i.e. *not* hidden behind `TYPE_CHECKING`), pydantic resolves it cleanly.

**Gotcha**: if you put a type-only import behind `TYPE_CHECKING` and then annotate a pydantic field with that type, pydantic fails to resolve it and raises a `NameError` when the model is built. drr-next avoids this by keeping all annotation-referenced names in normal imports. `TYPE_CHECKING` is reserved for things that genuinely never need to exist at runtime (alias-only types, helpers used only in stubs).

If you genuinely need to defer resolution — e.g. forward-referencing a class that isn't defined yet at model-construction time — call `Model.model_rebuild()` after both classes exist. Pydantic v2 handles this gracefully. See §15 (Gotchas) for the worked case.

### TL;DR

`from __future__ import annotations` = "treat all type annotations as strings; evaluate them lazily." Zero cost, enables forward references, composes with `TYPE_CHECKING`. Pydantic still works because it explicitly resolves the strings via `typing.get_type_hints(...)` when building models.

---

## 3. Union types and `Optional`

A union is "either this or that". Two equivalent ways to write it:

```python
from typing import Union, Optional

x: Union[int, str]    # old style
x: int | str          # modern (Python 3.10+)

y: Optional[int]      # old style
y: int | None         # modern — exactly equivalent
```

`drr-next` uses the modern `|` form everywhere. From [cervix.py](../../src/digital_registrar/schemas/pydantic/cervix.py):

```python
procedure: CervixProcedure | None = None
surgical_technique: CervixSurgicalTechnique | None = None
cancer_primary_site: CervixPrimarySite | None = None
histology: CervixHistology | None = None
grade: Literal[1, 2, 3] | None = None
tumor_size: int | None = None
```

**`Optional[X]` is purely syntactic sugar for `X | None`.** It does **not** mean "may be omitted" — it only means "the value can be None". Whether the field is required at construction is a separate question (see §9 below).

### `None` vs `NOT_STATED` in drr-next

You'll notice every closed-vocabulary enum in [src/digital_registrar/schemas/pydantic/_enums.py](../../src/digital_registrar/schemas/pydantic/_enums.py) ends with `NOT_STATED`. And the field annotation is also `MyEnum | None`. Why both?

```python
class CervixProcedure(LowerStrEnum):
    RADICAL_HYSTERECTOMY = auto()
    TOTAL_HYSTERECTOMY_BSO = auto()
    ...
    NOT_STATED = auto()
```

Two reasons:
- **`None`** — the field is absent / null in the JSON. This is the legacy shape from before the migration, kept for JSON-schema compatibility.
- **`NOT_STATED`** — the LM explicitly says "the report doesn't say". This is a richer signal than null for downstream consumers.

The field annotation `CervixProcedure | None` accommodates both: the LM can emit `"not_stated"` (a real enum value) or `null`. Both are valid.

---

## 4. `Literal` — closed vocabularies as types

`Literal[a, b, c]` says "the value must be exactly one of these literals":

```python
from typing import Literal

grade: Literal[1, 2, 3]                          # only 1, 2, or 3
side: Literal["left", "right", "midline"]        # only these three strings
```

This shows up in [cervix.py:150](../../src/digital_registrar/schemas/pydantic/cervix.py#L150):

```python
grade: Literal[1, 2, 3] | None = None
```

And in the dynamically-built router signature in [factory.py:322-323](../../src/digital_registrar/signatures/factory.py#L322-L323):

```python
keys = _registry_keys(registry) + ["others"]
cancer_category_t = Literal[tuple(keys)] | None
```

Yes, `Literal[tuple(...)]` is valid — `Literal` accepts a tuple as a way to splat a list of allowed values. The router's `cancer_category` field accepts only the auto-discovered organ names plus `"others"`.

### The StrEnum → Literal bridge

`drr-next` has a deliberate two-form trick. The pydantic layer uses **StrEnums**:

```python
class CervixProcedure(LowerStrEnum):
    RADICAL_HYSTERECTOMY = auto()
    ...
```

But the DSPy-facing layer rewrites them as `Literal[<member values>]`:

```python
# from src/digital_registrar/signatures/_signature_annotation.py:33-42
def enum_to_literal(enum_cls: type[Enum]) -> Any:
    values = tuple(member.value for member in enum_cls)
    return Literal[values]
```

So `CervixProcedure` becomes `Literal["radical_hysterectomy", "total_hysterectomy_bso", ..., "not_stated"]` in the schema DSPy emits to the LM.

**Why?** DSPy's JSON-schema generator produces cleaner output for `Literal` than for `Enum` references. The LM sees a flat list of allowed string values instead of a `$ref` to an enum definition.

**The round-trip**: the LM outputs the string `"radical_hysterectomy"`, and `CervixCancerCase.model_validate({"procedure": "radical_hysterectomy", ...})` coerces it **back** into `CervixProcedure.RADICAL_HYSTERECTOMY`. Pydantic v2 handles this natively because `StrEnum` members compare equal to their string value. You get the schema-side clarity *and* the python-side enum convenience for free.

---

## 5. `Annotated` — attach metadata to a type

`Annotated[X, ...]` lets you bolt arbitrary metadata onto a type without changing what it *is*:

```python
from typing import Annotated

UserId = Annotated[int, "must be positive"]
```

To a type checker, `UserId` is still `int`. The string is metadata that a library (or your own code) can read at runtime.

Pydantic uses `Annotated` for richer constraints:

```python
from typing import Annotated
from pydantic import Field

Age = Annotated[int, Field(ge=0, le=150)]
```

**`drr-next` does not lean on `Annotated`.** The codebase keeps metadata in **side tables** (Layer 2 = `FIELD_META` dict) instead of attaching it to the type. This is a deliberate choice: domain experts edit `schemas/extraction/<organ>.py`, and the Python file isn't cluttered with `Annotated[...]` chains.

You'll see `Annotated` in the wild (pydantic docs, FastAPI, etc.) so it's worth recognizing. In `drr-next` you can mostly ignore it.

---

## 6. `ClassVar` — "this is not a field"

Pydantic treats every annotated attribute on a `BaseModel` as a field by default. To tell pydantic "this is a class-level constant, leave it alone", use `ClassVar`:

```python
from typing import ClassVar
from pydantic import BaseModel

class Foo(BaseModel):
    name: str                # field (per-instance)
    VERSION: ClassVar = "1.0"   # class constant, NOT a field
```

`drr-next` uses this in [cervix.py:157-164](../../src/digital_registrar/schemas/pydantic/cervix.py#L157-L164):

```python
@assemble_case_model()
class CervixCancerCase(BaseModel):
    """Canonical extracted case record for cervix cancer."""

    procedure: CervixProcedure | None = None
    ...

    _STAGING: ClassVar = StagingSpec(
        pt=CervixPT, pn=CervixPN, pm=PMCategory,
        ...
    )
```

`_STAGING` is **not** a field on the model. The `@assemble_case_model` decorator reads it from the class as a *spec* and uses it to inject the staging fields (`pt_category`, `pn_category`, `stage_group`, etc.) on the rebuilt model.

If you forget the `ClassVar`, pydantic will try to treat `_STAGING = StagingSpec(...)` as a field with a frozen-dataclass default value — and the model will have a `_staging` field on every instance, which is wrong.

---

## 7. `TypedDict` — typed dictionary structure

A `TypedDict` describes the shape of a `dict` (keys + value types) without making it a class. It's still a `dict` at runtime.

[src/digital_registrar/schemas/extraction/__init__.py:43-62](../../src/digital_registrar/schemas/extraction/__init__.py#L43-L62):

```python
from typing import NotRequired, TypedDict

class FieldMeta(TypedDict):
    desc: str
    group: NotRequired[str]
    examples: NotRequired[list[object]]
    value_hints: NotRequired[dict[str, list[str]]]
```

What this says:
- A `FieldMeta` is a dict.
- `desc` is required and must be a `str`.
- `group`, `examples`, `value_hints` are *optional* keys (the `NotRequired` wrapper).

You use it as:

```python
m: FieldMeta = {"desc": "the tumor size in mm"}    # OK
m2: FieldMeta = {"desc": "x", "group": "nonnested"}  # OK
m3: FieldMeta = {"group": "x"}    # type error — missing required `desc`
```

**When to reach for `TypedDict`** vs `BaseModel`: TypedDict gives you type-checker support without the cost (validation, JSON schema) of pydantic. `drr-next` uses TypedDict for *metadata that's already trusted* (you wrote it yourself). It uses pydantic for *outputs that need validation* (LM responses, JSON data from disk).

---

## 8. Pydantic `BaseModel` — the basics

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str | None = None

p = Person(name="Ada", age=30)             # validates: OK
p = Person(name="Ada", age="thirty")       # ValidationError: age not int
p = Person.model_validate({"name": "Ada", "age": 30})   # same as above
```

What you get from inheriting `BaseModel`:
- An `__init__` that validates fields at construction.
- `model_validate(data: dict | obj)` — the canonical "build from external data" constructor.
- `model_dump()` — convert back to a plain dict.
- `model_dump_json()` — convert to a JSON string.
- `model_json_schema()` — emit a JSON Schema describing the model.
- `model_fields` — class-level dict of `FieldInfo` objects for introspection.

The first four are the day-to-day API. The fifth (`model_fields`) is what drr-next's introspection code uses (see §12 below).

---

## 9. `Field(...)` — required, defaults, descriptions

Pydantic provides a `Field` helper that lets you attach metadata to a field:

```python
from pydantic import BaseModel, Field

class Foo(BaseModel):
    name: str = Field(..., description="The full name")
    age: int = Field(0, ge=0, le=150)
    email: str | None = Field(None, description="Optional contact email")
```

Reading each line:
- `Field(...)` — `...` (Ellipsis) is the **required sentinel**. `name` must be provided.
- `Field(0, ge=0, le=150)` — default `0`; constraint: `0 <= age <= 150`.
- `Field(None, description=...)` — default `None`; with documentation.

You can drop `Field(...)` entirely and just use a regular default:

```python
class Foo(BaseModel):
    name: str               # required — no default at all
    age: int = 0            # required type, default 0
    email: str | None = None  # optional value, default None
```

The two notations coexist. `Field(...)` is needed when you want to **attach metadata** (description, constraints, alias).

Real example in [src/digital_registrar/schemas/pydantic/_common.py:24-46](../../src/digital_registrar/schemas/pydantic/_common.py#L24-L46):

```python
class IsCancerCase(BaseModel):
    """Top-level routing decision — mirrors `is_cancer` output fields."""

    cancer_excision_report: bool = Field(
        ...,
        description=(
            "Whether this report documents a PRIMARY cancer excision eligible for "
            "registry..."
        ),
    )
    cancer_category: Literal[_ORGAN_KEYS] | None = Field(
        None,
        description=(
            "Which organ the primary cancer arises from..."
        ),
    )
    cancer_category_others_description: str | None = Field(
        None,
        description="Free-text organ name when cancer_category == 'others'.",
    )
```

`cancer_excision_report` is **required** (`Field(...)`); `cancer_category` and `cancer_category_others_description` are **optional with default `None`**.

### Common `Field` parameters

| Parameter | What it does |
|---|---|
| `...` (positional default) | Mark as required |
| `default=value` / positional value | Set default |
| `default_factory=callable` | Call a function to produce the default (use for `dict`, `list`, etc.) |
| `description="..."` | Human-readable description (appears in JSON Schema) |
| `alias="other_name"` | Accept input under a different key |
| `ge=, le=, gt=, lt=` | Numeric bounds |
| `min_length=, max_length=` | String/list bounds |
| `pattern="..."` | Regex constraint |

### Why drr-next keeps descriptions OUT of Layer 1

Look at [cervix.py](../../src/digital_registrar/schemas/pydantic/cervix.py) and you'll see *no* `Field(description=...)` calls anywhere. Every field is just `name: type = None`.

That's deliberate. Descriptions live in [schemas/extraction/cervix.py](../../src/digital_registrar/schemas/extraction/cervix.py) as a `FIELD_META` dict:

```python
FIELD_META: dict[str, FieldMeta] = {
    "procedure":          {"group": "nonnested", "desc": "identify which surgery procedure was used. e.g. radical_hysterectomy"},
    "surgical_technique": {"group": "nonnested", "desc": "identify how the surgery was taken. e.g. laparoscopic"},
    ...
}
```

The descriptions get **injected** into the JSON Schema at generation time (and into DSPy signatures at build time). The pydantic class stays minimal so domain experts can read the shape at a glance, without prose getting in the way.

---

## 10. Validators — custom field rules

Pydantic v2 has three flavors of validator:

```python
from pydantic import BaseModel, field_validator, model_validator, computed_field

class Account(BaseModel):
    balance: int
    overdraft_limit: int

    @field_validator("balance")
    @classmethod
    def balance_must_be_int(cls, v):
        # Runs for each input to the `balance` field.
        if not isinstance(v, int):
            raise ValueError("balance must be int")
        return v

    @model_validator(mode="after")
    def check_overdraft(self):
        # Runs after all field validation, against the constructed instance.
        if self.balance < -self.overdraft_limit:
            raise ValueError("over overdraft limit")
        return self

    @computed_field
    @property
    def available(self) -> int:
        # A "fake" field, computed on demand, included in model_dump().
        return self.balance + self.overdraft_limit
```

What each does:
- **`@field_validator(field_name)`** — runs during field validation. Use to transform / reject values. Must be a `@classmethod`.
- **`@model_validator(mode="after")`** — runs after all fields are set. Use for cross-field rules. Returns `self`.
- **`@model_validator(mode="before")`** — runs *before* field validation, with the raw input dict. Use for pre-processing.
- **`@computed_field`** — exposes a `@property` as a virtual field that shows up in `model_dump()` and `model_json_schema()`.

`drr-next` doesn't lean heavily on custom validators — most validation is structural (types, enums) and pydantic handles it out of the box. But you'll recognize these patterns in other codebases.

---

## 11. `model_config` / `ConfigDict` — model-level settings

To configure behavior of an entire model:

```python
from pydantic import BaseModel, ConfigDict

class Foo(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,    # strip strings on input
        extra="forbid",               # reject unknown keys
        use_enum_values=True,         # store the .value, not the enum member
        frozen=True,                  # make instances immutable
    )

    name: str
```

Common knobs:

| Setting | Effect |
|---|---|
| `extra="forbid"` | Reject unknown keys (default: ignore) |
| `extra="allow"` | Keep unknown keys |
| `str_strip_whitespace=True` | `.strip()` strings on input |
| `frozen=True` | Make instances immutable (like `@dataclass(frozen=True)`) |
| `populate_by_name=True` | Accept input under field name OR alias |
| `use_enum_values=True` | Store `.value` rather than enum member |

`drr-next` mostly uses defaults. The implicit behaviors that matter most are:
- Pydantic v2 **coerces** strings to enum members when the field is annotated as an enum (the StrEnum round-trip).
- `model_validate(data)` is the right way to build a model from a dict — `Foo(**data)` works but is less flexible.

---

## 12. Introspection: `model_fields` and `.annotation`

Every `BaseModel` subclass exposes a `model_fields` class attribute. It's a dict from field name to `FieldInfo`. Each `FieldInfo` has `.annotation` (the type), `.default`, `.description`, etc.

`drr-next` walks this in [src/digital_registrar/schemas/generate.py](../../src/digital_registrar/schemas/generate.py) to inject descriptions into the JSON schema. And in [src/digital_registrar/signatures/_signature_annotation.py:113-124](../../src/digital_registrar/signatures/_signature_annotation.py#L113-L124):

```python
def _model_has_enums(cls: type[BaseModel]) -> bool:
    """True if any field annotation contains an Enum subclass anywhere in its tree."""

    def visit(tp: Any) -> bool:
        if _is_strenum(tp):
            return True
        for arg in _t.get_args(tp):
            if visit(arg):
                return True
        return False

    return any(visit(fi.annotation) for fi in cls.model_fields.values())
```

What's happening:
- `cls.model_fields.values()` — iterate over all `FieldInfo` objects.
- `fi.annotation` — get the type annotation of each field.
- `typing.get_args(tp)` — drill into generic parameters (e.g. `list[X]` → `(X,)`).

Combined with `typing.get_origin(tp)` (which returns `list`, `dict`, `tuple`, `Union`, etc. for parametrized types), this is the standard way to **walk a type tree**. The `signature_annotation` function in the same file uses the same pattern to rewrite annotations:

```python
# from _signature_annotation.py:78-101
origin = _t.get_origin(annotation)
args = _t.get_args(annotation)

if origin is None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        ...

if origin is list:
    if args:
        return list[signature_annotation(args[0], ...)]
    ...

if origin in (_t.Union, _pytypes.UnionType):
    new_args = tuple(signature_annotation(a, ...) for a in args)
    result = new_args[0]
    for arg in new_args[1:]:
        result = result | arg
    return result
```

This is general-purpose Python typing introspection. The same techniques work outside pydantic — for any annotated function or class.

---

## 13. The three-layer schema and how Field is split across them

A quick refresher on the architecture (you'll see it everywhere in `drr-next`):

**Layer 1 — Pydantic shape** ([schemas/pydantic/](../../src/digital_registrar/schemas/pydantic/)): types, enums, defaults. *No descriptions.*

```python
class CervixCancerCase(BaseModel):
    procedure: CervixProcedure | None = None
    ...
```

**Layer 2 — Extraction metadata** ([schemas/extraction/](../../src/digital_registrar/schemas/extraction/)): descriptions, group tags, per-group instructions. Lives in a `FIELD_META` dict.

```python
FIELD_META: dict[str, FieldMeta] = {
    "procedure": {"group": "nonnested", "desc": "identify which surgery procedure..."},
    ...
}
```

**Layer 3 — Aliases** ([schemas/aliases/](../../src/digital_registrar/schemas/aliases/)): TOML data files with surface-form synonyms ("ALK", "alk", "ALK D5F3" → `alk`).

The three layers get **merged** at runtime:
1. `_build_organ_meta` in [extraction/__init__.py](../../src/digital_registrar/schemas/extraction/__init__.py) merges Layer 2 (descriptions) with Layer 3 (aliases) into a single `OrganExtractionMeta`.
2. `generate.py` reads Layer 1's pydantic model, calls `.model_json_schema()`, then walks Layer 2 to **inject** descriptions into the schema by field path. Nested sub-fields use dotted keys (`"biomarkers.percentage"`).
3. The signature factory consumes the merged Layer 2 to build DSPy signatures, also walking Layer 1's annotations to handle StrEnum→Literal.

`Field(description=...)` *exists* in pydantic, but `drr-next` chose **not** to use it in Layer 1. The reason: domain experts shouldn't have to look at Python files to update LM-facing prose, and Python files shouldn't be polluted with prose that changes often.

---

## 14. Putting it together: read [cervix.py](../../src/digital_registrar/schemas/pydantic/cervix.py) line by line

You should now be able to read this without confusion:

```python
class CervixProcedure(LowerStrEnum):           # closed vocabulary (§4), StrEnum (§4)
    RADICAL_HYSTERECTOMY = auto()              # value = "radical_hysterectomy" (per LowerStrEnum)
    ...
    NOT_STATED = auto()                         # explicit sentinel (§3)


@assemble_case_model()                          # class decorator that REPLACES the class (Doc 01 §9)
class CervixCancerCase(BaseModel):              # pydantic model
    """Canonical extracted case record for cervix cancer."""

    procedure: CervixProcedure | None = None    # union type (§3), no description (Layer 1 stays clean)
    surgical_technique: CervixSurgicalTechnique | None = None
    ...
    grade: Literal[1, 2, 3] | None = None       # closed set of values (§4)
    tumor_size: int | None = None
    distant_metastasis: bool | None = None
    treatment_effect: str | None = None

    _STAGING: ClassVar = StagingSpec(...)       # NOT a field (§6); read by decorator

    margins: CervixMarginSite | None = MarginSpec()              # spec marker (Doc 01 §13)
    regional_lymph_node: CervixLNCategory | None = LNSpec(...)   # spec marker
    extranodal_extension: bool | None = None
    maximal_ln_size: int | None = None
```

Every line is one of: a type-hinted field with a default, a spec marker that the decorator will expand, or a ClassVar the decorator reads.

---

## 15. Gotchas

### `Optional[X]` ≠ "may be omitted"

```python
class Foo(BaseModel):
    x: int | None        # value can be None
    y: int | None = None # value can be None AND default is None

Foo()                    # ValidationError on x (required)
Foo(x=None)              # OK
```

`X | None` only says *the value can be `None`*. To make a field skippable at construction, you also need a default: `= None` or `Field(default=None)`.

### Bare `None` vs `Field(None)` as default

```python
x: int | None = None        # default None
y: int | None = Field(None)  # also default None
z: int | None = Field(default=None, description="...") # default None + metadata
```

All three work. Prefer the bare `= None` form unless you need metadata.

### `Field(...)` vs `Field()`

```python
x: int = Field(...)    # REQUIRED (Ellipsis is the sentinel)
y: int = Field()       # ALSO REQUIRED (no default given)
z: int = Field(0)      # default 0
```

It's the *absence* of a default that makes a field required. `...` is the explicit way to say "yes, I really mean required".

### Class-syntax vs `create_model` field-spec shapes

```python
# Class syntax — pydantic infers from class body
class Foo(BaseModel):
    name: str
    age: int | None = None

# Runtime — explicit (annotation, default_or_Field) tuple
Foo = create_model(
    "Foo",
    __base__=BaseModel,
    name=(str, ...),               # required
    age=(int | None, Field(None)), # optional
)
```

The runtime shape uses a **tuple of (annotation, default)**. drr-next's nested-type factories ([_common_factories.py](../../src/digital_registrar/schemas/pydantic/_common_factories.py)) use the runtime form because they need to parameterize the category type.

### `model_validate` vs construction syntax

```python
# These two are equivalent for clean input:
Foo(name="x", age=1)
Foo.model_validate({"name": "x", "age": 1})

# But model_validate also handles edge cases:
Foo.model_validate(other_pydantic_instance)   # OK, copies fields
Foo.model_validate(dataclass_instance)        # OK if shape matches
```

Prefer `model_validate(data)` when you have a dict from outside (JSON, LM output, etc.). Use `Foo(...)` when constructing inline.

### Forward references and circular imports

If `A.field: B` and `B.field: A`, you have a circular reference. The two tools for it:

1. **`from __future__ import annotations`** (see §2) — turns every annotation into a string, so neither class needs to exist when the other is defined.

2. **Explicit quoting** — without the future import, quote the type name:
   ```python
   class A(BaseModel):
       b: "B"

   class B(BaseModel):
       a: "A"

   A.model_rebuild()    # resolves forward refs after both classes exist
   ```

drr-next uses the `from __future__ import annotations` form everywhere, so it mostly doesn't need manual quoting. If you ever hit a "class not fully defined" error, `Model.model_rebuild()` is the escape hatch — call it after both classes are constructed.

### `model_fields` vs `__fields__` vs `dict(...)`

Pydantic v1 had `__fields__`. Pydantic v2 has `model_fields`. They are different shapes — don't mix examples from v1 docs with v2 code. drr-next is on pydantic v2. Always use `model_fields`.

---

## 16. Cheat sheet

| Thing | Meaning |
|---|---|
| `from __future__ import annotations` | Make all annotations in this file string-deferred (PEP 563) |
| `if TYPE_CHECKING:` | Imports that exist only for the type checker, not at runtime |
| `X | None` | Value can be `None` (same as `Optional[X]`) |
| `list[X]`, `dict[K, V]`, `tuple[A, B]` | Parametrized generics |
| `Literal[a, b, c]` | Value must be exactly one of these |
| `Annotated[X, meta]` | Type `X` plus metadata (rarely used in drr-next) |
| `ClassVar` | Class-level constant, NOT a pydantic field |
| `TypedDict` + `NotRequired` | Shape of a dict, for type-check only |
| `BaseModel` | Pydantic model base class |
| `Field(...)` | Required, with metadata |
| `Field(default, description=...)` | Optional, with default and description |
| `default_factory=callable` | Per-instance mutable default |
| `model_validate(data)` | The right way to build from external data |
| `model_dump()` | Convert back to plain dict |
| `model_json_schema()` | Emit JSON Schema |
| `model_fields` | Introspect fields (`dict[str, FieldInfo]`) |
| `@field_validator`, `@model_validator`, `@computed_field` | Custom rules |
| `model_config = ConfigDict(...)` | Model-level settings |
| `typing.get_origin(tp)` / `typing.get_args(tp)` | Walk a type tree |

You're now equipped to read every pydantic file in [schemas/](../../src/digital_registrar/schemas/) without guessing what anything means. Move on to [04 — DSPy in depth](04_dspy_in_depth.md) to see how these types become LM prompts.
