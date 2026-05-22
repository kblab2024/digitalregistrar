# 01 — Python class machinery: decorators, `@dataclass`, dynamic generation

> Goal: when you next open [src/digital_registrar/signatures/factory.py](../../src/digital_registrar/signatures/factory.py) and see classes being built out of dicts, none of it should feel like magic.

This doc walks **upward** through the toolbox: simple decorators, then `@dataclass`, then `Protocol`, then decorators that *replace* classes, then finally building classes at runtime. Every step uses a real snippet from `drr-next`.

If you only remember one thing, remember this: **a class is just a value**. You can store it in a dict, pass it to a function, build new ones on the fly. Everything else in this doc is mechanics around that fact.

---

## 0. ELI5: classes are values

In Python, `class Foo: ...` does two things:
1. **Run the class body** to collect the names and methods.
2. **Bind the resulting class object** to the name `Foo`.

That class object is a normal Python value. You can do this:

```python
class Cat: pass
class Dog: pass

animals = {"cat": Cat, "dog": Dog}
cls = animals["cat"]     # cls is the Cat class itself
fluffy = cls()           # call the class to get an instance
```

`drr-next` does exactly this in [src/digital_registrar/schemas/pydantic/_registry.py](../../src/digital_registrar/schemas/pydantic/_registry.py): it walks a folder, imports each module, grabs each `<Organ>CancerCase` class, and stores them in a dict:

```python
out: dict[str, type[BaseModel]] = {}
for mod_info in pkgutil.iter_modules(package.__path__):
    ...
    cls = getattr(module, expected_cls, None)
    out[name] = cls
return dict(sorted(out.items()))
```

The type of `cls` is `type[BaseModel]` — "a class that is a subclass of BaseModel". That's a value you can pass around.

---

## 1. Methods aren't magic either

A method is a function stored on a class. When you write:

```python
class Foo:
    def bar(self):
        return 42
```

…Python builds a function object and assigns it to `Foo.bar`. Accessing `foo.bar` on an instance gives you a *bound method* — the same function, but with `foo` pre-filled as the first argument.

Decorators on methods just wrap or transform that function. None of `@staticmethod`, `@classmethod`, or `@property` are special syntax — they're regular functions you could write yourself.

---

## 2. `@staticmethod` — a function that happens to live on a class

A static method has no `self` and no `cls`. It's grouped with the class for tidiness, not because it needs instance state.

`drr-next` uses one in [src/digital_registrar/pipeline_factory.py:102-106](../../src/digital_registrar/pipeline_factory.py#L102-L106):

```python
class CancerPipelineV2(dspy.Module):
    @staticmethod
    def _normalize_report(report: str | list[str]) -> list[str]:
        if isinstance(report, list):
            return [p.strip() for p in report if isinstance(p, str) and p.strip()]
        return [p.strip() for p in str(report).split("\n\n") if p.strip()]
```

It doesn't read or write any instance attribute. It logically belongs to the pipeline because that's where it's called, but it's a pure function. `@staticmethod` is the right shape.

**When to reach for it**: the helper is pure, but you only ever call it via the class.

---

## 3. `@classmethod` — receives the class, not an instance

A classmethod's first argument is the class itself (conventionally `cls`). The canonical use case is **alternate constructors**:

```python
class Date:
    def __init__(self, year, month, day): ...

    @classmethod
    def from_iso_string(cls, s: str):
        y, m, d = map(int, s.split("-"))
        return cls(y, m, d)   # <-- uses cls, not Date, so subclasses work

birthday = Date.from_iso_string("2026-05-22")
```

In `drr-next`, you'll see them mostly in pydantic-related code (pydantic v2 uses classmethods under the hood for validators). For example, if you wrote a custom validator on a `BaseModel`:

```python
from pydantic import field_validator

class Foo(BaseModel):
    age: int

    @field_validator("age")
    @classmethod
    def positive(cls, v):
        if v < 0:
            raise ValueError("age must be >= 0")
        return v
```

The `@classmethod` is required because the validator runs against the class, not against any particular `Foo` instance.

**`@staticmethod` vs `@classmethod`** — both let you call via `Foo.thing()`. Use `@classmethod` when you need a reference to the class (e.g. to call `cls(...)` for a subclass-aware constructor). Use `@staticmethod` otherwise.

---

## 4. `@property` — methods that look like attributes

`@property` turns `foo.bar` (attribute access) into a method call. Useful when a value is *computed* but should *look* like an attribute:

```python
class Rectangle:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    @property
    def area(self):
        return self.w * self.h

rect = Rectangle(3, 4)
print(rect.area)   # 12 — no parentheses
```

`drr-next` doesn't lean on `@property` directly, but pydantic provides its own version: `@computed_field`. That's covered in Doc 2.

The thing to remember: **`@property` runs your method every time** the attribute is accessed. Use `@functools.cached_property` if the result is expensive and won't change.

---

## 5. `@functools.cache` — memoize a function's output

This decorator stores the function's return value the first time it's called with given arguments. Subsequent calls with the same arguments return the cached result instantly.

`drr-next` uses it in [src/digital_registrar/schemas/pydantic/_registry.py:37-42](../../src/digital_registrar/schemas/pydantic/_registry.py#L37-L42):

```python
@functools.cache
def discover_case_models() -> dict[str, type[BaseModel]]:
    """Return ``{organ_key: <Organ>CancerCase}`` by walking this package.

    Cached so the discovery cost is paid once even if both
    ``__init__.py`` and ``_common.py`` call it during package import.
    """
    ...
```

**Why caching matters here**: `discover_case_models()` does real filesystem and import work. Multiple modules during package init will call it. Without `@cache`, each call would re-walk the directory; with `@cache`, only the first call does any work.

**Gotcha**: `@functools.cache` keeps references to all arguments alive forever (the lifetime of the process). If you cache results keyed by large objects, that's a leak. For long-running processes, prefer `@functools.lru_cache(maxsize=128)` which evicts old entries.

The difference vs `@property`:
- `@property` = "call this method when accessing an attribute"
- `@functools.cache` = "remember the answer so you don't have to recompute"
- `@functools.cached_property` = both, but only on instances.

---

## 6. `@dataclass` — auto-generate `__init__`, `__repr__`, `__eq__`

A *dataclass* is a class that mainly holds values. Writing `__init__`, `__repr__`, and `__eq__` by hand for every value-class is tedious; `@dataclass` writes them for you.

`drr-next` uses it heavily. The most important example is [src/digital_registrar/chunking/protocols.py:23-43](../../src/digital_registrar/chunking/protocols.py#L23-L43):

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Chunk:
    """A labeled slice of a report."""

    id: str
    text: str
    span: tuple[int, int]
    labels: frozenset[str] = field(default_factory=frozenset)
    meta: dict[str, Any] = field(default_factory=dict)

    def with_labels(self, labels: frozenset[str]) -> Chunk:
        """Return a copy of this chunk with ``labels`` replaced."""
        return Chunk(
            id=self.id, text=self.text, span=self.span,
            labels=labels, meta=self.meta,
        )
```

Let's decode every piece of this decoration.

### `@dataclass`

Reads each line of the class body that looks like `name: annotation` (or `name: annotation = default`) and generates a matching `__init__`. After the decorator runs, you can do:

```python
c = Chunk(id="c0", text="Hello", span=(0, 5))
print(c)
# Chunk(id='c0', text='Hello', span=(0, 5), labels=frozenset(), meta={})
```

You get `__init__` (constructor with keyword args), `__repr__` (the readable print), and `__eq__` (two `Chunk`s with the same fields compare equal) for free.

### `frozen=True`

Makes instances **immutable**: setting `c.id = "c1"` raises `dataclasses.FrozenInstanceError`. Frozen dataclasses are also **hashable** by default, so they can be dict keys or `set` members.

Why frozen for `Chunk`? Two reasons:
1. Chunks flow through the pipeline as values. If something mutates a chunk midstream, debugging is a nightmare.
2. Frozen + `frozenset[str]` for `labels` means a chunk is fully immutable end-to-end.

The cost: you can't write `chunk.labels = {...}`. The `with_labels(...)` method exists to return a *new* `Chunk` with the labels swapped — a functional update.

### `field(default_factory=frozenset)`

You **cannot** use a mutable default directly:

```python
# BUG — every Chunk would share the same dict!
@dataclass
class Chunk:
    meta: dict = {}
```

`field(default_factory=...)` says: "when no value is passed, call this factory to produce a fresh default." Pass the function itself (`frozenset`, `dict`, `list`), not the call result (`frozenset()`, `{}`, `[]`).

### The other dataclass in the codebase

[src/digital_registrar/signatures/factory.py:71-83](../../src/digital_registrar/signatures/factory.py#L71-L83) uses the same pattern:

```python
@dataclass(frozen=True)
class ExtractionStep:
    """One LM call produced by the factory for an organ's extraction."""

    name: str
    signature: type[dspy.Signature]
    output_field_names: tuple[str, ...]
    group: str  # source group tag; "<monolithic>" when not decomposed.
```

Note `output_field_names: tuple[str, ...]` — a tuple, not a list, because tuples are hashable and immutable. This pairs naturally with `frozen=True`.

**When to reach for `@dataclass`**: anytime you'd otherwise write a small class with `__init__` and a few attributes. **Always prefer `frozen=True` unless you need to mutate.**

---

## 7. `Protocol` — structural typing

A `Protocol` says "anything that has these methods counts as one of these", without requiring inheritance.

`drr-next` defines two protocols in [src/digital_registrar/chunking/protocols.py:46-72](../../src/digital_registrar/chunking/protocols.py#L46-L72):

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, report: str) -> list[Chunk]: ...

@runtime_checkable
class Router(Protocol):
    name: str

    def route(self, chunks: list[Chunk], groups: dict[str, str]) -> list[Chunk]: ...
```

The `...` is literal Python syntax (Ellipsis) — Protocol method bodies are placeholders.

**The structural-typing payoff**: `RegexSectionChunker` does *not* inherit from `Chunker`. It just happens to have a `name` attribute and a `chunk(report) -> list[Chunk]` method. Type-checkers (mypy, pyright) accept it as a `Chunker` because the shape matches.

This is why the registry in [registry.py](../../src/digital_registrar/chunking/registry.py) can type its dict as `dict[str, type[Chunker]]` even though no chunker class subclasses `Chunker`:

```python
CHUNKER_REGISTRY: dict[str, type[Chunker]] = {}
```

**`@runtime_checkable`** — by default, `isinstance(x, Chunker)` raises `TypeError` because Protocols are abstract. This decorator makes `isinstance` check structural equivalence at runtime. drr-next doesn't actually call `isinstance` against these protocols, but the decoration is there in case future code wants to.

**Contrast with `abc.abstractmethod`** — the *nominal* version, where you must explicitly subclass an abstract base class. drr-next doesn't use ABCs; it uses Protocols. The lesson: **structural typing is usually friendlier; ABCs are heavy.**

---

## 8. Class decorators that *register* a class

Now we move from method-level decorators to **class decorators**. A class decorator takes a class, optionally modifies or replaces it, and returns a class.

The simplest pattern: register the class somewhere, return it unchanged. [src/digital_registrar/chunking/registry.py:27-37](../../src/digital_registrar/chunking/registry.py#L27-L37):

```python
CHUNKER_REGISTRY: dict[str, type[Chunker]] = {}

def register_chunker(name: str):
    """Decorator: register a Chunker class under ``name``."""

    def deco(cls: C) -> C:
        if name in CHUNKER_REGISTRY:
            raise ValueError(f"Chunker {name!r} already registered")
        cls.name = name
        CHUNKER_REGISTRY[name] = cls
        return cls

    return deco
```

Use it like this:

```python
@register_chunker("my_chunker")
class MyChunker:
    def chunk(self, report: str) -> list[Chunk]:
        ...
```

Two things to notice:

1. **`register_chunker("my_chunker")` is a *factory*** — it returns a decorator. The outer call has the name, the inner `deco` is the actual decorator.
2. **The class is returned unchanged**, but as a side effect it's now in `CHUNKER_REGISTRY`. The GUI's chunker dropdown reads from that dict, so dropping a new file into `chunking/` and decorating its class is all you need.

This is the **registry pattern**. It's how `drr-next` keeps plugins discoverable: define a class, decorate it, the rest of the system picks it up automatically.

---

## 9. Class decorators that *replace* a class

The next step up: a class decorator that **builds a different class** and returns *that*. This is what `@assemble_case_model` does.

The full implementation is in [src/digital_registrar/schemas/pydantic/_case_builder.py:276-358](../../src/digital_registrar/schemas/pydantic/_case_builder.py#L276-L358). The short story:

```python
def assemble_case_model(*, organ=None, type_prefix=None):

    def _decorate(cls: type) -> type:
        # 1. Read fields off the decorated class.
        new_specs: dict[str, tuple[Any, FieldInfo]] = {}

        for fname, fi in cls.model_fields.items():
            ann = fi.annotation
            if isinstance(fi, MarginSpec):
                # 2. Detect spec markers and expand them into list-of-nested.
                cat_type = _strip_optional(ann)
                margin_type = make_margin_type(f"{prefix}Margin", category_type=cat_type)
                new_specs[fname] = (list[margin_type] | None, Field(default=None))
            elif isinstance(fi, LNSpec):
                ...
            elif isinstance(fi, BiomarkerSpec):
                ...
            else:
                new_specs[fname] = (ann, fi)

        # 3. Inject staging fields if the class declares _STAGING.
        staging = getattr(cls, "_STAGING", None)
        if isinstance(staging, StagingSpec):
            for fname, ann, fi in staging.emit_fields():
                new_specs[fname] = (ann, fi)

        # 4. Build a brand-new BaseModel from the expanded specs.
        new_cls = create_model(
            cls.__name__,
            __base__=BaseModel,
            __doc__=cls.__doc__,
            **new_specs,
        )
        return new_cls

    return _decorate
```

What the decorated class looks like in source (from [cervix.py](../../src/digital_registrar/schemas/pydantic/cervix.py)):

```python
@assemble_case_model()
class CervixCancerCase(BaseModel):
    """Canonical extracted case record for cervix cancer."""

    procedure: CervixProcedure | None = None
    ...
    _STAGING: ClassVar = StagingSpec(
        pt=CervixPT, pn=CervixPN, pm=PMCategory,
        tnm_descriptor=TNMDescriptor,
        stage_groups={"stage_group": CervixStageGroup},
        lean=True,
    )
    margins: CervixMarginSite | None = MarginSpec()
    regional_lymph_node: CervixLNCategory | None = LNSpec(side=LymphNodeSide)
    ...
```

After the decorator runs, `CervixCancerCase` is **a different class** than what you typed. It now has:
- `tnm_descriptor`, `pt_category`, `pn_category`, `pm_category`, `stage_group`, `ajcc_version` — injected from `_STAGING`.
- `margins: list[CervixMargin] | None` — `MarginSpec()` got expanded into a list of a freshly-generated `CervixMargin` model.
- `regional_lymph_node: list[CervixLN] | None` — same for `LNSpec`.

The author writes ~15 lines; the user sees ~25+ fields on the validated model.

**The key idea**: decorators don't have to be additive. They can read the input, throw it away, and return something completely different. The point is to give you a *declarative* surface (spec markers) while emitting verbose, validated pydantic underneath.

---

## 10. Building classes from scratch — the raw way

Python's built-in `type()` can build a class from a name, base tuple, and attribute dict:

```python
# These two are equivalent:
class Foo:
    x = 1
    def hello(self): return "hi"

Foo2 = type("Foo2", (), {"x": 1, "hello": lambda self: "hi"})
print(Foo().hello(), Foo2().hello())   # "hi hi"
```

`drr-next` **does not** use raw `type()` anywhere. Modern Python wraps this in friendlier tools — `pydantic.create_model`, `dataclasses.make_dataclass`, `dspy.signatures.signature.make_signature` — which generate the right shape for their domain. But it's worth seeing the underlying mechanism: `type(name, bases, attrs)` is what every higher-level class builder ultimately boils down to.

---

## 11. The two class builders `drr-next` actually uses

### `pydantic.create_model`

Builds a `BaseModel` subclass at runtime. The field-spec shape is `{name: (annotation, default_or_FieldInfo)}`.

[src/digital_registrar/schemas/pydantic/_common_factories.py:25-43](../../src/digital_registrar/schemas/pydantic/_common_factories.py#L25-L43) — a clean factory for `<Organ>Margin` types:

```python
def make_margin_type(
    name: str,
    *,
    category_type: Any,
) -> type[BaseModel]:
    """Build a Margin BaseModel with the standard four fields."""
    fields: dict[str, Any] = {
        "margin_category": (category_type | None, Field(None)),
        "margin_involved": (bool, ...),
        "distance":        (int | None, Field(None)),
        "description":     (str | None, None),
    }
    return create_model(name, __base__=BaseModel, **fields)
```

What `(bool, ...)` means:
- First element = annotation (`bool`)
- Second element = default. `...` (Ellipsis) is pydantic's sentinel for **required**.
- `Field(None)` is a `FieldInfo` with `default=None` — optional.
- `(str | None, None)` is also optional with default `None` (bare `None` works too).

The result is a real `BaseModel` subclass — `model_validate`, `model_dump`, `model_json_schema` all work on it.

**Why dynamic here?** The Margin shape (category, involved, distance, description) is the same across all 10 organs. Only the category vocabulary differs. Hard-coding 10 nearly-identical Margin classes is repetition; `make_margin_type(name, category_type=X)` makes one parameterized factory.

The same module has `make_ln_type` for lymph nodes. And [_case_builder.py:217-235](../../src/digital_registrar/schemas/pydantic/_case_builder.py#L217-L235) has `make_biomarker_type`. They all use `create_model`.

### `dspy.signatures.signature.make_signature`

Builds a `dspy.Signature` subclass at runtime. The field-spec shape is `{name: (annotation, dspy.InputField | dspy.OutputField)}`.

[src/digital_registrar/signatures/factory.py:86-120](../../src/digital_registrar/signatures/factory.py#L86-L120):

```python
def build_signature(
    name: str,
    instructions: str,
    output_fields: dict[str, tuple[Any, str]],
    *,
    input_fields: dict[str, tuple[Any, str]] | None = None,
    custom_types: dict[str, type] | None = None,
) -> type[dspy.Signature]:
    fields_dict: dict[str, tuple[Any, Any]] = {}
    for fname, (ann, desc) in (input_fields or {}).items():
        fields_dict[fname] = (ann, dspy.InputField(desc=desc))
    for fname, (ann, desc) in output_fields.items():
        fields_dict[fname] = (ann, dspy.OutputField(desc=desc))

    from dspy.signatures.signature import make_signature

    sig = make_signature(
        fields_dict,
        instructions=instructions,
        signature_name=name,
        custom_types=dict(custom_types or {}),
    )
    return sig
```

Two things specific to `make_signature` (vs `create_model`):
- `instructions=` becomes the docstring of the generated Signature, which DSPy renders into the LM prompt.
- `custom_types=` is a `{"BreastMargin": <class>, ...}` dict telling DSPy how to resolve nested types in JSON schema. Without it, `list[BreastMargin]` would fail to serialize.

---

## 12. Why dynamic generation at all?

Three reasons specific to `drr-next`:

1. **Schema lives in pydantic, not signatures.** Domain experts edit `schemas/pydantic/<organ>.py` (shape) and `schemas/extraction/<organ>.py` (descriptions). The signatures must reflect that schema; the factory keeps them in sync without manual edits.

2. **There are too many fields to hand-write.** Breast has ~50 fields across 7 groups. Cervix has ~13 across 4. Hand-authored signatures for each (organ, group) combo would be ~30 files of mostly-boilerplate.

3. **Decomposition is a runtime decision.** "Per-group" (one signature per group) vs "monolithic" (one signature for everything) depends on the LM you're using. The factory's `_choose_decomposition` consults `dspy.settings.lm.supports_response_schema` and picks at construction time. Static signatures can't do that.

The cost is paid once per process. Memoization helps where the same class shows up under multiple prefixes — see `_REBUILD_CACHE` in [_signature_annotation.py:127](../../src/digital_registrar/signatures/_signature_annotation.py#L127):

```python
_REBUILD_CACHE: dict[tuple[int, str], type[BaseModel]] = {}
```

The key is `(id(cls), prefix)`. `id(cls)` is the memory address — fast, unique per class, but **not pickle-safe**. The cache must be process-local, which is fine for an import-time builder.

---

## 13. Gotchas

### Spec markers look like defaults but aren't

```python
margins: CervixMarginSite | None = MarginSpec()
```

This looks like "default value = MarginSpec instance". Actually, `MarginSpec` is a *subclass of pydantic's `FieldInfo`*. The decorator detects it by `isinstance(fi, MarginSpec)` and rewrites the field. You can't tell from the call site that anything special is going on — that's why the `@assemble_case_model` decorator name is on the class above, as a hint.

### `_REBUILD_CACHE` lifetime

Because the cache keys on `id(cls)`, and `id` can be reused once an object is garbage-collected, the cache is **only safe within a process where the classes stay alive**. drr-next's case-model classes are module-level globals — they live as long as the process. So this is fine. Don't replicate the pattern in code where the input class might be created on the fly.

### `custom_types` is required for nested BaseModels

If your DSPy signature has `output_fields={"margins": (list[BreastMargin], "...")}` and you don't pass `custom_types={"BreastMargin": BreastMargin}`, DSPy's JSON-schema serializer can't resolve `BreastMargin` and will fail. The factory walks every rewritten annotation in [factory.py:159-167](../../src/digital_registrar/signatures/factory.py#L159-L167):

```python
def _collect_custom_types_from_annotation(annotation, out):
    if isinstance(annotation, type) and issubclass(annotation, BaseModel) and annotation is not BaseModel:
        out.setdefault(annotation.__name__, annotation)
        return
    for arg in _ty.get_args(annotation):
        _collect_custom_types_from_annotation(arg, out)
```

…and passes the collected dict to `make_signature(..., custom_types=...)`.

### Frozen dataclass instances need functional updates

```python
chunk = Chunk(id="c0", text="x", span=(0, 1))
chunk.labels = frozenset({"margins"})   # FrozenInstanceError
```

Use `chunk.with_labels(...)` (the codebase pattern), or `dataclasses.replace(chunk, labels=...)` for the general case.

### `@functools.cache` keeps references forever

If you `@functools.cache` a function whose argument is, say, a giant pandas DataFrame, that DataFrame stays in memory until the process exits. For long-running services, prefer `@lru_cache(maxsize=N)` or write an explicit cache with eviction.

### Protocols aren't enforced at runtime by default

```python
class Chunker(Protocol):
    def chunk(self, report: str) -> list[Chunk]: ...

class Broken:
    pass

isinstance(Broken(), Chunker)   # TypeError unless @runtime_checkable
```

Even with `@runtime_checkable`, the check is **structural** — does the object have a `chunk` attribute? — not deep (it doesn't verify the signature). For real safety, type-check at development time with mypy/pyright.

---

## 14. Cheat sheet

| Thing | What it does | When |
|---|---|---|
| `@staticmethod` | Function on a class, no self/cls | Pure helper that logically belongs with the class |
| `@classmethod` | Method receives `cls`, not `self` | Alternate constructors, pydantic validators |
| `@property` | Method looks like attribute access | Computed attributes |
| `@functools.cache` | Memoize a function's return | Pure function called repeatedly with same args |
| `@dataclass(frozen=True)` | Auto-`__init__`/`__repr__`/`__eq__`, immutable | Value classes (Chunk, ExtractionStep) |
| `field(default_factory=...)` | Per-instance mutable default | `dict`, `list`, `frozenset` fields |
| `Protocol` + `@runtime_checkable` | Structural typing | Plugin shapes (Chunker, Router) |
| `@register_chunker("x")` | Class decorator: register + return unchanged | Plugins discoverable by name |
| `@assemble_case_model()` | Class decorator: build a new class | Generative authoring (lean specs → full BaseModel) |
| `pydantic.create_model(name, __base__=BaseModel, **{n: (ann, default)})` | Build a BaseModel at runtime | Per-organ nested types, rebuilt-for-DSPy classes |
| `dspy.signatures.signature.make_signature(fields_dict, ..., custom_types=...)` | Build a Signature at runtime | The whole signature factory |

If you came here for the signature factory: the path is now visible.
- Spec markers (`@dataclass` machinery + FieldInfo subclasses, §6, §9) declare intent on a lean class.
- `@assemble_case_model` (class decorator, §9) reads them and calls `pydantic.create_model` (§11) to emit the full pydantic shape.
- `build_extraction_signatures` walks that shape, rewrites annotations (§11 again, via `rebuild_basemodel_for_dspy`), and calls `make_signature` (§11) to emit DSPy signatures.
- The output is wrapped in `dspy.Predict(...)` and called by the pipeline.

No raw `type()`, no metaclasses, no monkey-patching. Every step is just **a function that takes a class and returns a class**.
