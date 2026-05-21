# Python / dataclass / Pydantic / DSPy crash course

A grounded tutorial for reading and debugging the signature factory in this
repo. Written for someone who can read Python but finds the modern
type-system idioms (`from __future__ import annotations`, `type[X]`,
`Literal[...]`, dynamic class construction, etc.) confusing.

The toy example threaded through this doc is a miniature "pet adoption"
signature factory, structurally analogous to the real organ-schema → DSPy
factory in [src/digital_registrar_research/signatures/](../../src/digital_registrar_research/signatures/).

---

## 1. How to use this doc

Two modes:

- **Reading top-to-bottom** — each section adds one concept, with a tiny
  synthetic snippet first and then a callout pointing to the real code in
  this repo that uses the concept. By the time you reach section 11 you
  will have all the pieces to build a miniature signature factory from
  scratch.

- **Returning when stuck** — jump straight to section 13 ("Debugging
  recipes"). Each recipe maps a symptom (an error message, a confusing
  output) to a one-line cause and a fix.

The first time through, don't try to memorize. Read it, run the snippets
in a REPL, and let the patterns sink in. The factory in `signatures/` is
not magic — it's just every concept in this doc stacked together.

---

## 2. Python prerequisites

The five things that have to click before any of the framework-specific
material makes sense.

### 2.1 Classes are values too

In Python, a class is a runtime object you can pass around just like a
number or a string:

```python
class Dog:
    pass

# `Dog` is a value — you can put it in a variable...
my_class = Dog

# ...stick it in a list...
classes = [Dog, int, str]

# ...or call it to make an instance.
fido = Dog()             # instance
print(type(fido))        # <class '__main__.Dog'>
print(type(fido) is Dog) # True
```

The distinction that catches people: `Dog` is a *class*, `Dog()` is an
*instance*. Two completely different kinds of value. When you read a
function signature like:

```python
def register(cls: type[Dog]) -> None: ...
```

…the `type[Dog]` annotation means "the parameter `cls` should be the
*class* `Dog` (or a subclass), not an instance of it." We come back to
this in [section 10](#10-typex-and-the-class-as-value-mental-model). Hold
onto the idea for now.

### 2.2 Type hints are not runtime checks

```python
def add(a: int, b: int) -> int:
    return a + b

add("hello", "world")  # returns "helloworld" — no error!
```

Python does **not** enforce type hints at runtime. They are annotations
attached to the function for *other tools* to read — your editor, mypy,
Pydantic, DSPy. The interpreter itself happily ignores them.

This matters because every framework in this codebase (Pydantic, DSPy)
reads your annotations and uses them to do real work — validation,
JSON-schema generation, LM-output constraints. **The annotations are
data**, not enforcement.

### 2.3 `from __future__ import annotations` (PEP 563)

Every file in `signatures/` starts with this line. Here is what it does.

Without the import:

```python
class Node:
    children: list[Node]   # NameError: name 'Node' is not defined
```

At the time Python evaluates the class body, the name `Node` does not
yet exist (we're in the middle of creating it). So the annotation
`list[Node]` is evaluated eagerly and crashes.

With the import:

```python
from __future__ import annotations

class Node:
    children: list[Node]   # OK — annotation stored as a string
```

The `__future__` import tells Python: **don't evaluate annotations at
class-body / function-definition time. Store them as strings.** Tools
like Pydantic and DSPy resolve those strings later, when all the names
they reference are already defined.

> **In this repo:** every module in `signatures/` and `schemas/pydantic/`
> opens with this line. See [factory.py:17](../../src/digital_registrar_research/signatures/factory.py#L17),
> [_signature_annotation.py:20](../../src/digital_registrar_research/signatures/_signature_annotation.py#L20),
> [_enums.py:15](../../src/digital_registrar_research/schemas/pydantic/_enums.py#L15).

Side effect to know about: because annotations are now strings, you
can't introspect them with naive `Foo.__annotations__` — you must
resolve them with `typing.get_type_hints(Foo)` or
`typing.get_origin` / `typing.get_args`. The factory's annotation
walker uses exactly these introspection helpers.

### 2.4 The `typing` module essentials

```python
from typing import Optional, Union, Literal, Any, Annotated

# Optional[X] is sugar for Union[X, None]
x: Optional[int] = None              # i.e. int OR None

# Modern syntax (3.10+): same thing, with `|`
x: int | None = None

# Union of two non-None types
x: int | str = 0

# Literal — a *type* whose only values are the listed constants.
side: Literal["left", "right"] = "left"

# Any — opt out of typing. Treats the value as anything.
x: Any = ...

# Annotated — attach metadata to a type for tools to read.
from typing import Annotated
PositiveInt = Annotated[int, "must be > 0"]
```

The one most people stumble on is `Literal`. `Literal["a", "b"]` is **a
type**, not a value. It's the type whose only valid values are the
strings `"a"` or `"b"`. DSPy converts `Literal[...]` annotations into a
JSON-schema `enum` constraint, so the LM is required to emit one of the
listed strings.

> **In this repo:** the whole point of the StrEnum → `Literal[...]`
> rewrite in `_signature_annotation.py` is so that DSPy sees a closed
> vocabulary. See [_signature_annotation.py:33-42](../../src/digital_registrar_research/signatures/_signature_annotation.py#L33-L42).

### 2.5 Generic builtins and introspection

Since Python 3.9 you can subscript builtins directly:

```python
xs: list[int] = [1, 2, 3]
m: dict[str, float] = {"a": 1.0}
pair: tuple[str, int] = ("a", 1)

# Variadic tuple: any number of strings.
words: tuple[str, ...] = ("a", "b", "c")
```

For introspection (looking *at* a type, not using it), use
`typing.get_origin` and `typing.get_args`:

```python
import typing

typing.get_origin(list[int])      # <class 'list'>
typing.get_args(list[int])        # (int,)

typing.get_origin(int | None)     # types.UnionType
typing.get_args(int | None)       # (int, <class 'NoneType'>)
```

This is exactly how the factory's annotation walker peels apart a
nested annotation like `list[BreastMargin | None]` to recursively rewrite
each piece.

> **In this repo:** see [_signature_annotation.py:75-76](../../src/digital_registrar_research/signatures/_signature_annotation.py#L75-L76).
> Lines like `origin = _t.get_origin(annotation); args = _t.get_args(annotation)`
> are the entire mechanism for "look inside this type."

---

## 3. Decorators — what they actually are

The fastest way to demystify decorators: **`@deco` is just sugar for
`f = deco(f)`**. A decorator is a function (or class) that takes a
function (or class) and returns a function (or class).

```python
def shouty(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs).upper()
    return wrapper

@shouty
def greet(name):
    return f"hi {name}"

# Equivalent to:
# greet = shouty(greet)

greet("kai")  # "HI KAI"
```

Everything below — `@staticmethod`, `@classmethod`, `@cache`,
`@dataclass`, `@field_validator` — is just a function being applied to
the thing below it.

### 3.1 `@staticmethod`

A method declared inside a class that **ignores `self` and `cls`**. It's
a plain function that lives in the class's namespace.

```python
class TempConverter:
    @staticmethod
    def c_to_f(c):
        return c * 9 / 5 + 32

TempConverter.c_to_f(100)   # 212.0 — called on the class
TempConverter().c_to_f(0)   # 32.0 — same thing on an instance
```

Use it when the function has nothing to do with a specific instance but
*conceptually belongs* to the class.

> **In this repo:** `LowerStrEnum._generate_next_value_` is a
> `@staticmethod` because Python's enum machinery calls it without an
> instance — there is no instance yet at the moment Python is deciding
> what value to assign. See [_enums.py:32-34](../../src/digital_registrar_research/schemas/pydantic/_enums.py#L32-L34).

### 3.2 `@classmethod`

A method whose first argument is the **class itself**, not the instance.
Conventionally called `cls`.

```python
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    @classmethod
    def from_tuple(cls, pair):
        return cls(pair[0], pair[1])    # `cls` is `Point` here

p = Point.from_tuple((3, 4))
```

The classic use case is alternate constructors. The `cls` parameter
means a subclass calling `from_tuple` gets back the subclass, not
hardcoded `Point`.

### 3.3 `@cache` (and `@lru_cache`)

`functools.cache` memoizes a function — the same input arguments will
return the cached result on every subsequent call.

```python
from functools import cache

@cache
def slow_lookup(key):
    print(f"computing {key}")
    return key.upper()

slow_lookup("a")   # prints "computing a", returns "A"
slow_lookup("a")   # returns "A" — no print
slow_lookup("b")   # prints "computing b", returns "B"
```

Restrictions: all arguments must be hashable.

> **In this repo:** the factory doesn't use `@cache` directly — instead
> it manages a module-level dict `_REBUILD_CACHE` for the same reason
> (caching expensive `create_model` calls so DSPy's `custom_types` dict
> stays stable across re-builds). See
> [_signature_annotation.py:127](../../src/digital_registrar_research/signatures/_signature_annotation.py#L127).
> The manual approach is used because the cache key is a tuple including
> `id(...)` of the source class, which a function-level decorator
> couldn't express as cleanly.

### 3.4 `@dataclass`

Big enough to get its own section — see section 4 below.

### 3.5 Looking at multiple decorators

When you see a stack, read **bottom up**:

```python
@cache
@staticmethod
def helper(x):
    ...

# == staticmethod(helper) first, then cache(staticmethod(helper))
```

The decorator closest to `def` is applied first.

---

## 4. Dataclasses

`@dataclass` is a decorator from the standard library that **writes
boilerplate methods for you** based on the class's annotated attributes.

### 4.1 What it generates

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
```

The decorator inspects the class body and generates:

- `__init__(self, x, y)` — assigns the attributes
- `__repr__` — `Point(x=1.0, y=2.0)`
- `__eq__` — field-by-field equality

You get all of that without writing a single method.

```python
p = Point(1.0, 2.0)
q = Point(1.0, 2.0)
p == q        # True
repr(p)       # 'Point(x=1.0, y=2.0)'
```

### 4.2 Defaults and `field(default_factory=...)`

```python
@dataclass
class Inventory:
    items: list[str] = []           # WRONG — see below
```

This is **a trap**. The `[]` is evaluated once at class-definition time,
and *every instance* of `Inventory` will share the same list. Add an
item to one, you've added it to all. Python warns you (`TypeError:
mutable default ... is not allowed`) for built-in mutables in a
dataclass — but the safer idiom is:

```python
from dataclasses import dataclass, field

@dataclass
class Inventory:
    items: list[str] = field(default_factory=list)
```

`default_factory=list` says "call `list()` (i.e. make a fresh empty
list) every time a new instance is created."

### 4.3 `frozen=True`

```python
@dataclass(frozen=True)
class ExtractionStep:
    name: str
    group: str
```

`frozen=True` makes instances **immutable**. Assigning to a field on a
frozen instance raises `FrozenInstanceError`. Two consequences:

- You can put them in `set`s and use them as `dict` keys (frozen
  instances are hashable by default).
- Bugs from accidental mutation become impossible.

> **In this repo:** the central record produced by the factory is a
> frozen dataclass.
>
> ```python
> @dataclass(frozen=True)
> class ExtractionStep:
>     name: str
>     signature: type[dspy.Signature]
>     output_field_names: tuple[str, ...]
>     group: str
> ```
>
> See [factory.py:71-83](../../src/digital_registrar_research/signatures/factory.py#L71-L83).
> Note `signature: type[dspy.Signature]` — the field holds a class, not
> an instance. We will unpack `type[...]` in section 10.

### 4.4 `kw_only` and `slots`

```python
@dataclass(kw_only=True)
class Config:
    host: str
    port: int = 8080
```

`kw_only=True` means callers must pass every argument by keyword:
`Config(host="x", port=80)`, never `Config("x", 80)`. Avoids
positional-argument bugs when a class has many fields.

`slots=True` makes the class use `__slots__` — faster attribute access
and lower memory, at the cost of dynamic attribute assignment.

### 4.5 `__post_init__`

If the generated `__init__` isn't enough — for example, when you need
to validate or derive a field — define `__post_init__`:

```python
@dataclass
class Range:
    low: int
    high: int

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError("low must be <= high")
```

Python's dataclass machinery calls this **after** the generated
`__init__` has assigned all the fields.

### 4.6 `@dataclass` vs `pydantic.BaseModel`

Both let you declare classes with `name: type = default`. The
difference:

- **dataclass**: lightweight, stdlib, *no validation or coercion*.
  Good for internal records that you trust your own code to fill in
  correctly. `ExtractionStep` is a dataclass because nothing external
  ever constructs one — the factory does.
- **BaseModel**: third-party (Pydantic), validates and coerces inputs
  against the annotations, can dump/load JSON, exposes a JSON schema.
  Good for anything that crosses a boundary (LM outputs, API requests,
  GUI form data).

In this repo: case-models (`BreastCancerCase`, `ProstateCancerCase`) are
`BaseModel` because the LM produces them. `ExtractionStep` is a
dataclass because only the factory writes one.

---

## 5. Enums (and the Pydantic-v2 StrEnum pattern)

### 5.1 `Enum` basics

```python
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
```

Each member has a `.name` (the identifier you wrote — `"RED"`) and a
`.value` (what you assigned — `1`). Members compare by identity:

```python
Color.RED == Color.RED        # True
Color.RED == 1                # False — different types!
list(Color)                   # [Color.RED, Color.GREEN, Color.BLUE]
```

### 5.2 `StrEnum`

`StrEnum` (3.11+) is an `Enum` whose members **are strings**:

```python
from enum import StrEnum

class Side(StrEnum):
    LEFT = "left"
    RIGHT = "right"

Side.LEFT == "left"           # True!
"left" in Side                # not directly — use `Side("left")` to coerce
isinstance(Side.LEFT, str)    # True
```

This matters for JSON serialization: a `StrEnum` member serializes as
its string value directly, no extra step required. Pydantic v2 round-trips
this natively: emit `"left"` from the LM, `model_validate` re-coerces
back into `Side.LEFT` on parse.

### 5.3 `auto()` and `_generate_next_value_`

```python
from enum import StrEnum, auto

class Color(StrEnum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()
```

`auto()` is a sentinel that tells the enum machinery "pick a value for
me." For plain `Enum`, it returns `1, 2, 3, ...`. For `StrEnum`, the
default is `name.lower()` — so `Color.RED.value == "red"`.

How does it know what to pick? Each enum class can define
`_generate_next_value_`, which the metaclass calls every time it sees
`auto()`:

```python
class LowerStrEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()
```

The four arguments:

- `name`: the member name as written (`"RED"`)
- `start`: the start value passed to `Enum` (usually 1)
- `count`: how many members have already been defined
- `last_values`: list of values assigned so far

For `StrEnum`, the default `_generate_next_value_` already does
`name.lower()`. So why does this repo redefine it?

> **In this repo:** [_enums.py:20-34](../../src/digital_registrar_research/schemas/pydantic/_enums.py#L20-L34).
> The reason is that the upstream behavior changed across CPython
> versions, and pinning the override locally makes the contract explicit
> and version-independent. It also documents intent for readers — the
> docstring spells out that the canonical wire form is snake_case
> lowercase and that `auto()` is the only blessed way to declare
> members.

### 5.4 Why every enum has `NOT_STATED`

```python
class LymphNodeSide(LowerStrEnum):
    RIGHT = auto()
    LEFT = auto()
    MIDLINE = auto()
    NOT_STATED = auto()
```

`NOT_STATED` is **a convention, not a language feature**. The reason
it's everywhere:

- Clinical reports often don't mention a side at all.
- Without an explicit "report didn't say" member, the LM has to either
  fabricate one of the real values or emit `null`.
- An explicit member lets the LM say "the report did not state this"
  without ambiguity. It also gives the JSON schema a finite, closed set
  of values.

The field annotation in the case-model is typically `LymphNodeSide |
None` (default `None`) — the `None` covers structural absence (the
field isn't present in the response), and `NOT_STATED` covers
semantic absence (the LM read the report and concluded the side
wasn't stated).

---

## 6. Pydantic v2 — the minimum you need

Pydantic is the library that turns a class-of-annotated-fields into a
validator, serializer, and JSON-schema producer. Used **everywhere** in
this codebase for clinical case-models.

### 6.1 `BaseModel`

```python
from pydantic import BaseModel

class Patient(BaseModel):
    name: str
    age: int
    weight_kg: float | None = None
```

Looks like a dataclass — `name: type = default`. But:

```python
p = Patient(name="Kai", age="30")   # note: age is a string!
p.age                                # 30 — coerced to int
p.model_dump()                       # {'name': 'Kai', 'age': 30, 'weight_kg': None}

Patient(name="Kai", age="abc")
# pydantic.ValidationError: Input should be a valid integer ...
```

Pydantic validates **and** coerces. It's the boundary layer between
"untrusted string-shaped input" (LM output, HTTP request body) and
"typed Python object."

### 6.2 `Field(...)` — defaults + metadata

```python
from pydantic import Field

class Tumor(BaseModel):
    size_cm: float = Field(default=0.0, ge=0.0, description="tumor diameter in cm")
```

`Field` lets you attach extra info: validation constraints (`ge`, `le`,
`min_length`, `max_length`), a `description` (which shows up in the JSON
schema — important for DSPy), an `alias`, etc.

### 6.3 `model_config` vs old `class Config:`

Pydantic v1:

```python
class Patient(BaseModel):
    name: str

    class Config:
        extra = "forbid"
```

Pydantic v2:

```python
from pydantic import ConfigDict

class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
```

If you encounter the `class Config:` form in tutorials, that's a v1
pattern. This repo is v2 throughout. If you must mix v1 and v2 examples
when learning, look for `model_config` / `ConfigDict` — its presence
means v2.

### 6.4 Validators

Pydantic v2 uses `@field_validator` and `@model_validator`:

```python
from pydantic import BaseModel, field_validator

class Patient(BaseModel):
    age: int

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("age must be non-negative")
        return v
```

Two things to notice:

- Validators are **`@classmethod`** (you saw the `cls` parameter).
- v1 used `@validator(...)`. If you see that name in a code sample,
  it's v1.

### 6.5 The three methods you'll touch most

```python
p = Patient(name="Kai", age=30)

p.model_dump()                          # -> dict
p.model_dump_json()                     # -> JSON string
Patient.model_validate({"name": "K", "age": 30})  # dict -> instance
Patient.model_json_schema()             # JSON schema dict (sent to LM)
```

`model_validate` is the inverse of `model_dump`: take a dict (or a JSON
object), validate it, return an instance. This is the round-trip that
DSPy uses for LM responses — the LM emits a JSON object matching the
schema, DSPy calls `model_validate`, and bare strings get re-coerced
back into their StrEnum members. Pydantic v2 does this automatically.

> **In this repo:** the round-trip guarantee is what makes the
> StrEnum → `Literal[...]` rewrite safe. The factory tells DSPy "expect
> a string from this finite set," and Pydantic re-coerces the string
> back into the StrEnum member on parse. See the docstring at
> [_signature_annotation.py:16-18](../../src/digital_registrar_research/signatures/_signature_annotation.py#L16-L18).

### 6.6 `create_model` — building classes at runtime

Most of the time, you write a `BaseModel` the normal way:

```python
class Patient(BaseModel):
    name: str
    age: int
```

But sometimes you don't *know* the field names at code-writing time —
they depend on data, configuration, or another schema you're
rewriting. Pydantic exposes `create_model` for this:

```python
from pydantic import BaseModel, create_model, Field

Patient = create_model(
    "Patient",
    __base__=BaseModel,
    name=(str, Field(description="patient name")),
    age=(int, 0),                     # (annotation, default)
)
```

`create_model("Name", __base__=Base, field=(annotation, value_or_FieldInfo))`
returns a brand-new class equivalent to what you would have written
by hand. The string `"Name"` becomes the class's `__name__`.

> **In this repo:** the factory uses `create_model` to *rebuild* each
> nested `BaseModel` with rewritten annotations (StrEnum → Literal) and
> with descriptions injected from the extraction metadata. See
> [_signature_annotation.py:177-181](../../src/digital_registrar_research/signatures/_signature_annotation.py#L177-L181).
> The cleaned-up class is what gets handed to DSPy's `custom_types`
> dict so the LM sees the right JSON schema.

### 6.7 Why Pydantic v2 specifically

If you Google "pydantic", you'll get v1 and v2 examples mixed together.
Concrete v2 markers to spot:

- `model_config = ConfigDict(...)` (not `class Config:`)
- `@field_validator` / `@model_validator` (not `@validator`)
- `.model_dump()` / `.model_validate()` (not `.dict()` / `parse_obj`)
- `create_model(name, __base__=BaseModel, ...)` (the `__base__` keyword)

When in doubt, check `pyproject.toml` — this repo pins Pydantic v2.

---

## 7. DSPy Signatures — what they actually are

### 7.1 A Signature is a class describing one LM call

```python
import dspy

class SummarizeReport(dspy.Signature):
    """Summarize a clinical report in two sentences."""
    report: str = dspy.InputField()
    summary: str = dspy.OutputField()
```

That's a complete DSPy Signature. It says:

- **Inputs**: a single string field called `report`.
- **Outputs**: a single string field called `summary`.
- **Instructions** (the docstring): "Summarize a clinical report in
  two sentences."

When DSPy uses this signature to call an LM, it:

1. Reads the docstring as the prompt's instructions.
2. Reads the type annotations to build a JSON schema for the output.
3. Sends the inputs to the LM together with the schema.
4. Parses the LM's response back through the schema.

### 7.2 `InputField` and `OutputField`

```python
report: str = dspy.InputField()
summary: str = dspy.OutputField(desc="two sentences, plain English")
```

These are **marker values** — DSPy reads them to know which fields are
inputs and which are outputs. `desc=...` gets included in the prompt
so the LM knows what each field means.

`InputField` and `OutputField` are not "values" in the data-flow sense.
The values you'll work with later (the actual `report` string, the
actual `summary` string) come from calling the module that wraps the
Signature (see section 8).

### 7.3 The class form vs the dict form

You can build a Signature two ways:

**Class form** (static, what you see most often):

```python
class StageFromReport(dspy.Signature):
    """Derive the AJCC TNM stage from a pathology report."""
    report: str = dspy.InputField()
    organ: str = dspy.InputField()
    edition: str = dspy.InputField()
    stage: str = dspy.OutputField()
```

**Dict form** (dynamic, what the factory uses):

```python
from dspy.signatures.signature import make_signature

fields = {
    "report":  (str, dspy.InputField()),
    "organ":   (str, dspy.InputField()),
    "edition": (str, dspy.InputField()),
    "stage":   (str, dspy.OutputField()),
}

StageFromReport = make_signature(
    fields,
    instructions="Derive the AJCC TNM stage from a pathology report.",
    signature_name="StageFromReport",
    custom_types={},
)
```

Both produce **the same class**. The dict form is what you use when the
field set isn't known until runtime (because it comes from a Pydantic
schema, configuration, or extraction metadata). The class form is what
you use for static, hand-authored signatures.

> **In this repo:** the static example is `StageFromReport` in
> [staging/README.md:55-63](../../src/digital_registrar_research/staging/README.md#L55-L63).
> The dynamic call is in [factory.py:114-120](../../src/digital_registrar_research/signatures/factory.py#L114-L120).
> The factory specifically needs `make_signature` because it has to
> pass `custom_types` for nested `BaseModel` references — the class
> form doesn't expose that knob.

### 7.4 How DSPy uses the annotations

Annotations become JSON-schema constraints on the LM's output.
Concretely:

| Annotation | Effect on LM output |
| --- | --- |
| `str` | any string |
| `int`, `float` | the LM must emit a number |
| `bool` | emit `true` / `false` |
| `Literal["a", "b", "c"]` | must pick one of three strings — closed enum |
| `list[X]` | JSON array of items matching `X`'s schema |
| `MyBaseModel` | JSON object matching `MyBaseModel.model_json_schema()` |
| `X | None` | nullable version of `X` |

This is *why* the factory rewrites every StrEnum into `Literal[...]`:
the LM gets a strict, closed vocabulary at the JSON-schema level, and
the response is guaranteed to be one of those exact strings.

### 7.5 `custom_types` for nested `BaseModel`

When a Signature has a field like `margins: list[BreastMargin]`, DSPy
needs to be able to *resolve the name* `BreastMargin` when it generates
the JSON schema. It can't just walk the annotation — the name might
have been rewritten or be defined in a module DSPy doesn't import.

The fix: pass it in explicitly.

```python
sig = make_signature(
    fields,
    instructions=...,
    signature_name=...,
    custom_types={"BreastMargin": rebuilt_BreastMargin_class},
)
```

> **In this repo:** see [factory.py:13-15](../../src/digital_registrar_research/signatures/factory.py#L13-L15)
> for the rationale, and [factory.py:118](../../src/digital_registrar_research/signatures/factory.py#L118)
> for the call site. The `_collect_custom_types_from_annotation` helper
> at [factory.py:159-167](../../src/digital_registrar_research/signatures/factory.py#L159-L167)
> walks the rewritten annotation to populate this dict.

---

## 8. DSPy Modules — what consumes a Signature

A Signature by itself doesn't call any LM. To actually run one, you
wrap it in a `dspy.Module`. Three modules cover almost everything in
this repo.

### 8.1 `dspy.Predict` — the basic one

```python
import dspy

sig = StageFromReport            # the class from 7.3
predictor = dspy.Predict(sig)    # an instance of dspy.Predict
result = predictor(
    report="...",
    organ="breast",
    edition="AJCC 8",
)
print(result.stage)              # "IIA"
```

`Predict` takes a Signature *class*, and you call the *instance* like a
function. It runs one LM call, parses the response against the
Signature's schema, and returns a `dspy.Prediction` object whose
attributes match the output field names.

> **In this repo:** `dspy.Predict` is wired in the production pipeline at
> [pipeline_factory.py:83](../../src/digital_registrar_research/pipeline_factory.py#L83)
> (router) and [pipeline_factory.py:120](../../src/digital_registrar_research/pipeline_factory.py#L120)
> (per-step extractors). Each `ExtractionStep` produced by the factory
> gets paired with a `dspy.Predict` instance.

### 8.2 `dspy.ChainOfThought` — adds a reasoning field

```python
predictor = dspy.ChainOfThought(sig)
result = predictor(report="...")
print(result.reasoning)   # the LM's step-by-step
print(result.stage)
```

Same interface as `Predict`, but the LM is **asked for a `reasoning`
field before the real outputs**. This often improves answer quality on
tasks that benefit from explicit step-by-step thinking, at the cost of
a longer response.

The Signature stays *exactly the same*. The change is purely in the
wrapper.

> **In this repo:** see the ablation runner that swaps `Predict` for
> `ChainOfThought` at [ablations/runners/chain_of_thought.py:38](../../src/digital_registrar_research/ablations/runners/chain_of_thought.py#L38):
>
> ```python
> def _make_predict(sig, use_cot: bool):
>     return dspy.ChainOfThought(sig) if use_cot else dspy.Predict(sig)
> ```
>
> One-line swap, same Signature.

### 8.3 `dspy.ReAct` — the agentic loop

```python
predictor = dspy.ReAct(sig, tools=[tool_a, tool_b], max_iters=6)
result = predictor(report="...")
```

`ReAct` runs an iterative loop:

1. The LM reads the inputs.
2. The LM decides to either (a) **call a tool**, or (b) **emit the
   final outputs**.
3. If (a), the tool's Python function runs, its return value goes back
   to the LM, and the loop iterates.
4. If (b), the loop ends and `result` is the Prediction.

`max_iters` is a safety cap — if the LM hasn't emitted final outputs
after that many tool calls, the loop terminates.

> **In this repo:** [staging/README.md:62](../../src/digital_registrar_research/staging/README.md#L62)
> shows the agent:
>
> ```python
> agent = dspy.ReAct(StageFromReport, tools=STAGING_TOOLS, max_iters=6)
> agent(report=..., organ="breast", edition="AJCC 8")
> ```
>
> The same Signature class can be wrapped by `Predict`, `ChainOfThought`,
> or `ReAct` — the Signature describes *what* the LM should produce,
> the Module describes *how* it gets there.

### 8.4 Mental model

| Module | What it does |
| --- | --- |
| `Predict` | one LM call → parsed outputs |
| `ChainOfThought` | one LM call, plus an extra `reasoning` output |
| `ReAct` | a loop: LM → tool → LM → … → final outputs |

The Signature is *always* the same shape; the choice of Module is a
strategy decision.

---

## 9. DSPy Tools — letting the LM call your Python

`dspy.Tool` wraps a plain Python function with a name and a description
so the LM can decide when to invoke it.

### 9.1 The mechanics

```python
import dspy

def lookup_adoption_fee(species: str) -> float:
    fees = {"dog": 250.0, "cat": 175.0, "rabbit": 90.0}
    return fees.get(species, 0.0)

fee_tool = dspy.Tool(
    func=lookup_adoption_fee,
    name="lookup_adoption_fee",
    desc=(
        "Look up the adoption fee in dollars for a pet species. "
        "Args: species (str, one of 'dog', 'cat', 'rabbit'). "
        "Returns a float."
    ),
)
```

The wrapped function stays a normal Python function — you can still
call `lookup_adoption_fee("dog")` directly. The `dspy.Tool` is an
adapter that tells `dspy.ReAct` "here's a thing the LM can call."

### 9.2 The `desc` field IS the prompt

This is the most important rule: **`desc` is what the LM reads**. It is
not docs for human developers. Write it like a prompt:

- Spell out argument names and types.
- Describe the return shape.
- Mention valid values, units, edge cases.

> **In this repo:** look at the descriptions in [staging/tools.py:8-31](../../src/digital_registrar_research/staging/tools.py#L8-L31).
> Each one spells out every argument's type and meaning, the return
> shape, and when to call the tool. None of that is for a Python
> developer; it's for an LM that has never seen the wrapped function.

### 9.3 Type hints help — but `desc` is authoritative

If your wrapped function has good type hints (especially with simple
parameter types), DSPy can infer the parameter schema. But for any
case where the schema isn't obvious — for example a `dict` whose keys
vary by organ — the description has to spell it out. The staging tools
rely heavily on `desc` for exactly this reason.

### 9.4 One iteration of the staging ReAct loop

Walking the staging example end-to-end, with `STAGING_TOOLS =
[stage_from_observations_tool, observable_schema_tool]`:

1. LM gets the pathology report and the prompt.
2. LM calls `cancer_observable_schema("breast", "AJCC 8")` to learn what
   observations the staging engine wants.
3. Engine returns `{tumor_size_cm: ..., ln_positive_count: ..., ...}`.
4. LM extracts the relevant values from the report.
5. LM calls `cancer_stage_from_observations("breast", "AJCC 8",
   {tumor_size_cm: 2.3, ...})`.
6. Tool returns `{T: "T2", N: "N0", M: "M0", stage: "IIA"}`.
7. LM emits the final `stage` output: `"IIA"`.

The whole staging loop is just two tool calls. The LM is essentially
acting as a glue layer between "raw pathology report" and "structured
inputs to a deterministic staging engine."

### 9.5 Tool vs. OutputField

- **OutputField**: a value the LM should *always* produce on this call.
- **Tool**: an action the LM *might* take if it decides it needs to.

Rule of thumb: if the value can be looked up or computed
deterministically given a small input, expose it as a tool. If it must
come from the LM's reading of the inputs, make it an OutputField.

---

## 10. `type[X]` and the class-as-value mental model

We saw at the start that classes are values. The annotation `type[X]`
formalizes that.

```python
def register(cls: type[BaseModel]) -> None:
    ...

register(Patient)      # OK — Patient is a class
register(Patient())    # type error — passing an instance
```

`type[X]` means "the parameter is the *class* X (or any subclass of
X)." It's not "an instance of X" — that would be the unadorned `X`.

A small comparison:

```python
def store_instance(obj: dspy.Signature) -> None:
    ...   # `obj` is an instantiated signature object

def store_class(cls: type[dspy.Signature]) -> None:
    ...   # `cls` is the signature class itself
```

`dspy.Signature` instances are essentially never used directly — the
classes are what you pass around. So when you see:

```python
@dataclass(frozen=True)
class ExtractionStep:
    signature: type[dspy.Signature]
    ...
```

…the `type[dspy.Signature]` is telling you "this field holds a class
object — the class that `dspy.Predict(...)` will be called on later, not
an already-run prediction."

> **In this repo:** [factory.py:81](../../src/digital_registrar_research/signatures/factory.py#L81)
> declares `signature: type[dspy.Signature]`. Downstream at
> [pipeline_factory.py:120](../../src/digital_registrar_research/pipeline_factory.py#L120),
> the pipeline does `dspy.Predict(s.signature)` — it calls `Predict` on
> the class, not on an instance.

---

## 11. Putting it together — a tiny signature factory

Time to build a miniature version of the real factory using only the
concepts above. We'll use a "pet adoption" domain: a `PetAdoptionCase`
schema with one StrEnum field and one nested `BaseModel`. The build
proceeds in stages; each stage maps to a real-factory file:line.

### 11.1 Stage 1 — define the clean schema

```python
from __future__ import annotations
from enum import StrEnum, auto
from pydantic import BaseModel, Field


class LowerStrEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class PetSpecies(LowerStrEnum):
    DOG = auto()
    CAT = auto()
    RABBIT = auto()
    NOT_STATED = auto()


class VetVisit(BaseModel):
    """A single vet visit on the pet's record."""
    reason: str = Field(description="why the visit happened")
    date: str = Field(description="ISO date YYYY-MM-DD")


class PetAdoptionCase(BaseModel):
    """Adoption-record case model."""
    species: PetSpecies | None = Field(
        default=None,
        description="the kind of animal being adopted",
    )
    visits: list[VetVisit] = Field(
        default_factory=list,
        description="recent vet visits, most recent first",
    )
```

That's the *authoring* schema — the kind of thing a human would
hand-edit. It uses StrEnum (closed vocab), nested `BaseModel`
(`VetVisit`), and field descriptions on every annotation.

> Mirrors [schemas/pydantic/prostate.py](../../src/digital_registrar_research/schemas/pydantic/prostate.py)
> in style.

### 11.2 Stage 2 — rewrite the StrEnum to `Literal`

```python
from enum import Enum
from typing import Any, Literal

def enum_to_literal(enum_cls: type[Enum]) -> Any:
    values = tuple(member.value for member in enum_cls)
    return Literal[values]  # type: ignore[valid-type]
```

`enum_to_literal(PetSpecies)` returns
`Literal["dog", "cat", "rabbit", "not_stated"]`. DSPy will turn that
into a closed JSON-schema enum and the LM is forced to pick one.

> Mirrors [_signature_annotation.py:33-42](../../src/digital_registrar_research/signatures/_signature_annotation.py#L33-L42).

### 11.3 Stage 3 — walk a (possibly nested) annotation

```python
import typing as _t

def is_strenum(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, Enum)

def walk(annotation: Any) -> Any:
    """Rewrite StrEnum -> Literal, recursively, through list/Optional/Union."""
    if is_strenum(annotation):
        return enum_to_literal(annotation)

    origin = _t.get_origin(annotation)
    args = _t.get_args(annotation)
    if origin is None:
        return annotation                 # plain type, leave alone

    if origin is list:
        return list[walk(args[0])]
    if origin is _t.Union:
        return _t.Union[tuple(walk(a) for a in args)]
    return annotation
```

This is the heart of the real factory's annotation rewriter,
simplified. Run it once to confirm:

```python
walk(PetSpecies)                  # Literal['dog', 'cat', 'rabbit', 'not_stated']
walk(PetSpecies | None)           # Literal[...] | None
walk(list[PetSpecies])            # list[Literal[...]]
```

> Mirrors [_signature_annotation.py:50-95](../../src/digital_registrar_research/signatures/_signature_annotation.py#L50-L95).

### 11.4 Stage 4 — rebuild the nested `BaseModel` with `create_model`

Why not just *use* `VetVisit` as-is? Two reasons:

1. If `VetVisit` itself had StrEnum fields, they'd need rewriting too.
2. We may want to inject additional field descriptions from elsewhere
   (in the real factory: from per-organ extraction metadata).

```python
from pydantic import create_model

def rebuild(cls: type[BaseModel]) -> type[BaseModel]:
    new_fields = {}
    for fname, finfo in cls.model_fields.items():
        new_ann = walk(finfo.annotation)
        new_fields[fname] = (new_ann, finfo)
    return create_model(cls.__name__, __base__=BaseModel, **new_fields)
```

> Mirrors the real `rebuild_basemodel_for_dspy` at
> [_signature_annotation.py:130-185](../../src/digital_registrar_research/signatures/_signature_annotation.py#L130-L185).

### 11.5 Stage 5 — build the DSPy Signature

```python
import dspy
from dspy.signatures.signature import make_signature

def build_signature_for(case_cls: type[BaseModel]) -> type[dspy.Signature]:
    custom_types: dict[str, type] = {}
    output_fields: dict[str, tuple[Any, Any]] = {}

    for fname, finfo in case_cls.model_fields.items():
        new_ann = walk(finfo.annotation)
        # If the rewritten annotation references a BaseModel, rebuild + collect.
        for arg in _t.get_args(new_ann) + (new_ann,):
            if isinstance(arg, type) and issubclass(arg, BaseModel) and arg is not BaseModel:
                rebuilt = rebuild(arg)
                custom_types[rebuilt.__name__] = rebuilt
        output_fields[fname] = (
            new_ann,
            dspy.OutputField(desc=finfo.description or ""),
        )

    input_fields = {
        "report": (str, dspy.InputField(desc="the raw adoption record")),
    }

    return make_signature(
        {**input_fields, **output_fields},
        instructions=(case_cls.__doc__ or case_cls.__name__).strip(),
        signature_name=f"{case_cls.__name__}__extract",
        custom_types=custom_types,
    )
```

> Mirrors [factory.py:86-120](../../src/digital_registrar_research/signatures/factory.py#L86-L120).

### 11.6 Stage 6 — wrap the result in a step record

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TinyStep:
    name: str
    signature: type[dspy.Signature]
    output_field_names: tuple[str, ...]
```

> Mirrors [factory.py:71-83](../../src/digital_registrar_research/signatures/factory.py#L71-L83).

### 11.7 Stage 7 — verify by printing the JSON schema

```python
sig = build_signature_for(PetAdoptionCase)
step = TinyStep(
    name=sig.__name__,
    signature=sig,
    output_field_names=("species", "visits"),
)

print(step.signature.model_json_schema())
```

You will see a JSON schema where:

- `species` is an enum of `"dog", "cat", "rabbit", "not_stated"`
  (because of stage 2).
- `visits` is an array of objects with `reason` and `date` fields,
  each carrying its description (because of stage 4).
- The top-level `description` is the class docstring.

No LM call is required to confirm this — the schema is generated
statically.

### 11.8 Swap `Predict` for `ChainOfThought`

```python
predictor = dspy.Predict(step.signature)
# vs:
predictor_cot = dspy.ChainOfThought(step.signature)
```

Same Signature, different Module. The CoT version asks the LM for a
`reasoning` field in addition to `species` and `visits`. Nothing about
the Signature changed.

### 11.9 Side-by-side map

| Toy stage | Mirrors in the real factory |
| --- | --- |
| 11.1 `LowerStrEnum`, `PetAdoptionCase` | [_enums.py:20-34](../../src/digital_registrar_research/schemas/pydantic/_enums.py#L20-L34), [schemas/pydantic/prostate.py](../../src/digital_registrar_research/schemas/pydantic/prostate.py) |
| 11.2 `enum_to_literal` | [_signature_annotation.py:33-42](../../src/digital_registrar_research/signatures/_signature_annotation.py#L33-L42) |
| 11.3 `walk` | [_signature_annotation.py:50-95](../../src/digital_registrar_research/signatures/_signature_annotation.py#L50-L95) |
| 11.4 `rebuild` | [_signature_annotation.py:130-185](../../src/digital_registrar_research/signatures/_signature_annotation.py#L130-L185) |
| 11.5 `build_signature_for` | [factory.py:86-120](../../src/digital_registrar_research/signatures/factory.py#L86-L120) |
| 11.6 `TinyStep` | [factory.py:71-83](../../src/digital_registrar_research/signatures/factory.py#L71-L83) |
| 11.7 verification | use [pipeline_factory.py:120](../../src/digital_registrar_research/pipeline_factory.py#L120) for the production version |

---

## 12. A tiny ReAct agent with a Tool

Same pet-adoption domain. We'll let the LM look up an adoption fee from
a Python dictionary using a `dspy.Tool`.

### 12.1 A plain Python function — your business logic

```python
def lookup_adoption_fee(species: str) -> float:
    fees = {"dog": 250.0, "cat": 175.0, "rabbit": 90.0, "not_stated": 0.0}
    return fees.get(species, 0.0)
```

No LM in sight. Just a function.

### 12.2 Wrap it as a `dspy.Tool`

```python
import dspy

fee_tool = dspy.Tool(
    func=lookup_adoption_fee,
    name="lookup_adoption_fee",
    desc=(
        "Look up the adoption fee in US dollars for a pet species. "
        "Args: species (str, one of 'dog', 'cat', 'rabbit', 'not_stated'). "
        "Returns a float (the fee in USD)."
    ),
)
```

Notice how explicit `desc` is. The LM has never seen `lookup_adoption_fee`
— `desc` is the only thing it reads. Spell out arg names, valid values,
return type, and units.

### 12.3 A Signature for the recommendation task

```python
class RecommendPet(dspy.Signature):
    """Recommend a pet species for a user and quote the adoption fee."""
    user_request: str = dspy.InputField(desc="free-text user request")
    species: str = dspy.OutputField(desc="the recommended species name")
    fee: float = dspy.OutputField(desc="the adoption fee in USD")
    rationale: str = dspy.OutputField(desc="one-sentence explanation")
```

### 12.4 Wire up the agent

```python
agent = dspy.ReAct(RecommendPet, tools=[fee_tool], max_iters=4)
```

`max_iters=4` is a safety cap. If the LM hasn't emitted final outputs
after four tool-call rounds, the loop terminates.

### 12.5 What one iteration looks like

Concrete example with `user_request="I have a small apartment and I work
long hours"`:

1. LM reads the prompt and sees it can call `lookup_adoption_fee`.
2. LM reasons: "small apartment + long hours → cat is suitable; I should
   look up the fee."
3. LM calls `lookup_adoption_fee(species="cat")`.
4. Tool returns `175.0`. The LM sees the return value.
5. LM emits the final outputs: `species="cat"`, `fee=175.0`,
   `rationale="cats are independent and apartment-friendly"`.

The whole interaction is *one* tool call. If the LM had also wanted to
check the rabbit fee, it would have called the tool twice.

### 12.6 Side-by-side with the real staging code

| Toy stage | Real-staging mirror |
| --- | --- |
| 12.1 plain function | `stage_from_observations` in the staging adapter |
| 12.2 `dspy.Tool(func=..., desc=...)` | [staging/tools.py:8-20](../../src/digital_registrar_research/staging/tools.py#L8-L20) |
| 12.3 `RecommendPet` Signature (class form) | `StageFromReport` in [staging/README.md:55-63](../../src/digital_registrar_research/staging/README.md#L55-L63) |
| 12.4 `dspy.ReAct(...)` | [staging/README.md:62](../../src/digital_registrar_research/staging/README.md#L62) |

### 12.7 Two rules of thumb

- **Tool `desc` is a prompt, not docs.** Write it for an LM reader who
  has never seen the function. Mention every argument, every valid
  value, the return shape, and any units.
- **If your tool takes a `dict` with a varying shape, spell the shape
  out in `desc`.** This is exactly what `cancer_stage_from_observations`
  does — the observations dict depends on `(organ, edition)`, so the
  description tells the LM how to look it up first (using the other
  tool). DSPy can't infer a varying schema, so the description has to
  do the work.

---

## 13. Debugging recipes

Symptom → cause → fix. Skim down the table.

### 13.1 "I changed an enum and DSPy doesn't see the new value"

**Cause**: the per-process `_REBUILD_CACHE` in
[_signature_annotation.py:127](../../src/digital_registrar_research/signatures/_signature_annotation.py#L127)
is still holding the old rebuilt class.

**Fix**: restart the REPL / kernel. If you're hot-reloading in a
notebook, clear the cache explicitly:

```python
from digital_registrar_research.signatures import _signature_annotation
_signature_annotation._REBUILD_CACHE.clear()
```

### 13.2 "NameError: name 'X' is not defined" inside an annotation

**Cause**: either the file is missing `from __future__ import annotations`
and you have a forward reference, or the symbol genuinely isn't imported
at runtime.

**Fix**:

1. Add `from __future__ import annotations` to the top of the file.
2. If the import is already there, check that the symbol you reference
   is in scope at the point Pydantic / DSPy *resolves* the annotation
   — which may be a different module than where it was written.

### 13.3 "Pydantic complains about an unresolved nested type"

**Cause**: a nested `BaseModel` reference in the signature, but the
factory didn't pass it via `custom_types`.

**Fix**: trace through `_collect_custom_types_from_annotation` at
[factory.py:159-167](../../src/digital_registrar_research/signatures/factory.py#L159-L167)
on the rewritten annotation. The nested class needs to land in the dict
under its `__name__`.

### 13.4 "TypeError: unhashable type" or duplicate enum members

**Cause**: `auto()` returned a value that collides with another member.
Most commonly: a typo in `_generate_next_value_` returning the wrong
type (a list instead of a string, etc.).

**Fix**: print `[m.value for m in MyEnum]` and verify each value is a
unique, hashable string.

### 13.5 "I want to see exactly what JSON DSPy sends to the LM"

```python
print(step.signature.model_json_schema())
```

To see the *full LM history* (prompts + responses) after a call:

```python
dspy.inspect_history(n=1)        # last 1 LM call
```

### 13.6 "How do I add a new closed vocabulary?"

1. Add a new `LowerStrEnum` subclass in
   [schemas/pydantic/_enums.py](../../src/digital_registrar_research/schemas/pydantic/_enums.py).
2. Use it as a field annotation in the relevant case-model
   (`schemas/pydantic/<organ>.py`).
3. No factory changes needed — the annotation walker picks it up
   automatically.
4. Add a description to the field in `schemas/extraction/<organ>.py` so
   the LM knows what the vocabulary means.

### 13.7 "Why is my field showing up as `Optional[X]` in the JSON schema?"

**Cause**: the annotation is `X | None` (with or without a `None`
default). Pydantic + DSPy render that as a nullable schema.

**Fix**: drop the `| None` if the field is genuinely required. Or
accept the nullable form — most clinical fields *should* be nullable
because reports can omit anything.

### 13.8 "My ReAct agent keeps calling the wrong tool / never calls one"

**Cause**: the tool descriptions don't make the right tool obviously
right for the situation. The LM picks based on `desc` alone.

**Fix**: rewrite each `desc` to spell out:

- exactly when this tool should be called,
- exactly when it should *not* be called,
- what the args mean.

If two tools have overlapping responsibilities, give each `desc` a
phrase like "use this **first**" or "use this **after** you've called
…" so the LM knows the order.

### 13.9 "ReAct hit max_iters without emitting final outputs"

**Cause**: the LM is stuck in a tool-calling loop and can't satisfy
the Signature's output schema.

**Fix** (in order of preference):

1. Tighten the tool descriptions so the LM picks the right one faster.
2. Raise `max_iters`.
3. Simplify the Signature's output fields — fewer required outputs =
   fewer reasons to keep looping.
4. Inspect history with `dspy.inspect_history(n=10)` to see where the
   loop is going wrong.

---

## 14. Glossary

| Term | Meaning |
| --- | --- |
| **annotation** | The `: type` part of `x: int`. Stored on the class/function for tools (Pydantic, DSPy, mypy) to read. Not enforced by Python itself. |
| **BaseModel** | Pydantic's superclass for validated, JSON-serializable models. |
| **ChainOfThought** | A DSPy Module like `Predict` but with an extra `reasoning` output. |
| **classmethod** | A method whose first arg is the class, not the instance. |
| **ConfigDict** | The Pydantic v2 way to declare model-level config. Replaces `class Config:`. |
| **create_model** | Pydantic function that builds a `BaseModel` subclass at runtime. |
| **custom_types** | The keyword argument to `make_signature` that tells DSPy how to resolve nested-`BaseModel` references in annotations. |
| **dataclass** | Stdlib decorator that auto-generates `__init__`/`__repr__`/`__eq__` from annotated fields. |
| **default_factory** | `field(default_factory=list)` — produces a fresh default on each construction. Use instead of mutable literals. |
| **dspy.Predict** | DSPy Module that runs one LM call for a given Signature. |
| **dspy.ReAct** | DSPy Module that runs a tool-calling loop driven by an LM. |
| **dspy.Signature** | A class describing one LM call: typed inputs, typed outputs, an instructions string. |
| **dspy.Tool** | Wrapper around a Python function so a ReAct LM can invoke it. |
| **enum_to_literal** | Helper in this repo that converts a StrEnum into `Literal[<values>]`. |
| **field_validator** | Pydantic v2 decorator for per-field validation logic. |
| **frozen** | `@dataclass(frozen=True)` — immutable instances; hashable by default. |
| **from \_\_future\_\_ import annotations** | Defers annotation evaluation to runtime introspection time (PEP 563). |
| **get_origin / get_args** | Stdlib introspection on a parameterized type: `list[int]` → `list`, `(int,)`. |
| **InputField / OutputField** | Markers in a Signature class that tell DSPy which fields are inputs vs outputs. |
| **Literal** | Type whose only valid values are the listed constants. Becomes a closed enum in JSON schema. |
| **LowerStrEnum** | This repo's StrEnum subclass: `auto()` yields `name.lower()`. |
| **make_signature** | DSPy's dynamic Signature constructor; the only way to pass `custom_types`. |
| **model_config** | Pydantic v2 class-level config attribute. |
| **model_dump / model_validate** | Pydantic v2 dict↔instance methods. |
| **model_json_schema** | Returns the JSON schema DSPy emits to the LM. |
| **NOT_STATED** | A repo-wide convention: every closed vocabulary has this member as the explicit "report did not say" sentinel. |
| **StrEnum** | Enum whose members *are* strings (3.11+). |
| **type[X]** | "The class X (or a subclass), not an instance of X." |

---

## 15. Further reading

Python:

- [PEP 563 — postponed evaluation of annotations](https://peps.python.org/pep-0563/)
- [stdlib `typing` module](https://docs.python.org/3/library/typing.html)
- [stdlib `enum` module](https://docs.python.org/3/library/enum.html)
- [stdlib `dataclasses` module](https://docs.python.org/3/library/dataclasses.html)
- [stdlib `functools` (`cache`, `lru_cache`)](https://docs.python.org/3/library/functools.html)

Pydantic v2:

- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [Migration guide (v1 → v2)](https://docs.pydantic.dev/latest/migration/)
- [`create_model`](https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation)

DSPy:

- [DSPy docs — Signatures](https://dspy.ai/learn/programming/signatures/)
- [DSPy docs — Modules (Predict, ChainOfThought, ReAct)](https://dspy.ai/learn/programming/modules/)
- [DSPy docs — Tools](https://dspy.ai/learn/programming/tools/)

In this repo:

- [docs/architecture/schemas.md](../architecture/schemas.md) — how the two-layer schema authoring works.
- [docs/architecture/dspy_deep_dive.md](../architecture/dspy_deep_dive.md) — deeper notes on DSPy in this codebase.
- [docs/reference/schema_gui_blueprint.md](schema_gui_blueprint.md) — the schema-editor blueprint that the case-models feed into.
- [src/digital_registrar_research/staging/README.md](../../src/digital_registrar_research/staging/README.md) — the real-world ReAct + Tool example referenced throughout section 12.
