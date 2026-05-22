# Learning docs

ELI5-to-intermediate documentation for `drr-next`, anchored to the
actual codebase. Each doc opens with a real snippet from the project
and explains the underlying Python / pydantic / DSPy concepts that
make it tick.

These are written for an amateur programmer who wants to stop being
mystified by the codebase — not as a generic tutorial, but as a
reference you can keep open while reading the source.

## The five docs

| # | File | What it covers |
|---|---|---|
| 01 | [01_class_machinery.md](01_class_machinery.md) | `@staticmethod`, `@classmethod`, `@property`, `@functools.cache`, `@dataclass(frozen=True)`, `Protocol`, registry decorators, and finally `pydantic.create_model` / `dspy.make_signature` — the dynamic class generation that powers the signature factory. |
| 02 | [02_typing_and_pydantic.md](02_typing_and_pydantic.md) | Type hints, generics (`list[X]`, `dict[K, V]`), `Optional` / `\|`, `Literal`, `Annotated`, `ClassVar`, `TypedDict`. Then pydantic in depth: `BaseModel`, `Field(...)`, validators, `model_validate` / `model_dump` / `model_json_schema`, and the StrEnum↔Literal round-trip. |
| 03 | [03_chunking.md](03_chunking.md) | The `Chunk` dataclass, three chunkers, two routers, the filter loop in `pipeline_factory.py`, and three concrete ways to feed only lymph-node chunks to the lymph-node extraction signature. Runnable companion: [examples/lymph_node_focus.py](../../examples/lymph_node_focus.py). |
| 04 | [04_dspy_in_depth.md](04_dspy_in_depth.md) | `dspy.LM` / `configure`, `dspy.Signature` (subclass-based and dict-based), `InputField` / `OutputField`, `Predict`, `Module`, `ChainOfThought`, `Embedder` (used for chunk routing), `Tool` and `ReAct` (demo only). What drr-next does NOT use (`Retrieve`, teleprompters) and why. |
| 05 | [05_few_shot_demo_retrieval.md](05_few_shot_demo_retrieval.md) | A concrete design proposal for adding retrieval-augmented few-shot learning to drr-next: a small `demos/` package, one-hook pipeline integration, eval design, and the risks worth flagging. |

## Suggested reading order

`1 → 2 → 4 → 3 → 5`.

- **Doc 1** is the most universal foundation: decorators, `@dataclass`,
  `Protocol`. The early sections don't need pydantic or DSPy.
- **Doc 2** unlocks the pydantic-specific parts of Doc 1's later
  sections (`create_model`, the signature factory's pydantic backend).
- **Doc 4** explains DSPy as the consumer of the factory.
- **Doc 3** plugs into the pipeline that uses all of the above. It's
  the "manipulate the runtime" doc.
- **Doc 5** is the forward-looking proposal. Read last; it composes
  everything before.

If you only have time for one: read Doc 2 — typing and pydantic are
the most pervasive concepts in the codebase.

## Companion code

[examples/lymph_node_focus.py](../../examples/lymph_node_focus.py) — runnable
companion to Doc 3. Three flavors of "feed only lymph-node chunks to
the lymph-node signature" against a synthetic breast pathology report.
Requires the chunking extras: `pip install digital-registrar[chunking]`
and an LM backend (local Ollama or OpenAI creds).

## Note on scope

These docs only describe what's in the codebase **as of the time they
were written**. The forward-looking sections (Doc 5 in particular)
are proposals, not facts about the current code. If you're reading
this much later, run `git log -- docs/learning/` to see what's been
revised.
