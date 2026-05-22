# 04 — DSPy in depth (what's used, what's not, how)

> Goal: understand which DSPy primitives `drr-next` uses, what each one really does behind the scenes, and which DSPy concepts are *forward-looking* (ReAct, Tool, Embedder).

This doc assumes you've at least skimmed [01 — class machinery](01_class_machinery.md) and [02 — typing & pydantic](02_typing_and_pydantic.md). DSPy stands on both.

---

## 0. ELI5: what is DSPy?

DSPy is a thin layer that turns a **typed Python class** into a **prompt**, calls an **LM**, and **parses** the response back into typed Python.

You write:

```python
class IsCancer(dspy.Signature):
    """You are a cancer registrar..."""

    report: list = dspy.InputField(desc="this is a pathologic report...")
    cancer_excision_report: bool = dspy.OutputField(desc="identify whether...")
    cancer_category: Literal["breast", "lung", ...] | None = dspy.OutputField(desc="identify which organ...")
```

You call:

```python
predictor = dspy.Predict(IsCancer)
result = predictor(report=["Specimen...", "Microscopic..."])

print(result.cancer_excision_report)  # bool
print(result.cancer_category)         # "breast"
```

DSPy turns the class into a JSON-schema-shaped prompt, sends it to the configured LM, parses the response back, and returns a `Prediction` (an object with the output fields as attributes).

That's it. Everything else is variations on that core loop.

---

## 1. What drr-next actually uses from DSPy

| Primitive | Used? | Where |
|---|---|---|
| `dspy.LM` + `dspy.configure` | Yes (core) | [models/common.py](../../src/digital_registrar/models/common.py) |
| `dspy.Signature` | Yes (core) | Everywhere |
| `dspy.InputField` / `dspy.OutputField` | Yes (core) | Everywhere |
| `dspy.Predict` | Yes (core) | [pipeline_factory.py](../../src/digital_registrar/pipeline_factory.py), [models/](../../src/digital_registrar/models/) |
| `dspy.Module` | Yes (core) | `CancerPipelineV2` |
| `dspy.Embedder` | Yes (optional) | [chunking/dspy_embedder_router.py](../../src/digital_registrar/chunking/dspy_embedder_router.py) |
| `dspy.Tool` | Demo only | [staging/tools.py](../../src/digital_registrar/staging/tools.py) |
| `dspy.ReAct` | Demo only | [examples/staging_demo/react_agent.py](../../examples/staging_demo/react_agent.py) |
| `dspy.ChainOfThought` | Attic only | Old ablations |
| `dspy.Retrieve`, `dspy.ColBERTv2` | Not used | — |
| DSPy compilation / teleprompters | Not used (attic experiments) | — |

The bulk of this doc covers the **production-used** primitives. Sections 9–11 cover Embedder, Tool, ReAct.

---

## 2. `dspy.LM` and `dspy.configure` — the backend layer

DSPy is provider-agnostic. You build an `LM` for whichever model you want and tell DSPy to use it globally.

[src/digital_registrar/models/common.py:98-127](../../src/digital_registrar/models/common.py#L98-L127):

```python
def load_model(model_name: str, overrides: dict | None = None):
    model_id = model_list[model_name] if model_name in model_list else None
    ...
    kwargs = compute_lm_kwargs(model_name, overrides=overrides)

    if model_id.startswith("openai/"):
        from digital_registrar.util.secrets import load_openai_key
        api_key = load_openai_key()
        api_kwargs = {k: v for k, v in kwargs.items() if k not in _OLLAMA_ONLY_KEYS}
        lm = dspy.LM(
            model=model_id,
            api_key=api_key,
            model_type="chat",
            **api_kwargs,
        )
        ...
        return lm

    lm = dspy.LM(
        model=model_id,
        api_base=localaddr,
        api_key="",
        model_type="chat",
        **kwargs,
    )
    return lm

def autoconf_dspy(model_name: str, overrides: dict | None = None):
    lm = load_model(model_name, overrides=overrides)
    dspy.configure(lm=lm)
```

What this does:

1. **`dspy.LM(model=..., **kwargs)`** — build a callable that talks to a specific model. The `model` argument is parsed by LiteLLM (DSPy uses LiteLLM under the hood). Prefixes:
   - `openai/<model>` → OpenAI chat-completions API
   - `ollama_chat/<model>` → Local Ollama
   - `anthropic/<model>`, `azure/...`, many others — see LiteLLM docs

2. **`dspy.configure(lm=...)`** — set the global LM. Every `dspy.Predict(...)`, `dspy.ChainOfThought(...)`, `dspy.ReAct(...)` from this point on uses that LM.

You can also pass `lm=` directly to a module to override per-call:

```python
predictor = dspy.Predict(MySignature)
result = predictor(report="...", lm=other_lm)
```

But the global pattern is what drr-next uses.

**`dspy.configure(...)` can also set other things**:
- `disable_typeguard_warnings=True` — drr-next sets this in [pipeline_factory.py:52-56](../../src/digital_registrar/pipeline_factory.py#L52-L56) because DSPy 3.2 emits noisy warnings for dynamically-built signatures.
- `trace=[]` — DSPy's tracing hooks (advanced).

---

## 3. `dspy.Signature` — the contract

A Signature is a class that declares: "given these input fields, produce these output fields. Here are descriptions for each, and here's the overall instruction (the class docstring)."

There are **two ways to write one**.

### Subclass-based (legacy / hand-authored)

[src/digital_registrar/models/common.py:135-143](../../src/digital_registrar/models/common.py#L135-L143):

```python
class is_cancer(dspy.Signature):
    """You are a cancer registrar, you need to identify whether or not this report
    belongs to PRIMARY cancer excision eligible for cancer registry..."""

    report: list = dspy.InputField(
        desc='this is a pathologic report, separated into paragraphs...'
    )
    cancer_excision_report: bool = dspy.OutputField(
        desc='identify whether or not this report belongs to PRIMARY cancer excision...'
    )
    cancer_category: Literal["stomach","colorectal","breast",...]|None = dspy.OutputField(
        desc='identify which organ the primary cancer arises from...'
    )
    cancer_category_others_description: str|None = dspy.OutputField(
        desc='if is cancer_excision report AND cancer_category is others...'
    )
```

What's happening:
- The **docstring** becomes the LM-facing instructions.
- Each `dspy.InputField` / `dspy.OutputField` line declares one field with a type annotation, optional default, and a description.
- DSPy reads the class at construction time to build a JSON schema describing the expected I/O.

### Dict-based (`make_signature`) — what the factory uses

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

Same outputs as the class-based form, but built at runtime from a `{field_name: (annotation, InputField|OutputField)}` dict. drr-next uses this for **every** organ-specific extraction signature because they're parameterized by the pydantic schema and Layer-2 metadata.

For the deep dive on why dynamic construction, see [Doc 01](01_class_machinery.md). For the type-rewriting (StrEnum → Literal), see [Doc 02](02_typing_and_pydantic.md) §3 or [_signature_annotation.py](../../src/digital_registrar/signatures/_signature_annotation.py).

---

## 4. `dspy.InputField` / `dspy.OutputField` — field metadata

These are markers that tell DSPy whether a field is input or output. The `desc=` parameter becomes the LM-facing description.

```python
report: list = dspy.InputField(desc="paragraphs of the pathology report")
grade: int | None = dspy.OutputField(desc="histologic grade (1-3) if reported")
```

Two things to know:

1. **`desc=`** lands in the rendered prompt. DSPy emits something like:
   ```
   ## Inputs:
   - report: paragraphs of the pathology report

   ## Outputs:
   - grade: histologic grade (1-3) if reported
   ```
   (The actual format varies by *adapter* — see §8.)

2. **The annotation matters.** `report: list` vs `report: str` produces different prompt rendering and different parsing. `grade: int | None` tells DSPy the LM can emit an integer or `null`.

You can also pass `prefix=` to override how the field name appears in the prompt, and `format=` for custom serialization. drr-next mostly relies on the defaults.

---

## 5. `dspy.Predict` — the simplest module

Wrap a Signature in `dspy.Predict` to get a callable that:
1. Renders the signature to a prompt.
2. Calls the configured LM.
3. Parses the response.
4. Returns a `Prediction` object whose attributes are the output fields.

```python
predictor = dspy.Predict(is_cancer)
pred = predictor(report=["Specimen...", "Microscopic..."])

pred.cancer_excision_report   # bool
pred.cancer_category          # "breast"
pred.cancer_category_others_description  # None or str
```

In `drr-next`:

[src/digital_registrar/pipeline_factory.py:90-95](../../src/digital_registrar/pipeline_factory.py#L90-L95):

```python
self.router = dspy.Predict(build_router_signature(CASE_MODELS))
self.jsonize = (
    dspy.Predict(build_jsonize_signature(CASE_MODELS))
    if jsonize_enabled
    else None
)
```

And [pipeline_factory.py:127](../../src/digital_registrar/pipeline_factory.py#L127):

```python
self._extractor_cache[organ] = [(s, dspy.Predict(s.signature)) for s in steps]
```

So every signature — router, jsonize, per-group extractors — is wrapped in `dspy.Predict`.

**Demos**: `dspy.Predict` instances have a `.demos` attribute (a list of dicts with worked examples). DSPy renders those into the prompt as few-shot examples. Today drr-next leaves this empty — every call is zero-shot. **This is exactly the lever Doc 5 proposes pulling.**

---

## 6. `dspy.Module` — compose multiple signatures

`dspy.Module` is the base class for a pipeline. You subclass it, build sub-modules in `__init__`, and define a `forward()` method that orchestrates them.

`drr-next`'s `CancerPipelineV2` in [pipeline_factory.py:60-271](../../src/digital_registrar/pipeline_factory.py#L60-L271):

```python
class CancerPipelineV2(dspy.Module):
    def __init__(self, *, decomposition="auto", ...):
        super().__init__()
        self._decomposition = decomposition
        ...
        self.router = dspy.Predict(build_router_signature(CASE_MODELS))
        self.jsonize = (
            dspy.Predict(build_jsonize_signature(CASE_MODELS))
            if jsonize_enabled else None
        )
        self._extractor_cache: dict[str, list] = {}

    def forward(self, report, logger, fname=""):
        paragraphs = self._normalize_report(report)

        # Step 1: router — is this a cancer case? which organ?
        rsp = self.router(report=paragraphs)
        if not rsp.cancer_excision_report:
            return {...}

        # Step 2 (optional): rough JSON preprocessing
        if self._jsonize_enabled and self.jsonize is not None:
            json_resp = self.jsonize(report=paragraphs, cancer_category=rsp.cancer_category)
            json_report = json_resp.output if isinstance(json_resp.output, dict) else {}

        # Step 3: per-group extraction
        extractors = self._get_extractors(rsp.cancer_category)
        for step, predictor in extractors:
            ...
            pred = predictor(report=step_input, report_jsonized=json_report)
            organ_data = dump_prediction_plain(pred)
            kept = {k: organ_data.get(k) for k in step.output_field_names if k in organ_data}
            out["cancer_data"].update(kept)

        return out
```

What `dspy.Module` gives you:
- A consistent `forward()` invocation pattern.
- DSPy can introspect the module's sub-modules (its `dspy.Predict` instances) for compilation / optimization passes. drr-next doesn't use this, but it's why the structure matters.
- Subclassing forces you to be explicit about which steps are LM calls.

**You can call any module by calling the instance**: `pipeline(report=..., logger=..., fname=...)` → routes to `forward()`. This is standard PyTorch-style.

---

## 7. `dspy.ChainOfThought` — implicit reasoning field

`dspy.ChainOfThought(SomeSignature)` is like `dspy.Predict(SomeSignature)` but inserts a `reasoning` output field before everything else. The LM "thinks aloud" first, then produces structured outputs.

```python
class GradeFromReport(dspy.Signature):
    report: str = dspy.InputField()
    grade: int | None = dspy.OutputField()

cot = dspy.ChainOfThought(GradeFromReport)
result = cot(report="...")
print(result.reasoning)   # the model's chain of thought
print(result.grade)       # the structured answer
```

**Why use CoT**: helps weaker models reason about hard cases (rare histologies, ambiguous staging). The trade-off is **longer prompts and slower runs**.

**`drr-next` does not use `dspy.ChainOfThought` in production.** It's in attic ablations only — old experiments that compared CoT to direct prediction. The current production wisdom is that for **schema-constrained extraction**, `Predict` is fine; the structure of the output field already encodes most of what CoT would coax out.

Knowing CoT exists is still useful: if you have a *truly* hard inference (e.g. "what's the AJCC stage given these observations"), CoT is the first thing to try.

---

## 8. Adapters: how DSPy turns a Signature into a prompt

This is the part that's most often invisible. DSPy has an *adapter layer* that decides:
- How the prompt is formatted (chat vs completion).
- Whether to emit JSON Schema (`response_schema`) to constrain the LM's output.
- How to parse the response.

The two main adapters in DSPy 3:
- **`ChatAdapter`** — uses chat-style messages, parses out structured fields by markdown headings.
- **`JSONAdapter`** — sends a `response_schema` (when the LM supports it) and parses the LM's JSON output directly.

When the LM `supports_response_schema=True` (OpenAI's structured-outputs, vLLM with guided decoding, some Ollama models), DSPy can constrain the LM to emit valid JSON matching the signature's schema. This is much more reliable than parsing prose.

drr-next's auto-decomposition heuristic consults this exact flag — [factory.py:139-148](../../src/digital_registrar/signatures/factory.py#L139-L148):

```python
def _choose_decomposition(schema, field_meta, model_profile):
    if model_profile in _SMALL_MODEL_PROFILES:
        return "per_group"
    if model_profile in _LARGE_MODEL_PROFILES:
        try:
            lm = dspy.settings.lm
            if lm is not None and getattr(lm, "supports_response_schema", False):
                return "monolithic"
        ...
```

Translated: if the LM can emit a single JSON object matching the full schema reliably, do that. Otherwise, split into per-group prompts because small models choke on 50+ fields at once.

You usually don't think about adapters — DSPy picks one automatically. But knowing the layer is there explains why the same signature can produce subtly different prompts on different models.

---

## 9. `dspy.Embedder` — embedding text into vectors

DSPy provides a wrapper around LiteLLM-resolvable embedding backends:

```python
import dspy

embedder = dspy.Embedder("openai/text-embedding-3-small")
# or
embedder = dspy.Embedder("sentence-transformers/all-MiniLM-L6-v2")

vectors = embedder(["first piece of text", "second piece of text"])
# vectors is a (2, D) array
```

`drr-next` uses this for **chunk routing** in [src/digital_registrar/chunking/dspy_embedder_router.py:49-67](../../src/digital_registrar/chunking/dspy_embedder_router.py#L49-L67):

```python
def __init__(
    self,
    *,
    model: str | None = None,
    embedder=None,
    top_k: int = 2,
    min_similarity: float = 0.25,
) -> None:
    if model is None and embedder is None:
        raise ValueError("DSPyEmbedderRouter requires either `model` or `embedder`")
    ...
    if embedder is not None:
        self._embedder = embedder
    else:
        import dspy
        self._embedder = dspy.Embedder(model)
```

The router embeds each chunk and each extraction-group description, then assigns the top-K closest groups to each chunk by cosine similarity. See [Doc 3](03_chunking.md) for chunking specifics; for embeddings the key facts are:

- **Same interface across providers.** Swap `"openai/text-embedding-3-small"` for `"sentence-transformers/all-MiniLM-L6-v2"` and nothing else changes.
- **You can pass a custom callable.** The `embedder=` kwarg accepts anything that maps `list[str] -> 2D ndarray`. Useful for tests with stubbed embeddings.
- **No automatic batching, caching, or persistence.** Every call hits the embedder. If you embed the same text twice, you pay twice. (Doc 5 proposes a persistent demo store on top of this.)

---

## 10. `dspy.Tool` — wrap a function for LM-callable use

A `dspy.Tool` is a thin wrapper around a regular function. It carries:
- A `name` (what the LM sees and refers to).
- A `desc` (when to use it, what args it takes).
- The function itself.

`drr-next` defines two in [src/digital_registrar/staging/tools.py:8-31](../../src/digital_registrar/staging/tools.py#L8-L31):

```python
stage_from_observations_tool = dspy.Tool(
    func=stage_from_observations,
    name="cancer_stage_from_observations",
    desc=(
        "Compute the AJCC TNM stage from extracted observations. "
        "Args: organ (str, e.g. 'lung'), edition (str like 'AJCC 9' or 'AJCC 8'), "
        "observations (dict mapping observable name to value; strings for enums, "
        "numbers for numeric observables, bools for booleans), and pass-through "
        "kwargs ..."
    ),
)

observable_schema_tool = dspy.Tool(
    func=observable_schema,
    name="cancer_observable_schema",
    desc=(
        "Look up the input schema for an (organ, edition) pair before extracting..."
    ),
)

STAGING_TOOLS = [stage_from_observations_tool, observable_schema_tool]
```

A Tool alone does nothing. You feed a list of Tools to a module that **uses** them — that's what `dspy.ReAct` is for.

**`drr-next` uses Tools only in the demo.** The core extraction pipeline does not call tools.

---

## 11. `dspy.ReAct` — reasoning + acting (an agent loop)

`dspy.ReAct` is an agent that, in a loop, decides whether to call a tool or produce a final answer. The acronym is **Re**asoning + **Act**ion.

The mental model:
1. Read the signature inputs.
2. **Think** out loud (similar to chain-of-thought).
3. Either call a tool with arguments, or emit the final outputs.
4. If a tool was called, read its result, go back to step 2.
5. Cap at `max_iters` so the agent doesn't loop forever.

`drr-next`'s demo in [examples/staging_demo/react_agent.py](../../examples/staging_demo/react_agent.py):

```python
class StageFromReport(dspy.Signature):
    """Derive the AJCC TNM stage for the cancer described in the report.

    First call ``cancer_observable_schema`` with the given organ and
    edition to learn which observations the staging engine expects...
    Then read those observations off the report... Finally call
    ``cancer_stage_from_observations`` with the values you found and
    return its ``stage`` field as your output.
    """

    report: str = dspy.InputField(desc="A pathology report describing one cancer case.")
    organ: str = dspy.InputField(desc="The cancer organ, e.g. 'breast'.")
    edition: str = dspy.InputField(desc="The AJCC edition string, e.g. 'AJCC 8'.")
    stage: str = dspy.OutputField(desc="The derived stage group (e.g. 'IIA').")
    reasoning: str = dspy.OutputField(
        desc="One short sentence on which observations drove the stage assignment."
    )

def main():
    autoconf_dspy(MODEL)
    agent = dspy.ReAct(StageFromReport, tools=STAGING_TOOLS, max_iters=6)
    result = agent(report=REPORT, organ="breast", edition="AJCC 8")
    print(result.stage, result.reasoning)
    dspy.inspect_history(n=1)    # see the full trajectory
```

Each iteration emits a tuple of "thought / action / observation" or "thought / final answer". The signature's instructions hint at the *expected sequence* of tool calls.

**Why drr-next doesn't use ReAct in production extraction**: the extraction task is *not* agentic. The report is fixed input, the schema is fixed output, and there are no external systems to query. `dspy.Predict` (or per-group `Predict`s) maps directly onto that.

**When ReAct shines** (and why it's in the staging demo):
- The task needs a **tool with a learnable interface** (e.g. consult a schema before extracting).
- Multiple rounds of reasoning + grounding (e.g. retrieve, then validate, then refine).
- The model can decide *which* tool to call based on context.

A reasonable future direction is wrapping the validation step in a ReAct agent that, given an extracted record, decides whether to re-query the LM for clarification on conflicting fields. That's speculative.

---

## 12. `dspy.Prediction` — the result object

When you call `dspy.Predict(...)(input1=..., input2=...)`, you get back a `dspy.Prediction`. It's an object whose attributes are the output fields:

```python
result = predictor(report=["..."])
result.cancer_category               # "breast"
result.cancer_excision_report        # True
result.toDict()                      # {"cancer_category": "breast", "cancer_excision_report": True}
```

drr-next has a utility [src/digital_registrar/util/predictiondump.py](../../src/digital_registrar/util/predictiondump.py) — `dump_prediction_plain(pred)` — that converts a Prediction to a flat dict so the pipeline can `update()` its output dict:

```python
# in pipeline_factory.py:258-263
pred = predictor(report=step_input, report_jsonized=json_report)
organ_data = dump_prediction_plain(pred)
kept = {k: organ_data.get(k) for k in step.output_field_names if k in organ_data}
out["cancer_data"].update(kept)
```

The `kept` filter is because each per-group extractor might echo input field names back (e.g. `report` shows up in the Prediction too). drr-next only keeps the *output* fields the step is responsible for.

---

## 13. Inspecting what DSPy actually did

When something goes wrong, you want to see the actual prompt and response. DSPy offers:

```python
dspy.inspect_history(n=1)
```

Prints the last `n` LM calls (prompt + response) to stdout. **Use this aggressively** when debugging signatures — most prompt issues are obvious the moment you see the rendered output.

The staging demo calls `dspy.inspect_history(n=1)` at the end of its run for exactly this reason.

---

## 14. Things drr-next does NOT use (yet)

### `dspy.Retrieve` and `dspy.ColBERTv2`

DSPy ships built-in retrievers. `dspy.Retrieve(retriever)` is a module that takes a query and returns top-K passages. ColBERTv2 is a specific retriever DSPy ships out of the box.

drr-next has **no retrieval** today (see Doc 5 for what a retrieval layer would look like).

### Compilation / teleprompters

DSPy can *optimize* prompts and demonstrations against a dataset. The optimizers are called "teleprompters" (e.g. `dspy.BootstrapFewShot`, `dspy.MIPROv2`). You give them:
- A training set of (input, gold-output) pairs.
- A metric function.
- A module to compile.

…and they rewrite the module's demos / instructions to maximize the metric.

drr-next has **attic experiments** with compilation (`attic/ablations_scripts/compile_dspy.py`) but no production use. Compiling requires a curated labeled set, which the project doesn't have at scale yet.

This is **directly adjacent to Doc 5's few-shot demo retrieval proposal**. If you build the demo store described there, the next step up is using teleprompters to optimize *which* demos to include for which queries.

---

## 15. Gotchas

### `dspy.configure(lm=...)` is global

If you call `dspy.configure(lm=lm_a)` and then `dspy.configure(lm=lm_b)`, all subsequent calls use `lm_b`. There's no per-thread isolation by default. In a multi-tenant service this matters.

Per-call override:
```python
result = predictor(input=..., lm=other_lm)
```

### LMs are cached unless you say otherwise

By default DSPy caches LM responses by (prompt, kwargs). Same prompt → same response. drr-next disables this with `"cache": False` in [models/common.py:67](../../src/digital_registrar/models/common.py#L67):
```python
_BASE_KWARGS = {"repeat_penalty": 1.05, "keep_alive": "30m", "cache": False, "seed": 10}
```

If you turn caching back on, expect surprising determinism in test runs.

### Output field types are not enforced — they're *parsed*

If you declare `grade: int | None = dspy.OutputField(...)` and the LM emits `"grade": "two"`, DSPy will try to parse — and might give you `None`, or raise a parse error, or silently coerce to something weird, depending on the adapter. Always validate at the boundary with `case_model.model_validate(...)` (see [pipeline_factory.py:130-139](../../src/digital_registrar/pipeline_factory.py#L130-L139)).

### `make_signature` requires `custom_types` for nested BaseModels

Already covered in [Doc 1 §13](01_class_machinery.md#13-gotchas) — repeating because it's easy to miss. If your signature has output fields with nested `BaseModel` types (e.g. `list[BreastMargin]`), you must pass `custom_types={"BreastMargin": BreastMargin}` or DSPy's schema serializer fails. The factory does this for you; if you write a signature by hand, you do it yourself.

### The DSPy class-syntax form's class name shows up in errors

If you wrote `class is_cancer(dspy.Signature):`, error messages will say "is_cancer expected output field cancer_category to be one of …". Helps debugging but also means signature names ARE part of the interface.

### ReAct can loop

Set `max_iters` (drr-next's demo uses 6). The LM can get into a tool-call → think → tool-call cycle. Always cap.

### Tools' `desc` matters

The LM picks tools by reading their `desc`. If your tool's description is vague, the LM won't call it; if multiple tools have similar `desc`s, the LM may pick the wrong one. Make descriptions specific and disjoint.

---

## 16. Putting it together

The full extraction flow in drr-next:

```
load_model("gpt") → dspy.LM("ollama_chat/gpt-oss:20b", ...)
dspy.configure(lm=that_lm)

CancerPipelineV2.__init__()
  self.router  = dspy.Predict(build_router_signature(CASE_MODELS))
  self.jsonize = dspy.Predict(build_jsonize_signature(CASE_MODELS))  [optional]
  # extractors built lazily

CancerPipelineV2.forward(report=..., ...)
  1. router(report=paragraphs) → cancer_category
  2. (optional) jsonize(report=paragraphs, cancer_category=...) → rough JSON
  3. extractors = build_extraction_signatures(...) wrapped in dspy.Predict
     for each step:
       step_input = full report OR routed chunk texts (Doc 3)
       pred = predictor(report=step_input, report_jsonized=json_report)
       merge pred fields into cancer_data
  4. (optional) CASE_MODELS[organ].model_validate(cancer_data)  [degraded mode]
```

Every LM call goes through `dspy.Predict(...)`. Every Predict has a Signature behind it. Every Signature was built either by hand (legacy) or by the factory (dynamic). That's the whole story.

For forward-looking DSPy use in this project, see [Doc 5](05_few_shot_demo_retrieval.md) — it's a concrete proposal for using `.demos` on the existing Predicts to add retrieval-augmented few-shot learning without rebuilding anything.
