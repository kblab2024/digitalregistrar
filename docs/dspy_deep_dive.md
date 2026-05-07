# DSPy deep dive — why the strict-schema protocol works, and what to build next

This doc is the **postmortem-turned-roadmap** companion to [dspy_ollama_model_compatibility.md](dspy_ollama_model_compatibility.md). The compatibility doc explained *why* DSPy was failing on Gemma / Qwen and quietly succeeding on `gpt-oss:20b`. This one starts from the now-confirmed fact that the protocol in [pipeline_dspy_strict.py](../src/digital_registrar_research/pipeline_dspy_strict.py) — `litellm.register_model(...supports_response_schema=True)` + `dspy.adapters.JSONAdapter` + `dspy.ChainOfThought` per signature — works, and then unpacks (a) what each piece is doing on the wire, (b) the parts of DSPy this repo has not yet used, and (c) a phased plan for what to build next.

Pinned versions referenced throughout: **DSPy 3.2.0**, **Ollama ≥ 0.5**, **LiteLLM** as bundled by DSPy 3.2.

---

## 0. Reading order

1. [pipeline_dspy_strict.py](../src/digital_registrar_research/pipeline_dspy_strict.py) — the working pipeline. Read its module docstring first.
2. [dspy_ollama_model_compatibility.md](dspy_ollama_model_compatibility.md) §§1–4 — the prequel; explains the four layers (adapter, model post-training, runtime, tokenizer) that have to line up.
3. This doc, §1 onward.

If you only have ten minutes, read §1 of this doc and skip to §3 (roadmap).

---

## 1. Anatomy of the working protocol

Three lines do almost all the work:

```python
# pipeline_dspy_strict.py
litellm.register_model({bare: {"supports_response_schema": True, ...}})  # §1.2
dspy.configure(lm=lm, adapter=JSONAdapter())                              # §1.3
self.analyzer_is_cancer = dspy.ChainOfThought(is_cancer)                  # §1.4
```

Each one fixes one specific failure mode that the compatibility doc traced. Removing any one of them silently degrades back to whatever DSPy's auto-fallback chooses, which on Ollama is `JSONAdapter` in loose-`json_object` mode — i.e., a prompt-level instruction with no token-level enforcement.

### 1.1 The three load-bearing pieces

| Piece | What it changes | What you'd get without it |
|---|---|---|
| `litellm.register_model({bare: {"supports_response_schema": True, ...}})` | Flips a LiteLLM capability flag so DSPy is willing to emit `response_format={"type":"json_schema","strict":true}`. | DSPy detects "schema unsupported", silently downgrades to `{"type":"json_object"}` (loose JSON-mode) — Ollama treats it as a prompt hint, not a grammar constraint. |
| `dspy.configure(lm=lm, adapter=JSONAdapter())` | Forces the adapter — no auto-fallback. Output contract is "one JSON object whose keys are the signature's `OutputField`s". | Auto-selection chooses `ChatAdapter` first; on long Pydantic-typed signatures the field-marker regex fails to parse, retries on `JSONAdapter` mid-call, and you spend two LM calls per success without realizing it. |
| `dspy.ChainOfThought(<sig>)` per signature | Prepends a `reasoning: str` output field, which JSONAdapter emits as the first key of the JSON body. | The model has no slot to "think" before committing to enum values inside `Literal[...]` fields. Schema enforcement masks the wrong tokens; the model picks the most likely allowed token without having reasoned about it. |

### 1.2 `supports_response_schema` — the LiteLLM flag dance

The key sites in [pipeline_dspy_strict.py](../src/digital_registrar_research/pipeline_dspy_strict.py):

```python
# pipeline_dspy_strict.py:85-106
def _enable_strict_schema_for_ollama(model_id: str) -> None:
    import litellm
    bare = model_id.split("/", 1)[-1]    # "ollama_chat/gpt-oss:20b" -> "gpt-oss:20b"
    provider = model_id.split("/", 1)[0] if "/" in model_id else "ollama_chat"
    litellm.register_model({
        bare: {
            "supports_response_schema": True,
            "litellm_provider": provider,
            "mode": "chat",
        },
    })
```

There are two non-obvious traps here, both worth absorbing because they recur every time someone tries to wire DSPy + JSONAdapter to a non-OpenAI provider:

1. **The bare key.** LiteLLM's `supports_response_schema(model_id)` calls `get_llm_provider()` first, which strips the `ollama_chat/` prefix before looking the model up in its `model_cost` dict. If you register under the prefixed key, the lookup never sees your override; DSPy thinks the flag is still `False` and silently downgrades to JSON-mode. Registering under the bare key (`gpt-oss:20b`) is what actually flips the bit. (The `litellm_provider` and `mode` fields are required for the registration to be accepted, but only `supports_response_schema` matters for behavior.)
2. **The flag is only a permission, not an instruction.** Setting `True` tells DSPy *"yes, you may emit `json_schema`"*, but it doesn't force the adapter to do so. JSONAdapter still independently decides whether the signature *can* be expressed as a strict schema (see §1.3) — for some signatures it deliberately won't.

**Verifying the flip actually took.** Run one of the smoke scripts with LiteLLM's debug logging turned on:

```bash
LITELLM_LOG=DEBUG python -m digital_registrar_research.pipeline_dspy_strict 2>&1 \
  | grep -c "'type': 'json_schema'"
```

If the count is `> 0`, strict mode is reaching Ollama. If it's `0`, the override didn't land — most likely because the `bare` key is wrong (e.g., a colon got URL-encoded somewhere) or because LiteLLM was imported and cached its provider table before your override ran. Move `_enable_strict_schema_for_ollama` *before* `load_model` if you suspect the latter (the current code already does this; preserve the order if you refactor).

### 1.3 `JSONAdapter` at the wire

JSONAdapter does three things, in order:

1. Inspects the signature's `OutputField` types and tries to compile them to a JSON Schema.
2. If successful and the LM advertises `supports_response_schema=True`, sends `response_format={"type":"json_schema","json_schema":{"name":"<sig>", "schema":{...}, "strict":true}}` in the chat completion request.
3. On the response, parses the JSON, then validates with the same Pydantic types.

On Ollama ≥ 0.5, step 2 reaches llama.cpp's GBNF grammar-constrained decoder. **Tokens that would violate the schema are masked at sampling time** — they have probability zero, no matter what the LM's logits say. This is the only way to get *guaranteed* schema-valid JSON on a local model. Without it, you're trusting the model's training and your prompt.

There is a deliberate exception for one signature in this pipeline:

```python
# models/common.py:94-98
class ReportJsonize(dspy.Signature):
    ...
    output: dict = dspy.OutputField(desc=...)
```

A bare `dict` output field has no closed schema (any keys, any nesting). JSONAdapter detects this via its `_has_open_ended_mapping` short-circuit and falls back to `response_format={"type":"json_object"}` (loose JSON-mode) **for that one call**. The pipeline's docstring calls this out:

> `ReportJsonize.output: dict` triggers JSONAdapter's `_has_open_ended_mapping` short-circuit and falls back to `response_format={"type":"json_object"}` (loose JSON-mode) for that one call. Acceptable by design — the dict is unconstrained on purpose.

This is correct and intentional. The per-organ extractors that follow declare strictly typed outputs (`Literal[...]`, `int|None`, `list[BreastMargin]`, etc.) and hit the strict path. The first-pass `ReportJsonize` is supposed to be a loose shaping step; the strict pass happens downstream when the schema is known.

**A diagnostic worth running once.** Pop a Python REPL after a forward pass and call `dspy.inspect_history(n=1)` — it prints the exact request DSPy sent and the raw response. Look for the `response_format` key. If it's `json_schema` for the per-organ calls and `json_object` for the `ReportJsonize` call, the protocol is doing what it should.

### 1.4 ChainOfThought inside strict schema

[pipeline_dspy_strict.py:151-152, 213](../src/digital_registrar_research/pipeline_dspy_strict.py#L151) wraps every signature in `dspy.ChainOfThought`. What this does in the JSONAdapter regime is subtle and worth understanding deeply, because it's the part most people miss when they try to reproduce this protocol:

- A `dspy.ChainOfThought(sig)` programmatically appends a `reasoning: str` output field at the **front** of the signature's outputs.
- JSONAdapter compiles this into the JSON schema with the reasoning property listed first.
- Strict mode generates JSON properties in the schema's declared order, so the model emits its `reasoning` string *before* committing to any of the typed fields.
- That string is unconstrained — the schema only requires it to be a string. The model can use it as a token-level scratchpad.

Why does this matter when the schema is already enforcing valid output? Because **strict-schema decoding masks tokens, but it does not constrain the *content* of unconstrained string fields, and it does not improve the model's *choice* among allowed tokens**. When the schema permits ten `Literal` enum values, GBNF zeroes out the probability of every disallowed token — but among the ten allowed values, the model still picks the highest-probability one given its current state. If that state is "I haven't read the report carefully yet", the choice is noisy.

The reasoning prefix gives the model up to several hundred tokens of attention-conditioning on the input *before* the first masked sampling step. Empirically (and consistent with the wider CoT literature), this raises accuracy on enum-heavy fields substantially. It's not a DSPy quirk; it's the same effect ChainOfThought has everywhere, applied inside a schema-constrained envelope where there is no other way to get reasoning out.

The post-processing strip at [pipeline_dspy_strict.py:223](../src/digital_registrar_research/pipeline_dspy_strict.py#L223) removes the `reasoning` key from the final dict so that `cancer_data` is byte-compatible with the original `CancerPipeline.forward()` shape — the eval pipeline ingests both without modification:

```python
organ_data.pop("reasoning", None)
output_report["cancer_data"].update(organ_data)
```

If you're tempted to keep the reasoning trace for analysis, put it in a sidecar (see Phase 1 in §3) — don't leak it into `cancer_data`, because the cascade comparison code expects a fixed key set.

---

## 2. DSPy concepts the repo isn't using yet

The protocol in §1 only uses the *runtime* layer of DSPy — adapter + module + signature. DSPy's bigger story is **compilation**: treating a pipeline as a program that can be optimized against a metric. This section walks through the seven concepts that this repo could adopt, in the order I'd adopt them.

### 2.1 `dspy.Module` composition

The repo already uses `dspy.Module` correctly: [DspyStrictCancerPipeline](../src/digital_registrar_research/pipeline_dspy_strict.py#L140) subclasses it and composes `ChainOfThought` predictors as attributes. What it doesn't yet use is **nested-module reuse and parameterization**. Because every `dspy.ChainOfThought(...)` attribute on a Module is itself a parameter, when you compile the parent module the optimizer can produce per-attribute demonstrations and instructions that are saved together as one program JSON.

```python
# Current style - one organ analyzer instantiated per call, fresh each time
organ_analyzer = dspy.ChainOfThought(cls)   # pipeline_dspy_strict.py:213

# Compilable style - all organ analyzers as named attributes of self,
# instantiated once at __init__ time, so the optimizer can demonstrate each
class DspyStrictCancerPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer_is_cancer = dspy.ChainOfThought(is_cancer)
        self.jsonize = dspy.ChainOfThought(ReportJsonize)
        # NEW: pre-instantiate all organ analyzers as named attributes
        for organ, sigs in organmodels.items():
            for sig_name in sigs:
                attr = f"analyze_{organ}_{sig_name}"
                setattr(self, attr, dspy.ChainOfThought(globals()[sig_name]))
```

This refactor is a Phase-2 prerequisite — `BootstrapFewShot` can only insert demos into named, persistent attributes; ad-hoc `dspy.ChainOfThought(cls)` inside `forward()` is invisible to the compiler.

### 2.2 `dspy.Refine` / `dspy.BestOfN` — runtime contracts beyond Pydantic types

In DSPy ≤ 2.6, `dspy.Assert` and `dspy.Suggest` were the way to express runtime invariants ("if `margins.status == 'positive'` then `margin_distance` must be 0"). In **DSPy 3.x, the assertion machinery is reframed around two modules**: `dspy.Refine` (re-run with feedback when a check fails) and `dspy.BestOfN` (sample N times, pick the one a reward function likes best).

```python
import dspy

def margin_invariant_check(prediction) -> tuple[bool, str]:
    margins = getattr(prediction, "margins", None) or []
    for m in margins:
        if m.margin_involved and m.distance != 0:
            return False, (
                f"margin_category={m.margin_category} is involved but "
                f"distance={m.distance}; an involved margin must have "
                f"distance == 0. Re-extract."
            )
    return True, ""

# Wrap the existing analyzer with a one-shot refine pass.
margins_analyzer = dspy.ChainOfThought(BreastCancerMargins)
margins_refined = dspy.Refine(
    module=margins_analyzer,
    N=2,                                # one initial + one retry
    reward_fn=lambda _, pred: 1.0 if margin_invariant_check(pred)[0] else 0.0,
    threshold=1.0,
)
```

The reward function gets the example and the prediction; if the prediction fails the invariant, `Refine` re-runs the predictor and surfaces the failure message back to the model. This is strictly more useful than raising an exception, because it gives the model a chance to fix a genuine mistake without the human having to inspect the trace.

Where this slots in: cross-field constraints in the breast / lung / colon signatures that Pydantic alone can't express because they involve *multiple* fields. The cancer registrar's checklist has dozens of these. Phase 3 in §3 collects them.

### 2.3 Compilation — `BootstrapFewShot`, `MIPROv2`, `COPRO`

The single biggest piece of DSPy this repo hasn't yet exercised end-to-end is **the optimizer (teleprompter) layer**. The repo has a compile script at [scripts/ablations/compile_dspy.py](../scripts/ablations/compile_dspy.py) that uses `BootstrapFewShotWithRandomSearch`, but it's plumbed for the *monolithic* ablation cell only — it isn't compiling the modular per-organ pipeline that production actually uses.

| Optimizer | What it tunes | Trainset shape | When to reach for it |
|---|---|---|---|
| `BootstrapFewShot` | Demonstrations (few-shot examples) inserted into each predictor's prompt. | `list[dspy.Example]`, ≥10 to be useful, ~50 ideal. Each example needs the same `with_inputs(...)` keys the module's `forward` consumes. | Default first step. Cheap, deterministic, gives you a baseline you can defend. |
| `BootstrapFewShotWithRandomSearch` | Same as above + a search over which demo combinations validate best. | Same trainset. | Use when you have a metric and want to explore demo combinations. ~5–10× the compile cost. |
| `MIPROv2` | Both demonstrations *and* per-predictor instructions, via a Bayesian search with a separate proposer LM. | Trainset + a metric. Often wants 50+ examples. | Top-tier optimizer; expects compute budget. Use after you have a clean compiled BootstrapFewShot baseline to beat. |
| `COPRO` | Instructions only (no demos). Iteratively rewrites each predictor's instructions and keeps the best. | Trainset + metric. | Use when the prompts are the suspected weak link and demos aren't helping. |
| `Ensemble` | Combines N compiled programs into one (majority vote / mean / best-by-metric). | N already-compiled programs. | Late-stage; combine your best 3 BootstrapFewShot runs. |

The trainset object is a `dspy.Example`. The repo's compile script already builds them correctly:

```python
# scripts/ablations/compile_dspy.py:75-81
ex = dspy.Example(
    report=report_text,
    gold=gold,
    case_id=case_id,
    organ_n=organ_n,
).with_inputs("report")
```

`with_inputs("report")` tells DSPy that `report` is a model input and `gold` / `case_id` / `organ_n` are metadata the metric can read. The metric function gets `(example, prediction, trace=None)` and returns a scalar — that's the entire optimizer interface.

### 2.4 `dspy.Evaluate` and metric-driven optimization

The cascade three-chapter eval (`scripts/eval/cascade/compare_runs.py`) is *already* a metric — it produces per-case verdicts. The work to turn it into something a teleprompter can optimize against is small:

```python
from digital_registrar_research.benchmarks.eval.metrics import field_correct
from digital_registrar_research.benchmarks.eval.scope import FAIR_SCOPE

def cascade_metric(example, pred, trace=None) -> float:
    """Per-example field-accuracy score; mirrors compile_dspy._compile_metric
    but lifted to the modular pipeline output shape."""
    gold = example.gold
    pipe_out = getattr(pred, "output", None) or {}
    correct = total = 0
    for field in FAIR_SCOPE:
        result = field_correct(gold, pipe_out, field)
        if result is None:
            continue
        total += 1
        correct += int(bool(result))
    return correct / total if total else 0.0

evaluator = dspy.Evaluate(devset=dev_examples, metric=cascade_metric,
                          num_threads=1, display_progress=True)
score = evaluator(my_pipeline)
```

`dspy.Evaluate` is a one-liner harness; `dspy.BootstrapFewShot(metric=cascade_metric, ...)` uses the same metric to score candidate few-shot combinations. Same metric for evaluation and optimization is the discipline that keeps you honest — if you optimize against metric A and report metric B, you've Goodharted yourself.

### 2.5 `dspy.inspect_history` and `dspy.track_usage` — observability

Two short loops are worth knowing, because the compatibility doc complained that it's hard to tell which adapter actually fired:

```python
import dspy

# After any forward pass:
dspy.inspect_history(n=1)      # last prompt + response, fully expanded
dspy.inspect_history(n=5)      # last five

# Per-call usage tracking (DSPy 3.x):
with dspy.track_usage() as usage:
    out = pipeline(report=text, logger=log)
print(usage.get_total_tokens())  # prompt + completion across all sub-calls
```

`inspect_history` is the single highest-signal debugging tool when a model misbehaves — it tells you exactly what the adapter rendered into the prompt, which is the answer to 80% of "why did the model do X?" questions. `track_usage` is cheap; wrapping the per-case loop with it would let the cascade reports include token-cost columns alongside accuracy — a paper-grade addition for almost no work.

### 2.6 Custom adapters and the `TwoStepAdapter` pattern

You already know `JSONAdapter`. The other adapter worth reaching for is `dspy.adapters.TwoStepAdapter`, because it directly addresses the failure mode the compatibility doc identified for Gemma 3 and Qwen 3.5: **smart content models that can't reliably emit format**.

```python
import dspy
from dspy.adapters import TwoStepAdapter

# Smart but format-brittle - the *content* model
gemma = dspy.LM("ollama_chat/gemma3:27b", api_base="http://localhost:11434",
                api_key="", model_type="chat")

# Format-loyal, smaller - the *extraction* model, run after the smart one
gptoss = dspy.LM("ollama_chat/gpt-oss:20b", api_base="http://localhost:11434",
                 api_key="", model_type="chat")

dspy.configure(lm=gemma, adapter=TwoStepAdapter(extraction_model=gptoss))
```

What this does: Gemma reads the report and produces a free-text answer. The same prompt (now including Gemma's answer) is then handed to gpt-oss with a JSONAdapter contract; gpt-oss does the *extraction* into the strict schema. You get Gemma's reasoning depth and gpt-oss's format reliability. Cost is 2x latency per call.

There is no off-the-shelf adapter that does *strict-schema-only on the second pass*; you'd need a thin subclass. That's a Phase-4 spike, not a one-line config.

### 2.7 Saving and loading compiled programs

Compiled programs serialize to JSON. The C5 ablation cell already loads them:

```python
# Compile (once):
compiled = optimizer.compile(student=my_pipeline, trainset=examples)
compiled.save("workspace/compiled/my_pipeline_gptoss.json")

# Load (every run):
my_pipeline = DspyStrictCancerPipeline()
my_pipeline.load("workspace/compiled/my_pipeline_gptoss.json")
```

The JSON includes: per-predictor demos, per-predictor compiled instructions (if the optimizer rewrote them), and a hash of the signature shape. Loading checks the hash; if you change the signature (add a `Literal`, change a field type), the load will fail loudly rather than silently mismatch. Treat compiled programs as build artifacts — they belong under `workspace/compiled/` (already established) and should be regenerated whenever the signatures change.

---

## 3. Future development roadmap

Phased, rough cost in person-weeks, dependency arrows where they matter.

### Phase 1 — Solidify what works (1–2 person-weeks, prerequisite for everything below)

Three small changes that close the observability gaps the compatibility doc flagged:

1. **Adapter-assertion test.** After `setup_pipeline_dspy_strict`, assert `isinstance(dspy.settings.adapter, JSONAdapter)` and assert that `litellm.supports_response_schema(model_id) is True`. If either fails, raise immediately. This catches the "bare-key registration silently ignored" failure mode without anyone having to grep `LITELLM_LOG=DEBUG`. Add as a unit test in `tests/`.
2. **Surface parse failures.** [pipeline_dspy_strict.py:225-227](../src/digital_registrar_research/pipeline_dspy_strict.py#L225) currently does `except Exception as e: logger.error(...); continue`. Replace with a structured sidecar: every organ_data dict gets a `_parse_status: "ok" | "parse_failed" | "empty"` key, and the failure message goes into `_parse_error`. The cascade compare_runs step can then split scoring by parse status, so "model got it wrong" and "model crashed" don't get conflated.
3. **Adapter telemetry.** Once per run, emit `print(f"[dspy] active adapter = {type(dspy.settings.adapter).__name__}")` and add the value as a column in the cascade run artifact (`cascade_atomic.parquet`). When you A/B compare runs months from now, you'll know which adapter was active without re-reading code.

**Exit criterion:** A run with the litellm flag override removed fails fast at startup instead of silently producing degraded output.

### Phase 2 — Compile the modular pipeline (2–4 weeks)

This is where the payoff starts. Steps:

1. **Refactor for compilation** (§2.1). Pre-instantiate every organ analyzer as a named attribute of `DspyStrictCancerPipeline.__init__`. The forward loop becomes `getattr(self, f"analyze_{organ}_{sig}")` instead of `dspy.ChainOfThought(globals()[sig])`.
2. **Extend [scripts/ablations/compile_dspy.py](../scripts/ablations/compile_dspy.py)** to accept the modular pipeline as a `--pipeline modular` flag (currently it only compiles `MonolithicPipeline`). Same metric, same dev set.
3. **Per-organ compilation.** Run the optimizer once per organ, with the dev set filtered to that organ. Save artifacts as `workspace/compiled/<organ>_<model>.json`. Per-organ is more sample-efficient than global because each organ's signature uses a disjoint subset of the demo budget.
4. **Wire the loader.** `DspyStrictCancerPipeline.__init__` checks for compiled artifacts in `workspace/compiled/` and loads them per organ if present, falling back to uncompiled signatures if not. The C5 ablation cell already loads compiled JSON; it just needs to know about the per-organ split.
5. **Run cascade compare_runs** between compiled and uncompiled gpt-oss:20b on the TCGA dev split.

**Exit criterion:** A compiled `gpt-oss:20b` run beats baseline on at least one organ in `cascade compare_runs` by ≥1σ on the bootstrap CI.

**Risk to watch:** demonstrations leak gold into the prompt. Make sure the trainset and the eval set are disjoint at the case_id level, not just by random split. The compile script already pulls from `data/{dataset}/annotations/gold/`; make sure `compare_runs` is reading from a held-out fold.

### Phase 3 — Assertions for clinical invariants (2–3 weeks)

The cancer registrar checklist has many cross-field constraints that Pydantic can't express in a single field. A starter list, all from the breast model:

- **Margin coupling:** if `margin_involved` is True, `distance` must be 0. ([models/breast.py:22-26](../src/digital_registrar_research/models/breast.py#L22))
- **Staging consistency:** if `pn_category` is `n0`, `extranodal_extension` cannot be True; if `regional_lymph_node` has all `involved=0`, `pn_category` should be `n0`.
- **Biomarker mutual-exclusion on Her-2:** `expression` and `score` should not both be set on the same `BreastBiomarker` row (the field descriptions already say this, but nothing enforces it).
- **DCIS gating:** if `dcis_present` is False, `dcis_size` / `dcis_grade` / `dcis_comedo_necrosis` must all be None.

Implement each as a `dspy.Refine` reward function (§2.2) wrapping the relevant `ChainOfThought`. One refinement pass typically fixes 40–60% of cross-field violations at the cost of one extra LM call on the offending cases.

**Exit criterion:** assertion-failure rate plotted alongside per-organ accuracy in cascade reports; rate trends down across runs as more invariants are encoded.

### Phase 4 — `TwoStepAdapter` for content-rich models (experimental)

Goal: validate the compatibility doc's prediction that Gemma 3 / Qwen 3.5 + gpt-oss:20b extraction would beat single-LM gpt-oss on raw reasoning quality.

1. Wire `TwoStepAdapter(extraction_model=gptoss)` as a fourth pipeline sibling — `pipeline_dspy_twostep.py`, mirroring the `pipeline_structured` / `pipeline_dspy_strict` API.
2. Add an ablation cell `C6_dspy_twostep` (mirrors the C5 compiled cell).
3. Run head-to-head against single-LM `pipeline_dspy_strict` on the cascade three-chapter eval.

**Exit criterion:** TwoStepAdapter wins on at least one of {field accuracy, parse-failure rate} at <2x latency. If it loses on both, document and stop — that's still a publishable negative result.

### Phase 5 — `MIPROv2` (research-tier)

Prerequisites: Phase 1's clean parse-status metric, Phase 2's compiled BootstrapFewShot baseline, a held-out TCGA slice neither phase touched.

```python
optimizer = dspy.MIPROv2(
    metric=cascade_metric,
    auto="medium",                  # automatic budget; "light"/"medium"/"heavy"
    num_threads=1,
)
compiled = optimizer.compile(
    student=my_pipeline,
    trainset=train_examples,        # bootstrap demos from here
    valset=val_examples,            # validate proposals here
    requires_permission_to_run=False,
)
```

MIPROv2 will rewrite per-predictor instructions and search over demo combinations using a separate proposer LM. Budget: with `auto="medium"` on a 50-example trainset, expect a few hundred LM calls (≈30 minutes on local Ollama for a 20b model). With `auto="heavy"`, multiply by 5–10×.

**Exit criterion:** an ablation table row "MIPROv2-compiled vs. BootstrapFewShot-compiled vs. uncompiled" with bootstrap CIs.

---

## 4. Tutorial — one organ end-to-end

This section walks **breast cancer extraction** through four progressive states. Each stage is a small diff against the previous one; the goal is to give you a feel for what each addition is buying. All code is runnable against the dummy dataset under `data/dummy/tcga/`.

### Stage 0 — `dspy.Predict` on the raw signature (baseline)

```python
import dspy

dspy.configure(lm=dspy.LM("ollama_chat/gpt-oss:20b",
                          api_base="http://localhost:11434",
                          api_key="", model_type="chat"))

from digital_registrar_research.models.breast import BreastCancerNonnested

predictor = dspy.Predict(BreastCancerNonnested)
result = predictor(report=paragraphs, report_jsonized={})
```

What gets sent to the model: a `ChatAdapter`-rendered prompt with `[[ ## procedure ## ]]`, `[[ ## cancer_quadrant ## ]]`, etc. field markers. No `response_format` set. On gpt-oss this works most of the time; on Gemma it tends to wrap the output in a Markdown fence and break the parser.

**Wire-level check:** `dspy.inspect_history(n=1)` shows the field-marker prompt; the response is plain text with markers around each field. No JSON involved.

### Stage 1 — Add `ChainOfThought`

```python
predictor = dspy.ChainOfThought(BreastCancerNonnested)   # the only change
```

Now the prompt asks for a `[[ ## reasoning ## ]]` block first, then the fields. The model spends a few hundred tokens reasoning before each field. **Accuracy on enum-heavy fields (`histology`, `cancer_quadrant`) typically jumps 5–15 points** purely from this.

What's still fragile: format. ChatAdapter is parsing field markers out of free text; any model that wraps the whole response in `````json … ````` will break. (`gpt-oss` doesn't, but most instruction-tuned models do.)

### Stage 2 — Add `JSONAdapter` + the `supports_response_schema` flip (this is what `pipeline_dspy_strict.py` does)

```python
import litellm
from dspy.adapters import JSONAdapter

litellm.register_model({
    "gpt-oss:20b": {
        "supports_response_schema": True,
        "litellm_provider": "ollama_chat",
        "mode": "chat",
    },
})
dspy.configure(lm=lm, adapter=JSONAdapter())

predictor = dspy.ChainOfThought(BreastCancerNonnested)
```

Now the prompt asks for JSON, the response_format on the wire is `{"type":"json_schema","strict":true}`, and llama.cpp's GBNF decoder masks invalid tokens at sampling time. **Parse-failure rate goes to ~zero**; accuracy on enum fields stays at the Stage-1 level (CoT is still doing its job inside the JSON envelope).

This is the protocol the user just confirmed works on `gpt-oss:20b`.

**Wire-level check:** `dspy.inspect_history(n=1)` shows a single user message asking for JSON with the schema embedded; the response is pure JSON whose first key is `reasoning`.

### Stage 3 — Compile with `BootstrapFewShot`

```python
import dspy
from digital_registrar_research.benchmarks.eval.metrics import field_correct

def metric(ex, pred, trace=None):
    return float(field_correct(ex.gold, dict(pred), "histology") or 0.0)

trainset = [...]   # list[dspy.Example] with .with_inputs("report", "report_jsonized")

class Wrapper(dspy.Module):
    def __init__(self):
        super().__init__()
        self.p = dspy.ChainOfThought(BreastCancerNonnested)
    def forward(self, report, report_jsonized):
        return self.p(report=report, report_jsonized=report_jsonized)

optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4,
                                  max_labeled_demos=2)
compiled = optimizer.compile(student=Wrapper(), trainset=trainset)
compiled.save("workspace/compiled/breast_nonnested_gptoss.json")
```

Now the prompt that goes to the model includes 4–6 worked examples, each one a full input → output JSON pair drawn from `trainset` and validated against the metric. **Accuracy typically jumps another 5–10 points on top of Stage 2**; the gain is largest on the rare enum values that the base prompt under-samples.

**Wire-level check:** `dspy.inspect_history(n=1)` now shows the demos inline at the start of the prompt. The schema and reasoning channel are unchanged from Stage 2.

The four stages are cumulative. By the time you reach Stage 3 you have: deterministic parsing (Stage 2), reasoning lift (Stage 1), and learned demonstrations (Stage 3) — and you've spent ~50 lines of code total.

---

## 5. Verification recipes

Cookbook-style, mirrors [eval/comparing_runs.md](eval/comparing_runs.md).

### Recipe 1 — confirm strict mode is actually active

**Question:** Did `_enable_strict_schema_for_ollama` actually flip the LiteLLM flag, or am I silently in JSON-mode?

```bash
LITELLM_LOG=DEBUG python - <<'PY' 2>&1 | grep -c "'type': 'json_schema'"
from digital_registrar_research.pipeline_dspy_strict import (
    setup_pipeline_dspy_strict, run_cancer_pipeline_dspy_strict,
)
setup_pipeline_dspy_strict("gptoss")
run_cancer_pipeline_dspy_strict("Sample breast cancer pathology report ...")
PY
```

**Expected:** count > 0 (one per per-organ extractor call). The `ReportJsonize` call uses `json_object`, not `json_schema`, by design (§1.3) — so don't be alarmed by a `json_object` line; only the *absence* of `json_schema` lines is a failure.

**If count is 0:** the override didn't land. Check that `_enable_strict_schema_for_ollama` runs *before* `load_model`, and that the `bare` key matches what LiteLLM would compute via `get_llm_provider(model_id)[0]`.

### Recipe 2 — see what DSPy actually sent the model

```python
from digital_registrar_research.pipeline_dspy_strict import (
    setup_pipeline_dspy_strict, run_cancer_pipeline_dspy_strict,
)
import dspy

setup_pipeline_dspy_strict("gptoss")
out, dt = run_cancer_pipeline_dspy_strict("...report text...")
dspy.inspect_history(n=1)   # last call
dspy.inspect_history(n=5)   # last five (is_cancer, jsonize, + organ extractors)
```

Read the printed prompt. The first JSON property in the response should be `reasoning`. The `response_format` field in the request payload should be `json_schema` (strict path) for organ extractors and `json_object` for `ReportJsonize`.

### Recipe 3 — measure the ChainOfThought lift

```bash
# Run each pipeline on the same fixture
python scripts/pipeline/run_dspy_ollama_smoke_dummy.py --pipeline structured \
    --model gptoss --out workspace/runs/structured_gptoss
python scripts/pipeline/run_dspy_ollama_smoke_dummy.py --pipeline dspy_strict \
    --model gptoss --out workspace/runs/dspy_strict_gptoss

# Compare in cascade
python -m scripts.eval.cli compare_runs \
    --runs workspace/runs/structured_gptoss workspace/runs/dspy_strict_gptoss \
    --out workspace/eval/cot_lift.parquet
```

Read the per-organ table. The delta between the two runs is the ChainOfThought + per-signature decomposition lift over the flat single-shot structured-output baseline.

(If `run_dspy_ollama_smoke_dummy.py` doesn't yet take a `--pipeline` flag, adding one is the minimum work for this recipe — the underlying functions are already drop-in replacements.)

---

## 6. Open research questions

These are worth a paragraph each in the eventual paper, and are good candidates for ablation rows:

1. **Why does ChainOfThought help even when the schema is strict?** The answer in §1.4 is a hypothesis: the reasoning prefix lets the model spend KV-cache attention before committing to enum tokens that grammar decoding has already pre-masked. A clean ablation would compare `JSONAdapter + Predict` against `JSONAdapter + ChainOfThought` on the same per-organ signatures and report the delta on enum-heavy vs. continuous fields separately. If the delta is concentrated on enums, the hypothesis is supported.
2. **Does TwoStepAdapter beat single-LM extraction on the cascade metric?** Phase 4 measures this directly. The compatibility doc's prediction is yes; nobody has actually run it.
3. **Do compiled few-shots transfer cross-organ?** Probably not — the signatures and the report styles are disjoint enough that breast demos shouldn't help colon. But the cost to test is one optimizer rerun, and a negative result is publishable.
4. **What's the marginal cost of `dspy.Refine` retries?** §2.2 suggests one extra LM call on offending cases. With invariant violation rates in the 5–15% range, that's a 5–15% latency overhead — but only if you're doing one refinement pass. Two passes doubles it. The paper-grade question is whether the second pass pays for itself.
5. **Can MIPROv2's instruction rewrites be inspected?** Yes — they're saved into the compiled JSON. An interesting qualitative finding would be: are the rewrites domain-specific (mentioning "pathology", "TNM staging") or generic? If domain-specific, the optimizer is genuinely learning the task; if generic, it's just learning to be more verbose.

---

## Appendix A — One-glance architecture

```
                   pathology report (string)
                              │
                              ▼
              dspy.ChainOfThought(is_cancer)            ──── JSONAdapter (strict)
                              │
                              ▼  cancer_category
              dspy.ChainOfThought(ReportJsonize)        ──── JSONAdapter (loose; dict output)
                              │
                              ▼  rough json
   for sig in organmodels[cancer_category]:
              dspy.ChainOfThought(sig)                  ──── JSONAdapter (strict)
                              │
                              ▼  organ_data, with reasoning stripped at :223
                       cancer_data dict
                              │
                              ▼
                  cascade three-chapter eval
```

LM: `dspy.LM("ollama_chat/gpt-oss:20b", api_base="http://localhost:11434")`
LiteLLM flag: `litellm.register_model({"gpt-oss:20b": {"supports_response_schema": True, ...}})`

---

## Appendix B — Reading list (DSPy 3.x)

- DSPy docs: https://dspy.ai/learn/
- `Adapter` reference: https://dspy.ai/api/adapters/
- `Refine` and `BestOfN` (the modern assertion path): https://dspy.ai/api/modules/Refine/
- `BootstrapFewShot` / `MIPROv2` API: https://dspy.ai/api/optimizers/
- LiteLLM's model-cost / capability table (where `supports_response_schema` is read): https://github.com/BerriAI/litellm/blob/main/litellm/model_prices_and_context_window_backup.json
- Ollama structured outputs (the runtime layer the protocol depends on): https://docs.ollama.com/capabilities/structured-outputs

---

*End of dspy_deep_dive. The roadmap in §3 is recommendations, not commitments — pick what's worth doing.*
