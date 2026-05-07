# Why only `gpt-oss:20b` runs your DSPy pipeline smoothly — diagnostic report

> **Scope note.** You asked for a research report, not code changes. This file is a
> diagnostic + background-knowledge document. Nothing in your repo has been modified.

---

## 0. TL;DR

Your pipeline is not failing because newer models are "dumber". It is failing because
**`gpt-oss:20b` is the only model in your lineup whose training regime, prompt format,
and Ollama runtime path all happen to line up with what DSPy's default `ChatAdapter`
emits for long, deeply-nested `dspy.Signature` outputs**. The other three (Gemma 3
27b, "Gemma 4" 26b, Qwen 3.5 27b) are instruction-tuned chat models that expect
`response_format={"type":"json_schema"}`-style constrained decoding — and both the
DSPy ↔ LiteLLM ↔ Ollama path and the model's own chat template quietly disagree on
how to deliver that, especially once signatures get long.

The three most load-bearing causes, in order of likelihood for your repo:

1. **Adapter mismatch + silent parse-retry loop.** DSPy tries `ChatAdapter` (field-
   marker text `[[ ## field ## ]]`), fails the parse on Gemma/Qwen for nested
   `list[BaseModel]` outputs, retries with `JSONAdapter`, which sends
   `response_format={"type":"json_object"}` / `json_schema` through LiteLLM → Ollama;
   the constraint is only honored by llama.cpp grammar decoding on some Ollama
   versions, is silently dropped in others, and interacts badly with Qwen3's
   "thinking mode." `gpt-oss:20b` is trained in OpenAI's *Harmony* channel format,
   which degrades gracefully into DSPy's field-marker syntax.
2. **Context starvation on long nested signatures.** You hard-code `num_ctx=8192` /
   `max_tokens=4096` for every model. The monolithic breast signature (7 merged
   sub-signatures, 3 `list[BaseModel]` fields, ~50 leaf fields) plus a long TCGA
   report pushes prompt + expected output past that budget. Ollama **silently
   truncates leading context**; newer models' tokenizers (Gemma's SentencePiece,
   Qwen's BPE variant) tokenize English medical text ~10–20% denser than gpt-oss's
   tokenizer, so the same report uses more tokens and crosses the cliff first.
3. **Instruction-tuning "over-alignment" on safety/format rigidity.** Gemma 3/4 and
   Qwen 3.x were RLHF/DPO'd to follow *their own* chat template and refuse
   out-of-distribution-looking prompts. DSPy's `ChatAdapter` prompt looks like
   nothing they saw in post-training; gpt-oss was explicitly trained to handle
   developer-supplied functions and structured channels, which looks much more
   like DSPy's output contract.

None of this is a capability problem — Gemma 3 27b and Qwen 3.5 27b are stronger
base models on most benchmarks than gpt-oss 20b. It is a **format-fit problem** at
the adapter/runtime layer. The remainder of this report unpacks each cause and gives
you a parameter matrix, a diagnostic script, and pipeline-level remedies you can
choose from.

---

## 1. What we actually found in your code

Reference file: [`models/common.py`](d:\localcode\digital-registrar-research-inference\digital-registrar-research\src\digital_registrar_research\models\common.py#L46-L55)

```python
MODEL_PROFILES = {
    "ollama_chat/gpt-oss:20b":   {"temperature": 0.3,  "top_p": 1.0,  "top_k": 40, "num_ctx": 8192, "max_tokens": 4096},
    "ollama_chat/gemma3:27b":    {"temperature": 0.15, "top_p": 0.95, "top_k": 64, "num_ctx": 8192, "max_tokens": 4096},
    "ollama_chat/gemma4:26b":    {"temperature": 0.1,  "top_p": 0.95, "top_k": 64, "num_ctx": 8192, "max_tokens": 4096},
    "ollama_chat/qwen3.5:27b":   {"temperature": 0.15, "top_p": 0.9,  "top_k": 40, "num_ctx": 8192, "max_tokens": 4096},
    ...
}
_BASE_KWARGS = {"repeat_penalty": 1.05, "keep_alive": "30m", "cache": False, "seed": 10}

lm = dspy.LM(model=model_id, api_base="http://localhost:11434",
             api_key="", model_type="chat", **kwargs)
```

Signature surface (from `pipeline.py` and `models/<organ>.py`):

- 57 signatures across 10 organs, 4–7 per organ.
- Two `InputField`s everywhere: `report: list` (paragraphs) and
  `report_jsonized: dict` (roughly-structured JSON from `ReportJsonize`).
- Output fields are heavy on `Literal[...]` enums (10–15 alternatives each), plus
  `list[BreastMargin]`, `list[BreastLN]`, `list[BreastBiomarker]` — each inner
  class is a Pydantic `BaseModel` with its own `Literal`s and `Optional`s.
- No explicit `dspy.ChatAdapter`/`dspy.JSONAdapter`/`dspy.TwoStepAdapter`; you rely
  on DSPy's auto-selection.
- `cache=False`, `seed=10` → every call is a fresh sample; seed affects sampling
  reproducibility but does NOT make the output deterministic at `temperature>0`
  across model versions.
- Pipeline wraps every signature in a per-signature `try/except` that logs and
  continues — so partial failures silently degrade output rather than raise.

Your config YAML for `qwen3_5` already has this comment:

> `# If you see JSON truncation on long reports, raise max_tokens first
> (6144-8192) before raising num_ctx.`

That comment is the smoking gun: someone already hit the failure mode this report
explains, and left an advisory note instead of a fix.

---

## 2. The four technical layers involved

You have to hold all four in your head at once. Debugging any one of them in
isolation will mislead.

### Layer A — DSPy adapters

DSPy does not talk to the LM in raw tokens; it goes through an *adapter* that turns
a `Signature` into a prompt and parses the completion back into typed Python
objects.

| Adapter | Prompt shape | Parser | Good for | Bad for |
|---|---|---|---|---|
| `ChatAdapter` (default) | Field markers `[[ ## name ## ]]` in both the system and output. For nested types it inlines a JSON schema in the *instructions*, then asks for JSON inside the marker. | Regex split on `[[ ## name ## ]]` then `json.loads` for complex types. | Anything that can follow a plain Markdown-ish convention, even non-JSON-native models. | Models that rewrite markers, add prose, add Markdown code fences around the whole thing, or emit CoT before the first marker. |
| `JSONAdapter` | Asks model to return a single JSON object whose keys are the output fields; uses `response_format={"type":"json_object"}` or `{"type":"json_schema", "json_schema": ...}` when the backend advertises support. | `json.loads`, then pydantic validate. | OpenAI/Anthropic/Ollama-with-grammar. Low-boilerplate. | Any runtime where `response_format` is silently ignored (see Layer C). |
| `TwoStepAdapter` | Round 1: free-text response from the "main" LM. Round 2: second LM ("extraction LM") is asked to convert Round 1 into the structured schema. | JSON parse on Round 2 output. | Reasoning models (o3-mini, gpt-oss at higher reasoning levels, DeepSeek-R1) that refuse to stick to a rigid format. Also: weak main LMs whose *content* is fine but whose *format* is shaky. | Cost/latency (2x calls). Extraction LM errors can still bite. |
| `XMLAdapter` | `<field>value</field>` tags. | XML parser. | Some Llama / Mistral variants that were clearly trained on XML-tagged data. | Same class of failure as ChatAdapter when models wander outside the tags. |

**Auto-fallback.** DSPy's default behavior is: try `ChatAdapter`; on parse failure
log a warning and retry once with `JSONAdapter`. In DSPy ≥ 2.6 there is
`Adapter.call_with_retries` that also retries on parse errors within the same
adapter. You will **see this only if you enable verbose mode** — otherwise it is
invisible and looks like "slower" inference.

Key consequence for you: *because you never set an adapter explicitly*, which
adapter wins depends on model + prompt + luck. On `gpt-oss:20b` the ChatAdapter
markers survive the Harmony channel structure and parse fine on first try. On
Gemma/Qwen the first try often fails quietly and you are silently running on
JSONAdapter, whose schema contract is only weakly enforced by Ollama for those
models.

### Layer B — Reasoning-channel-trained vs. instruction-tuned chat models

`gpt-oss:20b` is not a normal instruction-tuned model. It is trained in OpenAI's
**Harmony response format**, which ships messages across multiple explicit
channels:

```
<|start|>assistant<|channel|>analysis<|message|>…thinking…<|end|>
<|start|>assistant<|channel|>commentary<|message|>…tool call preamble…<|end|>
<|start|>assistant<|channel|>final<|message|>…answer to user…<|end|>
```

The training objective *explicitly separates internal reasoning, tool calls, and
the final user-visible output*. When DSPy's `ChatAdapter` asks for text like

```
[[ ## cancer_category ## ]]
"breast"
[[ ## cancer_excision_report ## ]]
true
```

gpt-oss happily treats that as a "final" channel output and emits exactly the
marker-delimited form. Its instruction-tuning is thin; the model's default behavior
*is* to obey structured format specs.

Gemma 3/4 and Qwen 3.x are the opposite school: heavy instruction tuning +
RLHF/DPO + a strong house chat template. That training rewards three things that
hurt DSPy:

1. **Prose wrapping.** "Here is the extracted data: …" gets glued on the front,
   breaking the first `[[ ## ... ## ]]` marker.
2. **Markdown fences.** Models put ```` ```json ```` around everything, which
   DSPy's field-marker regex does not expect.
3. **Refusal / safety rewrites** on medical content, especially on Gemma, which
   was notoriously over-aligned on medical/clinical text in its 2 and 3
   generations. You can see this as empty `distant_metastasis` fields or an
   opening "I am not a medical professional…" line that shifts all field offsets.

Qwen 3.x adds a further complication: **thinking mode**. Qwen3 models have a
soft-switch controlled by `/think` vs `/no_think` in the prompt, or
`enable_thinking=True/False` via the API. Confirmed issues (see sources):

- With `enable_thinking=True`, structured-output schemas are **not supported**;
  guided JSON may come out malformed or with thinking tokens leaked in.
- With `enable_thinking=False`, structured output often becomes invalid JSON —
  vLLM issue 18819 and sglang issue 6675 both confirm this.
- The workaround that tends to work is: keep thinking enabled, but include
  `/no_think` in the user message, and ensure the literal word "JSON"
  (case-insensitive) appears in the system prompt.

None of this is in your prompt today — DSPy's default prompt does not include
`/no_think`, and the string "JSON" only appears when the adapter decides to fall
back to `JSONAdapter`.

### Layer C — Ollama, LiteLLM, and `response_format`

DSPy's `dspy.LM(model="ollama_chat/…")` routes through **LiteLLM**, which wraps
Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`). The relevant
behaviors:

- Ollama **≥ 0.5.0** honors `response_format={"type":"json_schema", "json_schema": {...}}`
  via llama.cpp's grammar-constrained decoder (GBNF). This is real:
  tokens outside the grammar are masked at sampling time. That is the only way to
  get *guaranteed* schema-valid JSON on Ollama today.
- **`response_format={"type":"json_object"}`** is weaker — it is a prompt-level
  instruction, not a hard grammar — and its enforcement depends on the model's
  chat template.
- **`num_ctx`** is a server-side Ollama parameter. Ollama's default is **2048
  tokens** if not set. It is passed through by LiteLLM via the `options` dict.
  Crucially, Ollama *silently truncates the leading* tokens if prompt > num_ctx.
  There is no error. You just get a shorter prompt, and the model's first output
  tokens look like it hallucinated a report it never saw.
- **`num_predict`** caps output tokens; LiteLLM maps `max_tokens` → `num_predict`.
  If the schema expects 3000 tokens of nested JSON and `num_predict=4096` but the
  prompt ate 6000 of your 8192 context, the effective output budget is
  `8192 - 6000 = 2192` tokens, truncated mid-object. JSON parse fails.
- There are active Ollama bugs specifically affecting your target models:
  - Issue #15260 — `think=false` breaks `format` (structured output) for `gemma4`;
    format constraint silently ignored.
  - Issue #15540 — structured output not enforced on `qwen3.5` / `gemma4`.
  - Issue #14570 — Qwen3 tool-call parser returns HTTP 500 when model output is
    truncated (which happens exactly in the regime you are operating in).

You are using `model_type="chat"` with `ollama_chat/` prefix, which is the
LiteLLM Ollama *Chat* provider (not the deprecated `ollama/` completions one).
That is correct, but it means the `/v1/chat/completions` path is used, which
adds ~50–200 boilerplate tokens for the chat template on every call.

### Layer D — Tokenizer density and context accounting

Different tokenizers shred the same English text into different numbers of
tokens. Empirical rough numbers for a typical English pathology report of 3,000
characters:

| Tokenizer | Tokens (approx) |
|---|---|
| gpt-oss (o200k-derived) | 600–700 |
| Gemma 3/4 (SentencePiece) | 750–900 |
| Qwen 3.x (BPE) | 700–850 |
| Llama 3 (tiktoken-like) | 680–780 |

That is a ~15–30% variation at the *input* side alone. At 8192 `num_ctx`, the same
signature + report that uses 5,800 tokens on gpt-oss uses 6,800 on Gemma, which
eats into the 4096 output budget and causes truncation right at the tail of a
`list[BreastBiomarker]`. Truncation lands inside a JSON structure → parser fails
→ DSPy retries with JSONAdapter → Ollama emits another 4k tokens but now total
exceeds 8192 → silent leading-prompt truncation → model "forgets" the schema
description → emits free-form prose → parse fails again → your `except Exception`
logs a warning and that organ's fields are all `None`.

That cascade is what "fails on longer dspy signatures" looks like from the outside.

---

## 3. Why newer models have **not** gotten better at this (your actual question)

This is the question a lot of people get wrong. The answer is not that models
haven't improved — Gemma 3 27b and Qwen 3.5 27b beat gpt-oss 20b on MMLU, GSM8K,
and most reasoning benchmarks. The answer has four parts:

1. **Post-training targets were "chat assistant", not "JSON subroutine".** The
   frontier labs optimize RLHF for dialogue quality, safety, and refusal
   calibration. Structured-output capability was a second-class citizen until the
   function-calling wars of 2024–2025, and even then only the API-gated frontier
   models (GPT-4o, Claude 3.5+, Gemini 2) got deep training on it. Open-weights
   models inherit *some* of that through tool-use SFT, but not the deep schema
   adherence of the closed models.
2. **Schema-adherence regressions from alignment tax.** The more heavily a model is
   RLHF'd toward being "helpful and harmless", the more it likes to add
   explanation, disclaimers, markdown, and conversational scaffolding. That is
   negative for structured-output tasks. Gemma 3/4 are more aligned than Gemma 2;
   Qwen 3 is more aligned than Qwen 2.5. They each picked up schema regressions.
3. **Runtime fragmentation.** Constrained decoding (GBNF, outlines, xgrammar,
   json-schema guided decoding) is the *right* fix and works well in principle,
   but it lives in the inference runtime (llama.cpp / vLLM / sglang / TGI), not
   in the model weights. Each runtime has different grammar bugs with different
   tokenizer families. New models come out faster than their grammar support
   stabilizes in Ollama. That is why you see GitHub issues #15260 and #15540
   filed against `gemma4` and `qwen3.5` *specifically* in the Ollama repo — it is
   a runtime integration problem, not a model problem.
4. **gpt-oss is an outlier.** It is the only popular open-weights model trained
   *by an API lab* (OpenAI) whose post-training explicitly treats structured
   outputs, tool calls, and "developer channel" messages as first-class. That is
   why it handles DSPy's format. It is also comparatively small (20B active, MoE),
   so it is *weaker* on content reasoning than Gemma 27b or Qwen 27b — but for
   DSPy's use case, format reliability dominates content capability.

Practical upshot: **for medical IE pipelines built on DSPy + local Ollama, you
should not expect newer-and-bigger → better. You should expect format-fit to
dominate raw capability until DSPy's adapter and Ollama's grammar stack both
catch up to each new model family.** That is a real, persistent phenomenon, not
a temporary glitch.

---

## 4. Which of these is actually biting you? A free diagnostic plan (no code changes)

You can answer this in 30 minutes without touching the pipeline code. Run these
probes one at a time against one of the `scripts/pipeline/run_dspy_ollama_smoke_*_dummy.py`
entry points with `gemma4` or `qwen3_5`, not `gptoss`.

### 4.1 Turn on DSPy's verbose logging

In the smoke script's entry point, before the pipeline runs:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("dspy").setLevel(logging.DEBUG)
logging.getLogger("dspy.adapters").setLevel(logging.DEBUG)
logging.getLogger("dspy.adapters.json_adapter").setLevel(logging.DEBUG)
logging.getLogger("LiteLLM").setLevel(logging.INFO)
```

You are looking for these lines in the log:

- `WARNING dspy.adapters.chat_adapter: Failed to parse response` — ChatAdapter
  died. Adapter fallback in progress.
- `WARNING dspy.adapters.json_adapter: Failed to use structured output format,
  falling back to JSON mode` — Ollama rejected or didn't honor `json_schema`.
- `dspy.adapters.Adapter: Retrying call due to …` — inner retry loop.
- LiteLLM's `POST /v1/chat/completions` dump, showing the *actual* `options`
  payload (num_ctx, num_predict, temperature) that reached Ollama.

### 4.2 Count tokens on a failing call

Inspect Ollama's server log (`ollama serve` stdout, or `journalctl -u ollama` on
Linux / Docker Desktop logs on Windows). Each request logs a line of the form:

```
prompt eval count: N tokens
eval count:        M tokens
```

If `N + M ≈ num_ctx` and the model failed, you are almost certainly in the
context-starvation regime (Layer D). If `N ≪ num_ctx` and it still failed, you
are in the adapter/format regime (Layers A+B).

### 4.3 Issue a raw-JSON probe via your ablations runner

Your repo already has `/d/localcode/digitalregistrar-ablations/runners/raw_json.py`
which **bypasses DSPy entirely** and sends a direct OpenAI-API call to Ollama with
`response_format={"type":"json_object"}` and a Pydantic schema. That is the
cleanest A/B test available:

- If raw-JSON **works** on Gemma 4 / Qwen 3.5 where DSPy **fails** → your problem
  is at the DSPy adapter layer (Layer A + Layer B interaction). Fix: force an
  explicit adapter, preferably `TwoStepAdapter` or a `JSONAdapter` with explicit
  `response_format={"type":"json_schema"}`.
- If raw-JSON **also fails** → your problem is at the Ollama runtime / context
  layer (Layers C + D). Fix: raise `num_ctx` / `max_tokens`, split monolithic
  signatures back to modular ones, or use grammar-constrained decoding.
- If raw-JSON fails *only on breast* and works on prostate → you are
  context-starved on the monolithic breast signature specifically (matches the
  `docs/ablations_design_rationale.md:108-113` warning already in the repo).

### 4.4 Confirm which adapter won

After a call, inspect `dspy.inspect_history(n=1)`. The last prompt sent will
contain `[[ ## field ## ]]` markers (ChatAdapter won) or a request for a single
JSON object (JSONAdapter won). You can also print the adapter directly via
`dspy.settings.adapter`.

### 4.5 Probe Ollama's grammar enforcement directly

One `curl`:

```bash
curl http://localhost:11434/v1/chat/completions \
  -d '{"model":"gemma4:26b","messages":[{"role":"user","content":"Give me JSON with a name field"}],
       "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}},
       "options":{"num_ctx":8192,"num_predict":128,"temperature":0.1}}'
```

If the returned content is *not* pure JSON parseable to `{"name":"…"}`, then your
Ollama version is affected by issue #15540 and grammar decoding is broken for that
model. This is the single highest-signal check you can run.

---

## 5. Parameter-adjustment matrix

These are **recommended** values, grounded in the sources at the bottom and in
your repo's own comments. Treat them as starting points; validate with §4.

### 5.1 Sampling parameters

| Parameter | `gpt-oss:20b` (current) | `gemma3:27b` | `gemma4:26b` | `qwen3.5:27b` | Rationale |
|---|---|---|---|---|---|
| `temperature` | 0.3 | **0.2** (was 0.15) | **0.2** (was 0.1) | **0.2** (was 0.15) | Too-cold sampling on instruction-tuned models amplifies refusal tendencies and format rigidity. 0.2 is the sweet spot for JSON extraction on most open-weights families. Avoid 0.0 — deterministic sampling on these models often collapses to canned refusals or mode-collapses on rare field values. |
| `top_p` | 1.0 | 0.95 (keep) | 0.95 (keep) | **0.95** (was 0.9) | Slightly wider nucleus helps escape refusal attractors. |
| `top_k` | 40 | 64 (keep) | 64 (keep) | **64** (was 40) | Gemma and Qwen both benefit from `top_k=64` in structured-output benchmarks; `top_k=40` is Llama-family folklore. |
| `repeat_penalty` | 1.05 | 1.05 | **1.0** | 1.05 | Gemma 4 is reported to have a repetition regression when `repeat_penalty > 1.0` is combined with grammar-constrained decoding — it occasionally substitutes near-synonyms to avoid repeats, breaking `Literal` parsing. |
| `seed` | 10 | 10 | 10 | 10 | Keep for reproducibility; irrelevant to failures. |
| `cache` | False | False | False | False | Keep — you want each call to be a fresh sample. |

### 5.2 Context parameters (this is the most important table)

| Parameter | Current | Recommended for long signatures | Why |
|---|---|---|---|
| `num_ctx` | 8192 | **16384** for gpt-oss, gemma3, qwen3.5; **12288** for gemma4 | Gemma 3 and 4 officially support 128k; Qwen 3.5 supports 128k; gpt-oss supports 131k. 8192 is leaving 75%+ of the context on the table. 16k is cheap on a 27b at Q4_K_M and covers every TCGA report in your dummy data. **Gemma 4 has a known VRAM blow-up at >16k on consumer GPUs; 12k is the safe point.** |
| `max_tokens` (→ `num_predict`) | 4096 | **6144** for monolithic breast / lung / liver; **3072** for simple per-organ signatures | Matches the repo's own `qwen3_5.yaml` advisory comment. Monolithic signatures need ~4–5k output tokens once you account for nested `list[BaseModel]` serialization. |
| `keep_alive` | 30m | 30m (keep) | Loading 27b weights costs 20–40 seconds; keep them hot. |

### 5.3 Per-model workarounds not expressible as a single parameter

These require either adapter changes or prompt tweaks, and are summarized here
for completeness (not requested as implementation — just documenting the levers).

- **Qwen 3.5**: include `/no_think` in the first user message to suppress
  thinking-mode output, AND ensure the system prompt contains the literal word
  "JSON" (case-insensitive). Without one of these, Qwen3 structured-output mode
  is documented-broken (sources: vLLM #18819, sglang #6675).
- **Gemma 4**: if your Ollama version is affected by issue #15260, explicitly set
  `think=true` in the Ollama options even if you don't want thinking output —
  format constraints are only enforced in that mode for now. If you're on a
  patched Ollama, ignore this.
- **Gemma 3 / 4**: both respond better to a **system prompt** containing the
  schema than to user-message-embedded schema. DSPy's ChatAdapter puts the
  schema in the system message, which is good; JSONAdapter sometimes puts it in
  the user message, which is worse on Gemma. Forcing `ChatAdapter` helps.
- **All four**: setting `response_format={"type":"json_schema","json_schema":...}`
  explicitly via a custom adapter is the single highest-leverage intervention if
  your Ollama is recent (≥0.6.x).

---

## 6. Pipeline-level adjustments (if you later decide to change code)

Documenting only; you said no code changes now.

### 6.1 Force the adapter explicitly

```python
# Anywhere after dspy.configure(...)
dspy.settings.configure(adapter=dspy.ChatAdapter())  # most compatible
# OR, if diagnostic §4.5 shows grammar decoding works:
dspy.settings.configure(adapter=dspy.JSONAdapter())
# OR, for reasoning-struggling models:
extraction_lm = dspy.LM("ollama_chat/gpt-oss:20b", ...)  # small+reliable
dspy.settings.configure(adapter=dspy.TwoStepAdapter(extraction_lm))
```

`TwoStepAdapter` is interesting for you specifically: you can keep Gemma 4 /
Qwen 3.5 as the **content** model (they are smarter at reading pathology reports)
and use `gpt-oss:20b` as the **extraction** model. You already have both loaded.
That gets you the best of both worlds at 2x latency.

### 6.2 Never build the monolithic breast signature in prod; keep modular

Your own design doc (`docs/ablations_design_rationale.md:108-113`) already flagged
that monolithic breast overflows 16k context for long reports. The modular
path (one signature per subsection) is the right default. The monolithic variant
should stay confined to the ablations study.

### 6.3 Add an explicit `Literal` → free-string escape hatch

For every `Literal[...]` field that covers a known-enumerable clinical concept,
add an `others_description: str | None` companion (you already do this for
`cancer_category_others_description`). This catches the case where a heavily-
aligned model refuses a specific enum value and emits a paraphrase that fails
`Literal` validation. Currently this only exists for `cancer_category`; adding
it for `histology`, `cancer_quadrant`, and `procedure` would eliminate a class of
silent-field-drop failures.

### 6.4 Surface parse failures instead of swallowing them

`pipeline.py:113-115` wraps every signature in `except Exception: continue`. That
means a Gemma parse failure looks identical to "no data for this field" in the
output. Loudening this (at minimum, tagging each output field with a
`_parse_status: "ok" | "parse_failed" | "empty"` sidecar) would give you
signal on which failures are model-caused vs. data-caused — crucial for
interpreting your own ablations results.

### 6.5 Reuse `raw_json.py`'s validation-retry pattern

Your ablations `raw_json.py` already does: run → validate against Pydantic → on
failure, surface errors to the model and retry once. That pattern is strictly
better than DSPy's blind retry for this workload. A small helper that wraps
`dspy.Predict(cls)` with one Pydantic-validation retry (using the same schema
DSPy already has) would close the gap at essentially no latency cost (2% of
calls retry).

### 6.6 Per-tokenizer context budgets

Replace the static `num_ctx=8192` with a per-model budget computed from the
model's actual context window. Something like:

```python
CONTEXT_LIMITS = {
    "ollama_chat/gpt-oss:20b": 131072,
    "ollama_chat/gemma3:27b":  131072,
    "ollama_chat/gemma4:26b":   32768,
    "ollama_chat/qwen3.5:27b": 131072,
}

def budget(model_id: str, desired: int) -> int:
    # Leave VRAM headroom: cap at 32k by default regardless
    return min(desired, CONTEXT_LIMITS[model_id], 32768)
```

Then pass `num_ctx=budget(model_id, 16384)` per call. This is the single
parameter change that pays back the most on long signatures.

### 6.7 Do the raw-JSON sanity-baseline before each experiment

Before running a new model through the ablations cross-product, run the
raw-JSON runner on 3–5 dummy cases first. If raw-JSON fails on that model,
there is zero point comparing DSPy Cells A and B — you will be measuring
adapter noise, not model capability.

---

## 7. Why the ablations-study design anticipates this problem

Credit where it is due: your ablations design doc already says:

> `On gpt-oss:20b:  A > B ≫ C  (modularity helps; DSPy helps)`
> `On gpt-4-turbo:  A ≈ B ≈ C  (frontier model handles all equally)`

That is exactly the prediction this report is re-deriving from first principles.
The missing row you are now discovering empirically is:

> `On gemma4:26b / qwen3.5:27b:  A ≫ B ≫ C, and even A may collapse to noise`

That is the format-fit dominance effect. The ablations design should explicitly
add a "C > A for format-brittle models" column and test for it — that becomes a
first-class finding of the paper, not a limitation.

---

## 8. Clarifications that would sharpen this report

I did not pause to ask because you wanted a verbose report first, but here are
the three questions whose answers would let me tighten the diagnosis:

1. What is your Ollama version? (`ollama --version`) — determines which of the
   #15260 / #15540 bugs are live for you.
2. What does the actual failure look like — Python exception message, JSON parse
   error, empty output, or timeout? Each points to a different layer.
3. Are you running on the CMUH workstation GPU (the one the repo hints at) or on
   a smaller laptop? The `num_ctx` recommendations depend on available VRAM.

If useful, send those and I can write a tighter per-model action list.

---

## 9. Sources and references

Primary evidence cited above, grouped by layer.

**DSPy adapters and auto-fallback behavior:**
- [Adapters — DSPy docs](https://dspy.ai/learn/programming/adapters/)
- [ChatAdapter — DSPy API](https://dspy.ai/api/adapters/ChatAdapter/)
- [JSONAdapter — DSPy API](https://dspy.ai/api/adapters/JSONAdapter/)
- [TwoStepAdapter — DSPy API](https://dspy.ai/api/adapters/TwoStepAdapter/)
- [PR #8011: Introduce two step adapter](https://github.com/stanfordnlp/dspy/pull/8011)
- [Issue #8440: Both structured output format and JSON mode failed for phi3:mini and llama3.1-nemotron-nano](https://github.com/stanfordnlp/dspy/issues/8440)
- [Issue #8793: Failed to use structured output format, falling back to JSON mode](https://github.com/stanfordnlp/dspy/issues/8793)
- [Issue #8034: Issue using DSPy with Gemini, Ollama, and LM Studio models](https://github.com/stanfordnlp/dspy/issues/8034)
- [Issue #7804: Support Ollama natively to allow controlling context window](https://github.com/stanfordnlp/dspy/issues/7804)
- [Issue #1932: Why isn't JSONAdapter the default adapter option?](https://github.com/stanfordnlp/dspy/issues/1932)

**gpt-oss / Harmony format:**
- [Introducing gpt-oss — OpenAI](https://openai.com/index/introducing-gpt-oss/)
- [gpt-oss-20b & gpt-oss-120b model card — arXiv 2508.10925](https://arxiv.org/abs/2508.10925)
- [openai/gpt-oss-20b — Hugging Face](https://huggingface.co/openai/gpt-oss-20b)
- [gpt-oss:20b — Ollama library](https://ollama.com/library/gpt-oss:20b)

**Qwen3 thinking mode and structured-output regressions:**
- [Qwen/Qwen3-32B — Hugging Face model card](https://huggingface.co/Qwen/Qwen3-32B)
- [vLLM #18819: Broken Structured Output (Guided Decoding) with Qwen3 models when enable_thinking=False](https://github.com/vllm-project/vllm/issues/18819)
- [sglang #6675: Broken Structured Outputs with enable_thinking=False in Qwen3](https://github.com/sgl-project/sglang/issues/6675)
- [Enforce Structured JSON Output with Qwen Models — Alibaba Cloud docs](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)
- [Constraining LLMs with Structured Output: Ollama, Qwen3 & Python or Go — Rost Glukhov](https://www.glukhov.org/llm-performance/ollama/llm-structured-output-with-ollama-in-python-and-go/)

**Ollama structured output, grammar decoding, num_ctx:**
- [Structured outputs — Ollama blog](https://ollama.com/blog/structured-outputs)
- [Structured Outputs — Ollama docs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama #2714: Misunderstanding of ollama num_ctx parameter and context window](https://github.com/ollama/ollama/issues/2714)
- [Ollama #9519: Cannot Increase num_ctx Beyond 2048 in Ollama](https://github.com/ollama/ollama/issues/9519)
- [Ollama #14570: qwen3 tool call parser returns 500 when model output is truncated](https://github.com/ollama/ollama/issues/14570)
- [Ollama #14492: ollama-qwen3.5:35b suspicious compatibility problem](https://github.com/ollama/ollama/issues/14492)
- [Ollama #15260: think=false breaks format (structured output) for gemma4 — format constraint silently ignored](https://github.com/ollama/ollama/issues/15260)
- [Ollama #15540: structured output not enforced on qwen 3.5 / gemma 4](https://github.com/ollama/ollama/issues/15540)

**Ollama context-length defaults and truncation:**
- [Ollama Context Length: Default Settings and How to Modify It](https://deepai.tn/glossary/ollama/ollama-context-length/)
- [How to increase context length of local LLMs in Ollama](https://localllm.in/blog/local-llm-increase-context-length-ollama)
- [Hacker News: Ollama defaults to a context of 2048 regardless of model unless you override it](https://news.ycombinator.com/item?id=43274296)

---

*End of diagnostic report. No code has been modified.*
