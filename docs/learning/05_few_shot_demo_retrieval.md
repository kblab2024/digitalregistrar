# 05 — Few-shot demo retrieval: a concrete design

> Goal: a real, scoped proposal for adding **retrieval-augmented few-shot learning** to drr-next. Not a tour of RAG options. The blueprint below is what I (writing this doc) recommend building first, with the trade-offs and risks I'd flag for the implementer.

This doc assumes you've read [04 — DSPy in depth](04_dspy_in_depth.md) §5 (Predict + `.demos`) and [03 — chunking](03_chunking.md) §5 (the per-step filter loop). It builds directly on top of both.

---

## 0. Where the project sits today

Every LM call in `CancerPipelineV2.forward()` is **zero-shot**. The signature carries an instruction (the case-model docstring + group instruction), the LM sees the rendered prompt with the report, and emits a structured output. There are no in-context examples.

drr-next has, today:
- A chunking + routing layer that focuses prompts on relevant sections (Doc 3).
- An embedding layer (`dspy.Embedder`) wired up for chunk routing (Doc 4 §9).
- Per-organ, per-group signatures built dynamically by the factory (Doc 1, Doc 4 §3).

What's missing: **learned signal**. Every report is treated as if no similar case has ever been seen.

---

## 1. Why few-shot demo retrieval is the right next move

Three reasons (all specific to this codebase):

1. **DSPy already has `.demos`.** `dspy.Predict` instances expose a `.demos` list. Whatever you put there gets rendered into the prompt as worked examples. No new LM-call infrastructure is needed.
2. **Per-group structure cleans up the retrieval index.** drr-next already builds **per-group** signatures. So the index is naturally keyed by `(organ, group)` — the lymph-node demos for breast cancer live in their own slice, separate from breast biomarkers and from lung lymph nodes. This keeps each retrieval space small (~hundreds to ~thousand demos, not millions) and on-topic.
3. **Chunking composes naturally.** The query for retrieval is the same `step_input` the filter loop already computes (Doc 3 §5). If routing pre-filtered the report down to lymph-node chunks for the `lymph_nodes` step, retrieval *also* queries with just those chunks. Better focus, both layers.

What this **doesn't** require:
- A vector DB. JSONL + a numpy `.npy` works while the store is small.
- New LM call shapes. Existing `dspy.Predict` already accepts demos.
- Pipeline-side changes outside one new hook.

It does add complexity (a demo store, an index, a curation workflow) — but the marginal complexity is minimal vs the alternatives (knowledge-base RAG, cross-report validation, compilation/teleprompters).

---

## 2. What a demo looks like

The shape is intentionally minimal:

```python
demo = {
    "report":  "...full pathology report text (or routed chunk text)...",
    "organ":   "breast",
    "group":   "margins",
    "outputs": {
        "margins": [
            {"margin_category": "anterior", "margin_involved": False, "distance": 4, "description": "..."},
            {"margin_category": "posterior", "margin_involved": False, "distance": 8, "description": "..."},
        ],
    },
    # Optional metadata for curation / debugging:
    "source": "tcga_BRCA_TCGA-XX-1234",
    "added_at": "2026-05-22",
    "reviewer": "dr_xyz",
    "notes": "Edge case: anterior margin <5 mm — kept for boundary discussion.",
}
```

Key choices:

- **Per (organ, group)**. One demo per group, not one demo for the whole case. This matches the per-group signature factory output.
- **`outputs` is a dict of field-name → value**. Same shape the LM emits today, so the demo is a *literal example* of the structured response.
- **`report` carries either the full report or the routed chunk text**. The store doesn't know about routing; the *write* side does. If you write a demo with the LN-section text only, retrieval at LN-extraction time will return that demo for LN-shaped queries.

---

## 3. Where demos come from

### Cold start

- A small batch of hand-annotated reports. The repo already has the
  [tests/fixtures/reference/](../../tests/fixtures/reference/) directory and
  [examples/dummy/data/](../../examples/dummy/data/) folder structure with
  per-case `gold` subfolders.
- For each gold case, decompose the case record by group and write one demo
  per (organ, group) where ground-truth fields exist.
- 5–20 demos per (organ, group) is plenty for cold start; small models
  benefit from even 1–3.

### Warm path (later)

- High-confidence outputs from production runs become *candidate* demos.
- A human reviews and accepts; on accept, the (report, organ, group, outputs)
  tuple becomes a real demo.
- Confidence here is *not* "the LM's reported confidence" — it's external:
  pydantic validation passed, key fields agree with downstream consumers, etc.

---

## 4. Module layout (new files)

Three small files, all under a new `demos/` package:

```
src/digital_registrar/demos/
├── __init__.py     # public API: add_demo, retrieve_demos
├── store.py        # add_demo, embed_text, persist JSONL + .npy
└── retrieve.py     # load store, compute cosine, return top-K
```

### `store.py` — write side

```python
# Sketch only — see Doc 4 §9 for dspy.Embedder details.

from pathlib import Path
import json
import numpy as np
import dspy

DEMO_ROOT = Path("data/demos")   # configurable

def _index_paths(organ: str, group: str) -> tuple[Path, Path]:
    base = DEMO_ROOT / organ / group
    base.mkdir(parents=True, exist_ok=True)
    return base / "demos.jsonl", base / "demos.npy"

def add_demo(
    *,
    organ: str,
    group: str,
    report: str,
    outputs: dict,
    source: str = "",
    notes: str = "",
    embedder: dspy.Embedder | None = None,
) -> None:
    """Append one demo to the (organ, group) store, recomputing the index."""
    jsonl_path, npy_path = _index_paths(organ, group)

    demo = {
        "report": report, "organ": organ, "group": group,
        "outputs": outputs, "source": source, "notes": notes,
    }

    # Append the JSONL row.
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(demo, ensure_ascii=False) + "\n")

    # Recompute the embedding index. (For larger stores, incrementally
    # append to the .npy instead — but rebuild is fine while N < 10k.)
    embedder = embedder or dspy.Embedder("sentence-transformers/all-MiniLM-L6-v2")
    all_demos = [json.loads(line) for line in jsonl_path.read_text("utf-8").splitlines()]
    texts = [d["report"] for d in all_demos]
    vecs = np.asarray(embedder(texts), dtype=np.float32)
    # L2-normalize so cosine = dot product.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    np.save(npy_path, vecs / norms)
```

### `retrieve.py` — read side

```python
import json
from pathlib import Path
import numpy as np
import dspy

# Process-level cache keyed by (organ, group) so we read each store once.
_CACHE: dict[tuple[str, str], tuple[list[dict], np.ndarray]] = {}
_EMBEDDER: dspy.Embedder | None = None

def _load(organ: str, group: str) -> tuple[list[dict], np.ndarray] | None:
    """Return (demos, embeddings) for an (organ, group), or None if no store."""
    key = (organ, group)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    base = DEMO_ROOT / organ / group
    jsonl, npy = base / "demos.jsonl", base / "demos.npy"
    if not jsonl.exists() or not npy.exists():
        return None
    demos = [json.loads(line) for line in jsonl.read_text("utf-8").splitlines()]
    vecs = np.load(npy)
    _CACHE[key] = (demos, vecs)
    return _CACHE[key]

def retrieve_demos(
    *,
    organ: str,
    group: str,
    query_text: str | list[str],
    k: int = 3,
) -> list[dict]:
    """Return top-K demos for this (organ, group) by cosine similarity."""
    loaded = _load(organ, group)
    if loaded is None:
        return []
    demos, vecs = loaded

    if isinstance(query_text, list):
        query_text = "\n\n".join(query_text)

    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = dspy.Embedder("sentence-transformers/all-MiniLM-L6-v2")
    q = np.asarray(_EMBEDDER([query_text]), dtype=np.float32)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    qn[qn == 0] = 1.0
    q = q / qn

    sims = (q @ vecs.T)[0]   # shape (N,)
    top = np.argsort(-sims)[:k]
    return [demos[i] for i in top]
```

### `__init__.py` — public API

```python
from .store import add_demo
from .retrieve import retrieve_demos

__all__ = ["add_demo", "retrieve_demos"]
```

That's the whole `demos` package. ~100 lines total.

---

## 5. Pipeline integration — one new hook

The change to [pipeline_factory.py](../../src/digital_registrar/pipeline_factory.py) is small. Two places:

### 5a. Constructor signature

```python
class CancerPipelineV2(dspy.Module):
    def __init__(
        self,
        *,
        decomposition: Literal["per_group", "monolithic", "auto"] = "auto",
        jsonize_enabled: bool = False,
        model_profile: str | None = None,
        validate_output: bool = True,
        chunker: Chunker | None = None,
        chunk_router: Router | None = None,
        routing_mode: Literal["off", "filter"] = "off",
        demo_retrieval_k: int = 0,         # NEW: 0 disables, >0 retrieves K
    ) -> None:
        ...
        self._demo_k = int(demo_retrieval_k)
```

`demo_retrieval_k=0` (default) means no demos — preserves current behavior exactly. Pass `=3` to inject 3 retrieved demos per per-group predictor at extraction time.

### 5b. Inside the per-step loop

In [forward()](../../src/digital_registrar/pipeline_factory.py#L236-L266), right before the `predictor(...)` call:

```python
for step, predictor in extractors:
    step_input = paragraphs
    if routed_chunks is not None:
        picked = [c for c in routed_chunks if step.group in c.labels]
        if picked:
            step_input = [c.text for c in picked]
        ...

    # NEW: inject retrieved demos for this step.
    if self._demo_k > 0:
        from .demos import retrieve_demos
        demos = retrieve_demos(
            organ=rsp.cancer_category,
            group=step.group,
            query_text=step_input,
            k=self._demo_k,
        )
        predictor.demos = [
            dspy.Example(report=d["report"], **d["outputs"])
            for d in demos
        ]

    try:
        pred = predictor(report=step_input, report_jsonized=json_report)
        ...
```

**Important**: `dspy.Example` is the shape DSPy expects for `.demos`. We're mapping our stored demo dict into the right shape. The exact field names in the kwargs must match the signature's output field names — that's why per-group storage helps (you know which output fields are in scope).

That's it. Two diffs. The rest of the pipeline (router, jsonize, validation, chunking) is unchanged.

---

## 6. Embedding backend choice

Reuse `dspy.Embedder` — already used by `DSPyEmbedderRouter` (Doc 3 §4, Doc 4 §9). Two reasons:

1. **One embedding model in the process.** If the chunk router uses `sentence-transformers/all-MiniLM-L6-v2`, the demo store should too. Same vector space → similarity scores are comparable.
2. **Future-proofing for provider swaps.** `dspy.Embedder("openai/text-embedding-3-small")` and `dspy.Embedder("sentence-transformers/all-MiniLM-L6-v2")` have identical interfaces. Switching only requires changing the `model=` string in *one* place.

Don't reach for FAISS / Chroma at the start. While N < ~10k per (organ, group), numpy cosine is fast (sub-millisecond) and has zero ops overhead. The day you have 100k+ demos for one group is the day to introduce a vector DB — *not* before.

---

## 7. Eval design — how to know it's working

Two-arm comparison on a held-out set:

| Arm | Setup | Measure |
|---|---|---|
| Control | `demo_retrieval_k=0` (current behavior) | Per-field accuracy, per-group F1 |
| Treatment | `demo_retrieval_k=3` | Per-field accuracy, per-group F1 |

Use the existing parity scaffold in [tests/pipeline/test_v2_parity.py](../../tests/pipeline/test_v2_parity.py) as a starting point. Add a small script (`eval_few_shot.py`) that:

1. Loads a fixed gold set (e.g. 50 cases from `examples/dummy/data/cmuh/annotations/gold/`).
2. Runs both arms.
3. For each (organ, group), compares per-field predictions to gold.
4. Reports the delta with confidence intervals (bootstrap, B = 500).

Expected outcome (educated guess, not promised): per-group F1 improves by 2-7 points for high-variance groups (margins, biomarkers, lymph nodes) and is flat for groups where signature instructions already nail the task (staging, nonnested).

**The eval is also the curation feedback loop** — if a group doesn't improve, your demos for it are off-distribution or too few.

---

## 8. Risks and non-obvious bits

### Prompt length explodes

Demos add tokens. Three demos × ~800-token average × N per-group steps = a lot more cost / latency. Mitigations:

- **Cap demo text** at ~500 tokens. Truncate longer reports before storing.
- **Store the routed chunk text, not the full report**, when chunking is on. The demo becomes "here's a lymph-node section and the right output for it" — much shorter.
- **Profile** before assuming. With `gpt-oss:20b` on Ollama, 3 short demos is fine; with a large hosted model, K may be more.

### StrEnum → Literal: store the wire form

Doc 2 §3 covered the round-trip. The same applies here:
- Pydantic instances internally hold `BreastProcedure.RADICAL_HYSTERECTOMY`.
- DSPy's prompt sees `Literal["radical_hysterectomy", ...]`.
- The LM emits the literal string `"radical_hysterectomy"`.

**Demos must store outputs in their *literal string form***. If you serialize a pydantic model with `model_dump()`, you get the literal form. If you serialize with `model_dump(mode="json")`, same. Don't `repr` enum members.

Practical rule: build the `outputs` dict by calling `case_model.model_dump(mode="json")` on the validated record, then peel off the keys belonging to this group.

### Demo selection bias

If all your `margins` demos for breast came from one hospital's report template (`"All margins free. Closest margin: anterior, 4 mm."`), retrieval will preferentially fetch them even when a different-template report comes in. The LM then over-mimics the template style.

Mitigations:
- **Source-stratified sampling at write time** — round-robin across known sources.
- **Diversity-aware retrieval at read time** — pick top-K but enforce a similarity diversity constraint (e.g. MMR — maximal marginal relevance). Overkill at first; add only if you observe the bias.

### When no chunks routed to a group: query with the full report

The existing fallback (Doc 3 §5: "no chunks routed → use the full report") needs to apply to *retrieval too*. The integration sketch in §5b does this naturally because `step_input = paragraphs` (the full report) in that case. But if you refactor later, keep this invariant.

### `dspy.Example` is *not* `dict`

`dspy.Example(**fields)` is the right way to build a demo for DSPy. Don't pass a raw dict to `.demos` — DSPy expects its own Example object. The factory call in §5b does this conversion.

### Cache invalidation

The `_CACHE` in `retrieve.py` caches the loaded demos + embeddings keyed by (organ, group). If you `add_demo(...)` mid-process, the cache is stale.

Two options:
- **Process-restart workflow** — accept that demos take effect on the next process. Fine for batch jobs, surprising in a long-running service.
- **Invalidate on write** — `add_demo` clears the cache for that (organ, group). One line of code; do this.

### "What if outputs have nested types like list[BreastMargin]?"

You stored them as a JSON list of dicts (per §2's example). DSPy will render them into the prompt as-is. The LM sees a JSON example of the structured output, which is exactly what we want. No conversion needed.

### Don't store partial-confidence demos

A demo with a missing field, or a hand-fixed value the LM consistently gets wrong, leaks bias. Two rules at write time:
- **All required fields for the group must be present.**
- **Pydantic validation on `outputs` against the per-group output schema must pass.**

You can enforce both in `add_demo(...)` by reconstructing a partial validator. Worth ~10 lines of guard code.

---

## 9. What's deliberately out of scope

- **Compiled DSPy / teleprompters.** Optimization of which demos to include for which queries — a layer on top of retrieval. Tackle after this is working.
- **Domain-knowledge RAG.** Embedding WHO classification, AJCC tables, organ guidelines. Orthogonal: you'd build it alongside, not on top of, demo retrieval.
- **Cross-report validation.** Retrieving similar past cases at validation time to anchor consistency checks. Orthogonal, validates rather than augments.
- **A vector DB.** FAISS / Chroma / pgvector. Premature until the demo store crosses ~50k entries per (organ, group).
- **Online learning.** Updating demos based on production confidence. Premature; needs a confidence model that isn't here yet.

These are all good ideas. They're just not first.

---

## 10. Estimated effort

| Item | Effort |
|---|---|
| `demos/store.py`, `demos/retrieve.py`, `demos/__init__.py` | 1 day |
| Pipeline integration (`demo_retrieval_k=` arg + per-step injection) | half day |
| Cold-start curation script: walk `tests/fixtures/reference/`, emit demos | 1 day |
| `eval_few_shot.py` — control vs treatment, per-group F1 + bootstrap | 1 day |
| First-pass eval, tune K, capped demo length | 2 days |
| Documentation (README in `demos/`, example in `examples/`) | half day |

Total: ~5-6 working days for a first cut you'd be comfortable A/B testing.

---

## 11. What to read next, if you want to build it

- [pipeline_factory.py:60-271](../../src/digital_registrar/pipeline_factory.py#L60-L271) — the integration site.
- [chunking/dspy_embedder_router.py:30-99](../../src/digital_registrar/chunking/dspy_embedder_router.py#L30-L99) — the embedding pattern you'll mirror.
- DSPy docs on `dspy.Example` and the `.demos` attribute of `dspy.Predict`.
- [tests/pipeline/test_v2_parity.py](../../tests/pipeline/test_v2_parity.py) — the eval scaffold to extend.

When you're ready, the change list in §5 is small enough to be a single PR. The bigger PR is the curation: turning the existing gold cases into per-(organ, group) demos. That's where the real iteration happens.

If you've been hesitant to introduce RAG because it sounds heavyweight: this design isn't. It's three small files, two lines in the pipeline, and zero new infrastructure beyond what `dspy.Embedder` already gives you. The hard part is the demo curation, not the code.
