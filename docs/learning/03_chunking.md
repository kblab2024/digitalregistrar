# 03 — Chunking: how it works, how to manipulate it

> Goal: understand what `Chunk` is, how the three chunkers and two routers compose, and how to feed *only* the lymph-node section to the `lymph_nodes` extraction signature (and nothing else).

Before reading: skim [02 — typing & pydantic](02_typing_and_pydantic.md) and [04 — DSPy in depth](04_dspy_in_depth.md) §6 (`dspy.Module`). The chunking layer plugs into `CancerPipelineV2.forward()` and you'll be reading that loop in §5 of this doc.

There's also a short usage cheat sheet in [_chunking_notes.md](../../src/digital_registrar/chunking/_chunking_notes.md) next to the code — that's a quick reference. This doc is the deep dive.

The runnable companion file for this doc is [examples/lymph_node_focus.py](../../examples/lymph_node_focus.py) — see §6.

---

## 0. ELI5: what's a chunk?

A chunk is a **labeled substring** of a pathology report:
- The original character offsets where the slice came from.
- A set of *labels* telling the pipeline which extraction groups this chunk should feed.

If a chunk is labeled `"lymph_nodes"`, the lymph-node extraction signature gets to see it. The margins signature does not.

That's the whole concept. The rest is plumbing: how to split, how to label, how the pipeline filters per signature.

---

## 1. The `Chunk` dataclass

The full definition is in [src/digital_registrar/chunking/protocols.py:23-43](../../src/digital_registrar/chunking/protocols.py#L23-L43):

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

Field by field:

| Field | What it is | Example |
|---|---|---|
| `id` | Stable identifier within one chunking run | `"c0"`, `"c1"`, `"c2"` |
| `text` | The actual substring | `"All margins free of tumor..."` |
| `span` | Char offsets `(start, end)` into the *original* report | `(450, 612)` |
| `labels` | Which extraction groups this chunk feeds | `frozenset({"margins"})` |
| `meta` | Free-form metadata (debugging, GUI highlighting, source info) | `{"section": "margins"}` |

Why `frozen=True` (immutable) + `frozenset` for labels? Because chunks flow through multiple pipeline stages. If anything mutates a chunk midstream, you can't reason about what's happening. Immutability removes that whole class of bug. See [Doc 01 §6](01_class_machinery.md#6-dataclass--auto-generate-init-repr-eq) for the dataclass deep dive.

To "change" a chunk's labels, you call `chunk.with_labels(...)` which returns a *new* `Chunk`. Functional updates, not mutation.

---

## 2. The two protocols

[src/digital_registrar/chunking/protocols.py:46-72](../../src/digital_registrar/chunking/protocols.py#L46-L72):

```python
@runtime_checkable
class Chunker(Protocol):
    name: str
    def chunk(self, report: str) -> list[Chunk]: ...

@runtime_checkable
class Router(Protocol):
    name: str
    def route(self, chunks: list[Chunk], groups: dict[str, str]) -> list[Chunk]: ...
```

**A `Chunker`** takes a report and splits it into chunks. It may or may not label them.

**A `Router`** takes already-split chunks plus group descriptions, and *labels* them.

The two roles separate concerns: splitting (boundaries) vs labeling (which group). Some chunkers self-label (`RegexSectionChunker` knows that everything under `MARGINS:` is the margins section), so they don't need a router. Others split semantically but don't know which group is which — they pair with a router.

For the Protocol/structural-typing mechanics, see [Doc 01 §7](01_class_machinery.md#7-protocol--structural-typing).

---

## 3. The three chunkers

### `ParagraphChunker` — split on blank lines (baseline)

[src/digital_registrar/chunking/paragraph.py:47-56](../../src/digital_registrar/chunking/paragraph.py#L47-L56):

```python
@register_chunker("paragraph")
class ParagraphChunker:
    """Splits on blank lines; one chunk per paragraph."""

    def chunk(self, report: str) -> list[Chunk]:
        spans = paragraph_spans(report)
        return [
            Chunk(id=f"c{i}", text=report[s:e], span=(s, e))
            for i, (s, e) in enumerate(spans)
        ]
```

**Output**: one chunk per paragraph, **no labels**, no metadata beyond id/text/span.

**When to use**: as input to a router. The paragraph chunker by itself is useless for routing because every chunk has empty `labels` — but pair it with `EmbeddingSimilarityRouter` or `DSPyEmbedderRouter` and each chunk gets labels assigned.

### `RegexSectionChunker` — header-based, self-labeling (the practical default)

[src/digital_registrar/chunking/regex_chunker.py:30-39](../../src/digital_registrar/chunking/regex_chunker.py#L30-L39) ships with default patterns matching standard pathology-report headers:

```python
_EOH = r"\s*(?::|$)"
DEFAULT_PATHOLOGY_PATTERNS: dict[str, str] = {
    "gross":       rf"^\s*(?:GROSS\s+(?:DESCRIPTION|EXAMINATION)|SPECIMEN(?:\s+DESCRIPTION)?){_EOH}",
    "microscopic": rf"^\s*MICROSCOPIC(?:\s+(?:DESCRIPTION|EXAMINATION|FINDINGS))?{_EOH}",
    "diagnosis":   rf"^\s*(?:FINAL\s+)?(?:DIAGNOSIS|DIAGNOSES|PATHOLOGIC\s+DIAGNOSIS){_EOH}",
    "margins":     rf"^\s*(?:RESECTION\s+)?MARGINS?{_EOH}",
    "lymph_nodes": rf"^\s*(?:REGIONAL\s+)?LYMPH(?:\s+NODES?)?{_EOH}",
    "staging":     rf"^\s*(?:AJCC\s+(?:CLASSIFICATION|STAGING)|PATHOLOGIC\s+STAGING|TNM\s+CLASSIFICATION|STAGING){_EOH}",
    "biomarkers":  rf"^\s*(?:BIOMARKERS|IMMUNOHISTOCHEMISTRY|IHC(?:\s+STUDIES)?|RECEPTORS?\s+STATUS){_EOH}",
    "comments":    rf"^\s*(?:COMMENTS?|NOTES?|ADDENDUM){_EOH}",
}
```

The chunker walks the report, finds every matching header, sorts them by position, and emits one chunk per header-to-next-header range. Each chunk is **auto-labeled** with the header name.

Critically, **the default keys match the organ `GROUP_INSTRUCTIONS` keys** (`margins`, `lymph_nodes`, `staging`, `biomarkers`). That means: with the default patterns, no router needed — chunks routed to `"lymph_nodes"` naturally feed the `lymph_nodes` extraction step.

When section names don't match group names, you can pass a translation map:

```python
chunker = RegexSectionChunker(
    section_to_groups={"diagnosis": ["nonnested", "grading"]},
)
```

Now any chunk labeled with the regex key `"diagnosis"` will get the labels `frozenset({"nonnested", "grading"})` — feeding both the `nonnested` and `grading` extraction signatures.

Also worth noting: `include_preamble=True` (default) emits a chunk for any text before the first header. Useful when reports have a header block of patient info that doesn't fall under any specific section.

### `EmbeddingBoundaryChunker` — semantic, no labels

[src/digital_registrar/chunking/embedding_chunker.py:65-141](../../src/digital_registrar/chunking/embedding_chunker.py#L65-L141): TextTiling-style chunker.

Algorithm:
1. Split report into paragraphs.
2. Embed each paragraph (default: `sentence-transformers/all-MiniLM-L6-v2`, ~22 MB local model).
3. Compute cosine similarity between adjacent paragraph embeddings.
4. Place a boundary where similarity drops below `mean - boundary_z * std`.
5. Greedily group paragraphs into chunks, respecting `min_chunk_paragraphs` and `max_chunk_paragraphs`.

**Output**: chunks with no labels (they need a router). Metadata includes `n_paragraphs` and the source paragraph indices.

**When to use**: reports without conventional headers. Free-text narratives where regex won't catch transitions.

The implementation is in pure NumPy on top of `sentence-transformers` — see [embedding_chunker.py:54-62](../../src/digital_registrar/chunking/embedding_chunker.py#L54-L62) for the encode step and lines 112-119 for the boundary detection.

---

## 4. The two routers

Routers assign group labels to chunks based on similarity to each group's description.

### `EmbeddingSimilarityRouter` — local sentence-transformers

[src/digital_registrar/chunking/embedding_chunker.py:174-219](../../src/digital_registrar/chunking/embedding_chunker.py#L174-L219):

```python
@register_router("embedding_similarity")
class EmbeddingSimilarityRouter:
    def __init__(self, *, model_name=_DEFAULT_MODEL, top_k=2, min_similarity=0.25):
        ...

    def route(self, chunks: list[Chunk], groups: dict[str, str]) -> list[Chunk]:
        if not chunks or not groups:
            return list(chunks)
        group_names = list(groups)
        group_texts = [groups[g] for g in group_names]

        chunk_emb = _encode(self._model, [c.text for c in chunks])
        group_emb = _encode(self._model, group_texts)
        sims = chunk_emb @ group_emb.T   # (n_chunks, n_groups) cosine

        k = min(self.top_k, len(group_names))
        out: list[Chunk] = []
        for i, chunk in enumerate(chunks):
            row = sims[i]
            top_idx = np.argsort(-row)[:k]
            picked = {
                group_names[j] for j in top_idx
                if float(row[j]) >= self.min_similarity
            }
            merged = chunk.labels | frozenset(picked)
            out.append(chunk.with_labels(merged))
        return out
```

Two knobs:
- `top_k` (default 2) — each chunk gets at most K group labels.
- `min_similarity` (default 0.25) — chunks below this threshold get NO label.

A chunk's existing labels are **merged with** new picks (`chunk.labels | frozenset(picked)`) — the router augments, doesn't overwrite.

### `DSPyEmbedderRouter` — anything via `dspy.Embedder`

[src/digital_registrar/chunking/dspy_embedder_router.py:30-99](../../src/digital_registrar/chunking/dspy_embedder_router.py#L30-L99):

Same scoring algorithm; different embedding backend. Pass `model=` and DSPy/LiteLLM resolves it (OpenAI, Cohere, Voyage, Ollama, etc.), or pass a `embedder=` callable directly for tests.

```python
router = DSPyEmbedderRouter(model="openai/text-embedding-3-small", top_k=2)
# or
router = DSPyEmbedderRouter(model="sentence-transformers/all-MiniLM-L6-v2", top_k=2)
# or — for tests / fakes:
router = DSPyEmbedderRouter(embedder=my_callable, top_k=2)
```

For embedding-backend specifics, see [Doc 04 §9](04_dspy_in_depth.md#9-dspyembedder--embedding-text-into-vectors).

---

## 5. The filter loop — where chunks become signature input

This is the heart of "send only X chunks to signature Y". From [pipeline_factory.py:143-266](../../src/digital_registrar/pipeline_factory.py#L143-L266):

### Step 1: compute routed chunks (once per organ, not per step)

```python
def _maybe_chunk(self, report, organ, logger):
    if self._routing_mode != "filter" or self._chunker is None:
        return None        # routing off → return None
    if not isinstance(report, str):
        logger.warning(
            "routing_mode=filter requires a `report: str` input; got "
            "list[str] — falling back to no routing.",
        )
        return None        # paragraphs in, can't chunk
    chunks = list(self._chunker.chunk(report))
    if self._chunk_router is not None:
        groups = EXTRACTION_META[organ]["groups"]
        chunks = list(self._chunk_router.route(chunks, groups))
    return chunks
```

Three possible returns:
- `None` — routing is off (callers fall back to full report).
- `[]` (empty list) — routing on but no chunks produced. Per-step loop falls back to full report per group.
- `[Chunk, Chunk, ...]` — labeled chunks ready for filtering.

### Step 2: per-step filter

```python
# pipeline_factory.py:223-256
routed_chunks = self._maybe_chunk(report, rsp.cancer_category, logger)
if routed_chunks is not None:
    out["chunks"] = [
        {"id": c.id, "span": list(c.span), "labels": sorted(c.labels), "meta": c.meta}
        for c in routed_chunks
    ]
    out["step_chunks"] = {}

extractors = self._get_extractors(rsp.cancer_category)
for step, predictor in extractors:
    step_input = paragraphs
    if routed_chunks is not None:
        picked = [c for c in routed_chunks if step.group in c.labels]
        if picked:
            step_input = [c.text for c in picked]
        else:
            logger.warning(
                "no chunks routed to group %r; falling back to full report",
                step.group,
            )
        out["step_chunks"][step.name] = [c.id for c in picked]
    try:
        pred = predictor(report=step_input, report_jsonized=json_report)
        ...
```

**The filter predicate** (line 248):

```python
picked = [c for c in routed_chunks if step.group in c.labels]
```

That's it. For each extraction step, pick the chunks whose `labels` contain that step's group tag.

**The fallback**: if no chunks match, use the full report. The pipeline still produces an output; it just doesn't get the focused-prompt benefit for that group. Logged as WARNING so you notice.

**Output enrichment**: `out["step_chunks"]` is a dict mapping `{step_name: [chunk_ids_fed]}`. This is what you read in `examples/lymph_node_focus.py` (next section) to verify routing worked.

---

## 6. Worked example: feed only lymph-node chunks to the lymph-node signature

Three flavors, in increasing manual control. The runnable companion file
[examples/lymph_node_focus.py](../../examples/lymph_node_focus.py) implements all three against a single sample report — run it and read `out["step_chunks"]` to see which chunks fed which step.

### Way 1 — the easy way: `RegexSectionChunker` (default patterns)

If your report has headers like `LYMPH NODES:` or `REGIONAL LYMPH NODES:`, this works out of the box.

```python
from digital_registrar.chunking import RegexSectionChunker
from digital_registrar.pipeline_factory import run_cancer_pipeline_v2, setup_pipeline_v2

setup_pipeline_v2("gpt")  # any model from models/common.py

out, elapsed = run_cancer_pipeline_v2(
    report=report_text,
    chunker=RegexSectionChunker(),
    routing_mode="filter",
)

print(out["step_chunks"])
# expected (approximate):
# {
#   "BreastCancer__nonnested":   [],
#   "BreastCancer__staging":     [],
#   "BreastCancer__margins":     ["c2"],
#   "BreastCancer__lymph_nodes": ["c3"],
#   "BreastCancer__biomarkers":  [],
#   ...
# }
```

Steps whose group name has no matching label get an empty chunk list — the pipeline falls back to the full report for those (with a WARNING log).

### Way 2 — the custom-router way: embedding similarity

If headers don't exist or are inconsistent, use an embedding router. You're trading determinism for flexibility.

```python
from digital_registrar.chunking import ParagraphChunker, DSPyEmbedderRouter
from digital_registrar.pipeline_factory import run_cancer_pipeline_v2

router = DSPyEmbedderRouter(
    model="sentence-transformers/all-MiniLM-L6-v2",
    top_k=2,
    min_similarity=0.3,
)

out, elapsed = run_cancer_pipeline_v2(
    report=report_text,
    chunker=ParagraphChunker(),
    chunk_router=router,
    routing_mode="filter",
)

print(out["step_chunks"])
```

The router reads `EXTRACTION_META[organ]["groups"]` — the per-group `GROUP_INSTRUCTIONS` text — and uses each group's instruction as the query against each chunk. Chunks scoring high on `"you need to extract... lymph_nodes..."` get the `lymph_nodes` label.

To **focus** routing on lymph nodes (e.g. ensure paragraphs talking about lymph nodes always make it), you can use a tighter `min_similarity` or a higher `top_k` — but the easiest control point is to override the group's description for routing purposes (a future enhancement; today the router uses whatever's in `GROUP_INSTRUCTIONS`).

### Way 3 — the fully manual way: custom chunker

When you want total control over chunking *and* labeling, write your own chunker.

```python
import re
from digital_registrar.chunking import Chunk, register_chunker, ParagraphChunker
from digital_registrar.pipeline_factory import run_cancer_pipeline_v2

@register_chunker("lymph_node_focus_only")
class LymphNodeFocusOnlyChunker:
    """Tag *only* paragraphs mentioning lymph nodes; leave the rest unlabeled."""

    LN_TERMS = re.compile(
        r"\b(?:lymph\s*nodes?|LN|sentinel|axillary|station\s*\d+|N\d|micrometastasis)\b",
        re.IGNORECASE,
    )

    def chunk(self, report: str) -> list[Chunk]:
        base = ParagraphChunker().chunk(report)
        out = []
        for c in base:
            if self.LN_TERMS.search(c.text):
                out.append(c.with_labels(frozenset({"lymph_nodes"})))
            else:
                out.append(c)   # unlabeled; falls back to full report
        return out

out, elapsed = run_cancer_pipeline_v2(
    report=report_text,
    chunker=LymphNodeFocusOnlyChunker(),
    routing_mode="filter",
)
```

What this guarantees:
- The `lymph_nodes` extraction step sees *only* the paragraphs that mention lymph-node terms.
- Every other step (margins, biomarkers, staging, nonnested, ...) sees the *full report* (fallback path).

That's the most surgical version of "feed only these chunks to this signature".

You can extend this further: assign multiple labels, use the chunk's `meta` dict to attach a score, or apply different scoring per organ. See `_chunking_notes.md` next to the code for more tips.

---

## 7. Extending — adding your own Chunker or Router

The pattern is the same for both: decorate the class with `@register_chunker(name)` or `@register_router(name)`. After registration:

- The GUI / playground's dropdown picks it up automatically.
- You can resolve it by name via `get_chunker("name")` or `get_router("name")`.

```python
from digital_registrar.chunking import Chunk, register_chunker

@register_chunker("by_sentence")
class SentenceChunker:
    """One chunk per sentence (rough heuristic)."""

    def chunk(self, report: str) -> list[Chunk]:
        sentences = report.split(". ")   # naive, but illustrative
        out = []
        cursor = 0
        for i, s in enumerate(sentences):
            start = report.find(s, cursor)
            end = start + len(s)
            out.append(Chunk(id=f"c{i}", text=s, span=(start, end)))
            cursor = end
        return out
```

And for routers:

```python
from digital_registrar.chunking import Chunk, register_router

@register_router("first_chunk_only")
class FirstChunkRouter:
    """Crude: label only the first chunk with every group. For debugging."""

    def route(self, chunks: list[Chunk], groups: dict[str, str]) -> list[Chunk]:
        if not chunks:
            return chunks
        labeled_first = chunks[0].with_labels(frozenset(groups))
        return [labeled_first] + list(chunks[1:])
```

A class doesn't need to inherit from `Chunker` / `Router` — the Protocols are structural (see [Doc 01 §7](01_class_machinery.md#7-protocol--structural-typing)). It only needs the right method shape.

---

## 8. The "GUI playground" connection

[apps/_shared/playground/](../../apps/_shared/playground/) (a Streamlit widget) reads the registries directly:

- The chunker dropdown is populated from `CHUNKER_REGISTRY.keys()`.
- The router dropdown is populated from `ROUTER_REGISTRY.keys()`.

So when you `@register_chunker("my_thing")`, you don't have to edit any GUI code — the dropdown updates on next reload. That's the value of the registry decorator pattern from [Doc 01 §8](01_class_machinery.md#8-class-decorators-that-register-a-class).

---

## 9. Gotchas

### `Chunk.labels` is a `frozenset`, not a `set` or `list`

You can't `chunk.labels.add(...)` — frozensets are immutable. Use `chunk.with_labels(chunk.labels | frozenset({"new_label"}))` to add a label functionally.

The reason is the chunk itself is `@dataclass(frozen=True)` (see Doc 1 §6). Immutability all the way down.

### `routing_mode="filter"` requires `report: str`, not `list[str]`

```python
# OK — chunker sees the raw text
run_cancer_pipeline_v2(report="...report text...", chunker=..., routing_mode="filter")

# Logs WARNING and silently disables routing
run_cancer_pipeline_v2(report=["paragraph 1", "paragraph 2"], chunker=..., routing_mode="filter")
```

If you pre-split paragraphs and pass a list, the chunker has no character offsets to compute from. The pipeline warns and falls through to no-routing.

### No chunks → full report fallback (logged at WARNING)

If your chunker / router fails to label any chunks for a particular group, the pipeline runs that extraction step against the full report. You don't get an error; you do get a WARNING log line:

```
no chunks routed to group 'lymph_nodes'; falling back to full report
```

If you're chasing a routing bug, grep for that exact log.

### Regex patterns are applied case-insensitive and multiline

[regex_chunker.py:82-85](../../src/digital_registrar/chunking/regex_chunker.py#L82-L85):
```python
self._compiled = {
    label: re.compile(p, re.IGNORECASE | re.MULTILINE)
    for label, p in self.pattern_map.items()
}
```

`re.MULTILINE` is why `^\s*MARGINS` matches "MARGINS" at the start of *any* line, not just the start of the document. If your custom pattern relies on a different anchoring rule, account for that.

### Overlapping header matches: the longer one wins

[regex_chunker.py:141-154](../../src/digital_registrar/chunking/regex_chunker.py#L141-L154):
```python
def _dedupe_overlapping(hits):
    """Drop hits whose header line is fully consumed by an earlier longer hit."""
    ...
    if h.header_start < kept[-1].body_start:
        # h's header is inside the previous header line; pick the longer.
        prev = kept[-1]
        if (h.body_start - h.header_start) > (prev.body_start - prev.header_start):
            kept[-1] = h
        continue
    ...
```

If two patterns both match at the same location — say `MARGINS` and `MARGINS AND LYMPH NODES` — the longer match wins. This avoids spurious double-labeling when one header is a substring of another.

### A chunk can have multiple labels

`labels: frozenset[str]` is a *set*. A chunk labeled `frozenset({"margins", "lymph_nodes"})` will be fed to both the margins extraction step AND the lymph-node extraction step. That's intentional — some report paragraphs genuinely talk about both.

### Embedding routers are non-deterministic across runs only via the embedder's nondeterminism

`sentence-transformers/all-MiniLM-L6-v2` is deterministic on a fixed device. OpenAI's `text-embedding-3-small` is also deterministic on the API side. But: numerical noise in `np.argsort` ties can occasionally flip the top-K when scores are very close. Set a higher `min_similarity` if you need rock-solid stability.

### `dspy.Embedder` requires `dspy.configure` to have been called for some providers

`DSPyEmbedderRouter(model="openai/...")` works as long as the LiteLLM environment can find an API key (set `OPENAI_API_KEY`). For Ollama-routed embeddings, the local Ollama process must be up. Errors are usually clear ("api key missing", "connection refused").

---

## 10. Cheat sheet

| Want | Use |
|---|---|
| Header-based, deterministic | `RegexSectionChunker()` (no router needed) |
| Custom headers | `RegexSectionChunker(pattern_map={...})` |
| Header → multiple groups | `RegexSectionChunker(section_to_groups={...})` |
| Semantic boundaries, no headers | `EmbeddingBoundaryChunker()` + a router |
| Paragraph chunks + ML routing | `ParagraphChunker()` + `EmbeddingSimilarityRouter()` |
| Same, but with OpenAI / other backends | `ParagraphChunker()` + `DSPyEmbedderRouter(model="...")` |
| Total control over what each chunk feeds | custom `@register_chunker` class |
| Verify routing worked | `out["step_chunks"]` after the pipeline run |

For the runnable end-to-end demo, see [examples/lymph_node_focus.py](../../examples/lymph_node_focus.py).

---

## 11. What's next

The routing layer is *one* lever for focused extraction. The next lever — completely independent — is **what demos go into each per-group prompt**. That's covered in [Doc 5 — few-shot demo retrieval](05_few_shot_demo_retrieval.md). Read it after you're comfortable with this chunking model; the two layers compose naturally.
