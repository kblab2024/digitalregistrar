# Chunking & routing — notes for users

This package lets you split a pathology report into chunks and route
each chunk to one or more extraction groups (e.g. send only the
`MARGINS:` section to the `margins` extractor, only the `LYMPH NODES:`
section to `lymph_nodes`). Two reasons you might want this:

1. **Smaller prompts.** Each per-group extractor sees less text — less
   noise, more focus.
2. **Faster iteration.** In the GUI playground, you can run a single
   group's signature against a single chunk and inspect prompt + output.

The pipeline still works exactly as before unless you pass
`routing_mode="filter"` to `run_cancer_pipeline_v2`. Default is `"off"`.

## Pick a chunker

| Chunker | When to use | Cost |
|---|---|---|
| `regex_section` | Pathology reports with conventional headers (`MARGINS:`, `LYMPH NODES:`, ...). Most predictable. | Free. |
| `embedding_boundary` | Free-form reports without explicit headers. Splits on semantic shifts. | One local embedding pass (~5 ms / paragraph on CPU). Needs `sentence-transformers`. |
| `paragraph` | Baseline — one chunk per paragraph. Pair with a router. | Free. |

Start with `regex_section`. It's deterministic, easy to debug, and the
default `pattern_map` already covers the common pathology headers. If
your reports use different conventions, override `pattern_map`:

```python
from digital_registrar.chunking import RegexSectionChunker

chunker = RegexSectionChunker(pattern_map={
    "margins":     r"^\s*RESECTION\s+MARGIN(?:S)?\s*(?::|$)",
    "lymph_nodes": r"^\s*REGIONAL\s+NODES?\s*(?::|$)",
    # ... add your own
})
```

If your section names don't match the organ's group names, pass
`section_to_groups`:

```python
chunker = RegexSectionChunker(
    section_to_groups={"diagnosis": ["nonnested", "grading"]},
)
```

## Pick a router

You only need a router if your chunker doesn't pre-label chunks.
`RegexSectionChunker` self-labels — no router needed.

| Router | Backed by | When to use |
|---|---|---|
| `embedding_similarity` | local `sentence-transformers` | No LM/API calls. Works offline. |
| `dspy_embedder` | `dspy.Embedder` (any provider) | When you want to swap embedding providers via DSPy config. |

Both score each chunk against each group's description (cosine
similarity), and assign the top-K groups whose score exceeds
`min_similarity`. Default `top_k=2`, `min_similarity=0.25`.

If a chunk doesn't clear `min_similarity` for any group, it gets no
labels. That's fine — the pipeline falls back to the full report for
any group with zero routed chunks (with a warning).

## Knobs to tune

- **regex_section** `pattern_map` keys → use the organ's group names
  directly to skip routing. Add `section_to_groups` only when names diverge.
- **embedding_boundary**
  - `min_chunk_paragraphs` (default 1), `max_chunk_paragraphs` (default 6)
    — clamp chunk size.
  - `boundary_z` (default 1.0) — how many std-devs below mean similarity
    counts as a boundary. Higher = fewer, larger chunks.
- **embedding_similarity / dspy_embedder**
  - `top_k` (default 2) — how many groups each chunk can feed.
  - `min_similarity` (default 0.25) — drop labels below this cosine score.

## Add a custom chunker

Decorate a class with `@register_chunker(name)`:

```python
from digital_registrar.chunking import register_chunker, Chunk

@register_chunker("my_chunker")
class MyChunker:
    def chunk(self, report: str) -> list[Chunk]:
        # ... your splitting logic ...
        return [Chunk(id="c0", text=..., span=(0, 100), labels=frozenset({"margins"}))]
```

The GUI's chunker dropdown picks it up automatically on next reload.

## Pipeline opt-in

```python
from digital_registrar.pipeline_factory import run_cancer_pipeline_v2
from digital_registrar.chunking import RegexSectionChunker

out, elapsed = run_cancer_pipeline_v2(
    report=report_text,
    chunker=RegexSectionChunker(),
    routing_mode="filter",
)
print(out["chunks"])       # list of chunk records (id, span, labels)
print(out["step_chunks"])  # {step_name: [chunk_ids fed to that step]}
```

`routing_mode="off"` (default) preserves current behaviour exactly —
every extractor sees the full report.

## Embedding model choice

The default `sentence-transformers/all-MiniLM-L6-v2` is 22 MB and runs
at ~5 ms / paragraph on CPU. Adequate for routing-style tasks over
English pathology text. Do not reach for medical-domain embeddings
until you have evidence the general one fails — it usually doesn't for
*routing*, where the distinction between e.g. "lymph nodes" and
"margins" descriptions is coarse.

If you want to swap providers (OpenAI, Cohere, a local Ollama embedder),
use `dspy_embedder` with a `model=` string DSPy / LiteLLM understands.
