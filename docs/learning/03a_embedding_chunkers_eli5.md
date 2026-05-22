# 03a — Embedding chunkers + routers, ELI5

> Companion to [03_chunking.md](03_chunking.md). That doc is the reference;
> this one builds the mental model from scratch for the embedding-based
> pieces specifically — `EmbeddingBoundaryChunker`,
> `EmbeddingSimilarityRouter`, and `DSPyEmbedderRouter` — and shows how to
> tune them when you want to find the LN section, or chunk shorter.

Three pieces matter: **embeddings**, the **boundary chunker**, and the
**similarity router**. The DSPy variant is just a swap of who does the
embedding.

## 1. Embeddings: "meaning fingerprints"

An embedding model takes a piece of text and turns it into a list of
~384 numbers (a vector). Two texts about similar things produce vectors
that point in nearly the same direction. So:

- `"All margins are negative"` → some vector A
- `"Resection margins clear"` → some vector B  ← A and B point the same way
- `"Lymph node 0/3 positive"` → some vector C  ← C points a *different* way

We measure "same direction" with **cosine similarity** — a number from
−1 (opposite) to 1 (identical). 0.9 = very similar, 0.2 = barely related.

That's it. Everything below is just clever bookkeeping on top of these
numbers.

## 2. `EmbeddingBoundaryChunker` — "cut where the topic changes"

Goal: group adjacent paragraphs that talk about the same thing.

Imagine the report has paragraphs P1, P2, P3, P4, P5. We:

1. Embed each → vectors v1, v2, v3, v4, v5.
2. Compute cosine similarity between **adjacent pairs**:
   - sim(P1, P2) = 0.85
   - sim(P2, P3) = 0.80
   - sim(P3, P4) = **0.30**  ← big drop, topic changed!
   - sim(P4, P5) = 0.82
3. Look at the distribution of those similarities. Mean ≈ 0.69, std ≈ 0.26.
4. With `boundary_z=1.0`, the threshold = 0.69 − 1.0 × 0.26 = **0.43**.
   Any drop below 0.43 is a "boundary."
5. So P3 → P4 is a boundary. Chunks become `[P1, P2, P3]` and `[P4, P5]`.

`min_chunk_paragraphs` / `max_chunk_paragraphs` clamp the size so a
single weird paragraph can't become its own micro-chunk, and a long
uniform run gets force-split when it gets too big.

**Want shorter chunks?** Two knobs:

```python
EmbeddingBoundaryChunker(
    max_chunk_paragraphs=2,   # default 6 → force-cut every 2 paragraphs
    boundary_z=0.3,           # default 1.0 → lower threshold = more boundaries detected
)
```

Or just use `ParagraphChunker()` — one chunk per paragraph, the maximum
granularity.

## 3. `EmbeddingSimilarityRouter` — "label each chunk by what it resembles"

Goal: decide which extraction group each chunk should feed.

You give it `groups = {"margins": "...description...", "lymph_nodes": "...description..."}`.
It:

1. Embeds every chunk's text → list of vectors `C[i]`.
2. Embeds every group's description → list of vectors `G[j]`.
3. Builds a similarity table:

   |              | margins | lymph_nodes | biomarkers |
   |--------------|---------|-------------|------------|
   | chunk c0     | 0.78    | 0.22        | 0.15       |
   | chunk c1     | 0.20    | **0.81**    | 0.18       |
   | chunk c2     | 0.30    | 0.25        | 0.72       |

4. For each chunk, picks the **top-K** groups whose similarity ≥
   `min_similarity`.

With defaults `top_k=2, min_similarity=0.25`:

- c0 → `{margins}` (lymph_nodes 0.22 is below 0.25, dropped)
- c1 → `{lymph_nodes, margins}` (only margins clears 0.25 among the rest)
- c2 → `{biomarkers}`

That's how the router decides "this chunk goes to the lymph_nodes
extractor."

## 4. `DSPyEmbedderRouter` — same math, different embedder

`EmbeddingSimilarityRouter` calls `sentence-transformers` locally.
`DSPyEmbedderRouter` calls `dspy.Embedder(model="...")`, which can wrap
**any** embedding provider (OpenAI, Cohere, local Ollama, your own
callable). Scoring, top-K, threshold — all identical. Use it when:

- You want to swap embedding providers via DSPy config.
- You want to bill embeddings to the same provider as your LM.
- Your reports are in a language the local MiniLM doesn't handle well,
  and you want a multilingual provider.

Otherwise, the local one is faster and free.

## 5. "How do I find the lymph node chunk?"

You don't need new code — the router already does this. But you can
also do it ad-hoc for inspection:

```python
from digital_registrar.chunking import (
    EmbeddingBoundaryChunker, EmbeddingSimilarityRouter,
)
from digital_registrar.schemas.extraction import EXTRACTION_META

chunker = EmbeddingBoundaryChunker(max_chunk_paragraphs=2)
router  = EmbeddingSimilarityRouter(top_k=2, min_similarity=0.25)

chunks = chunker.chunk(report_text)
groups = EXTRACTION_META["breast"]["groups"]   # has "lymph_nodes" key
chunks = router.route(chunks, groups)

ln_chunks = [c for c in chunks if "lymph_nodes" in c.labels]
for c in ln_chunks:
    print(c.id, c.span, c.text[:80])
```

If the "right" chunk isn't labeled lymph_nodes, three things to try, in
order:

1. **Lower the threshold.** `EmbeddingSimilarityRouter(min_similarity=0.15)`.
   The general MiniLM model rarely scores medical phrases above 0.4;
   0.25 may already be too strict.
2. **Improve the query (group description).** The current
   `EXTRACTION_META["breast"]["groups"]["lymph_nodes"]` is a long
   instruction. The router uses that whole string as the "query" —
   long generic instructions embed poorly. Try passing a tighter
   description just for routing:

   ```python
   tight_groups = {
       "lymph_nodes": "lymph node involvement, sentinel node, axillary nodes, station, positive nodes, examined",
       "margins":     "surgical resection margins, distance from tumor to inked margin, positive or negative for tumor",
   }
   chunks = router.route(chunks, tight_groups)
   ```
   Keyword-heavy strings embed better than long sentences for this kind
   of "topic matching."
3. **Chunk shorter.** A 6-paragraph chunk mixing margins + LN content
   will score in between for both. Shrink with `max_chunk_paragraphs=1`
   or `ParagraphChunker()` so each paragraph votes independently.

## 6. "Chunk even shorter" — full recipe

```python
from digital_registrar.chunking import ParagraphChunker, EmbeddingSimilarityRouter

# Maximum granularity: every paragraph is its own chunk.
chunker = ParagraphChunker()

# More aggressive labeling.
router = EmbeddingSimilarityRouter(top_k=3, min_similarity=0.15)
```

Or, if you want semantic clustering but at small sizes:

```python
chunker = EmbeddingBoundaryChunker(
    min_chunk_paragraphs=1,
    max_chunk_paragraphs=2,
    boundary_z=0.3,    # detect more boundaries
)
```

The trade-off: shorter chunks = more precise routing but more LM calls
(when you eventually feed them downstream) and less context per call.
For lymph node extraction specifically, very short chunks are usually
fine because LN content is self-contained.

## 7. Quick sanity test (no LM needed)

Paste this in a REPL with your real report to see numbers:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
queries = {
    "lymph_nodes": "lymph nodes sentinel axillary positive station",
    "margins":     "surgical margins resection distance involved",
}
chunks_text = [...]   # your chunks here
qv = m.encode(list(queries.values()), normalize_embeddings=True)
cv = m.encode(chunks_text, normalize_embeddings=True)
sims = cv @ qv.T   # rows = chunks, cols = queries
print(np.round(sims, 2))
```

You'll see the raw scores. That's exactly what the router is
thresholding on — once you can read this table you can predict what
labels every knob change will produce.

## Where to go next

- [03_chunking.md](03_chunking.md) — the reference: full `Chunk` /
  `Chunker` / `Router` contracts, the filter loop in `pipeline_factory.py`,
  and three worked examples of feeding only the LN chunk to the
  lymph-node signature.
- [04_dspy_in_depth.md](04_dspy_in_depth.md) §`Embedder` — what
  `dspy.Embedder` actually does under the hood, including LiteLLM
  provider strings.
