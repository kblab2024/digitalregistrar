"""Local-embedding chunker + similarity router.

Uses a small sentence-transformers model (default
``sentence-transformers/all-MiniLM-L6-v2`` — 22 MB, CPU-fast) to:

- segment a report into semantically coherent chunks
  (:class:`EmbeddingBoundaryChunker`), and
- label chunks with the extraction groups they most resemble
  (:class:`EmbeddingSimilarityRouter`).

No DSPy / LM calls. ``sentence-transformers`` is a soft dependency: the
package imports successfully without it, but instantiating either class
raises with a pip-install hint.
"""
from __future__ import annotations

import logging

import numpy as np

from .paragraph import paragraph_spans
from .protocols import Chunk
from .registry import register_chunker, register_router

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE: dict[str, object] = {}


def _load_model(name: str):
    """Lazy, cached sentence-transformers model loader.

    Importing ``sentence_transformers`` itself takes ~1 s on first call;
    the module is only imported when a caller actually instantiates an
    embedding-backed chunker/router.
    """
    cached = _MODEL_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"{e.name} is required for embedding-based chunking. "
            "Install with `pip install sentence-transformers` "
            "or `pip install 'digital-registrar[chunking]'`."
        ) from e
    model = SentenceTransformer(name)
    _MODEL_CACHE[name] = model
    return model


def _encode(model, texts: list[str]) -> np.ndarray:
    """Encode ``texts`` to L2-normalised float32 vectors."""
    arr = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(arr, dtype=np.float32)


@register_chunker("embedding_boundary")
class EmbeddingBoundaryChunker:
    """TextTiling-style semantic chunker.

    Embeds each paragraph, computes cosine similarity between adjacent
    paragraphs, and places a chunk boundary where similarity drops below
    ``mean - boundary_z * std``. Output chunks have no ``labels`` — pair
    with :class:`EmbeddingSimilarityRouter` (or any other router).

    Parameters
    ----------
    model_name
        sentence-transformers model identifier. The default 22 MB
        all-MiniLM-L6-v2 is fast and adequate for routing-style tasks.
    min_chunk_paragraphs, max_chunk_paragraphs
        Bound chunk size in paragraphs.
    boundary_z
        How many std-devs below the mean similarity counts as a
        boundary. Higher = fewer / larger chunks.
    """

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL,
        min_chunk_paragraphs: int = 1,
        max_chunk_paragraphs: int = 6,
        boundary_z: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.min_chunk_paragraphs = max(1, int(min_chunk_paragraphs))
        self.max_chunk_paragraphs = max(self.min_chunk_paragraphs, int(max_chunk_paragraphs))
        self.boundary_z = float(boundary_z)
        # Eagerly resolve the model so an instantiation-time failure is
        # surfaced now rather than at the first .chunk() call.
        self._model = _load_model(model_name)

    def chunk(self, report: str) -> list[Chunk]:
        spans = paragraph_spans(report)
        if not spans:
            return []
        paragraphs = [report[s:e] for s, e in spans]
        if len(paragraphs) == 1:
            s, e = spans[0]
            return [Chunk(id="c0", text=paragraphs[0], span=(s, e),
                          meta={"n_paragraphs": 1})]

        embeddings = _encode(self._model, paragraphs)
        # Adjacent cosine sim (vectors already L2-normalised).
        sims = (embeddings[:-1] * embeddings[1:]).sum(axis=1)

        threshold = float(sims.mean() - self.boundary_z * sims.std()) if sims.size > 1 else -1.0
        # Boundary at index i means a new chunk starts AT paragraph i+1
        # (so the drop is between i and i+1).
        boundary_set = {int(i) + 1 for i, s in enumerate(sims) if s < threshold}

        groups = _greedy_segment(
            len(paragraphs),
            boundary_set,
            min_size=self.min_chunk_paragraphs,
            max_size=self.max_chunk_paragraphs,
        )

        chunks: list[Chunk] = []
        for cid, idxs in enumerate(groups):
            s = spans[idxs[0]][0]
            e = spans[idxs[-1]][1]
            chunks.append(Chunk(
                id=f"c{cid}",
                text=report[s:e],
                span=(s, e),
                meta={
                    "n_paragraphs": len(idxs),
                    "paragraph_indices": idxs,
                },
            ))
        return chunks


def _greedy_segment(
    n: int,
    boundaries: set[int],
    *,
    min_size: int,
    max_size: int,
) -> list[list[int]]:
    """Group ``range(n)`` into runs honouring boundary hints and size bounds.

    ``boundaries`` is a set of paragraph indices where a new chunk should
    start. Boundaries are honoured only when the current chunk has at
    least ``min_size`` members; chunks are force-flushed when they reach
    ``max_size``.
    """
    groups: list[list[int]] = []
    current: list[int] = [0]
    for i in range(1, n):
        if i in boundaries and len(current) >= min_size:
            groups.append(current)
            current = [i]
            continue
        current.append(i)
        if len(current) >= max_size:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


@register_router("embedding_similarity")
class EmbeddingSimilarityRouter:
    """Label each chunk with the top-K groups by cosine similarity.

    Embeds each chunk and each group description, then assigns the
    top-K groups whose similarity exceeds ``min_similarity``. If no
    group clears the threshold the chunk's ``labels`` remain empty —
    the pipeline's routing fallback then sends the full report to any
    group with zero routed chunks.
    """

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL,
        top_k: int = 2,
        min_similarity: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.top_k = max(1, int(top_k))
        self.min_similarity = float(min_similarity)
        self._model = _load_model(model_name)

    def route(self, chunks: list[Chunk], groups: dict[str, str]) -> list[Chunk]:
        if not chunks or not groups:
            return list(chunks)
        group_names = list(groups)
        group_texts = [groups[g] for g in group_names]

        chunk_emb = _encode(self._model, [c.text for c in chunks])
        group_emb = _encode(self._model, group_texts)
        # (n_chunks, n_groups) cosine sim — both sides are L2-normalised.
        sims = chunk_emb @ group_emb.T

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


__all__ = [
    "EmbeddingBoundaryChunker",
    "EmbeddingSimilarityRouter",
]
