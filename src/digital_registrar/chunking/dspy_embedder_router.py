"""DSPy-backed embedding router.

Routes chunks to extraction groups using ``dspy.Embedder`` so the
embedding backend (LiteLLM provider, ``sentence-transformers``,
custom callable) is selected by DSPy config rather than hard-coded.

Pair with any :class:`Chunker` (typically ``ParagraphChunker`` or
``EmbeddingBoundaryChunker``). Same scoring rule as
:class:`EmbeddingSimilarityRouter` — cosine, top-K, min-threshold —
just a different embedding backend.
"""
from __future__ import annotations

import logging

import numpy as np

from .protocols import Chunk
from .registry import register_router

logger = logging.getLogger(__name__)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


@register_router("dspy_embedder")
class DSPyEmbedderRouter:
    """Label chunks via ``dspy.Embedder``-based cosine similarity.

    Parameters
    ----------
    model
        Forwarded to ``dspy.Embedder`` — e.g. ``"openai/text-embedding-3-small"``
        or ``"sentence-transformers/all-MiniLM-L6-v2"`` (DSPy resolves via
        LiteLLM). If ``None``, expects ``embedder`` to be passed directly.
    embedder
        Pre-built ``dspy.Embedder`` (or any callable mapping
        ``list[str] -> 2D ndarray``). Mutually exclusive with ``model``.
    top_k
        Maximum number of group labels assigned per chunk.
    min_similarity
        Cosine threshold; labels with score < this are dropped.
    """

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
        if model is not None and embedder is not None:
            raise ValueError("Pass only one of `model` or `embedder`")
        self.top_k = max(1, int(top_k))
        self.min_similarity = float(min_similarity)
        if embedder is not None:
            self._embedder = embedder
        else:
            import dspy
            self._embedder = dspy.Embedder(model)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._embedder(texts)
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(
                f"Embedder returned ndim={arr.ndim} array; expected 2D "
                f"(n_texts, dim)."
            )
        return _l2_normalize(arr)

    def route(self, chunks: list[Chunk], groups: dict[str, str]) -> list[Chunk]:
        if not chunks or not groups:
            return list(chunks)
        group_names = list(groups)
        group_texts = [groups[g] for g in group_names]

        chunk_emb = self._embed([c.text for c in chunks])
        group_emb = self._embed(group_texts)
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
            out.append(chunk.with_labels(chunk.labels | frozenset(picked)))
        return out


__all__ = ["DSPyEmbedderRouter"]
