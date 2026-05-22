"""Pluggable chunking + routing for extraction.

Two concerns, kept separate so users can mix and match:

- :class:`Chunker` splits a report into :class:`Chunk` slices.
- :class:`Router` labels chunks with the extraction groups they feed.

Self-labeling chunkers (e.g. :class:`RegexSectionChunker`) implement both
by writing ``labels`` directly during chunking; the GUI/pipeline can then
skip the router step.

See :file:`_chunking_notes.md` in this directory for usage guidance.
"""
from __future__ import annotations

from .paragraph import ParagraphChunker, paragraph_spans
from .protocols import Chunk, Chunker, Router
from .regex_chunker import DEFAULT_PATHOLOGY_PATTERNS, RegexSectionChunker
from .registry import (
    CHUNKER_REGISTRY,
    ROUTER_REGISTRY,
    get_chunker,
    get_router,
    register_chunker,
    register_router,
)

# Optional implementations: import for side-effect (registry registration) but
# never fail the package import if their soft deps (sentence-transformers,
# numpy) are unavailable.
try:  # pragma: no cover - import-time soft dependency
    from .embedding_chunker import (  # noqa: F401
        EmbeddingBoundaryChunker,
        EmbeddingSimilarityRouter,
    )
except ImportError:
    pass

try:  # pragma: no cover - import-time soft dependency
    from .dspy_embedder_router import DSPyEmbedderRouter  # noqa: F401
except ImportError:
    pass


__all__ = [
    "Chunk",
    "Chunker",
    "Router",
    "ParagraphChunker",
    "paragraph_spans",
    "RegexSectionChunker",
    "DEFAULT_PATHOLOGY_PATTERNS",
    "CHUNKER_REGISTRY",
    "ROUTER_REGISTRY",
    "register_chunker",
    "register_router",
    "get_chunker",
    "get_router",
]
