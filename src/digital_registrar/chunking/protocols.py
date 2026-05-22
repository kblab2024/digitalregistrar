"""Core types for the chunking layer.

A ``Chunk`` is a contiguous slice of an original report text, identified
by a stable id and char-offset span. ``labels`` carries the set of
extraction-group tags the chunk has been routed to (e.g. ``{"margins"}``).

The two protocols separate concerns:

- ``Chunker`` turns a report into chunks (no routing knowledge).
- ``Router`` consumes chunks plus per-group descriptions and returns the
  same chunks with ``labels`` populated.

Some chunkers naturally produce labeled chunks (regex section splitters):
they implement both responsibilities and can be paired with a no-op
router, or used standalone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Chunk:
    """A labeled slice of a report.

    ``span`` is a half-open char interval into the *original* report text
    (the same string passed to ``Chunker.chunk``). Keeping char offsets —
    not paragraph indices — lets the GUI highlight the exact substring.
    """

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


@runtime_checkable
class Chunker(Protocol):
    """Splits a report into a list of ``Chunk``.

    Implementations should set a class-level ``name`` attribute (the key
    used in ``CHUNKER_REGISTRY``). Implementations should be deterministic
    for a given input + configuration.
    """

    name: str

    def chunk(self, report: str) -> list[Chunk]: ...


@runtime_checkable
class Router(Protocol):
    """Labels chunks with the extraction groups they should feed.

    ``groups`` maps group name (e.g. ``"margins"``) to a description /
    instruction string the router can use as a query. The router returns
    the *same* chunks with ``labels`` populated — implementations should
    not mutate the input list.
    """

    name: str

    def route(self, chunks: list[Chunk], groups: dict[str, str]) -> list[Chunk]: ...


__all__ = ["Chunk", "Chunker", "Router"]
