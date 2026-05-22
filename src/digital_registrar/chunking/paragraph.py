"""Baseline paragraph chunker.

Matches the pipeline's existing paragraph-normalization behaviour
(``pipeline_factory._normalize_report``): split on blank lines, drop
empty runs, keep char offsets so the GUI can highlight the exact slice.

Useful as a no-op chunker (every paragraph is its own chunk) and as the
text-splitter feeding the embedding similarity router when the user
doesn't want semantic boundary detection.
"""
from __future__ import annotations

import re

from .protocols import Chunk
from .registry import register_chunker

_PARA_SEP = re.compile(r"\n\s*\n")


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Char-offset spans for each non-empty paragraph in ``text``.

    A paragraph is a non-blank run delimited by one or more blank lines.
    Trailing/leading whitespace inside a paragraph is preserved in the
    span; the surrounding blank-line separators are not.
    """
    spans: list[tuple[int, int]] = []
    n = len(text)
    cursor = 0
    while cursor < n:
        # Skip blank-line separators (and leading whitespace of the doc).
        m = re.match(r"\s+", text[cursor:])
        if m and ("\n" in m.group() or cursor == 0):
            cursor += m.end()
        if cursor >= n:
            break
        # Find next blank-line separator.
        sep = _PARA_SEP.search(text, cursor)
        end = sep.start() if sep is not None else n
        if end > cursor:
            spans.append((cursor, end))
        cursor = end
    return spans


@register_chunker("paragraph")
class ParagraphChunker:
    """Splits on blank lines; one chunk per paragraph."""

    def chunk(self, report: str) -> list[Chunk]:
        spans = paragraph_spans(report)
        return [
            Chunk(id=f"c{i}", text=report[s:e], span=(s, e))
            for i, (s, e) in enumerate(spans)
        ]


__all__ = ["ParagraphChunker", "paragraph_spans"]
