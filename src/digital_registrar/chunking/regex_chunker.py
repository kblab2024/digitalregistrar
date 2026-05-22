"""Section-header regex chunker.

For pathology reports a deterministic rule-based splitter on common
section headers (``MARGINS:``, ``LYMPH NODES:``, ``MICROSCOPIC
DESCRIPTION``, ...) gives most of the routing signal at zero ML cost.
Each chunk is auto-labeled with the section name it falls under, which
the pipeline / GUI can use directly when the section names already match
group names (e.g. ``margins``, ``lymph_nodes``).

For sections that don't map 1:1 to extraction groups, pass
``section_to_groups`` to translate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .protocols import Chunk
from .registry import register_chunker

# Default header patterns. Keys are deliberately chosen to match the
# breast/lung/etc. GROUP_INSTRUCTIONS keys where natural — so routing
# without an explicit ``section_to_groups`` mapping "just works" for
# margins / lymph_nodes / staging / biomarkers.
# Headers are anchored: each pattern must end with optional whitespace
# followed by either a colon (``MARGINS:``) or end-of-line (``MARGINS``
# alone on its line). Without that anchor, ``SPECIMEN`` would match the
# sentence ``Specimen received in formalin, ...`` mid-narrative.
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


@dataclass
class _HeaderHit:
    label: str
    header_start: int
    body_start: int


@register_chunker("regex_section")
class RegexSectionChunker:
    """Split a report on labeled section-header regexes.

    Parameters
    ----------
    pattern_map
        ``{section_label: regex}``. Patterns are compiled with
        ``re.IGNORECASE | re.MULTILINE`` so ``^`` and ``$`` line-anchor
        each header. If a header matches, the chunk is the text between
        the end of that header line and the start of the next header
        (or end of report). Defaults to :data:`DEFAULT_PATHOLOGY_PATTERNS`.
    section_to_groups
        Optional ``{section_label: [group_name, ...]}`` mapping. When set,
        a chunk's ``labels`` become the mapped groups instead of the raw
        section label. Use this when your section headers don't match the
        organ's group names (e.g. map ``"diagnosis" -> ["nonnested", "grading"]``).
    include_preamble
        If True (default), text before the first header becomes its own
        unlabeled chunk. The routing fallback will then send the
        full report to any group not otherwise covered.
    """

    def __init__(
        self,
        pattern_map: dict[str, str] | None = None,
        *,
        section_to_groups: dict[str, list[str]] | None = None,
        include_preamble: bool = True,
    ) -> None:
        self.pattern_map = dict(pattern_map or DEFAULT_PATHOLOGY_PATTERNS)
        self.section_to_groups = dict(section_to_groups or {})
        self.include_preamble = include_preamble
        self._compiled = {
            label: re.compile(p, re.IGNORECASE | re.MULTILINE)
            for label, p in self.pattern_map.items()
        }

    def chunk(self, report: str) -> list[Chunk]:
        hits: list[_HeaderHit] = []
        for label, regex in self._compiled.items():
            for m in regex.finditer(report):
                hits.append(_HeaderHit(
                    label=label, header_start=m.start(), body_start=m.end(),
                ))
        hits.sort(key=lambda h: h.header_start)

        # If two headers match at (or near) the same start, keep the longer
        # one — avoids "MARGINS" capturing both itself and a more specific
        # "MARGINS AND LYMPH NODES" header.
        hits = _dedupe_overlapping(hits)

        chunks: list[Chunk] = []
        cid = 0

        if self.include_preamble:
            preamble_end = hits[0].header_start if hits else len(report)
            preamble = report[:preamble_end]
            if preamble.strip():
                chunks.append(Chunk(
                    id=f"c{cid}",
                    text=preamble,
                    span=(0, preamble_end),
                    labels=frozenset(),
                    meta={"section": "<preamble>"},
                ))
                cid += 1

        if not hits:
            return chunks

        for i, h in enumerate(hits):
            content_start = h.body_start
            content_end = hits[i + 1].header_start if i + 1 < len(hits) else len(report)
            if content_end <= content_start:
                continue
            text = report[content_start:content_end]
            if not text.strip():
                continue
            labels_iter = self.section_to_groups.get(h.label, [h.label])
            chunks.append(Chunk(
                id=f"c{cid}",
                text=text,
                span=(content_start, content_end),
                labels=frozenset(labels_iter),
                meta={"section": h.label},
            ))
            cid += 1

        return chunks


def _dedupe_overlapping(hits: list[_HeaderHit]) -> list[_HeaderHit]:
    """Drop hits whose header line is fully consumed by an earlier longer hit."""
    if len(hits) <= 1:
        return hits
    kept: list[_HeaderHit] = []
    for h in hits:
        if kept and h.header_start < kept[-1].body_start:
            # h's header is inside the previous header line; pick the longer.
            prev = kept[-1]
            if (h.body_start - h.header_start) > (prev.body_start - prev.header_start):
                kept[-1] = h
            continue
        kept.append(h)
    return kept


__all__ = ["RegexSectionChunker", "DEFAULT_PATHOLOGY_PATTERNS"]
