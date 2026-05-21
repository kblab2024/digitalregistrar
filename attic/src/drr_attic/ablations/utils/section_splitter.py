"""Heuristic splitter for pathology reports into named sections.

Used by the per-section ablation (A5) so each section can be fed to a
section-specialised signature.

Supported sections (keys returned by :func:`split_report`):

    header     — clinical history, specimen list, gross specimen IDs
    gross      — gross description / macroscopic findings
    micro      — microscopic description
    dx         — final diagnosis / synoptic checklist
    comments   — comments / addendum / IHC discussion

Heuristic: locate ALL CAPS or Title-Case section headers ending with a
colon or newline, then assign every line until the next header to that
section. Reports without recognisable headers collapse into ``dx``
(the most defensive default — that's where the bulk of structured
findings live).

The splitter is intentionally simple — its job is to give the per-section
predictor a cleaner context window, not to be linguistically perfect.
"""
from __future__ import annotations

import re

# Order matters — first match wins when a header could plausibly belong
# to multiple sections (e.g. "FINAL DIAGNOSIS / COMMENT" → dx).
_HEADER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("dx", re.compile(
        r"(?im)^\s*(?:[A-Z]\.\s*)?"
        r"(?:FINAL\s+(?:PATH(?:OLOGIC)?\s+)?DIAGNOSIS|"
        r"DIAGNOSIS|"
        r"PATHOLOGIC\s+DIAGNOSIS|"
        r"SYNOPTIC\s+(?:REPORT|SUMMARY|CHECKLIST)|"
        r"CAP\s+SUMMARY)\s*[:.\-]?\s*$",
    )),
    ("gross", re.compile(
        r"(?im)^\s*(?:[A-Z]\.\s*)?"
        r"(?:GROSS\s+DESCRIPTION|"
        r"MACROSCOPIC\s+(?:DESCRIPTION|EXAMINATION)|"
        r"GROSS\s+EXAMINATION|"
        r"SPECIMEN\s+DESCRIPTION)\s*[:.\-]?\s*$",
    )),
    ("micro", re.compile(
        r"(?im)^\s*(?:[A-Z]\.\s*)?"
        r"(?:MICROSCOPIC\s+(?:DESCRIPTION|EXAMINATION)|"
        r"HISTOLOGY)\s*[:.\-]?\s*$",
    )),
    ("comments", re.compile(
        r"(?im)^\s*(?:[A-Z]\.\s*)?"
        r"(?:COMMENTS?|"
        r"ADDENDUM|"
        r"NOTE\s+TO\s+CLINICIAN|"
        r"IMMUNOHISTO(?:CHEMISTRY|CHEMICAL\s+STUDIES)|"
        r"ANCILLARY\s+(?:STUDIES|TESTING)|"
        r"DISCUSSION)\s*[:.\-]?\s*$",
    )),
    ("header", re.compile(
        r"(?im)^\s*(?:[A-Z]\.\s*)?"
        r"(?:CLINICAL\s+(?:HISTORY|INFORMATION|DATA)|"
        r"SPECIMEN(?:S)?(?:\s+SUBMITTED)?|"
        r"PRE-?OPERATIVE\s+DIAGNOSIS|"
        r"PROCEDURE)\s*[:.\-]?\s*$",
    )),
]

SECTION_NAMES: tuple[str, ...] = ("header", "gross", "micro", "dx", "comments")


def split_report(report: str) -> dict[str, str]:
    """Split a pathology report into named sections.

    Returns a dict with keys :data:`SECTION_NAMES`. Sections without
    matching content come back as the empty string. Reports with no
    recognisable headers collapse entirely into ``dx``.
    """
    sections: dict[str, list[str]] = {name: [] for name in SECTION_NAMES}

    lines = report.splitlines()
    current = "dx"  # default bucket before the first header
    found_any_header = False

    for line in lines:
        matched_section: str | None = None
        for section_name, pattern in _HEADER_PATTERNS:
            if pattern.match(line.strip()):
                matched_section = section_name
                found_any_header = True
                break

        if matched_section is not None:
            current = matched_section
            # Don't include the header line itself.
            continue

        sections[current].append(line)

    if not found_any_header:
        # No sectioning information available — whole report is "dx".
        return {name: ("\n".join(lines) if name == "dx" else "")
                for name in SECTION_NAMES}

    return {name: "\n".join(content).strip()
            for name, content in sections.items()}


__all__ = ["split_report", "SECTION_NAMES"]
