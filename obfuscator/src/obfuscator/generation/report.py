"""Layer 2a: render report .txt from report_realization.

CMUH = clean key:value layout.
TCGA = same fields shuffled and interleaved with filler clinical sentences.
"""
from __future__ import annotations

import random
from typing import Any

from ..id_mapper import synth_patient_filename


_CMUH_HEADER = """SPECIMEN: {procedure_label} {laterality_label}
PATIENT FILENAME: {patient_filename}
"""

_FILLER_SENTENCES = [
    "Specimen was received fresh and fixed in formalin.",
    "Sections were submitted in their entirety in cassettes for histologic examination.",
    "The specimen is oriented with sutures by the surgeon.",
    "Cut surfaces show fibrofatty tissue with focal areas of induration.",
    "Representative sections are submitted as labeled.",
    "No gross perforations identified in the area of adhesions.",
    "Sectioning reveals tan, well-circumscribed tissue.",
    "Microscopic examination confirms the gross findings.",
    "Inked margins are noted on all aspects of the specimen.",
    "Multiple representative sections submitted in sequential order.",
]


def render_report(realization: dict[str, Any], cancer_category: str | None,
                  cancer_excision_report: bool,
                  master_seed: int, case_id: str,
                  dataset: str, organ_n: str) -> str:
    """Build a synthetic report .txt for the given case.

    For non-cancer reports, returns a stub. For cancer reports, mirrors the
    realization fields so a downstream LLM extractor can plausibly recover
    the gold annotation.
    """
    patient_filename = synth_patient_filename(master_seed, case_id)
    rng = random.Random(hash((master_seed, case_id, "report")) & 0xFFFFFFFF)

    lines: list[str] = []
    lines.append(f"patient_filename: {patient_filename}")

    if not cancer_excision_report:
        body = "Specimen submitted for histologic examination. No malignancy identified. Benign findings only."
        lines.append(f"text: {body}")
        return "\n".join(lines) + "\n"

    flat = _flatten(realization)
    if dataset == "cmuh":
        body = _render_clean(flat, cancer_category, organ_n)
    else:
        body = _render_chaotic(flat, cancer_category, organ_n, rng)
    lines.append(f"text: {body}")
    return "\n".join(lines) + "\n"


def _render_clean(flat: dict[str, Any], cancer_category: str | None, organ_n: str) -> str:
    parts: list[str] = []
    parts.append(f"DIAGNOSIS: {cancer_category} carcinoma (organ_n={organ_n})")
    for k, v in flat.items():
        if v is None:
            continue
        parts.append(f"{k}: {_render_value(v)}")
    return ". ".join(parts) + "."


def _render_chaotic(flat: dict[str, Any], cancer_category: str | None,
                    organ_n: str, rng: random.Random) -> str:
    blocks: list[str] = []
    blocks.append(f"FINAL DIAGNOSIS: {cancer_category} carcinoma. organ_n={organ_n}.")
    items = [(k, v) for k, v in flat.items() if v is not None]
    rng.shuffle(items)
    for i, (k, v) in enumerate(items):
        blocks.append(f"{k}: {_render_value(v)}.")
        if i % 3 == 2 and _FILLER_SENTENCES:
            blocks.append(rng.choice(_FILLER_SENTENCES))
    return " ".join(blocks)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            sub = _flatten(v, prefix=f"{key}.")
            out.update(sub)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    sub = _flatten(item, prefix=f"{key}[{i}].")
                    out.update(sub)
                else:
                    out[f"{key}[{i}]"] = item
        else:
            out[key] = v
    return out


def _render_value(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)
