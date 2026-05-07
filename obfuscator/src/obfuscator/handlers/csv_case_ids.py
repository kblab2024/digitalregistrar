"""Replace TCGA-XX-XXXX strings in CSV files with deterministic synthetic IDs."""
from __future__ import annotations

import re
from pathlib import Path

from ..id_mapper import synth_tcga_case_id
from ..manifest import Manifests

_TCGA_RE = re.compile(r"\bTCGA-[A-Z0-9]{2}-[A-Z0-9]{4}\b")


def handle(src: Path, dst: Path, manifests: Manifests, master_seed: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    new_text = _TCGA_RE.sub(
        lambda m: synth_tcga_case_id(master_seed, m.group(0)),
        text,
    )
    dst.write_text(new_text, encoding="utf-8")
    manifests.record_output(src, dst, handler="csv_case_ids",
                            extra={"replacements": len(_TCGA_RE.findall(text))})
