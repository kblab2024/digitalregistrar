"""Remap case IDs inside data/{dataset}/splits.json.

Walks any list/dict shape, replacing string values that match a known case_id pattern.
Falls through (skip + record) if shape is unrecognized.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..manifest import Manifests, sha256_of_file


_CASE_ID_RE = re.compile(r"^[a-z]+\d+_\d+$")  # e.g. tcga1_5, cmuh3_42
_TCGA_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:\.[A-F0-9-]+)?$")


def handle(src: Path, dst: Path, manifests: Manifests, master_seed: int,
           id_map: dict[str, str]) -> None:
    """`id_map` maps original case_ids to synthetic equivalents (caller-provided)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with src.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        manifests.record_skipped(src, "splits.json parse failed",
                                 sha256=sha256_of_file(src))
        return
    remapped, n = _walk_remap(data, id_map)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(remapped, f, indent=2)
    manifests.record_output(src, dst, handler="splits_remapper",
                            extra={"ids_remapped": n})


def _walk_remap(obj: Any, id_map: dict[str, str]) -> tuple[Any, int]:
    n = 0
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            new_v, n_v = _walk_remap(v, id_map)
            n += n_v
            out[k] = new_v
        return out, n
    if isinstance(obj, list):
        out_list = []
        for item in obj:
            new_item, n_i = _walk_remap(item, id_map)
            n += n_i
            out_list.append(new_item)
        return out_list, n
    if isinstance(obj, str):
        if _CASE_ID_RE.match(obj):
            return id_map.get(obj, obj), 1 if obj in id_map else 0
        if _TCGA_RE.match(obj):
            return id_map.get(obj, obj), 1 if obj in id_map else 0
    return obj, 0
