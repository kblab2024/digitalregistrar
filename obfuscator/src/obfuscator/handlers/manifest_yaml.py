"""Allowlist + remap dataset_manifest.yaml.

Keeps known top-level keys (organs, counts, version, dataset). Remaps any case-ID-shaped
strings via id_map. If shape is wholly unrecognized, fall through to skip.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..manifest import Manifests, sha256_of_file


_CASE_ID_RE = re.compile(r"^[a-z]+\d+_\d+$")
_ALLOWLIST_KEYS = {
    "name", "dataset", "version", "created_at", "n_cases",
    "organs", "n_per_organ", "splits", "schema_version",
    "preannotation_model", "annotators", "modes",
}


def handle(src: Path, dst: Path, manifests: Manifests, id_map: dict[str, str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with src.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        manifests.record_skipped(src, "manifest yaml parse failed",
                                 sha256=sha256_of_file(src))
        return
    if not isinstance(data, dict):
        manifests.record_skipped(src, "manifest yaml not a top-level dict",
                                 sha256=sha256_of_file(src))
        return
    filtered = {k: v for k, v in data.items() if k in _ALLOWLIST_KEYS}
    remapped, n = _walk_remap(filtered, id_map)
    with dst.open("w", encoding="utf-8") as f:
        yaml.safe_dump(remapped, f, sort_keys=False)
    manifests.record_output(src, dst, handler="manifest_remapper",
                            extra={"keys_kept": list(filtered.keys()),
                                   "ids_remapped": n})


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
        out_l = []
        for item in obj:
            new_i, n_i = _walk_remap(item, id_map)
            n += n_i
            out_l.append(new_i)
        return out_l, n
    if isinstance(obj, str) and _CASE_ID_RE.match(obj):
        return id_map.get(obj, obj), 1 if obj in id_map else 0
    return obj, 0
