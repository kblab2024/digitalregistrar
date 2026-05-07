"""Folder discovery, sample indexing, and annotation load/save (flat on-disk format)."""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Sentinel used in session state to represent "intentional N/A".
# Serialised values are always plain null — this string only lives in
# Streamlit session state and is converted to None by _clean_value.
# Paths whose session value equals this sentinel are recorded in
# _meta.intentional_nulls so the intent survives save → reload.
NA_SENTINEL = "__NA__"


@dataclass
class SampleRef:
    sample_id: str          # e.g. "tcga1_37"
    stem: str               # e.g. "tcga1"
    n: str                  # e.g. "1" (numeric suffix of stem)
    result_path: str        # ...{prefix}_result_{date}/{n}/{sample_id}_output.json
    dataset_path: str       # ...{prefix}_dataset_{date}/{stem}/{sample_id}.txt
    annotation_path: str    # ...{prefix}_annotation_{date}/{n}/{sample_id}_annotation_{suffix}.json
    annotator_suffix: str   # e.g. "nhc"


@dataclass
class FolderSet:
    base_dir: str
    prefix: str              # e.g. "tcga"
    date: str                # e.g. "20251117"
    dataset_dir: str
    result_dir: str
    annotation_dir: str


# ── Folder discovery ───────────────────────────────────────────────────────────

_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z]+)_(?P<kind>dataset|result|annotation)_(?P<date>\d{6,})$")


def discover_folders(base_dir: str) -> FolderSet | None:
    """Scan base_dir for `{prefix}_{dataset|result|annotation}_{date}` subfolders.

    Returns a FolderSet when at least dataset + result are found. Annotation dir is
    inferred (and will be auto-created on first save) if missing.
    """
    if not base_dir or not os.path.isdir(base_dir):
        return None

    found: dict[tuple[str, str], dict[str, str]] = {}
    for entry in os.listdir(base_dir):
        full = os.path.join(base_dir, entry)
        if not os.path.isdir(full):
            continue
        m = _PATTERN.match(entry)
        if not m:
            continue
        key = (m.group("prefix"), m.group("date"))
        found.setdefault(key, {})[m.group("kind")] = full

    # Pick the group with the most complete set, preferring newest date.
    candidates = sorted(
        found.items(),
        key=lambda kv: (-len(kv[1]), kv[0][1]),  # more kinds first, then newest date
        reverse=True,
    )
    for (prefix, date), kinds in candidates:
        if "dataset" in kinds and "result" in kinds:
            annotation_dir = kinds.get(
                "annotation",
                os.path.join(base_dir, f"{prefix}_annotation_{date}"),
            )
            return FolderSet(
                base_dir=base_dir,
                prefix=prefix,
                date=date,
                dataset_dir=kinds["dataset"],
                result_dir=kinds["result"],
                annotation_dir=annotation_dir,
            )
    return None


# ── Sample indexing ────────────────────────────────────────────────────────────

_FILENAME_RE = re.compile(r"^(?P<stem>[A-Za-z]+\d+)_(?P<idx>\d+)_output\.json$")


def list_samples(folders: FolderSet, annotator_suffix: str) -> list[SampleRef]:
    """Walk result_dir/*/{stem}_{idx}_output.json and build SampleRef list.

    annotation_path is built with the given annotator suffix so each annotator
    has a fully independent set of files.
    """
    samples: list[SampleRef] = []
    pattern = os.path.join(folders.result_dir, "*", "*_output.json")
    for path in glob.glob(pattern):
        filename = os.path.basename(path)
        m = _FILENAME_RE.match(filename)
        if not m:
            continue
        stem = m.group("stem")
        idx = m.group("idx")
        sample_id = f"{stem}_{idx}"
        n_folder = os.path.basename(os.path.dirname(path))

        # n is conventionally the digits in the stem; fall back to n_folder.
        digits = "".join(ch for ch in stem if ch.isdigit())
        n = digits or n_folder

        dataset_path = os.path.join(folders.dataset_dir, stem, f"{sample_id}.txt")
        annotation_path = os.path.join(
            folders.annotation_dir, n, f"{sample_id}_annotation_{annotator_suffix}.json"
        )
        samples.append(SampleRef(
            sample_id=sample_id,
            stem=stem,
            n=n,
            result_path=path,
            dataset_path=dataset_path,
            annotation_path=annotation_path,
            annotator_suffix=annotator_suffix,
        ))

    samples.sort(key=_sample_sort_key)
    return samples


def _sample_sort_key(s: SampleRef) -> tuple:
    m = re.match(r"([A-Za-z]+)(\d+)_(\d+)$", s.sample_id)
    if m:
        return (m.group(1), int(m.group(2)), int(m.group(3)))
    return (s.sample_id, 0, 0)


# ── JSON I/O ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_report_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return "(Could not read dataset file: " + path + ")"


def strip_meta(data: dict) -> dict:
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def build_save_payload(annotation: dict, filename: str) -> dict:
    """Assemble final save dict from session annotation state (flat format).

    Session state mirrors the on-disk layout:
      - top-level: cancer_excision_report, cancer_category, cancer_category_others_description
      - cancer_data: {all cancer-specific fields; includes arrays margins/biomarkers/etc.}

    Every schema-declared field for the chosen cancer is written, defaulting
    to null when the annotator left it at "— 尚未設定 —". Fields the
    annotator explicitly marked as N/A also serialise to null, but their
    paths are additionally recorded in _meta.intentional_nulls so the
    "intentional" state can be restored on reload.
    """
    output: dict = {
        "_meta": {
            "filename": filename,
            "annotated_at": datetime.now().isoformat(timespec="seconds"),
        }
    }

    # Collect sentinel paths from the raw session annotation BEFORE _clean_value
    # collapses them to None.
    na_paths = _collect_sentinel_paths(annotation)

    excision = annotation.get("cancer_excision_report")
    category = annotation.get("cancer_category")
    output["cancer_excision_report"] = _clean_value(excision)
    output["cancer_category"] = _clean_value(category)
    output["cancer_category_others_description"] = _clean_value(
        annotation.get("cancer_category_others_description")
    )

    if excision is False:
        if na_paths:
            # Preserve only intent for fields still present in the truncated payload.
            kept = [p for p in na_paths if p in {
                "cancer_excision_report",
                "cancer_category",
                "cancer_category_others_description",
            }]
            if kept:
                output["_meta"]["intentional_nulls"] = kept
        return output

    cancer_data = annotation.get("cancer_data") or {}
    if category and category != "others":
        from .parser import parse_cancer_schema
        sections = parse_cancer_schema(category)
        expanded: dict = {}
        for sec in sections:
            for fspec in sec.flat_fields:
                expanded[fspec.name] = cancer_data.get(fspec.name)
            if sec.array_field_name:
                expanded[sec.array_field_name] = cancer_data.get(sec.array_field_name)
        for k, v in cancer_data.items():
            if k not in expanded:
                expanded[k] = v
        output["cancer_data"] = {k: _clean_value(v) for k, v in expanded.items()}
    elif cancer_data:
        output["cancer_data"] = _clean_value(cancer_data)

    if na_paths:
        output["_meta"]["intentional_nulls"] = na_paths

    return output


def _clean_value(val):
    if isinstance(val, dict):
        return {k: _clean_value(v) for k, v in val.items()}
    if isinstance(val, list):
        if not val:
            return None
        return [_clean_value(v) for v in val]
    if isinstance(val, str):
        if val == "" or val == NA_SENTINEL:
            return None
    return val


# ── Intentional-null intent tracking ───────────────────────────────────────────

_PATH_SEGMENT_RE = re.compile(r"^([^\[]+)(?:\[(\d+)\])?$")


def _collect_sentinel_paths(obj, prefix: str = "") -> list[str]:
    """Walk a session annotation dict and return dotted paths whose value is NA_SENTINEL.

    Array indices are written as ``field[i]``. The _meta key and any other
    underscore-prefixed keys are skipped so callers don't accidentally record
    paths into the meta subtree.
    """
    paths: list[str] = []
    if obj == NA_SENTINEL:
        if prefix:
            paths.append(prefix)
        return paths
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            paths.extend(_collect_sentinel_paths(v, next_prefix))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            paths.extend(_collect_sentinel_paths(v, f"{prefix}[{i}]"))
    return paths


def _parse_intent_path(path: str) -> list:
    """Break ``a.b[0].c`` into a flat segment list: ['a', 'b', 0, 'c']."""
    segments: list = []
    for part in path.split("."):
        m = _PATH_SEGMENT_RE.match(part)
        if not m:
            return []
        segments.append(m.group(1))
        if m.group(2) is not None:
            segments.append(int(m.group(2)))
    return segments


def rehydrate_sentinels(annotation: dict, meta: dict | None) -> None:
    """Replace ``None`` with NA_SENTINEL at every path listed in ``_meta.intentional_nulls``.

    Mutates ``annotation`` in place. Paths that no longer resolve (e.g. an
    array entry that has since been removed) are silently skipped. Only
    current ``None`` values are replaced — real data is never overwritten.
    """
    if not annotation or not isinstance(annotation, dict):
        return
    paths = (meta or {}).get("intentional_nulls") or []
    for path in paths:
        segments = _parse_intent_path(path)
        if not segments:
            continue
        try:
            cursor = annotation
            for seg in segments[:-1]:
                cursor = cursor[seg]
            last = segments[-1]
            current = cursor[last]
            if current is None:
                cursor[last] = NA_SENTINEL
        except (KeyError, IndexError, TypeError):
            continue


def save_annotation(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
