"""JSON loading + parse-error / runtime-error detection for prediction files.

Three load paths:

- :func:`load_json` returns the parsed dict or raises ``ParseError`` if
  the file is missing or malformed. Used for gold annotations where any
  failure is a hard error.
- :func:`load_prediction` is *defensive*: it returns a structured
  outcome (``ok`` + dict, or ``parse_error`` + error_mode + raw text) so
  the eval pipeline can record the failure mode rather than crashing.
- :func:`load_log_entry` reads optional ``_log.jsonl`` next to a run
  directory to enrich parse-error decomposition with timeout / refusal /
  schema-invalid signals.

Error-mode taxonomy:
    json_parse       — raised :class:`json.JSONDecodeError`
    schema_invalid   — file parsed but failed pydantic validation
    timeout          — log entry indicates the model didn't return
    refusal          — log entry indicates the model refused
    file_missing     — the file isn't on disk
    other            — unrecognised failure
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

ErrorMode = Literal[
    "json_parse",
    "schema_invalid",
    "timeout",
    "refusal",
    "file_missing",
    "other",
]


class ParseError(Exception):
    """Hard failure loading an annotation/prediction file."""


@dataclass(frozen=True)
class LoadOutcome:
    """Result of a defensive load attempt."""

    ok: bool
    data: dict[str, Any] | None
    error_mode: ErrorMode | None
    error_message: str | None
    path: Path

    def __bool__(self) -> bool:
        return self.ok


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return the parsed dict.

    Raises :class:`ParseError` on missing or malformed files. Use this
    for *gold* loads where missingness is a fatal error in the dataset.
    """
    if not path.exists():
        raise ParseError(f"file missing: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ParseError(f"json parse failed for {path}: {e}") from e
    if not isinstance(data, dict):
        raise ParseError(f"expected JSON object at {path}, got {type(data).__name__}")
    return data


def load_prediction(path: Path) -> LoadOutcome:
    """Defensively load a prediction file.

    Returns an :class:`LoadOutcome` with ``ok=False`` and an
    :data:`ErrorMode` rather than raising on parse failure. Honours an
    explicit ``_parse_error`` marker inside the JSON (some upstream
    runners write a sentinel object on failure).
    """
    if not path.exists():
        return LoadOutcome(
            ok=False, data=None, error_mode="file_missing",
            error_message=f"file missing: {path}", path=path,
        )
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return LoadOutcome(
            ok=False, data=None, error_mode="json_parse",
            error_message=str(e), path=path,
        )
    except OSError as e:
        return LoadOutcome(
            ok=False, data=None, error_mode="other",
            error_message=str(e), path=path,
        )
    if not isinstance(data, dict):
        return LoadOutcome(
            ok=False, data=None, error_mode="schema_invalid",
            error_message=f"expected dict, got {type(data).__name__}",
            path=path,
        )
    if data.get("_parse_error"):
        # Upstream runner explicitly flagged this case as failed.
        return LoadOutcome(
            ok=False, data=data, error_mode="schema_invalid",
            error_message=str(data.get("_parse_error_message") or "parse_error sentinel"),
            path=path,
        )
    return LoadOutcome(
        ok=True, data=data, error_mode=None,
        error_message=None, path=path,
    )


def extract_meta(annotation: dict[str, Any]) -> dict[str, Any]:
    """Pull the ``_meta`` block from a human annotation, or return ``{}``.

    Human annotations carry ``_meta = {annotator, mode, annotated_at}``
    for IAA/preann analysis. Gold and model predictions don't have it.
    """
    meta = annotation.get("_meta")
    return dict(meta) if isinstance(meta, dict) else {}


def index_runs(run_logs: dict[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """Read ``_log.jsonl`` files and return ``{run_id: {case_id: log_entry}}``.

    Optional — missing or malformed log files are silently skipped. The
    returned mapping is consulted by the outcome classifier to enrich
    parse-error decomposition with ``timeout``/``refusal`` signals.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for run_id, run_dir in run_logs.items():
        log_path = Path(run_dir) / "_log.jsonl"
        if not log_path.is_file():
            out[run_id] = {}
            continue
        entries: dict[str, dict[str, Any]] = {}
        try:
            with log_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cid = entry.get("case_id") or entry.get("case")
                    if cid:
                        entries[cid] = entry
        except OSError:
            pass
        out[run_id] = entries
    return out


def classify_log_error(log_entry: dict[str, Any] | None) -> ErrorMode | None:
    """Map a log entry to a refined error mode.

    Returns ``None`` if the entry doesn't indicate a failure — caller
    falls back to ``json_parse`` / ``schema_invalid`` from the file
    inspection. Heuristic field names match the conventions in
    ``scripts/run_*.py`` runners.
    """
    if not log_entry:
        return None
    if log_entry.get("timeout"):
        return "timeout"
    err = log_entry.get("error") or log_entry.get("error_type")
    if isinstance(err, str):
        err_l = err.lower()
        if "timeout" in err_l:
            return "timeout"
        if "refus" in err_l or "policy" in err_l:
            return "refusal"
        if "schema" in err_l or "validation" in err_l:
            return "schema_invalid"
        if "json" in err_l or "parse" in err_l:
            return "json_parse"
        return "other"
    return None


def cascade_scalar_only(df: pd.DataFrame) -> pd.DataFrame:
    """Subset a ``cascade_atomic`` frame to Stage-C scalar rows.

    Cascade emits one row per ``(run, case, cascade_stage, field)``. Stage-C
    rows for nested fields (``margins``, ``regional_lymph_node``,
    ``biomarkers``) carry a float F1 in the ``correct`` column instead of a
    bool — averaging or summing them alongside binary-correct scalar rows
    silently corrupts means. Every consumer that reduces ``correct`` over
    Stage-C should apply this helper first.

    Returns a view (not a copy) restricted to ``cascade_stage == "C"`` AND
    (when the column exists) ``field_kind != "nested_list"``. Tolerant of
    older atomic frames missing ``field_kind`` — those frames are returned
    as-is after the stage filter.
    """
    sub = df[df["cascade_stage"] == "C"]
    if "field_kind" in sub.columns:
        sub = sub[sub["field_kind"] != "nested_list"]
    return sub


def coerce_cascade_correct(s: pd.Series) -> pd.Series:
    """Map a possibly-JSON-stringified ``correct`` column back to floats.

    When :func:`scripts.eval._common.reporting.write_parquet` serialises a
    cascade_atomic frame whose ``correct`` column mixes scalar booleans
    (Stage A/B/C scalar) with nested floats (Stage C nested-list F1),
    the helper coerces the entire column to JSON strings ("true",
    "false", "0.5") so pyarrow can serialise it. Reading consumers that
    reduce on numeric correctness must coerce back, otherwise
    ``pd.to_numeric(..., errors="coerce")`` returns NaN for the
    stringified booleans and silently zeros every group reduction.

    Returns a Series of floats:
      - bool ``True`` / "true" / 1 → 1.0
      - bool ``False`` / "false" / 0 → 0.0
      - native floats / numeric strings → float(value)
      - None / "" / unrecognised → NaN
    """
    def _one(v):
        if v is None:
            return float("nan")
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            lv = v.strip().lower()
            if lv == "true":
                return 1.0
            if lv == "false":
                return 0.0
            if lv in ("", "none", "null"):
                return float("nan")
            try:
                return float(lv)
            except ValueError:
                return float("nan")
        return float("nan")
    return s.apply(_one)


def coerce_cascade_bool(s: pd.Series) -> pd.Series:
    """Map a possibly-JSON-stringified bool column back to native bools.

    Companion to :func:`coerce_cascade_correct` for the
    ``attempted`` / ``gold_present`` / ``field_missing`` / ``parse_error``
    columns, which can also be stringified when their dtype crosses
    pyarrow's mixed-type threshold (rare, but harmless to coerce
    defensively).
    """
    def _one(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v) if not pd.isna(v) else False
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return False
    return s.apply(_one).astype(bool)


__all__ = [
    "ParseError",
    "LoadOutcome",
    "ErrorMode",
    "load_json",
    "load_prediction",
    "extract_meta",
    "index_runs",
    "classify_log_error",
    "cascade_scalar_only",
    "coerce_cascade_correct",
    "coerce_cascade_bool",
]
