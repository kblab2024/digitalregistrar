"""Manifest stamping and CSV writing helpers.

The manifest captures every input that influences the output so a
reader can reproduce a run from a single file: CLI args, git SHA,
package version, dataset summary, UTC timestamp.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _git_sha() -> str | None:
    """Best-effort git SHA. Returns ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _package_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("digital_registrar_research")
    except Exception:
        return None


def write_manifest(
    out_dir: Path,
    args: argparse.Namespace | dict[str, Any],
    *,
    subcommand: str,
    n_cases_per_organ: dict[int, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``<out_dir>/manifest.json`` capturing everything needed to
    reproduce the outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(args, argparse.Namespace):
        raw = vars(args)
    else:
        raw = dict(args)
    # Filter out callables and dunders/underscore-prefixed (argparse stuffs
    # `_handler` etc. in the namespace which JSON can't serialise).
    args_dict = {
        k: _serialise(v) for k, v in raw.items()
        if not k.startswith("_") and not callable(v)
    }

    manifest = {
        "subcommand": subcommand,
        "args": args_dict,
        "git_sha": _git_sha(),
        "package_version": _package_version(),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cases_per_organ": n_cases_per_organ or {},
    }
    if extra:
        manifest["extra"] = {k: _serialise(v) for k, v in extra.items()}

    path = out_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=_serialise)
    logger.info("wrote manifest: %s", path)
    return path


def _serialise(v: Any) -> Any:
    """Make argparse / Path / set values JSON-friendly."""
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, dict):
        return {k: _serialise(val) for k, val in v.items()}
    if isinstance(v, list | tuple):
        return [_serialise(x) for x in v]
    return v


def write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    """Write a DataFrame to CSV with parent-dir creation and a log line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info("wrote %s (%d rows)", path, len(df))
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to parquet (requires pyarrow).

    Object columns with mixed types are coerced to JSON-encoded strings
    so pyarrow can serialise them without a type-inference failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        if _is_mixed_type(out[col]):
            out[col] = out[col].apply(_to_json_string)
    out.to_parquet(path, engine="pyarrow", index=False)
    logger.info("wrote %s (%d rows)", path, len(df))
    return path


def _is_mixed_type(series: pd.Series) -> bool:
    """True if an object-dtype column contains heterogeneous primitives."""
    seen: set[type] = set()
    for v in series:
        if v is None:
            continue
        # Treat all numeric scalars as compatible; only flag mixes
        # involving bools/strings/lists/dicts.
        if isinstance(v, bool):
            seen.add(bool)
        elif isinstance(v, (int, float)):
            seen.add(float)
        elif isinstance(v, str):
            seen.add(str)
        else:
            seen.add(type(v))
        if len(seen) > 1:
            return True
    return False


def _to_json_string(v):
    """Coerce a single value to a stable JSON-string representation."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    import json
    try:
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(v)


def setup_logging(verbose: bool) -> None:
    """Configure root-logger format. Subcommands call this once at start."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


__all__ = [
    "write_manifest",
    "write_csv",
    "write_parquet",
    "setup_logging",
]
