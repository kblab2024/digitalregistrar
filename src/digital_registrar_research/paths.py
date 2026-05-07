"""Centralised filesystem path resolution for the research package.

All hardcoded references to example-data folders / results / split files
funnel through this module so the repo can be moved or renamed without
breaking downstream code.

The workspace directory name is configurable via the
``DIGITAL_REGISTRAR_WORKSPACE`` env var. Defaults to ``workspace``.
Set it to ``workspace_obfustrated`` (or any other dir under repo root)
to point library code at an alternate workspace tree without touching
this file. Existing callers that import the constants by name still
work — they're computed from the env var at import time.
"""
import os
from pathlib import Path

# src/digital_registrar_research/paths.py → parents[2] is the repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Example data (TCGA gold set) — shipped in-repo. The flat three-folder
# layout (`{prefix}_dataset_{date}/`, `{prefix}_result_{date}/`,
# `{prefix}_annotation_{date}/`) is the contract the doctor-facing annotation
# UI expects (see annotation.io.discover_folders).
DATA_ROOT: Path = REPO_ROOT / "data"
RAW_REPORTS: Path = DATA_ROOT / "tcga_dataset_20251117"
PREANNOTATIONS: Path = DATA_ROOT / "tcga_result_20251117"
GOLD_ANNOTATIONS: Path = DATA_ROOT / "tcga_annotation_20251117"

# Workspace dir name — env-overridable so a Claude session can point library
# code at workspace_obfustrated/ without code edits.
WORKSPACE_DIR_NAME: str = os.environ.get("DIGITAL_REGISTRAR_WORKSPACE", "workspace")
WORKSPACE_ROOT: Path = REPO_ROOT / WORKSPACE_DIR_NAME

# Runtime artifacts (gitignored). The workspace tree mirrors the dummy/
# skeleton: workspace/results/{predictions,eval,ablations,benchmarks}/...
RESULTS_ROOT: Path = WORKSPACE_ROOT / "results"
BENCHMARKS_RESULTS: Path = RESULTS_ROOT / "benchmarks"
ABLATIONS_RESULTS: Path = RESULTS_ROOT / "ablations"


def workspace_root(workspace: str | Path | None = None) -> Path:
    """Get the workspace root, optionally with an explicit override.

    With no argument: returns the env-configured ``WORKSPACE_ROOT``.
    With a string: treated as a directory name under ``REPO_ROOT``
    (e.g. ``workspace``, ``workspace_obfustrated``, ``dummy``).
    With a Path: used as-is (must be absolute or relative to cwd).
    """
    if workspace is None:
        return WORKSPACE_ROOT
    p = Path(workspace)
    if not p.is_absolute() and "/" not in str(workspace) and "\\" not in str(workspace):
        return REPO_ROOT / str(workspace)
    return p


def results_root(workspace: str | Path | None = None) -> Path:
    return workspace_root(workspace) / "results"


# Packaged JSON schemas (generated from canonical Pydantic models)
SCHEMAS_DATA: Path = Path(__file__).resolve().parent / "schemas" / "data"

__all__ = [
    "REPO_ROOT", "DATA_ROOT", "RAW_REPORTS", "PREANNOTATIONS", "GOLD_ANNOTATIONS",
    "WORKSPACE_DIR_NAME", "WORKSPACE_ROOT",
    "RESULTS_ROOT", "BENCHMARKS_RESULTS", "ABLATIONS_RESULTS",
    "workspace_root", "results_root",
    "SCHEMAS_DATA",
]
