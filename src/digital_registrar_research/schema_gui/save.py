"""Atomic save flow for one organ across all three layers.

Render → format → smoke-import → atomic replace → regen JSON. Each
step records what happened so the UI can surface a clean status line
instead of a stack trace.

Race-condition detection: if a layer's on-disk mtime is newer than the
mtime we captured at load time, abort the save and ask the user to
reload. Two-tab editing is the failure mode we're guarding against.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .loaders import _schemas_root
from .state import OrganState
from .validation import validate
from .writers import render_layer1, render_layer2, render_layer3


@dataclass
class SaveReport:
    """What happened during a save attempt — surfaced to the UI."""

    success: bool = False
    files_written: list[Path] = field(default_factory=list)
    json_regenerated: bool = False
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    error: str = ""
    stderr: str = ""


def save_organ(state: OrganState, *, regen_json: bool = True) -> SaveReport:
    """Validate, render, and atomically persist ``state`` to disk.

    On success, the three layer files are updated and (if ``regen_json``)
    ``data/<organ>.json`` is refreshed. On any failure, no on-disk file
    is modified — temp files in the staging dir are cleaned up.
    """
    rep = SaveReport()

    # Step 1: validate.
    val = validate(state)
    rep.validation_errors = list(val.errors)
    rep.validation_warnings = list(val.warnings)
    if not val.ok():
        rep.error = f"validation failed ({len(val.errors)} errors)"
        return rep

    root = _schemas_root()
    layer1_path = root / "pydantic" / f"{state.organ_key}.py"
    layer2_path = root / "extraction" / f"{state.organ_key}.py"
    layer3_path = root / "aliases" / f"{state.organ_key}.toml"

    # Step 2: race-condition check.
    drift = _detect_mtime_drift(state, layer1_path, layer2_path, layer3_path)
    if drift:
        rep.error = (
            f"on-disk file changed since load: {drift}. "
            "Reload to merge before saving."
        )
        return rep

    # Step 3: render to temp files.
    try:
        l1_tmp = _stage(render_layer1(state), layer1_path)
        l2_tmp = _stage(render_layer2(state), layer2_path)
        l3_tmp = _stage(render_layer3(state), layer3_path)
    except Exception as e:
        rep.error = f"render failed: {e!r}"
        return rep

    # Step 4: best-effort format for Layer 1 / Layer 2.
    for tmp in (l1_tmp, l2_tmp):
        _format_python(tmp)

    # Step 5: smoke-import via subprocess (don't pollute current interpreter).
    smoke = _smoke_import(state.organ_key, l1_tmp, l2_tmp, l3_tmp)
    if smoke.returncode != 0:
        rep.error = "smoke-import failed; rendered file would not load"
        rep.stderr = smoke.stderr
        for tmp in (l1_tmp, l2_tmp, l3_tmp):
            tmp.unlink(missing_ok=True)
        return rep

    # Step 6: atomic replace.
    if not state.layer1_read_only_reason:
        os.replace(l1_tmp, layer1_path)
        rep.files_written.append(layer1_path)
    else:
        l1_tmp.unlink(missing_ok=True)
    os.replace(l2_tmp, layer2_path)
    rep.files_written.append(layer2_path)

    if state.aliases:
        os.replace(l3_tmp, layer3_path)
        rep.files_written.append(layer3_path)
    elif state.has_layer3_file:
        # User deleted all aliases — write empty file rather than removing.
        os.replace(l3_tmp, layer3_path)
        rep.files_written.append(layer3_path)
    else:
        l3_tmp.unlink(missing_ok=True)

    # Step 7: regenerate JSON.
    if regen_json:
        proc = subprocess.run(
            [sys.executable, "-m", "digital_registrar_research.schemas.generate"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            rep.error = "registrar-schemas (regen) failed; sources are saved but JSON may be stale"
            rep.stderr = proc.stderr
            return rep
        rep.json_regenerated = True

    rep.success = True
    return rep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_mtime_drift(
    state: OrganState,
    layer1: Path,
    layer2: Path,
    layer3: Path,
) -> str:
    """Return a non-empty reason if any layer was modified since load."""
    if layer1.is_file() and layer1.stat().st_mtime > state.layer1_mtime_at_load + 1e-6:
        return f"{layer1.name}"
    if layer2.is_file() and layer2.stat().st_mtime > state.layer2_mtime_at_load + 1e-6:
        return f"{layer2.name}"
    if (state.has_layer3_file and layer3.is_file()
            and layer3.stat().st_mtime > state.layer3_mtime_at_load + 1e-6):
        return f"{layer3.name}"
    return ""


def _stage(content: str, dest: Path) -> Path:
    """Write ``content`` to a sibling temp file in the same directory as ``dest``.

    Same-directory placement keeps ``os.replace`` atomic on every
    POSIX filesystem we care about. Named with ``.<basename>.tmp.<pid>``
    so partial writes are obvious if interrupted.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.stem}.", suffix=".tmp", dir=str(dest.parent)
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return Path(tmp_name)


def _format_python(path: Path) -> None:
    """Run ``ruff format`` on ``path`` if ruff is on PATH. Silent on failure."""
    if shutil.which("ruff") is None:
        return
    subprocess.run(
        ["ruff", "format", "--quiet", str(path)],
        capture_output=True, text=True,
    )


def _smoke_import(
    organ: str, l1_tmp: Path, l2_tmp: Path, l3_tmp: Path,
) -> subprocess.CompletedProcess[str]:
    """Spawn a fresh Python that imports the rendered Layer 1 + Layer 2.

    We swap the temp files into place inside the subprocess only — the
    parent process's filesystem state is untouched. If the import works,
    we'll commit the same swap in the parent.
    """
    snippet = (
        "import os, sys, shutil, importlib, tempfile\n"
        "from pathlib import Path\n"
        f"organ = {organ!r}\n"
        f"l1_tmp = Path({str(l1_tmp)!r})\n"
        f"l2_tmp = Path({str(l2_tmp)!r})\n"
        "from digital_registrar_research.schemas.pydantic import __file__ as PYD_INIT\n"
        "root = Path(PYD_INIT).resolve().parent.parent\n"
        "pyd_path = root / 'pydantic' / f'{organ}.py'\n"
        "extr_path = root / 'extraction' / f'{organ}.py'\n"
        "pyd_backup = pyd_path.read_bytes() if pyd_path.is_file() else None\n"
        "extr_backup = extr_path.read_bytes() if extr_path.is_file() else None\n"
        "try:\n"
        "    shutil.copy(l1_tmp, pyd_path)\n"
        "    shutil.copy(l2_tmp, extr_path)\n"
        "    importlib.invalidate_caches()\n"
        "    pyd = importlib.import_module(f'digital_registrar_research.schemas.pydantic.{organ}')\n"
        "    extr = importlib.import_module(f'digital_registrar_research.schemas.extraction.{organ}')\n"
        "    assert hasattr(extr, 'FIELD_META') and hasattr(extr, 'GROUP_INSTRUCTIONS')\n"
        "    sys.exit(0)\n"
        "finally:\n"
        "    if pyd_backup is not None:\n"
        "        pyd_path.write_bytes(pyd_backup)\n"
        "    if extr_backup is not None:\n"
        "        extr_path.write_bytes(extr_backup)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True, text=True,
    )


__all__ = ["save_organ", "SaveReport"]
