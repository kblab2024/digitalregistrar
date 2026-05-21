"""Scaffold a new organ across all three schema layers in one shot.

Usage (CLI)::

    python -m digital_registrar.schemas.create_organ skeletal
    python -m digital_registrar.schemas.create_organ skeletal --overwrite
    python -m digital_registrar.schemas.create_organ skeletal --no-regen-json

Usage (programmatic — for tests and a future GUI schema editor)::

    from digital_registrar.schemas.create_organ import create_organ
    report = create_organ("skeletal")
    print(report.files_written)

What this does
--------------
1. Validates the organ key (lowercase ASCII, `[a-z][a-z0-9_]*`, not already
   registered).
2. Renders three files from the templates in ``_templates/``:
   - ``schemas/pydantic/<organ_key>.py``        (Layer 1: shape)
   - ``schemas/extraction/<organ_key>.py``      (Layer 2: descriptions)
   - ``schemas/aliases/<organ_key>.toml``       (Layer 3: aliases, stub)
3. Smoke-imports the freshly written files (asserts they parse + register).
4. Optionally regenerates the JSON snapshot via ``registrar-schemas``.

The created files are the user's editing surface from this point on. The
scaffolder is a one-shot generator, not a long-lived dependency — editing
the generated files is the expected workflow, and auto-discovery means
no index files need updating.

A future GUI schema editor will populate its own organ specs and call
:func:`create_organ` with a custom ``base_dir`` pointing at a sandbox.
"""
from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ORGAN_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Resolve once: the directory containing this module is ``schemas/``.
_SCHEMAS_ROOT = Path(__file__).resolve().parent
_TEMPLATES_DIR = _SCHEMAS_ROOT / "_templates"


@dataclass
class CreatedOrganReport:
    """Return value from :func:`create_organ`.

    Captures everything a caller (CLI summary, test, future GUI) needs to
    know without re-querying the filesystem.
    """

    organ_key: str
    class_prefix: str
    files_written: list[Path] = field(default_factory=list)
    json_regenerated: bool = False


def _pascal_case(snake: str) -> str:
    """``"head_neck"`` -> ``"HeadNeck"``. Mirrors `_registry._pascal_case`."""
    return "".join(part.capitalize() for part in snake.split("_"))


def _validate_organ_key(organ_key: str, existing_keys: set[str]) -> None:
    """Raise ValueError with a precise message on bad input."""
    if not organ_key:
        raise ValueError("organ_key must not be empty.")
    if not _ORGAN_KEY_RE.fullmatch(organ_key):
        raise ValueError(
            f"organ_key {organ_key!r} must be lowercase ASCII starting with a "
            f"letter, followed by letters / digits / underscores "
            f"(pattern: {_ORGAN_KEY_RE.pattern})."
        )
    if organ_key in existing_keys:
        raise ValueError(
            f"organ_key {organ_key!r} is already registered "
            f"(existing organs: {sorted(existing_keys)}). "
            f"Pass --overwrite to clobber the existing per-organ files."
        )


def _render_template(template_path: Path, organ_key: str, class_prefix: str) -> str:
    """Substitute ``__ORGAN_KEY__`` and ``__CLASS_PREFIX__`` in a template file.

    Uses literal string replacement (not :class:`str.Template` / f-string)
    so the templates themselves can use any Python or TOML syntax without
    escaping concerns.
    """
    text = template_path.read_text(encoding="utf-8")
    return text.replace("__ORGAN_KEY__", organ_key).replace("__CLASS_PREFIX__", class_prefix)


def _existing_registry_keys(base_dir: Path) -> set[str]:
    """Return the set of organ keys currently in the Layer-1 registry.

    Looks at the on-disk Layer-1 directory rather than importing
    ``CASE_MODELS``, so the check works even when ``base_dir`` points
    at a sandbox tree (for tests / future GUI). Skips ``_*.py``.
    """
    pyd_dir = base_dir / "pydantic"
    if not pyd_dir.is_dir():
        return set()
    return {
        p.stem
        for p in pyd_dir.glob("*.py")
        if not p.stem.startswith("_")
    }


def create_organ(
    organ_key: str,
    *,
    class_prefix: str | None = None,
    overwrite: bool = False,
    regen_json: bool = True,
    base_dir: Path | None = None,
) -> CreatedOrganReport:
    """Scaffold a new organ across Layers 1, 2, 3 with the default schema.

    Parameters
    ----------
    organ_key
        Lowercase ASCII identifier used as the registry key, the
        per-organ filename, and the snake-case prefix in field
        descriptions. Pattern: ``[a-z][a-z0-9_]*``.
    class_prefix
        PascalCase prefix for the case-model class and per-organ enums
        (e.g. ``"Skeletal"`` for ``SkeletalCancerCase`` /
        ``SkeletalProcedure``). Defaults to ``_pascal_case(organ_key)``.
    overwrite
        If True, clobber any existing per-organ files. If False
        (default), refuse with FileExistsError on conflict.
    regen_json
        If True (default), invoke ``python -m
        digital_registrar.schemas.generate`` after writing the
        three source files to produce ``schemas/data/<organ_key>.json``.
        Pass False to skip the subprocess (faster; user runs
        ``registrar-schemas`` themselves).
    base_dir
        Directory containing the ``pydantic/``, ``extraction/``, and
        ``aliases/`` subpackages. Defaults to the installed ``schemas/``
        package. Override in tests or sandboxes.

    Returns
    -------
    CreatedOrganReport
        Records the three written paths plus whether the JSON snapshot
        was regenerated.
    """
    base = base_dir if base_dir is not None else _SCHEMAS_ROOT
    class_prefix = class_prefix or _pascal_case(organ_key)

    existing = _existing_registry_keys(base)
    if overwrite:
        existing = existing - {organ_key}
    _validate_organ_key(organ_key, existing)

    targets = {
        base / "pydantic"   / f"{organ_key}.py":   _TEMPLATES_DIR / "pydantic.py.tmpl",
        base / "extraction" / f"{organ_key}.py":   _TEMPLATES_DIR / "extraction.py.tmpl",
        base / "aliases"    / f"{organ_key}.toml": _TEMPLATES_DIR / "aliases.toml.tmpl",
    }

    if not overwrite:
        clashes = [out for out in targets if out.exists()]
        if clashes:
            raise FileExistsError(
                "Refusing to overwrite existing file(s): "
                + ", ".join(str(p) for p in clashes)
                + ". Pass overwrite=True (or --overwrite on the CLI) to clobber."
            )

    files_written: list[Path] = []
    for out_path, template_path in targets.items():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = _render_template(template_path, organ_key, class_prefix)
        out_path.write_text(rendered, encoding="utf-8")
        files_written.append(out_path)

    if base == _SCHEMAS_ROOT:
        # Reload the schema package so the new organ shows up in the live
        # process — verifies the generated files actually import. When
        # base_dir is a sandbox, skip the reload (tests do their own).
        _smoke_reload_and_assert(organ_key)

    json_regenerated = False
    if regen_json and base == _SCHEMAS_ROOT:
        subprocess.run(
            [sys.executable, "-m", "digital_registrar.schemas.generate"],
            check=True,
        )
        json_regenerated = True

    return CreatedOrganReport(
        organ_key=organ_key,
        class_prefix=class_prefix,
        files_written=files_written,
        json_regenerated=json_regenerated,
    )


def _smoke_reload_and_assert(organ_key: str) -> None:
    """Reload schemas.{pydantic,extraction} and assert the organ registered.

    Used only when scaffolding into the live package tree. The order
    matters: we must refresh pydantic's ``CASE_MODELS`` *before* causing
    ``extraction/__init__.py`` to run (which eagerly builds
    ``EXTRACTION_META`` against the live pydantic state). The fix is to
    pop extraction from ``sys.modules`` first so its __init__ runs fresh
    after pydantic has been reloaded.
    """
    import sys

    # Invalidate the import-machinery filesystem caches so
    # pkgutil.iter_modules sees the files we just wrote.
    importlib.invalidate_caches()

    # Reload Layer 1 first. Clear discover_case_models cache, then reload
    # pydantic so its CASE_MODELS includes the new organ.
    pyd_registry = importlib.import_module(
        "digital_registrar.schemas.pydantic._registry"
    )
    pyd_registry.discover_case_models.cache_clear()

    pyd = importlib.import_module("digital_registrar.schemas.pydantic")
    importlib.reload(pyd)

    if organ_key not in pyd.CASE_MODELS:
        raise RuntimeError(
            f"scaffolded {organ_key!r} but it did not register in CASE_MODELS "
            f"after reload. Check the generated pydantic/{organ_key}.py for "
            f"import-time errors."
        )

    # Now Layer 2. Pop extraction from sys.modules (along with its
    # registry submodule) so the next import runs __init__.py against
    # the freshly-reloaded pydantic. Just calling importlib.reload would
    # also work, but only if extraction had already been imported in
    # this process; the pop-then-import flow handles both first-import
    # and reload cases uniformly.
    for mod_name in (
        "digital_registrar.schemas.extraction",
        "digital_registrar.schemas.extraction._registry",
    ):
        sys.modules.pop(mod_name, None)

    extr = importlib.import_module("digital_registrar.schemas.extraction")

    if organ_key not in extr.EXTRACTION_META:
        raise RuntimeError(
            f"scaffolded {organ_key!r} but it did not register in "
            f"EXTRACTION_META after reload. Check the generated "
            f"extraction/{organ_key}.py for import-time errors."
        )


def _print_summary(report: CreatedOrganReport) -> None:
    print(f"[create_organ] scaffolded {report.organ_key!r} "
          f"(class prefix: {report.class_prefix})")
    print("  files written:")
    for p in report.files_written:
        print(f"    - {p}")
    if report.json_regenerated:
        print(f"  schemas/data/{report.organ_key}.json: regenerated")
    print()
    print("Next steps — customize the scaffolded defaults:")
    print(f"  - shape / enums          -> schemas/pydantic/{report.organ_key}.py")
    print(f"  - LM-facing descriptions -> schemas/extraction/{report.organ_key}.py")
    print(f"  - surface-form aliases   -> schemas/aliases/{report.organ_key}.toml")
    if not report.json_regenerated:
        print("  - regenerate JSON        -> `registrar-schemas`")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m digital_registrar.schemas.create_organ",
        description=(
            "Scaffold a new organ across all three schema layers "
            "(pydantic / extraction / aliases) with the default template. "
            "Auto-discovery picks the new organ up on next process import."
        ),
    )
    ap.add_argument("organ_key", help="lowercase identifier, e.g. 'skeletal'")
    ap.add_argument(
        "--class-prefix",
        default=None,
        help="PascalCase class prefix. Defaults to PascalCase(organ_key).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Clobber existing per-organ files. Default: refuse.",
    )
    ap.add_argument(
        "--no-regen-json",
        dest="regen_json",
        action="store_false",
        help="Skip the `registrar-schemas` subprocess after writing files.",
    )
    args = ap.parse_args(argv)

    try:
        report = create_organ(
            args.organ_key,
            class_prefix=args.class_prefix,
            overwrite=args.overwrite,
            regen_json=args.regen_json,
        )
    except (ValueError, FileExistsError, RuntimeError) as e:
        print(f"[create_organ] {e}", file=sys.stderr)
        return 1

    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
