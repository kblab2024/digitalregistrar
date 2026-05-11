"""Auto-discovery of per-organ extraction-metadata modules (Layer 2).

Walks the ``schemas.extraction`` package, imports every ``<organ>.py``
that doesn't start with ``_``, and returns the modules so the package
``__init__`` can build :data:`EXTRACTION_META` without a hand-maintained
import list.

Each discovered module must export ``FIELD_META`` (a ``dict[str, FieldMeta]``)
and ``GROUP_INSTRUCTIONS`` (a ``dict[str, str]``). The discoverer
validates presence but not contents — contents are validated by
:func:`._build_organ_meta` and the schema-concordance tests.
"""
from __future__ import annotations

import functools
import importlib
import pkgutil
from types import ModuleType


@functools.cache
def discover_extraction_modules() -> dict[str, ModuleType]:
    """Return ``{organ_key: module}`` for every per-organ file in this package."""
    package_name = __name__.rsplit(".", 1)[0]  # 'digital_registrar_research.schemas.extraction'
    package = importlib.import_module(package_name)
    out: dict[str, ModuleType] = {}

    for mod_info in pkgutil.iter_modules(package.__path__):
        name = mod_info.name
        if name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{name}")
        missing = [attr for attr in ("FIELD_META", "GROUP_INSTRUCTIONS") if not hasattr(module, attr)]
        if missing:
            raise RuntimeError(
                f"{package_name}.{name}: auto-discovery expected exports "
                f"{missing!r}, but they are not defined. Every per-organ "
                f"extraction module must export FIELD_META and "
                f"GROUP_INSTRUCTIONS."
            )
        out[name] = module

    return dict(sorted(out.items()))


__all__ = ["discover_extraction_modules"]
