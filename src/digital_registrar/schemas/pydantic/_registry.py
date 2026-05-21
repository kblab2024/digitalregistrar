"""Auto-discovery of per-organ Pydantic case-models (Layer 1).

Walks the ``schemas.pydantic`` package, imports every ``<organ>.py`` that
doesn't start with ``_``, and pulls out the class named ``<Organ>CancerCase``
(PascalCase of the snake_case file stem, plus the ``CancerCase`` suffix).

The registry replaces a hand-maintained list of imports in
``__init__.py``. Drop a new file into the directory and the next process
import picks it up — no edits to index files.

Naming convention enforced
--------------------------
- File ``lung.py`` must export class ``LungCancerCase``.
- File ``colorectal.py`` must export class ``ColorectalCancerCase``.
- File ``head_neck.py`` (hypothetical) must export class ``HeadNeckCancerCase``.

The internal nested-type-prefix override map
(``_case_builder._TYPE_PREFIX_OVERRIDES``, e.g. ``ColorectalCancerCase``
→ ``Colon`` for nested types like ``ColonMargin``) is NOT involved here.
The registry key is always the snake_case file stem; the override only
affects nested-type names emitted by ``@assemble_case_model``.
"""
from __future__ import annotations

import functools
import importlib
import pkgutil

from pydantic import BaseModel


def _pascal_case(snake: str) -> str:
    """``"head_neck"`` -> ``"HeadNeck"``. Works for single-word names too."""
    return "".join(part.capitalize() for part in snake.split("_"))


@functools.cache
def discover_case_models() -> dict[str, type[BaseModel]]:
    """Return ``{organ_key: <Organ>CancerCase}`` by walking this package.

    Cached so the discovery cost is paid once even if both
    ``__init__.py`` and ``_common.py`` call it during package import.
    """
    package_name = __name__.rsplit(".", 1)[0]  # 'digital_registrar.schemas.pydantic'
    package = importlib.import_module(package_name)
    out: dict[str, type[BaseModel]] = {}

    for mod_info in pkgutil.iter_modules(package.__path__):
        name = mod_info.name
        if name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{name}")
        expected_cls = f"{_pascal_case(name)}CancerCase"
        cls = getattr(module, expected_cls, None)
        if cls is None:
            raise RuntimeError(
                f"{package_name}.{name}: auto-discovery expected class "
                f"{expected_cls!r}, but the module does not export it. "
                f"Either rename the class to match the file, or rename the "
                f"file to match the class (file stem in snake_case, class "
                f"in PascalCase + 'CancerCase')."
            )
        if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            raise RuntimeError(
                f"{package_name}.{name}.{expected_cls} is not a BaseModel subclass."
            )
        out[name] = cls

    return dict(sorted(out.items()))


__all__ = ["discover_case_models"]
