"""Thin wrapper over the `tnmhelper` Layer-3 API.

Points `tnmhelper` at the data bundle shipped inside this package
(`staging/data/tnmhelper_data.zip`) and re-exports the public functions
so DSPy-side code never touches an uncompressed `tnmhelper_data/` tree.
"""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import tnmhelper
from tnmhelper import PASSTHROUGH_DEFAULTS
from tnmhelper.backend._shared import Edition
from tnmhelper.schema import DerivedStage, ModelSpec, ObservableSpec

_BUNDLE_PACKAGE = "digital_registrar_research.staging.data"
_BUNDLE_NAME = "tnmhelper_data.zip"

_configured = False
_active_source: Path | None = None


def _packaged_bundle_path() -> Path:
    res = files(_BUNDLE_PACKAGE).joinpath(_BUNDLE_NAME)
    return Path(str(res))


def _ensure_configured() -> None:
    global _configured, _active_source
    if _configured:
        return
    bundle = _packaged_bundle_path()
    if not bundle.is_file():
        raise RuntimeError(
            f"packaged staging bundle missing: {bundle}. "
            "Rebuild with `python -m tnmhelper.bundle export`."
        )
    tnmhelper.set_data_source(bundle)
    _active_source = bundle
    _configured = True


def set_data_source_override(path: str | Path) -> None:
    """Force a specific data source (zip or dir). Intended for tests."""
    global _configured, _active_source
    p = Path(path)
    tnmhelper.set_data_source(p)
    _active_source = p
    _configured = True


def active_data_source() -> Path | None:
    """Return the resolved data source path (None until first use)."""
    return _active_source


def organs() -> list[str]:
    _ensure_configured()
    return tnmhelper.organs()


def editions_for(organ: str) -> list[Edition]:
    _ensure_configured()
    return tnmhelper.editions_for(organ)


def model_spec(organ: str, edition: Edition | str) -> ModelSpec:
    _ensure_configured()
    return tnmhelper.model_spec(organ, edition)


def observable_schema(
    organ: str, edition: Edition | str
) -> dict[str, ObservableSpec]:
    _ensure_configured()
    return tnmhelper.observable_schema(organ, edition)


def derive_tnm(
    organ: str,
    edition: Edition | str,
    observations: dict[str, Any],
    *,
    pass_through: dict[str, Any] | None = None,
) -> dict[str, str]:
    _ensure_configured()
    return tnmhelper.derive_tnm(organ, edition, observations, pass_through=pass_through)


def stage_from_observations(
    organ: str,
    edition: Edition | str,
    observations: dict[str, Any],
    **pass_through: Any,
) -> DerivedStage:
    _ensure_configured()
    return tnmhelper.stage_from_observations(organ, edition, observations, **pass_through)


def stage(organ: str, edition: Edition | str, **column_values: str) -> DerivedStage:
    _ensure_configured()
    return tnmhelper.stage(organ, edition, **column_values)


__all__ = [
    "PASSTHROUGH_DEFAULTS",
    "active_data_source",
    "derive_tnm",
    "editions_for",
    "model_spec",
    "observable_schema",
    "organs",
    "set_data_source_override",
    "stage",
    "stage_from_observations",
]
