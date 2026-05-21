"""Streamlit GUI for editing the three-layer per-organ schema.

Entry point: ``registrar-schema-gui`` console script (see ``pyproject.toml``)
which delegates to :func:`app.main_cli`. The app loads any registered
organ (``CASE_MODELS``) into an in-memory state tree, lets the user
edit it across five tabs, then atomically writes the three source
files back and regenerates ``data/<organ>.json``.

Stack: Streamlit (already in the ``[annotation]`` extra), plus
``jinja2`` and ``tomli_w`` for the writers. ``black`` / ``ruff format``
are used opportunistically to keep saved Layer-1 files pre-commit-clean.

The package is intentionally a thin wrapper around the existing
``schemas/`` machinery — it never reimplements parsing or schema
generation. Loaders read on-disk files; writers re-emit them; the
existing ``registrar-schemas`` CLI regenerates the JSON snapshot.
"""
from __future__ import annotations

__all__: list[str] = []
