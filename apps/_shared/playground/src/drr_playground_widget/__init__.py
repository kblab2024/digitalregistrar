"""Shared Streamlit playground for chunker + single-signature inspection.

Used by both ``apps/infer-gui`` (as a "Playground" tab) and
``apps/schema-editor`` (as a "Try it" tab — passes the live ``OrganState``
so edits to group instructions show up immediately in the rendered LM
prompt).

Only public entry point is :func:`render_playground`. Heavy imports
(DSPy, the pipeline factory) happen lazily inside the renderer so
importing this package is cheap.
"""
from __future__ import annotations

from .widget import render_playground

__all__ = ["render_playground"]
