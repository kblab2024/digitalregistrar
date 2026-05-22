"""Name-based registries for chunker / router classes.

Custom implementations register via decorator::

    from digital_registrar.chunking import register_chunker, Chunk

    @register_chunker("my_chunker")
    class MyChunker:
        def chunk(self, report: str) -> list[Chunk]:
            ...

After registration the GUI's chunker dropdown picks up the new class
automatically — no edits to core needed.
"""
from __future__ import annotations

from typing import TypeVar

from .protocols import Chunker, Router

CHUNKER_REGISTRY: dict[str, type[Chunker]] = {}
ROUTER_REGISTRY: dict[str, type[Router]] = {}

C = TypeVar("C", bound=type)


def register_chunker(name: str):
    """Decorator: register a Chunker class under ``name``."""

    def deco(cls: C) -> C:
        if name in CHUNKER_REGISTRY:
            raise ValueError(f"Chunker {name!r} already registered")
        cls.name = name
        CHUNKER_REGISTRY[name] = cls  # type: ignore[assignment]
        return cls

    return deco


def register_router(name: str):
    """Decorator: register a Router class under ``name``."""

    def deco(cls: C) -> C:
        if name in ROUTER_REGISTRY:
            raise ValueError(f"Router {name!r} already registered")
        cls.name = name
        ROUTER_REGISTRY[name] = cls  # type: ignore[assignment]
        return cls

    return deco


def get_chunker(name: str) -> type[Chunker]:
    try:
        return CHUNKER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"No chunker registered as {name!r}. "
            f"Available: {sorted(CHUNKER_REGISTRY)}"
        ) from None


def get_router(name: str) -> type[Router]:
    try:
        return ROUTER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"No router registered as {name!r}. "
            f"Available: {sorted(ROUTER_REGISTRY)}"
        ) from None


__all__ = [
    "CHUNKER_REGISTRY",
    "ROUTER_REGISTRY",
    "register_chunker",
    "register_router",
    "get_chunker",
    "get_router",
]
