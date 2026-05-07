"""Run-ID helpers for multi-machine sweeps.

When the same dataset/model is processed on more than one host, every machine
needs its own slot space so concurrent writes do not collide. The mechanism is
deliberately minimal: a short, stable, opt-in slug is appended to each
``runNN`` directory name. Two machines with different slugs scan disjoint
slot sets and never see each other's directories.

Resolution order for the slug (first non-empty wins):
  1. ``DRR_MACHINE_ID`` env var — one-shot CLI override.
  2. ``machine_id:`` field in ``configs/local/runtime.yaml`` — versioned
     per-checkout setting (the ``configs/local/`` tree is gitignored).
  3. Empty string — emits legacy ``runNN`` with no suffix, preserving
     single-machine workflows untouched.

Hostname is intentionally NOT used as a fallback: laptop hostnames flap
between forms like ``mbp.local`` and ``mbp-2.local`` and would silently
fragment one machine's runs across multiple suffixes.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_PATH = REPO_ROOT / "configs" / "local" / "runtime.yaml"

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,11}$")
RUN_ID_RE = re.compile(r"^run(\d+)(?:-([a-z0-9][a-z0-9-]*))?$")


def _validate(slug: str) -> str:
    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"machine_id {slug!r} is invalid; must match {_SLUG_RE.pattern} "
            f"(lowercase alnum + hyphens, 1-12 chars, leading alnum)."
        )
    return slug


@lru_cache(maxsize=1)
def machine_slug() -> str:
    """Return the resolved machine slug (possibly empty)."""
    env = os.environ.get("DRR_MACHINE_ID", "").strip()
    if env:
        return _validate(env)
    if RUNTIME_CONFIG_PATH.is_file():
        import yaml
        with RUNTIME_CONFIG_PATH.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        candidate = (data.get("machine_id") or "").strip()
        if candidate:
            return _validate(candidate)
    return ""


def format_run_id(slot: int, *, padded: bool) -> str:
    """Return ``runNN[-slug]`` (zero-padded) or ``runN[-slug]``."""
    base = f"run{slot:02d}" if padded else f"run{slot}"
    slug = machine_slug()
    return f"{base}-{slug}" if slug else base


def parse_run_id(name: str) -> tuple[int, str]:
    """Inverse of :func:`format_run_id`. Returns ``(slot, slug)``; slug is
    ``""`` for legacy unsuffixed names. Raises ``ValueError`` on garbage."""
    m = RUN_ID_RE.fullmatch(name)
    if not m:
        raise ValueError(f"not a valid run id: {name!r}")
    return int(m.group(1)), (m.group(2) or "")
