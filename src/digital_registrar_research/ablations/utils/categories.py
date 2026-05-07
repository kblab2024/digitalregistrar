"""Single source of truth for the cancer-category enum used by every
ablation runner that emits or validates a ``cancer_category`` field.

Built from ``configs/organ_code.yaml`` via
:mod:`digital_registrar_research.benchmarks.organs`. CMUH is the
superset (10 organs); TCGA's 5 organs are a strict subset. The
``"others"`` sentinel matches :class:`models.common.is_cancer`'s
``cancer_category`` Literal.
"""
from __future__ import annotations

from ...benchmarks.organs import dataset_organs


def _build() -> list[str]:
    cmuh = dataset_organs("cmuh")
    return sorted(cmuh) + ["others"]


CANCER_CATEGORIES: list[str] = _build()
"""Sorted CMUH organs + ``"others"``. Use this everywhere in ablations
instead of hardcoding the list."""
