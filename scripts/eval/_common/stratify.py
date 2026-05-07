"""Organ index <-> name conversion + per-dataset organ-list helpers.

The dataset folder layout uses numeric organ indices, but the index ->
name mapping is **dataset-specific** (TCGA and CMUH number organs
differently — see ``configs/organ_code.yaml``). This module is a thin
dataset-aware wrapper around
``digital_registrar_research.benchmarks.organs`` so the eval pipeline
never has to load the yaml directly.
"""
from __future__ import annotations

from typing import Iterable

from digital_registrar_research.benchmarks import organs as _organs


def organ_name(dataset: str, idx: int) -> str:
    """Return the organ name for a (dataset, idx).

    Raises ``KeyError`` on unknown indices for that dataset.
    """
    return _organs.organ_name(dataset, idx)


def organ_index(dataset: str, name: str) -> int:
    """Return the numeric index for an organ name within a dataset.

    Raises ``KeyError`` if that dataset doesn't include the organ.
    """
    return _organs.organ_n_for(dataset, name)


def all_organ_indices(dataset: str) -> tuple[int, ...]:
    """Sorted tuple of organ indices defined for ``dataset``."""
    return tuple(sorted(_organs.organs_for(dataset).keys()))


def all_organ_names(dataset: str) -> tuple[str, ...]:
    """Sorted tuple of organ names defined for ``dataset``."""
    return tuple(sorted(_organs.organs_for(dataset).values()))


def parse_organ_arg(values: Iterable[str | int], dataset: str) -> list[int]:
    """Convert a mixed list of organ names / indices into indices for ``dataset``.

    Accepts ``["1", "2", "breast"]`` or ``[1, "lung"]``. Used by the
    ``--organs`` CLI argument so users can write either form. Raises
    :class:`ValueError` on entries the dataset doesn't recognize.
    """
    valid_idx = _organs.organs_for(dataset)
    valid_names_lc = {name.lower(): n for n, name in valid_idx.items()}
    out: list[int] = []
    for v in values:
        if isinstance(v, int):
            if v not in valid_idx:
                raise ValueError(
                    f"unknown organ index for dataset={dataset!r}: {v}"
                )
            out.append(v)
            continue
        s = str(v).strip().lower()
        if s.isdigit():
            i = int(s)
            if i not in valid_idx:
                raise ValueError(
                    f"unknown organ index for dataset={dataset!r}: {i}"
                )
            out.append(i)
        elif s in valid_names_lc:
            out.append(valid_names_lc[s])
        else:
            raise ValueError(
                f"unknown organ for dataset={dataset!r}: {v!r}"
            )
    return sorted(set(out))


def parse_case_id(case_id: str) -> tuple[str, int, int]:
    """Decode ``{dataset}{organ_idx}_{case_num}``.

    Delegates to :func:`organs.parse_case_id` which validates organ_n
    against the dataset's yaml mapping (so ``tcga6_1`` is rejected even
    though 6 is a valid CMUH index).
    """
    return _organs.parse_case_id(case_id)


__all__ = [
    "organ_name",
    "organ_index",
    "all_organ_indices",
    "all_organ_names",
    "parse_organ_arg",
    "parse_case_id",
]
