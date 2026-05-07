"""Paired-case discovery for IAA preann-effect analysis.

Two patterns:

- :func:`discover_paired_cases` — for one annotator (e.g. ``nhc``),
  return the case IDs present in BOTH ``<annotator>_with_preann/`` and
  ``<annotator>_without_preann/``. The without-preann set is a subset
  of the with-preann set; the intersection is the paired cohort.

- :func:`discover_trio` — for IAA against gold, return ``(case_id,
  organ_idx, paths)`` triples where every requested annotator has a
  file. Used by the IAA pairwise scoring.

All metrics computed on the paired cohort are paired-bootstrap eligible
because the case sample is identical across the with/without
conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import Paths
from .stratify import all_organ_indices


@dataclass(frozen=True)
class PairedCase:
    organ_idx: int
    case_id: str
    with_path: Path
    without_path: Path


@dataclass(frozen=True)
class TrioCase:
    organ_idx: int
    case_id: str
    paths: dict[str, Path]  # annotator -> path; only includes existing files


def discover_paired_cases(
    paths: Paths,
    annotator: str,
    *,
    organs: Iterable[int] | None = None,
) -> list[PairedCase]:
    """Return case IDs for which both ``<annotator>_with_preann`` and
    ``<annotator>_without_preann`` exist.

    ``annotator`` is the bare name (``"nhc"``, ``"kpc"``) — the suffix
    is appended internally. The without-preann set is a subset of the
    with-preann set; the returned list is the intersection in
    deterministic sorted order. ``organs`` defaults to the dataset's
    full organ-index set per ``configs/organ_code.yaml``.
    """
    with_dir_name = f"{annotator}_with_preann"
    without_dir_name = f"{annotator}_without_preann"
    organs_t = tuple(organs) if organs is not None else all_organ_indices(paths.dataset)

    with_index: dict[tuple[int, str], Path] = {
        (oi, cid): paths.annotation(with_dir_name, oi, cid)
        for oi, cid in paths.case_ids(with_dir_name, organs_t)
    }
    without_index: dict[tuple[int, str], Path] = {
        (oi, cid): paths.annotation(without_dir_name, oi, cid)
        for oi, cid in paths.case_ids(without_dir_name, organs_t)
    }

    shared = sorted(set(with_index.keys()) & set(without_index.keys()))
    return [
        PairedCase(
            organ_idx=oi,
            case_id=cid,
            with_path=with_index[(oi, cid)],
            without_path=without_index[(oi, cid)],
        )
        for oi, cid in shared
    ]


def discover_trio(
    paths: Paths,
    annotators: Iterable[str],
    *,
    organs: Iterable[int] | None = None,
    require_all: bool = False,
) -> list[TrioCase]:
    """Return ``TrioCase`` entries grouped by ``(organ_idx, case_id)``.

    ``annotators`` is a list of full annotator subdir names (e.g.
    ``["gold", "nhc_with_preann", "kpc_with_preann"]``). When
    ``require_all`` is True, only cases with a file from EVERY annotator
    are returned. Otherwise any case present in >=1 annotator is
    returned, with ``paths`` containing only the existing files.
    ``organs`` defaults to the dataset's full organ-index set per
    ``configs/organ_code.yaml``.
    """
    annotators = list(annotators)
    organs_t = tuple(organs) if organs is not None else all_organ_indices(paths.dataset)
    by_case: dict[tuple[int, str], dict[str, Path]] = {}
    for ann in annotators:
        for oi, cid in paths.case_ids(ann, organs_t):
            key = (oi, cid)
            entry = by_case.setdefault(key, {})
            entry[ann] = paths.annotation(ann, oi, cid)

    out: list[TrioCase] = []
    for (oi, cid), p in sorted(by_case.items()):
        if require_all and not all(a in p for a in annotators):
            continue
        out.append(TrioCase(organ_idx=oi, case_id=cid, paths=p))
    return out


__all__ = [
    "PairedCase",
    "TrioCase",
    "discover_paired_cases",
    "discover_trio",
]
