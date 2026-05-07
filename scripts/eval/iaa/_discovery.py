"""Shared case-discovery helper for the iaa subcommand family.

Both ``run_iaa`` and ``run_iaa_pair`` walk the dir-based annotation
layout (``annotations/<annotator>/<organ_idx>/<case_id>.json``) and
group annotations by ``case_id`` into the ``CaseEntry`` shape that
:func:`iaa.pairwise_iaa` expects. Centralising here avoids drift.
"""
from __future__ import annotations

import logging

from digital_registrar_research.benchmarks.eval.iaa import CaseEntry
from digital_registrar_research.benchmarks.eval.metrics import normalize

from .._common.loaders import ParseError, load_json
from .._common.paths import Paths
from .._common.stratify import organ_name

logger = logging.getLogger(__name__)


def discover_cases_dir_layout(
    *,
    paths: Paths,
    annotators: tuple[str, ...],
    organs: tuple[int, ...],
    case_filter: set[str] | None,
) -> tuple[dict[str, CaseEntry], dict[int, int]]:
    """Walk every annotator subdir and group annotations by case_id.

    Adapts the dir-based layout
    (``annotations/<annotator>/<organ_idx>/<case_id>.json``) to the
    ``CaseEntry`` shape that ``iaa.pairwise_iaa`` expects. The annotator
    name is used as the dict key AND the suffix parameter for downstream
    helpers; ``classify_section`` etc. only look at field names so they
    are unaffected.
    """
    cases: dict[str, CaseEntry] = {}
    n_per_organ: dict[int, int] = {}
    for annotator in annotators:
        for organ_idx, case_id in paths.case_ids(annotator, organs):
            if case_filter and case_id not in case_filter:
                continue
            ann_path = paths.annotation(annotator, organ_idx, case_id)
            try:
                ann = load_json(ann_path)
            except ParseError as e:
                logger.warning("skipping %s (%s): %s", case_id, annotator, e)
                continue
            organ = normalize(ann.get("cancer_category")) or organ_name(
                paths.dataset, organ_idx,
            )
            entry = cases.get(case_id)
            if entry is None:
                entry = CaseEntry(organ=organ, annotations={}, paths={})
                cases[case_id] = entry
                n_per_organ[organ_idx] = n_per_organ.get(organ_idx, 0) + 1
            entry.annotations[annotator] = ann
            entry.paths[annotator] = ann_path
            if entry.organ is None and organ:
                entry.organ = organ
    return cases, n_per_organ


__all__ = ["discover_cases_dir_layout"]
