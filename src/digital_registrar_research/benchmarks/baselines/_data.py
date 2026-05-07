"""Shared dataset loader for the ClinicalBERT and gpt4 baselines.

Two discovery entry points, with different label requirements:

* ``load_cases`` — walks the gold-annotation tree under
  ``<root>/data/<dataset>/annotations/gold/<organ_n>/*.json``. Every case
  has a label, so this is the training-side discovery (and any eval that
  needs gold). Reports are derived by convention as
  ``<root>/data/<dataset>/reports/<organ_n>/<case_id>.txt``.

* ``load_predict_cases`` — walks the reports tree directly at
  ``<root>/data/<dataset>/reports/<organ_n>/*.txt``. Gold annotations are
  best-effort enrichment: if the matching JSON exists, ``cancer_category``
  is populated; otherwise it stays ``None``. This is the predict-side
  discovery — prediction never needs labels, so the runners must not
  silently fail just because gold is missing or partial.

There are no train/test splits within a corpus: the cross-corpus baseline
uses every case of CMUH for training and every case of TCGA for prediction.
Disjointness is guaranteed by the dataset boundary (CMUH and TCGA are
disjoint corpora) and enforced at predict time by a dataset-disjointness
check on the checkpoint metadata.

Returns a uniform list of case dicts:
    {id, dataset, organ_n, organ_name, cancer_category,
     report_path, annotation_path}

(``organ_name`` and ``annotation_path`` may be ``None`` for predict-side
cases without gold; the others are always populated.)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..organs import organs_for

_LOGGER = logging.getLogger(__name__)
_CASE_ID_RE = re.compile(r"^([a-z]+)(\d+)_(\d+)$")


def _organ_n(case_id: str, dataset: str) -> str:
    m = re.match(rf"{re.escape(dataset)}(\d+)_", case_id)
    if not m:
        raise ValueError(
            f"could not parse organ_n from id {case_id!r} for dataset {dataset!r}"
        )
    return m.group(1)


def _walk_dataset(dataset: str, root: Path) -> list[dict]:
    """Walk ``<root>/data/<dataset>/annotations/gold/`` and return case dicts."""
    gold_root = root / "data" / dataset / "annotations" / "gold"
    reports_root = root / "data" / dataset / "reports"
    if not gold_root.is_dir():
        _LOGGER.error(
            "no gold annotations under %s for dataset %s — workspace not "
            "populated with labels? (training-side discovery; predict callers "
            "should use load_predict_cases.)",
            gold_root, dataset,
        )
        return []

    organ_map = organs_for(dataset)  # {organ_n_int: organ_name}
    cases: list[dict] = []
    for ann_path in sorted(gold_root.rglob("*.json")):
        organ_n = ann_path.parent.name
        case_id = ann_path.stem
        report_path = reports_root / organ_n / f"{case_id}.txt"
        try:
            with ann_path.open(encoding="utf-8") as f:
                ann = json.load(f)
        except json.JSONDecodeError as e:
            _LOGGER.warning("skipping malformed annotation %s: %s", ann_path, e)
            continue
        organ_name = organ_map.get(int(organ_n)) if organ_n.isdigit() else None
        cases.append({
            "id": case_id,
            "dataset": dataset,
            "organ_n": organ_n,
            "organ_name": organ_name,
            "cancer_category": ann.get("cancer_category"),
            "report_path": str(report_path),
            "annotation_path": str(ann_path),
        })
    return cases


def _walk_reports(dataset: str, root: Path) -> list[dict]:
    """Walk ``<root>/data/<dataset>/reports/<organ_n>/*.txt``.

    Gold enrichment is best-effort: if the matching JSON exists at
    ``<gold>/<organ_n>/<case_id>.json`` it's parsed and ``cancer_category``
    / ``annotation_path`` are populated; otherwise both stay ``None``.

    organ_n directories that aren't numeric, are hidden (start with ``_``),
    or aren't in ``configs/organ_code.yaml`` for this dataset are skipped
    with a warning.
    """
    reports_root = root / "data" / dataset / "reports"
    gold_root = root / "data" / dataset / "annotations" / "gold"
    if not reports_root.is_dir():
        _LOGGER.error(
            "no reports under %s for dataset %s — workspace not populated?",
            reports_root, dataset,
        )
        return []

    organ_map = organs_for(dataset)  # {organ_n_int: organ_name}
    cases: list[dict] = []
    for organ_dir in sorted(reports_root.iterdir(), key=lambda p: p.name):
        if not organ_dir.is_dir() or organ_dir.name.startswith("_"):
            continue
        if not organ_dir.name.isdigit():
            _LOGGER.warning(
                "non-numeric organ dir %s under %s — skipping",
                organ_dir.name, reports_root,
            )
            continue
        organ_n_int = int(organ_dir.name)
        if organ_n_int not in organ_map:
            _LOGGER.warning(
                "reports/%s/ has unknown organ_n=%s for dataset %s — skipping",
                organ_dir.name, organ_n_int, dataset,
            )
            continue
        organ_name = organ_map[organ_n_int]

        for report_path in sorted(organ_dir.glob("*.txt")):
            case_id = report_path.stem
            ann_path = gold_root / organ_dir.name / f"{case_id}.json"
            cancer_category: str | None = None
            ann_path_str: str | None = None
            if ann_path.is_file():
                try:
                    with ann_path.open(encoding="utf-8") as f:
                        cancer_category = json.load(f).get("cancer_category")
                    ann_path_str = str(ann_path)
                except json.JSONDecodeError as e:
                    _LOGGER.warning(
                        "skipping malformed gold %s: %s", ann_path, e,
                    )

            cases.append({
                "id": case_id,
                "dataset": dataset,
                "organ_n": organ_dir.name,
                "organ_name": organ_name,
                "cancer_category": cancer_category,
                "report_path": str(report_path),
                "annotation_path": ann_path_str,
            })
    return cases


def load_cases(
    datasets: list[str],
    root: Path,
    organs: set[str] | None = None,
    included_only: bool = False,
) -> list[dict]:
    """Pool gold-annotation cases across `datasets` and filter.

    Training-side discovery: walks ``annotations/gold/``. Every case has
    a label. For predict-side discovery (where gold is optional), use
    :func:`load_predict_cases` instead.

    Args:
        datasets: dataset names to include (e.g. ["cmuh", "tcga"]).
        root: experiment root containing ``data/<dataset>/`` subtrees
            (use ``dummy`` for dev, ``workspace`` for production).
        organs: keep only cases whose cancer_category is in this set.
        included_only: drop cases whose cancer_excision_report is False.
    """
    out: list[dict] = []
    for ds in datasets:
        out.extend(_walk_dataset(ds, root))

    if organs is not None:
        out = [c for c in out if c.get("cancer_category") in organs]

    if included_only:
        kept = []
        for c in out:
            with open(c["annotation_path"], encoding="utf-8") as f:
                if json.load(f).get("cancer_excision_report"):
                    kept.append(c)
        out = kept

    return out


def load_predict_cases(
    datasets: list[str],
    root: Path,
    organs: set[str] | None = None,
) -> list[dict]:
    """Predict-side case discovery: walks reports/, gold optional.

    Cases are enumerated from ``<root>/data/<dataset>/reports/<organ_n>/*.txt``
    so prediction works even when gold annotations are missing or partial.
    When a matching gold JSON does exist at
    ``<root>/data/<dataset>/annotations/gold/<organ_n>/<case_id>.json``,
    its ``cancer_category`` is attached to the case dict; otherwise that
    field stays ``None``.

    The ``organs`` filter compares against ``organ_name`` (derived from
    the directory number via ``configs/organ_code.yaml``), not against
    ``cancer_category``. So 'others' / null / non-excision cases are
    still in scope for prediction — the model has to classify them, not
    skip them.

    Args:
        datasets: dataset names to include (e.g. ["tcga"]).
        root: experiment root containing ``data/<dataset>/`` subtrees.
        organs: keep only cases whose organ_name is in this set
            (e.g. ``{"breast", "colorectal"}``).
    """
    out: list[dict] = []
    for ds in datasets:
        out.extend(_walk_reports(ds, root))
    if organs is not None:
        out = [c for c in out if c["organ_name"] in organs]
    return out


def per_dataset_counts(cases: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in cases:
        counts[c["dataset"]] = counts.get(c["dataset"], 0) + 1
    return counts
