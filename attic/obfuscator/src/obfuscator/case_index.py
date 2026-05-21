"""Walk a source workspace and group files by (dataset, organ_n, case_id).

If the source has no `data/{dataset}/reports/{organ_n}/*.txt` (e.g. just a dummy skeleton),
synthesize a default case set from configs/datasets/*.yaml + configs/annotators/annotators.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml


@dataclass
class CaseRecord:
    dataset: str
    organ_n: str
    case_id: str
    report_path: Path | None = None
    # All annotation files keyed by slot name (e.g. "gold", "nhc_with_preann").
    annotation_paths: dict[str, Path] = field(default_factory=dict)
    # Preannotation files keyed by model_slug.
    preannotation_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class WorkspaceIndex:
    src_root: Path
    cases: dict[tuple[str, str, str], CaseRecord]
    datasets: list[str]
    organs_per_dataset: dict[str, list[str]]
    other_files: list[Path]
    """Files outside the case-structured tree (configs, models/, splits, etc.)."""


_REPORTS_GLOB = "data/*/reports/*/*.txt"
_ANNOTATIONS_GLOB = "data/*/annotations/*/*/*.json"
_PREANNOTATION_GLOB = "data/*/preannotation/*/*/*.json"


def index_workspace(src: Path) -> WorkspaceIndex:
    """Build a complete case index of `src` plus a list of structural files."""
    src = src.resolve()
    cases: dict[tuple[str, str, str], CaseRecord] = {}

    # Pass 1: discover cases from reports (the canonical case-set).
    for report in src.glob(_REPORTS_GLOB):
        rel = report.relative_to(src).parts  # data, ds, reports, organ_n, case.txt
        if len(rel) < 5:
            continue
        dataset, organ_n, case_id = rel[1], rel[3], report.stem
        key = (dataset, organ_n, case_id)
        rec = cases.setdefault(key, CaseRecord(dataset, organ_n, case_id))
        rec.report_path = report

    # Pass 2: discover annotations.
    for ann in src.glob(_ANNOTATIONS_GLOB):
        rel = ann.relative_to(src).parts  # data, ds, annotations, slot, organ_n, case.json
        if len(rel) < 6:
            continue
        dataset, slot, organ_n, case_id = rel[1], rel[3], rel[4], ann.stem
        key = (dataset, organ_n, case_id)
        rec = cases.setdefault(key, CaseRecord(dataset, organ_n, case_id))
        rec.annotation_paths[slot] = ann

    # Pass 3: discover preannotations.
    for pre in src.glob(_PREANNOTATION_GLOB):
        rel = pre.relative_to(src).parts  # data, ds, preannotation, model, organ_n, case.json
        if len(rel) < 6:
            continue
        dataset, model_slug, organ_n, case_id = rel[1], rel[3], rel[4], pre.stem
        key = (dataset, organ_n, case_id)
        rec = cases.setdefault(key, CaseRecord(dataset, organ_n, case_id))
        rec.preannotation_paths[model_slug] = pre

    # If no cases found, synthesize from configs.
    if not cases:
        cases = _synthesize_cases(src)

    datasets = sorted({k[0] for k in cases.keys()})
    organs_per_dataset: dict[str, list[str]] = {}
    for ds in datasets:
        organs_per_dataset[ds] = sorted({k[1] for k in cases.keys() if k[0] == ds},
                                        key=lambda x: int(x) if x.isdigit() else 0)

    other_files = _discover_other_files(src)
    return WorkspaceIndex(
        src_root=src, cases=cases, datasets=datasets,
        organs_per_dataset=organs_per_dataset, other_files=other_files,
    )


def _synthesize_cases(src: Path) -> dict[tuple[str, str, str], CaseRecord]:
    """Generate a default case set from configs/datasets/*.yaml.

    Defaults match dummy/README: cmuh = 100 per organ, tcga = 50 per organ.
    """
    cases: dict[tuple[str, str, str], CaseRecord] = {}
    # Honor explicit configs/datasets/{ds}.yaml when present.
    configs_root = src / "configs" / "datasets"
    if not configs_root.is_dir():
        # Try the inference repo configs as fallback.
        configs_root = Path(__file__).resolve().parents[3] / "configs" / "datasets"
    n_per_organ = {"cmuh": 100, "tcga": 50}
    for ds_name, default_n in n_per_organ.items():
        cfg_path = configs_root / f"{ds_name}.yaml"
        if not cfg_path.is_file():
            continue
        with cfg_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        organs = cfg.get("organs") or []
        for organ_entry in organs:
            organ_n = str(organ_entry.get("n"))
            if not organ_n.isdigit():
                continue
            for i in range(1, default_n + 1):
                case_id = f"{ds_name}{organ_n}_{i}"
                key = (ds_name, organ_n, case_id)
                cases[key] = CaseRecord(ds_name, organ_n, case_id)
    return cases


def _discover_other_files(src: Path) -> list[Path]:
    """Return all files NOT covered by the case-walk handlers.

    Handler globs cover reports/annotations/preannotation; everything else
    (configs, models, splits, manifest, csvs, logs, .gitkeep, etc.) is reported
    here so the router can dispatch it.
    """
    out: list[Path] = []
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        if rel.startswith("data/") and (
            "/reports/" in rel or "/annotations/" in rel or "/preannotation/" in rel
        ) and (rel.endswith(".txt") or rel.endswith(".json")):
            continue
        out.append(p)
    return out


def iter_predictions_and_runs(src: Path) -> Iterator[Path]:
    """Yield existing prediction directories under results/predictions/...

    Used to remap _summary.json/_run_meta.json/_log.jsonl/_run.log.
    """
    pred_root = src / "results" / "predictions"
    if not pred_root.is_dir():
        return
    yield from (p for p in pred_root.rglob("*") if p.is_file())
