"""Per-case orchestration: drives the layered generation and writes the artifact set.

For each case, produces:
  - 5 annotation slot files (gold + 4 annotator/mode variants)
  - 1 preannotation file (gpt_oss_20b)
  - 1 report .txt
  - N predictions per (model, run) combo
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from . import profiles as profiles_mod
from .case_index import CaseRecord, WorkspaceIndex
from .generation.annotation import derive_annotation
from .generation.canonical import sample_canonical
from .generation.prediction import derive_prediction
from .generation.realization import derive_realization
from .generation.report import render_report
from .manifest import Manifests
from .profiles import ANNOTATOR_SLOTS, cancer_excision_rate
from .run_planner import RunPlan
from .schemas import CANCER_CATEGORIES, organ_name, validate_cancer_data
from .seeding import derive


PREANN_MODEL_SLUG = "gpt_oss_20b"


def process_case(case: CaseRecord, out_root: Path, master_seed: int,
                 plans: Iterable[RunPlan], manifests: Manifests,
                 *, shapes_only: bool = False, validate: bool = True) -> None:
    """Generate the full artifact set for one case and write to out_root."""
    seed_parts = (case.dataset, case.organ_n, case.case_id)
    rng = derive(master_seed, *seed_parts, "canonical")

    organ = organ_name(case.dataset, case.organ_n)
    if organ is None:
        manifests.record_skipped(
            Path(f"<synthetic:{case.case_id}>"),
            f"unknown organ_n {case.organ_n!r} for dataset {case.dataset!r}",
        )
        return

    # Decide cancer vs non-cancer (matches dummy README's 80/20 split).
    is_cancer = derive(master_seed, *seed_parts, "is_cancer").random() < cancer_excision_rate()
    cancer_category = organ if is_cancer else None

    if shapes_only:
        canonical: dict = {}
        realization: dict = {}
    else:
        canonical = sample_canonical(organ, rng) if is_cancer else {}
        realization = derive_realization(canonical, organ, master_seed, seed_parts) if is_cancer else {}

    # 1. Preannotation (used as bias source for *_with_preann annotators).
    preann = None
    if is_cancer and not shapes_only:
        preann = derive_prediction(realization, organ, PREANN_MODEL_SLUG, "run01",
                                   master_seed, seed_parts)
    _write_wrapped(
        out_root / "data" / case.dataset / "preannotation" / PREANN_MODEL_SLUG
        / case.organ_n / f"{case.case_id}.json",
        preann if is_cancer else {}, is_cancer, cancer_category,
        manifests, "preannotation", organ, validate=validate,
    )

    # 2. Five annotation slots.
    for slot in ANNOTATOR_SLOTS:
        if is_cancer and not shapes_only:
            ann = derive_annotation(
                realization, organ, slot, master_seed, seed_parts,
                preann=preann if "with_preann" in slot else None,
            )
        else:
            ann = {}
        _write_wrapped(
            out_root / "data" / case.dataset / "annotations" / slot
            / case.organ_n / f"{case.case_id}.json",
            ann, is_cancer, cancer_category, manifests,
            f"annotation:{slot}", organ, validate=validate,
        )

    # 3. Report .txt — derived from realization (text-truth), not from gold.
    report_text = render_report(
        realization=realization, cancer_category=cancer_category,
        cancer_excision_report=is_cancer, master_seed=master_seed,
        case_id=case.case_id, dataset=case.dataset, organ_n=case.organ_n,
    )
    if shapes_only:
        report_text = f"patient_filename: SYNTHETIC\ntext: lorem ipsum dolor sit amet\n"
    report_dst = out_root / "data" / case.dataset / "reports" / case.organ_n / f"{case.case_id}.txt"
    report_dst.parent.mkdir(parents=True, exist_ok=True)
    report_dst.write_text(report_text, encoding="utf-8")
    manifests.record_output(case.report_path, report_dst, handler="report_text")

    # 4. Predictions for each (model, run) combination.
    for plan in plans:
        if plan.family == "clinicalbert" and not is_cancer:
            # ClinicalBERT also predicts is-cancer; emit a stub.
            pass
        if is_cancer and not shapes_only:
            pred = derive_prediction(realization, organ, plan.model_slug, plan.run_id,
                                     master_seed, seed_parts)
        else:
            pred = {}
        pred_dst = (out_root / "results" / "predictions" / case.dataset
                    / plan.family / plan.model_slug / plan.run_id
                    / case.organ_n / f"{case.case_id}.json")
        _write_wrapped(pred_dst, pred, is_cancer, cancer_category, manifests,
                       f"prediction:{plan.model_slug}/{plan.run_id}", organ,
                       validate=validate)


def _write_wrapped(dst: Path, cancer_data: dict, is_cancer: bool,
                   cancer_category: str | None, manifests: Manifests,
                   handler: str, organ: str, *, validate: bool) -> None:
    payload = {
        "cancer_excision_report": is_cancer,
        "cancer_category": cancer_category,
        "cancer_category_others_description": None,
        "cancer_data": cancer_data,
    }
    if validate and is_cancer and cancer_data:
        errors = validate_cancer_data(organ, cancer_data)
        if errors:
            payload["_obfuscator_validation_errors"] = errors[:5]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    manifests.record_output(None, dst, handler=handler)


def write_predictions_run_metadata(out_root: Path, plans: Iterable[RunPlan],
                                   datasets: Iterable[str],
                                   manifests: Manifests) -> None:
    """Emit per-run-dir metadata banners (_run.log, _summary.json placeholder)."""
    for ds in datasets:
        for plan in plans:
            run_dir = (out_root / "results" / "predictions" / ds / plan.family
                       / plan.model_slug / plan.run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "_run.log"
            log_path.write_text(
                "[obfustrated workspace — synthetic predictions; "
                "no LLM was called to produce these.]\n", encoding="utf-8")
            summary_path = run_dir / "_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "run": plan.run_id,
                    "model_slug": plan.model_slug,
                    "dataset": ds,
                    "seed": plan.seed,
                    "obfustrated_synthetic": True,
                }, f, indent=2)
            manifests.record_output(None, log_path, handler="run_meta_remapper")
            manifests.record_output(None, summary_path, handler="run_meta_remapper")
