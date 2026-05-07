#!/usr/bin/env python3
"""One-time compile step for the C5 ablation.

Builds a dev-set from gold annotations under
``{folder}/data/{dataset}/annotations/gold/`` (paired with reports under
``{folder}/data/{dataset}/reports/``), optimises the monolithic
pipeline using ``BootstrapFewShotWithRandomSearch`` with a per-FAIR_SCOPE-
field-accuracy metric, and saves the compiled program JSON for
inference.

Dev set selection:
    - All cases with both a report (organ_n/case_id.txt) and gold
      annotation (organ_n/case_id.json), in deterministic sort order.
    - ``--dev-limit`` caps the size (default 20) so compile time stays
      bounded; the same cap each run yields a reproducible dev split.

Usage::

    python scripts/ablations/compile_dspy.py --folder dummy --dataset tcga \\
        --model gptoss --out workspace/compiled/dspy_compiled_gptoss.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402,F401
from _config_loader import resolve_folder  # noqa: E402

import dspy  # noqa: E402

from digital_registrar_research.ablations.runners._base import (  # noqa: E402
    DATASETS,
    UNIFIED_MODELS,
    discover_organs,
    load_decoding_overrides,
    setup_dspy_lm,
)
from digital_registrar_research.ablations.runners.dspy_monolithic import (  # noqa: E402
    MonolithicPipeline,
)
from digital_registrar_research.benchmarks.eval.metrics import (  # noqa: E402
    field_correct,
)
from digital_registrar_research.benchmarks.eval.scope import (  # noqa: E402
    BREAST_BIOMARKERS,
    FAIR_SCOPE,
)


def _load_dev_examples(folder: Path, dataset: str,
                       limit: int) -> list[dspy.Example]:
    reports_root = folder / "data" / dataset / "reports"
    gold_root = folder / "data" / dataset / "annotations" / "gold"
    if not reports_root.is_dir() or not gold_root.is_dir():
        sys.exit(
            f"need both {reports_root} and {gold_root}; one is missing.")

    examples: list[dspy.Example] = []
    for organ_n, organ_dir in discover_organs(reports_root, None):
        for report_path in sorted(organ_dir.glob("*.txt")):
            case_id = report_path.stem
            gold_path = gold_root / organ_n / f"{case_id}.json"
            if not gold_path.exists():
                continue
            try:
                report_text = report_path.read_text(encoding="utf-8")
                with gold_path.open(encoding="utf-8") as f:
                    gold = json.load(f)
            except Exception:
                continue
            ex = dspy.Example(
                report=report_text,
                gold=gold,
                case_id=case_id,
                organ_n=organ_n,
            ).with_inputs("report")
            examples.append(ex)
            if len(examples) >= limit:
                return examples
    return examples


def _wrap_pipeline_for_compile() -> dspy.Module:
    import logging
    logger = logging.getLogger("dspy_compile")

    class CompileWrapper(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pipe = MonolithicPipeline()

        def forward(self, report):
            return dspy.Prediction(
                output=self.pipe(report=report, logger=logger, fname="compile"))

    return CompileWrapper()


def _compile_metric(example, pred, trace=None) -> float:
    gold = example.gold
    pipe_out = getattr(pred, "output", None) or {}
    fields = list(FAIR_SCOPE)
    if (gold.get("cancer_category") or "").lower() == "breast":
        fields += [f"biomarker_{b}" for b in BREAST_BIOMARKERS]
    correct = 0
    total = 0
    for field in fields:
        result = field_correct(gold, pipe_out, field)
        if result is None:
            continue
        total += 1
        correct += int(bool(result))
    if total == 0:
        return 0.0
    return correct / total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", dest="experiment_root", required=True,
                    type=resolve_folder)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--model", required=True, choices=UNIFIED_MODELS)
    ap.add_argument("--out", required=True, type=Path,
                    help="path to write the compiled program JSON")
    ap.add_argument("--dev-limit", type=int, default=20,
                    help="cap dev-set size (default 20)")
    ap.add_argument("--max-bootstrapped-demos", type=int, default=4)
    ap.add_argument("--max-labeled-demos", type=int, default=2)
    ap.add_argument("--num-candidate-programs", type=int, default=8)
    args = ap.parse_args()

    overrides = load_decoding_overrides(args.model)
    setup_dspy_lm(args.model, overrides=overrides)

    dev_examples = _load_dev_examples(args.experiment_root, args.dataset,
                                      args.dev_limit)
    if len(dev_examples) < 5:
        sys.exit(f"Too few dev examples ({len(dev_examples)}); "
                 "need >= 5 for BootstrapFewShotWithRandomSearch.")

    print(f"[compile] folder={args.experiment_root} dataset={args.dataset} "
          f"model={args.model}")
    print(f"[compile] dev set size = {len(dev_examples)}")
    print(f"[compile] running BootstrapFewShotWithRandomSearch …")

    student = _wrap_pipeline_for_compile()
    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=_compile_metric,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
        num_candidate_programs=args.num_candidate_programs,
        num_threads=1,
    )
    compiled = optimizer.compile(student, trainset=dev_examples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(args.out))
    print(f"[compile] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
