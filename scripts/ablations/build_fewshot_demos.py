#!/usr/bin/env python3
"""Build the few-shot demo registry for the C2/C3 ablations.

Demos are sourced from gold annotations under
``{folder}/data/{dataset}/annotations/gold/`` (paired with reports
under ``{folder}/data/{dataset}/reports/``). Per organ we rank
candidates by FAIR_SCOPE coverage of the gold annotation; the top
``--n-max`` per organ are written to
``configs/ablations/fewshot_demos.yaml`` for the C2/C3 runner to read
at inference time.

Reproducible because:
    * candidates come from a deterministic file enumeration,
    * coverage scoring is purely a property of the gold annotation,
    * tie-breaking is by case_id (lexicographic).

Usage::

    python scripts/ablations/build_fewshot_demos.py --folder dummy --dataset tcga --n-max 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402
from _config_loader import resolve_folder  # noqa: E402

import yaml  # noqa: E402

from digital_registrar_research.ablations.runners._base import (  # noqa: E402
    DATASETS,
    discover_organs,
)
from digital_registrar_research.benchmarks.eval.scope import (  # noqa: E402
    BREAST_BIOMARKERS,
    FAIR_SCOPE,
    get_field_value,
)
from digital_registrar_research.benchmarks.organs import (  # noqa: E402
    organ_n_to_name,
)

DEFAULT_OUT = REPO_ROOT / "configs" / "ablations" / "fewshot_demos.yaml"


def _coverage(gold: dict) -> int:
    """Count populated FAIR_SCOPE fields in a gold annotation."""
    fields = list(FAIR_SCOPE)
    if (gold.get("cancer_category") or "").lower() == "breast":
        fields += [f"biomarker_{b}" for b in BREAST_BIOMARKERS]
    count = 0
    for field in fields:
        v = get_field_value(gold, field)
        if v not in (None, "", [], {}):
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", dest="experiment_root", required=True,
                    type=resolve_folder,
                    help="Experiment root containing data/{dataset}/")
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--n-max", type=int, default=5,
                    help="number of demos per organ to register (>= 5)")
    ap.add_argument("--seed", type=int, default=42,
                    help="recorded in the YAML for traceability")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    reports_root = args.experiment_root / "data" / args.dataset / "reports"
    gold_root = (args.experiment_root / "data" / args.dataset
                 / "annotations" / "gold")
    if not gold_root.is_dir():
        sys.exit(f"gold annotations not found at {gold_root}")

    by_organ: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for organ_n, _organ_dir in discover_organs(reports_root, None):
        organ_name = organ_n_to_name(args.dataset, organ_n)
        if not organ_name or organ_name == "others":
            continue
        gold_organ_dir = gold_root / organ_n
        if not gold_organ_dir.is_dir():
            continue
        for gold_path in sorted(gold_organ_dir.glob("*.json")):
            try:
                with gold_path.open(encoding="utf-8") as f:
                    gold = json.load(f)
            except Exception:
                continue
            score = _coverage(gold)
            by_organ[organ_name].append((score, gold_path.stem))

    organs_out: dict[str, list[str]] = {}
    for organ, scored in sorted(by_organ.items()):
        scored.sort(key=lambda kv: (-kv[0], kv[1]))
        organs_out[organ] = [cid for _score, cid in scored[: args.n_max]]

    out_doc = {
        "seed": args.seed,
        "n_max": args.n_max,
        "folder": str(args.experiment_root.resolve()),
        "dataset": args.dataset,
        "source": "max FAIR_SCOPE coverage per organ from gold annotations",
        "organs": organs_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(out_doc, sort_keys=False),
                        encoding="utf-8")
    print(f"[demos] wrote {args.out}  ({sum(len(v) for v in organs_out.values())} demos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
