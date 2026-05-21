#!/usr/bin/env python3
"""Deprecated — use scripts/baselines/run_bert.py instead.

The cross-platform runner ``run_bert.py`` auto-detects MPS / CUDA / CPU
and orchestrates both heads (CLS, QA, merged) with the canonical
predictions tree::

    {folder}/results/predictions/{dataset}/clinicalbert/{cls|qa|merged}/{organ_n}/{case_id}.json

Migration::

    # Old
    python scripts/baselines/predict_clinicalbert_cls_mac.py \\
        --annotations data/tcga_annotation_20251117 \\
        --dataset     data/tcga_dataset_20251117 \\
        --ckpt        ckpts/clinicalbert_cls.pt \\
        --out         workspace/results/benchmarks/clinicalbert_cls

    # New
    python scripts/baselines/run_bert.py \\
        --folder workspace --dataset tcga \\
        --heads cls qa merged \\
        --ckpt-cls ckpts/clinicalbert_cls.pt \\
        --ckpt-qa  ckpts/clinicalbert_qa
"""
from __future__ import annotations

import sys


_MIGRATION = """\
predict_clinicalbert_cls_mac.py has been retired in favor of run_bert.py.
Run instead:

    python scripts/baselines/run_bert.py --folder <dummy|workspace|path> \\
        --dataset <cmuh|tcga> --heads cls qa merged

See docs/benchmarks/ for the full benchmark workflow.
"""


def main() -> int:
    print(_MIGRATION, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
