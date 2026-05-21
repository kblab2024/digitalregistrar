#!/usr/bin/env python3
"""Deprecated — use scripts/baselines/train_bert.py instead.

The cross-platform trainer ``train_bert.py`` auto-detects MPS / CUDA /
CPU and trains both heads (CLS and/or QA) with the canonical pooled
train split contract.

Migration::

    # Old
    python scripts/baselines/train_clinicalbert_cls_mac.py \\
        --annotations data/tcga_annotation_20251117 \\
        --dataset     data/tcga_dataset_20251117 \\
        --epochs 5 --ckpt ckpts/clinicalbert_cls.pt

    # New
    python scripts/baselines/train_bert.py \\
        --folder workspace --datasets cmuh tcga --heads cls qa \\
        --ckpt-cls ckpts/clinicalbert_cls.pt --ckpt-qa ckpts/clinicalbert_qa \\
        --epochs-cls 5 --epochs-qa 3
"""
from __future__ import annotations

import sys


_MIGRATION = """\
train_clinicalbert_cls_mac.py has been retired in favor of train_bert.py.
Run instead:

    python scripts/baselines/train_bert.py --folder <dummy|workspace|path> \\
        --datasets cmuh tcga --heads cls qa

See docs/benchmarks/ for the full benchmark workflow.
"""


def main() -> int:
    print(_MIGRATION, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
