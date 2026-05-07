"""Select a curated subset of cases for ``without_preann`` annotation mode.

For each ``n/`` organ folder in ``<base>/data/<dataset>/preannotation/<model>/``
we pick a deterministic sample of cancer + non-cancer cases (default 15 + 5)
and copy their report ``.txt`` files into a sibling ``reports_without_preann/``
tree. The annotator UI in ``without_preann`` mode reads from that tree, so
human-from-scratch annotation runs against the curated subset instead of the
full dataset.

Cancer is read from the preannotation JSON: ``cancer_excision_report == true``
selects cancer (excluding ``cancer_category == "others"``); ``False`` selects
non-cancer. Bucketing is by the ``n/`` folder, not the preannotation's
``cancer_category`` field, so the subset lines up with reports/preannotation/
annotations/gold paths for downstream comparison.

Usage:
    python scripts/select_without_preann_subset.py --dummy
    python scripts/select_without_preann_subset.py --base /path/to/workspace
    python scripts/select_without_preann_subset.py --dummy --clean --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Matches PREANNOTATION_MODEL in src/digital_registrar_research/annotation/io_canonical.py.
DEFAULT_PREANN_MODEL = "gpt_oss_20b"
# Matches the seed used by scripts/gen_dummy_skeleton.py.
DEFAULT_SEED = 20251117


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=None,
                    help="Base dir containing data/<dataset>/... "
                         "(default: ./workspace, or ./dummy if --dummy)")
    ap.add_argument("--dummy", action="store_true",
                    help="Shortcut for --base ./dummy.")
    ap.add_argument("--dataset", default="cmuh",
                    help="Dataset name under <base>/data/ (default: cmuh)")
    ap.add_argument("--cancer-per-organ", type=int, default=15,
                    help="Cancer cases per n/ folder, excluding cancer_category=others (default: 15)")
    ap.add_argument("--noncancer-per-organ", type=int, default=5,
                    help="Non-cancer cases per n/ folder (default: 5)")
    ap.add_argument("--preann-model", default=DEFAULT_PREANN_MODEL,
                    help=f"Pre-annotation model dir name (default: {DEFAULT_PREANN_MODEL})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"RNG seed for deterministic selection (default: {DEFAULT_SEED})")
    ap.add_argument("--clean", action="store_true",
                    help="rmtree the destination reports_without_preann/ before copying.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan without copying anything.")
    args = ap.parse_args()

    if args.dummy:
        if args.base is not None and args.base != REPO_ROOT / "dummy":
            ap.error("--dummy and --base are mutually exclusive")
        args.base = REPO_ROOT / "dummy"
    elif args.base is None:
        args.base = REPO_ROOT / "workspace"

    if args.cancer_per_organ < 0 or args.noncancer_per_organ < 0:
        ap.error("--cancer-per-organ and --noncancer-per-organ must be >= 0")

    return args


def classify(preann: dict) -> str:
    """Return 'cancer', 'noncancer', or 'skip' for a preannotation payload."""
    flag = preann.get("cancer_excision_report")
    if flag is True:
        if preann.get("cancer_category") == "others":
            return "skip"
        return "cancer"
    if flag is False:
        return "noncancer"
    return "skip"


def gather_buckets(preann_root: Path) -> dict[str, dict[str, list[str]]]:
    """Walk preannotation/<model>/<n>/*.json, return {n: {bucket: [case_id, ...]}}."""
    buckets: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"cancer": [], "noncancer": []})
    for path in preann_root.rglob("*.json"):
        n = path.parent.name
        case_id = path.stem
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"warning: failed to parse {path}: {e}", file=sys.stderr)
            continue
        bucket = classify(data)
        if bucket == "skip":
            continue
        buckets[n][bucket].append(case_id)
    return buckets


def select_subset(buckets: dict[str, dict[str, list[str]]],
                  cancer_quota: int, noncancer_quota: int,
                  seed: int) -> list[tuple[str, str, str]]:
    """Return a deterministic [(n, bucket, case_id), ...] selection."""
    rng = random.Random(seed)
    out: list[tuple[str, str, str]] = []
    for n in sorted(buckets, key=_organ_sort_key):
        for bucket, quota in (("cancer", cancer_quota), ("noncancer", noncancer_quota)):
            ids = sorted(buckets[n][bucket])
            rng.shuffle(ids)
            chosen = ids[:quota]
            if len(chosen) < quota:
                print(f"warning: n={n} {bucket}: only {len(chosen)} available, "
                      f"requested {quota}", file=sys.stderr)
            for case_id in chosen:
                out.append((n, bucket, case_id))
    return out


def _organ_sort_key(n: str) -> tuple:
    try:
        return (0, int(n))
    except ValueError:
        return (1, n)


def copy_reports(selection: list[tuple[str, str, str]],
                 reports_root: Path, dst_root: Path,
                 dry_run: bool) -> tuple[int, int, dict[str, dict[str, int]]]:
    """Copy each selected report; return (copied, missing, per_n_per_bucket counts)."""
    copied = 0
    missing = 0
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"cancer": 0, "noncancer": 0})
    for n, bucket, case_id in selection:
        src = reports_root / n / f"{case_id}.txt"
        dst = dst_root / n / f"{case_id}.txt"
        if not src.is_file():
            print(f"warning: missing report {src}", file=sys.stderr)
            missing += 1
            continue
        if dry_run:
            print(f"[dry-run] {src} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        counts[n][bucket] += 1
        copied += 1
    return copied, missing, counts


def main() -> None:
    args = parse_args()

    ds_root = args.base / "data" / args.dataset
    preann_root = ds_root / "preannotation" / args.preann_model
    reports_root = ds_root / "reports"
    dst_root = ds_root / "reports_without_preann"

    if not preann_root.is_dir():
        sys.exit(f"error: preannotation root not found: {preann_root}")
    if not reports_root.is_dir():
        sys.exit(f"error: reports root not found: {reports_root}")

    buckets = gather_buckets(preann_root)
    if not buckets:
        sys.exit(f"error: no preannotation files found under {preann_root}")

    selection = select_subset(buckets, args.cancer_per_organ,
                              args.noncancer_per_organ, args.seed)

    if args.clean and dst_root.exists():
        if args.dry_run:
            print(f"[dry-run] would rmtree {dst_root}")
        else:
            shutil.rmtree(dst_root)

    copied, missing, counts = copy_reports(selection, reports_root, dst_root,
                                           args.dry_run)

    print()
    print(f"Base:           {args.base}")
    print(f"Dataset:        {args.dataset}")
    print(f"Preann model:   {args.preann_model}")
    print(f"Quota per n:    {args.cancer_per_organ} cancer + {args.noncancer_per_organ} noncancer "
          f"(seed={args.seed})")
    print(f"Source reports: {reports_root}")
    print(f"Destination:    {dst_root}{' [dry-run]' if args.dry_run else ''}")
    print()
    print(f"{'n':>4}  {'cancer':>7}  {'noncancer':>10}  {'total':>6}")
    grand_cancer = grand_noncancer = 0
    for n in sorted(counts, key=_organ_sort_key):
        c = counts[n]["cancer"]
        nc = counts[n]["noncancer"]
        grand_cancer += c
        grand_noncancer += nc
        print(f"{n:>4}  {c:>7}  {nc:>10}  {c + nc:>6}")
    print(f"{'all':>4}  {grand_cancer:>7}  {grand_noncancer:>10}  {grand_cancer + grand_noncancer:>6}")
    print()
    print(f"Copied:         {copied}")
    if missing:
        print(f"Missing report: {missing}")


if __name__ == "__main__":
    main()
