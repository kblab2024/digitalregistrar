#!/usr/bin/env python
"""Show divergence between the vendored research copy and the slim release.

The slim `digitalregistrar/` repo (sibling folder, untouched) is the
production-facing package. The vendored copy under
`src/digital_registrar_research/{pipeline.py, experiment.py, models/, util/}`
is the research tip-of-tree. This script prints a per-file diff so you can
see what's drifted before backporting to the slim release.

Usage:
    python scripts/repo/diff_against_slim.py
    python scripts/repo/diff_against_slim.py --slim-dir /path/to/digitalregistrar
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLIM = REPO_ROOT.parent / "digitalregistrar"
RESEARCH_ROOT = REPO_ROOT / "src" / "digital_registrar_research"

VENDORED_PATHS = [
    Path("pipeline.py"),
    Path("experiment.py"),
    *(Path("models") / p.name for p in (RESEARCH_ROOT / "models").glob("*.py")),
    *(Path("util") / p.name for p in (RESEARCH_ROOT / "util").glob("*.py")),
]


def _diff_one(slim_root: Path, rel: Path) -> str | None:
    """Return a unified diff string, or None if files are identical / missing."""
    slim_file = slim_root / rel
    research_file = RESEARCH_ROOT / rel
    if not slim_file.exists():
        return f"[NEW in research]   {rel}"
    if not research_file.exists():
        return f"[REMOVED in research] {rel}"
    a = slim_file.read_text(encoding="utf-8").splitlines(keepends=True)
    b = research_file.read_text(encoding="utf-8").splitlines(keepends=True)
    if a == b:
        return None
    diff = "".join(difflib.unified_diff(
        a, b,
        fromfile=f"slim/{rel}",
        tofile=f"research/{rel}",
        n=2,
    ))
    return diff or None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slim-dir", type=Path, default=DEFAULT_SLIM,
                    help=f"Path to the slim release (default: {DEFAULT_SLIM}).")
    ap.add_argument("--summary", action="store_true",
                    help="Only print which files differ, not the diffs themselves.")
    args = ap.parse_args()

    if not args.slim_dir.exists():
        print(f"[diff_against_slim] slim release not found at {args.slim_dir}", file=sys.stderr)
        sys.exit(2)

    diffs: list[tuple[Path, str]] = []
    for rel in VENDORED_PATHS:
        d = _diff_one(args.slim_dir, rel)
        if d:
            diffs.append((rel, d))

    if not diffs:
        print(f"[diff_against_slim] no divergence — research and slim ({args.slim_dir}) match.")
        return

    print(f"[diff_against_slim] {len(diffs)} file(s) differ vs {args.slim_dir}:")
    for rel, d in diffs:
        print(f"  - {rel}")
    if not args.summary:
        print()
        for _rel, d in diffs:
            print(d)


if __name__ == "__main__":
    main()
