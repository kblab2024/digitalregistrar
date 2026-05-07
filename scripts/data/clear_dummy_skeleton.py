"""Reset dummy/ to a clean known state — skeleton + canonical tracked files.

Inverse of `scripts/gen_dummy_skeleton.py`. Useful after a test run, or
any time you want a clean working tree without paying the cost of
re-running the generator.

Wipes:
    {out}/data/**          (everything under data/)
    {out}/results/**       (everything under results/)
Restores:
    {out}/data/.gitkeep
    {out}/data/cmuh/.gitkeep
    {out}/data/tcga/.gitkeep
    {out}/results/.gitkeep
Reverts to HEAD (only those that are tracked + currently modified):
    {out}/configs/**
    {out}/models/**
    {out}/README.md
    {out}/data/**            (any tracked .gitkeep that drifted)
    {out}/results/**

Usage:
    python scripts/clear_dummy_skeleton.py [--out dummy]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

# Mirrors DATASETS in scripts/gen_dummy_skeleton.py — the dataset folders
# we want to keep visible (with a .gitkeep) even when empty.
SKELETON_SUBDIRS = ["data", "data/cmuh", "data/tcga", "results"]

# Paths under {out} where the gen script writes tracked files. These get
# `git checkout HEAD --`'d so any post-generation drift is undone.
TRACKED_GEN_TARGETS = ["configs", "models", "README.md", "data", "results"]


def restore_tracked(out: Path) -> None:
    """git checkout HEAD -- the tracked files the generator overwrites.

    Silent no-op if {out} isn't inside a git repo. Restoration runs BEFORE
    the wipe so the wipe still produces a clean skeleton.
    """
    targets = [str(out / t) for t in TRACKED_GEN_TARGETS if (out / t).exists()]
    if not targets:
        return
    try:
        subprocess.run(
            ["git", "checkout", "HEAD", "--", *targets],
            check=False,
            capture_output=True,
            cwd=out.parent,
        )
    except FileNotFoundError:
        pass  # git not on PATH — skip silently


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("dummy"),
                    help="Dummy root to reset (default: ./dummy)")
    ap.add_argument("--no-git", action="store_true",
                    help="Skip the git-checkout step that restores tracked "
                         "configs/models/README to HEAD")
    args = ap.parse_args()

    out = args.out.resolve()
    if not out.exists():
        ap.error(f"{out} does not exist")

    if not args.no_git:
        restore_tracked(out)

    for sub in ("data", "results"):
        sub_root = out / sub
        if sub_root.exists():
            shutil.rmtree(sub_root)

    for d in SKELETON_SUBDIRS:
        path = out / d
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch()

    print(f"Reset {out}/data and {out}/results to skeleton:")
    for d in SKELETON_SUBDIRS:
        print(f"  {out}/{d}/.gitkeep")
    if not args.no_git:
        print(f"Restored tracked files under {out}/ to HEAD (configs, models, README).")


if __name__ == "__main__":
    main()
