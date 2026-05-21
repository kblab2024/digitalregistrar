"""obfuscate-workspace CLI entry point.

  obfuscate-workspace [--src workspace/] [--out workspace_obfustrated/]
                      [--seed 42] [--shapes-only] [--force] [--no-validate]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .case_index import index_workspace
from .id_mapper import synth_tcga_case_id
from .manifest import Manifests
from .orchestrator import process_case, write_predictions_run_metadata
from .router import route
from .run_planner import plan_runs


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO_ROOT / "workspace"
DEFAULT_OUT = REPO_ROOT / "workspace_obfustrated"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="obfuscate-workspace",
        description="Produce a schema-conformant, content-randomized copy of the workspace.",
    )
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help="Source workspace root (default: %(default)s).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Destination root (default: %(default)s).")
    p.add_argument("--seed", type=int, default=42,
                   help="Master seed for deterministic generation (default: %(default)s).")
    p.add_argument("--shapes-only", action="store_true",
                   help="Degraded mode: empty annotations + lorem-ipsum reports. "
                        "Eval F1 will be 0 — failure mode is obvious.")
    p.add_argument("--force", action="store_true",
                   help="Allow overwriting an existing non-empty --out.")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip jsonschema validation of generated annotations "
                        "(faster but risk silently-invalid output).")
    p.add_argument("--version", action="version", version=f"obfuscator {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    src = args.src.resolve()
    out = args.out.resolve()

    if not src.is_dir():
        print(f"error: --src not a directory: {src}", file=sys.stderr)
        return 2
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"error: --out exists and is non-empty: {out}\n"
              f"       pass --force to overwrite, or pick a different path.",
              file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    print(f"[obfuscator] src = {src}")
    print(f"[obfuscator] out = {out}")
    print(f"[obfuscator] seed = {args.seed}")
    if args.shapes_only:
        print("[obfuscator] mode = shapes-only (eval F1 will be 0)")

    print("[obfuscator] indexing source workspace ...")
    idx = index_workspace(src)
    print(f"[obfuscator] indexed {len(idx.cases)} cases across "
          f"{len(idx.datasets)} datasets ({sorted(idx.datasets)})")

    plans = plan_runs(src)
    print(f"[obfuscator] {len(plans)} (model, run) prediction combos")

    manifests = Manifests(out_root=out)

    # ID map: case_id -> synthetic case_id (used by splits/manifest remapping).
    # Filenames themselves are preserved (case_id stays the same on disk);
    # this map is only for content inside splits.json/manifest.yaml.
    id_map: dict[str, str] = {}
    for (ds, organ_n, case_id) in idx.cases.keys():
        # We preserve case_id on disk, so splits.json should keep them too.
        # We only synthesize the *patient_filename* embedded inside reports.
        pass

    # 1. Per-case generation (the layered noise pipeline).
    print("[obfuscator] generating per-case artifacts ...")
    for i, case in enumerate(idx.cases.values(), start=1):
        if i % 200 == 0:
            print(f"[obfuscator]   {i}/{len(idx.cases)} cases ...")
        process_case(case, out, args.seed, plans, manifests,
                     shapes_only=args.shapes_only,
                     validate=not args.no_validate)

    # 2. Per-run metadata (one _run.log + _summary.json per run dir per dataset).
    write_predictions_run_metadata(out, plans, idx.datasets, manifests)

    # 3. Route non-case files (configs, splits, manifests, logs, blobs, etc.).
    print(f"[obfuscator] routing {len(idx.other_files)} non-case files ...")
    for path in idx.other_files:
        try:
            route(path, src, out, manifests, master_seed=args.seed, id_map=id_map)
        except Exception as exc:
            manifests.record_skipped(path, f"router exception: {type(exc).__name__}: {exc}")

    # 4. Write manifests.
    manifests.write(args.seed)
    print(f"[obfuscator] done. {len(manifests.outputs)} outputs, "
          f"{len(manifests.skipped)} skipped.")
    print(f"[obfuscator] see {out}/_obfuscator_outputs.json and "
          f"_obfuscator_skipped.json for the audit trail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
