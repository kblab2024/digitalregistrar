#!/usr/bin/env python3
"""Smoke-test the multirun wrapper end-to-end against an isolated sandbox.

Validates that ``scripts/run_dspy_ollama_multirun.py`` actually:
  * fires N iterations,
  * lands each one in a distinct ``runNN/`` slot,
  * threads a different seed into every iteration,
  * accumulates one entry per iteration into ``_manifest.yaml`` with
    ``valid: true`` (no pipeline errors).

The wrapper is invoked in-process, so a green smoke proves the loop,
the seed rotation, the slot rotation, and the LM call path together.
The real Ollama daemon is hit — same trust model as
``scripts/run_inference_smoke.py`` and ``scripts/run_dspy_ollama_smoke.py``.

The sandbox is a fresh ``tempfile.mkdtemp`` directory with one report
per organ symlinked from ``dummy/data/<dataset>/reports/``, so we can
assert ``len(runs) == n`` without colliding with any in-progress sweeps
the user already has under ``dummy/results/predictions/...``.

Usage
-----
    python scripts/run_dspy_ollama_multirun_smoke.py --model gptoss
    python scripts/run_dspy_ollama_multirun_smoke.py --model gptoss --n 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pipeline"))  # for sibling imports

from run_dspy_ollama_single import (  # noqa: E402
    DATASETS,
    UNIFIED_MODELS,
    model_slug,
)

import run_dspy_ollama_multirun  # noqa: E402


def build_sandbox(src_root: Path, dataset: str,
                  reports_per_organ: int) -> Path:
    """Create an isolated experiment_root that mirrors the dummy layout
    but exposes only ``reports_per_organ`` reports per organ. Returns the
    sandbox path."""
    src_reports = src_root / "data" / dataset / "reports"
    if not src_reports.is_dir():
        raise FileNotFoundError(f"source reports not found: {src_reports}")

    sandbox = Path(tempfile.mkdtemp(prefix="multirun_smoke_"))
    dst_reports = sandbox / "data" / dataset / "reports"

    organ_dirs = sorted(d for d in src_reports.iterdir()
                        if d.is_dir() and not d.name.startswith("_"))
    if not organ_dirs:
        shutil.rmtree(sandbox, ignore_errors=True)
        raise FileNotFoundError(f"no organ dirs under {src_reports}")

    picked_any = False
    for organ_dir in organ_dirs:
        reports = sorted(organ_dir.glob("*.txt"))[:reports_per_organ]
        if not reports:
            continue
        out_organ = dst_reports / organ_dir.name
        out_organ.mkdir(parents=True, exist_ok=True)
        for rp in reports:
            link = out_organ / rp.name
            try:
                os.symlink(rp.resolve(), link)
            except OSError:
                # Fallback to copy if the platform/FS forbids symlinks.
                shutil.copy2(rp, link)
            picked_any = True

    if not picked_any:
        shutil.rmtree(sandbox, ignore_errors=True)
        raise FileNotFoundError(f"no *.txt reports under {src_reports}")
    return sandbox


def assert_smoke(sandbox: Path, dataset: str, model_alias: str,
                 expected_runs: int) -> None:
    """Walk the sandbox prediction tree and fail-loud on any mismatch."""
    import yaml  # lazy: matches the rest of the toolchain

    slug = model_slug(model_alias)
    model_dir = (sandbox / "results" / "predictions"
                 / dataset / "llm" / slug)
    if not model_dir.is_dir():
        raise AssertionError(f"model dir missing: {model_dir}")

    run_dirs = sorted(d for d in model_dir.iterdir()
                      if d.is_dir() and d.name.startswith("run"))
    if len(run_dirs) != expected_runs:
        raise AssertionError(
            f"expected {expected_runs} run dirs under {model_dir}, "
            f"got {len(run_dirs)}: {[d.name for d in run_dirs]}"
        )

    seeds_seen: list[int] = []
    for rd in run_dirs:
        summary_path = rd / "_summary.json"
        meta_path = rd / "_run_meta.json"
        if not summary_path.is_file():
            raise AssertionError(f"missing _summary.json in {rd}")
        if not meta_path.is_file():
            raise AssertionError(f"missing _run_meta.json in {rd}")
        summary = json.loads(summary_path.read_text())
        meta = json.loads(meta_path.read_text())
        if summary.get("n_pipeline_error", 0) != 0:
            raise AssertionError(
                f"{rd.name}: n_pipeline_error="
                f"{summary['n_pipeline_error']} (smoke must be clean)"
            )
        if summary.get("n_cases", 0) <= 0:
            raise AssertionError(f"{rd.name}: n_cases=0 — nothing ran")
        seed = meta.get("dspy_lm_kwargs", {}).get("seed")
        if seed is None:
            raise AssertionError(f"{rd.name}: seed missing in _run_meta.json")
        seeds_seen.append(int(seed))

    if len(set(seeds_seen)) != len(seeds_seen):
        raise AssertionError(
            f"seeds repeated across runs: {seeds_seen} "
            f"(wrapper must rotate seeds per iteration)"
        )

    manifest_path = model_dir / "_manifest.yaml"
    if not manifest_path.is_file():
        raise AssertionError(f"missing manifest: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    runs = list(manifest.get("runs") or [])
    if len(runs) != expected_runs:
        raise AssertionError(
            f"manifest runs={len(runs)}, expected {expected_runs}: {runs}"
        )
    manifest_seeds = [int(r.get("seed")) for r in runs
                      if r.get("seed") is not None]
    if sorted(manifest_seeds) != sorted(seeds_seen):
        raise AssertionError(
            f"manifest seeds {manifest_seeds} disagree with "
            f"_run_meta.json seeds {seeds_seen}"
        )
    if not all(r.get("valid") for r in runs):
        raise AssertionError(
            f"some manifest runs are not valid: {runs}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, choices=UNIFIED_MODELS,
                    help="Model alias forwarded to the wrapper.")
    ap.add_argument("--dataset", default="tcga", choices=DATASETS,
                    help="Source dataset to sample from (default: tcga).")
    ap.add_argument("--n", type=int, default=2,
                    help="Number of multirun iterations to fire "
                         "(default: 2; keep small — this hits Ollama).")
    ap.add_argument("--reports-per-organ", type=int, default=1,
                    help="Reports per organ to expose in the sandbox "
                         "(default: 1 — minimal smoke).")
    ap.add_argument("--master-seed", type=int, default=None,
                    help="Optional master seed forwarded to the wrapper "
                         "to make the seed sequence reproducible.")
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Restrict the wrapper to these numeric organ dirs.")
    ap.add_argument("--keep-sandbox", action="store_true",
                    help="Do not delete the sandbox dir on success "
                         "(handy for debugging).")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src_root = REPO_ROOT / "dummy"
    if not src_root.is_dir():
        print(f"error: dummy tree not found at {src_root}", file=sys.stderr)
        return 2

    try:
        sandbox = build_sandbox(src_root, args.dataset, args.reports_per_organ)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"smoke sandbox: {sandbox}")

    wrapper_argv = [
        "--model", args.model,
        "--folder", str(sandbox),
        "--dataset", args.dataset,
        "--n", str(args.n),
    ]
    if args.master_seed is not None:
        wrapper_argv += ["--master-seed", str(args.master_seed)]
    if args.organs:
        wrapper_argv += ["--organs", *args.organs]
    if args.verbose:
        wrapper_argv += ["-v"]

    rc = run_dspy_ollama_multirun.main(wrapper_argv)
    if rc != 0:
        print(f"SMOKE FAIL: wrapper returned rc={rc}", file=sys.stderr)
        print(f"  sandbox left at: {sandbox}", file=sys.stderr)
        return 1

    try:
        assert_smoke(sandbox, args.dataset, args.model, args.n)
    except AssertionError as exc:
        print(f"SMOKE FAIL: {exc}", file=sys.stderr)
        print(f"  sandbox left at: {sandbox}", file=sys.stderr)
        return 1

    print(f"SMOKE OK: {args.n} runs, distinct seeds, manifest consistent")
    if args.keep_sandbox:
        print(f"  sandbox kept at: {sandbox}")
    else:
        shutil.rmtree(sandbox, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
