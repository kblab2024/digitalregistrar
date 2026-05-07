#!/usr/bin/env python3
"""Full ablation grid driver — canonical layout.

Reads a YAML grid spec listing the (cell, model, args) tuples to run,
executes each in sequence, and aggregates at the end. Each run goes
through the canonical contract — ``--folder`` + ``--dataset`` + ``--model``
(unified alias) — so output lands at::

    {folder}/results/ablations/{dataset}/{cell_id}/{model_slug}/{run_id}/{organ_n}/{case_id}.json

YAML format
-----------
    name: grid_1
    description: "Minimum-viable lesion study"
    folder: dummy            # or workspace, or absolute path
    dataset: tcga
    runs:
      - cell: dspy_monolithic
        model: gptoss        # unified alias from models.common.UNIFIED_MODELS
        args: {}             # per-cell extra kwargs
      - cell: raw_json
        model: gptoss
        args: {}
      - cell: dspy_monolithic
        model: gptoss
        args:
          skip_jsonize: true

Usage
-----
    python scripts/ablations/run_grid.py --config configs/ablations/grid_1.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # for _config_loader

import yaml  # noqa: E402

from _config_loader import resolve_folder  # noqa: E402

CELL_DISPATCH = {
    # Cells A/B/C (originals).
    "dspy_modular":         "digital_registrar_research.ablations.runners.reuse_baseline",
    "dspy_monolithic":      "digital_registrar_research.ablations.runners.dspy_monolithic",
    "raw_json":             "digital_registrar_research.ablations.runners.raw_json",
    # Axis 1.
    "no_router":            "digital_registrar_research.ablations.runners.no_router",
    "per_section":          "digital_registrar_research.ablations.runners.per_section",
    # Axis 2.
    "str_outputs":          "digital_registrar_research.ablations.runners.str_outputs",
    "constrained_decoding": "digital_registrar_research.ablations.runners.constrained_decoding",
    "free_text_regex":      "digital_registrar_research.ablations.runners.free_text_regex",
    # Axis 3.
    "fewshot_demos":        "digital_registrar_research.ablations.runners.fewshot_demos",
    "chain_of_thought":     "digital_registrar_research.ablations.runners.chain_of_thought",
    "compiled_dspy":        "digital_registrar_research.ablations.runners.compiled_dspy",
    "minimal_prompt":       "digital_registrar_research.ablations.runners.minimal_prompt",
    # Axis 6.
    "union_schema":         "digital_registrar_research.ablations.runners.union_schema",
    "flat_schema":          "digital_registrar_research.ablations.runners.flat_schema",
}


def _git_sha(repo_root: Path) -> str | None:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return None
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            ref_path = repo_root / ".git" / ref.split(maxsplit=1)[1]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
        return ref
    except Exception:
        return None


def _build_args_for_cell(cell: str, model: str, experiment_root: Path,
                         dataset: str, extra: dict) -> argparse.Namespace:
    """Map the YAML ``args`` block onto the canonical Namespace shape
    accepted by every cell's ``run(args)`` entry point."""
    common = dict(
        experiment_root=experiment_root,
        dataset=dataset,
        model=model,
        run=extra.get("run"),
        organs=extra.get("organs"),
        limit=extra.get("limit"),
        overwrite=extra.get("overwrite", False),
        tolerate_errors=extra.get("tolerate_errors", False),
        verbose=extra.get("verbose", False),
    )
    if cell == "dspy_modular":
        return argparse.Namespace(
            **common, source_run=extra.get("source_run"))
    if cell == "dspy_monolithic":
        return argparse.Namespace(
            **common, skip_jsonize=extra.get("skip_jsonize", False))
    if cell == "raw_json":
        return argparse.Namespace(**common, api_base=extra.get("api_base"))
    if cell == "no_router":
        return argparse.Namespace(
            **common, include_jsonize=extra.get("include_jsonize", False))
    if cell == "per_section":
        return argparse.Namespace(**common)
    if cell == "str_outputs":
        return argparse.Namespace(
            **common, skip_jsonize=extra.get("skip_jsonize", False))
    if cell == "constrained_decoding":
        return argparse.Namespace(
            **common,
            backend=extra.get("backend", "vllm"),
            api_base=extra.get("api_base"),
            use_union_schema=extra.get("use_union_schema", False))
    if cell == "free_text_regex":
        return argparse.Namespace(**common, api_base=extra.get("api_base"))
    if cell == "fewshot_demos":
        return argparse.Namespace(
            **common,
            n_shots=extra.get("n_shots", 3),
            skip_jsonize=extra.get("skip_jsonize", False))
    if cell == "chain_of_thought":
        return argparse.Namespace(
            **common,
            skip_jsonize=extra.get("skip_jsonize", False),
            cot_everywhere=extra.get("cot_everywhere", False))
    if cell == "compiled_dspy":
        return argparse.Namespace(**common, compiled=Path(extra["compiled"]))
    if cell == "minimal_prompt":
        return argparse.Namespace(**common, api_base=extra.get("api_base"))
    if cell == "union_schema":
        return argparse.Namespace(**common, api_base=extra.get("api_base"))
    if cell == "flat_schema":
        return argparse.Namespace(**common, api_base=extra.get("api_base"))
    raise ValueError(f"Unknown cell {cell!r}")


def _run_one(cell: str, model: str, experiment_root: Path, dataset: str,
             extra: dict) -> dict:
    """Run a single (cell, model) tuple under the canonical layout."""
    if cell not in CELL_DISPATCH:
        raise ValueError(f"Unknown cell {cell!r} — registered cells: "
                         f"{sorted(CELL_DISPATCH)}")

    module = importlib.import_module(CELL_DISPATCH[cell])
    if not hasattr(module, "run"):
        raise RuntimeError(
            f"Cell module {module.__name__} has no run(args) function")

    ns = _build_args_for_cell(cell, model, experiment_root, dataset, extra)
    t0 = time.perf_counter()
    err: str | None = None
    rc: int = 0
    try:
        rc = module.run(ns) or 0
    except Exception as e:
        err = repr(e)
        raise
    finally:
        elapsed = time.perf_counter() - t0
        manifest = {
            "cell": cell,
            "model": model,
            "experiment_root": str(experiment_root.resolve()),
            "dataset": dataset,
            "args": extra,
            "elapsed_s": elapsed,
            "rc": rc,
            "git_sha": _git_sha(REPO_ROOT),
            "started_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "error": err,
        }
    return manifest


def _write_failures(experiment_root: Path, dataset: str,
                    failures: list[dict]) -> None:
    """Persist the per-cell failure list under the ablations root so
    the user can re-run only the failed cells (no need to grep logs)."""
    ablations_root = experiment_root / "results" / "ablations" / dataset
    ablations_root.mkdir(parents=True, exist_ok=True)
    out_path = ablations_root / "grid_failures.json"
    out_path.write_text(
        json.dumps({"failures": failures,
                    "completed_utc": dt.datetime.utcnow().isoformat(
                        timespec="seconds")},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")


def _preflight(spec: dict, experiment_root: Path, dataset: str,
               unified_models: tuple[str, ...]) -> list[str]:
    """Validate the YAML before running ANY cell. Returns a list of
    error messages — empty means OK. Catches typos / missing artifacts
    early instead of failing mid-grid."""
    errors: list[str] = []
    runs = spec.get("runs") or []
    cell_typos: set[str] = set()
    model_typos: set[str] = set()
    for run in runs:
        cell = run.get("cell")
        model = run.get("model")
        extra = run.get("args") or {}
        if cell not in CELL_DISPATCH:
            cell_typos.add(str(cell))
        if model not in unified_models:
            model_typos.add(str(model))
        if cell == "compiled_dspy":
            artifact = extra.get("compiled")
            if not artifact:
                errors.append("compiled_dspy run missing required "
                              "'compiled' artifact path")
            elif not Path(artifact).exists():
                errors.append(
                    f"compiled_dspy run: artifact {artifact!r} does not exist")
        if cell == "fewshot_demos":
            demos_yaml = (REPO_ROOT / "configs" / "ablations"
                          / "fewshot_demos.yaml")
            if not demos_yaml.exists():
                errors.append(
                    f"fewshot_demos run: {demos_yaml} not found — run "
                    "scripts/ablations/build_fewshot_demos.py first")
    if cell_typos:
        errors.append(
            f"Unknown cell-id(s): {sorted(cell_typos)}. "
            f"Known: {sorted(CELL_DISPATCH)}")
    if model_typos:
        errors.append(
            f"Unknown model alias(es): {sorted(model_typos)}. "
            f"Known: {sorted(unified_models)}")
    reports_root = experiment_root / "data" / dataset / "reports"
    if not reports_root.is_dir():
        errors.append(f"Reports root not found: {reports_root}")
    gold_root = (experiment_root / "data" / dataset / "annotations" / "gold")
    if not gold_root.is_dir():
        errors.append(f"Gold annotations root not found: {gold_root}")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True,
                    help="YAML grid spec — see configs/ablations/grid_1.yaml")
    ap.add_argument("--skip-aggregate", action="store_true",
                    help="run the cells but don't run run_ablations afterwards")
    ap.add_argument("--folder", dest="experiment_root_override",
                    type=resolve_folder, default=None,
                    help="override the YAML's `folder:` field")
    ap.add_argument("--dataset", default=None,
                    help="override the YAML's `dataset:` field")
    ap.add_argument("--continue-on-cell-error", action="store_true",
                    help="if a cell run raises, log the failure to "
                         "grid_failures.json and continue with the next cell "
                         "instead of halting the whole grid.")
    args = ap.parse_args(argv)

    if not args.config.is_file():
        sys.exit(f"Grid config not found: {args.config}")

    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not spec.get("runs"):
        sys.exit(f"No runs listed in {args.config}")

    experiment_root = (args.experiment_root_override
                       or resolve_folder(spec["folder"]))
    dataset = args.dataset or spec["dataset"]

    # Pre-flight: validate cell-ids, model aliases, required artifacts,
    # and data layout BEFORE the first cell runs. Catches typos that
    # would otherwise fail mid-grid hours later.
    from digital_registrar_research.ablations.runners._base import (  # noqa: E402
        UNIFIED_MODELS,
    )
    preflight_errors = _preflight(spec, experiment_root, dataset, UNIFIED_MODELS)
    if preflight_errors:
        print("[grid] PRE-FLIGHT FAILED:")
        for err in preflight_errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"[grid] config={args.config}")
    print(f"[grid] folder={experiment_root}  dataset={dataset}")
    print(f"[grid] runs={len(spec['runs'])}")

    manifests: list[dict] = []
    failures: list[dict] = []
    cells_seen: set[str] = set()
    models_seen: set[str] = set()

    t0 = time.perf_counter()
    for i, run in enumerate(spec["runs"], 1):
        cell = run["cell"]
        model = run["model"]
        extra = run.get("args") or {}
        print(f"\n[{i}/{len(spec['runs'])}] cell={cell} model={model}")
        try:
            manifest = _run_one(cell, model, experiment_root, dataset, extra)
            manifests.append(manifest)
            cells_seen.add(cell)
            models_seen.add(model)
        except Exception as exc:
            failure = {"cell": cell, "model": model, "args": extra,
                       "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"[grid] CELL FAILED: {failure['error']}")
            if not args.continue_on_cell_error:
                # Persist failures so far before halting.
                _write_failures(experiment_root, dataset, failures)
                raise

    print(f"\n[grid] all runs complete ({time.perf_counter() - t0:.1f}s)")
    if failures:
        _write_failures(experiment_root, dataset, failures)
        print(f"[grid] {len(failures)} cell(s) failed — see grid_failures.json")

    # Top-level grid manifest under the ablations root.
    ablations_root = (experiment_root / "results" / "ablations" / dataset)
    ablations_root.mkdir(parents=True, exist_ok=True)
    grid_meta = {
        "config_path": str(args.config),
        "spec": spec,
        "manifests": manifests,
        "git_sha": _git_sha(REPO_ROOT),
        "completed_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
    }
    (ablations_root / "_grid_meta.json").write_text(
        json.dumps(grid_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.skip_aggregate:
        print("[grid] --skip-aggregate set; not running run_ablations")
        return 0

    print("\n[grid] running aggregator…")
    from digital_registrar_research.ablations.eval.run_ablations import (
        main as eval_main,
    )
    eval_argv = [
        "--folder", str(experiment_root),
        "--dataset", dataset,
        "--cells", *sorted(cells_seen),
    ]
    eval_main(eval_argv)
    print(f"\n[grid] OK — see {ablations_root}/ablation_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
