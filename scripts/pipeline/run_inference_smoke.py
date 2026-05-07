#!/usr/bin/env python3
"""Pipeline smoke test on the dummy dataset — Ollama or OpenAI backend.

Purpose
-------
Pre-flight the *real* ``CancerPipeline`` (the DSPy is_cancer → jsonize
→ organ-specific graph) against the dummy layout before touching the
workstation and before touching CMUH. Swaps only the ``dspy.LM`` used
by the pipeline; everything downstream — signatures, organ dispatch,
per-case JSON writes, manifest book-keeping — is the production code
path unchanged.

Both backends talk through ``dspy.LM`` (LiteLLM under the hood):

  * ``--backend ollama`` → ``ollama_chat/<model>`` (e.g. ``gpt-oss:20b``)
    via a local Ollama daemon at ``--api-base`` (default
    ``http://localhost:11434``).
  * ``--backend openai`` → ``openai/<model>`` (e.g. ``gpt-4o-mini``)
    via ``https://api.openai.com/v1``; key read from a text file
    (default ``secrets/openai_api_key.txt``) or ``$OPENAI_API_KEY``.

Unlike ``run_dspy_ollama_smoke.py``, this script isn't pinned to the
static ``models.common.model_list`` — you pass any model name the
chosen backend can resolve. This is what lets the same script sanity-
check an OpenAI model *and* the Ollama models you'll run on the
workstation.

Fail-loud: any pipeline error or a mismatch between sampled-vs-ok
exits non-zero. A green smoke run is the go/no-go for the real run.

Usage
-----
    # Ollama (local daemon at :11434)
    python scripts/run_inference_smoke.py \\
        --backend ollama --model gpt-oss:20b

    # OpenAI (reads secrets/openai_api_key.txt by default)
    python scripts/run_inference_smoke.py \\
        --backend openai --model gpt-4o-mini

    # Different dummy dataset / sample size / organ filter
    python scripts/run_inference_smoke.py --backend ollama \\
        --model qwen3:30b --dataset tcga --n 6 --organs 1 2

Output
------
    {experiment_root}/results/predictions/{dataset}/llm/{slug}/_smoke_{ts}/
        {organ_n}/<case_id>.json    prediction (real pipeline output)
        _log.jsonl                   one row per case
        _summary.json                aggregate
        _run_meta.json               backend/model/env provenance
        _run.log                     pipeline log

The leading underscore on ``_smoke_{ts}`` keeps it out of any downstream
eval glob that filters ``not name.startswith('_')`` — smoke runs never
pollute real sweeps.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import platform
import random
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# Make the in-tree package importable without `pip install -e .`.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pipeline"))  # for sibling imports

import dspy  # noqa: E402

# Reuse production helpers so smoke output exactly matches real-run output.
from run_dspy_ollama_single import (  # noqa: E402
    PIPELINE_LOGGER_NAME,
    _atomic_write_json,
    _git_sha,
    _utc_now_iso,
    discover_organs,
    process_case,
)

from digital_registrar_research.pipeline import (  # noqa: E402
    run_cancer_pipeline,  # noqa: F401  # imported so DSPy signatures are registered
)
from digital_registrar_research.util.logger import setup_logger  # noqa: E402

DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "dummy"
DEFAULT_KEY_FILE = REPO_ROOT / "secrets" / "openai_api_key.txt"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
DATASETS = ("cmuh", "tcga")


# --- Backend / model plumbing -----------------------------------------------


def _slug(name: str) -> str:
    """Filesystem-safe slug: ``gpt-oss:20b`` → ``gpt_oss_20b``."""
    return re.sub(r"[-:./]+", "_", name).strip("_").lower()


def _load_api_key(backend: str, key_file: Path) -> str:
    """Resolve the API key for the chosen backend.

    Ollama ignores the key but DSPy/LiteLLM still requires *something*;
    OpenAI reads ``$OPENAI_API_KEY`` first, then the key file.
    """
    if backend == "ollama":
        return "ollama"  # placeholder — the daemon doesn't auth

    env = os.environ.get("OPENAI_API_KEY")
    if env:
        return env.strip()
    if not key_file.exists():
        raise FileNotFoundError(
            f"OpenAI key file not found at {key_file}. Create it with a single "
            f"line containing your key, or set OPENAI_API_KEY in the environment."
        )
    key = key_file.read_text(encoding="utf-8").strip()
    if (not key) or key.startswith("#") or "REPLACE_WITH" in key or "FILL_IN" in key:
        raise ValueError(
            f"API key file {key_file} still contains the placeholder. "
            f"Replace its contents with your real OpenAI key."
        )
    return key


def configure_dspy(backend: str, model: str, api_base: str, api_key: str, *,
                   temperature: float, max_tokens: int, seed: int) -> dspy.LM:
    """Build a ``dspy.LM`` for the requested backend and install it globally.

    The production pipeline uses ``dspy.configure(lm=...)``; we do the same
    here so ``run_cancer_pipeline`` picks up our LM without any code change.
    """
    if backend == "ollama":
        model_id = f"ollama_chat/{model}"
        lm = dspy.LM(
            model=model_id,
            api_base=api_base,
            api_key=api_key,
            model_type="chat",
            temperature=temperature,
            top_p=0.7,
            max_tokens=max_tokens,
            num_ctx=max_tokens,
            seed=seed,
        )
    elif backend == "openai":
        model_id = f"openai/{model}"
        lm = dspy.LM(
            model=model_id,
            api_base=api_base,
            api_key=api_key,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported backend: {backend!r}")

    dspy.configure(lm=lm)
    return lm


# --- Sampling ---------------------------------------------------------------


def sample_cases(organs: list[tuple[str, Path]], n: int, seed: int
                 ) -> list[tuple[str, Path]]:
    """Stratified sample across organ subdirs so small n hits every organ."""
    per_organ: list[list[tuple[str, Path]]] = []
    for organ_n, organ_dir in organs:
        reports = sorted(organ_dir.glob("*.txt"))
        if reports:
            per_organ.append([(organ_n, p) for p in reports])
    if not per_organ:
        return []
    rng = random.Random(seed)
    for lst in per_organ:
        rng.shuffle(lst)
    picked: list[tuple[str, Path]] = []
    while len(picked) < n and any(per_organ):
        for lst in per_organ:
            if lst:
                picked.append(lst.pop())
                if len(picked) >= n:
                    break
    return picked


# --- CLI --------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--backend", required=True, choices=("ollama", "openai"),
                    help="Which backend DSPy talks to.")
    ap.add_argument("--model", required=True,
                    help="Raw model name (e.g. 'gpt-oss:20b' for Ollama, "
                         "'gpt-4o-mini' for OpenAI).")
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT,
                    help="Root containing data/ and results/ "
                         "(default: ./dummy — this is the point).")
    ap.add_argument("--dataset", default="tcga", choices=DATASETS,
                    help="Dataset name under data/ (default: tcga).")
    ap.add_argument("--n", type=int, default=3,
                    help="Number of cases to sample, stratified by organ "
                         "(default: 3).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Sampling seed (default: 0).")
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Restrict sampling to these numeric organ dirs.")
    ap.add_argument("--api-base", default=None,
                    help="Override API base URL (default depends on --backend).")
    ap.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE,
                    help="Path to a text file containing the OpenAI API key. "
                         "Ignored for --backend ollama.")
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="Low by default so smoke results are reproducible.")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--model-seed", type=int, default=42,
                    help="Seed passed to the model when the backend honours it.")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    reports_root = args.experiment_root / "data" / args.dataset / "reports"
    if not reports_root.is_dir():
        print(f"error: reports not found at {reports_root}", file=sys.stderr)
        return 2

    organs = discover_organs(reports_root, args.organs)
    if not organs:
        suffix = f" matching {args.organs}" if args.organs else ""
        print(f"error: no organ dirs with *.txt found under {reports_root}"
              f"{suffix}", file=sys.stderr)
        return 2

    sampled = sample_cases(organs, args.n, args.seed)
    if len(sampled) < args.n:
        print(f"error: only {len(sampled)} report(s) available after "
              f"filtering; requested {args.n}", file=sys.stderr)
        return 2

    api_base = args.api_base or (
        DEFAULT_OLLAMA_BASE if args.backend == "ollama" else DEFAULT_OPENAI_BASE
    )
    try:
        api_key = _load_api_key(args.backend, args.api_key_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    slug = f"{args.backend}_{_slug(args.model)}"
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.experiment_root / "results" / "predictions"
               / args.dataset / "llm" / slug / f"_smoke_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name=PIPELINE_LOGGER_NAME,
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=str(out_dir / "_run.log"),
        json_format=False,
    )
    logger.info("backend: %s", args.backend)
    logger.info("model: %s (slug=%s)", args.model, slug)
    logger.info("api_base: %s", api_base)
    logger.info("experiment_root: %s", args.experiment_root)
    logger.info("dataset: %s", args.dataset)
    logger.info("sampled %d / %d reports (seed=%d)",
                len(sampled), sum(len(list(d.glob('*.txt'))) for _, d in organs),
                args.seed)
    for organ_n, rp in sampled:
        logger.info("  - %s (organ=%s)", rp.name, organ_n)

    try:
        lm = configure_dspy(
            args.backend, args.model, api_base, api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.model_seed,
        )
    except Exception as exc:
        print(f"error: failed to configure DSPy LM: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    run_name = out_dir.name  # e.g. "_smoke_20260423_101530"
    started_at = _utc_now_iso()
    t_run = time.perf_counter()

    summary: dict[str, Any] = {
        "run": run_name,
        "backend": args.backend,
        "model": args.model,
        "model_slug": slug,
        "dataset": args.dataset,
        "api_base": api_base,
        "seed": args.model_seed,
        "sampling_seed": args.seed,
        "temperature": args.temperature,
        "n_cases": 0, "n_ok": 0, "n_pipeline_error": 0, "n_cached": 0,
        "cancer_positive": 0,
        "sampled": [],
    }

    with (out_dir / "_log.jsonl").open("a", encoding="utf-8") as log_fh:
        for organ_n, report_path in sampled:
            row = process_case(
                report_path,
                organ=organ_n,
                run_name=run_name,
                seed=args.model_seed,
                out_dir=out_dir,
                log_fh=log_fh,
                logger=logger,
                overwrite=True,  # smoke runs should never silently skip
            )
            summary["n_cases"] += 1
            summary["sampled"].append({
                "case_id": row["case_id"], "organ": row["organ"],
                "status": row["status"], "latency_s": row["latency_s"],
                "is_cancer": row.get("is_cancer"),
                "cancer_category": row.get("cancer_category"),
            })
            if row["status"] == "ok":
                summary["n_ok"] += 1
                if row.get("is_cancer"):
                    summary["cancer_positive"] += 1
            elif row["status"] == "pipeline_error":
                summary["n_pipeline_error"] += 1

    summary["wall_time_s"] = round(time.perf_counter() - t_run, 2)
    finished_at = _utc_now_iso()

    _atomic_write_json(out_dir / "_summary.json", summary)
    _atomic_write_json(out_dir / "_run_meta.json", {
        "run": run_name,
        "backend": args.backend,
        "model": args.model,
        "model_slug": slug,
        "api_base": api_base,
        "dataset": args.dataset,
        "experiment_root": str(args.experiment_root.resolve()),
        "organs": [o[0] for o in organs],
        "started_at": started_at,
        "finished_at": finished_at,
        "dspy_lm_kwargs": {
            "temperature": getattr(lm, "temperature", None) or lm.kwargs.get("temperature"),
            "top_p": lm.kwargs.get("top_p"),
            "max_tokens": lm.kwargs.get("max_tokens"),
            "num_ctx": lm.kwargs.get("num_ctx"),
            "seed": lm.kwargs.get("seed"),
        },
        "git_sha": _git_sha(REPO_ROOT),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "argv": sys.argv,
    })

    line = (f"SMOKE OK={summary['n_ok']} ERR={summary['n_pipeline_error']} "
            f"N={summary['n_cases']} WALL={summary['wall_time_s']}s")
    logger.info(line)
    print(line)
    print(f"smoke dir: {out_dir}")

    return 0 if (summary["n_pipeline_error"] == 0
                 and summary["n_ok"] == args.n) else 1


if __name__ == "__main__":
    sys.exit(main())
