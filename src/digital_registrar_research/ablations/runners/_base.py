"""Shared scaffolding for canonical-layout ablation runners.

Every ablation runner uses the same ``--folder/--dataset/--model``
contract as ``scripts/pipeline/run_dspy_ollama_single.py`` so that
input data, output predictions, and model aliases all live where the
rest of the toolkit (eval, benchmarks, dummy fixture, workspace tree)
expects them. Concretely:

    {root}/
      data/{dataset}/
        reports/{organ_n}/{case_id}.txt          (input)
        annotations/gold/{organ_n}/{case_id}.json (gold for grading)
      results/ablations/{dataset}/{cell_id}/{model_slug}/{run_id}/
        _summary.json
        _log.jsonl
        _run.log
        _run_meta.json
        {organ_n}/{case_id}.json                 (per-case prediction)

      results/ablations/{dataset}/{cell_id}/{model_slug}/_manifest.yaml

This module is the single source of truth for that layout. It exposes:

* :func:`add_canonical_args` — argparse extension every runner reuses.
* :func:`resolve_run_paths`  — turn a parsed Namespace into the run dir,
                              model slug, and organ list.
* :func:`iterate_cases`      — yield ``(organ_n, report_path)`` tuples.
* :class:`RunSummary`        — accumulator + atomic ``_summary.json`` writer.
* :func:`write_case_json`    — atomic per-case write.
* :func:`finalize_run`       — writes ``_summary.json``, ``_run_meta.json``,
                               and updates the cell-level ``_manifest.yaml``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Make scripts/ importable so we can reuse the pipeline runner's helpers
# without copy-paste. ``REPO_ROOT/scripts`` is the canonical home for
# ``_config_loader``, ``_run_id``, and the pipeline single-runner.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _config_loader import (  # noqa: E402
    load_model_config,
    resolve_folder,
    split_decoding_overrides,
)
from _run_id import format_run_id, machine_slug  # noqa: E402

from ...models.common import load_model, localaddr, model_list  # noqa: E402
from ...util.logger import setup_logger  # noqa: E402

# --- Constants shared with run_dspy_ollama_single.py -------------------------

DATASETS = ("cmuh", "tcga")
MAX_RUN_SLOTS = 10
UNIFIED_MODELS = (
    "gptoss", "gemma3", "gemma4", "qwen3_5", "medgemmalarge", "medgemmasmall",
    "gemma4e2b",
)


# --- Naming ------------------------------------------------------------------

def model_slug(model_alias: str) -> str:
    """Canonical folder name for a model alias.

    ``ollama_chat/gpt-oss:20b`` → ``gpt_oss_20b``;
    ``ollama_chat/qwen3:30b``  → ``qwen3_30b``.

    Same convention as ``run_dspy_ollama_single.model_slug`` so ablation
    output dirs line up with the pipeline output dirs name-for-name.
    """
    if model_alias not in model_list:
        raise ValueError(
            f"unknown model alias {model_alias!r}; valid: {sorted(model_list)}")
    full = model_list[model_alias]
    tail = full.split("/", 1)[-1]
    return re.sub(r"[-:./]", "_", tail)


def ollama_tag(model_alias: str) -> str:
    """Bare Ollama / OpenAI-API model tag for a unified alias.

    ``gptoss`` → ``gpt-oss:20b``. Used by the non-DSPy runners that talk
    to the Ollama OpenAI-compatible endpoint directly (raw_json,
    free_text_regex, minimal_prompt, union_schema, flat_schema,
    constrained_decoding with ``--backend openai``).
    """
    if model_alias not in model_list:
        raise ValueError(
            f"unknown model alias {model_alias!r}; valid: {sorted(model_list)}")
    full = model_list[model_alias]
    return full.split("/", 1)[-1]


def default_api_base() -> str:
    """OpenAI-compatible base URL for the local Ollama daemon."""
    return f"{localaddr.rstrip('/')}/v1"


# --- Argparse contract -------------------------------------------------------

def add_canonical_args(ap: argparse.ArgumentParser, *,
                       require_model: bool = True) -> None:
    """Inject the standard ``--folder/--dataset/--model/--run/...`` flags
    onto an existing ArgumentParser. Keeps the surface identical across
    all 14 ablation runners and the pipeline runner.

    ``--obfustrated`` is an additive shortcut: when set without an explicit
    ``--folder``, defaults to ``workspace_obfustrated/`` (the obfuscator-
    produced synthetic copy). Existing ``--folder dummy`` / ``--folder workspace``
    invocations are unaffected.
    """
    ap.add_argument("--folder", dest="experiment_root", required=False, default=None,
                    type=resolve_folder,
                    help="Experiment root containing data/ and results/. "
                         "Shorthand 'dummy', 'workspace', or 'obfustrated' "
                         "resolves against the repo root. Required unless "
                         "--obfustrated is set.")
    ap.add_argument("--obfustrated", action="store_true",
                    help="Shortcut for --folder obfustrated (= workspace_obfustrated/). "
                         "Use when debugging without PHI; explicit --folder always wins.")
    ap.add_argument("--dataset", required=True, choices=DATASETS,
                    help="Dataset name under data/ (cmuh or tcga).")
    if require_model:
        ap.add_argument("--model", required=True, choices=UNIFIED_MODELS,
                        help="Model alias: " + ", ".join(UNIFIED_MODELS) +
                             ". Each alias auto-loads "
                             "configs/dspy_ollama_{alias}.yaml for "
                             "decoding overrides.")
    ap.add_argument("--run", default=None,
                    help="Run slot name, e.g. run01..run10 "
                         "(default: next free slot under the cell/model dir).")
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Only run these numeric organ directories (default: every organ).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap cases per organ (debugging).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess cases even if a valid output exists.")
    ap.add_argument("--tolerate-errors", action="store_true",
                    help="Always exit 0 if the script completes.")
    ap.add_argument("--trace-dspy", action="store_true",
                    help="Dump every DSPy LM call (rendered prompt + raw "
                         "response) into _dspy_trace.jsonl in the run dir. "
                         "Useful for diagnosing silent skips in the "
                         "is_cancer -> organ -> predictor chain.")
    ap.add_argument("-v", "--verbose", action="store_true")


# --- Path resolution ---------------------------------------------------------

@dataclass
class AblationPaths:
    """File-path producer for one ``(experiment_root, dataset, cell_id, model_slug)``.

    Produced by :func:`resolve_run_paths` from a parsed argparse Namespace.
    Mirrors :class:`scripts.eval._common.paths.Paths` semantically.
    """
    experiment_root: Path
    dataset: str
    cell_id: str
    model_slug: str

    @property
    def reports_dir(self) -> Path:
        return self.experiment_root / "data" / self.dataset / "reports"

    @property
    def gold_dir(self) -> Path:
        return self.experiment_root / "data" / self.dataset / "annotations" / "gold"

    @property
    def cell_dir(self) -> Path:
        return (self.experiment_root / "results" / "ablations" / self.dataset
                / self.cell_id / self.model_slug)

    def run_dir(self, run_id: str) -> Path:
        return self.cell_dir / run_id

    def case_path(self, run_id: str, organ_n: str, case_id: str) -> Path:
        return self.run_dir(run_id) / organ_n / f"{case_id}.json"

    def gold_path(self, organ_n: str, case_id: str) -> Path:
        return self.gold_dir / organ_n / f"{case_id}.json"


def discover_organs(reports_root: Path,
                    organ_filter: list[str] | None,
                    ) -> list[tuple[str, Path]]:
    """Return sorted ``(organ_n, organ_dir)`` for every organ subdir of
    ``reports_root`` that contains at least one ``*.txt``."""
    if not reports_root.is_dir():
        return []
    picked: list[tuple[str, Path]] = []
    for child in sorted(reports_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if organ_filter and child.name not in organ_filter:
            continue
        if not any(child.glob("*.txt")):
            continue
        picked.append((child.name, child))
    return picked


def discover_cases(organ_dir: Path, limit: int | None) -> list[Path]:
    cases = sorted(organ_dir.glob("*.txt"), key=lambda p: p.name)
    if limit is not None:
        cases = cases[:limit]
    return cases


def pick_next_run(cell_dir: Path) -> str:
    """Return the first run-id in ``run01..run{MAX_RUN_SLOTS}`` whose
    directory has no ``_summary.json`` yet (matches
    ``run_dspy_ollama_single.pick_next_run``).
    """
    for k in range(1, MAX_RUN_SLOTS + 1):
        name = format_run_id(k, padded=True)
        if not (cell_dir / name / "_summary.json").exists():
            return name
    slug = machine_slug()
    suffix_msg = f" (machine={slug!r})" if slug else ""
    raise RuntimeError(
        f"all {MAX_RUN_SLOTS} run slots are populated under {cell_dir}"
        f"{suffix_msg}; pass --run runNN explicitly.")


def apply_obfustrated_default(args: argparse.Namespace) -> None:
    """If ``--obfustrated`` was set without an explicit ``--folder``, resolve
    ``args.experiment_root`` to ``workspace_obfustrated/``.

    Idempotent. Call right after ``parse_args()``. Already invoked from
    inside ``resolve_run_paths``; runners that don't go through
    ``resolve_run_paths`` should call this explicitly.
    """
    if getattr(args, "experiment_root", None) is not None:
        return
    if getattr(args, "obfustrated", False):
        args.experiment_root = resolve_folder("obfustrated")
    else:
        raise SystemExit(
            "--folder is required (use 'dummy', 'workspace', "
            "'workspace_obfustrated', or an absolute path), "
            "or pass --obfustrated for the synthetic workspace.")


def resolve_run_paths(args: argparse.Namespace, cell_id: str,
                      ) -> tuple[AblationPaths, list[tuple[str, Path]], str]:
    """Resolve ``(paths, organs, run_name)`` from CLI args.

    Validates the model alias, builds the cell/model directory, picks
    the next free run slot (or honours ``--run``), and returns the
    discovered organ list.
    """
    apply_obfustrated_default(args)
    if not getattr(args, "model", None):
        raise SystemExit("--model is required")
    slug = model_slug(args.model)
    paths = AblationPaths(
        experiment_root=args.experiment_root,
        dataset=args.dataset,
        cell_id=cell_id,
        model_slug=slug,
    )
    if not paths.reports_dir.is_dir():
        raise SystemExit(f"reports not found at {paths.reports_dir}")

    organs = discover_organs(paths.reports_dir, args.organs)
    if not organs:
        suffix = f" matching {args.organs}" if args.organs else ""
        raise SystemExit(
            f"no organ dirs with *.txt found under {paths.reports_dir}{suffix}")

    if args.run:
        if not re.fullmatch(r"run\d{2}(-[a-z0-9][a-z0-9-]*)?", args.run):
            raise SystemExit(
                f"--run must look like 'run01' or 'run01-<machine>' (got {args.run!r})")
        run_name = args.run
    else:
        try:
            run_name = pick_next_run(paths.cell_dir)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    paths.run_dir(run_name).mkdir(parents=True, exist_ok=True)
    return paths, organs, run_name


# --- IO helpers --------------------------------------------------------------

def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _valid_existing(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return isinstance(data, dict) and not data.get("_pipeline_error") \
                                    and not data.get("_error")


def write_case_json(paths: AblationPaths, run_name: str, organ: str,
                    case_id: str, payload: dict) -> Path:
    """Atomic per-case write into the canonical layout."""
    out_path = paths.case_path(run_name, organ, case_id)
    _atomic_write_json(out_path, payload)
    return out_path


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


# --- Per-case iteration -----------------------------------------------------

def iterate_cases(organs: list[tuple[str, Path]], limit: int | None,
                  ) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(organ_n, case_id, report_path)`` for every report under
    each organ subdirectory, capped per-organ by ``limit``."""
    for organ_n, organ_dir in organs:
        for report_path in discover_cases(organ_dir, limit):
            yield organ_n, report_path.stem, report_path


# --- Summary + manifest -----------------------------------------------------

@dataclass
class RunSummary:
    """Per-run accumulator, written as ``_summary.json`` at finalize."""
    run: str
    cell: str
    model_slug: str
    model_alias: str
    dataset: str
    seed: Any = None
    n_cases: int = 0
    n_ok: int = 0
    n_pipeline_error: int = 0
    n_cached: int = 0
    cancer_positive: int = 0
    # Skip taxonomy — populated when a runner emits ``_skip_reason`` on its
    # per-case payload. Lets the user tell at a glance whether the
    # downstream organ predictor was even invoked.
    n_skipped_not_cancer: int = 0
    n_skipped_unknown_organ: int = 0
    n_downstream_called: int = 0
    per_organ: dict[str, dict[str, int]] = field(default_factory=dict)
    wall_time_s: float = 0.0
    started_at: str = field(default_factory=_utc_now_iso)
    finished_at: str = ""

    def record(self, organ: str, status: str, *,
               is_cancer: bool | None = None,
               skip_reason: str | None = None,
               downstream_called: bool = False) -> None:
        self.n_cases += 1
        per = self.per_organ.setdefault(organ, {
            "n_cases": 0, "n_ok": 0, "n_pipeline_error": 0,
            "n_cached": 0, "cancer_positive": 0,
        })
        per["n_cases"] += 1
        if status == "ok":
            self.n_ok += 1
            per["n_ok"] += 1
            if is_cancer:
                self.cancer_positive += 1
                per["cancer_positive"] += 1
            if skip_reason == "not_cancer":
                self.n_skipped_not_cancer += 1
            elif skip_reason == "unknown_organ":
                self.n_skipped_unknown_organ += 1
            if downstream_called:
                self.n_downstream_called += 1
        elif status == "cached":
            self.n_cached += 1
            per["n_cached"] += 1
        elif status == "pipeline_error":
            self.n_pipeline_error += 1
            per["n_pipeline_error"] += 1

    def to_dict(self) -> dict:
        return {
            "run": self.run,
            "cell": self.cell,
            "model_slug": self.model_slug,
            "model_alias": self.model_alias,
            "dataset": self.dataset,
            "seed": self.seed,
            "n_cases": self.n_cases,
            "n_ok": self.n_ok,
            "n_pipeline_error": self.n_pipeline_error,
            "n_cached": self.n_cached,
            "cancer_positive": self.cancer_positive,
            "n_skipped_not_cancer": self.n_skipped_not_cancer,
            "n_skipped_unknown_organ": self.n_skipped_unknown_organ,
            "n_downstream_called": self.n_downstream_called,
            "per_organ": self.per_organ,
            "wall_time_s": round(self.wall_time_s, 1),
            "parse_error_rate": (self.n_pipeline_error / max(self.n_cases, 1)),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _config_hash(model_id: str, decoding: dict) -> str:
    payload = json.dumps(
        {"model": model_id, "decoding": {k: decoding.get(k)
                                         for k in ("temperature", "top_p",
                                                   "max_tokens", "num_ctx",
                                                   "seed")}},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def update_cell_manifest(paths: AblationPaths, run_name: str,
                         summary: RunSummary, *,
                         model_alias: str, decoding: dict | None = None,
                         extra: dict | None = None) -> Path:
    """Read-modify-write the cell-level ``_manifest.yaml`` so each new run
    appends/updates its own entry without clobbering prior runs."""
    import yaml

    manifest_path = paths.cell_dir / "_manifest.yaml"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
    else:
        manifest = {}

    manifest.setdefault("experiment_id",
                        f"ablation_{paths.cell_id}_{paths.model_slug}_{paths.dataset}_v1")
    manifest["cell"] = paths.cell_id
    manifest["dataset"] = paths.dataset
    manifest["model_slug"] = paths.model_slug
    manifest["model_alias"] = model_alias
    manifest.setdefault("created_at",
                        dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"))
    if decoding is not None:
        manifest["config_hash"] = _config_hash(
            model_list.get(model_alias, model_alias), decoding)

    runs = list(manifest.get("runs") or [])
    entry = {
        "run": run_name,
        "seed": summary.seed,
        "valid": summary.n_pipeline_error == 0 and summary.n_cases > 0,
        "parse_error_rate": (summary.n_pipeline_error
                             / max(summary.n_cases, 1)),
    }
    if extra:
        entry.update(extra)
    replaced = False
    for i, r in enumerate(runs):
        if r.get("run") == run_name:
            runs[i] = entry
            replaced = True
            break
    if not replaced:
        runs.append(entry)
    runs.sort(key=lambda r: r.get("run", ""))
    manifest["runs"] = runs
    manifest["k"] = len(runs)
    _atomic_write_yaml(manifest_path, manifest)
    return manifest_path


def finalize_run(paths: AblationPaths, run_name: str, summary: RunSummary,
                 *, model_alias: str, decoding: dict | None = None,
                 extra_meta: dict | None = None,
                 manifest_extra: dict | None = None) -> None:
    """Write ``_summary.json`` + ``_run_meta.json`` + update ``_manifest.yaml``."""
    summary.finished_at = _utc_now_iso()
    run_dir = paths.run_dir(run_name)
    _atomic_write_json(run_dir / "_summary.json", summary.to_dict())

    meta = {
        "run": run_name,
        "cell": paths.cell_id,
        "model_alias": model_alias,
        "model_slug": paths.model_slug,
        "model_id": model_list.get(model_alias, model_alias),
        "dataset": paths.dataset,
        "experiment_root": str(paths.experiment_root.resolve()),
        "ollama_endpoint": localaddr,
        "started_at": summary.started_at,
        "finished_at": summary.finished_at,
        "decoding": decoding or {},
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "argv": sys.argv,
    }
    if extra_meta:
        meta.update(extra_meta)
    _atomic_write_json(run_dir / "_run_meta.json", meta)

    update_cell_manifest(paths, run_name, summary,
                         model_alias=model_alias, decoding=decoding,
                         extra=manifest_extra)


# --- DSPy bootstrap ---------------------------------------------------------

def setup_dspy_lm(model_alias: str, overrides: dict | None = None):
    """Configure DSPy's global LM and return the LM kwargs dict for logging.

    Wraps ``models.common.load_model`` + ``dspy.configure`` and returns the
    full kwargs that were applied — useful for the ``_run_meta.json`` and
    ``_manifest.yaml`` records.
    """
    import dspy

    lm = load_model(model_alias, overrides=overrides)
    dspy.configure(lm=lm)
    return {
        "temperature": getattr(lm, "temperature", None) or lm.kwargs.get("temperature"),
        "top_p": lm.kwargs.get("top_p"),
        "top_k": lm.kwargs.get("top_k"),
        "max_tokens": lm.kwargs.get("max_tokens"),
        "num_ctx": lm.kwargs.get("num_ctx"),
        "repeat_penalty": lm.kwargs.get("repeat_penalty"),
        "keep_alive": lm.kwargs.get("keep_alive"),
        "cache": lm.kwargs.get("cache"),
        "seed": lm.kwargs.get("seed"),
    }


def load_decoding_overrides(model_alias: str) -> dict:
    """Read ``configs/dspy_ollama_<alias>.yaml`` decoding overrides, if any."""
    cfg = load_model_config(model_alias)
    return split_decoding_overrides(cfg.get("decoding"))


# --- DSPy trace dump --------------------------------------------------------

_TRACE_TRUNCATE = 8000  # max chars per message / response in the JSONL


def _truncate(s: Any, n: int = _TRACE_TRUNCATE) -> Any:
    if isinstance(s, str) and len(s) > n:
        return s[:n] + f"…[truncated {len(s) - n} chars]"
    return s


def _try_get_dspy_lm():
    """Return ``dspy.settings.lm`` if DSPy is loaded, else None.

    Imported lazily so non-DSPy runners (raw_json, constrained_decoding,
    free_text_regex, ...) don't pay the import cost.
    """
    try:
        import dspy
    except Exception:
        return None
    return getattr(dspy.settings, "lm", None)


def dump_dspy_trace(case_id: str, run_dir: Path, since_index: int,
                    organ: str | None = None) -> int:
    """Append every ``lm.history`` entry since ``since_index`` to
    ``run_dir/_dspy_trace.jsonl`` and return the new history length.

    Each line is one JSON object: ``{case_id, organ, step, messages,
    response_text, prompt_tokens, completion_tokens, kwargs}``. Long
    fields are truncated at ``_TRACE_TRUNCATE`` chars to keep the file
    bounded; full content stays in ``lm.history`` for replay.
    """
    lm = _try_get_dspy_lm()
    if lm is None:
        return since_index
    history = list(getattr(lm, "history", []) or [])
    if since_index >= len(history):
        return len(history)
    out_path = run_dir / "_dspy_trace.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for entry in history[since_index:]:
            messages = entry.get("messages") or []
            # Each message is {role, content}; truncate content per-message.
            safe_messages = [
                {"role": m.get("role"),
                 "content": _truncate(m.get("content", ""))}
                for m in messages
            ]
            response = entry.get("response")
            if isinstance(response, dict):
                response_text = response.get("content")
            else:
                response_text = response
            fh.write(json.dumps({
                "case_id": case_id,
                "organ": organ,
                "step_index": since_index + 1,
                "model": entry.get("model"),
                "messages": safe_messages,
                "response_text": _truncate(response_text),
                "prompt_tokens": (entry.get("usage") or {}).get(
                    "prompt_tokens"),
                "completion_tokens": (entry.get("usage") or {}).get(
                    "completion_tokens"),
                "kwargs": entry.get("kwargs"),
                "cost": entry.get("cost"),
            }, ensure_ascii=False, default=str) + "\n")
            since_index += 1
    return len(history)


# --- Run-level loop helper --------------------------------------------------

def run_loop(paths: AblationPaths, organs: list[tuple[str, Path]],
             run_name: str, *, model_alias: str,
             predict: callable, args: argparse.Namespace,
             logger: logging.Logger, decoding: dict | None = None,
             manifest_extra: dict | None = None,
             extra_meta: dict | None = None) -> RunSummary:
    """Boilerplate per-run loop: iterate cases, call ``predict``, log,
    finalize. ``predict(report_text, organ, case_id) -> dict`` returns
    the per-case prediction payload."""
    run_dir = paths.run_dir(run_name)
    summary = RunSummary(
        run=run_name, cell=paths.cell_id, model_slug=paths.model_slug,
        model_alias=model_alias, dataset=paths.dataset,
        seed=(decoding or {}).get("seed"),
    )

    log_path = run_dir / "_log.jsonl"
    trace_dspy = bool(getattr(args, "trace_dspy", False)
                      or getattr(args, "verbose", False))
    trace_cursor = 0
    if trace_dspy:
        # Initialise cursor at current history length so we only capture
        # calls made by this run (not anything DSPy buffered earlier).
        lm = _try_get_dspy_lm()
        trace_cursor = len(getattr(lm, "history", []) or []) if lm else 0
    t_run = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log_fh:
        for organ_n, case_id, report_path in iterate_cases(organs, args.limit):
            out_path = paths.case_path(run_name, organ_n, case_id)
            started_at = _utc_now_iso()
            if not args.overwrite and _valid_existing(out_path):
                summary.record(organ_n, "cached")
                logger.info("[%s/%s/%s] cached — skipped",
                            run_name, organ_n, case_id)
                log_fh.write(json.dumps({
                    "case_id": case_id, "organ": organ_n, "run": run_name,
                    "status": "cached", "latency_s": 0.0,
                    "started_at": started_at,
                }, ensure_ascii=False) + "\n")
                log_fh.flush()
                continue

            report = report_path.read_text(encoding="utf-8")
            t0 = time.perf_counter()
            try:
                payload = predict(report, organ_n, case_id)
                latency_s = round(time.perf_counter() - t0, 3)
                write_case_json(paths, run_name, organ_n, case_id, payload)
                is_cancer = bool(payload.get("cancer_excision_report"))
                skip_reason = payload.get("_skip_reason")
                downstream_called = bool(payload.get("_downstream_called"))
                summary.record(organ_n, "ok", is_cancer=is_cancer,
                               skip_reason=skip_reason,
                               downstream_called=downstream_called)
                logger.info("[%s/%s/%s] ok (%.2fs, cancer=%s, skip=%s, downstream=%s)",
                            run_name, organ_n, case_id, latency_s,
                            is_cancer, skip_reason, downstream_called)
                row = {"case_id": case_id, "organ": organ_n, "run": run_name,
                       "status": "ok", "latency_s": latency_s,
                       "is_cancer": is_cancer,
                       "cancer_category": payload.get("cancer_category"),
                       "skip_reason": skip_reason,
                       "downstream_called": downstream_called,
                       "started_at": started_at}
            except Exception as exc:
                latency_s = round(time.perf_counter() - t0, 3)
                sentinel = {"_pipeline_error": True,
                            "reason": type(exc).__name__,
                            "message": str(exc)[:2000]}
                write_case_json(paths, run_name, organ_n, case_id, sentinel)
                summary.record(organ_n, "pipeline_error")
                logger.error("[%s/%s/%s] pipeline error: %s",
                             run_name, organ_n, case_id, exc)
                row = {"case_id": case_id, "organ": organ_n, "run": run_name,
                       "status": "pipeline_error", "latency_s": latency_s,
                       "error": f"{type(exc).__name__}: {exc}",
                       "started_at": started_at}
            log_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_fh.flush()
            if trace_dspy:
                trace_cursor = dump_dspy_trace(
                    case_id, run_dir, trace_cursor, organ=organ_n)

    summary.wall_time_s = time.perf_counter() - t_run
    finalize_run(paths, run_name, summary, model_alias=model_alias,
                 decoding=decoding, extra_meta=extra_meta,
                 manifest_extra=manifest_extra)
    return summary


def make_logger(name: str, run_dir: Path, verbose: bool) -> logging.Logger:
    return setup_logger(
        name=name,
        level=logging.DEBUG if verbose else logging.INFO,
        log_file=str(run_dir / "_run.log"),
        json_format=False,
    )


__all__ = [
    "DATASETS", "MAX_RUN_SLOTS", "UNIFIED_MODELS",
    "AblationPaths", "RunSummary",
    "model_slug", "ollama_tag", "default_api_base",
    "add_canonical_args", "apply_obfustrated_default", "resolve_run_paths",
    "discover_organs", "discover_cases", "pick_next_run",
    "iterate_cases", "write_case_json",
    "finalize_run", "update_cell_manifest", "run_loop",
    "setup_dspy_lm", "load_decoding_overrides",
    "dump_dspy_trace",
    "make_logger",
]
