"""
Aggregate ablation cell predictions written under the canonical layout
by delegating per-(cell, model) scoring to the cascade evaluator.

Canonical layout (see ``runners/_base.py``):

    {root}/results/ablations/{dataset}/{cell_id}/{model_slug}/{run_id}/{organ_n}/{case_id}.json

Per-(cell, model) scoring runs ``scripts.eval.cli cascade --method
ablation --cell <cell> --cell-model <model> ...`` which produces a
``cascade_atomic.parquet`` under
``{results_root}/{cell}/{model}/_cascade_eval/``. Each per-pair atomic
already carries ``cell`` and ``model_slug`` columns (Phase D2 of the
cascade-conformance migration). The orchestrator concatenates them
into the master ``cascade_atomic.parquet`` and ``ablation_grid.csv``
files used by the canonical-stats and ablation-stats layers.

Output files, all under ``--results-root`` (default:
``{folder}/results/ablations/{dataset}/``):

    cascade_atomic.parquet    full cascade atomic across (cell, model, run, case, field)
    atomic.parquet            backward-compat alias (same content)
    ablation_grid.csv         long-form: one row per (cell, model, run, case, field)
    ablation_summary.csv      per-(cell, model, field): accuracy + coverage
    ablation_table.csv        pivot: rows=field, cols=<cell>_<model>, cells=accuracy
    cell_deltas.csv           per-field deltas vs the configured baseline
    efficiency.csv            mean / median latency, schema-error rate, parse-error rate

Statistical CSVs (``ablation_paired_deltas.csv`` etc.) are written by
:mod:`stats` when ``--with-stats`` is on (default for non-smoke
results-roots).

Usage::

    python -m digital_registrar_research.ablations.eval.run_ablations \\
        --folder dummy --dataset tcga
    python -m digital_registrar_research.ablations.eval.run_ablations \\
        --results-root /custom/path/ablations --dataset tcga
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ...benchmarks.eval.metrics import (
    BREAST_BIOMARKERS,
    # Legacy aggregator imports retained for the soft-fail / fallback
    # path (`_grade_run`, `build_grid_dataframe`). The canonical scorer
    # is now the cascade subprocess, but unit tests and any in-process
    # caller that constructs grids directly still need these.
    FAIR_SCOPE,
    NESTED_LIST_FIELDS,
    match_nested_list,
    score_case,
)
from ...benchmarks.eval.scope import IMPLEMENTED_ORGANS

DEFAULT_BASELINE = "dspy_modular"


# ---------------------------------------------------------------------------
# Canonical path resolution
# ---------------------------------------------------------------------------

def _ablations_root(args: argparse.Namespace) -> Path:
    """Resolve the per-dataset ablations root from args.

    Three input shapes (in order of precedence):

    1. ``--results-root <path>`` — an absolute or relative path to a
       directory containing per-cell subdirs. Used as-is.
    2. ``--folder <root> --dataset <name>`` — canonical;
       resolves to ``{root}/results/ablations/{dataset}``.
    3. Neither — falls back to
       :data:`digital_registrar_research.paths.ABLATIONS_RESULTS`.
    """
    if args.results_root is not None:
        return Path(args.results_root)
    if args.experiment_root is not None and args.dataset:
        return (Path(args.experiment_root) / "results" / "ablations"
                / args.dataset)
    from ...paths import ABLATIONS_RESULTS
    return ABLATIONS_RESULTS


def _gold_root(args: argparse.Namespace) -> Path:
    """Locate the gold annotation tree.

    Canonical: ``{folder}/data/{dataset}/annotations/gold/``. Falls
    back to the legacy ``GOLD_ANNOTATIONS`` constant if --folder isn't
    given.
    """
    if args.experiment_root is not None and args.dataset:
        return (Path(args.experiment_root) / "data" / args.dataset
                / "annotations" / "gold")
    from ...paths import GOLD_ANNOTATIONS
    return GOLD_ANNOTATIONS


# ---------------------------------------------------------------------------
# Canonical-tree discovery
# ---------------------------------------------------------------------------

def _discover_runs(ablations_root: Path,
                   cells: list[str] | None = None,
                   models: list[str] | None = None,
                   ) -> list[tuple[str, str, str, Path]]:
    """Yield ``(cell_id, model_slug, run_id, run_dir)`` for every
    completed (with ``_summary.json``) run under the canonical tree."""
    if not ablations_root.is_dir():
        return []
    out: list[tuple[str, str, str, Path]] = []
    for cell_dir in sorted(ablations_root.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name.startswith("_"):
            continue
        if cells and cell_dir.name not in cells:
            continue
        for model_dir in sorted(cell_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("_"):
                continue
            if models and model_dir.name not in models:
                continue
            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir() or run_dir.name.startswith("_"):
                    continue
                if (run_dir / "_summary.json").exists():
                    out.append((cell_dir.name, model_dir.name,
                                run_dir.name, run_dir))
    return out


def _gold_for(case_id: str, organ_n: str, gold_root: Path) -> dict | None:
    """Read the gold annotation for ``(organ_n, case_id)`` from the
    canonical layout. Returns None if missing."""
    gold_path = gold_root / organ_n / f"{case_id}.json"
    if not gold_path.exists():
        return None
    try:
        with gold_path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Case- and field-level status classification
# ---------------------------------------------------------------------------
#
# Mutually-exclusive case_status precedence. Earlier entries win when a
# case carries multiple sentinels.
CASE_STATUS_PRECEDENCE: tuple[str, ...] = (
    "gold_missing",
    "prediction_unreadable",
    "pipeline_error",
    "parse_error",
    "schema_error",
    "renest_error",
    "b2_parse_error",
    "section_error",
    "skipped_intentional",
    "ok",
)

# Case statuses under which a field can still be graded normally. The
# others render every field unscoreable_due_to_case_error.
_GRADABLE_CASE_STATUS: frozenset[str] = frozenset({
    "ok",
    "schema_error",
    "renest_error",
    "b2_parse_error",
    "section_error",
})


def _classify_case(pred: dict | None,
                   pred_unreadable_reason: str | None = None,
                   gold_missing: bool = False,
                   ) -> tuple[str, list[str]]:
    """Resolve a case-level status from runner sentinel keys.

    Reads the runner-emitted sentinels (``_pipeline_error``,
    ``_parse_error`` / ``_error``, ``_schema_errors``,
    ``_renest_errors``, ``_b2_parse_errors``, ``_per_section_errors``,
    ``_skip_reason``) and resolves to a single ``case_status`` plus the
    full list of truthy flags.

    Args:
        pred: parsed prediction dict, or None if unreadable.
        pred_unreadable_reason: short repr of the parse exception, if
            the prediction file failed to load. Forces
            ``prediction_unreadable``.
        gold_missing: whether the matching gold annotation is absent.
            Forces ``gold_missing``.

    Returns:
        ``(case_status, flags)`` — ``flags`` is the list of every
        truthy sentinel observed, in precedence order. Useful for
        diagnostic columns; the chosen ``case_status`` is the first
        entry of the precedence list that fired.
    """
    flags: list[str] = []
    if gold_missing:
        flags.append("gold_missing")
    if pred_unreadable_reason is not None:
        flags.append("prediction_unreadable")
    if isinstance(pred, dict):
        if pred.get("_pipeline_error"):
            flags.append("pipeline_error")
        if pred.get("_parse_error") or pred.get("_error"):
            flags.append("parse_error")
        if pred.get("_schema_errors"):
            flags.append("schema_error")
        if pred.get("_renest_errors"):
            flags.append("renest_error")
        if pred.get("_b2_parse_errors"):
            flags.append("b2_parse_error")
        if pred.get("_per_section_errors"):
            flags.append("section_error")
        if pred.get("_skip_reason") in ("not_cancer", "unknown_organ"):
            flags.append("skipped_intentional")
    if not flags:
        flags.append("ok")
    # Precedence resolution.
    for status in CASE_STATUS_PRECEDENCE:
        if status in flags:
            return status, flags
    return "ok", flags


def _classify_field(case_status: str,
                    gold: dict | None,
                    pred: dict | None,
                    field: str,
                    scored: object,
                    is_nested: bool = False,
                    schema_errors: list[str] | None = None,
                    ) -> tuple[str, str]:
    """Resolve a field-level status given the case-level outcome and
    the per-field scoring result from ``score_case`` / ``match_nested_list``.

    Returns ``(field_status, field_error_detail)``. ``field_error_detail``
    is ≤120 chars and is empty for ``correct`` rows.
    """
    if case_status not in _GRADABLE_CASE_STATUS:
        return "unscoreable_due_to_case_error", ""

    # Gold-presence check. Missing gold field is distinct from gold
    # being null (which still scores against null).
    g_value = None
    if isinstance(gold, dict):
        if field in gold:
            g_value = gold.get(field)
        else:
            cd = gold.get("cancer_data") or {}
            if field in cd:
                g_value = cd.get(field)
            else:
                # biomarker_<X> fields live under cancer_data.biomarkers
                if field.startswith("biomarker_"):
                    bm = (cd.get("biomarkers") or {})
                    if field[len("biomarker_"):] not in bm:
                        return "gold_missing", ""
                else:
                    return "gold_missing", ""

    # Nested fields (regional_lymph_node, margins): scored is a float
    # F1 in (0, 1]. f1==1 → correct; f1==0 with any non-empty side →
    # treat as wrong_value; in-between → misaligned_list.
    if is_nested and isinstance(scored, (int, float)):
        if scored >= 1.0:
            return "correct", ""
        if scored <= 0.0:
            return "wrong_value", ""
        return "misaligned_list", f"nested_f1={float(scored):.2f}"

    # Scalar / list-of-literals fields.
    if scored is True:
        return "correct", ""
    if scored is False:
        # Distinguish wrong_type from wrong_value using schema errors
        # that name this field.
        detail = ""
        if schema_errors:
            for err in schema_errors:
                if isinstance(err, str) and field in err:
                    detail = err[:120]
                    return "wrong_type", detail
        return "wrong_value", _short_pred_gold(g_value, _pred_value(pred, field))
    # scored is None: not attempted. Distinguish missing_key vs null_value.
    if isinstance(pred, dict):
        if _pred_has_key(pred, field):
            return "null_value", ""
        return "missing_key", ""
    return "missing_key", ""


def _pred_has_key(pred: dict, field: str) -> bool:
    if field in pred:
        return True
    cd = pred.get("cancer_data") or {}
    if field in cd:
        return True
    if field.startswith("biomarker_"):
        bm = (cd.get("biomarkers") or {})
        return field[len("biomarker_"):] in bm
    return False


def _pred_value(pred: dict | None, field: str):
    if not isinstance(pred, dict):
        return None
    if field in pred:
        return pred[field]
    cd = pred.get("cancer_data") or {}
    if field in cd:
        return cd[field]
    if field.startswith("biomarker_"):
        bm = (cd.get("biomarkers") or {})
        return bm.get(field[len("biomarker_"):])
    return None


def _short_pred_gold(gold_value, pred_value) -> str:
    """Format a short ``gold=… pred=…`` description, ≤120 chars."""
    g = repr(gold_value)
    p = repr(pred_value)
    if len(g) > 50:
        g = g[:47] + "..."
    if len(p) > 50:
        p = p[:47] + "..."
    out = f"gold={g} pred={p}"
    return out[:120]


# ---------------------------------------------------------------------------
# Cascade-delegated scoring
# ---------------------------------------------------------------------------
#
# The cascade evaluator is the canonical scorer. We loop over discovered
# (cell, model) pairs and invoke ``python -m scripts.eval.cli cascade``
# as a subprocess for each pair. The per-pair output lives under
# ``_cascade_eval/`` so re-runs of run_ablations don't overwrite the
# ablation prediction tree, and ``_discover_runs`` skips it via the
# leading-underscore convention.


def _score_pair_via_cascade(
    *,
    cell: str,
    model: str,
    run_ids: list[str],
    experiment_root: Path,
    dataset: str,
    out_dir: Path,
    device: str = "cpu",
) -> pd.DataFrame:
    """Run cascade for one (cell, model) pair and return its atomic frame.

    Invokes ``python -m scripts.eval.cli cascade --method ablation``
    in a subprocess. Reads the resulting ``cascade_atomic.parquet`` and
    returns it as a DataFrame. Raises :class:`RuntimeError` if cascade
    exits non-zero or the output parquet is missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "scripts.eval.cli", "cascade",
        "--root", str(experiment_root),
        "--dataset", dataset,
        "--method", "ablation",
        "--cell", cell,
        "--cell-model", model,
        "--out", str(out_dir),
        "--device", device,
    ]
    if run_ids:
        cmd.extend(["--run-ids", *run_ids])
    print(f"[aggregate] cascade: cell={cell} model={model} runs={run_ids or 'auto'}")
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise RuntimeError(
            f"cascade subprocess for cell={cell} model={model} exited {rc}",
        )
    parquet = out_dir / "cascade_atomic.parquet"
    if not parquet.is_file():
        raise RuntimeError(
            f"cascade did not produce {parquet} (cell={cell} model={model})",
        )
    return pd.read_parquet(parquet)


def _remap_cascade_to_legacy_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt cascade_atomic schema to the legacy ablation_grid columns.

    Downstream consumers (canonical_stats, the legacy summary_table,
    cell_deltas) expect: ``cell, model, run, case_id, organ, field,
    correct, attempted, cancer_category_mismatch, method, case_status,
    case_flags, field_status, field_error_detail``.

    Mapping rules:
      - ``method`` is composed as ``f"{cell}_{model_slug}"``.
      - ``run`` <- ``run_id``.
      - ``model`` <- ``model_slug`` (the underlying model, not the
        composite). Preserves the legacy meaning of "model".
      - ``cancer_category_mismatch`` <- True for Stage-C scalar rows
        when ``cascade_stage == "C"`` and the case's Stage B was
        wrong. Cascade gating means Stage-C only emits when Stage B
        passed, so this column is always False — kept for schema
        compatibility.
      - ``case_status`` / ``case_flags``: derived from
        ``parse_error`` / ``error_mode``.
      - ``field_status`` / ``field_error_detail``: cascade's
        ``field_kind``-aware mapping (parse_error,
        unscoreable_due_to_case_error, gold_missing, missing_key,
        correct, wrong_value).
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "cell", "model", "run", "case_id", "organ", "field",
            "correct", "attempted", "cancer_category_mismatch", "method",
            "case_status", "case_flags", "field_status", "field_error_detail",
        ])
    out = df.copy()
    # `model_slug` is set by the cascade walker for ablation runs; fall
    # back to `model` if missing (older atomics).
    if "model_slug" not in out.columns:
        out["model_slug"] = out["model"]
    out["method"] = out["cell"].astype(str) + "_" + out["model_slug"].astype(str)
    out["run"] = out["run_id"]
    # Re-point `model` from the composite back to the underlying slug
    # for legacy consumers that expect (method=composite, model=slug).
    out["model"] = out["model_slug"]

    def _row_status(r: pd.Series) -> tuple[str, str, str, str]:
        """Returns (case_status, case_flags, field_status, field_error_detail)."""
        if bool(r.get("parse_error")):
            return ("parse_error", "parse_error",
                    "unscoreable_due_to_case_error", "")
        if not bool(r.get("gold_present", True)):
            return ("ok", "ok", "gold_missing", "")
        if bool(r.get("field_missing")):
            return ("ok", "ok", "missing_key", "")
        if r.get("correct") is True or r.get("correct") == 1:
            return ("ok", "ok", "correct", "")
        if r.get("correct") is False or r.get("correct") == 0:
            gold_v = r.get("gold_value")
            pred_v = r.get("pred_value")
            detail = f"gold={gold_v!r} pred={pred_v!r}"[:120]
            return ("ok", "ok", "wrong_value", detail)
        # Float / nested F1 case (correct in [0, 1]).
        c = r.get("correct")
        if isinstance(c, (int, float)) and 0.0 < float(c) < 1.0:
            return ("ok", "ok", "misaligned_list",
                    f"nested_f1={float(c):.2f}")
        return ("ok", "ok", "wrong_value", "")

    statuses = out.apply(_row_status, axis=1)
    out["case_status"] = [s[0] for s in statuses]
    out["case_flags"] = [s[1] for s in statuses]
    out["field_status"] = [s[2] for s in statuses]
    out["field_error_detail"] = [s[3] for s in statuses]
    out["cancer_category_mismatch"] = False  # gated to False under cascade

    keep = [
        "cell", "model", "run", "case_id", "organ", "field",
        "correct", "attempted", "cancer_category_mismatch", "method",
        "case_status", "case_flags", "field_status", "field_error_detail",
    ]
    return out[[c for c in keep if c in out.columns]].copy()


# Legacy grading helpers retained for the soft-fail empty-grid path and
# any unit tests that import them directly. The cascade walker is the
# canonical scorer for live runs.

def _grade_run(run_dir: Path, gold_root: Path,
               dataset: str | None = None) -> list[dict]:
    """Score every per-case JSON under a single run dir.

    Yields long-form rows ready for the master DataFrame. For each case
    where both gold and prediction supply ``cancer_category``, the rows
    carry ``cancer_category_mismatch=True`` when the two strings
    disagree — an accuracy signal, not a runtime error. Folder numbers
    are treated as case-id keys only and are not compared against
    ``cancer_category``.
    """
    rows: list[dict] = []
    for organ_dir in sorted(run_dir.iterdir()):
        if not organ_dir.is_dir() or organ_dir.name.startswith("_"):
            continue
        organ_n = organ_dir.name
        for pred_path in sorted(organ_dir.glob("*.json")):
            case_id = pred_path.stem
            gold = _gold_for(case_id, organ_n, gold_root)
            gold_missing = gold is None
            pred: dict | object = {}
            pred_unreadable_reason: str | None = None
            try:
                with pred_path.open(encoding="utf-8") as f:
                    pred = json.load(f)
            except Exception as exc:
                pred = {}
                pred_unreadable_reason = repr(exc)[:200]
            case_status, case_flags = _classify_case(
                pred if isinstance(pred, dict) else None,
                pred_unreadable_reason=pred_unreadable_reason,
                gold_missing=gold_missing,
            )
            cc_mismatch = (
                isinstance(pred, dict)
                and isinstance(gold, dict)
                and pred.get("cancer_category") is not None
                and gold.get("cancer_category") is not None
                and pred["cancer_category"] != gold["cancer_category"]
            )
            if cc_mismatch:
                print(f"[aggregate] cancer_category mismatch: case={case_id} "
                      f"folder={organ_n} gold={gold['cancer_category']!r} "
                      f"pred={pred['cancer_category']!r}")
            schema_errors = (pred.get("_schema_errors")
                             if isinstance(pred, dict) else None)
            flags_str = "|".join(case_flags)

            # When gold is missing OR the case is not gradable, emit
            # FAIR_SCOPE rows with correct=None / attempted=False and
            # the appropriate field_status.
            if gold_missing or case_status not in _GRADABLE_CASE_STATUS:
                for field in FAIR_SCOPE:
                    f_status, f_detail = _classify_field(
                        case_status, gold, pred if isinstance(pred, dict) else None,
                        field, scored=None, is_nested=False,
                        schema_errors=schema_errors,
                    )
                    rows.append({
                        "case_id": case_id, "organ": organ_n, "field": field,
                        "correct": None, "attempted": False,
                        "cancer_category_mismatch": cc_mismatch,
                        "case_status": case_status,
                        "case_flags": flags_str,
                        "field_status": f_status,
                        "field_error_detail": f_detail,
                    })
                continue

            result = score_case(gold, pred)
            for field in FAIR_SCOPE + [f"biomarker_{b}" for b in BREAST_BIOMARKERS]:
                if field not in result:
                    continue
                correct = result[field]
                f_status, f_detail = _classify_field(
                    case_status, gold, pred, field, scored=correct,
                    is_nested=False, schema_errors=schema_errors,
                )
                rows.append({
                    "case_id": case_id, "organ": organ_n, "field": field,
                    "correct": (bool(correct) if correct is not None else None),
                    "attempted": correct is not None,
                    "cancer_category_mismatch": cc_mismatch,
                    "case_status": case_status,
                    "case_flags": flags_str,
                    "field_status": f_status,
                    "field_error_detail": f_detail,
                })
            for nested_field, f1d in result.get("_nested", {}).items():
                f1_val = f1d["f1"]
                f_status, f_detail = _classify_field(
                    case_status, gold, pred, nested_field,
                    scored=f1_val, is_nested=True,
                    schema_errors=schema_errors,
                )
                rows.append({
                    "case_id": case_id, "organ": organ_n,
                    "field": nested_field,
                    "correct": f1_val, "attempted": True,
                    "cancer_category_mismatch": cc_mismatch,
                    "case_status": case_status,
                    "case_flags": flags_str,
                    "field_status": f_status,
                    "field_error_detail": f_detail,
                })
    return rows


def build_grid_dataframe(runs: list[tuple[str, str, str, Path]],
                         gold_root: Path,
                         dataset: str | None = None) -> pd.DataFrame:
    """Build the ablation_grid.csv master long-form table.

    Each row carries ``cell, model, run, case_id, organ, field, correct,
    attempted, cancer_category_mismatch, method`` (where
    ``method = f"{cell}_{model}"`` for backward compatibility with
    downstream stats code).
    """
    all_rows: list[dict] = []
    for cell, model, run_id, run_dir in runs:
        rows = _grade_run(run_dir, gold_root, dataset=dataset)
        for r in rows:
            r["cell"] = cell
            r["model"] = model
            r["run"] = run_id
            r["method"] = f"{cell}_{model}"
            all_rows.append(r)
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------

def compute_efficiency(runs: list[tuple[str, str, str, Path]]) -> pd.DataFrame:
    """Aggregate per-run timings + error counts from each ``_summary.json``."""
    rows = []
    for cell, model, run_id, run_dir in runs:
        try:
            with (run_dir / "_summary.json").open(encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue

        # Per-case latencies from _log.jsonl when present.
        latencies: list[float] = []
        log_path = run_dir / "_log.jsonl"
        if log_path.exists():
            with log_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    lat = rec.get("latency_s")
                    if isinstance(lat, (int, float)):
                        latencies.append(float(lat))

        # Per-case error counts from the on-disk JSONs.
        # Mutually-exclusive buckets so the rates can never sum past 1.0:
        #   - schema_only: case has _schema_errors AND no other failure
        #   - parse_only:  case has _parse_error / _error / _pipeline_error
        #                  AND no _schema_errors
        #   - both:        case has both flags
        # ``schema_errors`` and ``parse_errors`` are exposed separately so
        # downstream stats can still report each rate independently;
        # ``failed_total`` = schema_only + parse_only + both gives the
        # overall failure rate (always <= 1.0).
        schema_only = 0
        parse_only = 0
        both = 0
        for organ_dir in run_dir.iterdir():
            if not organ_dir.is_dir() or organ_dir.name.startswith("_"):
                continue
            for pred_path in organ_dir.glob("*.json"):
                try:
                    with pred_path.open(encoding="utf-8") as f:
                        pred = json.load(f)
                except Exception:
                    parse_only += 1
                    continue
                if not isinstance(pred, dict):
                    continue
                has_schema = bool(pred.get("_schema_errors"))
                has_parse = bool(pred.get("_parse_error")
                                 or pred.get("_error")
                                 or pred.get("_pipeline_error"))
                if has_schema and has_parse:
                    both += 1
                elif has_schema:
                    schema_only += 1
                elif has_parse:
                    parse_only += 1

        schema_errors = schema_only + both
        parse_errors = parse_only + both
        failed_total = schema_only + parse_only + both

        rows.append({
            "cell": cell,
            "model": model,
            "run": run_id,
            "n_cases": int(summary.get("n_cases", 0)),
            "mean_latency_s": (sum(latencies) / len(latencies)
                               if latencies else None),
            "median_latency_s": (sorted(latencies)[len(latencies) // 2]
                                 if latencies else None),
            "schema_errors": schema_errors,
            "parse_errors": parse_errors,
            "failed_total": failed_total,
            "validation_retries": summary.get("validation_retries", 0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cell deltas (legacy summary)
# ---------------------------------------------------------------------------

def compute_cell_deltas(long_df: pd.DataFrame,
                        baseline_method: str) -> pd.DataFrame:
    """For each (model, field): Δ accuracy of every other cell vs the
    configured baseline. Point-estimate only; the rich CIs live in
    ``ablation_paired_deltas.csv`` (written by :mod:`stats`).

    Accepts either the cascade_atomic schema (with ``cascade_stage`` and
    ``field_kind`` columns — filtered to Stage-C scalar before reduction)
    or the legacy ablation_grid schema (no stage column — uses every
    row).
    """
    if long_df.empty or "method" not in long_df.columns:
        return pd.DataFrame()
    df = long_df.copy()
    if "cascade_stage" in df.columns:
        df = df[df["cascade_stage"] == "C"]
    if "field_kind" in df.columns:
        df = df[df["field_kind"] != "nested_list"]
    df = df[df["attempted"] == True]  # noqa: E712
    df["accuracy"] = pd.to_numeric(df["correct"], errors="coerce")

    # Per-method × field mean accuracy.
    pivot = df.groupby(["method", "field"])["accuracy"].mean().unstack("method")
    if baseline_method not in pivot.columns:
        return pivot.reset_index()
    base = pivot[baseline_method]
    deltas = pivot.subtract(base, axis="index")
    deltas.columns = [f"delta_{c}_minus_{baseline_method}"
                      for c in pivot.columns]
    out = pd.concat([pivot, deltas], axis=1).reset_index()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", dest="experiment_root", default=None,
                    type=Path,
                    help="Experiment root containing data/ and results/. "
                         "Shorthand 'dummy' or 'workspace' resolves against "
                         "the repo root.")
    ap.add_argument("--dataset", default=None, choices=("cmuh", "tcga"),
                    help="Dataset name under data/ (cmuh or tcga).")
    ap.add_argument("--results-root", type=Path, default=None,
                    help="Override path to scan (overrides --folder/--dataset).")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="Restrict to these cell ids (default: all)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Restrict to these model slugs (default: all)")
    ap.add_argument("--baseline", default="dspy_modular_gpt_oss_20b",
                    help="<cell>_<model_slug> key to use as the Δ baseline")
    ap.add_argument("--with-stats", dest="with_stats",
                    action="store_true", default=None,
                    help="also call ablations.eval.stats.run_all "
                         "(default: ON for real results-root, OFF for _smoke_)")
    ap.add_argument("--no-stats", dest="with_stats", action="store_false")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                    default="cpu",
                    help="Device for the canonical_stats and ablations.stats "
                         "paired-bootstrap layers. Default: cpu (safety net).")
    args = ap.parse_args(argv)

    # Resolve --folder via the same shortcut as the runners.
    if args.experiment_root is not None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[4]
                                   / "scripts"))
            from _config_loader import resolve_folder  # noqa
            args.experiment_root = resolve_folder(args.experiment_root)
        except Exception:
            args.experiment_root = Path(args.experiment_root).resolve()

    results_root = _ablations_root(args)
    gold_root = _gold_root(args)

    runs = _discover_runs(results_root, cells=args.cells, models=args.models)
    if not runs:
        # Soft-fail: emit empty CSVs with full headers so downstream
        # consumers (multirun, paper-table orchestrators) can still run
        # to completion.
        print(f"[aggregate][warn] No completed runs found under "
              f"{results_root}. Writing empty summary scaffolding.",
              file=sys.stderr)
        results_root.mkdir(parents=True, exist_ok=True)
        empty_grid_cols = [
            "cell", "model", "run", "case_id", "organ", "field",
            "correct", "attempted", "cancer_category_mismatch", "method",
            "case_status", "case_flags", "field_status", "field_error_detail",
        ]
        pd.DataFrame(columns=empty_grid_cols).to_csv(
            results_root / "ablation_grid.csv", index=False)
        pd.DataFrame(columns=empty_grid_cols).to_parquet(
            results_root / "atomic.parquet")
        pd.DataFrame(columns=[
            "method", "field", "attempted", "total",
            "coverage", "accuracy_attempted",
        ]).to_csv(results_root / "ablation_summary.csv", index=False)
        return 0

    print(f"[aggregate] results_root={results_root}")
    print(f"[aggregate] gold_root={gold_root}")
    print(f"[aggregate] discovered {len(runs)} runs across "
          f"{len({(c, m) for c, m, _, _ in runs})} (cell, model) pairs")

    # --- Cascade-delegated scoring per (cell, model) pair --------------
    # Determine the dataset to pass to cascade. Prefer the explicit flag;
    # otherwise infer from the results_root path (which is canonically
    # ``{folder}/results/ablations/{dataset}``).
    dataset = args.dataset or _infer_dataset(results_root)
    if not dataset:
        raise SystemExit(
            "could not infer dataset for cascade scoring; "
            "pass --dataset explicitly.",
        )
    experiment_root = (
        Path(args.experiment_root)
        if args.experiment_root is not None
        else _infer_experiment_root(results_root)
    )
    if experiment_root is None:
        raise SystemExit(
            "could not infer experiment root for cascade scoring; "
            "pass --folder explicitly.",
        )

    # Group runs by (cell, model) and delegate one cascade invocation per pair.
    by_pair: dict[tuple[str, str], list[str]] = {}
    for cell, model, run_id, _run_dir in runs:
        by_pair.setdefault((cell, model), []).append(run_id)

    per_pair_atomics: list[pd.DataFrame] = []
    device = getattr(args, "device", "cpu")
    for (cell, model), run_ids in by_pair.items():
        cascade_out = results_root / cell / model / "_cascade_eval"
        atomic = _score_pair_via_cascade(
            cell=cell, model=model, run_ids=sorted(set(run_ids)),
            experiment_root=experiment_root, dataset=dataset,
            out_dir=cascade_out, device=device,
        )
        per_pair_atomics.append(atomic)

    cascade_atomic = pd.concat(per_pair_atomics, ignore_index=True)
    cascade_atomic_path = results_root / "cascade_atomic.parquet"
    try:
        cascade_atomic.to_parquet(cascade_atomic_path)
        print(f"Wrote {cascade_atomic_path}  ({len(cascade_atomic)} rows)")
    except Exception as exc:
        print(f"[aggregate][warn] failed to write {cascade_atomic_path}: {exc!r}",
              file=sys.stderr)

    # Backward-compat outputs: legacy ablation_grid schema for
    # canonical_stats and the legacy summary/pivot tables.
    grid_df = _remap_cascade_to_legacy_grid(cascade_atomic)
    grid_csv = results_root / "ablation_grid.csv"
    grid_df.to_csv(grid_csv, index=False)
    print(f"Wrote {grid_csv}  ({len(grid_df)} rows)")

    atomic_path = results_root / "atomic.parquet"
    try:
        grid_df.to_parquet(atomic_path)
        print(f"Wrote {atomic_path}  ({len(grid_df)} rows)")
    except Exception as exc:
        print(f"[aggregate][warn] failed to write {atomic_path}: {exc!r}",
              file=sys.stderr)

    # Per-method × field summary derived from cascade_atomic. Replaces
    # the legacy ``summary_table`` aggregator with a Stage-C scalar
    # reduction.
    summary = _summary_from_cascade_atomic(cascade_atomic)
    summary_path = results_root / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    if not summary.empty:
        pivot = summary.pivot_table(
            index="field", columns="method",
            values="accuracy_attempted", aggfunc="first")
        pivot.to_csv(results_root / "ablation_table.csv")
        print(f"Wrote {results_root / 'ablation_table.csv'}")
    else:
        pivot = pd.DataFrame()

    deltas = compute_cell_deltas(cascade_atomic, args.baseline)
    if not deltas.empty:
        deltas_path = results_root / "cell_deltas.csv"
        deltas.to_csv(deltas_path, index=False)
        print(f"Wrote {deltas_path}")

    eff = compute_efficiency(runs)
    if not eff.empty:
        eff.to_csv(results_root / "efficiency.csv", index=False)
        print(f"Wrote {results_root / 'efficiency.csv'}")

    # Canonical statistics suite — runs against the legacy long-form
    # grid (already remapped from cascade_atomic above). ``method``
    # column is built as f"{cell}_{model}"; the modular baseline
    # supplied via --baseline becomes the comparator.
    try:
        from . import canonical_stats
        canonical_stats.run_canonical_stats(
            grid_df, modular_method=args.baseline, out_dir=results_root,
            device=device)
    except Exception as exc:
        print(f"[aggregate][warn] canonical stats layer failed: {exc!r}",
              file=sys.stderr)

    is_smoke = results_root.name.startswith("_smoke")
    with_stats = args.with_stats if args.with_stats is not None else not is_smoke
    if with_stats:
        from . import stats as ablation_stats
        try:
            outputs = ablation_stats.run_all(
                results_root, baseline_method=args.baseline, device=device)
            for stage, path in outputs.items():
                print(f"Wrote {path}  (stats: {stage})")
        except Exception as exc:
            print(f"[warn] stats layer failed: {exc!r}")

    if not pivot.empty:
        print("\nper-method mean accuracy:")
        print(pivot.mean().to_string())
    return 0


# ---------------------------------------------------------------------------
# Cascade-atomic helpers
# ---------------------------------------------------------------------------

def _infer_dataset(results_root: Path) -> str | None:
    """Best-effort: parse the dataset from a canonical ablations path.

    Canonical: ``{root}/results/ablations/{dataset}``. The trailing dir
    name is the dataset. Returns None for non-canonical layouts.
    """
    name = results_root.name
    if name in ("cmuh", "tcga"):
        return name
    return None


def _infer_experiment_root(results_root: Path) -> Path | None:
    """Best-effort: walk up from ``{root}/results/ablations/{dataset}`` to ``{root}``."""
    parents = list(results_root.parents)
    # Expect parents[0]=ablations, parents[1]=results, parents[2]=root.
    if len(parents) >= 3 and parents[0].name == "ablations" and parents[1].name == "results":
        return parents[2]
    return None


def _summary_from_cascade_atomic(df: pd.DataFrame) -> pd.DataFrame:
    """Per (method, field) accuracy + coverage derived from cascade_atomic.

    Returns a long-form DataFrame with columns:
        method, field, attempted, total, coverage, accuracy_attempted

    Filters to Stage-C scalar rows; nested-list F1 means are reported in
    the chapter4/chapter5 outputs of each per-pair cascade run, not
    here.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["method", "field", "attempted", "total",
                     "coverage", "accuracy_attempted"],
        )
    sub = df.copy()
    if "cascade_stage" in sub.columns:
        sub = sub[sub["cascade_stage"] == "C"]
    if "field_kind" in sub.columns:
        sub = sub[sub["field_kind"] != "nested_list"]
    # `method` for ablation atomics is just "ablation"; the joint key
    # consumers rely on is `f"{cell}_{model_slug}"`.
    if "model_slug" not in sub.columns:
        sub = sub.assign(model_slug=sub.get("model"))
    sub = sub.assign(
        joint_method=sub["cell"].astype(str) + "_" + sub["model_slug"].astype(str),
    )
    rows: list[dict] = []
    for (method, field), grp in sub.groupby(["joint_method", "field"]):
        attempted = int(grp["attempted"].fillna(False).astype(bool).sum())
        total = int(len(grp))
        n_correct = int(
            pd.to_numeric(
                grp.loc[grp["attempted"].fillna(False).astype(bool), "correct"],
                errors="coerce",
            ).fillna(0).astype(bool).sum()
        )
        rows.append({
            "method": method,
            "field": field,
            "attempted": attempted,
            "total": total,
            "coverage": (attempted / total) if total else float("nan"),
            "accuracy_attempted": (n_correct / attempted) if attempted else float("nan"),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.exit(main())


# Silence unused-import warnings for re-exports kept for downstream use.
_ = (FAIR_SCOPE, NESTED_LIST_FIELDS, IMPLEMENTED_ORGANS, match_nested_list)
