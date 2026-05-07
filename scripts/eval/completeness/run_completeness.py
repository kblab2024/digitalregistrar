"""Completeness subcommand orchestrator.

Takes one or more (method, model, run_ids) triples plus a reference
annotator. Builds a unified atomic table across all methods, then
emits the cross-method missingness deltas, error-mode decomposition,
and the modularity-advantage table.

Output tree:
    manifest.json
    missingness_atomic.parquet
    per_method_per_field.csv
    per_method_per_organ.csv
    per_method_per_fieldtype.csv
    error_mode_decomposition.csv
    method_pairwise_deltas.csv      — paired Δ + McNemar between every method pair
    modularity_advantage.csv        — sorted ablation headline
    schema_conformance_deltas.csv
    refusal_calibration.csv
    position_in_schema_correlation.csv
    heatmap_attempted_rate.csv
"""
from __future__ import annotations

import argparse
import logging
from typing import Iterable

import pandas as pd

from digital_registrar_research.benchmarks.eval.ci_gpu import pick_device
from digital_registrar_research.benchmarks.eval.completeness import (
    aggregate_missingness, method_pair_deltas,
    position_in_schema_correlation, refusal_calibration,
)
from digital_registrar_research.benchmarks.eval.iaa import classify_field
from digital_registrar_research.benchmarks.eval.metrics import normalize
from digital_registrar_research.benchmarks.eval.scope import (
    BREAST_BIOMARKERS, FAIR_SCOPE,
)

from .._common.args import (
    add_common_args, parse_cases, parse_organs, parse_run_ids,
)
from .._common.loaders import load_json, load_prediction, ParseError
from .._common.outcome import CaseLoad, classify_outcome
from .._common.paths import KNOWN_METHODS, Paths, from_args
from .._common.reporting import (
    setup_logging, write_csv, write_manifest, write_parquet,
)
from .._common.stratify import organ_name

logger = logging.getLogger("scripts.eval.completeness")


# --- Argparse registration --------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "completeness",
        help="Detailed missingness analysis across methods.",
        description=__doc__,
    )
    add_common_args(parser, subcommand="completeness")
    parser.add_argument(
        "--methods", nargs="+", required=True,
        help="One or more 'method:model[:run_ids_csv]' triples. "
             "Examples: 'llm:gpt_oss_20b:run01,run02' "
             "(omitting run_ids means autodiscover); "
             "'clinicalbert:v2_finetuned'; 'rule_based:'.",
    )
    parser.add_argument(
        "--annotator", default="gold",
        help="Annotator subdir to score against (default: %(default)s).",
    )
    parser.set_defaults(_handler=_main)


# --- Main entry --------------------------------------------------------------


def _main(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    requested_device = getattr(args, "device", "cpu")
    resolved_device = pick_device(requested_device)
    logger.info("device: requested=%s resolved=%s",
                requested_device, resolved_device)
    args.resolved_device = resolved_device

    paths = from_args(args.root, args.dataset)
    paths.assert_exists()
    organs = parse_organs(args)
    case_filter = parse_cases(args)
    method_specs = [_parse_method_spec(s) for s in args.methods]

    # Build atomic across methods.
    frames: list[pd.DataFrame] = []
    n_per_organ: dict[int, int] = {}
    for method, model, run_ids in method_specs:
        if method == "llm" and not run_ids:
            run_ids = [rid for rid, _ in paths.discover_runs(model)]
        if method == "llm" and not run_ids:
            logger.warning("no runs for %s/%s, skipping", method, model)
            continue
        effective_runs = run_ids if method == "llm" else [""]
        df, npo = _build_atomic_for_method(
            paths=paths, method=method, model=model,
            run_ids=effective_runs, annotator=args.annotator,
            organs=organs, case_filter=case_filter,
        )
        if df.empty:
            continue
        frames.append(df)
        for k, v in npo.items():
            n_per_organ[k] = max(n_per_organ.get(k, 0), v)

    if not frames:
        logger.error("no atomic rows produced; check --methods.")
        return 1
    atomic = pd.concat(frames, ignore_index=True)
    logger.info("atomic table: %d rows across %d methods",
                len(atomic), atomic["method_label"].nunique())

    args.out.mkdir(parents=True, exist_ok=True)
    write_parquet(atomic, args.out / "missingness_atomic.parquet")

    # --- Aggregations -----------------------------------------------------

    write_csv(
        aggregate_missingness(atomic, by=("method_label", "field")),
        args.out / "per_method_per_field.csv",
    )
    write_csv(
        aggregate_missingness(atomic, by=("method_label", "organ")),
        args.out / "per_method_per_organ.csv",
    )
    write_csv(
        aggregate_missingness(atomic, by=("method_label", "field_kind")),
        args.out / "per_method_per_fieldtype.csv",
    )

    # --- Error-mode decomposition -----------------------------------------
    em = (
        atomic[atomic["parse_error"]]
        .assign(error_mode=atomic["error_mode"].fillna("other"))
        .groupby(["method_label", "error_mode"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if not em.empty:
        totals = em.groupby("method_label")["count"].transform("sum")
        em["share"] = em["count"] / totals
        write_csv(em, args.out / "error_mode_decomposition.csv")

    # --- Method-pair Δ ---------------------------------------------------
    deltas = _method_pair_deltas_using_label(
        atomic, by=("field", "organ"), device=resolved_device,
    )
    if not deltas.empty:
        write_csv(deltas, args.out / "method_pairwise_deltas.csv")

    # --- Modularity advantage ---------------------------------------------
    if not deltas.empty:
        adv = (deltas[["method_a", "method_b", "field", "organ",
                       "delta_attempted_rate", "n_paired"]]
               .sort_values("delta_attempted_rate",
                            key=lambda s: s.abs(), ascending=False))
        write_csv(adv, args.out / "modularity_advantage.csv")

    # --- Heatmap (rows=fields, cols=methods) -----------------------------
    heatmap = (
        atomic.groupby(["method_label", "field"])["attempted"]
        .mean()
        .reset_index(name="attempted_rate")
        .pivot(index="field", columns="method_label", values="attempted_rate")
        .reset_index()
    )
    write_csv(heatmap, args.out / "heatmap_attempted_rate.csv", index=False)

    # --- Refusal calibration ---------------------------------------------
    write_csv(
        refusal_calibration(atomic, by=("method_label", "field", "organ")),
        args.out / "refusal_calibration.csv",
    )

    # --- Schema conformance Δ between methods ----------------------------
    # Skip when the atomic table doesn't carry pred_value (lightweight
    # build path).
    if "pred_value" in atomic.columns:
        sc_rows = []
        for method, sub in atomic.groupby("method_label"):
            f = _schema_conformance(sub)
            if not f.empty:
                f["method_label"] = method
                sc_rows.append(f)
        if sc_rows:
            write_csv(pd.concat(sc_rows, ignore_index=True),
                      args.out / "schema_conformance_per_method.csv")

    # --- Position-in-schema correlation ---------------------------------
    field_order = list(FAIR_SCOPE) + [f"biomarker_{b}" for b in BREAST_BIOMARKERS]
    # Use method_label (cross-method axis) as the grouping key, so drop
    # the inner ``method`` column first to avoid duplicate names.
    pos_input = atomic.drop(columns=["method"]).rename(
        columns={"method_label": "method"}
    )
    write_csv(
        position_in_schema_correlation(
            pos_input,
            field_order=field_order, by="method",
        ),
        args.out / "position_in_schema_correlation.csv",
    )

    write_manifest(
        args.out, args, subcommand="completeness",
        n_cases_per_organ=n_per_organ,
        extra={
            "method_specs": [_describe_spec(s) for s in method_specs],
            "n_atomic_rows": int(len(atomic)),
            "device_requested": requested_device,
            "device_resolved": resolved_device,
        },
    )
    logger.info("done. outputs in %s", args.out)
    return 0


# --- Method-spec parsing ----------------------------------------------------


def _parse_method_spec(s: str) -> tuple[str, str | None, list[str] | None]:
    parts = s.split(":")
    if len(parts) < 1 or parts[0] not in KNOWN_METHODS:
        raise SystemExit(
            f"--methods entry must start with {KNOWN_METHODS}: got {s!r}"
        )
    method = parts[0]
    model = parts[1] if len(parts) > 1 and parts[1] else None
    run_ids = (parts[2].split(",") if len(parts) > 2 and parts[2] else None)
    if method != "rule_based" and not model:
        raise SystemExit(f"--methods entry needs model: {s!r}")
    return method, model, run_ids


def _describe_spec(spec: tuple[str, str | None, list[str] | None]) -> str:
    method, model, runs = spec
    return f"{method}/{model or ''}/{','.join(runs or []) or 'auto'}"


# --- Atomic-table builder (per-method) --------------------------------------


def _build_atomic_for_method(
    *, paths: Paths, method: str, model: str | None,
    run_ids: Iterable[str], annotator: str,
    organs: Iterable[int], case_filter: set[str] | None,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Build the atomic outcome table for one method.

    Adds a ``method_label`` column = ``"<method>/<model>"`` for
    cross-method comparison (so two LLM models can be compared).
    """
    fields = list(FAIR_SCOPE) + [f"biomarker_{b}" for b in BREAST_BIOMARKERS]
    rows: list[dict] = []
    n_per_organ: dict[int, int] = {}
    method_label = f"{method}/{model or 'rule_based'}"

    case_index: list[tuple[int, str]] = []
    for organ_idx, case_id in paths.case_ids(annotator, tuple(organs)):
        if case_filter and case_id not in case_filter:
            continue
        case_index.append((organ_idx, case_id))
    for oi, _ in case_index:
        n_per_organ[oi] = n_per_organ.get(oi, 0) + 1

    for run_id in run_ids:
        for organ_idx, case_id in case_index:
            try:
                gold = load_json(paths.annotation(annotator, organ_idx, case_id))
            except ParseError as e:
                logger.warning("skipping %s: %s", case_id, e)
                continue
            organ = normalize(gold.get("cancer_category")) or organ_name(paths.dataset, organ_idx)
            pred_path = paths.prediction(
                method=method, model=model, run_id=run_id or None,
                organ_idx=organ_idx, case_id=case_id,
            )
            lo = load_prediction(pred_path)
            case_load = CaseLoad.from_load_outcome(lo)
            for field in fields:
                if field.startswith("biomarker_") and organ != "breast":
                    continue
                if field.startswith("biomarker_"):
                    out = _classify_biomarker(gold, case_load, field)
                else:
                    out = classify_outcome(gold, case_load, field)

                if (not out.gold_present and not out.attempted
                        and not out.parse_error):
                    continue
                rows.append({
                    "run_id": run_id or "",
                    "method": method,
                    "model": model or "",
                    "method_label": method_label,
                    "annotator": annotator,
                    "case_id": case_id,
                    "organ_idx": int(organ_idx),
                    "organ": organ,
                    "field": field,
                    "field_kind": classify_field(field, organ),
                    "gold_present": out.gold_present,
                    "attempted": out.attempted,
                    "correct": out.correct,
                    "wrong": out.wrong,
                    "field_missing": out.field_missing,
                    "parse_error": out.parse_error,
                    "error_mode": out.error_mode,
                })
    return pd.DataFrame(rows), n_per_organ


def _method_pair_deltas_using_label(df: pd.DataFrame, *, by,
                                     device: str = "cpu") -> pd.DataFrame:
    """Wrapper to call src/.../eval/completeness.method_pair_deltas with
    ``method_label`` instead of ``method`` as the grouping column."""
    df = df.rename(columns={"method": "_orig_method", "method_label": "method"})
    out = method_pair_deltas(df, by=by, device=device)
    if out.empty:
        return out
    return out


# --- Inlined helpers (formerly imported from removed scripts.eval.non_nested) -


def _schema_conformance(df: pd.DataFrame) -> pd.DataFrame:
    """Per (field, organ) — out-of-vocabulary rate among attempted preds.

    Lifted from the removed ``scripts.eval.non_nested.metrics_non_nested``
    so the completeness pipeline doesn't depend on the legacy package.
    For categorical fields only; pairs with the modularity-advantage
    argument that schema-constrained pipelines should produce ~0% OOV.
    """
    from digital_registrar_research.benchmarks.eval.ci import wilson_ci
    from digital_registrar_research.benchmarks.eval.scope import (
        get_allowed_values,
    )

    rows: list[dict] = []
    fields = df["field"].dropna().unique()
    for field in fields:
        for organ in [*sorted(df["organ"].dropna().unique()), "ALL"]:
            organ_arg = organ if organ != "ALL" else None
            allowed = get_allowed_values(field, organ_arg)
            if not allowed:
                continue
            allowed_norm = {normalize(v) for v in allowed}
            sub = df[(df["field"] == field) & df["attempted"]]
            if organ != "ALL":
                sub = sub[sub["organ"] == organ]
            n_attempted = len(sub)
            n_oov = int(sub["pred_value"].apply(
                lambda v: normalize(v) not in allowed_norm and v is not None
            ).sum())
            if n_attempted == 0:
                continue
            lo, hi = wilson_ci(n_oov, n_attempted)
            rows.append({
                "field": field, "organ": organ,
                "n_attempted": n_attempted, "n_oov": n_oov,
                "oov_rate": n_oov / n_attempted,
                "oov_rate_ci_lo": lo, "oov_rate_ci_hi": hi,
            })
    return pd.DataFrame(rows)


def _classify_biomarker(gold, case_load, field: str):
    """Classify biomarker_<X> outcome from breast biomarkers list.

    Replaces the import of ``scripts.eval.non_nested.run_non_nested.
    _classify_biomarker`` (the legacy module no longer exists). For
    breast cases, looks up the named biomarker (e.g. ``biomarker_er``
    → ``er``) inside ``cancer_data.biomarkers`` and classifies as:

      - ``parse_error`` if the case failed to load
      - ``gold_present == False`` if no gold biomarker exists for this category
      - ``attempted`` if the prediction names this category
      - ``correct`` if the predicted expression matches gold

    Returns the same :class:`Outcome` dataclass that
    :func:`classify_outcome` produces.
    """
    from .._common.outcome import Outcome

    biomarker_key = field[len("biomarker_"):]

    def _find(bm_list, key):
        if not isinstance(bm_list, list):
            return None
        for item in bm_list:
            if not isinstance(item, dict):
                continue
            if normalize(item.get("biomarker_category")) == normalize(key):
                return item
        return None

    g_list = (gold.get("cancer_data") or {}).get("biomarkers")
    g_item = _find(g_list, biomarker_key)
    gold_present = g_item is not None

    if not case_load.ok:
        return Outcome(
            kind="parse_error", gold_present=gold_present,
            parse_error=True, field_missing=False,
            attempted=False, correct=False, wrong=False,
            error_mode=case_load.error_mode,
        )

    p = case_load.pred or {}
    p_list = (p.get("cancer_data") or {}).get("biomarkers")
    p_item = _find(p_list, biomarker_key)
    attempted = p_item is not None

    if not gold_present and not attempted:
        return Outcome(
            kind="ineligible", gold_present=False,
            parse_error=False, field_missing=False,
            attempted=False, correct=False, wrong=False,
            error_mode=None,
        )
    if gold_present and not attempted:
        return Outcome(
            kind="field_missing", gold_present=True,
            parse_error=False, field_missing=True,
            attempted=False, correct=False, wrong=False,
            error_mode=None,
        )
    # Attempted; compare expression as the headline scalar.
    g_exp = normalize(g_item.get("expression")) if g_item else None
    p_exp = normalize(p_item.get("expression"))
    correct = (g_exp is not None and p_exp is not None and g_exp == p_exp)
    return Outcome(
        kind="correct" if correct else "wrong",
        gold_present=gold_present,
        parse_error=False, field_missing=False,
        attempted=True, correct=correct, wrong=not correct,
        error_mode=None,
    )


__all__ = ["register"]
