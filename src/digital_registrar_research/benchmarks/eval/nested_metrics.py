"""
Per-field metrics for the two nested-list extraction targets
(`regional_lymph_node`, `margins`) that the flat-scoped `metrics.py`
reduces to a single F1.

These are reported against generative methods only — `rules.py`,
`clinicalbert_cls.py` and `clinicalbert_qa.py` do not emit list-of-dicts
output by design.

Two axes per field:
  - Case-level clinical summary (total counts, any-positive, closest
    margin distance) — what actually drives a treatment decision.
  - Per-item structural accuracy — did we identify the right
    stations / margins, and are their inner values correct.

Matching uses greedy bipartite descent on a similarity score, with a
composite key fallback (so a null `station_name` or `"others"` margin
category doesn't collapse to position-based zip the way
`metrics.match_nested_list` does).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .metrics import NUMERIC_TOLERANCE_MM, normalize
from .scope import get_field_value

LN_COUNT_TOLERANCE = 1  # +/- 1 node on examined / involved counts


# --- Token utilities for margin description Jaccard --------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str | None) -> set[str]:
    if not s:
        return set()
    return set(_TOKEN_RE.findall(s.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- Greedy bipartite matching -----------------------------------------------

def _greedy_match(gold_list: list[dict], pred_list: list[dict],
                  sim_fn) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Returns (matched_pairs, unmatched_gold_idx, unmatched_pred_idx).

    Iteratively picks the (g, p) pair with the highest positive similarity,
    removes both from the pool, repeats until nothing positive remains.
    Quadratic but fine for the ~1-10 items per case we see in practice.
    """
    if not gold_list or not pred_list:
        return [], list(range(len(gold_list))), list(range(len(pred_list)))

    sims = [
        [sim_fn(g, p) for p in pred_list]
        for g in gold_list
    ]
    gold_avail = set(range(len(gold_list)))
    pred_avail = set(range(len(pred_list)))
    matched: list[tuple[int, int]] = []

    while gold_avail and pred_avail:
        best = (0.0, -1, -1)
        for gi in gold_avail:
            for pi in pred_avail:
                s = sims[gi][pi]
                if s > best[0]:
                    best = (s, gi, pi)
        if best[1] < 0:
            break
        _, gi, pi = best
        matched.append((gi, pi))
        gold_avail.remove(gi)
        pred_avail.remove(pi)

    return matched, sorted(gold_avail), sorted(pred_avail)


# --- Similarity functions ----------------------------------------------------

def _ln_similarity(g: dict, p: dict) -> float:
    """Score a (gold, pred) lymph-node pair. Higher = better match.

    Primary-key components (all three optional in schema):
        station_name          +3 if both non-null and equal
        lymph_node_category   +2 if both non-null and equal (excluding "others")
        lymph_node_side       +1 if both non-null and equal

    Tie-break: +0.5 each if involved / examined are within tolerance.
    """
    score = 0.0
    gs, ps = normalize(g.get("station_name")), normalize(p.get("station_name"))
    if gs is not None and ps is not None and gs == ps:
        score += 3.0
    gc, pc = normalize(g.get("lymph_node_category")), normalize(p.get("lymph_node_category"))
    if gc is not None and pc is not None and gc == pc and gc != "others":
        score += 2.0
    gside, pside = normalize(g.get("lymph_node_side")), normalize(p.get("lymph_node_side"))
    if gside is not None and pside is not None and gside == pside:
        score += 1.0
    gi, pi = g.get("involved"), p.get("involved")
    if isinstance(gi, (int, float)) and isinstance(pi, (int, float)) \
            and abs(gi - pi) <= LN_COUNT_TOLERANCE:
        score += 0.5
    ge, pe = g.get("examined"), p.get("examined")
    if isinstance(ge, (int, float)) and isinstance(pe, (int, float)) \
            and abs(ge - pe) <= LN_COUNT_TOLERANCE:
        score += 0.5
    return score


def _margin_similarity(g: dict, p: dict) -> float:
    """Score a (gold, pred) margin pair.

    margin_category        +3 if both equal and not in {None, "others"}
    description Jaccard    +0..3 scaled by token overlap
    margin_involved        +1 if agree (tie-break only)
    """
    score = 0.0
    gc, pc = normalize(g.get("margin_category")), normalize(p.get("margin_category"))
    if gc is not None and pc is not None and gc == pc and gc != "others":
        score += 3.0
    j = _jaccard(_tokens(g.get("description")), _tokens(p.get("description")))
    score += 3.0 * j
    if g.get("margin_involved") is not None and p.get("margin_involved") is not None \
            and bool(g["margin_involved"]) == bool(p["margin_involved"]):
        score += 1.0
    return score


# --- Per-case scoring --------------------------------------------------------

def _safe_list(v) -> list[dict]:
    return v if isinstance(v, list) else []


def _per_station_counts(g_list: list[dict],
                        p_list: list[dict]) -> dict[str, dict]:
    """For each station_name appearing in gold OR pred, summarise the
    per-station (examined, involved) counts.

    Sums duplicate entries with the same station_name so the comparison
    is at the station-aggregate level, not the bipartite-row level.
    Items lacking a station_name are dropped (they cannot be grouped).
    """
    def _bucket(items: list[dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for it in items:
            sn = normalize(it.get("station_name"))
            if sn is None:
                continue
            cur = out.setdefault(sn, {"examined": 0, "involved": 0})
            for k in ("examined", "involved"):
                v = it.get(k)
                if isinstance(v, (int, float)):
                    cur[k] += int(v)
        return out

    g_buckets = _bucket(g_list)
    p_buckets = _bucket(p_list)
    stations = set(g_buckets) | set(p_buckets)
    out: dict[str, dict] = {}
    for sn in stations:
        g_ex = g_buckets.get(sn, {}).get("examined", 0)
        p_ex = p_buckets.get(sn, {}).get("examined", 0)
        g_in = g_buckets.get(sn, {}).get("involved", 0)
        p_in = p_buckets.get(sn, {}).get("involved", 0)
        out[sn] = {
            "gold_examined": g_ex,
            "pred_examined": p_ex,
            "examined_correct_tol": int(abs(g_ex - p_ex) <= LN_COUNT_TOLERANCE),
            "gold_involved": g_in,
            "pred_involved": p_in,
            "involved_correct_tol": int(abs(g_in - p_in) <= LN_COUNT_TOLERANCE),
        }
    return out


def _per_category_status(g_list: list[dict],
                         p_list: list[dict]) -> dict[str, dict]:
    """For each margin_category in gold ∪ pred, summarise whether ANY
    item under that category was involved.

    Independent of the bipartite-matched aggregate. Categories absent
    from one side default to involved=False on that side. Items lacking
    a margin_category are dropped.
    """
    def _by_cat(items: list[dict]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for it in items:
            c = normalize(it.get("margin_category"))
            if c is None:
                continue
            out.setdefault(c, []).append(it)
        return out

    g_by = _by_cat(g_list)
    p_by = _by_cat(p_list)
    cats = set(g_by) | set(p_by)
    out: dict[str, dict] = {}
    for c in cats:
        g_inv = any(bool(x.get("margin_involved")) for x in g_by.get(c, []))
        p_inv = any(bool(x.get("margin_involved")) for x in p_by.get(c, []))
        out[c] = {
            "gold_involved": int(g_inv),
            "pred_involved": int(p_inv),
            "involved_correct": int(g_inv == p_inv),
        }
    return out


def _aggregate_ln_by_group(items: list[dict]) -> dict[tuple, dict]:
    """Aggregate lymph-node items by (side, category), summing counts.

    Returns ``{(side, category): {"examined": int, "involved": int,
    "n_items": int}}``. ``side`` and ``category`` are normalised
    (lowercased / stripped). ``station_name`` is intentionally NOT in
    the key — that's the whole point of the redesign: station_name is
    an unstable free-text identifier that should not influence matching
    or scoring.
    """
    groups: dict[tuple, dict] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        side = normalize(it.get("lymph_node_side"))
        category = normalize(it.get("lymph_node_category"))
        key = (side, category)
        slot = groups.setdefault(
            key, {"examined": 0, "involved": 0, "n_items": 0},
        )
        e_v, i_v = it.get("examined"), it.get("involved")
        if isinstance(e_v, (int, float)):
            slot["examined"] += int(e_v)
        if isinstance(i_v, (int, float)):
            slot["involved"] += int(i_v)
        slot["n_items"] += 1
    return groups


def score_lymph_nodes(gold: dict, pred: dict) -> dict:
    """Per-case lymph-node metrics — category-aggregation scoring.

    The redesigned scorer sums ``examined`` and ``involved`` per
    ``(lymph_node_side, lymph_node_category)`` group on each side of
    the comparison, then matches groups by key. ``station_name`` is no
    longer used — the gold annotators' free-text station labels are
    unstable across cases and inflate disagreements that are textual
    rather than clinical (e.g. gold splits "nonsentinel" into two
    rows of 2 examined each, the model emits one row of 4).

    Return-dict keys preserve the legacy contract so the cascade
    reductions in :mod:`scripts.eval.cascade.reductions` and the
    legacy ``run_nested.py`` (until deletion) consume the same shape.
    Crucial difference: ``ln_station_*`` keys now refer to GROUPS,
    not per-list-item rows. ``ln_station_matched`` is the count of
    (side, category) groups present on both sides.

    New keys
    --------
        ln_group_recall            fraction of gold groups also produced
                                   by the model
        ln_group_precision         fraction of pred groups also in gold
        ln_n_groups_gold           number of distinct (side, category) groups
                                   on the gold side
        ln_n_groups_pred           same for pred
    """
    g_list = _safe_list(get_field_value(gold, "regional_lymph_node"))
    p_list = _safe_list(get_field_value(pred, "regional_lymph_node"))

    g_groups = _aggregate_ln_by_group(g_list)
    p_groups = _aggregate_ln_by_group(p_list)

    g_exam = sum(v["examined"] for v in g_groups.values())
    p_exam = sum(v["examined"] for v in p_groups.values())
    g_inv = sum(v["involved"] for v in g_groups.values())
    p_inv = sum(v["involved"] for v in p_groups.values())
    g_any = g_inv > 0
    p_any = p_inv > 0

    # Group-level matching: both sides must have the same (side, category).
    # No bipartite scoring — group keys are deterministic.
    common_keys = set(g_groups) & set(p_groups)
    only_gold = set(g_groups) - set(p_groups)
    only_pred = set(p_groups) - set(g_groups)
    tp = len(common_keys)
    fn = len(only_gold)
    fp = len(only_pred)

    n_g_groups = len(g_groups)
    n_p_groups = len(p_groups)

    examined_ok = involved_ok = category_ok = side_ok = 0
    for key in common_keys:
        g, p = g_groups[key], p_groups[key]
        # Examined / involved tolerance match.
        if abs(g["examined"] - p["examined"]) <= LN_COUNT_TOLERANCE:
            examined_ok += 1
        if abs(g["involved"] - p["involved"]) <= LN_COUNT_TOLERANCE:
            involved_ok += 1
        # By construction the side and category match — record as 1.0
        # so the downstream per-attribute reductions report this
        # explicitly rather than silently dropping the columns.
        category_ok += 1
        side_ok += 1

    group_recall = tp / n_g_groups if n_g_groups else float("nan")
    group_precision = tp / n_p_groups if n_p_groups else float("nan")

    # Per-group ledger for chapter-5 per-category reducers and the
    # single-group confusion view. Entries: list of dicts with side,
    # category, presence flags, and the matched counts when both sides
    # have the group.
    per_group_ledger: list[dict] = []
    for key in set(g_groups) | set(p_groups):
        side, category = key
        g_present = key in g_groups
        p_present = key in p_groups
        g = g_groups.get(key, {})
        p = p_groups.get(key, {})
        per_group_ledger.append({
            "side": side, "category": category,
            "gold_present": int(g_present),
            "pred_present": int(p_present),
            "gold_examined": int(g.get("examined", 0)),
            "pred_examined": int(p.get("examined", 0)),
            "gold_involved": int(g.get("involved", 0)),
            "pred_involved": int(p.get("involved", 0)),
            "examined_correct_tol": int(
                g_present and p_present
                and abs(g.get("examined", 0) - p.get("examined", 0))
                    <= LN_COUNT_TOLERANCE,
            ),
            "involved_correct_tol": int(
                g_present and p_present
                and abs(g.get("involved", 0) - p.get("involved", 0))
                    <= LN_COUNT_TOLERANCE,
            ),
        })

    # Single-group cases support a clean (gold_side, pred_side) and
    # (gold_cat, pred_cat) confusion observation. Multi-group cases are
    # ambiguous — we record them as None for downstream filtering.
    if n_g_groups == 1 and n_p_groups == 1:
        gk = next(iter(g_groups.keys()))
        pk = next(iter(p_groups.keys()))
        single_group_pair_side = (gk[0], pk[0])
        single_group_pair_category = (gk[1], pk[1])
    else:
        single_group_pair_side = None
        single_group_pair_category = None

    return {
        # Case-level totals (clinically actionable headlines).
        "ln_examined_total_gold": g_exam,
        "ln_examined_total_pred": p_exam,
        "ln_examined_total_abs_err": abs(g_exam - p_exam),
        "ln_examined_total_correct_tol": int(abs(g_exam - p_exam) <= LN_COUNT_TOLERANCE),
        "ln_involved_total_gold": g_inv,
        "ln_involved_total_pred": p_inv,
        "ln_involved_total_abs_err": abs(g_inv - p_inv),
        "ln_involved_total_correct_tol": int(abs(g_inv - p_inv) <= LN_COUNT_TOLERANCE),
        "ln_any_positive_gold": int(g_any),
        "ln_any_positive_pred": int(p_any),
        "ln_any_positive_correct": int(g_any == p_any),
        # Legacy-named "stations" keys now refer to GROUPS (side, category).
        "ln_total_stations_gold": n_g_groups,
        "ln_total_stations_pred": n_p_groups,
        "ln_total_stations_abs_err": abs(n_g_groups - n_p_groups),
        "ln_total_stations_correct_tol": int(
            abs(n_g_groups - n_p_groups) <= LN_COUNT_TOLERANCE),
        "ln_per_station": _per_station_counts(g_list, p_list),
        "ln_station_tp": tp,
        "ln_station_fp": fp,
        "ln_station_fn": fn,
        "ln_station_matched": tp,
        "ln_station_involved_correct": involved_ok,
        "ln_station_examined_correct": examined_ok,
        "ln_station_category_correct": category_ok,
        "ln_station_side_correct": side_ok,
        # New cascade-redesign keys.
        "ln_group_recall": group_recall,
        "ln_group_precision": group_precision,
        "ln_n_groups_gold": n_g_groups,
        "ln_n_groups_pred": n_p_groups,
        # Chapter-5 feeders.
        "ln_per_group": per_group_ledger,
        "ln_single_group_pair_side": single_group_pair_side,
        "ln_single_group_pair_category": single_group_pair_category,
    }


def score_margins(gold: dict, pred: dict) -> dict:
    """Returns a flat dict of per-case metrics for `margins`.

    Case-level:
      margin_any_involved_{gold,pred,correct}
      margin_closest_distance_{gold,pred,abs_err,correct_tol,has_both}
    Per-item:
      margin_{tp,fp,fn,matched,
              status_correct, distance_correct, category_correct}
    """
    g_list = _safe_list(get_field_value(gold, "margins"))
    p_list = _safe_list(get_field_value(pred, "margins"))

    def _any_involved(items):
        return any(bool(x.get("margin_involved")) for x in items)

    def _closest(items):
        vals = []
        for x in items:
            if x.get("margin_involved"):
                continue
            d = x.get("distance")
            if isinstance(d, (int, float)):
                vals.append(int(d))
        return min(vals) if vals else None

    g_any = _any_involved(g_list)
    p_any = _any_involved(p_list)
    g_close = _closest(g_list)
    p_close = _closest(p_list)

    has_both = g_close is not None and p_close is not None
    abs_err = abs(g_close - p_close) if has_both else None
    dist_correct_tol = int(abs_err <= NUMERIC_TOLERANCE_MM) if has_both else None

    matched, unm_g, unm_p = _greedy_match(g_list, p_list, _margin_similarity)
    tp = len(matched)
    fp = len(unm_p)
    fn = len(unm_g)

    # Matched-pair attribute tuples — feed the chapter-4 confusion
    # matrices and per-attribute reducers without re-running the
    # bipartite match downstream.
    matched_pair_categories: list[tuple] = []
    matched_pair_status: list[tuple] = []

    status_ok = distance_ok = category_ok = 0
    for gi, pi in matched:
        g, p = g_list[gi], p_list[pi]
        if bool(g.get("margin_involved")) == bool(p.get("margin_involved")):
            status_ok += 1
        gd, pd_ = g.get("distance"), p.get("distance")
        if gd is None and pd_ is None:
            distance_ok += 1
        elif isinstance(gd, (int, float)) and isinstance(pd_, (int, float)) \
                and abs(gd - pd_) <= NUMERIC_TOLERANCE_MM:
            distance_ok += 1
        if normalize(g.get("margin_category")) == normalize(p.get("margin_category")):
            category_ok += 1
        matched_pair_categories.append(
            (normalize(g.get("margin_category")),
             normalize(p.get("margin_category"))),
        )
        matched_pair_status.append(
            (bool(g.get("margin_involved")) if g.get("margin_involved") is not None else None,
             bool(p.get("margin_involved")) if p.get("margin_involved") is not None else None),
        )

    return {
        "margin_any_involved_gold": int(g_any),
        "margin_any_involved_pred": int(p_any),
        "margin_any_involved_correct": int(g_any == p_any),
        "margin_closest_distance_gold": g_close,
        "margin_closest_distance_pred": p_close,
        "margin_closest_distance_abs_err": abs_err,
        "margin_closest_distance_correct_tol": dist_correct_tol,
        "margin_closest_distance_has_both": int(has_both),
        "margin_per_category": _per_category_status(g_list, p_list),
        "margin_tp": tp,
        "margin_fp": fp,
        "margin_fn": fn,
        "margin_matched": tp,
        "margin_status_correct": status_ok,
        "margin_distance_correct": distance_ok,
        "margin_category_correct": category_ok,
        # Confusion-matrix feeders (long-form on demand by reducers).
        "margin_matched_pair_categories": matched_pair_categories,
        "margin_matched_pair_status": matched_pair_status,
    }


# --- Aggregation & summary ---------------------------------------------------

GENERATIVE_METHODS = {"digital_registrar", "gpt4_dspy"}


def _iter_test_cases(splits_path: Path):
    with splits_path.open(encoding="utf-8") as f:
        return json.load(f)["test"]


def _attempted(pred: dict, field: str) -> bool:
    if field in pred:
        return True
    cd = pred.get("cancer_data") or {}
    return field in cd


def _aggregate(method_to_preds: dict[str, Path],
               splits_path: Path,
               out_csv: Path,
               *,
               field: str,
               scorer) -> pd.DataFrame:
    cases = _iter_test_cases(splits_path)
    rows = []
    for method, preds_dir in method_to_preds.items():
        for case in cases:
            cid = case["id"]
            gold_path = Path(case["annotation_path"])
            pred_path = preds_dir / f"{cid}.json"
            row: dict = {"method": method, "case_id": cid, "attempted": False}
            if not pred_path.exists() or not gold_path.exists():
                rows.append(row)
                continue
            with gold_path.open(encoding="utf-8") as f:
                gold = json.load(f)
            with pred_path.open(encoding="utf-8") as f:
                pred = json.load(f)
            if not _attempted(pred, field):
                rows.append(row)
                continue
            row["attempted"] = True
            row.update(scorer(gold, pred))
            rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def aggregate_ln_to_csv(method_to_preds: dict[str, Path],
                        splits_path: Path,
                        out_csv: Path) -> pd.DataFrame:
    return _aggregate(method_to_preds, splits_path, out_csv,
                      field="regional_lymph_node", scorer=score_lymph_nodes)


def aggregate_margin_to_csv(method_to_preds: dict[str, Path],
                            splits_path: Path,
                            out_csv: Path) -> pd.DataFrame:
    return _aggregate(method_to_preds, splits_path, out_csv,
                      field="margins", scorer=score_margins)


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _bool_f1(df: pd.DataFrame, gold_col: str, pred_col: str) -> float:
    g = df[gold_col].fillna(0).astype(int)
    p = df[pred_col].fillna(0).astype(int)
    tp = int(((g == 1) & (p == 1)).sum())
    fp = int(((g == 0) & (p == 1)).sum())
    fn = int(((g == 1) & (p == 0)).sum())
    _, _, f1 = _prf(tp, fp, fn)
    return f1


def summarize_ln(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method summary of LN metrics."""
    attempted = df[df["attempted"]]
    if attempted.empty:
        return pd.DataFrame()

    def _per_method(sub):
        tp, fp, fn = sub["ln_station_tp"].sum(), sub["ln_station_fp"].sum(), sub["ln_station_fn"].sum()
        prec, rec, f1 = _prf(tp, fp, fn)
        matched = sub["ln_station_matched"].sum()
        out = {
            "coverage": len(sub) / len(df.loc[df["method"] == sub.name]),
            "cases_scored": len(sub),
            "examined_mae": sub["ln_examined_total_abs_err"].mean(),
            "examined_acc_tol1": sub["ln_examined_total_correct_tol"].mean(),
            "involved_mae": sub["ln_involved_total_abs_err"].mean(),
            "involved_acc_tol1": sub["ln_involved_total_correct_tol"].mean(),
            "any_positive_acc": sub["ln_any_positive_correct"].mean(),
            "any_positive_f1": _bool_f1(sub, "ln_any_positive_gold", "ln_any_positive_pred"),
            "station_precision": prec,
            "station_recall": rec,
            "station_f1": f1,
            "matched_involved_acc": (sub["ln_station_involved_correct"].sum() / matched) if matched else float("nan"),
            "matched_examined_acc": (sub["ln_station_examined_correct"].sum() / matched) if matched else float("nan"),
            "matched_category_acc": (sub["ln_station_category_correct"].sum() / matched) if matched else float("nan"),
            "matched_side_acc": (sub["ln_station_side_correct"].sum() / matched) if matched else float("nan"),
        }
        if "ln_total_stations_correct_tol" in sub.columns:
            out["total_stations_mae"] = sub["ln_total_stations_abs_err"].mean()
            out["total_stations_acc_tol1"] = sub["ln_total_stations_correct_tol"].mean()
        return pd.Series(out)

    return attempted.groupby("method").apply(_per_method, include_groups=False).reset_index()


def expand_ln_per_station(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the ``ln_per_station`` dict column into one row per
    (method, case_id, station_name).

    Output columns: method, case_id, station_name, gold_examined,
    pred_examined, examined_correct_tol, gold_involved, pred_involved,
    involved_correct_tol. Cases without a per-station dict are skipped.
    """
    if "ln_per_station" not in df.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for _, r in df.iterrows():
        per = r.get("ln_per_station")
        if not isinstance(per, dict) or not per:
            continue
        for sn, vals in per.items():
            row = {"method": r.get("method"), "case_id": r.get("case_id"),
                   "station_name": sn}
            row.update(vals)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_ln_per_station(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(method, station_name) accuracy of examined / involved counts.

    Consumes the long-form output of :func:`expand_ln_per_station`.
    """
    if df.empty:
        return df
    return (df.groupby(["method", "station_name"])
            .agg(n_cases=("case_id", "nunique"),
                 examined_acc_tol1=("examined_correct_tol", "mean"),
                 involved_acc_tol1=("involved_correct_tol", "mean"))
            .reset_index())


def summarize_margins(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method summary of margin metrics."""
    attempted = df[df["attempted"]]
    if attempted.empty:
        return pd.DataFrame()

    def _per_method(sub):
        tp, fp, fn = sub["margin_tp"].sum(), sub["margin_fp"].sum(), sub["margin_fn"].sum()
        prec, rec, f1 = _prf(tp, fp, fn)
        matched = sub["margin_matched"].sum()
        both = sub[sub["margin_closest_distance_has_both"] == 1]
        out = {
            "coverage": len(sub) / len(df.loc[df["method"] == sub.name]),
            "cases_scored": len(sub),
            "any_involved_acc": sub["margin_any_involved_correct"].mean(),
            "any_involved_f1": _bool_f1(sub, "margin_any_involved_gold", "margin_any_involved_pred"),
            "closest_dist_mae": both["margin_closest_distance_abs_err"].mean() if not both.empty else float("nan"),
            "closest_dist_acc_tol2": both["margin_closest_distance_correct_tol"].mean() if not both.empty else float("nan"),
            "closest_dist_both_rate": sub["margin_closest_distance_has_both"].mean(),
            "margin_precision": prec,
            "margin_recall": rec,
            "margin_f1": f1,
            "matched_status_acc": (sub["margin_status_correct"].sum() / matched) if matched else float("nan"),
            "matched_distance_acc": (sub["margin_distance_correct"].sum() / matched) if matched else float("nan"),
            "matched_category_acc": (sub["margin_category_correct"].sum() / matched) if matched else float("nan"),
        }
        if "margin_per_category" in sub.columns:
            # Average of per-category involved-correct rates, weighted
            # by the number of cases in which each category appeared.
            tot_pairs = 0
            tot_correct = 0
            for per in sub["margin_per_category"]:
                if not isinstance(per, dict):
                    continue
                for vals in per.values():
                    tot_pairs += 1
                    tot_correct += int(vals.get("involved_correct", 0))
            out["per_category_acc"] = (tot_correct / tot_pairs
                                       if tot_pairs else float("nan"))
        return pd.Series(out)

    return attempted.groupby("method").apply(_per_method, include_groups=False).reset_index()


def expand_margin_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the ``margin_per_category`` dict column into one row per
    (method, case_id, margin_category).

    Output columns: method, case_id, margin_category, gold_involved,
    pred_involved, involved_correct.
    """
    if "margin_per_category" not in df.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for _, r in df.iterrows():
        per = r.get("margin_per_category")
        if not isinstance(per, dict) or not per:
            continue
        for cat, vals in per.items():
            row = {"method": r.get("method"), "case_id": r.get("case_id"),
                   "margin_category": cat}
            row.update(vals)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_margin_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(method, margin_category) involved-correct accuracy.

    Consumes the long-form output of :func:`expand_margin_per_category`.
    """
    if df.empty:
        return df
    return (df.groupby(["method", "margin_category"])
            .agg(n_cases=("case_id", "nunique"),
                 involved_acc=("involved_correct", "mean"))
            .reset_index())


__all__ = [
    "GENERATIVE_METHODS",
    "LN_COUNT_TOLERANCE",
    "score_lymph_nodes",
    "score_margins",
    "aggregate_ln_to_csv",
    "aggregate_margin_to_csv",
    "summarize_ln",
    "summarize_margins",
    "expand_ln_per_station",
    "summarize_ln_per_station",
    "expand_margin_per_category",
    "summarize_margin_per_category",
]
