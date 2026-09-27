"""Folder pairing for ``registrar-eval``.

Matches a folder of predictions against a folder of gold annotations by
case id. File names are reduced to a case id by stripping one trailing
``_output`` / ``_annotation`` suffix (and a leftover ``.txt``), so all of
these pair up as case ``tcga4_1``::

    pred/tcga4_1_output.json        # registrar-pipeline output
    pred/tcga4_1.txt_output.json    # registrar-pipeline random-report mode
    gold/tcga4_1_annotation.json    # annotator export
    gold/tcga4_1.json               # plain

Both folders are searched recursively, so the per-dataset subfolders the
pipeline writes (``<run>/<dataset>/<stem>_output.json``) work unchanged.
Case ids must be unique within each folder.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CASE_ID_SUFFIXES = ("_output", "_annotation")


def case_id_from_path(path: Path | str) -> str:
    """Reduce a JSON file name to its case id (see module docstring)."""
    cid = Path(path).stem
    for suffix in CASE_ID_SUFFIXES:
        if cid.endswith(suffix) and len(cid) > len(suffix):
            cid = cid[: -len(suffix)]
            break
    if cid.endswith(".txt"):
        cid = cid[: -len(".txt")]
    return cid


def index_folder(root: Path | str) -> dict[str, Path]:
    """Map case id → JSON path for every ``*.json`` under ``root``.

    Raises ``FileNotFoundError`` if ``root`` is not a directory and
    ``ValueError`` if two files reduce to the same case id.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue
        cid = case_id_from_path(path)
        if cid in out:
            raise ValueError(
                f"duplicate case id {cid!r} under {root}: {out[cid]} and {path}"
            )
        out[cid] = path
    return out


def _load_record(path: Path) -> dict | None:
    """Load a JSON object; ``None`` if unreadable or not a JSON object."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        logger.warning("could not read %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        logger.warning("%s does not contain a JSON object", path)
        return None
    return data


@dataclass
class CasePair:
    """One gold case and its prediction (``pred`` is None if missing/unreadable)."""
    case_id: str
    dataset: str
    gold_path: Path
    pred_path: Path | None
    gold: dict
    pred: dict | None


def load_pairs(
    pred_root: Path | str, gold_root: Path | str,
) -> tuple[list[CasePair], dict[str, int]]:
    """Pair every gold case with its prediction.

    Returns ``(pairs, stats)``. ``pairs`` has one entry per readable gold
    file, sorted by case id. ``stats`` counts ``n_gold``, ``n_matched``,
    ``n_missing_pred``, ``n_unreadable_pred``, ``n_extra_pred`` (preds
    with no gold, ignored) and ``n_unreadable_gold`` (skipped).
    """
    gold_root = Path(gold_root)
    gold_index = index_folder(gold_root)
    pred_index = index_folder(pred_root)

    pairs: list[CasePair] = []
    stats = {"n_gold": 0, "n_matched": 0, "n_missing_pred": 0,
             "n_unreadable_pred": 0, "n_extra_pred": 0, "n_unreadable_gold": 0}
    for cid, gold_path in gold_index.items():
        gold = _load_record(gold_path)
        if gold is None:
            stats["n_unreadable_gold"] += 1
            continue
        stats["n_gold"] += 1
        pred_path = pred_index.get(cid)
        pred = None
        if pred_path is None:
            stats["n_missing_pred"] += 1
        else:
            pred = _load_record(pred_path)
            if pred is None:
                stats["n_unreadable_pred"] += 1
            else:
                stats["n_matched"] += 1
        dataset = gold_path.parent.relative_to(gold_root).as_posix()
        pairs.append(CasePair(
            case_id=cid,
            dataset="" if dataset == "." else dataset,
            gold_path=gold_path,
            pred_path=pred_path,
            gold=gold,
            pred=pred,
        ))

    extra = sorted(set(pred_index) - set(gold_index))
    stats["n_extra_pred"] = len(extra)
    if extra:
        logger.warning("%d prediction(s) have no gold annotation and were ignored "
                       "(e.g. %s)", len(extra), pred_index[extra[0]])
    if stats["n_unreadable_gold"]:
        logger.warning("%d gold file(s) could not be read and were skipped",
                       stats["n_unreadable_gold"])
    return pairs, stats


__all__ = ["CASE_ID_SUFFIXES", "CasePair", "case_id_from_path", "index_folder", "load_pairs"]
