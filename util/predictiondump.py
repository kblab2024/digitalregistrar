"""
    predictiondump.py
    ~~~~~~~~~~~~~~~~~~~~~~
    This module provides functions to convert model predictions into a structured format
    that can be easily serialized and saved for later analysis.

    Copyright 2025, Kai-Po Chang at Med NLP Lab, China Medical University, with aid from chatGPT. 
"""
__version__ = "0.1.0"
__date__ = "2025-10-05"
__author__ = ["Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"

import json
from typing import Any, Iterable, Callable, Optional
from dataclasses import is_dataclass, asdict
from datetime import datetime, date
from decimal import Decimal
from collections.abc import Mapping

# --- helpers to make everything JSON-safe ---
def _to_json_safe(x: Any):
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, Decimal):
        return float(x)
    try:
        import numpy as np  # optional
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass
    return x

# --- core recursive dumper ---

def dump_prediction_plain(pred) -> dict[str, Any]:
    """Recursively convert a DSPy Prediction into a plain dict."""
    from dspy.primitives.prediction import Prediction

    def to_plain(obj: Any):
        # Base primitives (return as-is)
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        # Nested Prediction → dict
        if isinstance(obj, Prediction):
            return {k: to_plain(v) for k, v in obj._store.items()
                    if not k.startswith(("_lm_usage", "_inputs", "_completions"))}

        # Mappings (dict-like)
        if isinstance(obj, Mapping):
            return {k: to_plain(v) for k, v in obj.items()
                    if not (isinstance(k, str) and k.startswith("_"))}

        # Lists / tuples
        if isinstance(obj, (list, tuple, set)):
            return [to_plain(v) for v in obj]

        # Objects with .dict() or .to_dict()
        for attr in ("dict", "to_dict"):
            if hasattr(obj, attr) and callable(getattr(obj, attr)):
                try:
                    return to_plain(getattr(obj, attr)())
                except Exception:
                    pass

        # Fallback: plain value
        return obj

    result = to_plain(pred)

    # If the top level isn't a dict, wrap it so we always return one
    return result if isinstance(result, dict) else {"value": result}

def dump_prediction(
    obj: Any,
    *,
    exclude_private: bool = True,
    exclude_keys: tuple[str, ...] = ("_lm_usage", "_inputs", "_completions"),
    custom_predicate: Optional[Callable[[str, Any], bool]] = None,
) -> Any:
    """
    Recursively convert a DSPy Prediction (or arbitrary nested structure)
    into JSON-serializable Python types, excluding selected internal fields.

    - exclude_private=True removes any dict keys starting with "_".
    - exclude_keys removes specific keys regardless of exclude_private.
    - custom_predicate(key, value) -> bool can veto inclusion of a field.
    """
    # Avoid circulars / trivial primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # DSPy Prediction (or anything duck-typing a store-like interface)
    try:
        from dspy.primitives.prediction import Prediction  # type: ignore
        is_prediction = isinstance(obj, Prediction)
    except Exception:
        is_prediction = False

    if is_prediction:
        # Most DSPy predictions expose their data via an internal store.
        store = getattr(obj, "_store", {}) or {}
        out = {}
        for k, v in store.items():
            if k in exclude_keys:
                continue
            if exclude_private and isinstance(k, str) and k.startswith("_"):
                continue
            if custom_predicate and not custom_predicate(k, v):
                continue
            out[k] = dump_prediction(
                v,
                exclude_private=exclude_private,
                exclude_keys=exclude_keys,
                custom_predicate=custom_predicate,
            )
        return out

    # Pydantic (v2 or v1) objects
    try:
        from pydantic import BaseModel  # type: ignore
        if isinstance(obj, BaseModel):
            data = (
                obj.model_dump()  # v2
                if hasattr(obj, "model_dump")
                else obj.dict()   # v1
            )
            return dump_prediction(
                data,
                exclude_private=exclude_private,
                exclude_keys=exclude_keys,
                custom_predicate=custom_predicate,
            )
    except Exception:
        pass

    # Dataclasses
    if is_dataclass(obj):
        return dump_prediction(
            asdict(obj),
            exclude_private=exclude_private,
            exclude_keys=exclude_keys,
            custom_predicate=custom_predicate,
        )

    # Mapping
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str):
                if k in exclude_keys:
                    continue
                if exclude_private and k.startswith("_"):
                    continue
                if custom_predicate and not custom_predicate(k, v):
                    continue
            out[k] = dump_prediction(
                v,
                exclude_private=exclude_private,
                exclude_keys=exclude_keys,
                custom_predicate=custom_predicate,
            )
        return out

    # Sequence
    if isinstance(obj, (list, tuple, set)):
        return [
            dump_prediction(
                v,
                exclude_private=exclude_private,
                exclude_keys=exclude_keys,
                custom_predicate=custom_predicate,
            )
            for v in obj
        ]

    # Objects with to_dict()/dict()
    for meth in ("to_dict", "dict"):
        if hasattr(obj, meth) and callable(getattr(obj, meth)):
            try:
                data = getattr(obj, meth)()
                return dump_prediction(
                    data,
                    exclude_private=exclude_private,
                    exclude_keys=exclude_keys,
                    custom_predicate=custom_predicate,
                )
            except Exception:
                pass

    # Fallback to JSON-safe conversion (datetime, numpy, decimal, etc.)
    return _to_json_safe(obj)

# --- combining many predictions into one JSON blob ---
def dump_many_predictions(
    preds: Iterable[Any],
    *,
    key_fn: Optional[Callable[[Any, int], Optional[str]]] = None,
    **dump_kwargs
) -> str:
    """
    Dump many predictions into a single JSON string.

    - By default returns a JSON array string.
    - If key_fn is provided and returns a non-None key, we build a dict
      mapping key -> dumped prediction.
    - dump_kwargs are forwarded to dump_prediction (e.g., exclude_keys=...).
    """
    dumped_list = [dump_prediction(p, **dump_kwargs) for p in preds]

    if key_fn:
        mapping = {}
        for i, (p, d) in enumerate(zip(preds, dumped_list)):
            k = key_fn(p, i)
            if k is not None:
                mapping[k] = d
        return json.dumps(mapping, ensure_ascii=False, indent=2)

    return json.dumps(dumped_list, ensure_ascii=False, indent=2)
