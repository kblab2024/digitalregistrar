"""Loads profiles.yaml and exposes typed accessors."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_PROFILES_PATH = Path(__file__).resolve().parent / "profiles.yaml"

ANNOTATOR_SLOTS = (
    "gold",
    "nhc_with_preann",
    "nhc_without_preann",
    "kpc_with_preann",
    "kpc_without_preann",
)


@dataclass(frozen=True)
class AnnotatorProfile:
    name: str
    rate: float
    bias: str
    preann_follow_prob: float = 0.0


@dataclass(frozen=True)
class ModelProfile:
    name: str
    rate: float
    family: str


@lru_cache(maxsize=1)
def _raw() -> dict[str, Any]:
    with _PROFILES_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def annotator(name: str) -> AnnotatorProfile:
    cfg = _raw().get("annotators", {}).get(name)
    if cfg is None:
        return AnnotatorProfile(name=name, rate=0.10, bias="none")
    return AnnotatorProfile(
        name=name,
        rate=float(cfg.get("rate", 0.10)),
        bias=str(cfg.get("bias", "none")),
        preann_follow_prob=float(cfg.get("preann_follow_prob", 0.0)),
    )


def model(name: str) -> ModelProfile:
    cfg = _raw().get("models", {}).get(name)
    if cfg is None:
        return ModelProfile(name=name, rate=0.20, family="llm")
    return ModelProfile(
        name=name,
        rate=float(cfg.get("rate", 0.20)),
        family=str(cfg.get("family", "llm")),
    )


def realization() -> dict[str, float]:
    cfg = _raw().get("realization", {})
    return {
        "drop_rate": float(cfg.get("drop_rate", 0.10)),
        "ambiguity_rate": float(cfg.get("ambiguity_rate", 0.05)),
    }


def numeric_range(field: str) -> dict[str, float] | None:
    return _raw().get("numeric_ranges", {}).get(field)


def near_neighbors(field: str) -> dict[Any, list[Any]]:
    return _raw().get("near_neighbors", {}).get(field, {}) or {}


def cancer_excision_rate() -> float:
    return float(_raw().get("cancer_excision_rate", 0.80))
