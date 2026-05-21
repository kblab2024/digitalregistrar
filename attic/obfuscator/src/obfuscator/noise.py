"""Per-field noise primitives. Each takes (value, seed_parts, rate) and returns perturbed value.

A "rate" is the probability of applying noise. With probability (1-rate) the value is unchanged.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

from .seeding import derive


def _rng(master_seed: int, seed_parts: tuple) -> random.Random:
    return derive(master_seed, *seed_parts)


def perturb_enum(value: Any, enum_values: Sequence[Any], master_seed: int,
                 seed_parts: tuple, rate: float,
                 near_neighbors: dict[Any, list[Any]] | None = None) -> Any:
    """Replace with a near-neighbor (or uniform random fallback) with probability rate."""
    rng = _rng(master_seed, seed_parts + ("enum",))
    if rng.random() >= rate:
        return value
    if near_neighbors and value in near_neighbors:
        candidates = [v for v in near_neighbors[value] if v in enum_values and v != value]
        if candidates:
            return rng.choice(candidates)
    others = [v for v in enum_values if v != value]
    if not others:
        return value
    return rng.choice(others)


def perturb_int(value: int | None, master_seed: int, seed_parts: tuple,
                rate: float, sigma_pct: float = 0.15,
                lo: int | None = None, hi: int | None = None) -> int | None:
    if value is None:
        return None
    rng = _rng(master_seed, seed_parts + ("int",))
    if rng.random() >= rate:
        return value
    sigma = max(1.0, abs(value) * sigma_pct)
    new = int(round(rng.gauss(value, sigma)))
    if lo is not None:
        new = max(lo, new)
    if hi is not None:
        new = min(hi, new)
    return new


def perturb_bool(value: bool | None, master_seed: int, seed_parts: tuple,
                 rate: float) -> bool | None:
    if value is None:
        return None
    rng = _rng(master_seed, seed_parts + ("bool",))
    if rng.random() < rate:
        return not value
    return value


def maybe_drop(value: Any, master_seed: int, seed_parts: tuple, drop_rate: float) -> Any:
    """Replace with None with probability drop_rate."""
    rng = _rng(master_seed, seed_parts + ("drop",))
    if rng.random() < drop_rate:
        return None
    return value


def perturb_string(value: str | None, master_seed: int, seed_parts: tuple,
                   rate: float, synonyms: dict[str, list[str]] | None = None) -> str | None:
    if value is None:
        return None
    rng = _rng(master_seed, seed_parts + ("str",))
    if rng.random() >= rate:
        return value
    if synonyms and value in synonyms:
        return rng.choice(synonyms[value])
    return value
