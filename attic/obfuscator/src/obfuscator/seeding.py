"""Deterministic seeding via blake2b — never uses Python's PYTHONHASHSEED-salted hash()."""
from __future__ import annotations

import hashlib
import random


def derive(master_seed: int, *parts: object) -> random.Random:
    """Derive a deterministic Random from a master seed and arbitrary key parts."""
    h = hashlib.blake2b(
        b"\x00".join(str(p).encode("utf-8") for p in parts),
        key=int(master_seed).to_bytes(8, "big", signed=False),
        digest_size=16,
    )
    return random.Random(int.from_bytes(h.digest()[:8], "big", signed=False))


def derive_int(master_seed: int, *parts: object, bits: int = 31) -> int:
    h = hashlib.blake2b(
        b"\x00".join(str(p).encode("utf-8") for p in parts),
        key=int(master_seed).to_bytes(8, "big", signed=False),
        digest_size=8,
    )
    return int.from_bytes(h.digest(), "big", signed=False) & ((1 << bits) - 1)
