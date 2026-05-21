"""Deterministic case-id → synthetic-id mapping. Same (master_seed, original_id) → same output."""
from __future__ import annotations

import string
import uuid

from .seeding import derive


_TCGA_ALPHABET = string.ascii_uppercase + string.digits


def synth_patient_filename(master_seed: int, case_id: str) -> str:
    """Map a case_id to a synthetic ``TCGA-XX-XXXX.<uuid>`` patient_filename.

    Same case_id under the same master_seed always yields the same output.
    """
    rng = derive(master_seed, "patient_filename", case_id)
    code1 = "".join(rng.choices(_TCGA_ALPHABET, k=2))
    code2 = "".join(rng.choices(_TCGA_ALPHABET, k=4))
    raw = bytearray(rng.getrandbits(8) for _ in range(16))
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    u = uuid.UUID(bytes=bytes(raw))
    return f"TCGA-{code1}-{code2}.{str(u).upper()}"


def synth_tcga_case_id(master_seed: int, original: str) -> str:
    """Map a TCGA-XX-XXXX style case ID to a synthetic one (same shape)."""
    rng = derive(master_seed, "tcga_case_id", original)
    code1 = "".join(rng.choices(_TCGA_ALPHABET, k=2))
    code2 = "".join(rng.choices(_TCGA_ALPHABET, k=4))
    return f"TCGA-{code1}-{code2}"
