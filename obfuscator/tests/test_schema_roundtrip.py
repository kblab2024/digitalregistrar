"""Generated annotations validate against the per-organ JSON Schema."""
from __future__ import annotations

import pytest

from obfuscator.generation.annotation import derive_annotation
from obfuscator.generation.canonical import sample_canonical
from obfuscator.generation.realization import derive_realization
from obfuscator.profiles import ANNOTATOR_SLOTS
from obfuscator.schemas import list_available_organs, validate_cancer_data
from obfuscator.seeding import derive


@pytest.mark.parametrize("organ", list_available_organs())
def test_canonical_validates(organ):
    """One canonical sample per organ must pass schema validation."""
    rng = derive(42, "test_canonical", organ)
    canonical = sample_canonical(organ, rng)
    errors = validate_cancer_data(organ, canonical)
    assert errors == [], f"{organ}: {errors[:3]}"


@pytest.mark.parametrize("organ", list_available_organs())
def test_all_5_slots_validate(organ):
    """All 5 annotation slots, derived from one realization, must validate."""
    rng = derive(42, "test_slots", organ)
    canonical = sample_canonical(organ, rng)
    case_seed_parts = ("test", organ, "case42")
    realization = derive_realization(canonical, organ, 42, case_seed_parts)
    for slot in ANNOTATOR_SLOTS:
        ann = derive_annotation(realization, organ, slot, 42, case_seed_parts)
        errors = validate_cancer_data(organ, ann)
        assert errors == [], f"{organ}/{slot}: {errors[:3]}"


def test_5_slots_actually_diverge():
    """Across N cases, the 5 slots produce non-identical JSONs at least most of the time."""
    organ = "breast"
    n_diverge = 0
    n_total = 50
    for i in range(n_total):
        case_seed_parts = ("test", organ, f"case{i}")
        rng = derive(42, "test_diverge", organ, str(i))
        canonical = sample_canonical(organ, rng)
        realization = derive_realization(canonical, organ, 42, case_seed_parts)
        slots = {
            slot: derive_annotation(realization, organ, slot, 42, case_seed_parts)
            for slot in ANNOTATOR_SLOTS
        }
        gold = slots["gold"]
        if any(slots[s] != gold for s in ANNOTATOR_SLOTS if s != "gold"):
            n_diverge += 1
    assert n_diverge >= int(n_total * 0.8), (
        f"slots diverged in only {n_diverge}/{n_total} cases — noise rates too low?"
    )
