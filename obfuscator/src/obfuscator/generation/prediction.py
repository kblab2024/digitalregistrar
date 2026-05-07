"""Layer 3: derive a per-(model, run) prediction from report_realization."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .. import profiles
from ..schemas import load_schema
from .annotation import _perturb_object


def derive_prediction(realization: dict[str, Any], organ: str,
                      model_slug: str, run_id: str,
                      master_seed: int, case_seed_parts: tuple) -> dict[str, Any]:
    """Return a noisy prediction for one (model, run) combination."""
    profile = profiles.model(model_slug)
    schema = load_schema(organ)
    out = deepcopy(realization)
    _perturb_object(out, schema, schema, master_seed,
                    case_seed_parts + ("prediction", model_slug, run_id),
                    profile.rate)
    return out
