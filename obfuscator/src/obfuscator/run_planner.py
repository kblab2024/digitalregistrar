"""Read configs/models/*.yaml from src to determine which (model, run) combos to emit predictions for."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class RunPlan:
    model_slug: str
    family: str          # 'llm' | 'clinicalbert' | 'rule_based'
    run_id: str          # 'run01', 'run02', ...
    seed: int


def plan_runs(src: Path) -> list[RunPlan]:
    """Return one RunPlan per (model, run) combination found in configs/models/."""
    configs_root = src / "configs" / "models"
    if not configs_root.is_dir():
        configs_root = _fallback_configs_root()
    plans: list[RunPlan] = []
    for path in sorted(configs_root.glob("*.yaml")):
        with path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        slug = cfg.get("name") or path.stem
        family = cfg.get("family") or "llm"
        if family in ("clinicalbert", "rule_based"):
            # No multi-run for these in current configs; emit a single run.
            plans.append(RunPlan(model_slug=slug, family=family, run_id="run01", seed=0))
            continue
        runs_cfg = cfg.get("runs") or {}
        seeds = runs_cfg.get("seeds") or []
        k = int(runs_cfg.get("k", len(seeds) or 1))
        if not seeds:
            seeds = list(range(k))
        for i, seed in enumerate(seeds[:k], start=1):
            plans.append(RunPlan(
                model_slug=slug, family=family,
                run_id=f"run{i:02d}", seed=int(seed),
            ))
    return plans


def _fallback_configs_root() -> Path:
    """Use the inference-repo configs as a last resort (e.g. when src has no configs/)."""
    return Path(__file__).resolve().parents[3] / "configs" / "models"


def models_by_family(plans: Iterable[RunPlan]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for p in plans:
        out.setdefault(p.family, set()).add(p.model_slug)
    return out
