"""Few-shot demo loader for the C2 / C3 ablations.

Demos are pre-registered in ``configs/ablations/fewshot_demos.yaml`` —
see ``scripts/ablations/build_fewshot_demos.py`` for the selection
script. The YAML records the source ``folder`` and ``dataset`` so the
loader can find the corresponding report + gold files in the canonical
tree without needing a separate splits.json.

YAML shape::

    seed: 42
    n_max: 5
    folder: /abs/path/to/dummy
    dataset: tcga
    organs:
      breast: [tcga1_47, tcga2_03, ...]
      lung:   [...]

The runner consumes :func:`load_demos(organ, n_shots)` and turns each
case into a ``dspy.Example`` keyed on ``report``.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import dspy
import yaml

from ...benchmarks.organs import organ_name_to_n
from ...paths import REPO_ROOT

DEMO_CONFIG_PATH = REPO_ROOT / "configs" / "ablations" / "fewshot_demos.yaml"


@cache
def _load_demo_config(path: Path = DEMO_CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Few-shot demo config not found at {path}. Run "
            f"scripts/ablations/build_fewshot_demos.py first.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_paths_from_config(config: dict, organ: str, case_id: str
                               ) -> tuple[Path, Path] | None:
    folder = Path(config.get("folder") or "")
    dataset = config.get("dataset")
    if not folder or not dataset:
        return None
    organ_n = organ_name_to_n(dataset, organ)
    if organ_n is None:
        return None
    report_path = folder / "data" / dataset / "reports" / organ_n / f"{case_id}.txt"
    gold_path = (folder / "data" / dataset / "annotations" / "gold"
                 / organ_n / f"{case_id}.json")
    return report_path, gold_path


def load_demos(organ: str, n_shots: int,
               config_path: Path = DEMO_CONFIG_PATH) -> list[dspy.Example]:
    """Return the first ``n_shots`` demos for ``organ`` as ``dspy.Example``s.

    Each example carries the keys the monolithic signature expects:
    ``report`` (list of paragraphs), ``report_jsonized`` (empty dict),
    and every output field populated from the gold annotation's
    ``cancer_data``.
    """
    config = _load_demo_config(config_path)
    organ_demos = config.get("organs", {}).get(organ, [])
    if not organ_demos:
        return []

    examples: list[dspy.Example] = []
    for case_id in organ_demos[:n_shots]:
        paths = _resolve_paths_from_config(config, organ, case_id)
        if paths is None:
            continue
        report_path, gold_path = paths
        if not report_path.exists() or not gold_path.exists():
            continue
        try:
            report_text = report_path.read_text(encoding="utf-8")
            with gold_path.open(encoding="utf-8") as f:
                gold = json.load(f)
        except Exception:
            continue
        cancer_data = gold.get("cancer_data") or {}
        paragraphs = [p.strip() for p in report_text.split("\n\n") if p.strip()]
        ex_kwargs = {"report": paragraphs, "report_jsonized": {}, **cancer_data}
        examples.append(
            dspy.Example(**ex_kwargs).with_inputs("report", "report_jsonized"))
    return examples


__all__ = ["load_demos", "DEMO_CONFIG_PATH"]
