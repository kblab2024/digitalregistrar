"""Path resolution for eval subcommands.

The :class:`Paths` instance is the **sole producer of disk paths** for
every eval subcommand. Subcommand modules must never construct path
strings directly — go through here so the layout convention is in one
place.

Layout (canonical example: ``/dummy``, real data: ``/workspace``):

    {root}/
    ├── data/{dataset}/
    │   ├── reports/{organ_idx}/{case_id}.txt
    │   ├── annotations/
    │   │   ├── gold/{organ_idx}/{case_id}.json
    │   │   ├── nhc_with_preann/{organ_idx}/{case_id}.json
    │   │   ├── nhc_without_preann/{organ_idx}/{case_id}.json
    │   │   ├── kpc_with_preann/{organ_idx}/{case_id}.json
    │   │   └── kpc_without_preann/{organ_idx}/{case_id}.json
    │   └── preannotation/{model}/{organ_idx}/{case_id}.json
    ├── results/predictions/{dataset}/
    │   ├── llm/{model}/{run_id}/{organ_idx}/{case_id}.json
    │   ├── clinicalbert/{model}/{organ_idx}/{case_id}.json
    │   └── rule_based/{organ_idx}/{case_id}.json
    └── results/ablations/{dataset}/{cell}/{model_slug}/{run_id}/{organ_idx}/{case_id}.json
        — for ``method="ablation"``. Cascade scoring of ablation cells
          uses this layout instead of the predictions tree.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from .stratify import all_organ_indices, parse_case_id

_RUN_ID_RE = re.compile(r"^run(\d+)(?:-([a-z0-9][a-z0-9-]*))?$")

Method = Literal["llm", "clinicalbert", "rule_based", "ablation"]
KNOWN_METHODS: tuple[Method, ...] = ("llm", "clinicalbert", "rule_based", "ablation")


@dataclass(frozen=True)
class Paths:
    """File-path producer for one ``(root, dataset)`` pair.

    Construct via ``Paths(root=resolve_folder(args.root), dataset=args.dataset)``
    where ``resolve_folder`` is :func:`scripts._config_loader.resolve_folder`.
    """

    root: Path
    dataset: str

    # --- Top-level dirs -------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / self.dataset

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def annotations_dir(self) -> Path:
        return self.data_dir / "annotations"

    @property
    def preannotation_dir(self) -> Path:
        return self.data_dir / "preannotation"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "results" / "predictions" / self.dataset

    @property
    def ablations_dir(self) -> Path:
        """Per-dataset ablations root.

        Cascade scoring of ablation cells reads predictions from
        ``{root}/results/ablations/{dataset}/{cell}/{model_slug}/{run_id}/...``
        rather than from ``predictions/llm/...``. The two trees are
        kept disjoint so re-running ablations doesn't overwrite the
        canonical LLM predictions used by the standard pipeline.
        """
        return self.root / "results" / "ablations" / self.dataset

    # --- Per-file accessors ---------------------------------------------------

    def report(self, organ_idx: int, case_id: str) -> Path:
        return self.reports_dir / str(organ_idx) / f"{case_id}.txt"

    def annotation(self, annotator: str, organ_idx: int, case_id: str) -> Path:
        """Annotation path for a given annotator subdirectory.

        ``annotator`` is the full subfolder name: ``gold``,
        ``nhc_with_preann``, ``nhc_without_preann``, ``kpc_with_preann``,
        ``kpc_without_preann``.
        """
        return self.annotations_dir / annotator / str(organ_idx) / f"{case_id}.json"

    def gold(self, organ_idx: int, case_id: str) -> Path:
        return self.annotation("gold", organ_idx, case_id)

    def preannotation(self, preann_model: str, organ_idx: int, case_id: str) -> Path:
        return self.preannotation_dir / preann_model / str(organ_idx) / f"{case_id}.json"

    def prediction(
        self,
        method: str,
        model: str | None,
        run_id: str | None,
        organ_idx: int,
        case_id: str,
        *,
        cell: str | None = None,
    ) -> Path:
        """Path to a model prediction file.

        ``method`` ∈ {"llm", "clinicalbert", "rule_based", "ablation"}.
        ``model`` and ``run_id`` are required for ``llm`` and
        ``ablation``; ``model`` only for ``clinicalbert``; both ignored
        for ``rule_based``. ``cell`` is required for ``ablation`` and
        identifies the cell directory under ``results/ablations/``.
        """
        if method == "llm":
            if not model or not run_id:
                raise ValueError("llm predictions require model and run_id")
            return (self.predictions_dir / "llm" / model / run_id
                    / str(organ_idx) / f"{case_id}.json")
        if method == "clinicalbert":
            if not model:
                raise ValueError("clinicalbert predictions require model")
            base = self.predictions_dir / "clinicalbert" / model
            if run_id:
                base = base / run_id
            return base / str(organ_idx) / f"{case_id}.json"
        if method == "rule_based":
            return self.predictions_dir / "rule_based" / str(organ_idx) / f"{case_id}.json"
        if method == "ablation":
            if not cell or not model or not run_id:
                raise ValueError(
                    "ablation predictions require cell, model, and run_id",
                )
            return (self.ablations_dir / cell / model / run_id
                    / str(organ_idx) / f"{case_id}.json")
        raise ValueError(f"unknown method: {method!r}")

    # --- Discovery ------------------------------------------------------------

    def case_ids(
        self,
        annotator: str = "gold",
        organs: tuple[int, ...] | None = None,
    ) -> Iterator[tuple[int, str]]:
        """Yield ``(organ_idx, case_id)`` for every annotation file under
        the given annotator. Iterated in deterministic sorted order.

        ``organs`` defaults to the dataset's full organ-index set
        (per ``configs/organ_code.yaml``).
        """
        base = self.annotations_dir / annotator
        if not base.is_dir():
            return
        organs = organs if organs is not None else all_organ_indices(self.dataset)
        for organ_idx in sorted(organs):
            organ_dir = base / str(organ_idx)
            if not organ_dir.is_dir():
                continue
            for p in sorted(organ_dir.glob("*.json")):
                yield organ_idx, p.stem

    def discover_runs(
        self, model: str, method: str = "llm"
    ) -> list[tuple[str, Path]]:
        """List ``(run_id, run_dir)`` for every ``run*`` subfolder of the
        model's prediction tree.

        Supports ``method="llm"`` (under ``predictions/{dataset}/llm/{model}/``)
        and ``method="clinicalbert"`` (under
        ``predictions/{dataset}/clinicalbert/{model}/`` where ``model`` is the
        head — typically ``merged`` for the headline comparison).
        Returns ``[]`` for ``rule_based`` (no run dimension) and for
        ablation runs (use :func:`discover_ablation_runs` instead).
        """
        if method == "llm":
            base = self.predictions_dir / "llm" / model
        elif method == "clinicalbert":
            base = self.predictions_dir / "clinicalbert" / model
        else:
            return []
        if not base.is_dir():
            return []
        runs: list[tuple[str, Path]] = []
        for p in sorted(base.iterdir()):
            if p.is_dir() and _RUN_ID_RE.fullmatch(p.name):
                runs.append((p.name, p))
        return runs

    def discover_ablation_runs(
        self, cell: str, model: str,
    ) -> list[tuple[str, Path]]:
        """List ``(run_id, run_dir)`` for every ``run*`` subfolder under
        ``results/ablations/{dataset}/{cell}/{model}/``.

        Returns ``[]`` if the cell or model dir doesn't exist. Cell-only
        directories with a single run write to ``runNN`` subfolders by
        convention; cells without a runNN tier are not supported by the
        cascade-scoring pipeline (use a sentinel ``run01`` if needed).
        """
        base = self.ablations_dir / cell / model
        if not base.is_dir():
            return []
        runs: list[tuple[str, Path]] = []
        for p in sorted(base.iterdir()):
            if p.is_dir() and _RUN_ID_RE.fullmatch(p.name):
                runs.append((p.name, p))
        return runs

    def assert_exists(self) -> None:
        """Sanity-check the root + dataset structure.

        Raises ``FileNotFoundError`` if the data dir is missing — fail
        fast rather than silently produce empty CSVs.
        """
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"data directory missing: {self.data_dir} "
                f"(check --root and --dataset)"
            )


# --- Convenience constructors -------------------------------------------------

def from_args(root: str | Path, dataset: str) -> Paths:
    """Build a :class:`Paths` from CLI args.

    Handles the ``dummy``/``workspace`` shortcut by deferring to
    :func:`scripts._config_loader.resolve_folder`.
    """
    # Local import to avoid a hard dependency loop during test isolation.
    from scripts._config_loader import resolve_folder
    return Paths(root=resolve_folder(root), dataset=dataset)


def parse_run_id_to_path_segment(run_id: str) -> str:
    """Validate a run-id string and return it unchanged.

    Run IDs are now arbitrary strings of the form ``runNN[-slug]`` (e.g.
    ``run01``, ``run02-alpha``). This helper catches malformed input
    early so subcommand code can trust ``run_id`` as a path segment.
    """
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            f"invalid run id: {run_id!r}; expected runNN[-slug] form"
        )
    return run_id


__all__ = [
    "Paths",
    "Method",
    "KNOWN_METHODS",
    "from_args",
    "parse_run_id_to_path_segment",
    "parse_case_id",
]
