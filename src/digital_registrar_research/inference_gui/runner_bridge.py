"""Thin adapter between the Streamlit UI and the extraction pipeline.

Keeps ``app.py`` free of pipeline / DSPy imports so the page module
loads quickly and is easy to reason about. Mirrors the contract of
``runner._select_runner`` so GUI output is interchangeable with the
``registrar-pipeline`` CLI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from digital_registrar_research.paths import REPO_ROOT


class PipelineSetupError(RuntimeError):
    """Raised when configuring DSPy for the chosen engine/model fails."""


def setup_for(engine: str, model: str) -> None:
    """Configure DSPy globally for ``engine`` × ``model``.

    Wraps the slow ``dspy.configure`` call behind a typed error so the
    UI can surface Ollama-down / missing-OpenAI-key failures cleanly.
    """
    try:
        if engine == "factory":
            from digital_registrar_research.pipeline_factory import setup_pipeline_v2
            setup_pipeline_v2(model)
        else:
            from digital_registrar_research.pipeline import setup_pipeline
            setup_pipeline(model)
    except Exception as e:
        raise PipelineSetupError(f"{engine}/{model}: {e}") from e


def run_one(
    report: str,
    fname: str,
    *,
    engine: str,
    decomposition: Literal["per_group", "monolithic", "auto"] = "auto",
    jsonize: bool = False,
    validate_output: bool = True,
) -> tuple[dict, float]:
    """Run the pipeline on a single report. Returns ``(output, elapsed_s)``."""
    if engine == "factory":
        from digital_registrar_research.pipeline_factory import run_cancer_pipeline_v2
        return run_cancer_pipeline_v2(
            report=report,
            fname=fname,
            decomposition=decomposition,
            jsonize_enabled=jsonize,
            validate_output=validate_output,
        )
    from digital_registrar_research.pipeline import run_cancer_pipeline
    return run_cancer_pipeline(report=report, fname=fname)


def save_output(run_dir: Path, stem: str, output: dict) -> Path:
    """Write ``output`` to ``run_dir/<stem>_output.json``.

    Byte-identical to the CLI runner's write at ``runner.run_folder``.
    """
    out_path = run_dir / f"{stem}_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return out_path


def make_run_dir() -> Path:
    """Create ``runs/run_<timestamp>/`` under the repo root."""
    from digital_registrar_research.runner import create_run_folder
    base = REPO_ROOT / "runs"
    base.mkdir(parents=True, exist_ok=True)
    return Path(create_run_folder(str(base)))


__all__ = [
    "PipelineSetupError",
    "setup_for",
    "run_one",
    "save_output",
    "make_run_dir",
]
